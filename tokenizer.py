"""
Copyright (c) 2020 Reverie Labs. All rights reserved.
This work is licensed under the terms of the MIT license .
"""

from enum import Enum
from typing import Iterable, List, Union

import numpy as np
from tqdm import tqdm

UNK_TOKEN = ""
MASK_TOKEN = "[*]"

element_abbreviations = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
    'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As',
    'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
    'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
    'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]


class TokenizerStatus(Enum):
    OK = 1
    INVALID_OTHER = 2
    INVALID_BRACKET = 3


class SmilesTokenizer:
    """General purpose NLP-style tokenizer for SMILES strings.

    Args:
        separator: When `return_type = str`, return strings will be joined by this separator. 
            Defaults to a single space.
        skip_invalid: If False (default), raises an error when a SMILES cannot be tokenized. If
            True, prints errors and returns TokenizerStatus.INVALID, but does not raise errors.
    """

    def __init__(self,
                 separator: str = " ",
                 skip_invalid: bool = False,
                 use_multiprocessing: bool = True):
        self.separator = separator
        self.skip_invalid = skip_invalid

        # List of element abbreviations
        self.abbrevs = element_abbreviations
        self.two_letter_abbrevs = list(filter(lambda x: len(x) == 2, self.abbrevs))

    def tokenize(self, smiles: Union[str, Iterable], return_type: Union[list, str] = list):
        """Main API method. Tokenizes a single SMILES string or iterable of SMILES."""
        if isinstance(smiles, str):
            tokens_list = self._tokenize(smiles)
            if isinstance(tokens_list, TokenizerStatus):
                return tokens_list
            if return_type == str:
                return self.tokens_to_string(tokens_list)
            elif return_type == list:
                return tokens_list
            else:
                raise ValueError(f"Unknown return_type {return_type}")

        # Assume smiles is an iterable
        for s in smiles:
            assert (isinstance(s, str))
        results = [self.tokenize(s, return_type) for s in tqdm(smiles)]
        return results

    def tokens_to_string(self, tokens_list: List[str]):
        """Converts a list of tokens to a string using `separator`."""
        return self.separator.join(tokens_list)

    def _tokenize(self, smiles: str):
        """Efficient, O(n) SMILES tokenizer.
        Anything in square brackets is considered a single token (e.g., [C@@H], [Na2+], etc.).
        Args:
            smiles: Single SMILES string.
        Returns:
            List of tokens.
        """
        assert (isinstance(smiles, str))
        tokens = []

        smiles += "&"
        buf = ""
        bracket_mode = False

        for char1, char2 in zip(smiles[0:-1], smiles[1:]):
            if char1 == "[":
                if buf:
                    return self._handle_error(smiles, TokenizerStatus.INVALID_OTHER)
                if bracket_mode:
                    return self._handle_error(smiles, TokenizerStatus.INVALID_BRACKET)
                bracket_mode = True
                buf += char1
            elif char1 == "]":
                if not bracket_mode:
                    return self._handle_error(smiles, TokenizerStatus.INVALID_BRACKET)
                bracket_mode = False
                buf += char1
                tokens.append(buf)
                buf = ""
            elif bracket_mode:
                buf += char1
            elif buf:
                tokens.append(buf + char1)
                buf = ""
            elif char1 + char2 in self.two_letter_abbrevs:
                buf += char1
            else:
                tokens.append(char1)
        return tokens

    def _handle_error(self, smiles: str, err: TokenizerStatus):
        """Raises or returns an error, depending on `skip_invalid` setting."""
        if self.skip_invalid:
            print(err, smiles)
            return err
        else:
            raise ValueError(err, smiles)