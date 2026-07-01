#!/usr/bin/env python3

# ===-- generate_tests.py - Generate M68k tests ---------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#
#
# This file generates some of the repetitive and error-prone encoding tests for
# the M68k target.
#
# Currently generated:
# - llvm/test/MC/M68k/MOVE.s
#
# ===----------------------------------------------------------------------===#

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Generator


# Utility functions:


# https://stackoverflow.com/a/952952
def flatten(xss):
    """Flattens a list of lists into a single list."""
    return [x for xs in xss for x in xs]


def word_to_bytes(word: int) -> bytes:
    """Converts a 16-bit word value to a pair of bytes."""
    return word.to_bytes(2, byteorder="big")


def words_to_bytes(words: list[int]) -> bytes:
    """Converts a list of words to a list of bytes."""
    return flatten(map(word_to_bytes, words))


def byte_to_hex(byte: int) -> str:
    """Returns hexadecimal representation of a number with 0x prefix"""
    return f"{byte:#04x}"


# Size field
# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf230


class Size(ABC):
    """Represents the amount of data being operated on."""

    @abstractmethod
    def asm() -> str:
        """Returns the size suffix in assembly language."""

    @abstractmethod
    def sizeField() -> int:
        """Returns the encoded value of the instruction's "size" field."""


@dataclass
class Byte(Size):
    """Indicates that an 8-bit value is being operated on."""

    def asm() -> str:
        return "b"

    def sizeField() -> int:
        return 0b01


@dataclass
class Word(Size):
    """Indicates that a 16-bit value is being operated on."""

    def asm() -> str:
        return "w"

    def sizeField() -> int:
        return 0b11


@dataclass
class Long(Size):
    """Indicates that a 32-bit value is being operated on."""

    def asm() -> str:
        return "l"

    def sizeField() -> int:
        return 0b10


# Data registers


class DataRegister(IntEnum):
    D0 = 0
    D1 = 1
    D2 = 2
    D3 = 3
    D4 = 4
    D5 = 5
    D6 = 6
    D7 = 7

    def permutations() -> Generator["DataRegister", None, None]:
        # d0, d1 and d7 are considered representative permutations, in order to test:
        # - all bits 0
        # - all bits 1
        # - the bit ordering
        for register in [DataRegister.D0, DataRegister.D1, DataRegister.D7]:
            yield register

    def asm(self) -> str:
        return f"%d{self}"


# Address registers


class AddressRegister(IntEnum):
    A0 = 0
    A1 = 1
    A2 = 2
    A3 = 3
    A4 = 4
    A5 = 5
    A6 = 6
    SP = 7

    def permutations() -> Generator["AddressRegister", None, None]:
        # a0, a1 and a7 (sp) are considered representative permutations, in order to test:
        # - all bits 0
        # - all bits 1
        # - the bit ordering
        for register in [AddressRegister.A0, AddressRegister.A1, AddressRegister.SP]:
            yield register

    def asm(self) -> str:
        if self is AddressRegister.SP:
            return "%sp"
        else:
            return f"%a{self}"


# Effective addressing modes
# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf3c


class EffectiveAddressingMode(ABC):
    """Represents a mechanism for addressing source or destination data."""

    @abstractmethod
    def permutations() -> Generator["EffectiveAddressingMode", None, None]:
        """Returns all the permutations of this addressing mode."""

    @abstractmethod
    def asm(self) -> str:
        """Returns the assembly language for the operand being addressed."""

    @abstractmethod
    def modeField(self) -> int:
        """Returns the encoded value of the instruction's "mode" field."""

    @abstractmethod
    def registerField(self) -> int:
        """Returns the encoded value of the instruction's "register" field."""


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2e
# Dn
@dataclass
class DataRegisterDirect(EffectiveAddressingMode):
    """Represents the contents of a particular data register."""

    register: DataRegister

    def permutations() -> Generator[EffectiveAddressingMode, None, None]:
        for register in DataRegister.permutations():
            yield DataRegisterDirect(register)

    def asm(self) -> str:
        return self.register.asm()

    def modeField(self) -> int:
        return 0b000

    def registerField(self) -> int:
        return self.register


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2e
# An
@dataclass
class AddressRegisterDirect(EffectiveAddressingMode):
    """Represents the contents of a particular address register."""

    register: AddressRegister

    def permutations() -> Generator[EffectiveAddressingMode, None, None]:
        for register in AddressRegister.permutations():
            yield AddressRegisterDirect(register)

    def asm(self) -> str:
        return self.register.asm()

    def modeField(self) -> int:
        return 0b001

    def registerField(self) -> int:
        return self.register


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2e
# (An)
@dataclass
class AddressRegisterIndirect(EffectiveAddressingMode):
    """Represents the contents of memory, with address in a particular address register."""

    register: AddressRegister

    def permutations() -> Generator[EffectiveAddressingMode, None, None]:
        for register in AddressRegister.permutations():
            yield AddressRegisterIndirect(register)

    def asm(self) -> str:
        return f"({self.register.asm()})"

    def modeField(self) -> int:
        return 0b010

    def registerField(self) -> int:
        return self.register


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2f
# (An)+
@dataclass
class AddressRegisterIndirectWithPostincrement(EffectiveAddressingMode):
    """Represents the contents of memory, with address in a particular address register. The address register is incremented by size after dereferencing."""

    register: AddressRegister

    def permutations() -> Generator[EffectiveAddressingMode, None, None]:
        for register in AddressRegister.permutations():
            yield AddressRegisterIndirectWithPostincrement(register)

    def asm(self) -> str:
        return f"({self.register.asm()})+"

    def modeField(self) -> int:
        return 0b011

    def registerField(self) -> int:
        return self.register


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf30
# -(An)
@dataclass
class AddressRegisterIndirectWithPredecrement(EffectiveAddressingMode):
    """Represents the contents of memory, with address in a particular address register. The address register is decremented by size before dereferencing."""

    register: AddressRegister

    def permutations() -> Generator[EffectiveAddressingMode, None, None]:
        for register in AddressRegister.permutations():
            yield AddressRegisterIndirectWithPredecrement(register)

    def asm(self) -> str:
        return f"-({self.register.asm()})"

    def modeField(self) -> int:
        return 0b100

    def registerField(self) -> int:
        return self.register


# Instructions


class Instruction(ABC):
    """An abstract class to represent a machine instruction."""

    @abstractmethod
    def permutations() -> Generator["Instruction", None, None]:
        """Returns all permutations of this instruction and its operands."""

    @abstractmethod
    def asm(self) -> str:
        """Returns the assembly language of this instruction."""

    @abstractmethod
    def words(self) -> list[int]:
        """Returns the encoded 16-bit words of this instruction."""

    def bytes(self) -> bytes:
        """Encodes the words of the instruction as a byte array."""
        words = self.words()
        lists_of_bytes = words_to_bytes(words)
        return lists_of_bytes


# https://m680x0.github.io/ref/integer-instructions.html#pfdc
@dataclass
class MOVE(Instruction):
    size: Size
    destination: EffectiveAddressingMode
    source: EffectiveAddressingMode

    def permutations() -> Generator["MOVE", None, None]:
        sourceModes = [
            DataRegisterDirect,
            AddressRegisterDirect,
            AddressRegisterIndirect,
            AddressRegisterIndirectWithPostincrement,
            AddressRegisterIndirectWithPredecrement,
        ]
        destinationModes = [
            DataRegisterDirect,
            AddressRegisterIndirect,
            AddressRegisterIndirectWithPostincrement,
            AddressRegisterIndirectWithPredecrement,
        ]

        for size in [Byte, Word, Long]:
            for sourceMode in sourceModes:
                # For byte size operation, address register direct is not allowed
                if (size is not Byte) or (sourceMode is not AddressRegisterDirect):
                    for source in sourceMode.permutations():
                        for destinationMode in destinationModes:
                            for destination in destinationMode.permutations():
                                yield MOVE(size, destination, source)

    def asm(self) -> str:
        return f"move.{self.size.asm()} {self.source.asm()}, {self.destination.asm()}"

    def words(self) -> list[int]:
        return [
            (
                (0b00 << 14)
                | (self.size.sizeField() << 12)
                | (self.destination.registerField() << 9)
                | (self.destination.modeField() << 6)
                | (self.source.modeField() << 3)
                | self.source.registerField()
            )
        ]


# Encoding tests


def write_encoding_tests_header(f):
    """Writes the comments at the top of the encoding test file."""
    header: list[str] = [
        "; Generated by llvm/test/MC/M68k/generate_tests.py\n",
        "; Please do not edit this file by hand.\n",
        "\n",
        "; RUN: llvm-mc -triple=m68k -mcpu=M68040 -show-encoding < %s | FileCheck %s\n",
    ]

    for line in header:
        f.write(line)


def write_encoding_test(f, instruction: Instruction):
    """Writes a single instruction's encoding test to a file."""
    bytes = ",".join(map(byte_to_hex, instruction.bytes()))
    f.write(f"; CHECK:      {instruction.asm()}\n")
    f.write(f"; CHECK-SAME: encoding: [{bytes}]\n")
    f.write(f"{instruction.asm()}\n")


def write_encoding_tests(cls):
    """Writes tests for all permutations of an instruction to a new file."""
    name = cls.__name__
    with open(f"{name}.s", "w") as f:
        write_encoding_tests_header(f)
        for instruction in cls.permutations():
            f.write("\n")
            write_encoding_test(f, instruction)


for cls in [MOVE]:
    write_encoding_tests(cls)
