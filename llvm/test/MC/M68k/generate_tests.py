#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator


# Utility functions:


# https://stackoverflow.com/a/952952
def flatten(xss):
    """Flattens a list of lists into a single list"""
    return [x for xs in xss for x in xs]


def word_to_bytes(word: int) -> bytes:
    """Converts a 16-bit word value to a pair of bytes"""
    return word.to_bytes(2, byteorder="big")


def words_to_bytes(words: list[int]) -> bytes:
    """Converts a list of words to a list of bytes"""
    return flatten(map(word_to_bytes, words))


# Size field
# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf230


class Size(ABC):
    @abstractmethod
    def asm() -> str:
        pass

    @abstractmethod
    def sizeField() -> int:
        pass


@dataclass
class Byte(Size):
    def asm() -> str:
        return "b"

    def sizeField() -> int:
        return 0b01


@dataclass
class Word(Size):
    def asm() -> str:
        return "w"

    def sizeField() -> int:
        return 0b11


@dataclass
class Long(Size):
    def asm() -> str:
        return "l"

    def sizeField() -> int:
        return 0b10


# Effective addressing modes
# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf3c


class EffectiveAddressingMode(ABC):
    @abstractmethod
    def permutations() -> Generator["EffectiveAddressingMode", None, None]:
        """Returns all the permutations of this addressing mode"""
        pass

    @abstractmethod
    def asm(self) -> str:
        pass

    @abstractmethod
    def modeField(self) -> int:
        pass

    @abstractmethod
    def registerField(self) -> int:
        pass


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2e
# Dn
@dataclass
class DataRegisterDirect(EffectiveAddressingMode):
    registerNumber: int

    def permutations() -> Generator["DataRegisterDirect", None, None]:
        for registerNumber in range(8):
            yield DataRegisterDirect(registerNumber)

    def asm(self) -> str:
        return f"%d{self.registerNumber}"

    def modeField(self) -> int:
        return 0b000

    def registerField(self) -> int:
        return self.registerNumber


# https://m680x0.github.io/ref/M68000PM_AD_Rev_1_Programmers_Reference_Manual_1992.html#pf2e
# An
@dataclass
class AddressRegisterDirect(EffectiveAddressingMode):
    registerNumber: int

    def permutations() -> Generator["AddressRegisterDirect", None, None]:
        for registerNumber in range(8):
            yield AddressRegisterDirect(registerNumber)

    def asm(self) -> str:
        if self.registerNumber == 7:
            return "%sp"
        else:
            return f"%a{self.registerNumber}"

    def modeField(self) -> int:
        return 0b001

    def registerField(self) -> int:
        return self.registerNumber


# Instructions


class Instruction(ABC):
    @abstractmethod
    def permutations() -> Generator["Instruction", None, None]:
        pass

    @abstractmethod
    def asm(self) -> str:
        pass

    @abstractmethod
    def words(self) -> list[int]:
        pass

    def bytes(self) -> bytes:
        """Encodes the instruction as a byte array"""
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
        sourceModes = [DataRegisterDirect, AddressRegisterDirect]
        destinationModes = [DataRegisterDirect]

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


# Utility functions


def byte_to_hex(byte: int) -> str:
    """Returns hexadecimal representation of a number with 0x prefix"""
    return f"{byte:#04x}"


# Encoding tests


def write_encoding_tests_header(f):
    header: list[str] = [
        "; Generated by llvm/test/MC/M68k/generate_tests.py\n",
        "; Please do not edit this file by hand.\n",
        "\n",
        "; RUN: llvm-mc -triple=m68k -mcpu=M68040 -show-encoding < %s | FileCheck %s\n",
    ]

    for line in header:
        f.write(line)


def write_encoding_test(f, instruction: Instruction):
    bytes = ",".join(map(byte_to_hex, instruction.bytes()))
    f.write(f"; CHECK:      {instruction.asm()}\n")
    f.write(f"; CHECK-SAME: encoding: [{bytes}]\n")
    f.write(f"{instruction.asm()}\n")


def write_encoding_tests(cls):
    name = cls.__name__
    with open(f"Generated/{name}.s", "w") as f:
        write_encoding_tests_header(f)
        for instruction in cls.permutations():
            f.write("\n")
            write_encoding_test(f, instruction)


for cls in [MOVE]:
    write_encoding_tests(cls)
