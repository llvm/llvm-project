from __future__ import print_function

from .base import TestBase

from ..disassembler import Disassembler, Option_UseMarkup

class TestDisassembler(TestBase):
    def test_instantiate(self):
         Disassembler('i686-apple-darwin9')

    def test_basic(self):
        sequence = '\x67\xe3\x81' # jcxz -127
        triple = 'i686-apple-darwin9'

        disassembler = Disassembler(triple)

        count, s = disassembler.get_instruction(sequence)
        self.assertEqual(count, 3)
        self.assertEqual(s, '\tjcxz\t-127')

    def test_nonexistent_triple(self):
        with self.assertRaisesRegex(Exception, "Could not obtain disassembler for triple"):
            Disassembler("nonexistent-triple-raises")

    def test_get_instructions(self):
        sequence = '\x67\xe3\x81\x01\xc7' # jcxz -127; addl %eax, %edi

        disassembler = Disassembler('i686-apple-darwin9')

        instructions = list(disassembler.get_instructions(sequence))
        self.assertEqual(len(instructions), 2)

        self.assertEqual(instructions[0], (0, 3, '\tjcxz\t-127'))
        self.assertEqual(instructions[1], (3, 2, '\taddl\t%eax, %edi'))

    def test_set_options(self):
        sequence = '\x10\x40\x2d\xe9'
        triple = 'arm-linux-android'

        disassembler = Disassembler(triple)
        disassembler.set_options(Option_UseMarkup)
        count, s = disassembler.get_instruction(sequence)
        print(s)
        self.assertEqual(count, 4)
        self.assertEqual(s, '\tpush\t{<reg:r4>, <reg:lr>}')
