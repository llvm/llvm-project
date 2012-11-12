from .base import TestBase

from ..disassembler import Disassembler

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

    def test_get_instructions(self):
        sequence = '\x67\xe3\x81\x01\xc7' # jcxz -127; addl %eax, %edi

        disassembler = Disassembler('i686-apple-darwin9')

        instructions = list(disassembler.get_instructions(sequence))
        self.assertEqual(len(instructions), 2)

        self.assertEqual(instructions[0], (0, 3, '\tjcxz\t-127'))
        self.assertEqual(instructions[1], (3, 2, '\taddl\t%eax, %edi'))
