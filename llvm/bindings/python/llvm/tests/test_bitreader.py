from __future__ import print_function

from .base import TestBase
from ..core import OpCode
from ..core import MemoryBuffer
from ..core import PassRegistry
from ..core import Context
from ..core import Module
from ..bit_reader import parse_bitcode

class TestBitReader(TestBase):

    def test_parse_bitcode(self):
        source = self.get_test_bc()
        m = parse_bitcode(MemoryBuffer(filename=source))
        print(m.target)
        print(m.datalayout)
