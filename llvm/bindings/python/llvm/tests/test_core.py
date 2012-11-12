from .base import TestBase
from ..core import OpCode
from ..core import MemoryBuffer

class TestCore(TestBase):
    def test_opcode(self):
        self.assertTrue(hasattr(OpCode, 'Ret'))
        self.assertTrue(isinstance(OpCode.Ret, OpCode))
        self.assertEqual(OpCode.Ret.value, 1)

        op = OpCode.from_value(1)
        self.assertTrue(isinstance(op, OpCode))
        self.assertEqual(op, OpCode.Ret)

    def test_memory_buffer_create_from_file(self):
        source = self.get_test_binary()

        MemoryBuffer(filename=source)

    def test_memory_buffer_failing(self):
        with self.assertRaises(Exception):
            MemoryBuffer(filename="/hopefully/this/path/doesnt/exist")

