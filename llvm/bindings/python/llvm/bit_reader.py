
from .common import LLVMObject
from .common import c_object_p
from .common import get_library
from . import enumerations
from .core import MemoryBuffer
from .core import Module
from .core import OpCode
from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p
from ctypes import cast
__all__ = ['parse_bitcode']
lib = get_library()

def parse_bitcode(mem_buffer):
    """Input is .core.MemoryBuffer"""
    module = c_object_p()
    result = lib.LLVMParseBitcode2(mem_buffer, byref(module))
    if result:
        raise RuntimeError('LLVM Error')
    m = Module(module)
    m.take_ownership(mem_buffer)
    return m

def register_library(library):
    library.LLVMParseBitcode2.argtypes = [MemoryBuffer, POINTER(c_object_p)]
    library.LLVMParseBitcode2.restype = bool

register_library(lib)
