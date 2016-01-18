
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
    out = c_char_p(None)
    result = lib.LLVMParseBitcode(mem_buffer, byref(module), byref(out))
    if result:
        raise RuntimeError('LLVM Error: %s' % out.value)
    m = Module(module)
    m.take_ownership(mem_buffer)
    return m

def register_library(library):
    library.LLVMParseBitcode.argtypes = [MemoryBuffer, POINTER(c_object_p), POINTER(c_char_p)]
    library.LLVMParseBitcode.restype = bool

register_library(lib)
