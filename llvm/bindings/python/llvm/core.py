#===- core.py - Python LLVM Bindings -------------------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#

from .common import LLVMObject
from .common import c_object_p
from .common import get_library

from . import enumerations

from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p

__all__ = [
    "lib",
    "MemoryBuffer",
]

lib = get_library()

class OpCode(object):
    """Represents an individual OpCode enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return 'OpCode.%s' % self.name

    @staticmethod
    def from_value(value):
        """Obtain an OpCode instance from a numeric value."""
        result = OpCode._value_map.get(value, None)

        if result is None:
            raise ValueError('Unknown OpCode: %d' % value)

        return result

    @staticmethod
    def register(name, value):
        """Registers a new OpCode enumeration.

        This is called by this module for each enumeration defined in
        enumerations. You should not need to call this outside this module.
        """
        if value in OpCode._value_map:
            raise ValueError('OpCode value already registered: %d' % value)

        opcode = OpCode(name, value)
        OpCode._value_map[value] = opcode
        setattr(OpCode, name, opcode)

class MemoryBuffer(LLVMObject):
    """Represents an opaque memory buffer."""

    def __init__(self, filename=None):
        """Create a new memory buffer.

        Currently, we support creating from the contents of a file at the
        specified filename.
        """
        if filename is None:
            raise Exception("filename argument must be defined")

        memory = c_object_p()
        out = c_char_p(None)

        result = lib.LLVMCreateMemoryBufferWithContentsOfFile(filename,
                byref(memory), byref(out))

        if result:
            raise Exception("Could not create memory buffer: %s" % out.value)

        LLVMObject.__init__(self, memory, disposer=lib.LLVMDisposeMemoryBuffer)

def register_library(library):
    library.LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [c_char_p,
            POINTER(c_object_p), POINTER(c_char_p)]
    library.LLVMCreateMemoryBufferWithContentsOfFile.restype = bool

    library.LLVMDisposeMemoryBuffer.argtypes = [MemoryBuffer]

def register_enumerations():
    for name, value in enumerations.OpCodes:
        OpCode.register(name, value)

register_library(lib)
register_enumerations()
