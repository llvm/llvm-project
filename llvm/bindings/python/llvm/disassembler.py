#===- disassembler.py - Python LLVM Bindings -----------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

from ctypes import CFUNCTYPE
from ctypes import POINTER
from ctypes import addressof
from ctypes import c_byte
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_size_t
from ctypes import c_ubyte
from ctypes import c_uint64
from ctypes import c_void_p
from ctypes import cast

from .common import LLVMObject
from .common import c_object_p
from .common import get_library

__all__ = [
    'Disassembler',
]

lib = get_library()
callbacks = {}

# Constants for set_options
Option_UseMarkup = 1



_initialized = False
_targets = ['AArch64', 'ARM', 'Hexagon', 'MSP430', 'Mips', 'NVPTX', 'PowerPC', 'R600', 'Sparc', 'SystemZ', 'X86', 'XCore']
def _ensure_initialized():
    global _initialized
    if not _initialized:
        # Here one would want to call the functions
        # LLVMInitializeAll{TargetInfo,TargetMC,Disassembler}s, but
        # unfortunately they are only defined as static inline
        # functions in the header files of llvm-c, so they don't exist
        # as symbols in the shared library.
        # So until that is fixed use this hack to initialize them all
        for tgt in _targets:
            for initializer in ("TargetInfo", "TargetMC", "Disassembler"):
                try:
                    f = getattr(lib, "LLVMInitialize" + tgt + initializer)
                except AttributeError:
                    continue
                f()
        _initialized = True


class Disassembler(LLVMObject):
    """Represents a disassembler instance.

    Disassembler instances are tied to specific "triple," which must be defined
    at creation time.

    Disassembler instances can disassemble instructions from multiple sources.
    """
    def __init__(self, triple):
        """Create a new disassembler instance.

        The triple argument is the triple to create the disassembler for. This
        is something like 'i386-apple-darwin9'.
        """

        _ensure_initialized()

        ptr = lib.LLVMCreateDisasm(c_char_p(triple), c_void_p(None), c_int(0),
                callbacks['op_info'](0), callbacks['symbol_lookup'](0))
        if not ptr:
            raise Exception('Could not obtain disassembler for triple: %s' %
                            triple)

        LLVMObject.__init__(self, ptr, disposer=lib.LLVMDisasmDispose)

    def get_instruction(self, source, pc=0):
        """Obtain the next instruction from an input source.

        The input source should be a str or bytearray or something that
        represents a sequence of bytes.

        This function will start reading bytes from the beginning of the
        source.

        The pc argument specifies the address that the first byte is at.

        This returns a 2-tuple of:

          long number of bytes read. 0 if no instruction was read.
          str representation of instruction. This will be the assembly that
            represents the instruction.
        """
        buf = cast(c_char_p(source), POINTER(c_ubyte))
        out_str = cast((c_byte * 255)(), c_char_p)

        result = lib.LLVMDisasmInstruction(self, buf, c_uint64(len(source)),
                                           c_uint64(pc), out_str, 255)

        return (result, out_str.value)

    def get_instructions(self, source, pc=0):
        """Obtain multiple instructions from an input source.

        This is like get_instruction() except it is a generator for all
        instructions within the source. It starts at the beginning of the
        source and reads instructions until no more can be read.

        This generator returns 3-tuple of:

          long address of instruction.
          long size of instruction, in bytes.
          str representation of instruction.
        """
        source_bytes = c_char_p(source)
        out_str = cast((c_byte * 255)(), c_char_p)

        # This could probably be written cleaner. But, it does work.
        buf = cast(source_bytes, POINTER(c_ubyte * len(source))).contents
        offset = 0
        address = pc
        end_address = pc + len(source)
        while address < end_address:
            b = cast(addressof(buf) + offset, POINTER(c_ubyte))
            result = lib.LLVMDisasmInstruction(self, b,
                    c_uint64(len(source) - offset), c_uint64(address),
                    out_str, 255)

            if result == 0:
                break

            yield (address, result, out_str.value)

            address += result
            offset += result

    def set_options(self, options):
        if not lib.LLVMSetDisasmOptions(self, options):
            raise Exception('Unable to set all disassembler options in %i' % options)


def register_library(library):
    library.LLVMCreateDisasm.argtypes = [c_char_p, c_void_p, c_int,
        callbacks['op_info'], callbacks['symbol_lookup']]
    library.LLVMCreateDisasm.restype = c_object_p

    library.LLVMDisasmDispose.argtypes = [Disassembler]

    library.LLVMDisasmInstruction.argtypes = [Disassembler, POINTER(c_ubyte),
            c_uint64, c_uint64, c_char_p, c_size_t]
    library.LLVMDisasmInstruction.restype = c_size_t

    library.LLVMSetDisasmOptions.argtypes = [Disassembler, c_uint64]
    library.LLVMSetDisasmOptions.restype = c_int


callbacks['op_info'] = CFUNCTYPE(c_int, c_void_p, c_uint64, c_uint64, c_uint64,
                                 c_int, c_void_p)
callbacks['symbol_lookup'] = CFUNCTYPE(c_char_p, c_void_p, c_uint64,
                                       POINTER(c_uint64), c_uint64,
                                       POINTER(c_char_p))

register_library(lib)
