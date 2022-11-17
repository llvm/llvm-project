#===- core.py - Python LLVM Bindings -------------------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
from __future__ import print_function

from .common import LLVMObject
from .common import c_object_p
from .common import get_library

from . import enumerations

from ctypes import POINTER
from ctypes import byref
from ctypes import c_char_p
from ctypes import c_uint

import sys

__all__ = [
    "lib",
    "Enums",
    "OpCode",
    "MemoryBuffer",
    "Module",
    "Value",
    "Function",
    "BasicBlock",
    "Instruction",
    "Context",
    "PassRegistry"
]

lib = get_library()
Enums = []

class LLVMEnumeration(object):
    """Represents an individual LLVM enumeration."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return '%s.%s' % (self.__class__.__name__,
                          self.name)

    @classmethod
    def from_value(cls, value):
        """Obtain an enumeration instance from a numeric value."""
        result = cls._value_map.get(value, None)

        if result is None:
            raise ValueError('Unknown %s: %d' % (cls.__name__,
                                                 value))

        return result

    @classmethod
    def register(cls, name, value):
        """Registers a new enumeration.

        This is called by this module for each enumeration defined in
        enumerations. You should not need to call this outside this module.
        """
        if value in cls._value_map:
            raise ValueError('%s value already registered: %d' % (cls.__name__,
                                                                  value))
        enum = cls(name, value)
        cls._value_map[value] = enum
        setattr(cls, name, enum)

class Attribute(LLVMEnumeration):
    """Represents an individual Attribute enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(Attribute, self).__init__(name, value)

class OpCode(LLVMEnumeration):
    """Represents an individual OpCode enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(OpCode, self).__init__(name, value)

class TypeKind(LLVMEnumeration):
    """Represents an individual TypeKind enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(TypeKind, self).__init__(name, value)

class Linkage(LLVMEnumeration):
    """Represents an individual Linkage enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(Linkage, self).__init__(name, value)

class Visibility(LLVMEnumeration):
    """Represents an individual visibility enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(Visibility, self).__init__(name, value)

class CallConv(LLVMEnumeration):
    """Represents an individual calling convention enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(CallConv, self).__init__(name, value)

class IntPredicate(LLVMEnumeration):
    """Represents an individual IntPredicate enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(IntPredicate, self).__init__(name, value)

class RealPredicate(LLVMEnumeration):
    """Represents an individual RealPredicate enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(RealPredicate, self).__init__(name, value)

class LandingPadClauseTy(LLVMEnumeration):
    """Represents an individual LandingPadClauseTy enumeration."""

    _value_map = {}

    def __init__(self, name, value):
        super(LandingPadClauseTy, self).__init__(name, value)

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

    def __len__(self):
        return lib.LLVMGetBufferSize(self)

class Value(LLVMObject):
    
    def __init__(self, value):
        LLVMObject.__init__(self, value)

    @property
    def name(self):
        return lib.LLVMGetValueName(self)

    def dump(self):
        lib.LLVMDumpValue(self)
    
    def get_operand(self, i):
        return Value(lib.LLVMGetOperand(self, i))
    
    def set_operand(self, i, v):
        return lib.LLVMSetOperand(self, i, v)
    
    def __len__(self):
        return lib.LLVMGetNumOperands(self)

class Module(LLVMObject):
    """Represents the top-level structure of an llvm program in an opaque object."""

    def __init__(self, module, name=None, context=None):
        LLVMObject.__init__(self, module, disposer=lib.LLVMDisposeModule)

    @classmethod
    def CreateWithName(cls, module_id):
        m = Module(lib.LLVMModuleCreateWithName(module_id))
        Context.GetGlobalContext().take_ownership(m)
        return m

    @property
    def datalayout(self):
        return lib.LLVMGetDataLayout(self)

    @datalayout.setter
    def datalayout(self, new_data_layout):
        """new_data_layout is a string."""
        lib.LLVMSetDataLayout(self, new_data_layout)

    @property
    def target(self):
        return lib.LLVMGetTarget(self)

    @target.setter
    def target(self, new_target):
        """new_target is a string."""
        lib.LLVMSetTarget(self, new_target)

    def dump(self):
        lib.LLVMDumpModule(self)

    class __function_iterator(object):
        def __init__(self, module, reverse=False):
            self.module = module
            self.reverse = reverse
            if self.reverse:
                self.function = self.module.last
            else:
                self.function = self.module.first
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if not isinstance(self.function, Function):
                raise StopIteration("")
            result = self.function
            if self.reverse:
                self.function = self.function.prev
            else:
                self.function = self.function.next
            return result

        if sys.version_info.major == 2:
            next = __next__

    def __iter__(self):
        return Module.__function_iterator(self)

    def __reversed__(self):
        return Module.__function_iterator(self, reverse=True)

    @property
    def first(self):
        return Function(lib.LLVMGetFirstFunction(self))

    @property
    def last(self):
        return Function(lib.LLVMGetLastFunction(self))

    def print_module_to_file(self, filename):
        out = c_char_p(None)
        # Result is inverted so 0 means everything was ok.
        result = lib.LLVMPrintModuleToFile(self, filename, byref(out))        
        if result:
            raise RuntimeError("LLVM Error: %s" % out.value)

class Function(Value):

    def __init__(self, value):
        Value.__init__(self, value)
    
    @property
    def next(self):
        f = lib.LLVMGetNextFunction(self)
        return f and Function(f)
    
    @property
    def prev(self):
        f = lib.LLVMGetPreviousFunction(self)
        return f and Function(f)
    
    @property
    def first(self):
        b = lib.LLVMGetFirstBasicBlock(self)
        return b and BasicBlock(b)

    @property
    def last(self):
        b = lib.LLVMGetLastBasicBlock(self)
        return b and BasicBlock(b)

    class __bb_iterator(object):
        def __init__(self, function, reverse=False):
            self.function = function
            self.reverse = reverse
            if self.reverse:
                self.bb = function.last
            else:
                self.bb = function.first
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if not isinstance(self.bb, BasicBlock):
                raise StopIteration("")
            result = self.bb
            if self.reverse:
                self.bb = self.bb.prev
            else:
                self.bb = self.bb.next
            return result

        if sys.version_info.major == 2:
            next = __next__
    
    def __iter__(self):
        return Function.__bb_iterator(self)

    def __reversed__(self):
        return Function.__bb_iterator(self, reverse=True)
    
    def __len__(self):
        return lib.LLVMCountBasicBlocks(self)

class BasicBlock(LLVMObject):
    
    def __init__(self, value):
        LLVMObject.__init__(self, value)

    @property
    def next(self):
        b = lib.LLVMGetNextBasicBlock(self)
        return b and BasicBlock(b)

    @property
    def prev(self):
        b = lib.LLVMGetPreviousBasicBlock(self)
        return b and BasicBlock(b)
    
    @property
    def first(self):
        i = lib.LLVMGetFirstInstruction(self)
        return i and Instruction(i)

    @property
    def last(self):
        i = lib.LLVMGetLastInstruction(self)
        return i and Instruction(i)

    def __as_value(self):
        return Value(lib.LLVMBasicBlockAsValue(self))
    
    @property
    def name(self):
        return lib.LLVMGetValueName(self.__as_value())

    def dump(self):
        lib.LLVMDumpValue(self.__as_value())

    def get_operand(self, i):
        return Value(lib.LLVMGetOperand(self.__as_value(),
                                        i))
    
    def set_operand(self, i, v):
        return lib.LLVMSetOperand(self.__as_value(),
                                  i, v)
    
    def __len__(self):
        return lib.LLVMGetNumOperands(self.__as_value())

    class __inst_iterator(object):
        def __init__(self, bb, reverse=False):            
            self.bb = bb
            self.reverse = reverse
            if self.reverse:
                self.inst = self.bb.last
            else:
                self.inst = self.bb.first
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if not isinstance(self.inst, Instruction):
                raise StopIteration("")
            result = self.inst
            if self.reverse:
                self.inst = self.inst.prev
            else:
                self.inst = self.inst.next
            return result

        if sys.version_info.major == 2:
            next = __next__

    def __iter__(self):
        return BasicBlock.__inst_iterator(self)

    def __reversed__(self):
        return BasicBlock.__inst_iterator(self, reverse=True)


class Instruction(Value):

    def __init__(self, value):
        Value.__init__(self, value)

    @property
    def next(self):
        i = lib.LLVMGetNextInstruction(self)
        return i and Instruction(i)

    @property
    def prev(self):
        i = lib.LLVMGetPreviousInstruction(self)
        return i and Instruction(i)

    @property
    def opcode(self):
        return OpCode.from_value(lib.LLVMGetInstructionOpcode(self))

class Context(LLVMObject):

    def __init__(self, context=None):
        if context is None:
            context = lib.LLVMContextCreate()
            LLVMObject.__init__(self, context, disposer=lib.LLVMContextDispose)
        else:
            LLVMObject.__init__(self, context)

    @classmethod
    def GetGlobalContext(cls):
        return Context(lib.LLVMGetGlobalContext())

class PassRegistry(LLVMObject):
    """Represents an opaque pass registry object."""

    def __init__(self):
        LLVMObject.__init__(self,
                            lib.LLVMGetGlobalPassRegistry())

def register_library(library):
    # Initialization/Shutdown declarations.
    library.LLVMInitializeCore.argtypes = [PassRegistry]
    library.LLVMInitializeCore.restype = None

    library.LLVMInitializeTransformUtils.argtypes = [PassRegistry]
    library.LLVMInitializeTransformUtils.restype = None

    library.LLVMInitializeScalarOpts.argtypes = [PassRegistry]
    library.LLVMInitializeScalarOpts.restype = None

    library.LLVMInitializeVectorization.argtypes = [PassRegistry]
    library.LLVMInitializeVectorization.restype = None

    library.LLVMInitializeInstCombine.argtypes = [PassRegistry]
    library.LLVMInitializeInstCombine.restype = None

    library.LLVMInitializeIPO.argtypes = [PassRegistry]
    library.LLVMInitializeIPO.restype = None

    library.LLVMInitializeAnalysis.argtypes = [PassRegistry]
    library.LLVMInitializeAnalysis.restype = None

    library.LLVMInitializeCodeGen.argtypes = [PassRegistry]
    library.LLVMInitializeCodeGen.restype = None

    library.LLVMInitializeTarget.argtypes = [PassRegistry]
    library.LLVMInitializeTarget.restype = None

    library.LLVMShutdown.argtypes = []
    library.LLVMShutdown.restype = None

    # Pass Registry declarations.
    library.LLVMGetGlobalPassRegistry.argtypes = []
    library.LLVMGetGlobalPassRegistry.restype = c_object_p

    # Context declarations.
    library.LLVMContextCreate.argtypes = []
    library.LLVMContextCreate.restype = c_object_p

    library.LLVMContextDispose.argtypes = [Context]
    library.LLVMContextDispose.restype = None

    library.LLVMGetGlobalContext.argtypes = []
    library.LLVMGetGlobalContext.restype = c_object_p

    # Memory buffer declarations
    library.LLVMCreateMemoryBufferWithContentsOfFile.argtypes = [c_char_p,
            POINTER(c_object_p), POINTER(c_char_p)]
    library.LLVMCreateMemoryBufferWithContentsOfFile.restype = bool

    library.LLVMGetBufferSize.argtypes = [MemoryBuffer]

    library.LLVMDisposeMemoryBuffer.argtypes = [MemoryBuffer]

    # Module declarations
    library.LLVMModuleCreateWithName.argtypes = [c_char_p]
    library.LLVMModuleCreateWithName.restype = c_object_p

    library.LLVMDisposeModule.argtypes = [Module]
    library.LLVMDisposeModule.restype = None

    library.LLVMGetDataLayout.argtypes = [Module]
    library.LLVMGetDataLayout.restype = c_char_p

    library.LLVMSetDataLayout.argtypes = [Module, c_char_p]
    library.LLVMSetDataLayout.restype = None

    library.LLVMGetTarget.argtypes = [Module]
    library.LLVMGetTarget.restype = c_char_p

    library.LLVMSetTarget.argtypes = [Module, c_char_p]
    library.LLVMSetTarget.restype = None

    library.LLVMDumpModule.argtypes = [Module]
    library.LLVMDumpModule.restype = None

    library.LLVMPrintModuleToFile.argtypes = [Module, c_char_p,
                                              POINTER(c_char_p)]
    library.LLVMPrintModuleToFile.restype = bool

    library.LLVMGetFirstFunction.argtypes = [Module]
    library.LLVMGetFirstFunction.restype = c_object_p

    library.LLVMGetLastFunction.argtypes = [Module]
    library.LLVMGetLastFunction.restype = c_object_p

    library.LLVMGetNextFunction.argtypes = [Function]
    library.LLVMGetNextFunction.restype = c_object_p

    library.LLVMGetPreviousFunction.argtypes = [Function]
    library.LLVMGetPreviousFunction.restype = c_object_p

    # Value declarations.
    library.LLVMGetValueName.argtypes = [Value]
    library.LLVMGetValueName.restype = c_char_p

    library.LLVMDumpValue.argtypes = [Value]
    library.LLVMDumpValue.restype = None

    library.LLVMGetOperand.argtypes = [Value, c_uint]
    library.LLVMGetOperand.restype = c_object_p

    library.LLVMSetOperand.argtypes = [Value, Value, c_uint]
    library.LLVMSetOperand.restype = None

    library.LLVMGetNumOperands.argtypes = [Value]
    library.LLVMGetNumOperands.restype = c_uint

    # Basic Block Declarations.
    library.LLVMGetFirstBasicBlock.argtypes = [Function]
    library.LLVMGetFirstBasicBlock.restype = c_object_p

    library.LLVMGetLastBasicBlock.argtypes = [Function]
    library.LLVMGetLastBasicBlock.restype = c_object_p

    library.LLVMGetNextBasicBlock.argtypes = [BasicBlock]
    library.LLVMGetNextBasicBlock.restype = c_object_p

    library.LLVMGetPreviousBasicBlock.argtypes = [BasicBlock]
    library.LLVMGetPreviousBasicBlock.restype = c_object_p

    library.LLVMGetFirstInstruction.argtypes = [BasicBlock]
    library.LLVMGetFirstInstruction.restype = c_object_p

    library.LLVMGetLastInstruction.argtypes = [BasicBlock]
    library.LLVMGetLastInstruction.restype = c_object_p

    library.LLVMBasicBlockAsValue.argtypes = [BasicBlock]
    library.LLVMBasicBlockAsValue.restype = c_object_p

    library.LLVMCountBasicBlocks.argtypes = [Function]
    library.LLVMCountBasicBlocks.restype = c_uint

    # Instruction Declarations.
    library.LLVMGetNextInstruction.argtypes = [Instruction]
    library.LLVMGetNextInstruction.restype = c_object_p

    library.LLVMGetPreviousInstruction.argtypes = [Instruction]
    library.LLVMGetPreviousInstruction.restype = c_object_p

    library.LLVMGetInstructionOpcode.argtypes = [Instruction]
    library.LLVMGetInstructionOpcode.restype = c_uint

def register_enumerations():
    if Enums:
        return None
    enums = [
        (Attribute, enumerations.Attributes),
        (OpCode, enumerations.OpCodes),
        (TypeKind, enumerations.TypeKinds),
        (Linkage, enumerations.Linkages),
        (Visibility, enumerations.Visibility),
        (CallConv, enumerations.CallConv),
        (IntPredicate, enumerations.IntPredicate),
        (RealPredicate, enumerations.RealPredicate),
        (LandingPadClauseTy, enumerations.LandingPadClauseTy),
    ]
    for enum_class, enum_spec in enums:
        for name, value in enum_spec:
            print(name, value)
            enum_class.register(name, value)
    return enums

def initialize_llvm():
    Context.GetGlobalContext()
    p = PassRegistry()
    lib.LLVMInitializeCore(p)
    lib.LLVMInitializeTransformUtils(p)
    lib.LLVMInitializeScalarOpts(p)
    lib.LLVMInitializeVectorization(p)
    lib.LLVMInitializeInstCombine(p)
    lib.LLVMInitializeIPO(p)
    lib.LLVMInitializeAnalysis(p)
    lib.LLVMInitializeCodeGen(p)
    lib.LLVMInitializeTarget(p)

register_library(lib)
Enums = register_enumerations()
initialize_llvm()
