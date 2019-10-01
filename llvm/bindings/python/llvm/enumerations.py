#===- enumerations.py - Python LLVM Enumerations -------------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

r"""
LLVM Enumerations
=================

This file defines enumerations from LLVM.

Each enumeration is exposed as a list of 2-tuples. These lists are consumed by
dedicated types elsewhere in the package. The enumerations are centrally
defined in this file so they are easier to locate and maintain.
"""

__all__ = [
    'Attributes',
    'OpCodes',
    'TypeKinds',
    'Linkages',
    'Visibility',
    'CallConv',
    'IntPredicate',
    'RealPredicate',
    'LandingPadClauseTy',
]

Attributes = [
    ('ZExt', 1 << 0),
    ('MSExt', 1 << 1),
    ('NoReturn', 1 << 2),
    ('InReg', 1 << 3),
    ('StructRet', 1 << 4),
    ('NoUnwind', 1 << 5),
    ('NoAlias', 1 << 6),
    ('ByVal', 1 << 7),
    ('Nest', 1 << 8),
    ('ReadNone', 1 << 9),
    ('ReadOnly', 1 << 10),
    ('NoInline', 1 << 11),
    ('AlwaysInline', 1 << 12),
    ('OptimizeForSize', 1 << 13),
    ('StackProtect', 1 << 14),
    ('StackProtectReq', 1 << 15),
    ('Alignment', 31 << 16),
    ('NoCapture', 1 << 21),
    ('NoRedZone', 1 << 22),
    ('ImplicitFloat', 1 << 23),
    ('Naked', 1 << 24),
    ('InlineHint', 1 << 25),
    ('StackAlignment', 7 << 26),
    ('ReturnsTwice', 1 << 29),
    ('UWTable', 1 << 30),
    ('NonLazyBind', 1 << 31),
]

OpCodes = [
    ('Ret', 1),
    ('Br', 2),
    ('Switch', 3),
    ('IndirectBr', 4),
    ('Invoke', 5),
    ('Unreachable', 7),
    ('Add', 8),
    ('FAdd', 9),
    ('Sub', 10),
    ('FSub', 11),
    ('Mul', 12),
    ('FMul', 13),
    ('UDiv', 14),
    ('SDiv', 15),
    ('FDiv', 16),
    ('URem', 17),
    ('SRem', 18),
    ('FRem', 19),
    ('Shl', 20),
    ('LShr', 21),
    ('AShr', 22),
    ('And', 23),
    ('Or', 24),
    ('Xor', 25),
    ('Alloca', 26),
    ('Load', 27),
    ('Store', 28),
    ('GetElementPtr', 29),
    ('Trunc', 30),
    ('ZExt', 31),
    ('SExt', 32),
    ('FPToUI', 33),
    ('FPToSI', 34),
    ('UIToFP', 35),
    ('SIToFP', 36),
    ('FPTrunc', 37),
    ('FPExt', 38),
    ('PtrToInt', 39),
    ('IntToPtr', 40),
    ('BitCast', 41),
    ('ICmp', 42),
    ('FCmpl', 43),
    ('PHI', 44),
    ('Call', 45),
    ('Select', 46),
    ('UserOp1', 47),
    ('UserOp2', 48),
    ('AArg', 49),
    ('ExtractElement', 50),
    ('InsertElement', 51),
    ('ShuffleVector', 52),
    ('ExtractValue', 53),
    ('InsertValue', 54),
    ('Fence', 55),
    ('AtomicCmpXchg', 56),
    ('AtomicRMW', 57),
    ('Resume', 58),
    ('LandingPad', 59),
]

TypeKinds = [
    ('Void', 0),
    ('Half', 1),
    ('Float', 2),
    ('Double', 3),
    ('X86_FP80', 4),
    ('FP128', 5),
    ('PPC_FP128', 6),
    ('Label', 7),
    ('Integer', 8),
    ('Function', 9),
    ('Struct', 10),
    ('Array', 11),
    ('Pointer', 12),
    ('Vector', 13),
    ('Metadata', 14),
    ('X86_MMX', 15),
]

Linkages = [
    ('External', 0),
    ('AvailableExternally', 1),
    ('LinkOnceAny', 2),
    ('LinkOnceODR', 3),
    ('WeakAny', 4),
    ('WeakODR', 5),
    ('Appending', 6),
    ('Internal', 7),
    ('Private', 8),
    ('DLLImport', 9),
    ('DLLExport', 10),
    ('ExternalWeak', 11),
    ('Ghost', 12),
    ('Common', 13),
    ('LinkerPrivate', 14),
    ('LinkerPrivateWeak', 15),
    ('LinkerPrivateWeakDefAuto', 16),
]

Visibility = [
    ('Default', 0),
    ('Hidden', 1),
    ('Protected', 2),
]

CallConv = [
    ('CCall', 0),
    ('FastCall', 8),
    ('ColdCall', 9),
    ('X86StdcallCall', 64),
    ('X86FastcallCall', 65),
]

IntPredicate = [
    ('EQ', 32),
    ('NE', 33),
    ('UGT', 34),
    ('UGE', 35),
    ('ULT', 36),
    ('ULE', 37),
    ('SGT', 38),
    ('SGE', 39),
    ('SLT', 40),
    ('SLE', 41),
]

RealPredicate = [
    ('PredicateFalse', 0),
    ('OEQ', 1),
    ('OGT', 2),
    ('OGE', 3),
    ('OLT', 4),
    ('OLE', 5),
    ('ONE', 6),
    ('ORD', 7),
    ('UNO', 8),
    ('UEQ', 9),
    ('UGT', 10),
    ('UGE', 11),
    ('ULT', 12),
    ('ULE', 13),
    ('UNE', 14),
    ('PredicateTrue', 15),
]

LandingPadClauseTy = [
    ('Catch', 0),
    ('Filter', 1),
]
