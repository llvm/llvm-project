//===-- PtrEnums.td - Ptr dialect enumerations -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_ENUMS
#define PTR_ENUMS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Atomic binary op enum attribute
//===----------------------------------------------------------------------===//

def AtomicBinOpXchg : I64EnumAttrCase<"xchg", 0, "xchg">;
def AtomicBinOpAdd  : I64EnumAttrCase<"add", 1, "add">;
def AtomicBinOpSub  : I64EnumAttrCase<"sub", 2, "sub">;
def AtomicBinOpAnd  : I64EnumAttrCase<"_and", 3, "_and">;
def AtomicBinOpNand : I64EnumAttrCase<"nand", 4, "nand">;
def AtomicBinOpOr   : I64EnumAttrCase<"_or", 5, "_or">;
def AtomicBinOpXor  : I64EnumAttrCase<"_xor", 6, "_xor">;
def AtomicBinOpMax  : I64EnumAttrCase<"max", 7, "max">;
def AtomicBinOpMin  : I64EnumAttrCase<"min", 8, "min">;
def AtomicBinOpUMax : I64EnumAttrCase<"umax", 9, "umax">;
def AtomicBinOpUMin : I64EnumAttrCase<"umin", 10, "umin">;
def AtomicBinOpFAdd : I64EnumAttrCase<"fadd", 11, "fadd">;
def AtomicBinOpFSub : I64EnumAttrCase<"fsub", 12, "fsub">;
def AtomicBinOpFMax : I64EnumAttrCase<"fmax", 13, "fmax">;
def AtomicBinOpFMin : I64EnumAttrCase<"fmin", 14, "fmin">;
def AtomicBinOpUIncWrap : I64EnumAttrCase<"uinc_wrap", 15, "uinc_wrap">;
def AtomicBinOpUDecWrap : I64EnumAttrCase<"udec_wrap", 16, "udec_wrap">;

def AtomicBinOp : I64EnumAttr<
    "AtomicBinOp",
    "ptr.atomicrmw binary operations",
    [AtomicBinOpXchg, AtomicBinOpAdd, AtomicBinOpSub, AtomicBinOpAnd,
     AtomicBinOpNand, AtomicBinOpOr, AtomicBinOpXor, AtomicBinOpMax,
     AtomicBinOpMin, AtomicBinOpUMax, AtomicBinOpUMin, AtomicBinOpFAdd,
     AtomicBinOpFSub, AtomicBinOpFMax, AtomicBinOpFMin, AtomicBinOpUIncWrap,
     AtomicBinOpUDecWrap]> {
  let cppNamespace = "::mlir::ptr";
}

//===----------------------------------------------------------------------===//
// Atomic ordering enum attribute
//===----------------------------------------------------------------------===//

def AtomicOrderingNotAtomic : I64EnumAttrCase<"not_atomic", 0, "not_atomic">;
def AtomicOrderingUnordered : I64EnumAttrCase<"unordered", 1, "unordered">;
def AtomicOrderingMonotonic : I64EnumAttrCase<"monotonic", 2, "monotonic">;
def AtomicOrderingAcquire   : I64EnumAttrCase<"acquire", 3, "acquire">;
def AtomicOrderingRelease   : I64EnumAttrCase<"release", 4, "release">;
def AtomicOrderingAcqRel    : I64EnumAttrCase<"acq_rel", 5, "acq_rel">;
def AtomicOrderingSeqCst    : I64EnumAttrCase<"seq_cst", 6, "seq_cst">;

def AtomicOrdering : I64EnumAttr<
    "AtomicOrdering",
    "Atomic ordering for LLVM's memory model",
    [AtomicOrderingNotAtomic, AtomicOrderingUnordered, AtomicOrderingMonotonic,
     AtomicOrderingAcquire, AtomicOrderingRelease, AtomicOrderingAcqRel,
     AtomicOrderingSeqCst
    ]> {
  let cppNamespace = "::mlir::ptr";
}

#endif // PTR_ENUMS
