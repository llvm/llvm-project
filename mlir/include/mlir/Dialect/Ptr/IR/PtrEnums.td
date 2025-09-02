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
// Atomic binary op enum attribute.
//===----------------------------------------------------------------------===//

def AtomicBinOpXchg : I32EnumCase<"xchg", 0, "xchg">;
def AtomicBinOpAdd  : I32EnumCase<"add", 1, "add">;
def AtomicBinOpSub  : I32EnumCase<"sub", 2, "sub">;
def AtomicBinOpAnd  : I32EnumCase<"_and", 3, "_and">;
def AtomicBinOpNand : I32EnumCase<"nand", 4, "nand">;
def AtomicBinOpOr   : I32EnumCase<"_or", 5, "_or">;
def AtomicBinOpXor  : I32EnumCase<"_xor", 6, "_xor">;
def AtomicBinOpMax  : I32EnumCase<"max", 7, "max">;
def AtomicBinOpMin  : I32EnumCase<"min", 8, "min">;
def AtomicBinOpUMax : I32EnumCase<"umax", 9, "umax">;
def AtomicBinOpUMin : I32EnumCase<"umin", 10, "umin">;
def AtomicBinOpFAdd : I32EnumCase<"fadd", 11, "fadd">;
def AtomicBinOpFSub : I32EnumCase<"fsub", 12, "fsub">;
def AtomicBinOpFMax : I32EnumCase<"fmax", 13, "fmax">;
def AtomicBinOpFMin : I32EnumCase<"fmin", 14, "fmin">;
def AtomicBinOpUIncWrap : I32EnumCase<"uinc_wrap", 15, "uinc_wrap">;
def AtomicBinOpUDecWrap : I32EnumCase<"udec_wrap", 16, "udec_wrap">;

def AtomicBinOp : I32Enum<
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
// Atomic ordering enum attribute.
//===----------------------------------------------------------------------===//

def AtomicOrderingNotAtomic : I32EnumCase<"not_atomic", 0, "not_atomic">;
def AtomicOrderingUnordered : I32EnumCase<"unordered", 1, "unordered">;
def AtomicOrderingMonotonic : I32EnumCase<"monotonic", 2, "monotonic">;
def AtomicOrderingAcquire   : I32EnumCase<"acquire", 3, "acquire">;
def AtomicOrderingRelease   : I32EnumCase<"release", 4, "release">;
def AtomicOrderingAcqRel    : I32EnumCase<"acq_rel", 5, "acq_rel">;
def AtomicOrderingSeqCst    : I32EnumCase<"seq_cst", 6, "seq_cst">;

def AtomicOrdering : I32Enum<
    "AtomicOrdering",
    "Atomic ordering for LLVM's memory model",
    [AtomicOrderingNotAtomic, AtomicOrderingUnordered, AtomicOrderingMonotonic,
     AtomicOrderingAcquire, AtomicOrderingRelease, AtomicOrderingAcqRel,
     AtomicOrderingSeqCst
    ]> {
  let cppNamespace = "::mlir::ptr";
}

def AtomicOrderingProp : EnumProp<AtomicOrdering>;

//===----------------------------------------------------------------------===//
// Ptr add flags enum properties.
//===----------------------------------------------------------------------===//

def Ptr_PtrAddFlags : I32Enum<"PtrAddFlags", "Pointer add flags", [
    I32EnumCase<"none", 0>, I32EnumCase<"nusw", 1>, I32EnumCase<"nuw", 2>,
    I32EnumCase<"inbounds", 3>
  ]> {
  let cppNamespace = "::mlir::ptr";
}

#endif // PTR_ENUMS
