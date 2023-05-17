// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s

// This test is validating that it an Intrinsic with an LLVMPointerType to
// llvm_any_ty still properly work after r363233. That patch rewrote the
// substitution handling code in the Intrinsic Emitter, and didn't consider this
// case, so TableGen would hit an assertion in EncodeFixedType that was checking
// to ensure that the substitution being processed was correctly replaced.

include "llvm/IR/Intrinsics.td"

def int_has_ptr_to_any : Intrinsic<[LLVMPointerType<llvm_any_ty>, llvm_i8_ty]>;
// CHECK: /* 0 */ 21, 14, 15, 0, 2, 0
