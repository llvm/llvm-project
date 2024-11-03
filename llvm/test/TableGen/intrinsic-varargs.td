// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s
// XFAIL: vg_leak

include "llvm/IR/Intrinsics.td"

// CHECK: /* 0 */ 0, 29, 0,
def int_foo : Intrinsic<[], [llvm_vararg_ty]>;
