// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s
// XFAIL: vg_leak

include "llvm/IR/Intrinsics.td"

// Make sure we can return up to 8 values
// CHECK: returns_8_results = {{[0-9]+}}, // llvm.returns.8.results
def int_returns_8_results : Intrinsic<
    [llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty,
     llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty, llvm_anyint_ty],
     [], [], "llvm.returns.8.results">;
