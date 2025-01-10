// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s --check-prefix=CHECK-ENUM
// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS > /dev/null 2>&1
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DENABLE_ERROR 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// XFAIL: vg_leak

include "llvm/IR/Intrinsics.td"

// Make sure we can return up to 9 values.
// CHECK-ENUM: returns_9_results = {{[0-9]+}}, // llvm.returns.9.results
def int_returns_9_results : Intrinsic<
                              !listsplat(llvm_anyint_ty, 9),
                              [], [], "llvm.returns.9.results">;

#ifdef ENABLE_ERROR
// CHECK-ERROR: error: intrinsics can only return upto 9 values, 'int_returns_10_results' returns 10 values
// CHECK-ERROR-NEXT: def int_returns_10_results : Intrinsic<
def int_returns_10_results : Intrinsic<
                              !listsplat(llvm_anyint_ty, 10),
                              [], [], "llvm.returns.10.results">;

#endif
