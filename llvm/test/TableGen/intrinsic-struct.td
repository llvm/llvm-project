// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s --check-prefix=CHECK-ENUM
// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s --check-prefix=CHECK-IMPL
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS -DENABLE_ERROR 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// XFAIL: vg_leak

include "llvm/IR/Intrinsics.td"

// Make sure we can return up to 257 values. Intrinsics are in alphabetical order.
// CHECK-ENUM: returns_a0_results = {{[0-9]+}}, // llvm.returns.a0.results
// CHECK-ENUM: returns_b1_results, // llvm.returns.b1.results
// CHECK-ENUM: returns_c2_results, // llvm.returns.c2.results
// CHECK-ENUM: returns_d9_results, // llvm.returns.d9.results
// CHECK-ENUM: returns_e10_results, // llvm.returns.e10.results
// CHECK-ENUM: returns_f257_results, // llvm.returns.f257.results

// Make sure the encoding table is correctly generated.
// CHECK-IMPL: IIT_LongEncodingTable
// CHECK-IMPL-NEXT: 21, 255
// CHECK-IMPL-SAME: 15, 1, 15, 9, 15, 17, 15, 25, 15, 33, 15, 41, 15, 49, 15, 57, 15, 65, 15, 73, 15, 81,
// CHECK-IMPL-NEXT: 21, 0
// CHECK-IMPL-SAME: 15, 1, 15, 9, 0
// CHECK-IMPL-NEXT: 21, 7
// CHECK-IMPL-SAME: 15, 1, 15, 9, 15, 17, 15, 25, 15, 33, 15, 41, 15, 49, 15, 57, 15, 65, 0
// CHECK-IMPL-NEXT: 21, 8
// CHECK-IMPL-SAME: 15, 1, 15, 9, 15, 17, 15, 25, 15, 33, 15, 41, 15, 49, 15, 57, 15, 65, 15, 73, 0
def int_returns_a0_results : Intrinsic<
                             [],
                             [], [], "llvm.returns.a0.results">;

def int_returns_b1_results : Intrinsic<
                             [llvm_anyint_ty],
                             [], [], "llvm.returns.b1.results">;

def int_returns_c2_results : Intrinsic<
                             !listsplat(llvm_anyint_ty, 2),
                             [], [], "llvm.returns.c2.results">;

def int_returns_d9_results : Intrinsic<
                             !listsplat(llvm_anyint_ty, 9),
                             [], [], "llvm.returns.d9.results">;

def int_returns_e10_results : Intrinsic<
                              !listsplat(llvm_anyint_ty, 10),
                              [], [], "llvm.returns.e10.results">;

def int_returns_f257_results : Intrinsic<
                               !listsplat(llvm_anyint_ty, 257),
                               [], [], "llvm.returns.f257.results">;

#ifdef ENABLE_ERROR
// CHECK-ERROR: error: intrinsics can only return upto 257 values, 'int_returns_g258_results' returns 258 values
// CHECK-ERROR-NEXT: def int_returns_g258_results : Intrinsic<
def int_returns_g258_results : Intrinsic<
                               !listsplat(llvm_anyint_ty, 258),
                               [], [], "llvm.returns.g258.results">;

#endif
