// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-ENUM
// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s  | FileCheck %s --check-prefix=CHECK-IMPL
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DENABLE_ERROR 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

#define TEST_INTRINSICS_SUPPRESS_DEFS
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
// CHECK-IMPL-NEXT: 21, 0
// CHECK-IMPL-SAME: 15, 0, 1, 15, 1, 1, 0

// CHECK-IMPL-NEXT: 21, 7
// CHECK-IMPL-SAME: 15, 0, 1, 15, 1, 1, 15, 2, 1, 15, 3, 1, 15, 4, 1, 15, 5, 1
// CHECK-IMPL-SAME: 15, 6, 1, 15, 7, 1, 15, 8, 1, 0

// CHECK-IMPL-NEXT: 21, 8
// CHECK-IMPL-SAME: 15, 0, 1, 15, 1, 1, 15, 2, 1, 15, 3, 1, 15, 4, 1, 15, 5, 1
// CHECK-IMPL-SAME: 15, 6, 1, 15, 7, 1, 15, 8, 1, 15, 9, 1, 0

// CHECK-IMPL-NEXT: 21, 255
// There should be list of 257 '4's (IIT code for i32) followed by 0. We just
// match the suffix that ends with 0 (IIT_Done).
// CHECK-IMPL-SAME: 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0
def int_returns_a0_results : Intrinsic<
                             [],
                             [], []>;

def int_returns_b1_results : Intrinsic<
                             [llvm_anyint_ty],
                             [], []>;

def int_returns_c2_results : Intrinsic<
                             !listsplat(llvm_anyint_ty, 2),
                             [], []>;

def int_returns_d9_results : Intrinsic<
                             !listsplat(llvm_anyint_ty, 9),
                             [], []>;

def int_returns_e10_results : Intrinsic<
                              !listsplat(llvm_anyint_ty, 10),
                              [], []>;

def int_returns_f257_results : Intrinsic<
                               !listsplat(llvm_i32_ty, 257),
                               [], []>;

#ifdef ENABLE_ERROR
// CHECK-ERROR: error: intrinsics can only return upto 257 values, 'int_returns_g258_results' returns 258 values
// CHECK-ERROR-NEXT: def int_returns_g258_results : Intrinsic<
def int_returns_g258_results : Intrinsic<
                               !listsplat(llvm_i32_ty, 258),
                               [], []>;

#endif
