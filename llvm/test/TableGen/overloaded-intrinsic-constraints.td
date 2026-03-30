// RUN: llvm-tblgen -gen-intrinsic-enums -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s --check-prefix=CHECK-ENUM
// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST_INTRINSICS_SUPPRESS_DEFS | FileCheck %s --check-prefix=CHECK-IMPL

include "llvm/IR/Intrinsics.td"

// Test AnyTypeOf constraints for overloaded intrinsics.
def int_anytypeof_multi : Intrinsic<
  [llvm_i32_ty],
  [
    AnyTypeOf<[llvm_i16_ty, llvm_i32_ty]>,
    AnyTypeOf<[llvm_anyint_ty, llvm_float_ty, llvm_double_ty]>
  ],
  [IntrNoMem]
>;

// Verify enum generation.
// CHECK-ENUM: anytypeof_multi = {{[0-9]+}}, // llvm.anytypeof.multi

// Verify encoding table generation.
// CHECK-IMPL: static constexpr unsigned char IIT_LongEncodingTable[] = {
// CHECK-IMPL-NEXT: /* 0 */ 4, 15, 5, 2, 3, 4, 15, 13, 3, 15, 9, 7, 8, 0,
// CHECK-IMPL-NEXT: 255
