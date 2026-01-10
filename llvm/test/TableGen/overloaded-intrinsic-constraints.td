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

// Test mixed AnyTypeOf + NoneTypeOf.
def int_mixed_constraints : Intrinsic<
  [llvm_i32_ty],
  [
    AnyTypeOf<[llvm_i16_ty, llvm_i32_ty]>,
    NoneTypeOf<[llvm_i8_ty]>
  ],
  [IntrNoMem]
>;

// Test NoneTypeOf on return type.
def int_nonetypeof_return : Intrinsic<
  [NoneTypeOf<[llvm_i8_ty, llvm_double_ty]>],
  [llvm_i32_ty],
  [IntrNoMem]
>;

// Verify enum generation.
// CHECK-ENUM: anytypeof_multi = {{[0-9]+}}, // llvm.anytypeof.multi
// CHECK-ENUM: mixed_constraints{{.*}}// llvm.mixed.constraints
// CHECK-ENUM: nonetypeof_return{{.*}}// llvm.nonetypeof.return

// Verify encoding table generation.
// CHECK-IMPL: static constexpr unsigned char IIT_LongEncodingTable[] = {
// CHECK-IMPL-NEXT: /* 0 */ 4, 15, 0, 93, 2, 3, 4, 15, 8, 94, 1, 2, 0,
// CHECK-IMPL-NEXT: /* 13 */ 15, 0, 94, 2, 2, 8, 4, 0,
// CHECK-IMPL-NEXT: /* 21 */ 4, 15, 0, 93, 2, 3, 4, 15, 8, 93, 3, 15, 9, 7, 8, 0,
// CHECK-IMPL-NEXT: 255
