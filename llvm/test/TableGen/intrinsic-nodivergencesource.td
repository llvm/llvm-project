// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_a_none : Intrinsic<[llvm_i32_ty], [], [Throws]>;
def int_b_uniform : Intrinsic<[llvm_i32_ty], [], [Throws, IntrNoDivergenceSource]>;
def int_c_uniform : Intrinsic<[llvm_i32_ty], [], [Throws, IntrNoDivergenceSource]>;
def int_d_nounwind : Intrinsic<[llvm_i32_ty]>;
def int_e_nounwind_uniform : Intrinsic<[llvm_i32_ty], [], [IntrNoDivergenceSource]>;

// The property is sufficient on its own and participates in attribute grouping.
// CHECK: static AttributeSet getIntrinsicFnAttributeSet(
// CHECK: case 0: // llvm.b.uniform
// CHECK-NEXT: return AttributeSet::get(C, {
// CHECK-NEXT: Attribute::get(C, Attribute::NoDivergenceSource),
// CHECK-NEXT: });
// CHECK: case 1: // llvm.d.nounwind
// CHECK: case 2: // llvm.e.nounwind.uniform
// CHECK: static constexpr uint8_t IntrinsicsToAttributesMap[] = {
// CHECK-NEXT: 3 << 0 | 0, // llvm.a.none
// CHECK-NEXT: 0 << 0 | 0, // llvm.b.uniform
// CHECK-NEXT: 0 << 0 | 0, // llvm.c.uniform
// CHECK-NEXT: 1 << 0 | 0, // llvm.d.nounwind
// CHECK-NEXT: 2 << 0 | 0, // llvm.e.nounwind.uniform
// CHECK-NEXT: }; // IntrinsicsToAttributesMap
