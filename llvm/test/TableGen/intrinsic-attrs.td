// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_random_gen   : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects]>;

def int_deref_ptr_ret : Intrinsic<[llvm_ptr_ty], [], [Dereferenceable<RetIndex, 16>]>;

// CHECK: static AttributeSet getIntrinsicArgAttributeSet(LLVMContext &C, unsigned ID,
// CHECK-NEXT: Type *ArgType) {
// CHECK-NEXT: unsigned BitWidth = ArgType->getScalarSizeInBits();
// CHECK-NEXT: switch (ID) {
// CHECK-NEXT: default: llvm_unreachable("Invalid attribute set number");
// CHECK-NEXT: case 0:
// CHECK-NEXT:     return AttributeSet::get(C, {
// CHECK-NEXT:       Attribute::get(C, Attribute::Dereferenceable, 16),
// CHECK-NEXT: });
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK: static AttributeSet getIntrinsicFnAttributeSet(
// CHECK: case 0:
// CHECK-NEXT: return AttributeSet::get(C, {
// CHECK-NEXT: Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT: });

// CHECK: static constexpr uint16_t IntrinsicsToAttributesMap[] = {
// CHECK: 0 << 2 | 0, // llvm.deref.ptr.ret
// CHECK: 1 << 2 | 1, // llvm.random.gen
// CHECK: }; // IntrinsicsToAttributesMap

// CHECK: static constexpr ArgNoAttrIDPair ArgAttrIdTable[] = {
// CHECK-NEXT:   {0, 0},
// CHECK: }; // ArgAttrIdTable

// CHECK: static constexpr ArgAttributesInfo ArgAttributesInfoTable[] = {
// CHECK-NEXT:  {0, 1},
// CHECK-NEXT:  {0, 0},
// CHECK-NEXT: }; // ArgAttributesInfoTable

// CHECK: getAttributes(LLVMContext &C, ID id,
// CHECK-NEXT: FunctionType *FT) {
