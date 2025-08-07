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
// CHECK: 0 << 8 | 0, // llvm.deref.ptr.ret
// CHECK: 1 << 8 | 1, // llvm.random.gen

// CHECK: getAttributes(LLVMContext &C, ID id,
// CHECK-NEXT: FunctionType *FT) {
// CHECK:      case 1:
// CHECK-NEXT:   HasFnAttr = true;
// CHECK-NEXT:   break;
// CHECK-NEXT: case 0:
// CHECK-NEXT:   AS[0] = {0, getIntrinsicArgAttributeSet(C, 0, FT->getContainedType(0))};
// CHECK-NEXT:   HasFnAttr = true;
// CHECK-NEXT:   NumAttrs = 1
// CHECK-NEXT:   break;
