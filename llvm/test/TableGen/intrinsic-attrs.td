// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

// ... this intrinsic.
def int_random_gen   : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects]>;

def int_deref_ptr_ret : Intrinsic<[llvm_ptr_ty], [], [Dereferenceable<RetIndex, 16>]>;

// CHECK: static AttributeSet getIntrinsicArgAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:   switch (ID) {
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


// CHECK: 1, // llvm.deref.ptr.ret
// CHECK: 2, // llvm.random.gen

// CHECK: case 1:
// CHECK-NEXT: AS[0] = {0, getIntrinsicArgAttributeSet(C, 0)};
// CHECK-NEXT: AS[1] = {AttributeList::FunctionIndex, getIntrinsicFnAttributeSet(C, 0)};
// CHECK-NEXT: NumAttrs = 2;

// CHECK: case 2:
// CHECK-NEXT: AS[0] = {AttributeList::FunctionIndex, getIntrinsicFnAttributeSet(C, 1)};
// CHECK-NEXT: NumAttrs = 1;
