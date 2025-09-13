// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_aarch64_set_fpmr_2   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleWriteMemOnly<AArch64_FPMR>]>;

def int_aarch64_get_za_2   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadMemOnly<AArch64_ZA>]>;

def int_aarch64_get_fpmr_set_za   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadMemOnly<AArch64_FPMR>, IntrInaccessibleWriteMemOnly<AArch64_ZA>]>;

// CHECK:    static constexpr unsigned IntrinsicNameOffsetTable[] = {
// CHECK-NEXT:  1, // not_intrinsic
// CHECK-NEXT:  15, // llvm.aarch64.get.fpmr.set.za
// CHECK-NEXT:  44, // llvm.aarch64.get.za.2
// CHECK-NEXT:  66, // llvm.aarch64.set.fpmr.2

// CHECK:   static AttributeSet getIntrinsicFnAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:  switch (ID) {
// CHECK-NEXT:    default: llvm_unreachable("Invalid attribute set number");
// CHECK-NEXT:  case 0:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, MEM_TARGET_0: Ref, MEM_TARGET_1: Mod, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(576)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 1:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, MEM_TARGET_0: NoModRef, MEM_TARGET_1: Ref, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(256)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 2:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, MEM_TARGET_0: Mod, MEM_TARGET_1: NoModRef, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(128)),

// CHECK: static constexpr uint16_t IntrinsicsToAttributesMap[] = {
// CHECK-NEXT:    0 << 8 | 0, // llvm.aarch64.get.fpmr.set.za
// CHECK-NEXT:    1 << 8 | 0, // llvm.aarch64.get.za.2
// CHECK-NEXT:    2 << 8 | 0, // llvm.aarch64.set.fpmr.2
// CHECK-NEXT:};
