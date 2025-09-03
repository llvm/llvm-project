// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_aarch64_set_fpmr_2   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleWriteMemOnly<AArch64_FPMR>]>;

def int_aarch64_get_za_2   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadMemOnly<AArch64_ZA>]>;

def int_aarch64_get_fpmr_set_za   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadMemOnly<AArch64_FPMR>, IntrInaccessibleWriteMemOnly<AArch64_ZA>]>;

def int_aarch64_get_set_fpmr  : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadWriteMem<AArch64_FPMR>]>;

def int_aarch64_get_set_za  : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadWriteMem<AArch64_ZA>]>;

def int_aarch64_get_set_fpmr_get_za  : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrInaccessibleReadWriteMem<AArch64_FPMR>, IntrInaccessibleReadMemOnly<AArch64_ZA>]>;

def int_aarch64_get_fpmr_get_set_za  : DefaultAttrsIntrinsic<[], [llvm_i64_ty],  [IntrInaccessibleReadMemOnly<AArch64_FPMR>, IntrInaccessibleReadWriteMem<AArch64_ZA>]>;

// CHECK:    static constexpr unsigned IntrinsicNameOffsetTable[] = {
// CHECK-NEXT:  1, // not_intrinsic
// CHECK-NEXT:  15, // llvm.aarch64.get.fpmr.get.set.za
// CHECK-NEXT:  48, // llvm.aarch64.get.fpmr.set.za
// CHECK-NEXT:  77, // llvm.aarch64.get.set.fpmr
// CHECK-NEXT:  103, // llvm.aarch64.get.set.fpmr.get.za
// CHECK-NEXT:  136, // llvm.aarch64.get.set.za
// CHECK-NEXT:  160, // llvm.aarch64.get.za.2
// CHECK-NEXT:  182, // llvm.aarch64.set.fpmr.2

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
// CHECK-NEXT:     // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: Ref, AARCH64_ZA: ModRef, Other: NoModRef
// CHECK-NEXT:     Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(832)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 1:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: Ref, AARCH64_ZA: Mod, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(576)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 2:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: ModRef, AARCH64_ZA: NoModRef, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(192)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 3:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: ModRef, AARCH64_ZA: Ref, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(448)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 4:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: NoModRef, AARCH64_ZA: ModRef, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(768)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 5:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: NoModRef, AARCH64_ZA: Ref, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(256)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 6:
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, AARCH64_FPMR: Mod, AARCH64_ZA: NoModRef, Other: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(128)),

// CHECK: static constexpr uint16_t IntrinsicsToAttributesMap[] = {
// CHECK-NEXT:    0 << 8 | 0, // llvm.aarch64.get.fpmr.get.set.za
// CHECK-NEXT:    1 << 8 | 0, // llvm.aarch64.get.fpmr.set.za
// CHECK-NEXT:    2 << 8 | 0, // llvm.aarch64.get.set.fpmr
// CHECK-NEXT:    3 << 8 | 0, // llvm.aarch64.get.set.fpmr.get.za
// CHECK-NEXT:    4 << 8 | 0, // llvm.aarch64.get.set.za
// CHECK-NEXT:    5 << 8 | 0, // llvm.aarch64.get.za.2
// CHECK-NEXT:    6 << 8 | 0, // llvm.aarch64.set.fpmr.2
// CHECK-NEXT:};
