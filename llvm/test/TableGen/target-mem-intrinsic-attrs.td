// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_aarch64_get_target_mem0_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrReadMem, IntrRead<[TargetMem0, TargetMem1]>]>;

def int_aarch64_get_target_mem0_set_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem0]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_get_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrReadMem,  IntrRead<[TargetMem1]>]>;

def int_aarch64_get_target_mem1_set_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem1]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_set_inaccessible_mem   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrWriteMem, IntrWrite<[InaccessibleMem]>]>;

def int_aarch64_set_target_mem0   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrWriteMem, IntrWrite<[TargetMem0]>]>;

// CHECK:   static AttributeSet getIntrinsicFnAttributeSet(LLVMContext &C, unsigned ID) {
// CHECK-NEXT:  switch (ID) {
// CHECK-NEXT:    default: llvm_unreachable("Invalid attribute set number");
// CHECK-NEXT:  case 0: // llvm.aarch64.get.target.mem0.mem1
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Ref, TargetMem1: Ref
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(1280)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 1: // llvm.aarch64.get.target.mem0.set.target.mem1
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:     Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Ref, TargetMem1: Mod
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(2304)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 2: // llvm.aarch64.get.target.mem1
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: Ref
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(1024)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 3: // llvm.aarch64.get.target.mem1.set.target.mem1
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: ModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(3072)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 4: // llvm.aarch64.set.inaccessible.mem
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: Mod, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(8)),
// CHECK-NEXT:    });
// CHECK-NEXT:  case 5: // llvm.aarch64.set.target.mem0
// CHECK-NEXT:    return AttributeSet::get(C, {
// CHECK-NEXT:      Attribute::get(C, Attribute::NoUnwind),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoCallback),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoSync),
// CHECK-NEXT:      Attribute::get(C, Attribute::NoFree),
// CHECK-NEXT:      Attribute::get(C, Attribute::WillReturn),
// CHECK-NEXT:      // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Mod, TargetMem1: NoModRef
// CHECK-NEXT:      Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(512)),
