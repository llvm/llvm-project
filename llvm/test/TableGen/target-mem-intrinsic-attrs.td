// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

include "llvm/IR/Intrinsics.td"

def int_aarch64_get_target_mem0_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrReadMem, IntrRead<[TargetMem0, TargetMem1]>]>;

def int_aarch64_get_target_mem0_set_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem0]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_get_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrReadMem,  IntrRead<[TargetMem1]>]>;

def int_aarch64_get_target_mem1_set_target_mem1   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem1]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_set_inaccessible_mem   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrWriteMem, IntrWrite<[InaccessibleMem]>]>;

def int_aarch64_set_target_mem0   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrWriteMem, IntrWrite<[TargetMem0]>]>;

def int_aarch64_argmem_read_target_mem1_write   :  DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[ArgMem]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_argmem_target_mem1_read_target_mem1_write  : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[ArgMem, TargetMem1]>, IntrWrite<[TargetMem1]>]>;

def int_aarch64_target_mem1_read_argmem_write   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem1]>, IntrWrite<[ArgMem]>]>;

def int_aarch64_target_mem0_read_argmem_write   : DefaultAttrsIntrinsic<[], [llvm_i64_ty], [IntrRead<[TargetMem0]>, IntrWrite<[ArgMem]>]>;

// CHECK-LABEL: case 0: // llvm.aarch64.argmem.read.target.mem1.write
// CHECK: // ArgMem: Ref, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: Mod
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(2049)),

// CHECK-LABEL: case 1: // llvm.aarch64.argmem.target.mem1.read.target.mem1.write
// CHECK: // ArgMem: Ref, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: ModRef
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(3073)),

// CHECK-LABEL: case 2: // llvm.aarch64.get.target.mem0.mem1
// CHECK: // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Ref, TargetMem1: Ref
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(1280)),

// CHECK-LABEL: case 3: // llvm.aarch64.get.target.mem0.set.target.mem1
// CHECK: // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Ref, TargetMem1: Mod
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(2304)),

// CHECK-LABEL: case 4: // llvm.aarch64.get.target.mem1
// CHECK: // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: Ref
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(1024)),

// CHECK-LABEL: case 5: // llvm.aarch64.get.target.mem1.set.target.mem1
// CHECK: // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: ModRef
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(3072)),

// CHECK-LABEL: case 6: // llvm.aarch64.set.inaccessible.mem
// CHECK: // ArgMem: NoModRef, InaccessibleMem: Mod, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: NoModRef
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(8)),

// CHECK-LABEL: case 7: // llvm.aarch64.set.target.mem0
// CHECK: // ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Mod, TargetMem1: NoModRef
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(512)),

// CHECK-LABEL: case 8: // llvm.aarch64.target.mem0.read.argmem.write
// CHECK: // ArgMem: Mod, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: Ref, TargetMem1: NoModRef
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(258)),

// CHECK-LABEL: case 9: // llvm.aarch64.target.mem1.read.argmem.write
// CHECK: // ArgMem: Mod, InaccessibleMem: NoModRef, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: Ref
// CHECK-NEXT: Attribute::getWithMemoryEffects(C, MemoryEffects::createFromIntValue(1026)),
