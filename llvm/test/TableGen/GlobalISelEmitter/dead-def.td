// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false \
// RUN:   -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Check that $same_name from I2 isn't copied to the root instruction.

def I1 : I<(outs GPR32:$same_name), (ins GPR32:$rs), []>;
def I2 : I<(outs GPR32:$other_name, GPR32:$same_name), (ins GPR32:$rs), []>;

def : Pat<(abs i32:$x), (I1 (I2 $x))>;

// CHECK-LABEL: // (abs:{ *:[i32] } i32:{ *:[i32] }:$x)  =>  (I1:{ *:[i32] } (I2:{ *:[i32] }:{ *:[i32] } ?:{ *:[i32] }:$x))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::I2),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define|RegState::Dead),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // x
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::I1),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[same_name]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 0,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
