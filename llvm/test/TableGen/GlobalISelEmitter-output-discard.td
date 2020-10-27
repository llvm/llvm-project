// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test that extra explicit results are treated as dead defs.
def ADD_CO : I<(outs GPR32:$dst, GPR8:$flag),
               (ins GPR32Op:$src0, GPR32Op:$src1), []>;

// GISEL: GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ADD,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/MyTarget::GPR32RegClassID,
// GISEL-NEXT: // (add:{ *:[i32] } i32:{ *:[i32] }:$src0, i32:{ *:[i32] }:$src1)  =>  (ADD_CO:{ *:[i32] }:{ *:[i8] } GPR32:{ *:[i32] }:$src0, GPR32:{ *:[i32] }:$src1)
// GISEL-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s8,
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::ADD_CO,
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// GISEL-NEXT: GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/RegState::Define|RegState::Dead,
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
// GISEL-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// GISEL-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
def : Pat <
  (add i32:$src0, i32:$src1),
  (ADD_CO GPR32:$src0, GPR32:$src1)
>;
