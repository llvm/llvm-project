// RUN: llvm-tblgen -gen-global-isel -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck -check-prefix=GISEL %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

def int_tgt_foo : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

// Test that using srcvalue works for discarded pattern inputs.
def FOO : I<(outs GPR32:$dst), (ins GPR32Op:$src0, GPR32Op:$src1), []>;

// GISEL: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS),
// GISEL-NEXT: GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// GISEL-NEXT: GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::tgt_foo),
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// GISEL-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// GISEL-NEXT: // (intrinsic_w_chain:{ *:[i32] } {{[0-9]+}}:{ *:[iPTR] }, srcvalue:{ *:[i32] }, i32:{ *:[i32] }:$src1)  =>  (FOO:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), GPR32:{ *:[i32] }:$src1)
// GISEL-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::IMPLICIT_DEF),
// GISEL-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// GISEL-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// GISEL-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::FOO),
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// GISEL-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// GISEL-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src1
// GISEL-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// GISEL-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
def : Pat <
  (int_tgt_foo (i32 srcvalue), i32:$src1),
  (FOO (IMPLICIT_DEF), GPR32:$src1)
>;
