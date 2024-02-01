// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Verify that patterns will add temp registers if NumDstDef > NumSrcDef when NumSrcDef >= 1
// and that these temp registers are marked as dead
// Note: This is an extension of the test GlobalISelEmitter-output-discard.td

def THREE_OUTS : I<(outs GPR32:$out1, GPR32:$out2, GPR32:$out3), (ins GPR32:$in1), []>;

def SDTTwoOut : SDTypeProfile<2, 1, [
  SDTCisInt<0>, SDTCisInt<1>
]>;
def two_out : SDNode<"MyTgt::ONE_OUT", SDTTwoOut, []>;
def G_TWO_OUT : MyTargetGenericInstruction{
  let OutOperandList = (outs type0:$out1, type0:$out2);
  let InOperandList = (ins type0:$in);
}
def : GINodeEquiv<G_TWO_OUT, two_out>;

def : Pat<(two_out GPR32:$val), (THREE_OUTS GPR32:$val)>;

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_TWO_OUT),
// CHECK-NEXT: // MIs[0] DstI[out1]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] DstI[out2]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] val
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (two_out:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$val)  =>  (THREE_OUTS:{ *:[i32] }:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$val)
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::THREE_OUTS),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[out1]
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // DstI[out2]
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define|RegState::Dead),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // val
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,
