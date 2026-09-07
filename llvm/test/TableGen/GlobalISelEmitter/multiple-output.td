// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../../include -I %p/../Common %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test the generation of patterns with multiple output operands and makes sure that
// we are able to create a new instruction if necessary, or just simply change the
// opcode if the input and output operands of the generic instruction are the same
// as the target-specific instruction

// Verify that patterns with multiple outputs are translated

//-----------------------------------------------------------------------------
// Test where only the opcode is mutated during ISel

let Constraints = "$ptr_out = $addr" in
def LDPost : I<(outs GPR32:$val, GPR32:$ptr_out), (ins GPR32:$addr, GPR32:$off), []>;
def SDTLoadPost : SDTypeProfile<2, 2, [
  SDTCisInt<0>, SDTCisSameAs<1,2>, SDTCisPtrTy<2>, SDTCisInt<3>,
]>;
def loadpost : SDNode<"MyTgt::LOADPOST", SDTLoadPost, [
  SDNPHasChain, SDNPMayLoad, SDNPMemOperand
]>;
def G_POST_LOAD : MyTargetGenericInstruction{
  let OutOperandList = (outs type0:$val, type1:$ptr_out);
  let InOperandList = (ins type1:$ptr, type2:$off);
}
def : GINodeEquiv<G_POST_LOAD, loadpost>;
def : Pat<(loadpost (p0 GPR32:$addr), (i32 GPR32:$off)),
  (LDPost GPR32:$addr, GPR32:$off)
>;

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_NINE_IN),
// CHECK:      GIM_RootCheckType8, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass8, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK:      GIM_RootCheckType, /*Op*/9, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass, /*Op*/9, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK:      GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::NINE_INS),
// CHECK:      GIR_RootToRootCopy, /*OpIdx*/9,
// CHECK-NEXT: GIR_RootToRootCopy8,

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_POST_LOAD),
// CHECK-NEXT: // MIs[0] DstI[val]
// CHECK-NEXT: GIM_RootCheckType0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] DstI[ptr_out]
// CHECK-NEXT: GIM_RootCheckType1, /*Type*/GILLT_p0s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] addr
// CHECK-NEXT: GIM_RootCheckType2, /*Type*/GILLT_p0s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] off
// CHECK-NEXT: GIM_RootCheckType3, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass3, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (loadpost:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$addr, GPR32:{ *:[i32] }:$off)  =>  (LDPost:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$addr, GPR32:{ *:[i32] }:$off)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LDPost),
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,

//-----------------------------------------------------------------------------
// Test where a whole new MIR instruction is created during ISel

def TWO_INS : I<(outs GPR32:$out1, GPR32:$out2), (ins GPR32:$in1, GPR32:$in2), []>;

def SDTTwoIn : SDTypeProfile<2, 2, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>
]>;
def two_in : SDNode<"MyTgt::TWO_IN", SDTTwoIn, []>;
def G_TWO_IN : MyTargetGenericInstruction{
  let OutOperandList = (outs type0:$out1, type0:$out2);
  let InOperandList = (ins type0:$in1, type0:$in2);
}
def : GINodeEquiv<G_TWO_IN, two_in>;

// Swap the input operands for an easy way to force the creation of a new instruction
def : Pat<(two_in GPR32:$i1, GPR32:$i2), (TWO_INS GPR32:$i2, GPR32:$i1)>;

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_TWO_IN),
// CHECK-NEXT: // MIs[0] DstI[out1]
// CHECK-NEXT: GIM_RootCheckType0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] DstI[out2]
// CHECK-NEXT: GIM_RootCheckType1, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] i1
// CHECK-NEXT: GIM_RootCheckType2, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] i2
// CHECK-NEXT: GIM_RootCheckType3, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass3, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (two_in:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$i1, GPR32:{ *:[i32] }:$i2)  =>  (TWO_INS:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$i2, GPR32:{ *:[i32] }:$i1)
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::TWO_INS),
// CHECK-NEXT: GIR_RootToRootCopy0, // DstI[out1]
// CHECK-NEXT: GIR_RootToRootCopy1, // DstI[out2]
// CHECK-NEXT: GIR_RootToRootCopy3, // i2
// CHECK-NEXT: GIR_RootToRootCopy2, // i1
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

//-----------------------------------------------------------------------------
// Test root operand indices 8 and 9, including the ULEB128 fallback for 9.

def NINE_INS : I<(outs GPR32:$out),
                 (ins GPR32:$in0, GPR32:$in1, GPR32:$in2, GPR32:$in3,
                      GPR32:$in4, GPR32:$in5, GPR32:$in6, GPR32:$in7,
                      GPR32:$in8), []>;

def SDTNineIn : SDTypeProfile<1, 9, [
  SDTCisInt<0>, SDTCisInt<1>, SDTCisInt<2>, SDTCisInt<3>, SDTCisInt<4>,
  SDTCisInt<5>, SDTCisInt<6>, SDTCisInt<7>, SDTCisInt<8>, SDTCisInt<9>
]>;
def nine_in : SDNode<"MyTgt::NINE_IN", SDTNineIn, []>;
def G_NINE_IN : MyTargetGenericInstruction {
  let OutOperandList = (outs type0:$out);
  let InOperandList = (ins type0:$in0, type0:$in1, type0:$in2, type0:$in3,
                           type0:$in4, type0:$in5, type0:$in6, type0:$in7,
                           type0:$in8);
}
def : GINodeEquiv<G_NINE_IN, nine_in>;
def : Pat<(nine_in GPR32:$in0, GPR32:$in1, GPR32:$in2, GPR32:$in3,
                   GPR32:$in4, GPR32:$in5, GPR32:$in6, GPR32:$in7,
                   GPR32:$in8),
          (NINE_INS GPR32:$in8, GPR32:$in7, GPR32:$in6, GPR32:$in5,
                    GPR32:$in4, GPR32:$in3, GPR32:$in2, GPR32:$in1,
                    GPR32:$in0)>;

//-----------------------------------------------------------------------------
// Test where implicit defs are added using Defs.

let Defs = [R0] in
def ImplicitDefInstr : I<(outs GPR32:$dst), (ins GPR32:$src), []>;
def OtherInstr : I<(outs GPR32:$dst), (ins GPR32:$src), []>;

def : Pat<(i32 (add i32:$src, i32:$src)),
  (OtherInstr (ImplicitDefInstr GPR32:$src))>;

// CHECK: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_RootCheckType0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_RootCheckType1, /*Type*/GILLT_s32,
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
// CHECK-NEXT: // (add:{ *:[i32] } i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src)  =>  (OtherInstr:{ *:[i32] } (ImplicitDefInstr:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$src))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::ImplicitDefInstr),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_SetImplicitDefDead, /*InsnID*/1, /*OpIdx for MyTarget::R0*/0,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::OtherInstr),
// CHECK-NEXT: GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

//-----------------------------------------------------------------------------
// Test when the inner instruction in the output pattern has two outs

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_RootCheckType0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_RootCheckRegBankForClass0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_RootCheckType1, /*Type*/GILLT_s32,
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
// CHECK-NEXT: // (add:{ *:[i32] } i32:{ *:[i32] }:$src, i32:{ *:[i32] }:$src)  =>  (OtherInstr:{ *:[i32] } (TwoOutsInstr:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$src))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::TwoOutsInstr),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define|RegState::Dead)),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::OtherInstr),
// CHECK-NEXT: GIR_RootToRootCopy0, // DstI[dst]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

def TwoOutsInstr : I<(outs GPR32:$out1, GPR32:$out2), (ins GPR32:$src), []>;

def : Pat<(i32 (add i32:$src, i32:$src)),
  (OtherInstr (TwoOutsInstr GPR32:$src))>;
