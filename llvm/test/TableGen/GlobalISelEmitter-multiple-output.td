// RUN: llvm-tblgen -gen-global-isel -optimize-match-table=false -warn-on-skipped-patterns -I %p/../../include -I %p/Common %s -o - < %s | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Test the generation of patterns with multiple output operands and makes sure that
// we are able to create a new instruction if necessary, or just simply change the
// opcode if the input and output operands of the generic instruction are the same
// as the target-specific instruction

// Verify that patterns with multiple outputs are translated

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

// CHECK:      GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(MyTarget::G_POST_LOAD),
// CHECK-NEXT: // MIs[0] DstI[val]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] DstI[ptr_out]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_p0s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] addr
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_p0s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] off
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (loadpost:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$addr, GPR32:{ *:[i32] }:$off)  =>  (LDPost:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$addr, GPR32:{ *:[i32] }:$off)
// CHECK-NEXT: GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::LDPost),
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,

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
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] DstI[out2]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] i1
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // MIs[0] i2
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/GIMT_Encode2(MyTarget::GPR32RegClassID),
// CHECK-NEXT: // (two_in:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$i1, GPR32:{ *:[i32] }:$i2)  =>  (TWO_INS:{ *:[i32] }:{ *:[i32] } GPR32:{ *:[i32] }:$i2, GPR32:{ *:[i32] }:$i1)
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::TWO_INS),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[out1]
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // DstI[out2]
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // i2
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // i1
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
