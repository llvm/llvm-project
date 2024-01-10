// RUN: llvm-tblgen %s -gen-global-isel -optimize-match-table=false -I %p/../../include -I %p/Common -o - | FileCheck %s

include "llvm/Target/Target.td"
include "GlobalISelEmitterCommon.td"

// Boilerplate code for setting up some registers with subregs.
class MyReg<string n, list<Register> subregs = []>
  : Register<n> {
  let SubRegs = subregs;
}

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, size, registers> {
  let Size = size;
}

def sub0 : SubRegIndex<16>;
def sub1 : SubRegIndex<16, 16>;
def S0 : MyReg<"s0">;
def S1 : MyReg<"s1">;
def SRegs : MyClass<16, [i16], (sequence "S%u", 0, 1)>;

let SubRegIndices = [sub0, sub1] in {
def D0 : MyReg<"d0", [S0, S1]>;
}

def DRegs : MyClass<32, [i32], (sequence "D%u", 0, 0)>;
def SOP : RegisterOperand<SRegs>;
def DOP : RegisterOperand<DRegs>;
def SOME_INSN : I<(outs DRegs:$dst), (ins DOP:$src), []>;
def SUBSOME_INSN : I<(outs SRegs:$dst), (ins SOP:$src), []>;

// CHECK: GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SEXT),
// CHECK-NEXT: // MIs[0] DstI[dst]
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: // MIs[0] src
// CHECK-NEXT: GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: // (sext:{ *:[i32] } SOP:{ *:[i16] }:$src)  =>  (REG_SEQUENCE:{ *:[i32] } DRegs:{ *:[i32] }, (SUBSOME_INSN:{ *:[i16] } SOP:{ *:[i16] }:$src), sub0:{ *:[i32] }, (SUBSOME_INSN:{ *:[i16] } SOP:{ *:[i16] }:$src), sub1:{ *:[i32] })
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s16,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::REG_SEQUENCE),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_AddImm8, /*InsnID*/0, /*SubRegIndex*/1,
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT: GIR_AddImm8, /*InsnID*/0, /*SubRegIndex*/2,
// CHECK-NEXT: GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/3, GIMT_Encode2(Test::SRegsRegClassID),
def : Pat<(i32 (sext SOP:$src)),
          (REG_SEQUENCE DRegs, (SUBSOME_INSN SOP:$src), sub0,
                               (SUBSOME_INSN SOP:$src), sub1)>;


// CHECK: GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::REG_SEQUENCE),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/1,
// CHECK-NEXT: GIR_AddImm8, /*InsnID*/1, /*SubRegIndex*/1,
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/2,
// CHECK-NEXT: GIR_AddImm8, /*InsnID*/1, /*SubRegIndex*/2,
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/3, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(MyTarget::SOME_INSN),
// Make sure operands are constrained when REG_SEQUENCE isn't the root instruction.
def : Pat<(i32 (zext SOP:$src)),
          (SOME_INSN (REG_SEQUENCE DRegs, (SUBSOME_INSN SOP:$src), sub0,
                                          (SUBSOME_INSN SOP:$src), sub1))>;
