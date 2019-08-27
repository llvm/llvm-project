// RUN: llvm-tblgen %s -gen-global-isel -warn-on-skipped-patterns -I %p/../../include -I %p/Common -o - 2> %t.skipped | FileCheck %s
// RUN: cat %t.skipped | FileCheck %s --check-prefix=SKIPPED

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
def S2 : MyReg<"s14">;
def S3 : MyReg<"s15">;
def SRegs : MyClass<16, [i16], (sequence "S%u", 0, 3)>;

let SubRegIndices = [sub0, sub1] in {
def D0 : MyReg<"d0", [S0, S1]>;
def E0 : MyReg<"e0", [S2, S3]>;
}

def DRegs : MyClass<32, [i32], (sequence "D%u", 0, 0)>;
def ERegs : MyClass<32, [i32], (sequence "E%u", 0, 0)>;
def SOP : RegisterOperand<SRegs>;
def DOP : RegisterOperand<DRegs>;
def SOME_INSN : I<(outs DRegs:$dst), (ins DOP:$src), []>;

// We should skip cases where we don't have a given register class for the
// subregister source.
// SKIPPED: def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0)>;
// SKIPPED: def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0))>;
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0)>;
def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0))>;

// Test that we import INSERT_SUBREG when its subregister source has a given
// class.
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), SOP:$src, sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] })
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT:            GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::INSERT_SUBREG,
// CHECK-NEXT:            GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:            GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:            GIR_AddImm, /*InsnID*/0, /*Imm*/1,
// CHECK-NEXT:            GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC DRegs*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC DRegs*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, /*RC SRegs*/0,


// Test that we can import INSERT_SUBREG when it is a subinstruction of another
// instruction.
def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), SOP:$src, sub0))>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (SOME_INSN:{ *:[i32] } (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] }))
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT:            GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::INSERT_SUBREG,
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
// CHECK-NEXT:            GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:            GIR_AddImm, /*InsnID*/1, /*Imm*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, /*RC DRegs*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, /*RC DRegs*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/2, /*RC SRegs*/0,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/0, /*Opcode*/MyTarget::SOME_INSN,
// CHECK-NEXT:            GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:            GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:            GIR_ConstrainSelectedInstOperands, /*InsnID*/0,


// Test that we correctly infer the super register class for INSERT_SUBREG when
// we have COPY_TO_REGCLASS. We want to make sure we get an E register here,
// not a D register.
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (COPY_TO_REGCLASS SOP:$src, ERegs)), SOP:$src, sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (COPY_TO_REGCLASS:{ *:[i32] } SOP:{ *:[i16] }:$src, ERegs:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] })
// CHECK:                GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::INSERT_SUBREG,
// CHECK-DAG:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC ERegs*/2,
// CHECK-NEXT:           GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC ERegs*/2,
// CHECK-NEXT:           GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, /*RC SRegs*/0,

// Test that we can import INSERT_SUBREG when its subregister source is defined
// by a subinstruction.
def SUBSOME_INSN : I<(outs SRegs:$dst), (ins SOP:$src), []>;
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), (SUBSOME_INSN SOP:$src), sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), (SUBSOME_INSN:{ *:[i16] } SOP:{ *:[i16] }:$src), sub0:{ *:[i32] })
// CHECK-NEXT:          GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:          GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s16,
// CHECK-NEXT:          GIR_BuildMI, /*InsnID*/2, /*Opcode*/MyTarget::SUBSOME_INSN,
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT:          GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:          GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT:          GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
// CHECK-NEXT:          GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:          GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::INSERT_SUBREG,
// CHECK-NEXT:          GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/1, /*TempRegFlags*/0,
// CHECK-NEXT:          GIR_AddImm, /*InsnID*/0, /*Imm*/1,
// CHECK-NEXT:          GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC DRegs*/1,
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC DRegs*/1,
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, /*RC SRegs*/0,
