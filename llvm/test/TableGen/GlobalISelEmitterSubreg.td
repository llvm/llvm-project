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
def SOME_INSN2 : I<(outs DRegs:$dst), (ins DOP:$src1, DOP:$src2), []>;
def SUBSOME_INSN : I<(outs SRegs:$dst), (ins SOP:$src), []>;
def SUBSOME_INSN2 : I<(outs SRegs:$dst), (ins SOP:$src), []>;

// Adding this enables the tests below to check that we are not using this class
// for constraining the operand register classes, since it is unallocatable.
let isAllocatable = 0 in
def SuperDRegs : MyClass<32, [i32], (add DRegs, ERegs)>;

def complex : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexOperand", []>;
def gi_complex : GIComplexOperandMatcher<s32, "selectComplexOperand">,
                 GIComplexPatternEquiv<complex>;

// We should skip cases where we don't have a given register class for the
// subregister source.
// SKIPPED: def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0)>;
// SKIPPED: def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0))>;
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0)>;
def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), i16:$src, sub0))>;

// Test EXTRACT_SUBREG applied to suboperands of a ComplexPattern
// operand.
def : Pat<(sub (complex DOP:$src1, DOP:$src2), 77),
          (SOME_INSN2 (EXTRACT_SUBREG DOP:$src1, sub0),
                      (EXTRACT_SUBREG DOP:$src2, sub1))>;
// CHECK-LABEL: // (sub:{ *:[i32] } (complex:{ *:[i32] } DOP:{ *:[i32] }:$src1, DOP:{ *:[i32] }:$src2), 77:{ *:[i32] })  =>  (SOME_INSN2:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i32] } DOP:{ *:[i32] }:$src1, sub0:{ *:[i32] }), (EXTRACT_SUBREG:{ *:[i32] } DOP:{ *:[i32] }:$src2, sub1:{ *:[i32] }))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_ComplexSubOperandSubRegRenderer, /*InsnID*/2, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/1, /*SubRegIdx*/GIMT_Encode2(2), // src2
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/2, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/2, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_ComplexSubOperandSubRegRenderer, /*InsnID*/1, /*RendererID*/GIMT_Encode2(0), /*SubOperand*/0, /*SubRegIdx*/GIMT_Encode2(1), // src1
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::SOME_INSN2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 2,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// Test that we import INSERT_SUBREG when its subregister source has a given
// class.
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), SOP:$src, sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] })
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::IMPLICIT_DEF),
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:            GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:            GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::INSERT_SUBREG),
// CHECK-NEXT:            GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:            GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:            GIR_RootToRootCopy, /*OpIdx*/1, // src
// CHECK-NEXT:            GIR_AddImm8, /*InsnID*/0, /*Imm*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT:            // GIR_Coverage, 3,
// CHECK-NEXT:            GIR_EraseRootFromParent_Done,


// Test that we can import INSERT_SUBREG when it is a subinstruction of another
// instruction.
def : Pat<(i32 (anyext i16:$src)), (SOME_INSN (INSERT_SUBREG (i32 (IMPLICIT_DEF)), SOP:$src, sub0))>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (SOME_INSN:{ *:[i32] } (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] }))
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(TargetOpcode::IMPLICIT_DEF),
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:            GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT:            GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::INSERT_SUBREG),
// CHECK-NEXT:            GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:            GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/1,
// CHECK-NEXT:            GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:            GIR_AddImm8, /*InsnID*/1, /*Imm*/1,
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:            GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/2, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT:            GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::SOME_INSN),
// CHECK-NEXT:            GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:            GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:            GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT:            // GIR_Coverage, 4,
// CHECK-NEXT:            GIR_EraseRootFromParent_Done,


// Test that we correctly infer the super register class for INSERT_SUBREG when
// we have COPY_TO_REGCLASS. We want to make sure we get an E register here,
// not a D register.
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (COPY_TO_REGCLASS SOP:$src, ERegs)), SOP:$src, sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (COPY_TO_REGCLASS:{ *:[i32] } SOP:{ *:[i16] }:$src, ERegs:{ *:[i32] }), SOP:{ *:[i16] }:$src, sub0:{ *:[i32] })
// CHECK:                GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::INSERT_SUBREG),
// CHECK-DAG:            GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::ERegsRegClassID),
// CHECK-NEXT:           GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::ERegsRegClassID),
// CHECK-NEXT:           GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, GIMT_Encode2(Test::SRegsRegClassID),

// Test that we can import INSERT_SUBREG when its subregister source is defined
// by a subinstruction.
def : Pat<(i32 (anyext i16:$src)), (INSERT_SUBREG (i32 (IMPLICIT_DEF)), (SUBSOME_INSN SOP:$src), sub0)>;
// CHECK-LABEL:  (anyext:{ *:[i32] } i16:{ *:[i16] }:$src)  =>  (INSERT_SUBREG:{ *:[i32] } (IMPLICIT_DEF:{ *:[i32] }), (SUBSOME_INSN:{ *:[i16] } SOP:{ *:[i16] }:$src), sub0:{ *:[i32] })
// CHECK-NEXT:          GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:          GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s16,
// CHECK-NEXT:          GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN),
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:          GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:          GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT:          GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::IMPLICIT_DEF),
// CHECK-NEXT:          GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:          GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:          GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::INSERT_SUBREG),
// CHECK-NEXT:          GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:          GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:          GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:          GIR_AddImm8, /*InsnID*/0, /*Imm*/1,
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:          GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT:          // GIR_Coverage, 6,
// CHECK-NEXT:          GIR_EraseRootFromParent_Done,

// Test an EXTRACT_SUBREG that is a sub instruction. The individual
// operands should be constrained to specific register classes, and
// not use GIR_ConstrainSelectedInstOperands.
def : Pat<(i16 (trunc (not DOP:$src))),
          (SUBSOME_INSN (EXTRACT_SUBREG DOP:$src, sub0))>;
// CHECK-LABEL: // (trunc:{ *:[i16] } (xor:{ *:[i32] } DOP:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (SUBSOME_INSN:{ *:[i16] } (EXTRACT_SUBREG:{ *:[i16] } DOP:{ *:[i32] }:$src, sub0:{ *:[i32] }))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, /*SubRegIdx*/GIMT_Encode2(1), // src
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN),

// Test an extract from an output instruction result (nonleaf)
def : Pat<(i16 (trunc (bitreverse DOP:$src))),
           (EXTRACT_SUBREG (SOME_INSN DOP:$src), sub0)>;
// CHECK-LABEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BITREVERSE),
// CHECK-NEXT:  GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:  GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:  GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:  // (trunc:{ *:[i16] } (bitreverse:{ *:[i32] } DOP:{ *:[i32] }:$src))  =>  (EXTRACT_SUBREG:{ *:[i16] } (SOME_INSN:{ *:[i32] } DOP:{ *:[i32] }:$src), sub0:{ *:[i32] })
// CHECK-NEXT:  GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:  GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::SOME_INSN),
// CHECK-NEXT:  GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:  GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:  GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:  GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:  GIR_AddTempSubRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(0), GIMT_Encode2(sub0),
// CHECK-NEXT:  GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT:  GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:  // GIR_Coverage, 8,
// CHECK-NEXT:  GIR_EraseRootFromParent_Done,

// EXTRACT_SUBREG is subinstruction, but also doesn't have a leaf input

// CHECK-LABEL: GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_CTPOP),
// CHECK-NEXT: GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT: GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT: // (trunc:{ *:[i16] } (ctpop:{ *:[i32] } DOP:{ *:[i32] }:$src))  =>  (SUBSOME_INSN2:{ *:[i16] } (EXTRACT_SUBREG:{ *:[i16] } (SOME_INSN:{ *:[i32] } DOP:{ *:[i32] }:$src), sub0:{ *:[i32] }))
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
// CHECK-NEXT: GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_s32,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/2, /*Opcode*/GIMT_Encode2(MyTarget::SOME_INSN),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT: GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
// CHECK-NEXT: GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT: GIR_AddTempSubRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/GIMT_Encode2(0), GIMT_Encode2(sub0),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/1, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN2),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT: GIR_RootConstrainSelectedInstOperands,
// CHECK-NEXT: // GIR_Coverage, 9,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,
def : Pat<(i16 (trunc (ctpop DOP:$src))),
           (SUBSOME_INSN2 (EXTRACT_SUBREG (SOME_INSN DOP:$src), sub0))>;

// Test an EXTRACT_SUBREG that is the final instruction.
def : Pat<(i16 (trunc DOP:$src)),
           (EXTRACT_SUBREG DOP:$src, sub0)>;
// CHECK-LABEL: // (trunc:{ *:[i16] } DOP:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i16] } DOP:{ *:[i32] }:$src, sub0:{ *:[i32] })
// CHECK-NEXT: GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT: GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT: GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/GIMT_Encode2(1), // src
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT: GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT: // GIR_Coverage, 10,
// CHECK-NEXT: GIR_EraseRootFromParent_Done,

// Test that we can import SUBREG_TO_REG
def : Pat<(i32 (zext SOP:$src)),
          (SUBREG_TO_REG (i64 0), (SUBSOME_INSN SOP:$src), sub0)>;
// CHECK-LABEL:  (zext:{ *:[i32] } SOP:{ *:[i16] }:$src)  =>  (SUBREG_TO_REG:{ *:[i32] } 0:{ *:[i64] }, (SUBSOME_INSN:{ *:[i16] } SOP:{ *:[i16] }:$src), sub0:{ *:[i32] })
// CHECK-NEXT:        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
// CHECK-NEXT:        GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(MyTarget::SUBSOME_INSN),
// CHECK-NEXT:        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
// CHECK-NEXT:        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::SUBREG_TO_REG),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // DstI[dst]
// CHECK-NEXT:        GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:        GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:        GIR_AddImm8, /*InsnID*/0, /*Imm*/1,
// CHECK-NEXT:        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, GIMT_Encode2(Test::DRegsRegClassID),
// CHECK-NEXT:        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/2, GIMT_Encode2(Test::SRegsRegClassID),
// CHECK-NEXT:        // GIR_Coverage, 11,
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
