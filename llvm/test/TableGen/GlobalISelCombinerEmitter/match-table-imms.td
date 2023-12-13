// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// emits a imm
def InstTest0 : GICombineRule<
  (defs root:$a),
  (match (COPY $a, (i32 0))),
  (apply (COPY $a, 0))>;

// implicitly emits a G_CONSTANT
def InstTest1 : GICombineRule<
  (defs root:$a),
  (match (G_ZEXT $a, 0)),
  (apply (COPY $a, (i32 0)))>;

// emits a CImm because G_CONSTANT is a special case.
def CImmInstTest1 : GICombineRule<
  (defs root:$a),
  (match (G_CONSTANT $a, (i32 0))),
  (apply (G_CONSTANT $a, (i32 42)))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  InstTest1,
  CImmInstTest1
]>;

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(21), GIMT_Encode2(129), /*)*//*default:*//*Label 3*/ GIMT_Encode4(559),
// CHECK-NEXT:     /*TargetOpcode::COPY*//*Label 0*/ GIMT_Encode4(442), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_CONSTANT*//*Label 1*/ GIMT_Encode4(473), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_ZEXT*//*Label 2*/ GIMT_Encode4(519),
// CHECK-NEXT:     // Label 0: @442
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(472), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:       // Combiner Rule #0: InstTest0
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 4: @472
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @473
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(518), // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// CHECK-NEXT:       // Combiner Rule #2: CImmInstTest1
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_CONSTANT),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/0, /*Type*/GILLT_s32, /*Imm*/GIMT_Encode8(42),
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 5: @518
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @519
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(558), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] Operand 1
// CHECK-NEXT:       GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:       // Combiner Rule #1: InstTest1
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 6: @558
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 3: @559
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 560 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
