// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner -gicombiner-debug-cxxpreds %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def MatchFooPerms: GICombinePatFrag<
    (outs root:$foo),
    (ins gi_imm:$cst),
    [
      (pattern (G_ZEXT $foo, $b), (G_TRUNC $b, $x):$dbg0),
      (pattern (G_TRUNC $foo, $z):$dbg1),
      (pattern (G_FPEXT $foo, $z):$dbg1)
    ]>;

def Test0 : GICombineRule<
  (defs root:$root),
  (match (MatchFooPerms $root, (i32 10))),
  (apply (COPY $root, (i32 0)))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  Test0
]>;

// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 44, // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ZEXT,
// CHECK-NEXT:       // MIs[0] root
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] __Test0_match_0.b
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_TRUNC,
// CHECK-NEXT:       // MIs[1] __Test0_match_0.x
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/1, /*Type*/GILLT_s32, /*Imm*/0,
// CHECK-NEXT:       // Combiner Rule #0: Test0 @ [__Test0_match_0[0]]
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 0: @44
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 1*/ 79, // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_TRUNC,
// CHECK-NEXT:       // MIs[0] root
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] __Test0_match_0.z
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/1, /*Type*/GILLT_s32, /*Imm*/0,
// CHECK-NEXT:       // Combiner Rule #0: Test0 @ [__Test0_match_0[1]]
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 1: @79
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 2*/ 114, // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_FPEXT,
// CHECK-NEXT:       // MIs[0] root
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] __Test0_match_0.z
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/1, /*Type*/GILLT_s32, /*Imm*/0,
// CHECK-NEXT:       // Combiner Rule #0: Test0 @ [__Test0_match_0[2]]
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 2: @114
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
