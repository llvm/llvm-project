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

// G_FCONSTANT apply with a concrete f32 IEEE bit pattern. The pattern grammar
// has no fp literals, so 0x3f800000 is 1.0f (not the integer 1).
def FPImmF32 : GICombineRule<
  (defs root:$a),
  (match (G_FCONSTANT $a, (f32 0))),
  (apply (G_FCONSTANT $a, (f32 0x3f800000)))>;

// Same for f64: 0x3ff0000000000000 is 1.0.
def FPImmF64 : GICombineRule<
  (defs root:$a),
  (match (G_FCONSTANT $a, (f64 0))),
  (apply (G_FCONSTANT $a, (f64 0x3ff0000000000000)))>;

// GITypeOf +0.0. Bit pattern 0 is +0.0 for every FP width, so this is the
// portable "replace with a typed zero" form used with nested matches.
def FPImmTypeOfZero : GICombineRule<
  (defs root:$dst),
  (match (G_FNEG $t, $src),
         (G_FNEG $dst, $t)),
  (apply (G_FCONSTANT $dst, (GITypeOf<"$dst"> 0)))>;

// GITypeOf with a non-zero pattern that has both an integer and a fractional
// part as f32 (0x3fc00000 == 1.5f). The bits are applied using $a's matched type.
def FPImmTypeOfFrac : GICombineRule<
  (defs root:$a),
  (match (G_FABS $a, $src)),
  (apply (G_FCONSTANT $a, (GITypeOf<"$a"> 0x3fc00000)))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  InstTest1,
  CImmInstTest1,
  FPImmF32,
  FPImmF64,
  FPImmTypeOfZero,
  FPImmTypeOfFrac
]>;

//      CHECK: const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:      GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(20), GIMT_Encode2({{[0-9]+}}), /*)*//*default:*//*Label 6*/ GIMT_Encode4({{[0-9]+}}),
// CHECK-NEXT:      /*TargetOpcode::COPY*//*Label 0*/ GIMT_Encode4(814), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /*TargetOpcode::G_CONSTANT*//*Label 1*/ GIMT_Encode4(839),
// CHECK-NEXT:      /*TargetOpcode::G_FCONSTANT*//*Label 2*/ GIMT_Encode4(879), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /*TargetOpcode::G_ZEXT*//*Label 3*/ GIMT_Encode4(973), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /*TargetOpcode::G_FNEG*//*Label 4*/ GIMT_Encode4(1008), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /*TargetOpcode::G_FABS*//*Label 5*/ GIMT_Encode4(1048),
// CHECK-NEXT:      // Label 0: @814
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(838), // Rule ID 0 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:        // Combiner Rule #0: InstTest0
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 7: @838
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 1: @839
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(878), // Rule ID 2 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// CHECK-NEXT:        // Combiner Rule #2: CImmInstTest1
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_CONSTANT),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddCImm, /*InsnID*/0, /*Type*/GILLT_s32, /*Imm*/GIMT_Encode8(42),
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 8: @878
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 2: @879
// CHECK-NEXT:      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/GIMT_Encode2(0), GIMT_Encode2(2), /*)*//*default:*//*Label 11*/ GIMT_Encode4(972),
// CHECK-NEXT:      /*GILLT_s32*//*Label 9*/ GIMT_Encode4(898),
// CHECK-NEXT:      /*GILLT_s64*//*Label 10*/ GIMT_Encode4(935),
// CHECK-NEXT:      // Label 9: @898
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 12*/ GIMT_Encode4(934), // Rule ID 3 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// CHECK-NEXT:        // Combiner Rule #3: FPImmF32
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s32, /*Imm*/GIMT_Encode8(1065353216),
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 12: @934
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 10: @935
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 13*/ GIMT_Encode4(971), // Rule ID 4 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// CHECK-NEXT:        // Combiner Rule #4: FPImmF64
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s64, /*Imm*/GIMT_Encode8(4607182418800017408),
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 13: @971
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 11: @972
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 3: @973
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 14*/ GIMT_Encode4(1007), // Rule ID 1 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        // MIs[0] Operand 1
// CHECK-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:        GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:        // Combiner Rule #1: InstTest1
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 14: @1007
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 4: @1008
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 15*/ GIMT_Encode4(1047), // Rule ID 5 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule5Enabled),
// CHECK-NEXT:        // MIs[0] dst
// CHECK-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:        // MIs[0] t
// CHECK-NEXT:        GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:        GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FNEG),
// CHECK-NEXT:        // MIs[1] src
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:        // Combiner Rule #5: FPImmTypeOfZero
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // dst
// CHECK-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(0),
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 15: @1047
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 5: @1048
// CHECK-NEXT:      GIM_Try, /*On fail goto*//*Label 16*/ GIMT_Encode4(1077), // Rule ID 6 //
// CHECK-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule6Enabled),
// CHECK-NEXT:        // MIs[0] a
// CHECK-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:        // MIs[0] src
// CHECK-NEXT:        // No operand predicates
// CHECK-NEXT:        // Combiner Rule #6: FPImmTypeOfFrac
// CHECK-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// CHECK-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(1069547520),
// CHECK-NEXT:        GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      // Label 16: @1077
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      // Label 6: @1078
// CHECK-NEXT:      GIM_Reject,
// CHECK-NEXT:      }; // Size: {{[0-9]+}} bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
