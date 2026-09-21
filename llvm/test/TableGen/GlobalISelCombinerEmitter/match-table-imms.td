// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck --check-prefix=DEFAULT %s
// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner -gisel-extended-llt %s | \
// RUN: FileCheck --check-prefix=EXTLLT %s

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

// fp16 and bf16 share a bit width but differ in exponent/mantissa split:
// 0x3c00 is 1.0 as f16, 0x3f80 is 1.0 as bf16.
def FPImmF16 : GICombineRule<
  (defs root:$a),
  (match (G_FCONSTANT $a, (f16 0))),
  (apply (G_FCONSTANT $a, (f16 0x3c00)))>;

def FPImmBF16 : GICombineRule<
  (defs root:$a),
  (match (G_FCONSTANT $a, (bf16 0))),
  (apply (G_FCONSTANT $a, (bf16 0x3f80)))>;

// GITypeOf +0.0. Bit pattern 0 is +0.0 for every FP width, so this is the
// portable "replace with a typed zero" form used with nested matches.
def FPImmTypeOfZero : GICombineRule<
  (defs root:$dst),
  (match (G_FNEG $t, $src),
         (G_FNEG f32:$dst, $t)),
  (apply (G_FCONSTANT $dst, (GITypeOf<"$dst"> 0)))>;

// GITypeOf with a non-zero pattern that has both an integer and a fractional
// part as f32 (0x3fc00000 == 1.5f). The bits are applied using $a's matched type.
def FPImmTypeOfFrac : GICombineRule<
  (defs root:$a),
  (match (G_FABS f32:$a, $src)),
  (apply (G_FCONSTANT $a, (GITypeOf<"$a"> 0x3fc00000)))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  InstTest1,
  CImmInstTest1,
  FPImmF32,
  FPImmF64,
  FPImmF16,
  FPImmBF16,
  FPImmTypeOfZero,
  FPImmTypeOfFrac
]>;

// Default (no -gisel-extended-llt): f16 and bf16 both collapse to the
// generic scalar GILLT_s16 and share one GIM_SwitchType case.
// GIR_AddCFPImm tags both with GILLT_s16, so getFltSemanticForLLT() cannot
// tell them apart and assumes IEEEhalf.
//      DEFAULT: const uint8_t *GenMyCombiner::getMatchTable() const {
// DEFAULT-NEXT:   constexpr static uint8_t MatchTable0[] = {
// DEFAULT-NEXT:      GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(20), GIMT_Encode2({{[0-9]+}}), /*)*//*default:*//*Label 6*/ GIMT_Encode4([[LBL6:[0-9]+]]),
// DEFAULT-NEXT:      /*TargetOpcode::COPY*//*Label 0*/ GIMT_Encode4([[LBL0:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// DEFAULT-NEXT:      /*TargetOpcode::G_CONSTANT*//*Label 1*/ GIMT_Encode4([[LBL1:[0-9]+]]),
// DEFAULT-NEXT:      /*TargetOpcode::G_FCONSTANT*//*Label 2*/ GIMT_Encode4([[LBL2:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// DEFAULT-NEXT:      /*TargetOpcode::G_ZEXT*//*Label 3*/ GIMT_Encode4([[LBL3:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// DEFAULT-NEXT:      /*TargetOpcode::G_FNEG*//*Label 4*/ GIMT_Encode4([[LBL4:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// DEFAULT-NEXT:      /*TargetOpcode::G_FABS*//*Label 5*/ GIMT_Encode4([[LBL5:[0-9]+]]),
// DEFAULT-NEXT:      // Label 0: @[[LBL0]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4([[LBL7:[0-9]+]]), // Rule ID 0 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// DEFAULT-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// DEFAULT-NEXT:        // MIs[0] a
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// DEFAULT-NEXT:        // Combiner Rule #0: InstTest0
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 7: @[[LBL7]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 1: @[[LBL1]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4([[LBL8:[0-9]+]]), // Rule ID 2 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// DEFAULT-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_s32,
// DEFAULT-NEXT:        // MIs[0] a
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// DEFAULT-NEXT:        // Combiner Rule #2: CImmInstTest1
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_CONSTANT),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddCImm, /*InsnID*/0, /*Type*/GILLT_s32, /*Imm*/GIMT_Encode8(42),
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 8: @[[LBL8]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 2: @[[LBL2]]
// DEFAULT-NEXT:      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/GIMT_Encode2(0), GIMT_Encode2(3), /*)*//*default:*//*Label 12*/ GIMT_Encode4([[LBL12:[0-9]+]]),
// DEFAULT-NEXT:      /*GILLT_s16*//*Label 9*/ GIMT_Encode4([[LBL9:[0-9]+]]),
// DEFAULT-NEXT:      /*GILLT_s32*//*Label 10*/ GIMT_Encode4([[LBL10:[0-9]+]]),
// DEFAULT-NEXT:      /*GILLT_s64*//*Label 11*/ GIMT_Encode4([[LBL11:[0-9]+]]),
// DEFAULT-NEXT:      // Label 9: @[[LBL9]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 13*/ GIMT_Encode4([[LBL13:[0-9]+]]),
// DEFAULT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// DEFAULT-NEXT:        GIM_Try, /*On fail goto*//*Label 14*/ GIMT_Encode4([[LBL14:[0-9]+]]), // Rule ID 5 //
// DEFAULT-NEXT:          GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule5Enabled),
// DEFAULT-NEXT:          // MIs[0] a
// DEFAULT-NEXT:          // No operand predicates
// DEFAULT-NEXT:          // Combiner Rule #5: FPImmF16
// DEFAULT-NEXT:          GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:          GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:          GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s16, /*Imm*/GIMT_Encode8(15360),
// DEFAULT-NEXT:          GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:        // Label 14: @[[LBL14]]
// DEFAULT-NEXT:        GIM_Try, /*On fail goto*//*Label 15*/ GIMT_Encode4([[LBL15:[0-9]+]]), // Rule ID 6 //
// DEFAULT-NEXT:          GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule6Enabled),
// DEFAULT-NEXT:          // MIs[0] a
// DEFAULT-NEXT:          // No operand predicates
// DEFAULT-NEXT:          // Combiner Rule #6: FPImmBF16
// DEFAULT-NEXT:          GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:          GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:          GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s16, /*Imm*/GIMT_Encode8(16256),
// DEFAULT-NEXT:          GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:        // Label 15: @[[LBL15]]
// DEFAULT-NEXT:        GIM_Reject,
// DEFAULT-NEXT:      // Label 13: @[[LBL13]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 10: @[[LBL10]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 16*/ GIMT_Encode4([[LBL16:[0-9]+]]), // Rule ID 3 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// DEFAULT-NEXT:        // MIs[0] a
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// DEFAULT-NEXT:        // Combiner Rule #3: FPImmF32
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s32, /*Imm*/GIMT_Encode8(1065353216),
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 16: @[[LBL16]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 11: @[[LBL11]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 17*/ GIMT_Encode4([[LBL17:[0-9]+]]), // Rule ID 4 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// DEFAULT-NEXT:        // MIs[0] a
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// DEFAULT-NEXT:        // Combiner Rule #4: FPImmF64
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_s64, /*Imm*/GIMT_Encode8(4607182418800017408),
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 17: @[[LBL17]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 12: @[[LBL12]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 3: @[[LBL3]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 18*/ GIMT_Encode4([[LBL18:[0-9]+]]), // Rule ID 1 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// DEFAULT-NEXT:        // MIs[0] a
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        // MIs[0] Operand 1
// DEFAULT-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// DEFAULT-NEXT:        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// DEFAULT-NEXT:        GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// DEFAULT-NEXT:        // Combiner Rule #1: InstTest1
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 18: @[[LBL18]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 4: @[[LBL4]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 19*/ GIMT_Encode4([[LBL19:[0-9]+]]), // Rule ID 7 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule7Enabled),
// DEFAULT-NEXT:        GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// DEFAULT-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// DEFAULT-NEXT:        // MIs[0] t
// DEFAULT-NEXT:        GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// DEFAULT-NEXT:        GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FNEG),
// DEFAULT-NEXT:        // MIs[1] src
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        GIM_CheckIsSafeToFold, /*NumInsns*/1,
// DEFAULT-NEXT:        // Combiner Rule #7: FPImmTypeOfZero
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // dst
// DEFAULT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(0),
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 19: @[[LBL19]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 5: @[[LBL5]]
// DEFAULT-NEXT:      GIM_Try, /*On fail goto*//*Label 20*/ GIMT_Encode4([[LBL20:[0-9]+]]), // Rule ID 8 //
// DEFAULT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule8Enabled),
// DEFAULT-NEXT:        GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_s32,
// DEFAULT-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// DEFAULT-NEXT:        // MIs[0] src
// DEFAULT-NEXT:        // No operand predicates
// DEFAULT-NEXT:        // Combiner Rule #8: FPImmTypeOfFrac
// DEFAULT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// DEFAULT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// DEFAULT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(1069547520),
// DEFAULT-NEXT:        GIR_EraseRootFromParent_Done,
// DEFAULT-NEXT:      // Label 20: @[[LBL20]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      // Label 6: @[[LBL6]]
// DEFAULT-NEXT:      GIM_Reject,
// DEFAULT-NEXT:      }; // Size: {{[0-9]+}} bytes
// DEFAULT-NEXT:   return MatchTable0;
// DEFAULT-NEXT: }

// With -gisel-extended-llt (used by AMDGPU's combiner tables): f16 and
// bf16 get distinct GILLT_f16/GILLT_bf16 entries tagged with their real
// FpSemantics, so GIR_AddCFPImm's type disambiguates them correctly.
//      EXTLLT: enum {
// EXTLLT-NEXT:   GILLT_i32,
// EXTLLT-NEXT:   GILLT_f16,
// EXTLLT-NEXT:   GILLT_bf16,
// EXTLLT-NEXT:   GILLT_f32,
// EXTLLT-NEXT:   GILLT_f64,
// EXTLLT-NEXT: };
//      EXTLLT: const uint8_t *GenMyCombiner::getMatchTable() const {
// EXTLLT-NEXT:   constexpr static uint8_t MatchTable0[] = {
// EXTLLT-NEXT:      GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(20), GIMT_Encode2({{[0-9]+}}), /*)*//*default:*//*Label 6*/ GIMT_Encode4([[LBL6:[0-9]+]]),
// EXTLLT-NEXT:      /*TargetOpcode::COPY*//*Label 0*/ GIMT_Encode4([[LBL0:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// EXTLLT-NEXT:      /*TargetOpcode::G_CONSTANT*//*Label 1*/ GIMT_Encode4([[LBL1:[0-9]+]]),
// EXTLLT-NEXT:      /*TargetOpcode::G_FCONSTANT*//*Label 2*/ GIMT_Encode4([[LBL2:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// EXTLLT-NEXT:      /*TargetOpcode::G_ZEXT*//*Label 3*/ GIMT_Encode4([[LBL3:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// EXTLLT-NEXT:      /*TargetOpcode::G_FNEG*//*Label 4*/ GIMT_Encode4([[LBL4:[0-9]+]]), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// EXTLLT-NEXT:      /*TargetOpcode::G_FABS*//*Label 5*/ GIMT_Encode4([[LBL5:[0-9]+]]),
// EXTLLT-NEXT:      // Label 0: @[[LBL0]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4([[LBL7:[0-9]+]]), // Rule ID 0 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// EXTLLT-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_i32,
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// EXTLLT-NEXT:        // Combiner Rule #0: InstTest0
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddImm8, /*InsnID*/0, /*Imm*/0,
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 7: @[[LBL7]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 1: @[[LBL1]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4([[LBL8:[0-9]+]]), // Rule ID 2 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// EXTLLT-NEXT:        GIM_RootCheckType, /*Op*/1, /*Type*/GILLT_i32,
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #2: CImmInstTest1
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_CONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCImm, /*InsnID*/0, /*Type*/GILLT_i32, /*Imm*/GIMT_Encode8(42),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 8: @[[LBL8]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 2: @[[LBL2]]
// EXTLLT-NEXT:      GIM_SwitchType, /*MI*/0, /*Op*/1, /*[*/GIMT_Encode2(1), GIMT_Encode2(5), /*)*//*default:*//*Label 13*/ GIMT_Encode4([[LBL13:[0-9]+]]),
// EXTLLT-NEXT:      /*GILLT_f16*//*Label 9*/ GIMT_Encode4([[LBL9:[0-9]+]]),
// EXTLLT-NEXT:      /*GILLT_bf16*//*Label 10*/ GIMT_Encode4([[LBL10:[0-9]+]]),
// EXTLLT-NEXT:      /*GILLT_f32*//*Label 11*/ GIMT_Encode4([[LBL11:[0-9]+]]),
// EXTLLT-NEXT:      /*GILLT_f64*//*Label 12*/ GIMT_Encode4([[LBL12:[0-9]+]]),
// EXTLLT-NEXT:      // Label 9: @[[LBL9]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 14*/ GIMT_Encode4([[LBL14:[0-9]+]]), // Rule ID 5 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule5Enabled),
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #5: FPImmF16
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_f16, /*Imm*/GIMT_Encode8(15360),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 14: @[[LBL14]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 10: @[[LBL10]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 15*/ GIMT_Encode4([[LBL15:[0-9]+]]), // Rule ID 6 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule6Enabled),
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #6: FPImmBF16
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_bf16, /*Imm*/GIMT_Encode8(16256),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 15: @[[LBL15]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 11: @[[LBL11]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 16*/ GIMT_Encode4([[LBL16:[0-9]+]]), // Rule ID 3 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #3: FPImmF32
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_f32, /*Imm*/GIMT_Encode8(1065353216),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 16: @[[LBL16]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 12: @[[LBL12]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 17*/ GIMT_Encode4([[LBL17:[0-9]+]]), // Rule ID 4 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #4: FPImmF64
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/GILLT_f64, /*Imm*/GIMT_Encode8(4607182418800017408),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 17: @[[LBL17]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 13: @[[LBL13]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 3: @[[LBL3]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 18*/ GIMT_Encode4([[LBL18:[0-9]+]]), // Rule ID 1 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// EXTLLT-NEXT:        // MIs[0] a
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        // MIs[0] Operand 1
// EXTLLT-NEXT:        GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// EXTLLT-NEXT:        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_i32,
// EXTLLT-NEXT:        GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// EXTLLT-NEXT:        // Combiner Rule #1: InstTest1
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 18: @[[LBL18]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 4: @[[LBL4]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 19*/ GIMT_Encode4([[LBL19:[0-9]+]]), // Rule ID 7 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule7Enabled),
// EXTLLT-NEXT:        GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_f32,
// EXTLLT-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// EXTLLT-NEXT:        // MIs[0] t
// EXTLLT-NEXT:        GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// EXTLLT-NEXT:        GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FNEG),
// EXTLLT-NEXT:        // MIs[1] src
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        GIM_CheckIsSafeToFold, /*NumInsns*/1,
// EXTLLT-NEXT:        // Combiner Rule #7: FPImmTypeOfZero
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // dst
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(0),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 19: @[[LBL19]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 5: @[[LBL5]]
// EXTLLT-NEXT:      GIM_Try, /*On fail goto*//*Label 20*/ GIMT_Encode4([[LBL20:[0-9]+]]), // Rule ID 8 //
// EXTLLT-NEXT:        GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule8Enabled),
// EXTLLT-NEXT:        GIM_RootCheckType, /*Op*/0, /*Type*/GILLT_f32,
// EXTLLT-NEXT:        GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// EXTLLT-NEXT:        // MIs[0] src
// EXTLLT-NEXT:        // No operand predicates
// EXTLLT-NEXT:        // Combiner Rule #8: FPImmTypeOfFrac
// EXTLLT-NEXT:        GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// EXTLLT-NEXT:        GIR_RootToRootCopy, /*OpIdx*/0, // a
// EXTLLT-NEXT:        GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(1069547520),
// EXTLLT-NEXT:        GIR_EraseRootFromParent_Done,
// EXTLLT-NEXT:      // Label 20: @[[LBL20]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      // Label 6: @[[LBL6]]
// EXTLLT-NEXT:      GIM_Reject,
// EXTLLT-NEXT:      }; // Size: {{[0-9]+}} bytes
// EXTLLT-NEXT:   return MatchTable0;
// EXTLLT-NEXT: }
