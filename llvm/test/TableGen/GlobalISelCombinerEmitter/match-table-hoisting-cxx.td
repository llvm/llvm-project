// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"


def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// These two combine rules both use an identical C++ predicate, but we cannot
// hoist because the combiner assumes C++ predicates always run last.
// We do not depend on explicitly recording operands for C++ predicates to access them.
def SharedCXXPred0: GICombineRule<
   (defs root:$root),
   (match (G_SUB $sub, 0, $A),
          (G_UMAX $max, $A, $sub),
          (G_SUB $root, 0, $max),
          [{ return MRI.hasOneNonDBGUse(${sub}.getReg()) &&
                    MRI.hasOneNonDBGUse(${max}.getReg()); }]),
   (apply (G_SUB $sub1, 0, $A),
          (G_UMIN $root, $A, $sub1))>;

def SharedCXXPred1: GICombineRule<
   (defs root:$root),
   (match (G_SUB $sub, 0, $A),
          (G_SMAX $max, $A, $sub),
          (G_SUB $root, 0, $max),
          [{ return MRI.hasOneNonDBGUse(${sub}.getReg()) &&
                    MRI.hasOneNonDBGUse(${max}.getReg()); }]),
   (apply (G_SUB $sub1, 0, $A),
          (G_SMIN $root, $A, $sub1))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  SharedCXXPred0,
  SharedCXXPred1
]>;

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:      /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(386),
// CHECK-NEXT:      /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*   9 */   GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(103), // Rule ID 2 //
// CHECK-NEXT:      /*  14 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /*  17 */     // MIs[0] root
// CHECK-NEXT:      /*  17 */     GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:      /*  21 */     // MIs[0] Operand 1
// CHECK-NEXT:      /*  21 */     GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:      /*  25 */     // MIs[0] max
// CHECK-NEXT:      /*  25 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:      /*  29 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_SMAX),
// CHECK-NEXT:      /*  33 */     // MIs[1] A
// CHECK-NEXT:      /*  33 */     // No operand predicates
// CHECK-NEXT:      /*  33 */     // MIs[1] sub
// CHECK-NEXT:      /*  33 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /*  37 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*  41 */     // MIs[2] Operand 1
// CHECK-NEXT:      /*  41 */     GIM_CheckConstantInt8, /*MI*/2, /*Op*/1, 0,
// CHECK-NEXT:      /*  45 */     // MIs[2] A
// CHECK-NEXT:      /*  45 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
// CHECK-NEXT:      /*  50 */     GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:      /*  54 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /*  56 */     GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/255,
// CHECK-NEXT:      /*  59 */     GIR_BuildConstant, /*TempRegID*/1, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:      /*  69 */     GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/255,
// CHECK-NEXT:      /*  72 */     // Combiner Rule #1: SharedCXXPred1
// CHECK-NEXT:      /*  72 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*  75 */     GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:      /*  80 */     GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:      /*  83 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  87 */     GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SMIN),
// CHECK-NEXT:      /*  91 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:      /*  95 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  99 */     GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:      /* 102 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 103 */   // Label 1: @103
// CHECK-NEXT:      /* 103 */   GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(197), // Rule ID 3 //
// CHECK-NEXT:      /* 108 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /* 111 */     // MIs[0] root
// CHECK-NEXT:      /* 111 */     GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:      /* 115 */     // MIs[0] Operand 1
// CHECK-NEXT:      /* 115 */     GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:      /* 119 */     // MIs[0] max
// CHECK-NEXT:      /* 119 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:      /* 123 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_SMAX),
// CHECK-NEXT:      /* 127 */     // MIs[1] A
// CHECK-NEXT:      /* 127 */     // No operand predicates
// CHECK-NEXT:      /* 127 */     // MIs[1] sub
// CHECK-NEXT:      /* 127 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:      /* 131 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 135 */     // MIs[2] Operand 1
// CHECK-NEXT:      /* 135 */     GIM_CheckConstantInt8, /*MI*/2, /*Op*/1, 0,
// CHECK-NEXT:      /* 139 */     // MIs[2] A
// CHECK-NEXT:      /* 139 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
// CHECK-NEXT:      /* 144 */     GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:      /* 148 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 150 */     GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/255,
// CHECK-NEXT:      /* 153 */     GIR_BuildConstant, /*TempRegID*/1, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:      /* 163 */     GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/255,
// CHECK-NEXT:      /* 166 */     // Combiner Rule #1: SharedCXXPred1 @ [__SharedCXXPred1_match_1[commuted]]
// CHECK-NEXT:      /* 166 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 169 */     GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:      /* 174 */     GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:      /* 177 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 181 */     GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SMIN),
// CHECK-NEXT:      /* 185 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 189 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 193 */     GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:      /* 196 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 197 */   // Label 2: @197
// CHECK-NEXT:      /* 197 */   GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(291), // Rule ID 0 //
// CHECK-NEXT:      /* 202 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 205 */     // MIs[0] root
// CHECK-NEXT:      /* 205 */     GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:      /* 209 */     // MIs[0] Operand 1
// CHECK-NEXT:      /* 209 */     GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:      /* 213 */     // MIs[0] max
// CHECK-NEXT:      /* 213 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:      /* 217 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_UMAX),
// CHECK-NEXT:      /* 221 */     // MIs[1] A
// CHECK-NEXT:      /* 221 */     // No operand predicates
// CHECK-NEXT:      /* 221 */     // MIs[1] sub
// CHECK-NEXT:      /* 221 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /* 225 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 229 */     // MIs[2] Operand 1
// CHECK-NEXT:      /* 229 */     GIM_CheckConstantInt8, /*MI*/2, /*Op*/1, 0,
// CHECK-NEXT:      /* 233 */     // MIs[2] A
// CHECK-NEXT:      /* 233 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
// CHECK-NEXT:      /* 238 */     GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:      /* 242 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 244 */     GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/255,
// CHECK-NEXT:      /* 247 */     GIR_BuildConstant, /*TempRegID*/1, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:      /* 257 */     GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/255,
// CHECK-NEXT:      /* 260 */     // Combiner Rule #0: SharedCXXPred0
// CHECK-NEXT:      /* 260 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 263 */     GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:      /* 268 */     GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:      /* 271 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /* 275 */     GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_UMIN),
// CHECK-NEXT:      /* 279 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 283 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /* 287 */     GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:      /* 290 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 291 */   // Label 3: @291
// CHECK-NEXT:      /* 291 */   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(385), // Rule ID 1 //
// CHECK-NEXT:      /* 296 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 299 */     // MIs[0] root
// CHECK-NEXT:      /* 299 */     GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:      /* 303 */     // MIs[0] Operand 1
// CHECK-NEXT:      /* 303 */     GIM_CheckConstantInt8, /*MI*/0, /*Op*/1, 0,
// CHECK-NEXT:      /* 307 */     // MIs[0] max
// CHECK-NEXT:      /* 307 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:      /* 311 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_UMAX),
// CHECK-NEXT:      /* 315 */     // MIs[1] A
// CHECK-NEXT:      /* 315 */     // No operand predicates
// CHECK-NEXT:      /* 315 */     // MIs[1] sub
// CHECK-NEXT:      /* 315 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:      /* 319 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 323 */     // MIs[2] Operand 1
// CHECK-NEXT:      /* 323 */     GIM_CheckConstantInt8, /*MI*/2, /*Op*/1, 0,
// CHECK-NEXT:      /* 327 */     // MIs[2] A
// CHECK-NEXT:      /* 327 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/2, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
// CHECK-NEXT:      /* 332 */     GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:      /* 336 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 338 */     GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/255,
// CHECK-NEXT:      /* 341 */     GIR_BuildConstant, /*TempRegID*/1, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:      /* 351 */     GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/255,
// CHECK-NEXT:      /* 354 */     // Combiner Rule #0: SharedCXXPred0 @ [__SharedCXXPred0_match_1[commuted]]
// CHECK-NEXT:      /* 354 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 357 */     GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:      /* 362 */     GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:      /* 365 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 369 */     GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_UMIN),
// CHECK-NEXT:      /* 373 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 377 */     GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 381 */     GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:      /* 384 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 385 */   // Label 4: @385
// CHECK-NEXT:      /* 385 */   GIM_Reject,
// CHECK-NEXT:      /* 386 */ // Label 0: @386
// CHECK-NEXT:      /* 386 */ GIM_Reject,
// CHECK-NEXT:      /* 387 */ }; // Size: 387 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
