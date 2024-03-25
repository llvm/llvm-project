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
      (pattern (G_ZEXT $foo, $b), (G_TRUNC $b, $x):$dbg0, "return foo(${x}, ${cst})"),
      (pattern (G_TRUNC $foo, $z):$dbg1, "return bar(${foo}, ${cst})")
    ]>;

def Test0 : GICombineRule<
  (defs root:$dst),
  (match (G_AND $dst, $cst0, $tmp),
         (G_AND $tmp, $cst1, $cst2),
         (MatchFooPerms $cst0, (i32 10)):$a,
         (MatchFooPerms $cst1, (i32 20)):$b,
         (MatchFooPerms $cst2, (i32 30)):$c
  ),
  (apply (COPY $dst, (i32 0)), "APPLY ${cst0}")>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  Test0
]>;

// CHECK:      bool GenMyCombiner::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI, const MatcherState &State) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[0]]
// CHECK-NEXT:     return foo(State.MIs[2]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner0 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[0]]
// CHECK-NEXT:     return foo(State.MIs[5]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner1 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner2: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[0]]
// CHECK-NEXT:     return foo(State.MIs[7]->getOperand(1), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner2 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner3: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[1]]
// CHECK-NEXT:     return foo(State.MIs[2]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner3 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner4: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[1]]
// CHECK-NEXT:     return foo(State.MIs[5]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner4 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner5: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[0], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(2), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner5 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner6: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[0]]
// CHECK-NEXT:     return foo(State.MIs[2]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner6 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner7: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[0]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner7 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner8: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[0]]
// CHECK-NEXT:     return foo(State.MIs[6]->getOperand(1), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner8 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner9: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[1]]
// CHECK-NEXT:     return foo(State.MIs[2]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner9 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner10: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner10 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner11: {
// CHECK-NEXT:     // Pattern Alternatives: [a[0], b[1], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(2), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner11 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner12: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[0]]
// CHECK-NEXT:     return bar(State.MIs[0]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner12 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner13: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[0]]
// CHECK-NEXT:     return foo(State.MIs[4]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner13 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner14: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[0]]
// CHECK-NEXT:     return foo(State.MIs[6]->getOperand(1), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner14 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner15: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[1]]
// CHECK-NEXT:     return bar(State.MIs[0]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner15 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner16: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[1]]
// CHECK-NEXT:     return foo(State.MIs[4]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner16 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner17: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[0], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(2), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner17 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner18: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[0]]
// CHECK-NEXT:     return bar(State.MIs[0]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner18 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner19: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[0]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner19 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner20: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[0]]
// CHECK-NEXT:     return foo(State.MIs[5]->getOperand(1), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner20 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner21: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[1]]
// CHECK-NEXT:     return bar(State.MIs[0]->getOperand(1), 10)
// CHECK-NEXT:     llvm_unreachable("GICombiner21 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner22: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(1), 20)
// CHECK-NEXT:     llvm_unreachable("GICombiner22 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner23: {
// CHECK-NEXT:     // Pattern Alternatives: [a[1], b[1], c[1]]
// CHECK-NEXT:     return bar(State.MIs[3]->getOperand(2), 30)
// CHECK-NEXT:     llvm_unreachable("GICombiner23 should have returned");
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown predicate");
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(850),
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(99), // Rule ID 7 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[1] a.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[2] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[3] b.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[2] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/2, /*OpIdx*/2, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[4] c.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner21),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner22),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner23),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[1], b[1], c[1]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 1: @99
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(199), // Rule ID 6 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[1] a.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[2] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[3] b.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[2] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/2, /*OpIdx*/2, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[4] c.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/4, /*OpIdx*/1, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[5] c.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner18),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner19),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner20),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[1], b[1], c[0]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 2: @199
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(299), // Rule ID 5 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[1] a.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[2] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[3] b.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[4] b.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[2] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/2, /*OpIdx*/2, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[5] c.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner15),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner16),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner17),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[1], b[0], c[1]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 3: @299
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(409), // Rule ID 4 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[1] a.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[2] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/2, /*OpIdx*/1, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[3] b.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[4] b.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[2] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/2, /*OpIdx*/2, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[5] c.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/6, /*MI*/5, /*OpIdx*/1, // MIs[6]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/6, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[6] c.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner12),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner13),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner14),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/6,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[1], b[0], c[0]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 4: @409
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(509), // Rule ID 3 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[1] a.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[2] a.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/0, /*OpIdx*/2, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[3] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[4] b.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[3] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/3, /*OpIdx*/2, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[5] c.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner9),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner10),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner11),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[0], b[1], c[1]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 5: @509
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(619), // Rule ID 2 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[1] a.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[2] a.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/0, /*OpIdx*/2, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[3] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[4] b.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[3] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/3, /*OpIdx*/2, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[5] c.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/6, /*MI*/5, /*OpIdx*/1, // MIs[6]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/6, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[6] c.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner6),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner7),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner8),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/6,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[0], b[1], c[0]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 6: @619
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(729), // Rule ID 1 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[1] a.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[2] a.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/0, /*OpIdx*/2, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[3] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[4] b.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/4, /*OpIdx*/1, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[5] b.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[3] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/6, /*MI*/3, /*OpIdx*/2, // MIs[6]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/6, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[6] c.z
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner3),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner4),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner5),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/6,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[0], b[0], c[1]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 7: @729
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(849), // Rule ID 0 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:         // MIs[0] dst
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] cst0
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[1] a.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[2] a.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] tmp
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/3, /*MI*/0, /*OpIdx*/2, // MIs[3]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/3, GIMT_Encode2(TargetOpcode::G_AND),
// CHECK-NEXT:         // MIs[3] cst1
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/4, /*MI*/3, /*OpIdx*/1, // MIs[4]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/4, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[4] b.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/5, /*MI*/4, /*OpIdx*/1, // MIs[5]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/5, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[5] b.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[3] cst2
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/6, /*MI*/3, /*OpIdx*/2, // MIs[6]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/6, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:         // MIs[6] c.b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/7, /*MI*/6, /*OpIdx*/1, // MIs[7]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/7, GIMT_Encode2(TargetOpcode::G_TRUNC),
// CHECK-NEXT:         // MIs[7] c.x
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner2),
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/2,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/3,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/4,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/5,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/6,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/7,
// CHECK-NEXT:         GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:         GIR_BuildConstant, /*TempRegID*/0, /*Val*/GIMT_Encode8(0),
// CHECK-NEXT:         // Combiner Rule #0: Test0 @ [a[0], b[0], c[0]]
// CHECK-NEXT:         GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::COPY),
// CHECK-NEXT:         GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:         GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/0,
// CHECK-NEXT:         GIR_CustomAction, GIMT_Encode2(GICXXCustomAction_CombineApplyGICombiner0),
// CHECK-NEXT:         GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 8: @849
// CHECK-NEXT:       GIM_Reject,
// CHECK-NEXT:     // Label 0: @850
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 851 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
