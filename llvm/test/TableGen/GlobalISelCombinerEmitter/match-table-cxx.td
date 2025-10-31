// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def OneMatchOneApply : GICombineRule<
  (defs root:$a),
  (match (G_FABS $a, $b), "return MATCH0;"),
  (apply "APPLY0")>;

def TwoMatchTwoApply : GICombineRule<
  (defs root:$a),
  (match (G_FNEG $a, $b), "return MATCH0;", "return MATCH1;"),
  (apply "APPLY0", "APPLY1")>;

def TwoMatchNoApply : GICombineRule<
  (defs root:$a),
  (match (G_STORE $x, $y):$a, "return MATCH0;", "return MATCH1;"),
  (apply (GIEraseRoot))>;

def NoMatchTwoApply : GICombineRule<
  (defs root:$a),
  (match (G_SEXT $a, $y)),
  (apply "APPLY0", "APPLY1")>;

def CombineCXXOrder : GICombineRule<
  (defs root:$a),
  (combine (G_ZEXT $a, $y), "A0", "return A1")>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  OneMatchOneApply,
  TwoMatchTwoApply,
  TwoMatchNoApply,
  NoMatchTwoApply,
  CombineCXXOrder
]>;

// CHECK:      bool GenMyCombiner::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI, const MatcherState &State) const {
// CHECK-NEXT:   switch (PredicateID) {
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:     return MATCH0;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:     return MATCH1;
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown predicate");
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK:      bool GenMyCombiner::runCustomAction(unsigned ApplyID, const MatcherState &State, NewMIVector &OutMIs) const {
// CHECK-NEXT:   Helper.getBuilder().setInstrAndDebugLoc(*State.MIs[0]);
// CHECK-NEXT:   switch(ApplyID) {
// CHECK-NEXT:   case GICXXCustomAction_GICombiner0:{
// CHECK-NEXT:     // Match Patterns
// CHECK-NEXT:     if(![&](){return MATCH0;}()) {
// CHECK-NEXT:       return false;
// CHECK-NEXT:     }
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     APPLY0
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_GICombiner1:{
// CHECK-NEXT:     // Match Patterns
// CHECK-NEXT:     if(![&](){return MATCH0;}()) {
// CHECK-NEXT:       return false;
// CHECK-NEXT:     }
// CHECK-NEXT:     if(![&](){return MATCH1;}()) {
// CHECK-NEXT:       return false;
// CHECK-NEXT:     }
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     APPLY0
// CHECK-NEXT:     APPLY1
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_GICombiner2:{
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     APPLY0
// CHECK-NEXT:     APPLY1
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_GICombiner3:{
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     A0
// CHECK-NEXT:     return A1
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown Apply Action");
// CHECK-NEXT: }

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     /*   0 */ GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(99), GIMT_Encode2(212), /*)*//*default:*//*Label 5*/ GIMT_Encode4(528),
// CHECK-NEXT:     /*  10 */ /*TargetOpcode::G_STORE*//*Label 0*/ GIMT_Encode4(462), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /* 182 */ /*TargetOpcode::G_SEXT*//*Label 1*/ GIMT_Encode4(480), GIMT_Encode4(0),
// CHECK-NEXT:     /* 190 */ /*TargetOpcode::G_ZEXT*//*Label 2*/ GIMT_Encode4(492), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /* 418 */ /*TargetOpcode::G_FNEG*//*Label 3*/ GIMT_Encode4(504), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /* 458 */ /*TargetOpcode::G_FABS*//*Label 4*/ GIMT_Encode4(516),
// CHECK-NEXT:     /* 462 */ // Label 0: @462
// CHECK-NEXT:     /* 462 */ GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(479), // Rule ID 2 //
// CHECK-NEXT:     /* 467 */   GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:     /* 470 */   // MIs[0] x
// CHECK-NEXT:     /* 470 */   // No operand predicates
// CHECK-NEXT:     /* 470 */   // MIs[0] y
// CHECK-NEXT:     /* 470 */   // No operand predicates
// CHECK-NEXT:     /* 470 */   GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:     /* 474 */   GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:     /* 478 */   // Combiner Rule #2: TwoMatchNoApply
// CHECK-NEXT:     /* 478 */   GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     /* 479 */ // Label 6: @479
// CHECK-NEXT:     /* 479 */ GIM_Reject,
// CHECK-NEXT:     /* 480 */ // Label 1: @480
// CHECK-NEXT:     /* 480 */ GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(491), // Rule ID 3 //
// CHECK-NEXT:     /* 485 */   GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:     /* 488 */   // MIs[0] a
// CHECK-NEXT:     /* 488 */   // No operand predicates
// CHECK-NEXT:     /* 488 */   // MIs[0] y
// CHECK-NEXT:     /* 488 */   // No operand predicates
// CHECK-NEXT:     /* 488 */   // Combiner Rule #3: NoMatchTwoApply
// CHECK-NEXT:     /* 488 */   GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner2),
// CHECK-NEXT:     /* 491 */ // Label 7: @491
// CHECK-NEXT:     /* 491 */ GIM_Reject,
// CHECK-NEXT:     /* 492 */ // Label 2: @492
// CHECK-NEXT:     /* 492 */ GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(503), // Rule ID 4 //
// CHECK-NEXT:     /* 497 */   GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// CHECK-NEXT:     /* 500 */   // MIs[0] a
// CHECK-NEXT:     /* 500 */   // No operand predicates
// CHECK-NEXT:     /* 500 */   // MIs[0] y
// CHECK-NEXT:     /* 500 */   // No operand predicates
// CHECK-NEXT:     /* 500 */   // Combiner Rule #4: CombineCXXOrder
// CHECK-NEXT:     /* 500 */   GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner3),
// CHECK-NEXT:     /* 503 */ // Label 8: @503
// CHECK-NEXT:     /* 503 */ GIM_Reject,
// CHECK-NEXT:     /* 504 */ // Label 3: @504
// CHECK-NEXT:     /* 504 */ GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(515), // Rule ID 1 //
// CHECK-NEXT:     /* 509 */   GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:     /* 512 */   // MIs[0] a
// CHECK-NEXT:     /* 512 */   // No operand predicates
// CHECK-NEXT:     /* 512 */   // MIs[0] b
// CHECK-NEXT:     /* 512 */   // No operand predicates
// CHECK-NEXT:     /* 512 */   // Combiner Rule #1: TwoMatchTwoApply
// CHECK-NEXT:     /* 512 */   GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner1),
// CHECK-NEXT:     /* 515 */ // Label 9: @515
// CHECK-NEXT:     /* 515 */ GIM_Reject,
// CHECK-NEXT:     /* 516 */ // Label 4: @516
// CHECK-NEXT:     /* 516 */ GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(527), // Rule ID 0 //
// CHECK-NEXT:     /* 521 */   GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:     /* 524 */   // MIs[0] a
// CHECK-NEXT:     /* 524 */   // No operand predicates
// CHECK-NEXT:     /* 524 */   // MIs[0] b
// CHECK-NEXT:     /* 524 */   // No operand predicates
// CHECK-NEXT:     /* 524 */   // Combiner Rule #0: OneMatchOneApply
// CHECK-NEXT:     /* 524 */   GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     /* 527 */ // Label 10: @527
// CHECK-NEXT:     /* 527 */ GIM_Reject,
// CHECK-NEXT:     /* 528 */ // Label 5: @528
// CHECK-NEXT:     /* 528 */ GIM_Reject,
// CHECK-NEXT:     /* 529 */ }; // Size: 529 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
