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
// CHECK-NEXT:      /*   0 */ GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(99), GIMT_Encode2(210), /*)*//*default:*//*Label 5*/ GIMT_Encode4(520),
// CHECK-NEXT:      /* 10 */ /*TargetOpcode::G_STORE*//*Label 0*/ GIMT_Encode4(454), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 182 */ /*TargetOpcode::G_SEXT*//*Label 1*/ GIMT_Encode4(472), GIMT_Encode4(0),
// CHECK-NEXT:      /* 190 */ /*TargetOpcode::G_ZEXT*//*Label 2*/ GIMT_Encode4(484), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 414 */ /*TargetOpcode::G_FNEG*//*Label 3*/ GIMT_Encode4(496), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 450 */ /*TargetOpcode::G_FABS*//*Label 4*/ GIMT_Encode4(508),
// CHECK-NEXT:      /* 454 */ // Label 0: @454
// CHECK-NEXT:      /* 454 */ GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(471), // Rule ID 2 //
// CHECK-NEXT:      /* 459 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:      /* 462 */ // MIs[0] x
// CHECK-NEXT:      /* 462 */ // No operand predicates
// CHECK-NEXT:      /* 462 */ // MIs[0] y
// CHECK-NEXT:      /* 462 */ // No operand predicates
// CHECK-NEXT:      /* 462 */ GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:      /* 466 */ GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:      /* 470 */ // Combiner Rule #2: TwoMatchNoApply
// CHECK-NEXT:      /* 470 */ GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 471 */ // Label 6: @471
// CHECK-NEXT:      /* 471 */ GIM_Reject,
// CHECK-NEXT:      /* 472 */ // Label 1: @472
// CHECK-NEXT:      /* 472 */ GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(483), // Rule ID 3 //
// CHECK-NEXT:      /* 477 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:      /* 480 */ // MIs[0] a
// CHECK-NEXT:      /* 480 */ // No operand predicates
// CHECK-NEXT:      /* 480 */ // MIs[0] y
// CHECK-NEXT:      /* 480 */ // No operand predicates
// CHECK-NEXT:      /* 480 */ // Combiner Rule #3: NoMatchTwoApply
// CHECK-NEXT:      /* 480 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner2),
// CHECK-NEXT:      /* 483 */ // Label 7: @483
// CHECK-NEXT:      /* 483 */ GIM_Reject,
// CHECK-NEXT:      /* 484 */ // Label 2: @484
// CHECK-NEXT:      /* 484 */ GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(495), // Rule ID 4 //
// CHECK-NEXT:      /* 489 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// CHECK-NEXT:      /* 492 */ // MIs[0] a
// CHECK-NEXT:      /* 492 */ // No operand predicates
// CHECK-NEXT:      /* 492 */ // MIs[0] y
// CHECK-NEXT:      /* 492 */ // No operand predicates
// CHECK-NEXT:      /* 492 */ // Combiner Rule #4: CombineCXXOrder
// CHECK-NEXT:      /* 492 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner3),
// CHECK-NEXT:      /* 495 */ // Label 8: @495
// CHECK-NEXT:      /* 495 */ GIM_Reject,
// CHECK-NEXT:      /* 496 */ // Label 3: @496
// CHECK-NEXT:      /* 496 */ GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(507), // Rule ID 1 //
// CHECK-NEXT:      /* 501 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /* 504 */ // MIs[0] a
// CHECK-NEXT:      /* 504 */ // No operand predicates
// CHECK-NEXT:      /* 504 */ // MIs[0] b
// CHECK-NEXT:      /* 504 */ // No operand predicates
// CHECK-NEXT:      /* 504 */ // Combiner Rule #1: TwoMatchTwoApply
// CHECK-NEXT:      /* 504 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner1),
// CHECK-NEXT:      /* 507 */ // Label 9: @507
// CHECK-NEXT:      /* 507 */ GIM_Reject,
// CHECK-NEXT:      /* 508 */ // Label 4: @508
// CHECK-NEXT:      /* 508 */ GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(519), // Rule ID 0 //
// CHECK-NEXT:      /* 513 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 516 */ // MIs[0] a
// CHECK-NEXT:      /* 516 */ // No operand predicates
// CHECK-NEXT:      /* 516 */ // MIs[0] b
// CHECK-NEXT:      /* 516 */ // No operand predicates
// CHECK-NEXT:      /* 516 */ // Combiner Rule #0: OneMatchOneApply
// CHECK-NEXT:      /* 516 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:      /* 519 */ // Label 10: @519
// CHECK-NEXT:      /* 519 */ GIM_Reject,
// CHECK-NEXT:      /* 520 */ // Label 5: @520
// CHECK-NEXT:      /* 520 */ GIM_Reject,
// CHECK-NEXT:      /* 521 */ }; // Size: 521 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
