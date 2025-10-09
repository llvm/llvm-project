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
// CHECK-NEXT:      /*   0 */ GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(99), GIMT_Encode2(211), /*)*//*default:*//*Label 5*/ GIMT_Encode4(524),
// CHECK-NEXT:      /* 10 */ /*TargetOpcode::G_STORE*//*Label 0*/ GIMT_Encode4(458), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 182 */ /*TargetOpcode::G_SEXT*//*Label 1*/ GIMT_Encode4(476), GIMT_Encode4(0),
// CHECK-NEXT:      /* 190 */ /*TargetOpcode::G_ZEXT*//*Label 2*/ GIMT_Encode4(488), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 418 */ /*TargetOpcode::G_FNEG*//*Label 3*/ GIMT_Encode4(500), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:      /* 454 */ /*TargetOpcode::G_FABS*//*Label 4*/ GIMT_Encode4(512),
// CHECK-NEXT:      /* 458 */ // Label 0: @458
// CHECK-NEXT:      /* 458 */ GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(475), // Rule ID 2 //
// CHECK-NEXT:      /* 463 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:      /* 466 */ // MIs[0] x
// CHECK-NEXT:      /* 466 */ // No operand predicates
// CHECK-NEXT:      /* 466 */ // MIs[0] y
// CHECK-NEXT:      /* 466 */ // No operand predicates
// CHECK-NEXT:      /* 466 */ GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:      /* 470 */ GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:      /* 474 */ // Combiner Rule #2: TwoMatchNoApply
// CHECK-NEXT:      /* 474 */ GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 475 */ // Label 6: @475
// CHECK-NEXT:      /* 475 */ GIM_Reject,
// CHECK-NEXT:      /* 476 */ // Label 1: @476
// CHECK-NEXT:      /* 476 */ GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(487), // Rule ID 3 //
// CHECK-NEXT:      /* 481 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:      /* 484 */ // MIs[0] a
// CHECK-NEXT:      /* 484 */ // No operand predicates
// CHECK-NEXT:      /* 484 */ // MIs[0] y
// CHECK-NEXT:      /* 484 */ // No operand predicates
// CHECK-NEXT:      /* 484 */ // Combiner Rule #3: NoMatchTwoApply
// CHECK-NEXT:      /* 484 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner2),
// CHECK-NEXT:      /* 487 */ // Label 7: @487
// CHECK-NEXT:      /* 487 */ GIM_Reject,
// CHECK-NEXT:      /* 488 */ // Label 2: @488
// CHECK-NEXT:      /* 488 */ GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(499), // Rule ID 4 //
// CHECK-NEXT:      /* 493 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// CHECK-NEXT:      /* 496 */ // MIs[0] a
// CHECK-NEXT:      /* 496 */ // No operand predicates
// CHECK-NEXT:      /* 496 */ // MIs[0] y
// CHECK-NEXT:      /* 496 */ // No operand predicates
// CHECK-NEXT:      /* 496 */ // Combiner Rule #4: CombineCXXOrder
// CHECK-NEXT:      /* 496 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner3),
// CHECK-NEXT:      /* 499 */ // Label 8: @499
// CHECK-NEXT:      /* 499 */ GIM_Reject,
// CHECK-NEXT:      /* 500 */ // Label 3: @500
// CHECK-NEXT:      /* 500 */ GIM_Try, /*On fail goto*//*Label 9*/ GIMT_Encode4(511), // Rule ID 1 //
// CHECK-NEXT:      /* 505 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /* 508 */ // MIs[0] a
// CHECK-NEXT:      /* 508 */ // No operand predicates
// CHECK-NEXT:      /* 508 */ // MIs[0] b
// CHECK-NEXT:      /* 508 */ // No operand predicates
// CHECK-NEXT:      /* 508 */ // Combiner Rule #1: TwoMatchTwoApply
// CHECK-NEXT:      /* 508 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner1),
// CHECK-NEXT:      /* 511 */ // Label 9: @511
// CHECK-NEXT:      /* 511 */ GIM_Reject,
// CHECK-NEXT:      /* 512 */ // Label 4: @512
// CHECK-NEXT:      /* 512 */ GIM_Try, /*On fail goto*//*Label 10*/ GIMT_Encode4(523), // Rule ID 0 //
// CHECK-NEXT:      /* 517 */ GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 520 */ // MIs[0] a
// CHECK-NEXT:      /* 520 */ // No operand predicates
// CHECK-NEXT:      /* 520 */ // MIs[0] b
// CHECK-NEXT:      /* 520 */ // No operand predicates
// CHECK-NEXT:      /* 520 */ // Combiner Rule #0: OneMatchOneApply
// CHECK-NEXT:      /* 520 */ GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:      /* 523 */ // Label 10: @523
// CHECK-NEXT:      /* 523 */ GIM_Reject,
// CHECK-NEXT:      /* 524 */ // Label 5: @524
// CHECK-NEXT:      /* 524 */ GIM_Reject,
// CHECK-NEXT:      /* 525 */ }; // Size: 525 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
