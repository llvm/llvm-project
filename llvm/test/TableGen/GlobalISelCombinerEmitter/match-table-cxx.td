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

def MyCombiner: GICombiner<"GenMyCombiner", [
  OneMatchOneApply,
  TwoMatchTwoApply,
  TwoMatchNoApply,
  NoMatchTwoApply
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
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown Apply Action");
// CHECK-NEXT: }

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(94), GIMT_Encode2(194), /*)*//*default:*//*Label 4*/ GIMT_Encode4(464),
// CHECK-NEXT:     /*TargetOpcode::G_STORE*//*Label 0*/ GIMT_Encode4(410), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_SEXT*//*Label 1*/ GIMT_Encode4(428), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_FNEG*//*Label 2*/ GIMT_Encode4(440), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_FABS*//*Label 3*/ GIMT_Encode4(452),
// CHECK-NEXT:     // Label 0: @410
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(427), // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:       // MIs[0] x
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] y
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner0),
// CHECK-NEXT:       GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GIMT_Encode2(GICXXPred_MI_Predicate_GICombiner1),
// CHECK-NEXT:       // Combiner Rule #2: TwoMatchNoApply
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 5: @427
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @428
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(439), // Rule ID 3 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] y
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #3: NoMatchTwoApply
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner2),
// CHECK-NEXT:     // Label 6: @439
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @440
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4(451), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #1: TwoMatchTwoApply
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner1),
// CHECK-NEXT:     // Label 7: @451
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 3: @452
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4(463), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #0: OneMatchOneApply
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     // Label 8: @463
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 4: @464
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 465 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
