// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def InstTest0 : GICombineRule<
  (defs root:$a),
  (match (G_BUILD_VECTOR $a, $b, $c, $d)),
  (apply [{ APPLY }])>;

def InstTest1 : GICombineRule<
  (defs root:$a),
  (match (G_BUILD_VECTOR $a, $b)),
  (apply [{ APPLY }])>;

def InstTest2 : GICombineRule<
  (defs root:$a),
  (match (G_UNMERGE_VALUES $a, $b)),
  (apply [{ APPLY }])>;

def InstTest3 : GICombineRule<
  (defs root:$a),
  (match (G_UNMERGE_VALUES $a, $b, $c, $d)),
  (apply [{ APPLY }])>;

def VariadicTypeTestCxx : GICombineRule<
  (defs root:$a),
  (match (G_BUILD_VECTOR $a, GIVariadic<2, 4>:$b)),
  (apply [{ ${b} }])>;

def VariadicTypeTestReuse : GICombineRule<
  (defs root:$a),
  (match (G_BUILD_VECTOR $a, $c, GIVariadic<2, 4>:$b)),
  (apply (G_MERGE_VALUES $a, $b, $c))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  InstTest1,
  InstTest2,
  InstTest3,
  VariadicTypeTestCxx,
  VariadicTypeTestReuse
]>;

// CHECK:     bool GenMyCombiner::runCustomAction(unsigned ApplyID, const MatcherState &State, NewMIVector &OutMIs) const {
// CHECK-NEXT:   Helper.getBuilder().setInstrAndDebugLoc(*State.MIs[0]);
// CHECK-NEXT:   switch(ApplyID) {
// CHECK-NEXT:   case GICXXCustomAction_GICombiner0:{
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     APPLY
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_GICombiner1:{
// CHECK-NEXT:     // Apply Patterns
// CHECK-NEXT:     getRemainingOperands(*State.MIs[0], 1)
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable("Unknown Apply Action");
// CHECK-NEXT: }

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2({{[0-9]+}}), GIMT_Encode2({{[0-9]+}}), /*)*//*default:*//*Label 2*/ GIMT_Encode4({{[0-9]+}}),
// CHECK-NEXT:     /*TargetOpcode::G_UNMERGE_VALUES*//*Label 0*/ GIMT_Encode4({{[0-9]+}}), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_BUILD_VECTOR*//*Label 1*/ GIMT_Encode4({{[0-9]+}}),
// CHECK-NEXT:     // Label 0: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule2Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #2: InstTest2
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     // Label 3: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 3 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule3Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] c
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] d
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #3: InstTest3
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     // Label 4: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #1: InstTest1
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     // Label 5: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 4 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule4Enabled),
// CHECK-NEXT:       GIM_CheckNumOperandsGE, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:       GIM_CheckNumOperandsLE, /*MI*/0, /*Expected*/5,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #4: VariadicTypeTestCxx
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner1),
// CHECK-NEXT:     // Label 6: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 7*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 5 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule5Enabled),
// CHECK-NEXT:       GIM_CheckNumOperandsGE, /*MI*/0, /*Expected*/4,
// CHECK-NEXT:       GIM_CheckNumOperandsLE, /*MI*/0, /*Expected*/6,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] c
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #5: VariadicTypeTestReuse
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_MERGE_VALUES),
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_CopyRemaining, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // b
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/1, // c
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 7: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 8*/ GIMT_Encode4({{[0-9]+}}), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] c
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] d
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #0: InstTest0
// CHECK-NEXT:       GIR_DoneWithCustomAction, /*Fn*/GIMT_Encode2(GICXXCustomAction_GICombiner0),
// CHECK-NEXT:     // Label 8: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @{{[0-9]+}}
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 128 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
