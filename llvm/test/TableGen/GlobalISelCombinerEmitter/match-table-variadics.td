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

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
  InstTest1,
  InstTest2,
  InstTest3
]>;

// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/65, 69, /*)*//*default:*//*Label 2*/ 51,
// CHECK-NEXT:     /*TargetOpcode::G_UNMERGE_VALUES*//*Label 0*/ 9, 0, 0,
// CHECK-NEXT:     /*TargetOpcode::G_BUILD_VECTOR*//*Label 1*/ 30,
// CHECK-NEXT:     // Label 0: @9
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ 19, // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule2Enabled,
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #2: InstTest2
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 3: @19
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ 29, // Rule ID 3 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule3Enabled,
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
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 4: @29
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @30
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 5*/ 40, // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // Combiner Rule #1: InstTest1
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 5: @40
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 6*/ 50, // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
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
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 6: @50
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @51
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
