// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"


def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// These two combine rules both use GIM_CheckIsSameOperandIgnoreCopies.
// Check that the match table does not try to combine them by hoisting them in the GroupMatcher,
// which would cause a runtime crash as the "other inst" used by the predicate is
// only defined in the leaf RuleMatcher.
def SharedPreds0 : GICombineRule<
   (defs root:$root),
   (match (G_SUB $sub1, $B, $C),
          (G_ADD $add1, $A, $sub1),
          (G_SUB $root, $add1, $B)),
   (apply (G_SUB $root, $A, $C))>;

def SharedPreds1 : GICombineRule<
   (defs root:$root),
   (match (G_ADD $add1, $B, $C),
          (G_ADD $add2, $A, $add1),
          (G_SUB $root, $add2, $B)),
   (apply (G_ADD $root, $A, $C))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  SharedPreds0,
  SharedPreds1
]>;


// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:      /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(98),
// CHECK-NEXT:      /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*   9 */   GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(53), // Rule ID 1 //
// CHECK-NEXT:      /*  14 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /*  17 */     // MIs[0] root
// CHECK-NEXT:      /*  17 */     // No operand predicates
// CHECK-NEXT:      /*  17 */     // MIs[0] add2
// CHECK-NEXT:      /*  17 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /*  21 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  25 */     // MIs[1] A
// CHECK-NEXT:      /*  25 */     // No operand predicates
// CHECK-NEXT:      /*  25 */     // MIs[1] add1
// CHECK-NEXT:      /*  25 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /*  29 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  33 */     // MIs[2] B
// CHECK-NEXT:      /*  33 */     // No operand predicates
// CHECK-NEXT:      /*  33 */     // MIs[2] C
// CHECK-NEXT:      /*  33 */     // No operand predicates
// CHECK-NEXT:      /*  33 */     // MIs[0] B
// CHECK-NEXT:      /*  33 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK-NEXT:      /*  38 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /*  40 */     // Combiner Rule #1: SharedPreds1
// CHECK-NEXT:      /*  40 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  43 */     GIR_RootToRootCopy0, // root
// CHECK-NEXT:      /*  44 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  48 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /*  52 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  53 */   // Label 1: @53
// CHECK-NEXT:      /*  53 */   GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(97), // Rule ID 0 //
// CHECK-NEXT:      /*  58 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /*  61 */     // MIs[0] root
// CHECK-NEXT:      /*  61 */     // No operand predicates
// CHECK-NEXT:      /*  61 */     // MIs[0] add1
// CHECK-NEXT:      /*  61 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /*  65 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  69 */     // MIs[1] A
// CHECK-NEXT:      /*  69 */     // No operand predicates
// CHECK-NEXT:      /*  69 */     // MIs[1] sub1
// CHECK-NEXT:      /*  69 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /*  73 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*  77 */     // MIs[2] B
// CHECK-NEXT:      /*  77 */     // No operand predicates
// CHECK-NEXT:      /*  77 */     // MIs[2] C
// CHECK-NEXT:      /*  77 */     // No operand predicates
// CHECK-NEXT:      /*  77 */     // MIs[0] B
// CHECK-NEXT:      /*  77 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK-NEXT:      /*  82 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /*  84 */     // Combiner Rule #0: SharedPreds0
// CHECK-NEXT:      /*  84 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*  87 */     GIR_RootToRootCopy0, // root
// CHECK-NEXT:      /*  88 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  92 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /*  96 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  97 */   // Label 2: @97
// CHECK-NEXT:      /*  97 */   GIM_Reject,
// CHECK-NEXT:      /* 98 */ // Label 0: @98
// CHECK-NEXT:      /* 98 */ GIM_Reject,
// CHECK-NEXT:      /* 99 */ }; // Size: 99 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
