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
// CHECK-NEXT:      /*   0 */ GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(280),
// CHECK-NEXT:      /*   5 */   GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /*   9 */   GIM_Try, /*On fail goto*//*Label 1*/ GIMT_Encode4(54), // Rule ID 2 //
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
// CHECK-NEXT:      /*  43 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /*  45 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  49 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /*  53 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  54 */   // Label 1: @54
// CHECK-NEXT:      /*  54 */   GIM_Try, /*On fail goto*//*Label 2*/ GIMT_Encode4(99), // Rule ID 3 //
// CHECK-NEXT:      /*  59 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /*  62 */     // MIs[0] root
// CHECK-NEXT:      /*  62 */     // No operand predicates
// CHECK-NEXT:      /*  62 */     // MIs[0] add2
// CHECK-NEXT:      /*  62 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /*  66 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  70 */     // MIs[1] A
// CHECK-NEXT:      /*  70 */     // No operand predicates
// CHECK-NEXT:      /*  70 */     // MIs[1] add1
// CHECK-NEXT:      /*  70 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /*  74 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  78 */     // MIs[2] B
// CHECK-NEXT:      /*  78 */     // No operand predicates
// CHECK-NEXT:      /*  78 */     // MIs[2] C
// CHECK-NEXT:      /*  78 */     // No operand predicates
// CHECK-NEXT:      /*  78 */     // MIs[0] B
// CHECK-NEXT:      /*  78 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/2,
// CHECK-NEXT:      /*  83 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /*  85 */     // Combiner Rule #1: SharedPreds1 @ [__SharedPreds1_match_0[commuted]]
// CHECK-NEXT:      /*  85 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /*  88 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /*  90 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /*  94 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // C
// CHECK-NEXT:      /*  98 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /*  99 */   // Label 2: @99
// CHECK-NEXT:      /*  99 */   GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(144), // Rule ID 4 //
// CHECK-NEXT:      /* 104 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /* 107 */     // MIs[0] root
// CHECK-NEXT:      /* 107 */     // No operand predicates
// CHECK-NEXT:      /* 107 */     // MIs[0] add2
// CHECK-NEXT:      /* 107 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 111 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 115 */     // MIs[1] A
// CHECK-NEXT:      /* 115 */     // No operand predicates
// CHECK-NEXT:      /* 115 */     // MIs[1] add1
// CHECK-NEXT:      /* 115 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:      /* 119 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 123 */     // MIs[2] B
// CHECK-NEXT:      /* 123 */     // No operand predicates
// CHECK-NEXT:      /* 123 */     // MIs[2] C
// CHECK-NEXT:      /* 123 */     // No operand predicates
// CHECK-NEXT:      /* 123 */     // MIs[0] B
// CHECK-NEXT:      /* 123 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK-NEXT:      /* 128 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 130 */     // Combiner Rule #1: SharedPreds1 @ [__SharedPreds1_match_1[commuted]]
// CHECK-NEXT:      /* 130 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 133 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 135 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 139 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /* 143 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 144 */   // Label 3: @144
// CHECK-NEXT:      /* 144 */   GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(189), // Rule ID 5 //
// CHECK-NEXT:      /* 149 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:      /* 152 */     // MIs[0] root
// CHECK-NEXT:      /* 152 */     // No operand predicates
// CHECK-NEXT:      /* 152 */     // MIs[0] add2
// CHECK-NEXT:      /* 152 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 156 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 160 */     // MIs[1] A
// CHECK-NEXT:      /* 160 */     // No operand predicates
// CHECK-NEXT:      /* 160 */     // MIs[1] add1
// CHECK-NEXT:      /* 160 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:      /* 164 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 168 */     // MIs[2] B
// CHECK-NEXT:      /* 168 */     // No operand predicates
// CHECK-NEXT:      /* 168 */     // MIs[2] C
// CHECK-NEXT:      /* 168 */     // No operand predicates
// CHECK-NEXT:      /* 168 */     // MIs[0] B
// CHECK-NEXT:      /* 168 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/2,
// CHECK-NEXT:      /* 173 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 175 */     // Combiner Rule #1: SharedPreds1 @ [__SharedPreds1_match_0[commuted], __SharedPreds1_match_1[commuted]]
// CHECK-NEXT:      /* 175 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 178 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 180 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 184 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // C
// CHECK-NEXT:      /* 188 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 189 */   // Label 4: @189
// CHECK-NEXT:      /* 189 */   GIM_Try, /*On fail goto*//*Label 5*/ GIMT_Encode4(234), // Rule ID 0 //
// CHECK-NEXT:      /* 194 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 197 */     // MIs[0] root
// CHECK-NEXT:      /* 197 */     // No operand predicates
// CHECK-NEXT:      /* 197 */     // MIs[0] add1
// CHECK-NEXT:      /* 197 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 201 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 205 */     // MIs[1] A
// CHECK-NEXT:      /* 205 */     // No operand predicates
// CHECK-NEXT:      /* 205 */     // MIs[1] sub1
// CHECK-NEXT:      /* 205 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/2, // MIs[2]
// CHECK-NEXT:      /* 209 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 213 */     // MIs[2] B
// CHECK-NEXT:      /* 213 */     // No operand predicates
// CHECK-NEXT:      /* 213 */     // MIs[2] C
// CHECK-NEXT:      /* 213 */     // No operand predicates
// CHECK-NEXT:      /* 213 */     // MIs[0] B
// CHECK-NEXT:      /* 213 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK-NEXT:      /* 218 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 220 */     // Combiner Rule #0: SharedPreds0
// CHECK-NEXT:      /* 220 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 223 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 225 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // A
// CHECK-NEXT:      /* 229 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /* 233 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 234 */   // Label 5: @234
// CHECK-NEXT:      /* 234 */   GIM_Try, /*On fail goto*//*Label 6*/ GIMT_Encode4(279), // Rule ID 1 //
// CHECK-NEXT:      /* 239 */     GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:      /* 242 */     // MIs[0] root
// CHECK-NEXT:      /* 242 */     // No operand predicates
// CHECK-NEXT:      /* 242 */     // MIs[0] add1
// CHECK-NEXT:      /* 242 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:      /* 246 */     GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:      /* 250 */     // MIs[1] A
// CHECK-NEXT:      /* 250 */     // No operand predicates
// CHECK-NEXT:      /* 250 */     // MIs[1] sub1
// CHECK-NEXT:      /* 250 */     GIM_RecordInsnIgnoreCopies, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
// CHECK-NEXT:      /* 254 */     GIM_CheckOpcode, /*MI*/2, GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 258 */     // MIs[2] B
// CHECK-NEXT:      /* 258 */     // No operand predicates
// CHECK-NEXT:      /* 258 */     // MIs[2] C
// CHECK-NEXT:      /* 258 */     // No operand predicates
// CHECK-NEXT:      /* 258 */     // MIs[0] B
// CHECK-NEXT:      /* 258 */     GIM_CheckIsSameOperandIgnoreCopies, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
// CHECK-NEXT:      /* 263 */     GIM_CheckIsSafeToFold, /*NumInsns*/2,
// CHECK-NEXT:      /* 265 */     // Combiner Rule #0: SharedPreds0 @ [__SharedPreds0_match_1[commuted]]
// CHECK-NEXT:      /* 265 */     GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_SUB),
// CHECK-NEXT:      /* 268 */     GIR_RootToRootCopy, /*OpIdx*/0, // root
// CHECK-NEXT:      /* 270 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // A
// CHECK-NEXT:      /* 274 */     GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/2, // C
// CHECK-NEXT:      /* 278 */     GIR_EraseRootFromParent_Done,
// CHECK-NEXT:      /* 279 */   // Label 6: @279
// CHECK-NEXT:      /* 279 */   GIM_Reject,
// CHECK-NEXT:      /* 280 */ // Label 0: @280
// CHECK-NEXT:      /* 280 */ GIM_Reject,
// CHECK-NEXT:      /* 281 */ }; // Size: 281 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
