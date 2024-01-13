// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def ReplaceMatched : GICombineRule<
  (defs root:$dst),
  (match  (G_FNEG $tmp, $src),
          (G_FNEG $dst, $tmp)),
  (apply  (GIReplaceReg $dst, $src))>;

def ReplaceTemp : GICombineRule<
  (defs root:$a),
  (match    (G_BUILD_VECTOR $tmp, $x, $y),
            (G_UNMERGE_VALUES $a, $b, $tmp)),
  (apply  (G_UNMERGE_VALUES $a, i32:$new, $y),
          (GIReplaceReg $b, $new))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  ReplaceMatched,
  ReplaceTemp
]>;

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2(67), GIMT_Encode2(183), /*)*//*default:*//*Label 2*/ GIMT_Encode4(558),
// CHECK-NEXT:     /*TargetOpcode::G_UNMERGE_VALUES*//*Label 0*/ GIMT_Encode4(474), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0), GIMT_Encode4(0),
// CHECK-NEXT:     /*TargetOpcode::G_FNEG*//*Label 1*/ GIMT_Encode4(526),
// CHECK-NEXT:     // Label 0: @474
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4(525), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] tmp
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_BUILD_VECTOR),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/1, /*Expected*/3,
// CHECK-NEXT:       // MIs[1] x
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[1] y
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       // Combiner Rule #1: ReplaceTemp
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_UNMERGE_VALUES),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // y
// CHECK-NEXT:       GIR_ReplaceRegWithTempReg, /*OldInsnID*/0, /*OldOpIdx*/1, /*TempRegID*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 3: @525
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @526
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4(557), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] tmp
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FNEG),
// CHECK-NEXT:       // MIs[1] src
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckCanReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/1,
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       // Combiner Rule #0: ReplaceMatched
// CHECK-NEXT:       GIR_ReplaceReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*NewInsnId*/1, /*NewOpIdx*/1,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 4: @557
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @558
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 559 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
