// RUN: llvm-tblgen -I %S/Inputs -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

include "test-intrinsics.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def IntrinTest0 : GICombineRule<
  (defs root:$a),
  (match (int_1in_1out $a, 0)),
  (apply (int_1in_1out $a, $x),
         (int_0in_1out i32:$x))>;

def SpecialIntrins : GICombineRule<
  (defs root:$a),
  (match (int_sideeffects_1in_1out $a, $b)),
  (apply (int_convergent_1in_1out i32:$x, $b),
         (int_convergent_sideeffects_1in_1out $a, $x))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  IntrinTest0,
  SpecialIntrins
]>;

//      CHECK: const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_SwitchOpcode, /*MI*/0, /*[*/GIMT_Encode2({{[0-9]+}}), GIMT_Encode2({{[0-9]+}}), /*)*//*default:*//*Label 2*/ GIMT_Encode4([[L132:[0-9]+]]),
// CHECK-NEXT:     /*TargetOpcode::G_INTRINSIC*//*Label 0*/ GIMT_Encode4(18),
// CHECK-NEXT:     /*TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS*//*Label 1*/ GIMT_Encode4([[L73:[0-9]+]]),
// CHECK-NEXT:     // Label 0: @18
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ GIMT_Encode4([[L72:[0-9]+]]), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:       GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::1in_1out),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] Operand 2
// CHECK-NEXT:       GIM_CheckConstantInt8, /*MI*/0, /*Op*/2, 0,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       // Combiner Rule #0: IntrinTest0
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_INTRINSIC),
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:       GIR_AddIntrinsicID, /*MI*/0, GIMT_Encode2(Intrinsic::0in_1out),
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_INTRINSIC),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddIntrinsicID, /*MI*/1, GIMT_Encode2(Intrinsic::1in_1out),
// CHECK-NEXT:       GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 3: @[[L72]]
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 1: @[[L73]]
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ GIMT_Encode4([[L131:[0-9]+]]), // Rule ID 1 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule1Enabled),
// CHECK-NEXT:       GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
// CHECK-NEXT:       GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, GIMT_Encode2(Intrinsic::sideeffects_1in_1out),
// CHECK-NEXT:       // MIs[0] a
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] b
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       // Combiner Rule #1: SpecialIntrins
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_INTRINSIC_CONVERGENT),
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:       GIR_AddIntrinsicID, /*MI*/0, GIMT_Encode2(Intrinsic::convergent_1in_1out),
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/2, // b
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/GIMT_Encode2(TargetOpcode::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_AddIntrinsicID, /*MI*/1, GIMT_Encode2(Intrinsic::convergent_sideeffects_1in_1out),
// CHECK-NEXT:       GIR_AddSimpleTempRegister, /*InsnID*/1, /*TempRegID*/0,
// CHECK-NEXT:       GIR_MergeMemOperands, /*InsnID*/1, /*NumInsns*/1, /*MergeInsnID's*/0,
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 4: @[[L131]]
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     // Label 2: @[[L132]]
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: 125 bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
