// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def HasAnswerToEverything : Predicate<"Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42">;

def WipOpcodeTest0 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_TRUNC):$d),
  (apply [{ APPLY }])>;

def WipOpcodeTest1 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_TRUNC, G_SEXT):$d),
  (apply [{ APPLY }])>;

// Note: also checks that spaces in the type name are removed.
def reg_matchinfo : GIDefMatchData<"Register  ">;
def InstTest0 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0, reg_matchinfo:$r1),
  (match (COPY $a, $b):$d),
  (apply [{ APPLY ${r0}, ${r1} }])>;

let Predicates = [HasAnswerToEverything] in
def InstTest1 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0),
  (match (COPY $a, $b):$d,
         (G_ZEXT $b, $c),
         [{ return CHECK ${a}, ${b}, ${c}, ${d} }]),
  (apply [{ APPLY }])>;

def InOutInstTest0 : GICombineRule<
  (defs root:$root),
  (match (G_ZEXT $tmp, $ext),
         (G_STORE $tmp, $ptr):$root),
  (apply (G_STORE $ext, $ptr):$root, "APPLY ${ext} ${ptr} ${root}")>;

// Imm operand of G_CONSTANT should match a literal int, while the second
// should match a constant.
def InOutInstTest1 : GICombineRule<
  (defs root:$dst),
  (match
      (G_CONSTANT $x, -42:$z),
      (G_AND $dst, $x, (i32 43))),
  (apply (G_TRUNC $dst, $z))>;

def MatchICst: GICombinePatFrag<
    (outs),
    (ins gi_mo:$foo, gi_imm:$cst),
    [(pattern "return matchIConstant(${foo}, ${cst});")]>;

def PatFragTest0 : GICombineRule<
  (defs root:$dst),
  (match (G_ZEXT $dst, $cst), (MatchICst $cst, (i32 0))),
  (apply (COPY $dst, (i32 0)))>;

// TODO: add test with temp reg use

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
  WipOpcodeTest0,
  WipOpcodeTest1,
  InstTest0,
  InstTest1,
  InOutInstTest0,
  InOutInstTest1,
  PatFragTest0
]>;

// We have at most 2 registers used by one rule at a time, so we should only have 2 registers MDInfos.

// CHECK:      struct MatchInfosTy {
// CHECK-NEXT:   Register MDInfo0, MDInfo1;
// CHECK-NEXT: };

// Check predicates
// CHECK:      switch (PredicateID) {
// CHECK-NEXT: case GICXXPred_MI_Predicate_GICombiner0: {
// CHECK-NEXT:   return CHECK State.MIs[0]->getOperand(0), State.MIs[0]->getOperand(1), State.MIs[1]->getOperand(1), State.MIs[0]
// CHECK-NEXT: }
// CHECK-NEXT: case GICXXPred_MI_Predicate_GICombiner1: {
// CHECK-NEXT:   return matchIConstant(State.MIs[0]->getOperand(1), 0);
// CHECK-NEXT: }

// Verify we reset MatchData on each tryCombineAll
// CHECK:      bool GenMyCombiner::tryCombineAll(MachineInstr &I) const {
// CHECK-NEXT:   const TargetSubtargetInfo &ST = MF.getSubtarget();
// CHECK-NEXT:   const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT:   NewMIVector OutMIs;
// CHECK-NEXT:   State.MIs.clear();
// CHECK-NEXT:   State.MIs.push_back(&I);
// CHECK-NEXT:   MatchInfos = MatchInfosTy();
// CHECK-EMPTY:
// CHECK-NEXT:   if (executeMatchTable(*this, OutMIs, State, ExecInfo, getMatchTable(), *ST.getInstrInfo(), MRI, *MRI.getTargetRegisterInfo(), *ST.getRegBankInfo(), AvailableFeatures, /*CoverageInfo*/ nullptr, &Observer))
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// Check apply
// CHECK:      enum {
// CHECK-NEXT:   GICXXCustomAction_CombineApplyGICombiner0 = GICXXCustomAction_Invalid + 1,
// CHECK-NEXT:   GICXXCustomAction_CombineApplyGICombiner1,
// CHECK-NEXT:   GICXXCustomAction_CombineApplyGICombiner2,
// CHECK-NEXT: };
// CHECK-NEXT: void GenMyCombiner::runCustomAction(unsigned ApplyID, const MatcherState &State, NewMIVector &OutMIs) const {
// CHECK-NEXT:   switch(ApplyID) {
// CHECK-NEXT:   case GICXXCustomAction_CombineApplyGICombiner0:{
// CHECK-NEXT:     APPLY
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_CombineApplyGICombiner1:{
// CHECK-NEXT:     APPLY MatchInfos.MDInfo0, MatchInfos.MDInfo1
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT:   case GICXXCustomAction_CombineApplyGICombiner2:{
// CHECK-NEXT:     APPLY State.MIs[1]->getOperand(1) State.MIs[0]->getOperand(1) OutMIs[0]
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT:   llvm_unreachable("Unknown Apply Action");
// CHECK-NEXT: }

// Verify match table.
// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 20,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_TRUNC,
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 1*/ 12, // Rule ID 0 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:         // Combiner Rule #0: WipOpcodeTest0; wip_match_opcode 'G_TRUNC'
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 1: @12
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 2*/ 19, // Rule ID 1 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:         // Combiner Rule #1: WipOpcodeTest1; wip_match_opcode 'G_TRUNC'
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 2: @19
// CHECK-NEXT:       GIM_Reject,
// CHECK-NEXT:     // Label 0: @20
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ 30, // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_SEXT,
// CHECK-NEXT:       // Combiner Rule #1: WipOpcodeTest1; wip_match_opcode 'G_SEXT'
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 3: @30
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ 64,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::COPY,
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 5*/ 42, // Rule ID 3 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule2Enabled,
// CHECK-NEXT:         // MIs[0] a
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] b
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // Combiner Rule #2: InstTest0
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner1,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 5: @42
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 6*/ 63, // Rule ID 4 //
// CHECK-NEXT:         GIM_CheckFeatures, GIFBS_HasAnswerToEverything,
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule3Enabled,
// CHECK-NEXT:         // MIs[0] a
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         // MIs[0] b
// CHECK-NEXT:         GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ZEXT,
// CHECK-NEXT:         // MIs[1] c
// CHECK-NEXT:         // No operand predicates
// CHECK-NEXT:         GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GICXXPred_MI_Predicate_GICombiner0,
// CHECK-NEXT:         GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:         // Combiner Rule #3: InstTest1
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 6: @63
// CHECK-NEXT:       GIM_Reject,
// CHECK-NEXT:     // Label 4: @64
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 7*/ 101, // Rule ID 5 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule4Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_STORE,
// CHECK-NEXT:       // MIs[0] tmp
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/0, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ZEXT,
// CHECK-NEXT:       // MIs[1] ext
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] ptr
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       // Combiner Rule #4: InOutInstTest0
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::G_STORE,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // ext
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // ptr
// CHECK-NEXT:       GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner2,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 7: @101
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 8*/ 143, // Rule ID 6 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule5Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_AND,
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] x
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       // MIs[1] z
// CHECK-NEXT:       GIM_CheckLiteralInt, /*MI*/1, /*Op*/1, -42,
// CHECK-NEXT:       GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 43,
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       // Combiner Rule #5: InOutInstTest1
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::G_TRUNC,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // z
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 8: @143
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 9*/ 181, // Rule ID 7 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule6Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_ZEXT,
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] cst
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckCxxInsnPredicate, /*MI*/0, /*FnId*/GICXXPred_MI_Predicate_GICombiner1,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/1, /*Type*/GILLT_s32, /*Imm*/0,
// CHECK-NEXT:       // Combiner Rule #6: PatFragTest0 @ [__PatFragTest0_match_1[0]]
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 9: @181
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
