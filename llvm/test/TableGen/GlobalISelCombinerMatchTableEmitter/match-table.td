// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner-matchtable \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def dummy;

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}
def MOV : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def TRUNC : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def ZEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;
def SEXT : I<(outs GPR32:$dst), (ins GPR32:$src1), []>;

def HasAnswerToEverything : Predicate<"Subtarget->getAnswerToUniverse() == 42 && Subtarget->getAnswerToLife() == 42">;

def WipOpcodeTest0 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode TRUNC):$d),
  (apply [{ APPLY }])>;

def WipOpcodeTest1 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode TRUNC, SEXT):$d),
  (apply [{ APPLY }])>;

// Note: also checks that spaces in the type name are removed.
def reg_matchinfo : GIDefMatchData<"Register  ">;
def InstTest0 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0, reg_matchinfo:$r1),
  (match (MOV $a, $b):$d),
  (apply [{ APPLY ${r0}, ${r1} }])>;

let Predicates = [HasAnswerToEverything] in
def InstTest1 : GICombineRule<
  (defs root:$d, reg_matchinfo:$r0),
  (match (MOV $a, $b):$d,
         (ZEXT $b, $c),
         [{ return CHECK ${a}, ${b}, ${c}, ${d} }]),
  (apply [{ APPLY }])>;

def MyCombiner: GICombinerHelper<"GenMyCombiner", [
  WipOpcodeTest0,
  WipOpcodeTest1,
  InstTest0,
  InstTest1
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

// Verify we reset MatchData on each tryCombineAll
// CHECK:      bool GenMyCombiner::tryCombineAll(MachineInstr &I) const {
// CHECK-NEXT:   const TargetSubtargetInfo &ST = MF.getSubtarget();
// CHECK-NEXT:   const PredicateBitset AvailableFeatures = getAvailableFeatures();
// CHECK-NEXT:   NewMIVector OutMIs;
// CHECK-NEXT:   State.MIs.clear();
// CHECK-NEXT:   State.MIs.push_back(&I);
// CHECK-NEXT:   MatchInfos = MatchInfosTy();
// CHECK-EMPTY:
// CHECK-NEXT:   if (executeMatchTable(*this, OutMIs, State, ExecInfo, getMatchTable(), *ST.getInstrInfo(), MRI, *MRI.getTargetRegisterInfo(), *ST.getRegBankInfo(), AvailableFeatures, /*CoverageInfo*/ nullptr))
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   return false;
// CHECK-NEXT: }


// Verify match table.
// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 20,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, MyTarget::TRUNC,
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 1*/ 12, // Rule ID 0 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:         // Combiner Rule #0: WipOpcodeTest0; wip_match_opcode alternative 'TRUNC'
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 1: @12
// CHECK-NEXT:       GIM_Try, /*On fail goto*//*Label 2*/ 19, // Rule ID 1 //
// CHECK-NEXT:         GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:         // Combiner Rule #1: WipOpcodeTest1; wip_match_opcode alternative 'TRUNC'
// CHECK-NEXT:         GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:         GIR_Done,
// CHECK-NEXT:       // Label 2: @19
// CHECK-NEXT:       GIM_Reject,
// CHECK-NEXT:     // Label 0: @20
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 3*/ 30, // Rule ID 2 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule1Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, MyTarget::SEXT,
// CHECK-NEXT:       // Combiner Rule #1: WipOpcodeTest1; wip_match_opcode alternative 'SEXT'
// CHECK-NEXT:       GIR_CustomAction, GICXXCustomAction_CombineApplyGICombiner0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 3: @30
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 4*/ 64,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, MyTarget::MOV,
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
// CHECK-NEXT:         GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:         GIM_CheckOpcode, /*MI*/1, MyTarget::ZEXT,
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
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
