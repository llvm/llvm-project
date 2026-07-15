// RUN: llvm-tblgen -gen-macro-fusion-pred -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-PREDICATOR
// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-SUBTARGET

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo { }
def TestAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def Test : Target {
  let InstructionSet = TestInstrInfo;
  let AssemblyWriters = [TestAsmWriter];
}

let Namespace = "Test" in {
  foreach i = 0-32 in {
    def X#i : Register<"x"#i>;
  }

  def GPR : RegisterClass<"GPR", [i32], 32, (sequence  "X%u", 0, 32)>;

  class TestInst<int Opc> : Instruction {
    field bits<32> Inst;
    field bits<32> SoftFail = 0;
    let Size = 4;
    let Inst = Opc;
    let OutOperandList = (outs);
    let InOperandList = (ins);
    let AsmString = NAME;
  }
}

def Inst0 : TestInst<0>;
def Inst1 : TestInst<1>;
let isCommutable = true in
def Inst2 : TestInst<2>;

def BothFusionPredicate: BothFusionPredicateWithMCInstPredicate<CheckRegOperand<0, X0>>;
def TestBothFusionPredicate: Fusion<"test-both-fusion-predicate", "HasBothFusionPredicate",
                                    "Test BothFusionPredicate",
                                    [BothFusionPredicate]>;

def TestFusion: SimpleFusion<"test", "HasTestFusion", "Test Fusion",
                             CheckOpcode<[Inst0, Inst1]>,
                             CheckAll<[
                               CheckOpcode<[Inst1]>,
                               CheckRegOperand<0, X0>
                             ]>>;

let IsCommutable = 1 in
def TestCommutableFusion: SimpleFusion<"test-commutable", "HasTestCommutableFusion",
                                       "Test Commutable Fusion",
                                       CheckOpcode<[Inst0]>,
                                       CheckAll<[
                                         CheckOpcode<[Inst1]>,
                                         CheckRegOperand<0, X0>
                                       ]>>;

def TestSingleFusion: SingleFusion<"test-single", "HasTestSingleFusion",
                                   "Test SingleFusion",
                                   Inst0, Inst2,
                                   secondInstPred=CheckRegOperand<0, X0>>;

def TestFirstSameRegFusion: Fusion<"test-first-same-reg", "HasTestFirstSameRegFusion",
                                   "Test FirstSameReg",
                                   [FirstInstHasSameReg<0, 1>]> {
  bit IsCommutable = 1;
}

// CHECK-PREDICATOR:       #ifdef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  bool isTestBothFusionPredicate(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestCommutableFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestFirstSameRegFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestSingleFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  } // namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif // GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #ifdef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestBothFusionPredicatePreRA, "Times TestBothFusionPredicate Triggered (pre-ra)");
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestBothFusionPredicatePostRA, "Times TestBothFusionPredicate Triggered (post-ra)");
// CHECK-PREDICATOR-NEXT:  bool isTestBothFusionPredicate(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (MI->getOperand(0).getReg() != Test::X0)
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (MI->getOperand(0).getReg() != Test::X0)
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (SecondMI.getMF()->getProperties().hasNoVRegs())
// CHECK-PREDICATOR-NEXT:      ++NumTestBothFusionPredicatePostRA;
// CHECK-PREDICATOR-NEXT:    else
// CHECK-PREDICATOR-NEXT:      ++NumTestBothFusionPredicatePreRA;
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestCommutableFusionPreRA, "Times TestCommutableFusion Triggered (pre-ra)");
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestCommutableFusionPostRA, "Times TestCommutableFusion Triggered (post-ra)");
// CHECK-PREDICATOR-NEXT:  bool isTestCommutableFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst1 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (( MI->getOpcode() != Test::Inst0 ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!SecondMI.getOperand(0).getReg().isVirtual()) {
// CHECK-PREDICATOR-NEXT:      if (SecondMI.getOperand(0).getReg() != SecondMI.getOperand(1).getReg()) {
// CHECK-PREDICATOR-NEXT:        if (!SecondMI.getDesc().isCommutable())
// CHECK-PREDICATOR-NEXT:          return false;
// CHECK-PREDICATOR-NEXT:        unsigned SrcOpIdx1 = 1, SrcOpIdx2 = TargetInstrInfo::CommuteAnyOperandIndex;
// CHECK-PREDICATOR-NEXT:        if (TII.findCommutedOpIndices(SecondMI, SrcOpIdx1, SrcOpIdx2))
// CHECK-PREDICATOR-NEXT:          if (SecondMI.getOperand(0).getReg() != SecondMI.getOperand(SrcOpIdx2).getReg())
// CHECK-PREDICATOR-NEXT:            return false;
// CHECK-PREDICATOR-NEXT:      }
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      Register FirstDest = FirstMI->getOperand(0).getReg();
// CHECK-PREDICATOR-NEXT:      if (FirstDest.isVirtual() && !MRI.hasOneNonDBGUse(FirstDest))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!(FirstMI->getOperand(0).isReg() &&
// CHECK-PREDICATOR-NEXT:          SecondMI.getOperand(1).isReg() &&
// CHECK-PREDICATOR-NEXT:          FirstMI->getOperand(0).getReg() == SecondMI.getOperand(1).getReg())) {
// CHECK-PREDICATOR-NEXT:      if (!SecondMI.getDesc().isCommutable())
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:      unsigned SrcOpIdx1 = 1, SrcOpIdx2 = TargetInstrInfo::CommuteAnyOperandIndex;
// CHECK-PREDICATOR-NEXT:      if (TII.findCommutedOpIndices(SecondMI, SrcOpIdx1, SrcOpIdx2))
// CHECK-PREDICATOR-NEXT:        if (FirstMI->getOperand(0).getReg() != SecondMI.getOperand(SrcOpIdx2).getReg())
// CHECK-PREDICATOR-NEXT:          return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (SecondMI.getMF()->getProperties().hasNoVRegs())
// CHECK-PREDICATOR-NEXT:      ++NumTestCommutableFusionPostRA;
// CHECK-PREDICATOR-NEXT:    else
// CHECK-PREDICATOR-NEXT:      ++NumTestCommutableFusionPreRA;
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestFirstSameRegFusionPreRA, "Times TestFirstSameRegFusion Triggered (pre-ra)");
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestFirstSameRegFusionPostRA, "Times TestFirstSameRegFusion Triggered (post-ra)");
// CHECK-PREDICATOR-NEXT:  bool isTestFirstSameRegFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    if (!FirstMI->getOperand(0).getReg().isVirtual()) {
// CHECK-PREDICATOR-NEXT:      if (FirstMI->getOperand(0).getReg() != FirstMI->getOperand(1).getReg()) {
// CHECK-PREDICATOR-NEXT:        if (!FirstMI->getDesc().isCommutable())
// CHECK-PREDICATOR-NEXT:          return false;
// CHECK-PREDICATOR-NEXT:        unsigned SrcOpIdx1 = 1, SrcOpIdx2 = TargetInstrInfo::CommuteAnyOperandIndex;
// CHECK-PREDICATOR-NEXT:        if (TII.findCommutedOpIndices(FirstMI, SrcOpIdx1, SrcOpIdx2))
// CHECK-PREDICATOR-NEXT:          if (FirstMI->getOperand(0).getReg() != FirstMI->getOperand(SrcOpIdx2).getReg())
// CHECK-PREDICATOR-NEXT:            return false;
// CHECK-PREDICATOR-NEXT:      }
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (SecondMI.getMF()->getProperties().hasNoVRegs())
// CHECK-PREDICATOR-NEXT:      ++NumTestFirstSameRegFusionPostRA;
// CHECK-PREDICATOR-NEXT:    else
// CHECK-PREDICATOR-NEXT:      ++NumTestFirstSameRegFusionPreRA;
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestFusionPreRA, "Times TestFusion Triggered (pre-ra)");
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestFusionPostRA, "Times TestFusion Triggered (post-ra)");
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst1 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (!llvm::is_contained({Test::Inst0, Test::Inst1}, MI->getOpcode()))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!SecondMI.getOperand(0).getReg().isVirtual()) {
// CHECK-PREDICATOR-NEXT:      if (SecondMI.getOperand(0).getReg() != SecondMI.getOperand(1).getReg())
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      Register FirstDest = FirstMI->getOperand(0).getReg();
// CHECK-PREDICATOR-NEXT:      if (FirstDest.isVirtual() && !MRI.hasOneNonDBGUse(FirstDest))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!(FirstMI->getOperand(0).isReg() &&
// CHECK-PREDICATOR-NEXT:          SecondMI.getOperand(1).isReg() &&
// CHECK-PREDICATOR-NEXT:          FirstMI->getOperand(0).getReg() == SecondMI.getOperand(1).getReg()))
// CHECK-PREDICATOR-NEXT:      return false;
// CHECK-PREDICATOR-NEXT:    if (SecondMI.getMF()->getProperties().hasNoVRegs())
// CHECK-PREDICATOR-NEXT:      ++NumTestFusionPostRA;
// CHECK-PREDICATOR-NEXT:    else
// CHECK-PREDICATOR-NEXT:      ++NumTestFusionPreRA;
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestSingleFusionPreRA, "Times TestSingleFusion Triggered (pre-ra)");
// CHECK-PREDICATOR-NEXT:  STATISTIC(NumTestSingleFusionPostRA, "Times TestSingleFusion Triggered (post-ra)");
// CHECK-PREDICATOR-NEXT:  bool isTestSingleFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst2 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst0 )
// CHECK-PREDICATOR-NEXT:          && true
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!SecondMI.getOperand(0).getReg().isVirtual()) {
// CHECK-PREDICATOR-NEXT:      if (SecondMI.getOperand(0).getReg() != SecondMI.getOperand(1).getReg()) {
// CHECK-PREDICATOR-NEXT:        if (!SecondMI.getDesc().isCommutable())
// CHECK-PREDICATOR-NEXT:          return false;
// CHECK-PREDICATOR-NEXT:        unsigned SrcOpIdx1 = 1, SrcOpIdx2 = TargetInstrInfo::CommuteAnyOperandIndex;
// CHECK-PREDICATOR-NEXT:        if (TII.findCommutedOpIndices(SecondMI, SrcOpIdx1, SrcOpIdx2))
// CHECK-PREDICATOR-NEXT:          if (SecondMI.getOperand(0).getReg() != SecondMI.getOperand(SrcOpIdx2).getReg())
// CHECK-PREDICATOR-NEXT:            return false;
// CHECK-PREDICATOR-NEXT:      }
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      Register FirstDest = FirstMI->getOperand(0).getReg();
// CHECK-PREDICATOR-NEXT:      if (FirstDest.isVirtual() && !MRI.hasOneNonDBGUse(FirstDest))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!(FirstMI->getOperand(0).isReg() &&
// CHECK-PREDICATOR-NEXT:          SecondMI.getOperand(1).isReg() &&
// CHECK-PREDICATOR-NEXT:          FirstMI->getOperand(0).getReg() == SecondMI.getOperand(1).getReg())) {
// CHECK-PREDICATOR-NEXT:      if (!SecondMI.getDesc().isCommutable())
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:      unsigned SrcOpIdx1 = 1, SrcOpIdx2 = TargetInstrInfo::CommuteAnyOperandIndex;
// CHECK-PREDICATOR-NEXT:      if (TII.findCommutedOpIndices(SecondMI, SrcOpIdx1, SrcOpIdx2))
// CHECK-PREDICATOR-NEXT:        if (FirstMI->getOperand(0).getReg() != SecondMI.getOperand(SrcOpIdx2).getReg())
// CHECK-PREDICATOR-NEXT:          return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (SecondMI.getMF()->getProperties().hasNoVRegs())
// CHECK-PREDICATOR-NEXT:      ++NumTestSingleFusionPostRA;
// CHECK-PREDICATOR-NEXT:    else
// CHECK-PREDICATOR-NEXT:      ++NumTestSingleFusionPreRA;
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  } // namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif // GET_Test_MACRO_FUSION_PRED_IMPL

// Check that we have generated `getMacroFusions()` function.
// CHECK-SUBTARGET:      std::vector<MacroFusionPredTy> getMacroFusions() const final;

// CHECK-SUBTARGET:      std::vector<MacroFusionPredTy> TestGenSubtargetInfo::getMacroFusions() const {
// CHECK-SUBTARGET-NEXT:   std::vector<MacroFusionPredTy> Fusions;
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestBothFusionPredicate)) Fusions.push_back(llvm::isTestBothFusionPredicate);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestCommutableFusion)) Fusions.push_back(llvm::isTestCommutableFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestFirstSameRegFusion)) Fusions.push_back(llvm::isTestFirstSameRegFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestFusion)) Fusions.push_back(llvm::isTestFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestSingleFusion)) Fusions.push_back(llvm::isTestSingleFusion);
// CHECK-SUBTARGET-NEXT:   return Fusions;
// CHECK-SUBTARGET-NEXT: }
