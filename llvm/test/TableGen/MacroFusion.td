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

def TestFusion: SimpleFusion<"test-fusion", "HasTestFusion", "Test Fusion",
                             CheckOpcode<[Inst0]>,
                             CheckAll<[
                               CheckOpcode<[Inst1]>,
                               CheckRegOperand<0, X0>
                             ]>>;

let IsCommutable = 1 in
def TestCommutableFusion: SimpleFusion<"test-commutable-fusion", "HasTestCommutableFusion",
                                       "Test Commutable Fusion",
                                       CheckOpcode<[Inst0]>,
                                       CheckAll<[
                                         CheckOpcode<[Inst1]>,
                                         CheckRegOperand<0, X0>
                                       ]>>;

def TestSingleFusion: SingleFusion<"test-single-fusion", "HasTestSingleFusion",
                                   "Test SingleFusion",
                                   Inst0, Inst2,
                                   secondInstPred=CheckRegOperand<0, X0>>;

// CHECK-PREDICATOR:       #ifdef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-NEXT:  bool isTestBothFusionPredicate(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestCommutableFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);   
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  bool isTestSingleFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  } // end namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif

// CHECK-PREDICATOR:       #ifdef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-NEXT:  bool isTestBothFusionPredicate(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (MI->getOperand(0).getReg() != Test::X0)
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (MI->getOperand(0).getReg() != Test::X0)
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  bool isTestCommutableFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst1 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = FirstMI;
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
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst1 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = FirstMI;
// CHECK-PREDICATOR-NEXT:      if (( MI->getOpcode() != Test::Inst0 ))
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
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  bool isTestSingleFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    {{[[]}}{{[[]}}maybe_unused{{[]]}}{{[]]}} auto &MRI = SecondMI.getMF()->getRegInfo();
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          ( MI->getOpcode() == Test::Inst2 )
// CHECK-PREDICATOR-NEXT:          && MI->getOperand(0).getReg() == Test::X0
// CHECK-PREDICATOR-NEXT:        ))
// CHECK-PREDICATOR-NEXT:        return false;
// CHECK-PREDICATOR-NEXT:    }
// CHECK-PREDICATOR-NEXT:    if (!FirstMI)
// CHECK-PREDICATOR-NEXT:      return true;
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = FirstMI;
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
// CHECK-PREDICATOR-NEXT:    return true;
// CHECK-PREDICATOR-NEXT:  }
// CHECK-PREDICATOR-NEXT:  } // end namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif

// Check that we have generated target subfeature.
// CHECK-SUBTARGET: { "test-both-fusion-predicate", "Test BothFusionPredicate", Test::TestBothFusionPredicate
// CHECK-SUBTARGET: { "test-commutable-fusion", "Test Commutable Fusion", Test::TestCommutableFusion
// CHECK-SUBTARGET: { "test-fusion", "Test Fusion", Test::TestFusion
// CHECK-SUBTARGET: { "test-single-fusion", "Test SingleFusion", Test::TestSingleFusion

// Check that we have generated `getMacroFusions()` function.
// CHECK-SUBTARGET:      std::vector<MacroFusionPredTy> getMacroFusions() const override;

// CHECK-SUBTARGET:      std::vector<MacroFusionPredTy> TestGenSubtargetInfo::getMacroFusions() const {
// CHECK-SUBTARGET-NEXT:   std::vector<MacroFusionPredTy> Fusions;
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestBothFusionPredicate)) Fusions.push_back(llvm::isTestBothFusionPredicate); 
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestCommutableFusion)) Fusions.push_back(llvm::isTestCommutableFusion); 
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestFusion)) Fusions.push_back(llvm::isTestFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestSingleFusion)) Fusions.push_back(llvm::isTestSingleFusion);
// CHECK-SUBTARGET-NEXT:   return Fusions;
// CHECK-SUBTARGET-NEXT: }
