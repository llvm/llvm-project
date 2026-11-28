// RUN: llvm-tblgen -gen-macro-fusion-pred -I %p/../../include %s | FileCheck --check-prefix=CHECK-PREDICATOR %s
// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s | FileCheck --check-prefix=CHECK-SUBTARGET %s

// Check FusionFor and SimpleFusionFor. They are named after the feature and
// sorted with the other fusions.

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

def TestAFusion : SubtargetFeature<"fusion-a", "HasAFusion", "true",
                                   "Test A Fusion">;
def TestCFusion : SubtargetFeature<"fusion-c", "HasCFusion", "true",
                                   "Test C Fusion">;

def : SimpleFusionFor<TestCFusion, CheckOpcode<[Inst0]>, CheckOpcode<[Inst1]>>;

def TestBFusion : SimpleFusion<"b", "HasBFusion", "Test B Fusion",
                               CheckOpcode<[Inst1]>, CheckOpcode<[Inst0]>>;

let IsCommutable = 1 in
def : FusionFor<TestAFusion, [FirstInstHasSameReg<0, 1>]>;

// CHECK-PREDICATOR:      bool isTestAFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &, const SDep *);{{$}}
// CHECK-PREDICATOR-NEXT: bool isTestBFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &, const SDep *);{{$}}
// CHECK-PREDICATOR-NEXT: bool isTestCFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &, const SDep *);{{$}}

// CHECK-PREDICATOR:      STATISTIC(NumTestAFusionPreRA, "Times TestAFusion Triggered (pre-ra)");
// CHECK-PREDICATOR:      bool isTestAFusion(
// CHECK-PREDICATOR:        if (!FirstMI->getDesc().isCommutable())
// CHECK-PREDICATOR:      bool isTestBFusion(
// CHECK-PREDICATOR:        if (( MI->getOpcode() != Test::Inst0 ))
// CHECK-PREDICATOR:        if (( MI->getOpcode() != Test::Inst1 ))
// CHECK-PREDICATOR:      bool isTestCFusion(
// CHECK-PREDICATOR:        if (( MI->getOpcode() != Test::Inst1 ))
// CHECK-PREDICATOR:        if (( MI->getOpcode() != Test::Inst0 ))

// CHECK-SUBTARGET:      std::vector<MacroFusionPredTy> TestGenSubtargetInfo::getMacroFusions() const {
// CHECK-SUBTARGET-NEXT:   std::vector<MacroFusionPredTy> Fusions;
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestAFusion)) Fusions.push_back(llvm::isTestAFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestBFusion)) Fusions.push_back(llvm::isTestBFusion);
// CHECK-SUBTARGET-NEXT:   if (hasFeature(Test::TestCFusion)) Fusions.push_back(llvm::isTestCFusion);
// CHECK-SUBTARGET-NEXT:   return Fusions;
// CHECK-SUBTARGET-NEXT: }
