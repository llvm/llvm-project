// RUN: llvm-tblgen -gen-macro-fusion-pred -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-PREDICATOR

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

def TestFusion: SimpleFusion<CheckOpcode<[Inst0]>,
                             CheckAll<[
                              CheckOpcode<[Inst1]>,
                              CheckRegOperand<0, X0>
                             ]>>;

// CHECK-PREDICATOR:       #ifdef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_DECL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(const TargetInstrInfo &, const TargetSubtargetInfo &, const MachineInstr *, const MachineInstr &);
// CHECK-PREDICATOR-NEXT:  } // end namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif

// CHECK-PREDICATOR:       #ifdef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-NEXT:  #undef GET_Test_MACRO_FUSION_PRED_IMPL
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  namespace llvm {
// CHECK-PREDICATOR-NEXT:  bool isTestFusion(
// CHECK-PREDICATOR-NEXT:      const TargetInstrInfo &TII,
// CHECK-PREDICATOR-NEXT:      const TargetSubtargetInfo &STI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr *FirstMI,
// CHECK-PREDICATOR-NEXT:      const MachineInstr &SecondMI) {
// CHECK-PREDICATOR-NEXT:    auto &MRI = SecondMI.getMF()->getRegInfo();
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
// CHECK-PREDICATOR-NEXT:    {
// CHECK-PREDICATOR-NEXT:      const MachineInstr *MI = &SecondMI;
// CHECK-PREDICATOR-NEXT:      if (!(
// CHECK-PREDICATOR-NEXT:          MI->getOperand(0).getReg().isVirtual()
// CHECK-PREDICATOR-NEXT:          || MI->getOperand(0).getReg() == MI->getOperand(1).getReg()
// CHECK-PREDICATOR-NEXT:        ))
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
// CHECK-PREDICATOR-NEXT:  } // end namespace llvm
// CHECK-PREDICATOR-EMPTY:
// CHECK-PREDICATOR-NEXT:  #endif
