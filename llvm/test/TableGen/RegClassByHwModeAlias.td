// RUN: llvm-tblgen -gen-asm-writer  -I %S -I %p/../../include %s -o - | FileCheck %s

include "Common/RegClassByHwModeCommon.td"

def IsPtrY : Predicate<"Subtarget->isPtrY()">;
defvar PtrX = DefaultMode;
def PtrY : HwMode<[IsPtrY]>;

// Define more restrictive subset classes to check that those are handled.
def EvenXRegs : RegisterClass<"MyTarget", [i64], 64, (add X0, X2, X4, X6)>;
def EvenYRegs : RegisterClass<"MyTarget", [i64], 64, (add Y0, Y2, Y4, Y6)>;
def PtrRC : RegClassByHwMode<[PtrX, PtrY], [XRegs, YRegs]>;
def EvenPtrRC : RegClassByHwMode<[PtrX, PtrY], [EvenXRegs, EvenYRegs]>;
def NullReg : RegisterByHwMode<PtrRC, [PtrX, PtrY], [X0, Y0]>;

def TEST_XREG : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "t_x $dst, $src";
  let opcode = 0;
}
def TEST_PTR : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "t_ptr $dst, $src";
  let opcode = 0;
}

def MY_T_X : InstAlias<"t_x $src", (TEST_XREG X0, XRegs:$src)>;
def MY_T_X_EVEN : InstAlias<"t_x.even $src", (TEST_XREG EvenXRegs:$dst, EvenXRegs:$src)>;

def MY_T_PTR : InstAlias<"t_ptr $src", (TEST_PTR NullReg, PtrRC:$src)>;
def MY_T_PTR_EVEN : InstAlias<"t_ptr.even $src", (TEST_PTR EvenPtrRC:$dst, EvenPtrRC:$src)>;

// CHECK-LABEL: static const AliasPatternCond Conds[] = {
// CHECK-NEXT:    // (TEST_PTR NullReg, PtrRC:$src) - 0
// CHECK-NEXT:    {AliasPatternCond::K_Custom, 1/*NullReg*/},
// CHECK-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::PtrRC},
// CHECK-NEXT:    // (TEST_PTR EvenPtrRC:$dst, EvenPtrRC:$src) - 2
// CHECK-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// CHECK-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// CHECK-NEXT:    // (TEST_XREG X0, XRegs:$src) - 4
// CHECK-NEXT:    {AliasPatternCond::K_Reg, MyTarget::X0},
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::XRegsRegClassID},
// CHECK-NEXT:    // (TEST_XREG EvenXRegs:$dst, EvenXRegs:$src) - 6
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenXRegsRegClassID},
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenXRegsRegClassID},
// CHECK-NEXT:  };

// CHECK-LABEL: static bool MyTargetInstPrinterValidateMCOperand(const MCOperand &MCOp,
// CHECK-NEXT:                    const MCSubtargetInfo &STI,
// CHECK-NEXT:                    unsigned PredicateIndex) {
// CHECK-NEXT:    switch (PredicateIndex) {
// CHECK-NEXT:    default:
// CHECK-NEXT:      llvm_unreachable("Unknown MCOperandPredicate kind");
// CHECK-NEXT:      break;
// CHECK-NEXT:    case 1: {
// CHECK-NEXT:      return MCOp.isReg() && MCOp.getReg() == MyTarget::RegisterByHwMode::getNullReg(STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo));
// CHECK-NEXT:    }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

def MyTargetISA : InstrInfo;
def MyTargetAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
  let AssemblyWriters = [MyTargetAsmWriter];
}
