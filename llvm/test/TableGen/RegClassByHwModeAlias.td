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

// TODO: Can't use a fixed register for this instruction, would need RegisterByHwMode.
// def MY_T_PTR : InstAlias<"t_ptr $src", (TEST_PTR X0, XRegs:$src)>;
def MY_T_PTR_EVEN : InstAlias<"t_ptr.even $src", (TEST_PTR EvenPtrRC:$dst, EvenPtrRC:$src)>;

// CHECK-LABEL: static const AliasPatternCond Conds[] = {
// CHECK-NEXT:    // (TEST_PTR EvenPtrRC:$dst, EvenPtrRC:$src) - 0
// CHECK-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// CHECK-NEXT:    {AliasPatternCond::K_RegClassByHwMode, MyTarget::EvenPtrRC},
// CHECK-NEXT:    // (TEST_XREG X0, XRegs:$src) - 2
// CHECK-NEXT:    {AliasPatternCond::K_Reg, MyTarget::X0},
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::XRegsRegClassID},
// CHECK-NEXT:    // (TEST_XREG EvenXRegs:$dst, EvenXRegs:$src) - 4
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenXRegsRegClassID},
// CHECK-NEXT:    {AliasPatternCond::K_RegClass, MyTarget::EvenXRegsRegClassID},
// CHECK-NEXT:  };

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
