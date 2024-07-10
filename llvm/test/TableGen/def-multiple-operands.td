// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo {}

def arch : Target {
  let InstructionSet = archInstrInfo;
}

def R0 : Register<"r0">;
def P0 : Register<"p0">;
def R32 : RegisterClass<"MyNS", [i32], 0, (add R0)>;
def P1 : RegisterClass<"MyNS", [i1], 0, (add P0)>;

def Reg3Opnd : Operand<OtherVT> {
  let MIOperandInfo = (ops R32, R32, P1);
}

// The following checks verify that 'MCInstrDesc' entry for 'InstA' has the
// expected 'NumOperands' and 'NumDefs', i.e. 'InstA' should have 3 defs out of
// 4 operands.

// CHECK: archInstrTable {{.* = \{}}
// CHECK: {{\{}}
// CHECK: {{\{}} [[ID:[0-9]+]], 4, 3, 13, {{.+\}, \/\/}}
// CHECK-SAME: Inst #[[ID]] = InstA
def InstA : Instruction {
  let Namespace = "MyNS";
  let Size = 13;
  // InstA should have 3 defs out of 4 operands.
  let OutOperandList = (outs Reg3Opnd:$dst);
  let InOperandList = (ins i32imm:$c);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
  let hasSideEffects = false;
}
