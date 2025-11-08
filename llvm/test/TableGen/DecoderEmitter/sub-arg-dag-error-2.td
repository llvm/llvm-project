// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=error:

include "llvm/Target/Target.td"

def CustomOp : Operand<i32>;

// Used to crash.
// CHECK: error: In instruction 'I', operand #0 has 1 sub-arg names, expected 0

def I : Instruction {
  let Size = 1;
  bits<8> Inst;
  bits<1> i;

  let Inst{0} = 0;
  let Inst{1} = i;

  let OutOperandList = (outs);
  let InOperandList = (ins (CustomOp $i):$op);
}

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
