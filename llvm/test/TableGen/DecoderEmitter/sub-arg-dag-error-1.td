// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=error:

include "llvm/Target/Target.td"

def R0 : Register<"r0">;
def RC : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

// Used to crash.
// CHECK: error: In instruction 'I', operand #0 has 1 sub-arg names, but no sub-operands

def I : Instruction {
  let Size = 1;
  bits<8> Inst;
  bits<1> r;

  let Inst{0} = 0;
  let Inst{1} = r;

  let OutOperandList = (outs);
  let InOperandList = (ins (RC $r):$op);
}

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
