// RUN: not llvm-tblgen -gen-emitter -I %p/../../include %s 2>&1 | FileCheck %s --implicit-check-not=error:

// Check that TableGen doesn't crash on insufficient positional
// instruction operands.

// TODO: remove this test when removing useDeprecatedPositionallyEncodedOperands

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo {
  let useDeprecatedPositionallyEncodedOperands = 1;
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

// CHECK: error: Too few operands in record foo (no match for variable rs)
// CHECK: note: Dumping record for previous error:
def foo : Instruction {
  bits<3> rd;
  bits<3> rs;

  bits<8> Inst;
  let Inst{1-0} = 0;
  let Inst{4-2} = rd;
  let Inst{7-5} = rs;

  let OutOperandList = (outs Regs:$xd);
  let InOperandList = (ins);
}
