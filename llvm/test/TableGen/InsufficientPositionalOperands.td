// RUN: not llvm-tblgen -gen-emitter -I %p/../../include %s 2>&1 | FileCheck %s

// Check that TableGen doesn't crash on insufficient positional
// instruction operands.

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

def foo : Instruction {
  bits<3> rd;
  bits<3> rs;

  bits<8> Inst;
  let Inst{1-0} = 0;
  let Inst{4-2} = rd;
  let Inst{7-5} = rs;

// CHECK: Too few operands in record foo (no match for variable rs)
  let OutOperandList = (outs Regs:$xd);
  let InOperandList = (ins);
}
