// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s 2>&1 | FileCheck %s --implicit-check-not=error:

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

def complex_nodec : Operand<i32> {
  let MIOperandInfo = (ops Regs, Regs);
}

def complex_withdec : Operand<i32> {
  let MIOperandInfo = (ops Regs, Regs);
  let DecoderMethod = "DecodeComplex";
}

class ArchInstr : Instruction {
  let Size = 1;
  bits<8> Inst;
}

// This definition is broken in both directions:
// 1. Uses a complex operand without a decoder, and without named sub-ops.
// 2. Uses a complex operand with named sub-ops, but with a decoder as well.

// CHECK: error: DecoderEmitter: operand "r1c" uses MIOperandInfo with multiple ops, but doesn't have a custom decoder!
// CHECK: note: Dumping record for previous error:
// CHECK: error: DecoderEmitter: operand "r1ab" has type "complex_withdec" with a custom DecoderMethod, but also named sub-operands.
def foo1 : ArchInstr {
  bits<2> r1a;
  bits<2> r1b;
  bits<2> r1c;

  let Inst{1-0} = r1a;
  let Inst{3-2} = r1b;
  let Inst{5-4} = r1c;
  let Inst{7-6} = 0b00;

  let OutOperandList = (outs complex_nodec:$r1c);
  let InOperandList = (ins (complex_withdec $r1a, $r1b):$r1ab);
}

// This definition has no errors.
def foo2 : ArchInstr {
  bits<2> r2a;
  bits<2> r2b;
  bits<2> r2c;

  let Inst{1-0} = r2a;
  let Inst{3-2} = r2b;
  let Inst{5-4} = r2c;
  let Inst{7-6} = 0b01;

  let OutOperandList = (outs complex_withdec:$r2c);
  let InOperandList = (ins (complex_nodec $r2a, $r2b):$r2ab);
}
