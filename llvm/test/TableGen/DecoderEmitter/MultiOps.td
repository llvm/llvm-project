// RUN: not llvm-tblgen -gen-disassembler -I %p/../../../include %s 2>&1 | FileCheck %s --implicit-check-not=error:

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

def complex_nodec1 : Operand<i32> {
  let MIOperandInfo = (ops Regs);
}

def complex_nodec2 : Operand<i32> {
  let MIOperandInfo = (ops Regs, Regs);
}

def complex_withdec1 : Operand<i32> {
  let MIOperandInfo = (ops Regs);
  let DecoderMethod = "DecodeComplex";
}

def complex_withdec2 : Operand<i32> {
  let MIOperandInfo = (ops Regs, Regs);
  let DecoderMethod = "DecodeComplex";
}

class ArchInstr : Instruction {
  let Size = 2;
  bits<16> Inst;
}

// This definition is broken in both directions:
// 1. Uses a complex operand without a decoder, and without named sub-ops.
// 2. Uses a complex operand with named sub-ops, but with a decoder as well.

// CHECK: error: DecoderEmitter: operand "r1c" has non-empty MIOperandInfo, but doesn't have a custom decoder!
// CHECK: note: Dumping record for previous error:
// CHECK: error: DecoderEmitter: operand "r2b" has non-empty MIOperandInfo, but doesn't have a custom decoder!
// CHECK: note: Dumping record for previous error:
// CHECK: error: DecoderEmitter: operand "r1" has type "complex_withdec2" with a custom DecoderMethod, but also named sub-operands.
// CHECK: error: DecoderEmitter: operand "r2" has type "complex_withdec1" with a custom DecoderMethod, but also named sub-operands.
def foo1 : ArchInstr {
  bits<2> r1a;
  bits<2> r1b;
  bits<2> r1c;
  bits<2> r2a;
  bits<2> r2b;

  let Inst{1-0} = r1a;
  let Inst{3-2} = r1b;
  let Inst{5-4} = r1c;
  let Inst{7-6} = r2a;
  let Inst{9-8} = r2b;
  let Inst{11-10} = 0b00;

  let OutOperandList = (outs complex_nodec2:$r1c, complex_nodec1:$r2b);
  let InOperandList = (ins (complex_withdec2 $r1a, $r1b):$r1,
                           (complex_withdec1 $r2a):$r2);
}

// This definition has no errors.
def foo2 : ArchInstr {
  bits<2> r2a;
  bits<2> r2b;
  bits<2> r2c;
  bits<2> r1a;
  bits<2> r1b;

  let Inst{1-0} = r2a;
  let Inst{3-2} = r2b;
  let Inst{5-4} = r2c;
  let Inst{7-6} = r1a;
  let Inst{9-8} = r1b;
  let Inst{11-10} = 0b01;

  let OutOperandList = (outs complex_withdec2:$r2c, complex_withdec1:$r1b);
  let InOperandList = (ins (complex_nodec2 $r2a, $r2b):$r2,
                           (complex_nodec1 $r1a):$r1);
}
