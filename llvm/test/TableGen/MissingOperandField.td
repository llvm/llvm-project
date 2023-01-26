// RUN: not llvm-tblgen -gen-emitter -I %p/../../include %s 2>&1 | FileCheck %s --implicit-check-not=error:

// Check that we emit reasonable diagnostics when fields do not have
// corresponding operands.

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;

def Regs : RegisterClass<"foo", [i32], 0, (add Reg)>;

// CHECK: error: No operand named rd in record foo (would've used positional operand #0 ('xd') sub-op #0 with useDeprecatedPositionallyEncodedOperands=true)
// CHECK: error: No operand named rs in record foo (would've given 'too few operands' error with useDeprecatedPositionallyEncodedOperands=true)
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
