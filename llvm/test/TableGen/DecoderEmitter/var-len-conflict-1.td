// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class I : Instruction {
  let InOperandList = (ins i32imm:$op);
  let OutOperandList = (outs);
}

// Check that we don't try to read the second byte without ruling out
// 1-byte encodings first. This should actually be a decoding conflict,
// but DecoderEmitter heuristics decide that I8_0 and I8_1 are more specific
// than the rest and give them priority.

//          _______0  I8_0
//          _______1  I8_1
// 00000000 ________  I16_0
// 00000001 ________  I16_1
// 00000010 ________  I16_2

// CHECK: switch Inst[0]
// CHECK: decode to I8_0
// CHECK: decode to I8_1
// CHECK: switch Inst[15:8]
// CHECK: decode to I16_0
// CHECK: decode to I16_1

def I8_0  : I { dag Inst = (descend (operand "$op", 7), 0b0); }
def I8_1  : I { dag Inst = (descend (operand "$op", 7), 0b1); }
def I16_0 : I { dag Inst = (descend 0b00000000, (operand "$op", 8)); }
def I16_1 : I { dag Inst = (descend 0b00000001, (operand "$op", 8)); }
def I16_2 : I { dag Inst = (descend 0b00000010, (operand "$op", 8)); }

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
