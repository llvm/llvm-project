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

// CHECK:      /* 0 */  MCD::OPC_Scope, 17, 0,               // Skip to: 20
// CHECK-NEXT: /* 3 */  MCD::OPC_ExtractField, 0, 1,         // Inst{0} ...
// CHECK-NEXT: /* 6 */  MCD::OPC_FilterValueOrSkip, 0, 4, 0, // Skip to: 14
// CHECK-NEXT: /* 10 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,   // Opcode: I8_0, DecodeIdx: 0
// CHECK-NEXT: /* 14 */ MCD::OPC_FilterValue, 1,
// CHECK-NEXT: /* 16 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,   // Opcode: I8_1, DecodeIdx: 0
// CHECK-NEXT: /* 20 */ MCD::OPC_ExtractField, 8, 8,         // Inst{15-8} ...
// CHECK-NEXT: /* 23 */ MCD::OPC_FilterValueOrSkip, 0, 4, 0, // Skip to: 31
// CHECK-NEXT: /* 27 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 1,   // Opcode: I16_0, DecodeIdx: 1
// CHECK-NEXT: /* 31 */ MCD::OPC_FilterValueOrSkip, 1, 4, 0, // Skip to: 39
// CHECK-NEXT: /* 35 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 1,   // Opcode: I16_1, DecodeIdx: 1
// CHECK-NEXT: /* 39 */ MCD::OPC_FilterValue, 2,
// CHECK-NEXT: /* 41 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 1,   // Opcode: I16_2, DecodeIdx: 1

def I8_0  : I { dag Inst = (descend (operand "$op", 7), 0b0); }
def I8_1  : I { dag Inst = (descend (operand "$op", 7), 0b1); }
def I16_0 : I { dag Inst = (descend 0b00000000, (operand "$op", 8)); }
def I16_1 : I { dag Inst = (descend 0b00000001, (operand "$op", 8)); }
def I16_2 : I { dag Inst = (descend 0b00000010, (operand "$op", 8)); }

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
