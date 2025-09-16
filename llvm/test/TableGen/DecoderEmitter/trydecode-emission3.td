// RUN: llvm-tblgen -gen-disassembler  -I %p/../../../include %s | FileCheck %s
// RUN: llvm-tblgen -gen-disassembler --large-decoder-table -I %p/../../../include %s | FileCheck %s --check-prefix=CHECK-LARGE

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

class TestInstruction : Instruction {
  let Size = 1;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
}

def InstA : TestInstruction {
  let Inst = {0,0,0,0,?,?,?,?};
  let AsmString = "InstA";
}

def InstBOp : Operand<i32> {
  let DecoderMethod = "DecodeInstBOp";
  let hasCompleteDecoder = 0;
}

def InstB : TestInstruction {
  bits<2> op;
  let Inst{7...2} = {0,0,0,0,0,0};
  let Inst{1...0} = op;
  let OutOperandList = (outs InstBOp:$op);
  let AsmString = "InstB";
}

// CHECK:      /* 0 */       OPC_CheckField, 4, 4, 0,
// CHECK-NEXT: /* 4 */       OPC_Scope, 8, 0, // end scope at 15
// CHECK-NEXT: /* 7 */       OPC_CheckField, 2, 2, 0,
// CHECK-NEXT: /* 11 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-NEXT: /* 15 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: InstA, DecodeIdx: 1
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstBOp(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }

// CHECK-LARGE:      /* 0 */       OPC_CheckField, 4, 4, 0,
// CHECK-LARGE-NEXT: /* 4 */       OPC_Scope, 8, 0, 0, // end scope at 16
// CHECK-LARGE-NEXT: /* 8 */       OPC_CheckField, 2, 2, 0,
// CHECK-LARGE-NEXT: /* 12 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-LARGE-NEXT: /* 16 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: InstA, DecodeIdx: 1
// CHECK-LARGE-NEXT: };

// CHECK-LARGE: if (!Check(S, DecodeInstBOp(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
