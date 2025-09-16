// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s
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
  let Inst = {0,0,0,0,0,0,?,?};
  let AsmString = "InstA";
  let DecoderMethod = "DecodeInstA";
  let hasCompleteDecoder = 0;
}

def InstB : TestInstruction {
  let Inst = {0,0,0,?,?,0,1,1};
  let AsmString = "InstB";
  let DecoderMethod = "DecodeInstB";
  let hasCompleteDecoder = 0;
}

// CHECK:      /* 0 */       OPC_CheckField, 2, 1, 0,
// CHECK-NEXT: /* 4 */       OPC_CheckField, 5, 3, 0,
// CHECK-NEXT: /* 8 */       OPC_Scope, 8, 0, // end scope at 19
// CHECK-NEXT: /* 11 */      OPC_CheckField, 0, 2, 3,
// CHECK-NEXT: /* 15 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-NEXT: /* 19 */      OPC_CheckField, 3, 2, 0,
// CHECK-NEXT: /* 23 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1,

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }

// CHECK-LARGE:      /* 0 */       OPC_CheckField, 2, 1, 0,
// CHECK-LARGE-NEXT: /* 4 */       OPC_CheckField, 5, 3, 0,
// CHECK-LARGE-NEXT: /* 8 */       OPC_Scope, 8, 0, 0, // end scope at 20
// CHECK-LARGE-NEXT: /* 12 */      OPC_CheckField, 0, 2, 3,
// CHECK-LARGE-NEXT: /* 16 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-LARGE-NEXT: /* 20 */      OPC_CheckField, 3, 2, 0,
// CHECK-LARGE-NEXT: /* 24 */      OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1,
// CHECK-LARGE-NEXT: };

// CHECK-LARGE: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK-LARGE: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
