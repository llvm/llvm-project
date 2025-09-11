// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s
// RUN: llvm-tblgen -gen-disassembler --large-decoder-table -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-LARGE

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

// CHECK:      /* 0 */       MCD::OPC_CheckField, 2, 1, 0,
// CHECK-NEXT: /* 4 */       MCD::OPC_CheckField, 5, 3, 0,
// CHECK-NEXT: /* 8 */       MCD::OPC_Scope, 8, 0, // Skip to: 19
// CHECK-NEXT: /* 11 */      MCD::OPC_CheckField, 0, 2, 3,
// CHECK-NEXT: /* 15 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-NEXT: /* 19 */      MCD::OPC_CheckField, 3, 2, 0,
// CHECK-NEXT: /* 23 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 1,

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }

// CHECK-LARGE:      /* 0 */       MCD::OPC_CheckField, 2, 1, 0,
// CHECK-LARGE-NEXT: /* 4 */       MCD::OPC_CheckField, 5, 3, 0,
// CHECK-LARGE-NEXT: /* 8 */       MCD::OPC_Scope, 8, 0, 0, // Skip to: 20
// CHECK-LARGE-NEXT: /* 12 */      MCD::OPC_CheckField, 0, 2, 3,
// CHECK-LARGE-NEXT: /* 16 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0,
// CHECK-LARGE-NEXT: /* 20 */      MCD::OPC_CheckField, 3, 2, 0,
// CHECK-LARGE-NEXT: /* 24 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 1,
// CHECK-LARGE-NEXT: };

// CHECK-LARGE: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK-LARGE: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
