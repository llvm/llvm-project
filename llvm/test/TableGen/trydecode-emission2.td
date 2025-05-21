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

// CHECK:      /* 0 */       MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
// CHECK-NEXT: /* 3 */       MCD::OPC_FilterValue, 0, 31, 0, // Skip to: 38
// CHECK-NEXT: /* 7 */       MCD::OPC_ExtractField, 5, 3,  // Inst{7-5} ...
// CHECK-NEXT: /* 10 */      MCD::OPC_FilterValue, 0, 24, 0, // Skip to: 38
// CHECK-NEXT: /* 14 */      MCD::OPC_CheckField, 0, 2, 3, 6, 0, // Skip to: 26
// CHECK-NEXT: /* 20 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, // Opcode: InstB, skip to: 26
// CHECK-NEXT: /* 26 */      MCD::OPC_CheckField, 3, 2, 0, 6, 0, // Skip to: 38
// CHECK-NEXT: /* 32 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 1, 0, 0, // Opcode: InstA, skip to: 38
// CHECK-NEXT: /* 38 */      MCD::OPC_Fail,

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }

// CHECK-LARGE:      /* 0 */       MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
// CHECK-LARGE-NEXT: /* 3 */       MCD::OPC_FilterValue, 0, 36, 0, 0, // Skip to: 44
// CHECK-LARGE-NEXT: /* 8 */       MCD::OPC_ExtractField, 5, 3,  // Inst{7-5} ...
// CHECK-LARGE-NEXT: /* 11 */      MCD::OPC_FilterValue, 0, 28, 0, 0, // Skip to: 44
// CHECK-LARGE-NEXT: /* 16 */      MCD::OPC_CheckField, 0, 2, 3, 7, 0, 0, // Skip to: 30
// CHECK-LARGE-NEXT: /* 23 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, 0, // Opcode: InstB, skip to: 30
// CHECK-LARGE-NEXT: /* 30 */      MCD::OPC_CheckField, 3, 2, 0, 7, 0, 0, // Skip to: 44
// CHECK-LARGE-NEXT: /* 37 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 1, 0, 0, 0, // Opcode: InstA, skip to: 44
// CHECK-LARGE-NEXT: /* 44 */      MCD::OPC_Fail,

// CHECK-LARGE: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK-LARGE: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
