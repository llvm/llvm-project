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
// CHECK-NEXT: /* 3 */       MCD::OPC_FilterValueOrFail, 0,
// CHECK-NEXT: /* 5 */       MCD::OPC_ExtractField, 5, 3,  // Inst{7-5} ...
// CHECK-NEXT: /* 8 */       MCD::OPC_FilterValueOrFail, 0
// CHECK-NEXT: /* 10 */      MCD::OPC_CheckField, 0, 2, 3, 6, 0, // Skip to: 22
// CHECK-NEXT: /* 16 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, // Opcode: InstB, DecodeIdx: {{[0-9]+}}, Skip to: 22
// CHECK-NEXT: /* 22 */      MCD::OPC_CheckFieldOrFail, 3, 2, 0,
// CHECK-NEXT: /* 26 */      MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, {{[0-9]+}}, 1,
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }

// CHECK-LARGE:      /* 0 */       MCD::OPC_ExtractField, 2, 1,  // Inst{2} ...
// CHECK-LARGE-NEXT: /* 3 */       MCD::OPC_FilterValueOrFail, 0,
// CHECK-LARGE-NEXT: /* 5 */       MCD::OPC_ExtractField, 5, 3,  // Inst{7-5} ...
// CHECK-LARGE-NEXT: /* 8 */       MCD::OPC_FilterValueOrFail, 0,
// CHECK-LARGE-NEXT: /* 10 */      MCD::OPC_CheckField, 0, 2, 3, 7, 0, 0, // Skip to: 24
// CHECK-LARGE-NEXT: /* 17 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, 0, // Opcode: InstB, DecodeIdx: {{[0-9]+}}, Skip to: 24
// CHECK-LARGE-NEXT: /* 24 */      MCD::OPC_CheckFieldOrFail, 3, 2, 0,
// CHECK-LARGE-NEXT: /* 28 */      MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, {{[0-9]+}}, 1,
// CHECK-LARGE-NEXT: };

// CHECK-LARGE: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK-LARGE: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
