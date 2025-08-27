// RUN: llvm-tblgen -gen-disassembler  -I %p/../../include %s | FileCheck %s
// RUN: llvm-tblgen -gen-disassembler --large-decoder-table -I %p/../../include %s | FileCheck %s --check-prefix=CHECK-LARGE

// Test for OPC_ExtractField/OPC_CheckField with start bit > 255.
// These large start values may arise for architectures with long instruction
// words.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

class TestInstruction : Instruction {
  let Size = 64;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  field bits<512> Inst;
  field bits<512> SoftFail = 0;
}

def InstA : TestInstruction {
  let Inst{509-502} = {0,0,0,0,?,?,?,?};
  let AsmString = "InstA";
}

def InstB : TestInstruction {
  let Inst{509-502} = {0,0,0,0,0,0,?,?};
  let AsmString = "InstB";
  let DecoderMethod = "DecodeInstB";
  let hasCompleteDecoder = 0;
}

// CHECK:      /* 0 */       MCD::OPC_CheckFieldOrFail, 250, 3, 4, 0,
// CHECK-NEXT: /* 5 */       MCD::OPC_CheckField, 248, 3, 2, 0, 6, 0, // Skip to: 18
// CHECK-NEXT: /* 12 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, // Opcode: InstB, DecodeIdx: {{[0-9]+}}, Skip to: 18
// CHECK-NEXT: /* 18 */      MCD::OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: InstA, DecodeIdx: {{[0-9]+}}
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }


// CHECK-LARGE:      /* 0 */       MCD::OPC_CheckFieldOrFail, 250, 3, 4, 0,
// CHECK-LARGE-NEXT: /* 5 */       MCD::OPC_CheckField, 248, 3, 2, 0, 7, 0, 0, // Skip to: 20
// CHECK-LARGE-NEXT: /* 13 */      MCD::OPC_TryDecode, {{[0-9]+}}, {{[0-9]+}}, 0, 0, 0, 0, // Opcode: InstB, DecodeIdx: {{[0-9]+}}, Skip to: 20
// CHECK-LARGE-NEXT: /* 20 */      MCD::OPC_Decode, {{[0-9]+}}, {{[0-9]+}}, 1, // Opcode: InstA, DecodeIdx: {{[0-9]+}}
// CHECK-LARGE-NEXT: };

// CHECK-LARGE: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
