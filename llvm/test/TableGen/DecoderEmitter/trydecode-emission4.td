// RUN: llvm-tblgen -gen-disassembler  -I %p/../../../include %s | FileCheck %s

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

// CHECK-LABEL: static const uint8_t DecoderTable512[20] = {
// CHECK-NEXT:    OPC_CheckField, 250, 3, 4, 0, //  0: check Inst[509:506] == 0x0
// CHECK-NEXT:    OPC_Scope, 9,                 //  5: try {
// CHECK-NEXT:    OPC_CheckField, 248, 3, 2, 0, //  7:  check Inst[505:504] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0,  // 12:  decode to InstB using decoder 0
// CHECK-NEXT:                                  // 12: } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1,  // 16:  decode to InstA using decoder 1
// CHECK-NEXT:                                  // 16: }
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
