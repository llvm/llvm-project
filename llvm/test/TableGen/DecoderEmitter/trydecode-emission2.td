// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

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

// CHECK-LABEL: static const uint8_t DecoderTable8[26] = {
// CHECK-NEXT:    OPC_CheckField, 2, 1, 0,     //  0: check Inst[2] == 0x0
// CHECK-NEXT:    OPC_CheckField, 5, 3, 0,     //  4: check Inst[7:5] == 0x0
// CHECK-NEXT:    OPC_Scope, 8,                //  8: try {
// CHECK-NEXT:    OPC_CheckField, 0, 2, 3,     // 10:  check Inst[1:0] == 0x3
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 14:  decode to InstB using decoder 0
// CHECK-NEXT:                                 // 14: } else try {
// CHECK-NEXT:    OPC_CheckField, 3, 2, 0,     // 18:  check Inst[4:3] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 22:  decode to InstA using decoder 1
// CHECK-NEXT:                                 // 22: }
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstB(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
// CHECK: if (!Check(S, DecodeInstA(MI, insn, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
