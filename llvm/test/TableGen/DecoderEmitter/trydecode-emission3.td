// RUN: llvm-tblgen -gen-disassembler  -I %p/../../../include %s | FileCheck %s

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

// CHECK-LABEL: static const uint8_t DecoderTable8[18] = {
// CHECK-NEXT:    OPC_CheckField, 4, 4, 0,     //  0: check Inst[7:4] == 0x0
// CHECK-NEXT:    OPC_Scope, 8,                //  4: try {
// CHECK-NEXT:    OPC_CheckField, 2, 2, 0,     //  6:  check Inst[3:2] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 10:  decode to InstB using decoder 0
// CHECK-NEXT:                                 // 10: } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 14:  decode to InstA using decoder 1
// CHECK-NEXT:                                 // 14: }
// CHECK-NEXT: };

// CHECK: if (!Check(S, DecodeInstBOp(MI, tmp, Address, Decoder))) { DecodeComplete = false; return MCDisassembler::Fail; }
