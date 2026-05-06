// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class Enc {
  int Size = 2;
  bits<16> Inst;
}

class EncSHIFT<bits<2> opc> : Enc {
  bits<6> shamt;
  let Inst{15...14} = {0, 0};
  let Inst{13...12} = opc;
  let Inst{11...6} = shamt;
}

class EncNOP<bits<2> opc> : Enc {
  let Inst{15...14} = {0, 0};
  let Inst{13...12} = opc;
  let Inst{11...6} = {0, 0, 0, 0, 0, 0};
}

def ShAmtOp : Operand<i32> {
  let DecoderMethod = "decodeShAmt";
  let hasCompleteDecoder = false;
}

class I<dag out_ops, dag in_ops> : Instruction {
  let InOperandList = in_ops;
  let OutOperandList = out_ops;
}

// CHECK-LABEL: static const uint8_t DecoderTable16[67] = {
// CHECK-NEXT:    OPC_SwitchField, 12, 4,      //  0: switch Inst[15:12] {
// CHECK-NEXT:    0, 14,                       //  3: case 0x0: {
// CHECK-NEXT:    OPC_Scope, 8,                //  5:  try {
// CHECK-NEXT:    OPC_CheckField, 6, 6, 0,     //  7:   check Inst[11:6] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 11:   decode to NOP using decoder 0
// CHECK-NEXT:                                 // 11:  } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 15:   decode to SHIFT0 using decoder 1
// CHECK-NEXT:                                 // 15:  }
// CHECK-NEXT:                                 // 15: }
// CHECK-NEXT:    1, 14,                       // 19: case 0x1: {
// CHECK-NEXT:    OPC_Scope, 8,                // 21:  try {
// CHECK-NEXT:    OPC_CheckField, 6, 6, 0,     // 23:   check Inst[11:6] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 27:   decode to {{.*}}:NOP using decoder 0
// CHECK-NEXT:                                 // 27:  } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 31:   decode to SHIFT1 using decoder 1
// CHECK-NEXT:                                 // 31:  }
// CHECK-NEXT:                                 // 31: }
// CHECK-NEXT:    2, 14,                       // 35: case 0x2: {
// CHECK-NEXT:    OPC_Scope, 8,                // 37:  try {
// CHECK-NEXT:    OPC_CheckField, 6, 6, 0,     // 39:   check Inst[11:6] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 43:   decode to {{.*}}:NOP using decoder 0
// CHECK-NEXT:                                 // 43:  } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 47:   decode to SHIFT2 using decoder 1
// CHECK-NEXT:                                 // 47:  }
// CHECK-NEXT:                                 // 47: }
// CHECK-NEXT:    3, 0,                        // 51: case 0x3: {
// CHECK-NEXT:    OPC_Scope, 8,                // 53:  try {
// CHECK-NEXT:    OPC_CheckField, 6, 6, 0,     // 55:   check Inst[11:6] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 59:   decode to {{.*}}:NOP using decoder 0
// CHECK-NEXT:                                 // 59:  } else try {
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 1, // 63:   decode to SHIFT3 using decoder 1
// CHECK-NEXT:                                 // 63:  }
// CHECK-NEXT:                                 // 63: }
// CHECK-NEXT:                                 // 63: } // switch Inst[15:12]
// CHECK-NEXT:  };

class SHIFT<bits<2> opc> : I<(outs), (ins ShAmtOp:$shamt)>, EncSHIFT<opc>;
def SHIFT0 : SHIFT<0>;
def SHIFT1 : SHIFT<1>;
def SHIFT2 : SHIFT<2>;
def SHIFT3 : SHIFT<3>;

def NOP : I<(outs), (ins)>, EncNOP<0>;
def : AdditionalEncoding<NOP>, EncNOP<1>;
def : AdditionalEncoding<NOP>, EncNOP<2>;
def : AdditionalEncoding<NOP>, EncNOP<3>;

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
