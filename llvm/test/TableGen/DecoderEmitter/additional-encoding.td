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

// CHECK:      /* 0 */  OPC_ExtractField, 12, 4,               // Field = Inst{15-12}
// CHECK-NEXT: /* 3 */  OPC_FilterValueOrSkip, 0, 15, 0,       // if Field != 0x0 skip to 22
// CHECK-NEXT: /* 7 */  OPC_Scope, 8, 0,                       // end scope at 18
// CHECK-NEXT: /* 10 */ OPC_CheckField, 6, 6, 0,               // if Inst{11-6} != 0x0
// CHECK-NEXT: /* 14 */ OPC_Decode, {{[0-9]+}}, 2, 0,          // Opcode: {{.*}}:NOP, DecodeIdx: 0
// CHECK-NEXT: /* 18 */ OPC_Decode, {{[0-9]+}}, 2, 1,          // Opcode: SHIFT0, DecodeIdx: 1
// CHECK-NEXT: /* 22 */ OPC_FilterValueOrSkip, 1, 15, 0,       // if Field != 0x1 skip to 41
// CHECK-NEXT: /* 26 */ OPC_Scope, 8, 0,                       // end scope at 37
// CHECK-NEXT: /* 29 */ OPC_CheckField, 6, 6, 0,               // if Inst{11-6} != 0x0
// CHECK-NEXT: /* 33 */ OPC_Decode, {{[0-9]+}}, 2, 0,          // Opcode: {{.*}}:NOP, DecodeIdx: 0
// CHECK-NEXT: /* 37 */ OPC_Decode, {{[0-9]+}}, 2, 1,          // Opcode: SHIFT1, DecodeIdx: 1
// CHECK-NEXT: /* 41 */ OPC_FilterValueOrSkip, 2, 15, 0,       // if Field != 0x2 skip to 60
// CHECK-NEXT: /* 45 */ OPC_Scope, 8, 0,                       // end scope at 56
// CHECK-NEXT: /* 48 */ OPC_CheckField, 6, 6, 0,               // if Inst{11-6} != 0x0
// CHECK-NEXT: /* 52 */ OPC_Decode, {{[0-9]+}}, 2, 0,          // Opcode: {{.*}}:NOP, DecodeIdx: 0
// CHECK-NEXT: /* 56 */ OPC_Decode, {{[0-9]+}}, 2, 1,          // Opcode: SHIFT2, DecodeIdx: 1
// CHECK-NEXT: /* 60 */ OPC_FilterValue, 3,                    // if Field != 0x3
// CHECK-NEXT: /* 62 */ OPC_Scope, 8, 0,                       // end scope at 73
// CHECK-NEXT: /* 65 */ OPC_CheckField, 6, 6, 0,               // if Inst{11-6} != 0x0
// CHECK-NEXT: /* 69 */ OPC_Decode, {{[0-9]+}}, 2, 0,          // Opcode: {{.*}}:NOP, DecodeIdx: 0
// CHECK-NEXT: /* 73 */ OPC_Decode, {{[0-9]+}}, 2, 1,          // Opcode: SHIFT3, DecodeIdx: 1


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
