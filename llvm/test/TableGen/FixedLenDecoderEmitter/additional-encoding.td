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

// CHECK:      /* 0 */  MCD::OPC_ExtractField, 12, 4,                // Inst{15-12} ...
// CHECK-NEXT: /* 3 */  MCD::OPC_FilterValue, 0, 14, 0,              // Skip to: 21
// CHECK-NEXT: /* 7 */  MCD::OPC_CheckField, 6, 6, 0, 4, 0,          // Skip to: 17
// CHECK-NEXT: /* 13 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,           // Opcode: {{.*}}:NOP
// CHECK-NEXT: /* 17 */ MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, 2, 1,
// CHECK-NEXT: /* 21 */ MCD::OPC_FilterValue, 1, 14, 0,              // Skip to: 39
// CHECK-NEXT: /* 25 */ MCD::OPC_CheckField, 6, 6, 0, 4, 0,          // Skip to: 35
// CHECK-NEXT: /* 31 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,           // Opcode: {{.*}}:NOP
// CHECK-NEXT: /* 35 */ MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, 2, 1,
// CHECK-NEXT: /* 39 */ MCD::OPC_FilterValue, 2, 14, 0,              // Skip to: 57
// CHECK-NEXT: /* 43 */ MCD::OPC_CheckField, 6, 6, 0, 4, 0,          // Skip to: 53
// CHECK-NEXT: /* 49 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,           // Opcode: {{.*}}:NOP
// CHECK-NEXT: /* 53 */ MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, 2, 1,
// CHECK-NEXT: /* 57 */ MCD::OPC_FilterValueOrFail, 3,
// CHECK-NEXT: /* 59 */ MCD::OPC_CheckField, 6, 6, 0, 4, 0,          // Skip to: 69
// CHECK-NEXT: /* 65 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0,           // Opcode: {{.*}}:NOP
// CHECK-NEXT: /* 69 */ MCD::OPC_TryDecodeOrFail, {{[0-9]+}}, 2, 1,

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
