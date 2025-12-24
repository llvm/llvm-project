// RUN: llvm-tblgen -gen-disassembler -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

class I : Instruction {
  let InOperandList = (ins);
  let OutOperandList = (outs);
  let Size = 16;
  bits<128> Inst;
}

// Check that a 64-bit filter with all bits set does not confuse DecoderEmitter.
//
// CHECK-LABEL: static const uint8_t DecoderTable128[32] = {
// CHECK-NEXT:    OPC_SwitchField, 0, 64,      //  0: switch Inst[63:0] {
// CHECK-NEXT:    1, 8,                        //  3: case 0x1: {
// CHECK-NEXT:    OPC_CheckField, 127, 1, 1,   //  5:  check Inst[127] == 0x1
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, //  9:  decode to I2 using decoder 0
// CHECK-NEXT:                                 //  9: }
// CHECK-NEXT:    255, 255, 255, 255, 255, 255, 255, 255, 255, 1, 0,
// CHECK-NEXT:                                 // 13: case 0xffffffffffffffff: {
// CHECK-NEXT:    OPC_CheckField, 127, 1, 0,   // 24:  check Inst[127] == 0x0
// CHECK-NEXT:    OPC_Decode, {{[0-9, ]+}}, 0, // 28:  decode to I1 using decoder 0
// CHECK-NEXT:                                 // 28: }
// CHECK-NEXT:                                 // 28: } // switch Inst[63:0]
// CHECK-NEXT:  };

def I1 : I {
  let Inst{63...0} = -1;
  let Inst{127} = 0;
}

def I2 : I {
  let Inst{63...0} = 1;
  let Inst{127} = 1;
}

def II : InstrInfo;

def MyTarget : Target {
  let InstructionSet = II;
}
