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
// CHECK-LABEL: static const uint8_t DecoderTable128[34] = {
// CHECK-NEXT:  /* 0 */  MCD::OPC_ExtractField, 0, 64,  // Inst{63-0} ...
// CHECK-NEXT:  /* 3 */  MCD::OPC_FilterValueOrSkip, 1, 8, 0, // Skip to: 15
// CHECK-NEXT:  /* 7 */  MCD::OPC_CheckField, 127, 1, 1,
// CHECK-NEXT:  /* 11 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0, // Opcode: I2, DecodeIdx: 0
// CHECK-NEXT:  /* 15 */ MCD::OPC_FilterValue, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1,
// CHECK-NEXT:  /* 26 */ MCD::OPC_CheckField, 127, 1, 0,
// CHECK-NEXT:  /* 30 */ MCD::OPC_Decode, {{[0-9]+}}, 2, 0, // Opcode: I1, DecodeIdx: 0
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
