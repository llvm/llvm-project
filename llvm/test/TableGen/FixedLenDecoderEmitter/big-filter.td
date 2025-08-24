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
// CHECK-LABEL: static const uint8_t DecoderTable128[35] = {
// CHECK-NEXT:    MCD::OPC_ExtractField, 0, 64,
// CHECK-NEXT:    MCD::OPC_FilterValue, 1, 8, 0,
// CHECK-NEXT:    MCD::OPC_CheckFieldOrFail, 127, 1, 1,
// CHECK-NEXT:    MCD::OPC_Decode, 187, 2, 0,
// CHECK-NEXT:    MCD::OPC_FilterValueOrFail, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1,
// CHECK-NEXT:    MCD::OPC_CheckFieldOrFail, 127, 1, 0,
// CHECK-NEXT:    MCD::OPC_Decode, 186, 2, 0,
// CHECK-NEXT:    MCD::OPC_Fail,
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
