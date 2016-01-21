// RUN: llvm-tblgen -gen-disassembler -I %p/../../include %s | FileCheck %s

// Check that if decoding of an instruction fails and the instruction does not
// have a complete decoder method that can determine if the bitpattern is valid
// or not then the decoder tries to find a more general instruction that
// matches the bitpattern too.

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

def InstB : TestInstruction {
  let Inst = {0,0,0,0,0,0,?,?};
  let AsmString = "InstB";
  let DecoderMethod = "DecodeInstB";
  let hasCompleteDecoder = 0;
}

// CHECK:      /* 0 */       MCD::OPC_ExtractField, 4, 4,  // Inst{7-4} ...
// CHECK-NEXT: /* 3 */       MCD::OPC_FilterValue, 0, 14, 0, // Skip to: 21
// CHECK-NEXT: /* 7 */       MCD::OPC_CheckField, 2, 2, 0, 5, 0, // Skip to: 18
// CHECK-NEXT: /* 13 */      MCD::OPC_TryDecode, 25, 0, 0, 0, // Opcode: InstB, skip to: 18
// CHECK-NEXT: /* 18 */      MCD::OPC_Decode, 24, 1, // Opcode: InstA
// CHECK-NEXT: /* 21 */      MCD::OPC_Fail,

// CHECK: if (DecodeInstB(MI, insn, Address, Decoder) == MCDisassembler::Fail) { DecodeComplete = false; return MCDisassembler::Fail; }
