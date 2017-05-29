//===- Nios2InstrInfo.td - Target Description for Nios2 ------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Nios2 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "Nios2InstrFormats.td"

//===----------------------------------------------------------------------===//
// Nios2 Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

def simm16      : Operand<i32> {
  let DecoderMethod= "DecodeSimm16";
}

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{ return isInt<16>(N->getSExtValue()); }]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic and logical instructions with 2 register operands.
class ArithLogicI<bits<6> op, string instr_asm, SDNode OpNode,
                  Operand Od, PatLeaf imm_type, RegisterClass RC> :
  FI<op, (outs RC:$rB), (ins RC:$rA, Od:$imm16),
     !strconcat(instr_asm, "\t$rB, $rA, $imm16"),
     [(set RC:$rB, (OpNode RC:$rA, imm_type:$imm16))]> {
  let isReMaterializable = 1;
}

//===----------------------------------------------------------------------===//
// Nios2 R1 Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def ADDi : ArithLogicI<0x04, "addi", add, simm16, immSExt16, CPURegs>;
