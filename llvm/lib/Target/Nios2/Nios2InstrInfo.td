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

def simm16     : Operand<i32> {
  let DecoderMethod= "DecodeSimm16";
}

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{ return isInt<16>(N->getSExtValue()); }]>;

// Custom return SDNode
def Nios2Ret : SDNode<"Nios2ISD::Ret", SDTNone,
    [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic and logical instructions with 2 registers and 16-bit immediate
// value.
multiclass ArithLogicRegImm16<bits<6> op, string mnemonic, SDNode opNode,
                              Operand immOp, PatLeaf immType>:
           CommonInstr_I_F2I16<op, (outs CPURegs:$rB),
	                       (ins CPURegs:$rA, immOp:$imm),
                               !strconcat(mnemonic, "\t$rB, $rA, $imm"),
                               [(set CPURegs:$rB,
			         (opNode CPURegs:$rA, immType:$imm))],
                               IIAlu>;

// Arithmetic and logical instructions with 3 register operands.
// Defines R1 and R2 instruction at the same time.
multiclass ArithLogicReg<bits<6> opx, string mnemonic,
                         SDNode opNode>:
  CommonInstr_R_F3X6<opx, (outs CPURegs:$rC),
                     (ins CPURegs:$rA, CPURegs:$rB),
                     !strconcat(mnemonic, "\t$rC, $rA, $rB"),
                     [(set CPURegs:$rC, (opNode CPURegs:$rA, CPURegs:$rB))],
                     IIAlu>;

multiclass Return<bits<6> opx, dag outs, dag ins, string mnemonic> {
  let rB = 0, rC = 0,
      isReturn = 1,
      isCodeGenOnly = 1,
      hasCtrlDep = 1,
      hasExtraSrcRegAllocReq = 1 in {
    defm NAME# : CommonInstr_R_F3X6<opx, outs, ins, mnemonic, [], IIBranch>;
  }
}

//===----------------------------------------------------------------------===//
// Nios2 Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic instructions operating on registers.
let isCommutable = 1 ,
    isReMaterializable = 1 in {
  defm ADD    : ArithLogicReg<0x31, "add",    add>;
  defm AND    : ArithLogicReg<0x0e, "and",    and>;
  defm OR     : ArithLogicReg<0x16, "or",     or>;
  defm XOR    : ArithLogicReg<0x1e, "xor",    xor>;
  defm MUL    : ArithLogicReg<0x27, "mul",    mul>;
}

let isReMaterializable = 1 in {
  defm SUB    : ArithLogicReg<0x39, "sub",    sub>;
}

defm DIVU : ArithLogicReg<0x24, "divu",   udiv>;
defm DIV  : ArithLogicReg<0x25, "div",    sdiv>;

defm SLL : ArithLogicReg<0x13, "sll",  shl>;
defm SRL : ArithLogicReg<0x1b, "srl",  srl>;
defm SRA : ArithLogicReg<0x3b, "sra",  sra>;

/// Arithmetic Instructions (ALU Immediate)
defm ADDI  : ArithLogicRegImm16<0x04, "addi",  add, simm16, immSExt16>;

// Returns:
defm RET  : Return<0x05, (outs), (ins CPURegs:$rA), "ret">;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// Return RA.
let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1 in
def RetRA : Nios2Pseudo<(outs), (ins), "", [(Nios2Ret)]>;
