//=- NanoMipsInstrInfo.td - NanoMips Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes NanoMips instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

def MipsFullAddr : SDNode<"MipsISD::FullAddr", SDTIntUnaryOp>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

// 32-bit arithmetic and logical instructions with 3 register operands.
class ArithLoginR32<string opstr, RegisterOperand RO,
                    SDPatternOperator OpNode = null_frag>:
  InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (OpNode RO:$rs, RO:$rt))]>, InstSize32;

// 16-bit arithmetic and logical instructions with 2 register operands.
class ArithLogicR16<string opstr, RegisterOperand RO,
                    SDPatternOperator OpNode = null_frag>:
    InstNM<(outs RO:$rt), (ins RO:$rs), !strconcat(opstr, "\t$rt, $rs"),
           [(set RO:$rt, (OpNode RO:$rs))]>, InstSize16;

// Arithmetic and logical instructions with 2 register operands and immediate.
class ArithLogicINM<string opstr, Operand Od, RegisterOperand RO,
                    SDPatternOperator imm_type = null_frag,
                    SDPatternOperator OpNode = null_frag> :
  InstNM<(outs RO:$rt), (ins RO:$rs, Od:$imm16),
         !strconcat(opstr, "\t$rt, $rs, $imm16"),
         [(set RO:$rt, (OpNode RO:$rs, imm_type:$imm16))]>, InstSize32 {
  let isReMaterializable = 1;
}

// Instructions with 1 register (dest) and 1 immediate operand.
class RegImmNM<string opstr, Operand Od, RegisterOperand RO,
               SDPatternOperator imm_type = null_frag,
               SDPatternOperator OpNode = null_frag> :
  InstNM<(outs RO:$rt), (ins Od:$imm), !strconcat(opstr, "\t$rt, $imm"),
         [(set RO:$rt, (OpNode imm_type:$imm))]>;

class LoadMemoryNM<string opstr, DAGOperand RO, DAGOperand MO,
                 SDPatternOperator OpNode = null_frag,
                 ComplexPattern Addr = addr> :
    InstNM<(outs RO:$rt), (ins MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
           [(set RO:$rt, (OpNode Addr:$addr))]> {
  let DecoderMethod = "DecodeMem";
  let canFoldAsLoad = 1;
  let mayLoad = 1;
  string BaseOpcode = opstr;
}

class StoreMemoryNM<string opstr, DAGOperand RO, DAGOperand MO,
                    SDPatternOperator OpNode = null_frag,
                    ComplexPattern Addr = addr> :
    InstNM<(outs), (ins RO:$rt, MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
           [(OpNode RO:$rt, Addr:$addr)]> {
  let DecoderMethod = "DecodeMem";
  let mayStore = 1;
  string BaseOpcode = opstr;
}

class CallNM<string opstr,
             SDPatternOperator OpNode = null_frag,
             SDPatternOperator imm_type = null_frag> :
  InstNM<(outs), (ins calltarget:$addr), !strconcat(opstr, "\t$addr"),
         [(OpNode imm_type:$addr)]> {
  let DecoderMethod = "DecodeJumpTarget";
  let isCall = 1;
  let isCTI = 1;
  let Defs = [RA_NM];
}

// Indirect branch instruction.
class IndirectBranchNM<string opstr, RegisterOperand RO,
                       SDPatternOperator operator = null_frag>:
    InstNM<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"),
           [(operator RO:$rs)]>, InstSize16 {
  let isTerminator = 1;
  let isBarrier = 1;
  let isCTI = 1;
  let isBranch = 1;
  let isIndirectBranch = 1;
}

class AdjustStack<SDPatternOperator operator> :
    PseudoInstNM<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                 [(operator timm:$amt1, timm:$amt2)]> {
  let Defs = [SP_NM];
  let Uses = [SP_NM];
  let hasSideEffects = 1;
  let hasNoSchedulingInfo = 1;
}

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {

def SLLV_NM  : ArithLoginR32<"sllv", GPR32NMOpnd, shl>;
def SRLV_NM  : ArithLoginR32<"srlv", GPR32NMOpnd, srl>;
def SRAV_NM  : ArithLoginR32<"srav", GPR32NMOpnd, sra>;
def ROTRV_NM : ArithLoginR32<"rotrv", GPR32NMOpnd, rotr>;
def ADD_NM   : ArithLoginR32<"add", GPR32NMOpnd>;
def ADDu_NM  : ArithLoginR32<"addu", GPR32NMOpnd, add>;
def ADDiu_NM : ArithLogicINM<"addiu", simm16_relaxed, GPR32NMOpnd,
                             imm32SExt16, add>;
def SUB_NM   : ArithLoginR32<"sub", GPR32NMOpnd>;
def SUBu_NM  : ArithLoginR32<"subu", GPR32NMOpnd, sub>;

def AND_NM : ArithLoginR32<"and", GPR32NMOpnd, and>;
def OR_NM  : ArithLoginR32<"or", GPR32NMOpnd, or>;
def XOR_NM : ArithLoginR32<"xor", GPR32NMOpnd, xor>;
def NOR_NM : ArithLoginR32<"nor", GPR32NMOpnd>;

def NOT_NM   : ArithLogicR16<"not", GPR3Opnd, not>;
def XOR16_NM : ArithLogicR16<"xor16", GPR3Opnd>;
def AND16_NM : ArithLogicR16<"and16", GPR3Opnd>;
def OR16_NM  : ArithLogicR16<"or16", GPR3Opnd>;

def LW_NM : LoadMemoryNM<"lw", GPR32NMOpnd, mem, load>, InstSize32;
def SW_NM : StoreMemoryNM<"sw", GPR32NMOpnd, mem, store>, InstSize32;

def JRC_NM : IndirectBranchNM<"jrc", GPR32NMOpnd>;

def BALC_NM : CallNM<"balc", MipsJmpLink, tglobaladdr>, InstSize32;

def Li_NM : RegImmNM<"li", simm32_relaxed, GPR32NMOpnd>, InstSize48;

def ADJCALLSTACKDOWN_NM : AdjustStack<callseq_start>;
def ADJCALLSTACKUP_NM   : AdjustStack<callseq_end>;

// Return instruction is matched as RetRA, then expanded into PseudoReturnNM
// after register allocation. Finally, MipsAsmPrinter expands this into JRC_NM.
def PseudoReturnNM : PseudoInstNM<(outs), (ins GPR32NMOpnd:$rs), []> {
  let isTerminator = 1;
  let isBarrier = 1;
  let isReturn = 1;
  let isCodeGenOnly = 1;
  let hasCtrlDep = 1;
  let hasExtraSrcRegAllocReq = 1;
  bit isCTI = 1;
}

}

def imm32_NM : IntImmLeaf<i32, [{
    return (uint32_t)Imm.getZExtValue() == (uint64_t)Imm.getZExtValue();
}]>;
def : MipsPat<(i32 imm32_NM:$imm), (Li_NM imm:$imm)>, ISA_NANOMIPS;

def : MipsPat<(MipsJmpLink (i32 texternalsym:$dst)),
              (BALC_NM texternalsym:$dst)>, ISA_NANOMIPS;

def : MipsPat<(MipsFullAddr tglobaladdr:$in),
              (Li_NM tglobaladdr:$in)>, ISA_NANOMIPS;

def : MipsPat<(not (or GPR32NM:$rs, GPR32NM:$rt)),
              (NOR_NM GPR32NM:$rs, GPR32NM:$rt)>, ISA_NANOMIPS;


//===----------------------------------------------------------------------===//
//
// Jump and Branch Instructions.
//
//===----------------------------------------------------------------------===//

// NanoMips has only 'compact' jumps and branches, with no delay slots.
// Described using IsntSE class for the moment, should use specific
// NanoMips instruction formats later to get correct sizes.  Some
// branch instructions exist in multiple different sizes, and should
// likely be distinct instructions later.

// This class is missing instruction size, because these instructions can be
// encoded in formats of different size (eg. BEQC16 and BEQC32).
class CBranchNM<string opstr, DAGOperand opnd, PatFrag cond_op,
                RegisterOperand RO>
    : InstNM<(outs), (ins RO:$rs, RO:$rt, opnd:$offset),
             !strconcat(opstr, "\t$rs, $rt, $offset"),
             [(brcond (i32 (cond_op RO:$rs, RO:$rt)), bb:$offset)]> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let Defs = [AT_NM];
  bit isCTI = 1;
}

class CBranchImmNM<string opstr, DAGOperand opnd, PatFrag cond_op,
                   RegisterOperand RO,
                   Operand imm_type, SDPatternOperator ImmOp>
    : InstSize32,
      InstNM<(outs), (ins RO:$rt, imm_type:$u, opnd:$offset),
             !strconcat(opstr, "\t$rt, $u, $offset"),
             [(brcond (i32 (cond_op RO:$rt, ImmOp:$u)), bb:$offset)]> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let Defs = [AT_NM];
  bit isCTI = 1;
}

class CBranchZeroNM<string opstr, DAGOperand opnd, PatFrag cond_op,
                    RegisterOperand RO>
    : InstSize16,
      InstNM<(outs), (ins RO:$rs, opnd:$offset),
             !strconcat(opstr, "\t$rs, $offset"),
             [(brcond (i32 (cond_op RO:$rs, 0)), bb:$offset)]> {
  let isBranch = 1;
  let isTerminator = 1;
  let Defs = [AT_NM];
  bit isCTI = 1;
}

let isCodeGenOnly = 1, hasNoSchedulingInfo = 1,
    hasDelaySlot = 0 in {

// TODO: BBEQZ/BBNEZ bit-test conditional branch instructions
def BEQC_NM  : CBranchNM<"beqc", brtarget, seteq, GPR32NMOpnd>, InstSize32;
def BEQIC_NM : CBranchImmNM<"beqic", brtarget, seteq, GPR32NMOpnd, uimm7, immZExt7>;
def BEQZC_NM : CBranchZeroNM<"beqzc", brtarget, seteq, GPR32NMOpnd>;

def BGEC_NM   : CBranchNM<"bgec", brtarget, setge, GPR32NMOpnd>, InstSize32;
def BGEIC_NM  : CBranchImmNM<"bgeic", brtarget, setge, GPR32NMOpnd, uimm7, immZExt7>;
def BGEIUC_NM : CBranchImmNM<"bgeiuc", brtarget, setuge, GPR32NMOpnd, uimm7, immZExt7>;
def BGEUC_NM  : CBranchNM<"bgeuc", brtarget, setuge, GPR32NMOpnd>, InstSize32;

def BLTC_NM   : CBranchNM<"bltc", brtarget, setlt, GPR32NMOpnd>, InstSize32;
def BLTIC_NM  : CBranchImmNM<"bltic", brtarget, setlt, GPR32NMOpnd, uimm7, immZExt7>;
def BLTIUC_NM : CBranchImmNM<"bltiuc", brtarget, setult, GPR32NMOpnd, uimm7, immZExt7>;
def BLTUC_NM  : CBranchNM<"bltuc", brtarget, setult, GPR32NMOpnd>, InstSize32;

def BNEC_NM  : CBranchNM<"bnec", brtarget, setne, GPR32NMOpnd>, InstSize32;
def BNEIC_NM : CBranchImmNM<"bnec", brtarget, setne, GPR32NMOpnd, uimm7, immZExt7>;
def BNEZC_NM : CBranchZeroNM<"beqzc", brtarget, setne, GPR32NMOpnd>;


// Unconditional branch
// TODO: native nanomips formats
def BC_NM       : JumpFJ<jmptarget, "bc", br, bb, "b">, IsBranch, ISA_NANOMIPS;

//def JAL  : MMRel, JumpLink<"jal", calltarget>, FJ<3>, ISA_MIPS1;

}

// Reversed branch conditions:
def : Pat<(brcond (i32 (setugt GPR32NM:$lhs, GPR32NM:$rhs)), bb:$dst),
          (BLTUC_NM GPR32NM:$rhs, GPR32NM:$lhs, bb:$dst)>, ISA_NANOMIPS;
def : Pat<(brcond (i32 (setgt GPR32NM:$lhs, GPR32NM:$rhs)), bb:$dst),
          (BLTC_NM GPR32NM:$rhs, GPR32NM:$lhs, bb:$dst)>, ISA_NANOMIPS;
def : Pat<(brcond (i32 (setule GPR32NM:$lhs, GPR32NM:$rhs)), bb:$dst),
          (BGEUC_NM GPR32NM:$rhs, GPR32NM:$lhs, bb:$dst)>, ISA_NANOMIPS;
def : Pat<(brcond (i32 (setle GPR32NM:$lhs, GPR32NM:$rhs)), bb:$dst),
          (BGEC_NM GPR32NM:$rhs, GPR32NM:$lhs, bb:$dst)>, ISA_NANOMIPS;
