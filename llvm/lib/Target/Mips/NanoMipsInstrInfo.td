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

// Arithmetic and logical instructions with 3 register operands.
class ArithLogicRNM<string opstr, RegisterOperand RO,
                    SDPatternOperator OpNode = null_frag>:
  InstNM<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (OpNode RO:$rs, RO:$rt))]>, InstSize32;

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

class LoadNM<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
             InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr> :
             LoadMemory<opstr, RO, mem, OpNode, Itin, Addr> {
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
}

class StoreNM<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
              InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr,
              DAGOperand MO = mem> :
              StoreMemory<opstr, RO, MO, OpNode, Itin, Addr> {
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
}

class CallNM<string opstr,
             SDPatternOperator OpNode = null_frag,
             SDPatternOperator imm_type = null_frag> :
  InstNM<(outs), (ins calltarget:$addr), !strconcat(opstr, "\t$addr"),
         [(OpNode imm_type:$addr)]> {
  let Namespace = "Mips";
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
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

def ADDu_NM : ArithLogicRNM<"addu", GPR32NMOpnd, add>;
def ADDiu_NM : ArithLogicINM<"addiu", simm16_relaxed, GPR32NMOpnd,
                             imm32SExt16, add>;
def AND_NM : ArithLogicRNM<"and", GPR32NMOpnd, and>;
def LW_NM : StdMMR6Rel, LoadNM<"lw", GPR32NMOpnd, load>, ISA_NANOMIPS;
def OR_NM : ArithLogicRNM<"or", GPR32NMOpnd, or>;
def SUBu_NM : ArithLogicRNM<"subu", GPR32NMOpnd, sub>;
def SW_NM : StoreNM<"sw", GPR32NMOpnd, store>, ISA_NANOMIPS;

def JRC_NM : IndirectBranchNM<"jrc", GPR32NMOpnd>;

let hasDelaySlot = 0 in
  def PseudoReturnNM : PseudoReturnBase<GPR32NMOpnd>, ISA_NANOMIPS;

def BALC_NM : CallNM<"balc", MipsJmpLink, tglobaladdr>, InstSize32;

def Li_NM : RegImmNM<"li", simm32_relaxed, GPR32NMOpnd>, InstSize48;

def ADJCALLSTACKDOWN_NM : AdjustStack<callseq_start>;
def ADJCALLSTACKUP_NM   : AdjustStack<callseq_end>;

}

def imm32_NM : IntImmLeaf<i32, [{
    return (uint32_t)Imm.getZExtValue() == (uint64_t)Imm.getZExtValue();
}]>;
def : MipsPat<(i32 imm32_NM:$imm), (Li_NM imm:$imm)>, ISA_NANOMIPS;

def : MipsPat<(MipsJmpLink (i32 texternalsym:$dst)),
              (BALC_NM texternalsym:$dst)>, ISA_NANOMIPS;

def : MipsPat<(MipsFullAddr tglobaladdr:$in),
              (Li_NM tglobaladdr:$in)>, ISA_NANOMIPS;


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
