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
		    SDPatternOperator OpNode = null_frag> :
      ArithLogicR<opstr, RO, 0, NoItinerary, OpNode> {
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
}

// Arithmetic and logical instructions with 2 register operands.
class ArithLogicINM<string opstr, Operand Od, RegisterOperand RO,
                    SDPatternOperator imm_type = null_frag,
                    SDPatternOperator OpNode = null_frag> :
  ArithLogicI<opstr, Od, RO, NoItinerary, imm_type, OpNode> {
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
}

// Instructions with 1 register (dest) and 1 immediate operand.
class RegImmNM<string opstr, Operand Od, RegisterOperand RO,
               InstrItinClass Itin = NoItinerary,
               SDPatternOperator imm_type = null_frag,
               SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RO:$rt), (ins Od:$imm), !strconcat(opstr, "\t$rt, $imm"),
         [(set RO:$rt, (OpNode imm_type:$imm))], Itin, FrmI, opstr> {
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
}

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
  InstSE<(outs), (ins calltarget:$addr), !strconcat(opstr, "\t$addr"),
         [(OpNode imm_type:$addr)], NoItinerary, FrmI, opstr> {
  let Namespace = "Mips";
  let DecoderNamespace = "NanoMips";
  let EncodingPredicates = [HasNanoMips];
  let DecoderMethod = "DecodeJumpTarget";
  let isCall = 1;
  let isCTI = 1;
  let Defs = [RA_NM];
}

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {

def ADDu_NM : ArithLogicRNM<"addu", GPR32NMOpnd, add>, ISA_NANOMIPS;
def ADDiu_NM : ArithLogicINM<"addiu", simm16_relaxed, GPR32NMOpnd,
                             imm32SExt16, add>, ISA_NANOMIPS;
def AND_NM : ArithLogicRNM<"and", GPR32NMOpnd, and>, ISA_NANOMIPS;
def LW_NM : StdMMR6Rel, LoadNM<"lw", GPR32NMOpnd, load>, ISA_NANOMIPS;
def OR_NM : ArithLogicRNM<"or", GPR32NMOpnd, or>, ISA_NANOMIPS;
def SUBu_NM : ArithLogicRNM<"subu", GPR32NMOpnd, sub>, ISA_NANOMIPS;
def SW_NM : StoreNM<"sw", GPR32NMOpnd, store>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def JRC_NM : IndirectBranch<"jrc", GPR32NMOpnd>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def PseudoReturnNM : PseudoReturnBase<GPR32NMOpnd>, ISA_NANOMIPS;

def BALC_NM : CallNM<"balc", MipsJmpLink, tglobaladdr>, ISA_NANOMIPS;

def Li_NM : RegImmNM<"li", simm32_relaxed, GPR32NMOpnd>, ISA_NANOMIPS;

let Defs = [SP_NM], Uses = [SP_NM], hasSideEffects = 1, hasNoSchedulingInfo = 1 in {
  def ADJCALLSTACKDOWN_NM : MipsPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                       [(callseq_start timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP_NM : MipsPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                     [(callseq_end timm:$amt1, timm:$amt2)]>;
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
