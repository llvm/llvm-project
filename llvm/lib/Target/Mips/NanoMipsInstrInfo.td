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
//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {

def ADDu_NM : ArithLogicRNM<"addu", GPR32NMOpnd, add>, ISA_NANOMIPS;
def ADDiu_NM : ArithLogicINM<"addiu", simm16_relaxed, GPR32NMOpnd,
                             imm32SExt16, add>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def JRC_NM : IndirectBranch<"jrc", GPR32NMOpnd>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def PseudoReturnNM : PseudoReturnBase<GPR32NMOpnd>, ISA_NANOMIPS;

def Li_NM : RegImmNM<"li", simm32_relaxed, GPR32NMOpnd>, ISA_NANOMIPS;
}

def imm32_NM : IntImmLeaf<i32, [{
    return (uint32_t)Imm.getZExtValue() == (uint64_t)Imm.getZExtValue();
}]>;
def : MipsPat<(i32 imm32_NM:$imm), (Li_NM imm:$imm)>, ISA_NANOMIPS;
