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

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {

def ADDu_NM : ArithLogicRNM<"addu", GPR32Opnd, add>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def JRC_NM : IndirectBranch<"jrc", GPR32Opnd>, ISA_NANOMIPS;

let hasDelaySlot = 0 in
  def PseudoReturnNM : PseudoReturnBase<GPR32Opnd>, ISA_NANOMIPS;

}

