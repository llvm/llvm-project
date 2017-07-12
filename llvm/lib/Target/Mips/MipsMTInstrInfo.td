//===-- MipsMTInstrInfo.td - Mips MT Instruction Infos -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Encodings
//===----------------------------------------------------------------------===//

class DMT_ENC : COP0_MFMC0_MT<FIELD5_1_DMT_EMT, FIELD5_2_DMT_EMT,
                              OPCODE_SC_DMT>;

class EMT_ENC : COP0_MFMC0_MT<FIELD5_1_DMT_EMT, FIELD5_2_DMT_EMT,
                              OPCODE_SC_EMT>;

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Descriptions
//===----------------------------------------------------------------------===//

class MT_1R_DESC_BASE<string instr_asm, InstrItinClass Itin = NoItinerary> {
  dag OutOperandList = (outs GPR32Opnd:$rt);
  dag InOperandList = (ins);
  string AsmString = !strconcat(instr_asm, "\t$rt");
  list<dag> Pattern = [];
  InstrItinClass Itinerary = Itin;
}

class DMT_DESC : MT_1R_DESC_BASE<"dmt", II_DMT>;

class EMT_DESC : MT_1R_DESC_BASE<"emt", II_EMT>;

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Definitions
//===----------------------------------------------------------------------===//
let hasSideEffects = 1, isNotDuplicable = 1,
    AdditionalPredicates = [NotInMicroMips] in {
  def DMT : DMT_ENC, DMT_DESC, ASE_MT;

  def EMT : EMT_ENC, EMT_DESC, ASE_MT;
}

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Definitions
//===----------------------------------------------------------------------===//

let AdditionalPredicates = [NotInMicroMips] in {
  def : MipsInstAlias<"dmt", (DMT ZERO), 1>, ASE_MT;

  def : MipsInstAlias<"emt", (EMT ZERO), 1>, ASE_MT;
}
