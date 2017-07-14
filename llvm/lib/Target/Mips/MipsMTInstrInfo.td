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
                              OPCODE_SC_D>;

class EMT_ENC : COP0_MFMC0_MT<FIELD5_1_DMT_EMT, FIELD5_2_DMT_EMT,
                              OPCODE_SC_E>;

class DVPE_ENC : COP0_MFMC0_MT<FIELD5_1_2_DVPE_EVPE, FIELD5_1_2_DVPE_EVPE,
                               OPCODE_SC_D>;

class EVPE_ENC : COP0_MFMC0_MT<FIELD5_1_2_DVPE_EVPE, FIELD5_1_2_DVPE_EVPE,
                               OPCODE_SC_E>;

class FORK_ENC : SPECIAL3_MT_FORK;

class YIELD_ENC : SPECIAL3_MT_YIELD;

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

class FORK_DESC {
  dag OutOperandList = (outs GPR32Opnd:$rs, GPR32Opnd:$rd);
  dag InOperandList = (ins GPR32Opnd:$rt);
  string AsmString = "fork\t$rd, $rs, $rt";
  list<dag> Pattern = [];
  InstrItinClass Itinerary = II_FORK;
}

class YIELD_DESC {
  dag OutOperandList = (outs GPR32Opnd:$rd);
  dag InOperandList = (ins GPR32Opnd:$rs);
  string AsmString = "yield\t$rd, $rs";
  list<dag> Pattern = [];
  InstrItinClass Itinerary = II_YIELD;
}

class DMT_DESC : MT_1R_DESC_BASE<"dmt", II_DMT>;

class EMT_DESC : MT_1R_DESC_BASE<"emt", II_EMT>;

class DVPE_DESC : MT_1R_DESC_BASE<"dvpe", II_DVPE>;

class EVPE_DESC : MT_1R_DESC_BASE<"evpe", II_EVPE>;

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Definitions
//===----------------------------------------------------------------------===//
let hasSideEffects = 1, isNotDuplicable = 1,
    AdditionalPredicates = [NotInMicroMips] in {
  def DMT : DMT_ENC, DMT_DESC, ASE_MT;

  def EMT : EMT_ENC, EMT_DESC, ASE_MT;

  def DVPE : DVPE_ENC, DVPE_DESC, ASE_MT;

  def EVPE : EVPE_ENC, EVPE_DESC, ASE_MT;

  def FORK : FORK_ENC, FORK_DESC, ASE_MT;

  def YIELD : YIELD_ENC, YIELD_DESC, ASE_MT;
}

//===----------------------------------------------------------------------===//
// MIPS MT Instruction Definitions
//===----------------------------------------------------------------------===//

let AdditionalPredicates = [NotInMicroMips] in {
  def : MipsInstAlias<"dmt", (DMT ZERO), 1>, ASE_MT;

  def : MipsInstAlias<"emt", (EMT ZERO), 1>, ASE_MT;

  def : MipsInstAlias<"dvpe", (DVPE ZERO), 1>, ASE_MT;

  def : MipsInstAlias<"evpe", (EVPE ZERO), 1>, ASE_MT;

  def : MipsInstAlias<"yield $rs", (YIELD ZERO, GPR32Opnd:$rs), 1>, ASE_MT;
}
