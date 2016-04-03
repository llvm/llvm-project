//=-  MicroMips64r6InstrInfo.td - Instruction Information  -*- tablegen -*- -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes MicroMips64r6 instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class DAUI_MMR6_ENC : DAUI_FM_MMR6;
class DAHI_MMR6_ENC : POOL32I_ADD_IMM_FM_MMR6<0b10001>;
class DATI_MMR6_ENC : POOL32I_ADD_IMM_FM_MMR6<0b10000>;
class DEXT_MMR6_ENC : POOL32S_EXTBITS_FM_MMR6<0b101100>;
class DEXTM_MMR6_ENC : POOL32S_EXTBITS_FM_MMR6<0b100100>;
class DEXTU_MMR6_ENC : POOL32S_EXTBITS_FM_MMR6<0b010100>;
class DALIGN_MMR6_ENC : POOL32S_DALIGN_FM_MMR6;
class DDIV_MM64R6_ENC : POOL32A_DIVMOD_FM_MMR6<"ddiv", 0b100011000>;
class DMOD_MM64R6_ENC : POOL32A_DIVMOD_FM_MMR6<"dmod", 0b101011000>;
class DDIVU_MM64R6_ENC : POOL32A_DIVMOD_FM_MMR6<"ddivu", 0b110011000>;
class DMODU_MM64R6_ENC : POOL32A_DIVMOD_FM_MMR6<"dmodu", 0b111011000>;
class DINSU_MM64R6_ENC : POOL32S_EXTBITS_FM_MMR6<0b110100>;
class DINSM_MM64R6_ENC : POOL32S_EXTBITS_FM_MMR6<0b000100>;
class DINS_MM64R6_ENC : POOL32S_EXTBITS_FM_MMR6<0b001100>;
class DMTC0_MM64R6_ENC : POOL32S_DMFTC0_FM_MMR6<"dmtc0", 0b01011>;
class DMTC1_MM64R6_ENC : POOL32F_MFTC1_FM_MMR6<"dmtc1", 0b10110000>;
class DMTC2_MM64R6_ENC : POOL32A_MFTC2_FM_MMR6<"dmtc2", 0b0111110100>;
class DMFC0_MM64R6_ENC : POOL32S_DMFTC0_FM_MMR6<"dmfc0", 0b00011>;
class DMFC1_MM64R6_ENC : POOL32F_MFTC1_FM_MMR6<"dmfc1", 0b10010000>;
class DMFC2_MM64R6_ENC : POOL32A_MFTC2_FM_MMR6<"dmfc2", 0b0110110100>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class DAUI_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $imm");
  list<dag> Pattern = [];
}
class DAUI_MMR6_DESC : DAUI_MMR6_DESC_BASE<"daui", GPR64Opnd>;

class DAHI_DATI_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins GPROpnd:$rt, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  string Constraints = "$rs = $rt";
}
class DAHI_MMR6_DESC : DAHI_DATI_DESC_BASE<"dahi", GPR64Opnd>;
class DATI_MMR6_DESC : DAHI_DATI_DESC_BASE<"dati", GPR64Opnd>;

class EXTBITS_DESC_BASE<string instr_asm, RegisterOperand RO, Operand PosOpnd,
                        Operand SizeOpnd, SDPatternOperator Op = null_frag>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs RO:$rt);
  dag InOperandList = (ins RO:$rs, PosOpnd:$pos, SizeOpnd:$size);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $pos, $size");
  list<dag> Pattern = [(set RO:$rt, (Op RO:$rs, imm:$pos, imm:$size))];
  InstrItinClass Itinerary = II_EXT;
  Format Form = FrmR;
  string BaseOpcode = instr_asm;
}
// TODO: Add 'pos + size' constraint check to dext* instructions
//       DEXT: 0 < pos + size <= 63
//       DEXTM, DEXTU: 32 < pos + size <= 64
class DEXT_MMR6_DESC : EXTBITS_DESC_BASE<"dext", GPR64Opnd, uimm5_report_uimm6,
                                         uimm5_plus1, MipsExt>;
class DEXTM_MMR6_DESC : EXTBITS_DESC_BASE<"dextm", GPR64Opnd, uimm5,
                                          uimm5_plus33, MipsExt>;
class DEXTU_MMR6_DESC : EXTBITS_DESC_BASE<"dextu", GPR64Opnd, uimm5_plus32,
                                          uimm5_plus1, MipsExt>;

class DALIGN_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd> : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$bp);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $bp");
  list<dag> Pattern = [];
}

class DALIGN_MMR6_DESC : DALIGN_DESC_BASE<"dalign", GPR64Opnd, uimm3>;

class DDIV_MM64R6_DESC : ArithLogicR<"ddiv", GPR32Opnd>;
class DMOD_MM64R6_DESC : ArithLogicR<"dmod", GPR32Opnd>;
class DDIVU_MM64R6_DESC : ArithLogicR<"ddivu", GPR32Opnd>;
class DMODU_MM64R6_DESC : ArithLogicR<"dmodu", GPR32Opnd>;

class DINSU_MM64R6_DESC : InsBase<"dinsu", GPR64Opnd, uimm5_plus32,
                                  uimm5_inssize_plus1, MipsIns>;
class DINSM_MM64R6_DESC : InsBase<"dinsm", GPR64Opnd, uimm5, uimm_range_2_64>;
class DINS_MM64R6_DESC : InsBase<"dins", GPR64Opnd, uimm5, uimm5_inssize_plus1,
                                 MipsIns>;
class DMTC0_MM64R6_DESC : MTC0_MMR6_DESC_BASE<"dmtc0", COP0Opnd, GPR64Opnd>;
class DMTC1_MM64R6_DESC : MTC1_MMR6_DESC_BASE<"dmtc1", FGR64Opnd, GPR64Opnd,
                                              II_DMTC1, bitconvert>;
class DMTC2_MM64R6_DESC : MTC2_MMR6_DESC_BASE<"dmtc2", COP2Opnd, GPR64Opnd>;

class DMFC0_MM64R6_DESC : MFC0_MMR6_DESC_BASE<"dmfc0", GPR64Opnd, COP0Opnd>;
class DMFC1_MM64R6_DESC : MFC1_MMR6_DESC_BASE<"dmfc1", GPR64Opnd, FGR64Opnd,
                                              II_DMFC1, bitconvert>;
class DMFC2_MM64R6_DESC : MFC2_MMR6_DESC_BASE<"dmfc2", GPR64Opnd, COP2Opnd>;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "MicroMipsR6" in {
  def DAUI_MM64R6 : StdMMR6Rel, DAUI_MMR6_DESC, DAUI_MMR6_ENC, ISA_MICROMIPS64R6;
  def DAHI_MM64R6 : StdMMR6Rel, DAHI_MMR6_DESC, DAHI_MMR6_ENC, ISA_MICROMIPS64R6;
  def DATI_MM64R6 : StdMMR6Rel, DATI_MMR6_DESC, DATI_MMR6_ENC, ISA_MICROMIPS64R6;
  def DEXT_MM64R6 : StdMMR6Rel, DEXT_MMR6_DESC, DEXT_MMR6_ENC,
                    ISA_MICROMIPS64R6;
  def DEXTM_MM64R6 : StdMMR6Rel, DEXTM_MMR6_DESC, DEXTM_MMR6_ENC,
                     ISA_MICROMIPS64R6;
  def DEXTU_MM64R6 : StdMMR6Rel, DEXTU_MMR6_DESC, DEXTU_MMR6_ENC,
                     ISA_MICROMIPS64R6;
  def DALIGN_MM64R6 : StdMMR6Rel, DALIGN_MMR6_DESC, DALIGN_MMR6_ENC,
                      ISA_MICROMIPS64R6;
  def DDIV_MM64R6 : R6MMR6Rel, DDIV_MM64R6_DESC, DDIV_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DMOD_MM64R6 : R6MMR6Rel, DMOD_MM64R6_DESC, DMOD_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DDIVU_MM64R6 : R6MMR6Rel, DDIVU_MM64R6_DESC, DDIVU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DMODU_MM64R6 : R6MMR6Rel, DMODU_MM64R6_DESC, DMODU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DINSU_MM64R6: R6MMR6Rel, DINSU_MM64R6_DESC, DINSU_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DINSM_MM64R6: R6MMR6Rel, DINSM_MM64R6_DESC, DINSM_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DINS_MM64R6: R6MMR6Rel, DINS_MM64R6_DESC, DINS_MM64R6_ENC,
                   ISA_MICROMIPS64R6;
  def DMTC0_MM64R6 : StdMMR6Rel, DMTC0_MM64R6_ENC, DMTC0_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DMTC1_MM64R6 : StdMMR6Rel, DMTC1_MM64R6_DESC, DMTC1_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DMTC2_MM64R6 : StdMMR6Rel, DMTC2_MM64R6_ENC, DMTC2_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DMFC0_MM64R6 : StdMMR6Rel, DMFC0_MM64R6_ENC, DMFC0_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DMFC1_MM64R6 : StdMMR6Rel, DMFC1_MM64R6_DESC, DMFC1_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DMFC2_MM64R6 : StdMMR6Rel, DMFC2_MM64R6_ENC, DMFC2_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
}

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : MipsInstAlias<"dmtc0 $rt, $rd",
                    (DMTC0_MM64R6 COP0Opnd:$rd, GPR64Opnd:$rt, 0), 0>;
def : MipsInstAlias<"dmfc0 $rt, $rd",
                    (DMFC0_MM64R6 GPR64Opnd:$rt, COP0Opnd:$rd, 0), 0>,
                    ISA_MICROMIPS64R6;
