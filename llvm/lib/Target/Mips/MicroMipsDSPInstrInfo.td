//===- MicroMipsDSPInstrInfo.td - Micromips DSP instructions -*- tablegen *-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes MicroMips DSP instructions.
//
//===----------------------------------------------------------------------===//

// Instruction encoding.
class ADDQ_PH_MM_ENC : POOL32A_3R_FMT<"addq.ph", 0b00000001101>;
class ADDQ_S_PH_MM_ENC : POOL32A_3R_FMT<"addq_s.ph", 0b10000001101>;
class ADDQ_S_W_MM_ENC : POOL32A_3RB0_FMT<"addq_s.w", 0b1100000101>;
class ADDQH_PH_MMR2_ENC : POOL32A_3R_FMT<"addqh.ph", 0b00001001101>;
class ADDQH_R_PH_MMR2_ENC : POOL32A_3R_FMT<"addqh_r.ph", 0b10001001101>;
class ADDQH_W_MMR2_ENC: POOL32A_3R_FMT<"addqh.w", 0b00010001101>;
class ADDQH_R_W_MMR2_ENC : POOL32A_3R_FMT<"addqh_r.w", 0b10010001101>;
class ADDU_PH_MMR2_ENC : POOL32A_3R_FMT<"addu.ph", 0b00100001101>;
class ADDU_S_PH_MMR2_ENC : POOL32A_3R_FMT<"addu_s.ph", 0b10100001101>;
class ADDU_QB_MM_ENC : POOL32A_3R_FMT<"addu.qb", 0b00011001101>;
class ADDU_S_QB_MM_ENC : POOL32A_3R_FMT<"addu_s.qb", 0b10011001101>;
class ADDUH_QB_MMR2_ENC : POOL32A_3R_FMT<"adduh.qb", 0b00101001101>;
class ADDUH_R_QB_MMR2_ENC : POOL32A_3R_FMT<"adduh_r.qb", 0b10101001101>;
class ADDSC_MM_ENC : POOL32A_3RB0_FMT<"addsc", 0b1110000101>;
class ADDWC_MM_ENC : POOL32A_3RB0_FMT<"addwc", 0b1111000101>;
class DPA_W_PH_MMR2_ENC : POOL32A_2RAC_FMT<"dpa.w.ph", 0b00000010>;
class DPAQ_S_W_PH_MM_ENC : POOL32A_2RAC_FMT<"dpaq_s.w.ph", 0b00001010>;
class DPAQ_SA_L_W_MM_ENC : POOL32A_2RAC_FMT<"dpaq_sa.l.w", 0b01001010>;
class DPAQX_S_W_PH_MMR2_ENC : POOL32A_2RAC_FMT<"dpaqx_s.w.ph", 0b10001010>;
class DPAQX_SA_W_PH_MMR2_ENC : POOL32A_2RAC_FMT<"dpaqx_sa.w.ph", 0b11001010>;
class DPAU_H_QBL_MM_ENC : POOL32A_2RAC_FMT<"dpau.h.qbl", 0b10000010>;
class DPAU_H_QBR_MM_ENC : POOL32A_2RAC_FMT<"dpau.h.qbr", 0b11000010>;
class DPAX_W_PH_MMR2_ENC : POOL32A_2RAC_FMT<"dpax.w.ph", 0b01000010>;
class ABSQ_S_PH_MM_ENC : POOL32A_2R_FMT<"absq_s.ph", 0b0001000100>;
class ABSQ_S_W_MM_ENC : POOL32A_2R_FMT<"absq_s.w", 0b0010000100>;
class ABSQ_S_QB_MMR2_ENC : POOL32A_2R_FMT<"absq_s.qb", 0b0000000100>;
class INSV_MM_ENC : POOL32A_2R_FMT<"insv", 0b0100000100>;
class MADD_DSP_MM_ENC : POOL32A_2RAC_FMT<"madd", 0b00101010>;
class MADDU_DSP_MM_ENC : POOL32A_2RAC_FMT<"maddu", 0b01101010>;
class MSUB_DSP_MM_ENC : POOL32A_2RAC_FMT<"msub", 0b10101010>;
class MSUBU_DSP_MM_ENC : POOL32A_2RAC_FMT<"msubu", 0b11101010>;
class MULT_DSP_MM_ENC : POOL32A_2RAC_FMT<"mult", 0b00110010>;
class MULTU_DSP_MM_ENC : POOL32A_2RAC_FMT<"multu", 0b01110010>;
class SHLL_PH_MM_ENC : POOL32A_2RSA4_FMT<"shll.ph", 0b001110110101>;
class SHLL_S_PH_MM_ENC : POOL32A_2RSA4_FMT<"shll_s.ph", 0b101110110101>;
class SHLL_QB_MM_ENC : POOL32A_2RSA3_FMT<"shll.qb", 0b0100001>;
class SHLLV_PH_MM_ENC : POOL32A_3R_FMT<"shllv.ph", 0b00000001110>;
class SHLLV_S_PH_MM_ENC : POOL32A_3R_FMT<"shllv_s.ph", 0b10000001110>;
class SHLLV_QB_MM_ENC : POOL32A_3RB0_FMT<"shllv.qb", 0b1110010101>;
class SHLLV_S_W_MM_ENC : POOL32A_3RB0_FMT<"shllv_s.w", 0b1111010101>;
class SHLL_S_W_MM_ENC : POOL32A_2RSA5B0_FMT<"shll_s.w", 0b1111110101>;
class SHRA_QB_MMR2_ENC : POOL32A_2RSA3_FMT<"shra.qb", 0b0000111>;
class SHRA_R_QB_MMR2_ENC : POOL32A_2RSA3_FMT<"shra_r.qb", 0b1000111>;
class SHRA_PH_MM_ENC : POOL32A_2RSA4B0_FMT<"shra.ph", 0b01100110101>;
class SHRA_R_PH_MM_ENC : POOL32A_2RSA4B0_FMT<"shra_r.ph", 0b11100110101>;
class SHRAV_PH_MM_ENC : POOL32A_3R_FMT<"shrav.ph", 0b00110001101>;
class SHRAV_R_PH_MM_ENC : POOL32A_3R_FMT<"shrav_r.ph", 0b10110001101>;
class SHRAV_QB_MMR2_ENC : POOL32A_3R_FMT<"shrav.qb", 0b00111001101>;
class SHRAV_R_QB_MMR2_ENC : POOL32A_3R_FMT<"shrav_r.qb", 0b10111001101>;
class SHRAV_R_W_MM_ENC : POOL32A_3RB0_FMT<"shrav_r.w", 0b1011010101>;
class SHRA_R_W_MM_ENC : POOL32A_2RSA5B0_FMT<"shra_r.w", 0b1011110101>;
class SHRL_PH_MMR2_ENC : POOL32A_2RSA4OP6_FMT<"shrl.ph", 0b001111>;
class SHRL_QB_MM_ENC : POOL32A_2RSA3_FMT<"shrl.qb", 0b1100001>;
class SHRLV_PH_MMR2_ENC : POOL32A_3RB0_FMT<"shrlv.ph", 0b1100010101>;
class SHRLV_QB_MM_ENC : POOL32A_3RB0_FMT<"shrlv.qb", 0b1101010101>;
class PRECEQ_W_PHL_MM_ENC : POOL32A_2R_FMT<"preceq.w.phl", 0b0101000100>;
class PRECEQ_W_PHR_MM_ENC : POOL32A_2R_FMT<"preceq.w.phr", 0b0110000100>;
class PRECEQU_PH_QBL_MM_ENC : POOL32A_2R_FMT<"precequ.ph.qbl", 0b0111000100>;
class PRECEQU_PH_QBLA_MM_ENC : POOL32A_2R_FMT<"precequ.ph.qbla", 0b0111001100>;
class PRECEQU_PH_QBR_MM_ENC : POOL32A_2R_FMT<"precequ.ph.qbr", 0b1001000100>;
class PRECEQU_PH_QBRA_MM_ENC : POOL32A_2R_FMT<"precequ.ph.qbra", 0b1001001100>;
class PRECEU_PH_QBL_MM_ENC : POOL32A_2R_FMT<"preceu.ph.qbl", 0b1011000100>;
class PRECEU_PH_QBLA_MM_ENC : POOL32A_2R_FMT<"preceu.ph.qbla", 0b1011001100>;
class PRECEU_PH_QBR_MM_ENC : POOL32A_2R_FMT<"preceu.ph.qbr", 0b1101000100>;
class PRECEU_PH_QBRA_MM_ENC : POOL32A_2R_FMT<"preceu.ph.qbra", 0b1101001100>;
class SUBQ_PH_MM_ENC : POOL32A_3R_FMT<"subq.ph", 0b01000001101>;
class SUBQ_S_PH_MM_ENC : POOL32A_3R_FMT<"subq_s.ph", 0b11000001101>;
class SUBQ_S_W_MM_ENC : POOL32A_3RB0_FMT<"subq_s.w", 0b1101000101>;
class SUBQH_PH_MMR2_ENC : POOL32A_3R_FMT<"subqh.ph", 0b01001001101>;
class SUBQH_R_PH_MMR2_ENC : POOL32A_3R_FMT<"subqh_r.ph", 0b11001001101>;
class SUBQH_W_MMR2_ENC : POOL32A_3R_FMT<"subqh.w", 0b01010001101>;
class SUBQH_R_W_MMR2_ENC : POOL32A_3R_FMT<"subqh_r.w", 0b11010001101>;
class SUBU_PH_MMR2_ENC : POOL32A_3R_FMT<"subu.ph", 0b01100001101>;
class SUBU_S_PH_MMR2_ENC : POOL32A_3R_FMT<"subu_s.ph", 0b11100001101>;
class SUBU_QB_MM_ENC : POOL32A_3R_FMT<"subu.qb", 0b01011001101>;
class SUBU_S_QB_MM_ENC : POOL32A_3R_FMT<"subu_s.qb", 0b11011001101>;
class SUBUH_QB_MMR2_ENC : POOL32A_3R_FMT<"subuh.qb", 0b01101001101>;
class SUBUH_R_QB_MMR2_ENC : POOL32A_3R_FMT<"subuh_r.qb", 0b11101001101>;

// Instruction desc.
class ABSQ_S_PH_MM_R2_DESC_BASE<string opstr, SDPatternOperator OpNode,
                                InstrItinClass itin, RegisterOperand ROD,
                                RegisterOperand ROS = ROD> {
  dag OutOperandList = (outs ROD:$rt);
  dag InOperandList = (ins ROS:$rs);
  string AsmString = !strconcat(opstr, "\t$rt, $rs");
  list<dag> Pattern = [(set ROD:$rt, (OpNode ROS:$rs))];
  InstrItinClass Itinerary = itin;
}
class ABSQ_S_PH_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.ph", int_mips_absq_s_ph, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag20]>;
class ABSQ_S_W_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.w", int_mips_absq_s_w, NoItinerary, GPR32Opnd>, Defs<[DSPOutFlag20]>;
class ABSQ_S_QB_MMR2_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "absq_s.qb", int_mips_absq_s_qb, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag20]>;
class PRECEQ_W_PHL_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceq.w.phl", int_mips_preceq_w_phl, NoItinerary, GPR32Opnd, DSPROpnd>;
class PRECEQ_W_PHR_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceq.w.phr", int_mips_preceq_w_phr, NoItinerary, GPR32Opnd, DSPROpnd>;
class PRECEQU_PH_QBL_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "precequ.ph.qbl", int_mips_precequ_ph_qbl, NoItinerary, DSPROpnd>;
class PRECEQU_PH_QBLA_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "precequ.ph.qbla", int_mips_precequ_ph_qbla, NoItinerary, DSPROpnd>;
class PRECEQU_PH_QBR_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "precequ.ph.qbr", int_mips_precequ_ph_qbr, NoItinerary, DSPROpnd>;
class PRECEQU_PH_QBRA_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "precequ.ph.qbra", int_mips_precequ_ph_qbra, NoItinerary, DSPROpnd>;
class PRECEU_PH_QBL_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceu.ph.qbl", int_mips_preceu_ph_qbl, NoItinerary, DSPROpnd>;
class PRECEU_PH_QBLA_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceu.ph.qbla", int_mips_preceu_ph_qbla, NoItinerary, DSPROpnd>;
class PRECEU_PH_QBR_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceu.ph.qbr", int_mips_preceu_ph_qbr, NoItinerary, DSPROpnd>;
class PRECEU_PH_QBRA_MM_DESC : ABSQ_S_PH_MM_R2_DESC_BASE<
  "preceu.ph.qbra", int_mips_preceu_ph_qbra, NoItinerary, DSPROpnd>;

class SHLL_R2_MM_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                           SDPatternOperator ImmPat, InstrItinClass itin,
                           RegisterOperand RO, Operand ImmOpnd> {
  dag OutOperandList = (outs RO:$rt);
  dag InOperandList = (ins RO:$rs, ImmOpnd:$sa);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $sa");
  list<dag> Pattern = [(set RO:$rt, (OpNode RO:$rs, ImmPat:$sa))];
  InstrItinClass Itinerary = itin;
  bit hasSideEffects = 1;
}
class SHLL_PH_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shll.ph", null_frag, immZExt4, NoItinerary, DSPROpnd, uimm4>,
  Defs<[DSPOutFlag22]>;
class SHLL_S_PH_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shll_s.ph", int_mips_shll_s_ph, immZExt4, NoItinerary, DSPROpnd, uimm4>,
  Defs<[DSPOutFlag22]>;
class SHLL_QB_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shll.qb", null_frag, immZExt3, NoItinerary, DSPROpnd, uimm3>,
  Defs<[DSPOutFlag22]>;
class SHLL_S_W_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shll_s.w", int_mips_shll_s_w, immZExt5, NoItinerary, GPR32Opnd, uimm5>,
  Defs<[DSPOutFlag22]>;
class SHRA_QB_MMR2_DESC : SHLL_R2_MM_DESC_BASE<
  "shra.qb", null_frag, immZExt3, NoItinerary, DSPROpnd, uimm3>;
class SHRA_R_QB_MMR2_DESC : SHLL_R2_MM_DESC_BASE<
  "shra_r.qb", int_mips_shra_r_qb, immZExt3, NoItinerary, DSPROpnd, uimm3>;
class SHRA_PH_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shra.ph", null_frag, immZExt4, NoItinerary, DSPROpnd, uimm4>;
class SHRA_R_PH_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shra_r.ph", int_mips_shra_r_ph, immZExt4, NoItinerary, DSPROpnd, uimm4>;
class SHRA_R_W_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shra_r.w", int_mips_shra_r_w, immZExt5, NoItinerary, GPR32Opnd, uimm5>;
class SHRL_QB_MM_DESC : SHLL_R2_MM_DESC_BASE<
  "shrl.qb", null_frag, immZExt3, NoItinerary, DSPROpnd, uimm3>;
class SHRL_PH_MMR2_DESC : SHLL_R2_MM_DESC_BASE<
  "shrl.ph", null_frag, immZExt4, NoItinerary, DSPROpnd, uimm4>;

class SHLLV_R3_MM_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                            InstrItinClass itin, RegisterOperand RO> {
  dag OutOperandList = (outs RO:$rd);
  dag InOperandList =  (ins RO:$rt, GPR32Opnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt, $rs");
  list<dag> Pattern = [(set RO:$rd, (OpNode RO:$rt, GPR32Opnd:$rs))];
  InstrItinClass Itinerary = itin;
}
class SHLLV_PH_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shllv.ph", int_mips_shll_ph, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag22]>;
class SHLLV_S_PH_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shllv_s.ph", int_mips_shll_s_ph, NoItinerary, DSPROpnd>,
  Defs<[DSPOutFlag22]>;
class SHLLV_QB_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shllv.qb", int_mips_shll_qb, NoItinerary, DSPROpnd>, Defs<[DSPOutFlag22]>;
class SHLLV_S_W_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shllv_s.w", int_mips_shll_s_w, NoItinerary, GPR32Opnd>, Defs<[DSPOutFlag22]>;
class SHRAV_PH_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrav.ph", int_mips_shra_ph, NoItinerary, DSPROpnd>;
class SHRAV_R_PH_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrav_r.ph", int_mips_shra_r_ph, NoItinerary, DSPROpnd>;
class SHRAV_QB_MMR2_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrav.qb", int_mips_shra_qb, NoItinerary, DSPROpnd>;
class SHRAV_R_QB_MMR2_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrav_r.qb", int_mips_shra_r_qb, NoItinerary, DSPROpnd>;
class SHRAV_R_W_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrav_r.w", int_mips_shra_r_w, NoItinerary, GPR32Opnd>;
class SHRLV_PH_MMR2_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrlv.ph", int_mips_shrl_ph, NoItinerary, DSPROpnd>;
class SHRLV_QB_MM_DESC : SHLLV_R3_MM_DESC_BASE<
  "shrlv.qb", int_mips_shrl_qb, NoItinerary, DSPROpnd>;

// Instruction defs.
// microMIPS DSP Rev 1
def ADDQ_PH_MM : DspMMRel, ADDQ_PH_MM_ENC, ADDQ_PH_DESC;
def ADDQ_S_PH_MM : DspMMRel, ADDQ_S_PH_MM_ENC, ADDQ_S_PH_DESC;
def ADDQ_S_W_MM : DspMMRel, ADDQ_S_W_MM_ENC, ADDQ_S_W_DESC;
def ADDU_QB_MM : DspMMRel, ADDU_QB_MM_ENC, ADDU_QB_DESC;
def ADDU_S_QB_MM : DspMMRel, ADDU_S_QB_MM_ENC, ADDU_S_QB_DESC;
def ADDSC_MM : DspMMRel, ADDSC_MM_ENC, ADDSC_DESC;
def ADDWC_MM : DspMMRel, ADDWC_MM_ENC, ADDWC_DESC;
def DPAQ_S_W_PH_MM : DspMMRel, DPAQ_S_W_PH_MM_ENC, DPAQ_S_W_PH_DESC;
def DPAQ_SA_L_W_MM : DspMMRel, DPAQ_SA_L_W_MM_ENC, DPAQ_SA_L_W_DESC;
def DPAU_H_QBL_MM : DspMMRel, DPAU_H_QBL_MM_ENC, DPAU_H_QBL_DESC;
def DPAU_H_QBR_MM : DspMMRel, DPAU_H_QBR_MM_ENC, DPAU_H_QBR_DESC;
def ABSQ_S_PH_MM : DspMMRel, ABSQ_S_PH_MM_ENC, ABSQ_S_PH_MM_DESC;
def ABSQ_S_W_MM : DspMMRel, ABSQ_S_W_MM_ENC, ABSQ_S_W_MM_DESC;
def INSV_MM : DspMMRel, INSV_MM_ENC, INSV_DESC;
def MADD_DSP_MM : DspMMRel, MADD_DSP_MM_ENC, MADD_DSP_DESC;
def MADDU_DSP_MM : DspMMRel, MADDU_DSP_MM_ENC, MADDU_DSP_DESC;
def MSUB_DSP_MM : DspMMRel, MSUB_DSP_MM_ENC, MSUB_DSP_DESC;
def MSUBU_DSP_MM : DspMMRel, MSUBU_DSP_MM_ENC, MSUBU_DSP_DESC;
def MULT_DSP_MM : DspMMRel, MULT_DSP_MM_ENC, MULT_DSP_DESC;
def MULTU_DSP_MM : DspMMRel, MULTU_DSP_MM_ENC, MULTU_DSP_DESC;
def SHLL_PH_MM : DspMMRel, SHLL_PH_MM_ENC, SHLL_PH_MM_DESC;
def SHLL_S_PH_MM : DspMMRel, SHLL_S_PH_MM_ENC, SHLL_S_PH_MM_DESC;
def SHLL_QB_MM : DspMMRel, SHLL_QB_MM_ENC, SHLL_QB_MM_DESC;
def SHLLV_PH_MM : DspMMRel, SHLLV_PH_MM_ENC, SHLLV_PH_MM_DESC;
def SHLLV_S_PH_MM : DspMMRel, SHLLV_S_PH_MM_ENC, SHLLV_S_PH_MM_DESC;
def SHLLV_QB_MM : DspMMRel, SHLLV_QB_MM_ENC, SHLLV_QB_MM_DESC;
def SHLLV_S_W_MM : DspMMRel, SHLLV_S_W_MM_ENC, SHLLV_S_W_MM_DESC;
def SHLL_S_W_MM : DspMMRel, SHLL_S_W_MM_ENC, SHLL_S_W_MM_DESC;
def SHRA_PH_MM : DspMMRel, SHRA_PH_MM_ENC, SHRA_PH_MM_DESC;
def SHRA_R_PH_MM : DspMMRel, SHRA_R_PH_MM_ENC, SHRA_R_PH_MM_DESC;
def SHRAV_PH_MM : DspMMRel, SHRAV_PH_MM_ENC, SHRAV_PH_MM_DESC;
def SHRAV_R_PH_MM : DspMMRel, SHRAV_R_PH_MM_ENC, SHRAV_R_PH_MM_DESC;
def SHRAV_R_W_MM : DspMMRel, SHRAV_R_W_MM_ENC, SHRAV_R_W_MM_DESC;
def SHRA_R_W_MM : DspMMRel, SHRA_R_W_MM_ENC, SHRA_R_W_MM_DESC;
def SHRL_QB_MM : DspMMRel, SHRL_QB_MM_ENC, SHRL_QB_MM_DESC;
def SHRLV_QB_MM : DspMMRel, SHRLV_QB_MM_ENC, SHRLV_QB_MM_DESC;
def PRECEQ_W_PHL_MM : DspMMRel, PRECEQ_W_PHL_MM_ENC, PRECEQ_W_PHL_MM_DESC;
def PRECEQ_W_PHR_MM : DspMMRel, PRECEQ_W_PHR_MM_ENC, PRECEQ_W_PHR_MM_DESC;
def PRECEQU_PH_QBL_MM : DspMMRel, PRECEQU_PH_QBL_MM_ENC, PRECEQU_PH_QBL_MM_DESC;
def PRECEQU_PH_QBLA_MM : DspMMRel, PRECEQU_PH_QBLA_MM_ENC,
                         PRECEQU_PH_QBLA_MM_DESC;
def PRECEQU_PH_QBR_MM : DspMMRel, PRECEQU_PH_QBR_MM_ENC, PRECEQU_PH_QBR_MM_DESC;
def PRECEQU_PH_QBRA_MM : DspMMRel, PRECEQU_PH_QBRA_MM_ENC,
                         PRECEQU_PH_QBRA_MM_DESC;
def PRECEU_PH_QBL_MM : DspMMRel, PRECEU_PH_QBL_MM_ENC, PRECEU_PH_QBL_MM_DESC;
def PRECEU_PH_QBLA_MM : DspMMRel, PRECEU_PH_QBLA_MM_ENC, PRECEU_PH_QBLA_MM_DESC;
def PRECEU_PH_QBR_MM : DspMMRel, PRECEU_PH_QBR_MM_ENC, PRECEU_PH_QBR_MM_DESC;
def PRECEU_PH_QBRA_MM : DspMMRel, PRECEU_PH_QBRA_MM_ENC, PRECEU_PH_QBRA_MM_DESC;
def SUBQ_PH_MM : DspMMRel, SUBQ_PH_MM_ENC, SUBQ_PH_DESC;
def SUBQ_S_PH_MM : DspMMRel, SUBQ_S_PH_MM_ENC, SUBQ_S_PH_DESC;
def SUBQ_S_W_MM : DspMMRel, SUBQ_S_W_MM_ENC, SUBQ_S_W_DESC;
def SUBU_QB_MM : DspMMRel, SUBU_QB_MM_ENC, SUBU_QB_DESC;
def SUBU_S_QB_MM : DspMMRel, SUBU_S_QB_MM_ENC, SUBU_S_QB_DESC;
// microMIPS DSP Rev 2
def ABSQ_S_QB_MMR2 : DspMMRel, ABSQ_S_QB_MMR2_ENC, ABSQ_S_QB_MMR2_DESC,
                     ISA_DSPR2;
def ADDQH_PH_MMR2 : DspMMRel, ADDQH_PH_MMR2_ENC, ADDQH_PH_DESC, ISA_DSPR2;
def ADDQH_R_PH_MMR2 : DspMMRel, ADDQH_R_PH_MMR2_ENC, ADDQH_R_PH_DESC, ISA_DSPR2;
def ADDQH_W_MMR2 : DspMMRel, ADDQH_W_MMR2_ENC, ADDQH_W_DESC, ISA_DSPR2;
def ADDQH_R_W_MMR2 : DspMMRel, ADDQH_R_W_MMR2_ENC, ADDQH_R_W_DESC, ISA_DSPR2;
def ADDU_PH_MMR2 : DspMMRel, ADDU_PH_MMR2_ENC, ADDU_PH_DESC, ISA_DSPR2;
def ADDU_S_PH_MMR2 : DspMMRel, ADDU_S_PH_MMR2_ENC, ADDU_S_PH_DESC, ISA_DSPR2;
def ADDUH_QB_MMR2 : DspMMRel, ADDUH_QB_MMR2_ENC, ADDUH_QB_DESC, ISA_DSPR2;
def ADDUH_R_QB_MMR2 : DspMMRel, ADDUH_R_QB_MMR2_ENC, ADDUH_R_QB_DESC, ISA_DSPR2;
def DPA_W_PH_MMR2 : DspMMRel, DPA_W_PH_MMR2_ENC, DPA_W_PH_DESC, ISA_DSPR2;
def DPAQX_S_W_PH_MMR2 : DspMMRel, DPAQX_S_W_PH_MMR2_ENC, DPAQX_S_W_PH_DESC,
                        ISA_DSPR2;
def DPAQX_SA_W_PH_MMR2 : DspMMRel, DPAQX_SA_W_PH_MMR2_ENC, DPAQX_SA_W_PH_DESC,
                         ISA_DSPR2;
def DPAX_W_PH_MMR2 : DspMMRel, DPAX_W_PH_MMR2_ENC, DPAX_W_PH_DESC, ISA_DSPR2;
def SHRA_QB_MMR2 : DspMMRel, SHRA_QB_MMR2_ENC, SHRA_QB_MMR2_DESC, ISA_DSPR2;
def SHRA_R_QB_MMR2 : DspMMRel, SHRA_R_QB_MMR2_ENC, SHRA_R_QB_MMR2_DESC,
                     ISA_DSPR2;
def SHRAV_QB_MMR2 : DspMMRel, SHRAV_QB_MMR2_ENC, SHRAV_QB_MMR2_DESC, ISA_DSPR2;
def SHRAV_R_QB_MMR2 : DspMMRel, SHRAV_R_QB_MMR2_ENC, SHRAV_R_QB_MMR2_DESC,
                      ISA_DSPR2;
def SHRL_PH_MMR2 : DspMMRel, SHRL_PH_MMR2_ENC, SHRL_PH_MMR2_DESC, ISA_DSPR2;
def SHRLV_PH_MMR2 : DspMMRel, SHRLV_PH_MMR2_ENC, SHRLV_PH_MMR2_DESC, ISA_DSPR2;
def SUBQH_PH_MMR2 : DspMMRel, SUBQH_PH_MMR2_ENC, SUBQH_PH_DESC, ISA_DSPR2;
def SUBQH_R_PH_MMR2 : DspMMRel, SUBQH_R_PH_MMR2_ENC, SUBQH_R_PH_DESC, ISA_DSPR2;
def SUBQH_W_MMR2 : DspMMRel, SUBQH_W_MMR2_ENC, SUBQH_W_DESC, ISA_DSPR2;
def SUBQH_R_W_MMR2 : DspMMRel, SUBQH_R_W_MMR2_ENC, SUBQH_R_W_DESC, ISA_DSPR2;
def SUBU_PH_MMR2 : DspMMRel, SUBU_PH_MMR2_ENC, SUBU_PH_DESC, ISA_DSPR2;
def SUBU_S_PH_MMR2 : DspMMRel, SUBU_S_PH_MMR2_ENC, SUBU_S_PH_DESC, ISA_DSPR2;
def SUBUH_QB_MMR2 : DspMMRel, SUBUH_QB_MMR2_ENC, SUBUH_QB_DESC, ISA_DSPR2;
def SUBUH_R_QB_MMR2 : DspMMRel, SUBUH_R_QB_MMR2_ENC, SUBUH_R_QB_DESC, ISA_DSPR2;
