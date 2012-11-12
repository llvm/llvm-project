//===- MipsDSPInstrInfo.td - DSP ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips DSP ASE instructions.
//
//===----------------------------------------------------------------------===//

// ImmLeaf
def immZExt2 : ImmLeaf<i32, [{return isUInt<2>(Imm);}]>;
def immZExt3 : ImmLeaf<i32, [{return isUInt<3>(Imm);}]>;
def immZExt4 : ImmLeaf<i32, [{return isUInt<4>(Imm);}]>;
def immZExt8 : ImmLeaf<i32, [{return isUInt<8>(Imm);}]>;
def immZExt10 : ImmLeaf<i32, [{return isUInt<10>(Imm);}]>;
def immSExt6 : ImmLeaf<i32, [{return isInt<6>(Imm);}]>;

// Mips-specific dsp nodes
def SDT_MipsExtr : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisSameAs<0, 1>]>;
def SDT_MipsShilo : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def SDT_MipsDPA : SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisSameAs<0, 1>]>;

class MipsDSPBase<string Opc, SDTypeProfile Prof> :
  SDNode<!strconcat("MipsISD::", Opc), Prof,
         [SDNPHasChain, SDNPInGlue, SDNPOutGlue]>;

class MipsDSPSideEffectBase<string Opc, SDTypeProfile Prof> :
  SDNode<!strconcat("MipsISD::", Opc), Prof,
         [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPSideEffect]>;

def MipsEXTP : MipsDSPSideEffectBase<"EXTP", SDT_MipsExtr>;
def MipsEXTPDP : MipsDSPSideEffectBase<"EXTPDP", SDT_MipsExtr>;
def MipsEXTR_S_H : MipsDSPSideEffectBase<"EXTR_S_H", SDT_MipsExtr>;
def MipsEXTR_W : MipsDSPSideEffectBase<"EXTR_W", SDT_MipsExtr>;
def MipsEXTR_R_W : MipsDSPSideEffectBase<"EXTR_R_W", SDT_MipsExtr>;
def MipsEXTR_RS_W : MipsDSPSideEffectBase<"EXTR_RS_W", SDT_MipsExtr>;

def MipsSHILO : MipsDSPBase<"SHILO", SDT_MipsShilo>;
def MipsMTHLIP : MipsDSPBase<"MTHLIP", SDT_MipsShilo>;

def MipsMULSAQ_S_W_PH : MipsDSPSideEffectBase<"MULSAQ_S_W_PH", SDT_MipsDPA>;
def MipsMAQ_S_W_PHL : MipsDSPSideEffectBase<"MAQ_S_W_PHL", SDT_MipsDPA>;
def MipsMAQ_S_W_PHR : MipsDSPSideEffectBase<"MAQ_S_W_PHR", SDT_MipsDPA>;
def MipsMAQ_SA_W_PHL : MipsDSPSideEffectBase<"MAQ_SA_W_PHL", SDT_MipsDPA>;
def MipsMAQ_SA_W_PHR : MipsDSPSideEffectBase<"MAQ_SA_W_PHR", SDT_MipsDPA>;

def MipsDPAU_H_QBL : MipsDSPBase<"DPAU_H_QBL", SDT_MipsDPA>;
def MipsDPAU_H_QBR : MipsDSPBase<"DPAU_H_QBR", SDT_MipsDPA>;
def MipsDPSU_H_QBL : MipsDSPBase<"DPSU_H_QBL", SDT_MipsDPA>;
def MipsDPSU_H_QBR : MipsDSPBase<"DPSU_H_QBR", SDT_MipsDPA>;
def MipsDPAQ_S_W_PH : MipsDSPSideEffectBase<"DPAQ_S_W_PH", SDT_MipsDPA>;
def MipsDPSQ_S_W_PH : MipsDSPSideEffectBase<"DPSQ_S_W_PH", SDT_MipsDPA>;
def MipsDPAQ_SA_L_W : MipsDSPSideEffectBase<"DPAQ_SA_L_W", SDT_MipsDPA>;
def MipsDPSQ_SA_L_W : MipsDSPSideEffectBase<"DPSQ_SA_L_W", SDT_MipsDPA>;

def MipsDPA_W_PH : MipsDSPBase<"DPA_W_PH", SDT_MipsDPA>;
def MipsDPS_W_PH : MipsDSPBase<"DPS_W_PH", SDT_MipsDPA>;
def MipsDPAQX_S_W_PH : MipsDSPSideEffectBase<"DPAQX_S_W_PH", SDT_MipsDPA>;
def MipsDPAQX_SA_W_PH : MipsDSPSideEffectBase<"DPAQX_SA_W_PH", SDT_MipsDPA>;
def MipsDPAX_W_PH : MipsDSPBase<"DPAX_W_PH", SDT_MipsDPA>;
def MipsDPSX_W_PH : MipsDSPBase<"DPSX_W_PH", SDT_MipsDPA>;
def MipsDPSQX_S_W_PH : MipsDSPSideEffectBase<"DPSQX_S_W_PH", SDT_MipsDPA>;
def MipsDPSQX_SA_W_PH : MipsDSPSideEffectBase<"DPSQX_SA_W_PH", SDT_MipsDPA>;
def MipsMULSA_W_PH : MipsDSPBase<"MULSA_W_PH", SDT_MipsDPA>;

def MipsMULT : MipsDSPBase<"MULT", SDT_MipsDPA>;
def MipsMULTU : MipsDSPBase<"MULTU", SDT_MipsDPA>;
def MipsMADD_DSP : MipsDSPBase<"MADD_DSP", SDT_MipsDPA>;
def MipsMADDU_DSP : MipsDSPBase<"MADDU_DSP", SDT_MipsDPA>;
def MipsMSUB_DSP : MipsDSPBase<"MSUB_DSP", SDT_MipsDPA>;
def MipsMSUBU_DSP : MipsDSPBase<"MSUBU_DSP", SDT_MipsDPA>;

// Flags.
class IsCommutable {
  bit isCommutable = 1;
}

class UseAC {
  list<Register> Uses = [AC0];
}

class UseDSPCtrl {
  list<Register> Uses = [DSPCtrl];
}

class ClearDefs {
  list<Register> Defs = [];
}

// Instruction encoding.
class ADDU_QB_ENC : ADDU_QB_FMT<0b00000>;
class ADDU_S_QB_ENC : ADDU_QB_FMT<0b00100>;
class SUBU_QB_ENC : ADDU_QB_FMT<0b00001>;
class SUBU_S_QB_ENC : ADDU_QB_FMT<0b00101>;
class ADDQ_PH_ENC : ADDU_QB_FMT<0b01010>;
class ADDQ_S_PH_ENC : ADDU_QB_FMT<0b01110>;
class SUBQ_PH_ENC : ADDU_QB_FMT<0b01011>;
class SUBQ_S_PH_ENC : ADDU_QB_FMT<0b01111>;
class ADDQ_S_W_ENC : ADDU_QB_FMT<0b10110>;
class SUBQ_S_W_ENC : ADDU_QB_FMT<0b10111>;
class ADDSC_ENC : ADDU_QB_FMT<0b10000>;
class ADDWC_ENC : ADDU_QB_FMT<0b10001>;
class MODSUB_ENC : ADDU_QB_FMT<0b10010>;
class RADDU_W_QB_ENC : RADDU_W_QB_FMT<0b10100>;
class ABSQ_S_PH_ENC : ABSQ_S_PH_R2_FMT<0b01001>;
class ABSQ_S_W_ENC : ABSQ_S_PH_R2_FMT<0b10001>;
class PRECRQ_QB_PH_ENC : CMP_EQ_QB_R3_FMT<0b01100>;
class PRECRQ_PH_W_ENC : CMP_EQ_QB_R3_FMT<0b10100>;
class PRECRQ_RS_PH_W_ENC : CMP_EQ_QB_R3_FMT<0b10101>;
class PRECRQU_S_QB_PH_ENC : CMP_EQ_QB_R3_FMT<0b01111>;
class PRECEQ_W_PHL_ENC : ABSQ_S_PH_R2_FMT<0b01100>;
class PRECEQ_W_PHR_ENC : ABSQ_S_PH_R2_FMT<0b01101>;
class PRECEQU_PH_QBL_ENC : ABSQ_S_PH_R2_FMT<0b00100>;
class PRECEQU_PH_QBR_ENC : ABSQ_S_PH_R2_FMT<0b00101>;
class PRECEQU_PH_QBLA_ENC : ABSQ_S_PH_R2_FMT<0b00110>;
class PRECEQU_PH_QBRA_ENC : ABSQ_S_PH_R2_FMT<0b00111>;
class PRECEU_PH_QBL_ENC : ABSQ_S_PH_R2_FMT<0b11100>;
class PRECEU_PH_QBR_ENC : ABSQ_S_PH_R2_FMT<0b11101>;
class PRECEU_PH_QBLA_ENC : ABSQ_S_PH_R2_FMT<0b11110>;
class PRECEU_PH_QBRA_ENC : ABSQ_S_PH_R2_FMT<0b11111>;
class SHLL_QB_ENC : SHLL_QB_FMT<0b00000>;
class SHLLV_QB_ENC : SHLL_QB_FMT<0b00010>;
class SHRL_QB_ENC : SHLL_QB_FMT<0b00001>;
class SHRLV_QB_ENC : SHLL_QB_FMT<0b00011>;
class SHLL_PH_ENC : SHLL_QB_FMT<0b01000>;
class SHLLV_PH_ENC : SHLL_QB_FMT<0b01010>;
class SHLL_S_PH_ENC : SHLL_QB_FMT<0b01100>;
class SHLLV_S_PH_ENC : SHLL_QB_FMT<0b01110>;
class SHRA_PH_ENC : SHLL_QB_FMT<0b01001>;
class SHRAV_PH_ENC : SHLL_QB_FMT<0b01011>;
class SHRA_R_PH_ENC : SHLL_QB_FMT<0b01101>;
class SHRAV_R_PH_ENC : SHLL_QB_FMT<0b01111>;
class SHLL_S_W_ENC : SHLL_QB_FMT<0b10100>;
class SHLLV_S_W_ENC : SHLL_QB_FMT<0b10110>;
class SHRA_R_W_ENC : SHLL_QB_FMT<0b10101>;
class SHRAV_R_W_ENC : SHLL_QB_FMT<0b10111>;
class MULEU_S_PH_QBL_ENC : ADDU_QB_FMT<0b00110>;
class MULEU_S_PH_QBR_ENC : ADDU_QB_FMT<0b00111>;
class MULEQ_S_W_PHL_ENC : ADDU_QB_FMT<0b11100>;
class MULEQ_S_W_PHR_ENC : ADDU_QB_FMT<0b11101>;
class MULQ_RS_PH_ENC : ADDU_QB_FMT<0b11111>;
class MULSAQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00110>;
class MAQ_S_W_PHL_ENC : DPA_W_PH_FMT<0b10100>;
class MAQ_S_W_PHR_ENC : DPA_W_PH_FMT<0b10110>;
class MAQ_SA_W_PHL_ENC : DPA_W_PH_FMT<0b10000>;
class MAQ_SA_W_PHR_ENC : DPA_W_PH_FMT<0b10010>;
class DPAU_H_QBL_ENC : DPA_W_PH_FMT<0b00011>;
class DPAU_H_QBR_ENC : DPA_W_PH_FMT<0b00111>;
class DPSU_H_QBL_ENC : DPA_W_PH_FMT<0b01011>;
class DPSU_H_QBR_ENC : DPA_W_PH_FMT<0b01111>;
class DPAQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00100>;
class DPSQ_S_W_PH_ENC : DPA_W_PH_FMT<0b00101>;
class DPAQ_SA_L_W_ENC : DPA_W_PH_FMT<0b01100>;
class DPSQ_SA_L_W_ENC : DPA_W_PH_FMT<0b01101>;
class MULT_DSP_ENC : MULT_FMT<0b000000, 0b011000>;
class MULTU_DSP_ENC : MULT_FMT<0b000000, 0b011001>;
class MADD_DSP_ENC : MULT_FMT<0b011100, 0b000000>;
class MADDU_DSP_ENC : MULT_FMT<0b011100, 0b000001>;
class MSUB_DSP_ENC : MULT_FMT<0b011100, 0b000100>;
class MSUBU_DSP_ENC : MULT_FMT<0b011100, 0b000101>;
class CMPU_EQ_QB_ENC : CMP_EQ_QB_R2_FMT<0b00000>;
class CMPU_LT_QB_ENC : CMP_EQ_QB_R2_FMT<0b00001>;
class CMPU_LE_QB_ENC : CMP_EQ_QB_R2_FMT<0b00010>;
class CMPGU_EQ_QB_ENC : CMP_EQ_QB_R3_FMT<0b00100>;
class CMPGU_LT_QB_ENC : CMP_EQ_QB_R3_FMT<0b00101>;
class CMPGU_LE_QB_ENC : CMP_EQ_QB_R3_FMT<0b00110>;
class CMP_EQ_PH_ENC : CMP_EQ_QB_R2_FMT<0b01000>;
class CMP_LT_PH_ENC : CMP_EQ_QB_R2_FMT<0b01001>;
class CMP_LE_PH_ENC : CMP_EQ_QB_R2_FMT<0b01010>;
class BITREV_ENC : ABSQ_S_PH_R2_FMT<0b11011>;
class PACKRL_PH_ENC : CMP_EQ_QB_R3_FMT<0b01110>;
class REPL_QB_ENC : REPL_FMT<0b00010>;
class REPL_PH_ENC : REPL_FMT<0b01010>;
class REPLV_QB_ENC : ABSQ_S_PH_R2_FMT<0b00011>;
class REPLV_PH_ENC : ABSQ_S_PH_R2_FMT<0b01011>;
class PICK_QB_ENC : CMP_EQ_QB_R3_FMT<0b00011>;
class PICK_PH_ENC : CMP_EQ_QB_R3_FMT<0b01011>;
class LWX_ENC : LX_FMT<0b00000>;
class LHX_ENC : LX_FMT<0b00100>;
class LBUX_ENC : LX_FMT<0b00110>;
class BPOSGE32_ENC : BPOSGE32_FMT<0b11100>;
class INSV_ENC : INSV_FMT<0b001100>;

class EXTP_ENC : EXTR_W_TY1_FMT<0b00010>;
class EXTPV_ENC : EXTR_W_TY1_FMT<0b00011>;
class EXTPDP_ENC : EXTR_W_TY1_FMT<0b01010>;
class EXTPDPV_ENC : EXTR_W_TY1_FMT<0b01011>;
class EXTR_W_ENC : EXTR_W_TY1_FMT<0b00000>;
class EXTRV_W_ENC : EXTR_W_TY1_FMT<0b00001>;
class EXTR_R_W_ENC : EXTR_W_TY1_FMT<0b00100>;
class EXTRV_R_W_ENC : EXTR_W_TY1_FMT<0b00101>;
class EXTR_RS_W_ENC : EXTR_W_TY1_FMT<0b00110>;
class EXTRV_RS_W_ENC : EXTR_W_TY1_FMT<0b00111>;
class EXTR_S_H_ENC : EXTR_W_TY1_FMT<0b01110>;
class EXTRV_S_H_ENC : EXTR_W_TY1_FMT<0b01111>;
class SHILO_ENC : SHILO_R1_FMT<0b11010>;
class SHILOV_ENC : SHILO_R2_FMT<0b11011>;
class MTHLIP_ENC : SHILO_R2_FMT<0b11111>;

class RDDSP_ENC : RDDSP_FMT<0b10010>;
class WRDSP_ENC : WRDSP_FMT<0b10011>;
class ADDU_PH_ENC : ADDU_QB_FMT<0b01000>;
class ADDU_S_PH_ENC : ADDU_QB_FMT<0b01100>;
class SUBU_PH_ENC : ADDU_QB_FMT<0b01001>;
class SUBU_S_PH_ENC : ADDU_QB_FMT<0b01101>;
class CMPGDU_EQ_QB_ENC : CMP_EQ_QB_R3_FMT<0b11000>;
class CMPGDU_LT_QB_ENC : CMP_EQ_QB_R3_FMT<0b11001>;
class CMPGDU_LE_QB_ENC : CMP_EQ_QB_R3_FMT<0b11010>;
class ABSQ_S_QB_ENC : ABSQ_S_PH_R2_FMT<0b00001>;
class ADDUH_QB_ENC : ADDUH_QB_FMT<0b00000>;
class ADDUH_R_QB_ENC : ADDUH_QB_FMT<0b00010>;
class SUBUH_QB_ENC : ADDUH_QB_FMT<0b00001>;
class SUBUH_R_QB_ENC : ADDUH_QB_FMT<0b00011>;
class ADDQH_PH_ENC : ADDUH_QB_FMT<0b01000>;
class ADDQH_R_PH_ENC : ADDUH_QB_FMT<0b01010>;
class SUBQH_PH_ENC : ADDUH_QB_FMT<0b01001>;
class SUBQH_R_PH_ENC : ADDUH_QB_FMT<0b01011>;
class ADDQH_W_ENC : ADDUH_QB_FMT<0b10000>;
class ADDQH_R_W_ENC : ADDUH_QB_FMT<0b10010>;
class SUBQH_W_ENC : ADDUH_QB_FMT<0b10001>;
class SUBQH_R_W_ENC : ADDUH_QB_FMT<0b10011>;
class MUL_PH_ENC : ADDUH_QB_FMT<0b01100>;
class MUL_S_PH_ENC : ADDUH_QB_FMT<0b01110>;
class MULQ_S_W_ENC : ADDUH_QB_FMT<0b10110>;
class MULQ_RS_W_ENC : ADDUH_QB_FMT<0b10111>;
class MULQ_S_PH_ENC : ADDU_QB_FMT<0b11110>;
class DPA_W_PH_ENC : DPA_W_PH_FMT<0b00000>;
class DPS_W_PH_ENC : DPA_W_PH_FMT<0b00001>;
class DPAQX_S_W_PH_ENC : DPA_W_PH_FMT<0b11000>;
class DPAQX_SA_W_PH_ENC : DPA_W_PH_FMT<0b11010>;
class DPAX_W_PH_ENC : DPA_W_PH_FMT<0b01000>;
class DPSX_W_PH_ENC : DPA_W_PH_FMT<0b01001>;
class DPSQX_S_W_PH_ENC : DPA_W_PH_FMT<0b11001>;
class DPSQX_SA_W_PH_ENC : DPA_W_PH_FMT<0b11011>;
class MULSA_W_PH_ENC : DPA_W_PH_FMT<0b00010>;
class PRECR_QB_PH_ENC : CMP_EQ_QB_R3_FMT<0b01101>;
class PRECR_SRA_PH_W_ENC : PRECR_SRA_PH_W_FMT<0b11110>;
class PRECR_SRA_R_PH_W_ENC : PRECR_SRA_PH_W_FMT<0b11111>;
class SHRA_QB_ENC : SHLL_QB_FMT<0b00100>;
class SHRAV_QB_ENC : SHLL_QB_FMT<0b00110>;
class SHRA_R_QB_ENC : SHLL_QB_FMT<0b00101>;
class SHRAV_R_QB_ENC : SHLL_QB_FMT<0b00111>;
class SHRL_PH_ENC : SHLL_QB_FMT<0b11001>;
class SHRLV_PH_ENC : SHLL_QB_FMT<0b11011>;
class APPEND_ENC : APPEND_FMT<0b00000>;
class BALIGN_ENC : APPEND_FMT<0b10000>;
class PREPEND_ENC : APPEND_FMT<0b00001>;

// Instruction desc.
class ADDU_QB_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                        InstrItinClass itin, RegisterClass RCD,
                        RegisterClass RCS,  RegisterClass RCT = RCS> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCS:$rs, RCT:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCS:$rs, RCT:$rt))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class RADDU_W_QB_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                           InstrItinClass itin, RegisterClass RCD,
                           RegisterClass RCS = RCD> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCS:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCS:$rs))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class CMP_EQ_QB_R2_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                             InstrItinClass itin, RegisterClass RCS,
                             RegisterClass RCT = RCS> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCS:$rs, RCT:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt");
  list<dag> Pattern = [(OpNode RCS:$rs, RCT:$rt)];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class CMP_EQ_QB_R3_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                             InstrItinClass itin, RegisterClass RCD,
                             RegisterClass RCS,  RegisterClass RCT = RCS> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCS:$rs, RCT:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCS:$rs, RCT:$rt))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class PRECR_SRA_PH_W_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                               InstrItinClass itin, RegisterClass RCT,
                               RegisterClass RCS = RCT> {
  dag OutOperandList = (outs RCT:$rt);
  dag InOperandList = (ins RCS:$rs, shamt:$sa, RCS:$src);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $sa");
  list<dag> Pattern = [(set RCT:$rt, (OpNode RCS:$src, RCS:$rs, immZExt5:$sa))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
  string Constraints = "$src = $rt";
}

class ABSQ_S_PH_R2_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                             InstrItinClass itin, RegisterClass RCD,
                             RegisterClass RCT = RCD> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCT:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCT:$rt))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class REPL_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                     ImmLeaf immPat, InstrItinClass itin, RegisterClass RC> {
  dag OutOperandList = (outs RC:$rd);
  dag InOperandList = (ins uimm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rd, $imm");
  list<dag> Pattern = [(set RC:$rd, (OpNode immPat:$imm))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class SHLL_QB_R3_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                           InstrItinClass itin, RegisterClass RC> {
  dag OutOperandList = (outs RC:$rd);
  dag InOperandList =  (ins RC:$rt, CPURegs:$rs_sa);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt, $rs_sa");
  list<dag> Pattern = [(set RC:$rd, (OpNode RC:$rt, CPURegs:$rs_sa))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class SHLL_QB_R2_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                           SDPatternOperator ImmPat, InstrItinClass itin,
                           RegisterClass RC> {
  dag OutOperandList = (outs RC:$rd);
  dag InOperandList = (ins RC:$rt, uimm16:$rs_sa);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt, $rs_sa");
  list<dag> Pattern = [(set RC:$rd, (OpNode RC:$rt, ImmPat:$rs_sa))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class LX_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rd);
  dag InOperandList = (ins CPURegs:$base, CPURegs:$index);
  string AsmString = !strconcat(instr_asm, "\t$rd, ${index}(${base})");
  list<dag> Pattern = [(set CPURegs:$rd,
                       (OpNode CPURegs:$base, CPURegs:$index))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
  bit mayLoad = 1;
}

class ADDUH_QB_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                         InstrItinClass itin, RegisterClass RCD,
                         RegisterClass RCS = RCD,  RegisterClass RCT = RCD> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCS:$rs, RCT:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCS:$rs, RCT:$rt))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class APPEND_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       SDPatternOperator ImmOp, InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins CPURegs:$rs, shamt:$sa, CPURegs:$src);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $sa");
  list<dag> Pattern =  [(set CPURegs:$rt,
                        (OpNode CPURegs:$src, CPURegs:$rs, ImmOp:$sa))];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
  string Constraints = "$src = $rt";
}

class EXTR_W_TY1_R2_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                              InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins ACRegs:$ac, CPURegs:$shift_rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $ac, $shift_rs");
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class EXTR_W_TY1_R1_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                              InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins ACRegs:$ac, uimm16:$shift_rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $ac, $shift_rs");
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class SHILO_R1_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins simm16:$shift), [(OpNode immSExt6:$shift)]>,
  PseudoInstExpansion<(realinst AC0, simm16:$shift)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class SHILO_R1_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins simm16:$shift);
  string AsmString = !strconcat(instr_asm, "\t$ac, $shift");
}

class SHILO_R2_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs), [(OpNode CPURegs:$rs)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class SHILO_R2_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs");
}

class MTHLIP_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $ac");
}

class RDDSP_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                      InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rd);
  dag InOperandList = (ins uimm16:$mask);
  string AsmString = !strconcat(instr_asm, "\t$rd, $mask");
  list<dag> Pattern = [(set CPURegs:$rd, (OpNode immZExt10:$mask))];
  InstrItinClass Itinerary = itin;
  list<Register> Uses = [DSPCtrl];
}

class WRDSP_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                      InstrItinClass itin> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins CPURegs:$rs, uimm16:$mask);
  string AsmString = !strconcat(instr_asm, "\t$rs, $mask");
  list<dag> Pattern = [(OpNode CPURegs:$rs, immZExt10:$mask)];
  InstrItinClass Itinerary = itin;
  list<Register> Defs = [DSPCtrl];
}

class DPA_W_PH_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                           Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs, CPURegs:$rt),
            [(OpNode CPURegs:$rs, CPURegs:$rt)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs, CPURegs:$rt)> {
  list<Register> Defs = [DSPCtrl, AC0];
  list<Register> Uses = [AC0];
  InstrItinClass Itinerary = itin;
}

class DPA_W_PH_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs, CPURegs:$rt);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs, $rt");
}

class MULT_PSEUDO_BASE<SDPatternOperator OpNode, InstrItinClass itin,
                       Instruction realinst> :
  PseudoDSP<(outs), (ins CPURegs:$rs, CPURegs:$rt),
            [(OpNode CPURegs:$rs, CPURegs:$rt)]>,
  PseudoInstExpansion<(realinst AC0, CPURegs:$rs, CPURegs:$rt)> {
  list<Register> Defs = [DSPCtrl, AC0];
  InstrItinClass Itinerary = itin;
}

class MULT_DESC_BASE<string instr_asm> {
  dag OutOperandList = (outs ACRegs:$ac);
  dag InOperandList = (ins CPURegs:$rs, CPURegs:$rt);
  string AsmString = !strconcat(instr_asm, "\t$ac, $rs, $rt");
}

class BPOSGE32_PSEUDO_DESC_BASE<SDPatternOperator OpNode, InstrItinClass itin> :
  MipsPseudo<(outs CPURegs:$dst), (ins), "", [(set CPURegs:$dst, (OpNode))]> {
  list<Register> Uses = [DSPCtrl];
  bit usesCustomInserter = 1;
}

class BPOSGE32_DESC_BASE<string instr_asm, InstrItinClass itin> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins brtarget:$offset);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  InstrItinClass Itinerary = itin;
  list<Register> Uses = [DSPCtrl];
  bit isBranch = 1;
  bit isTerminator = 1;
  bit hasDelaySlot = 1;
}

class INSV_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                     InstrItinClass itin> {
  dag OutOperandList = (outs CPURegs:$rt);
  dag InOperandList = (ins CPURegs:$src, CPURegs:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs");
  list<dag> Pattern = [(set CPURegs:$rt, (OpNode CPURegs:$src, CPURegs:$rs))];
  InstrItinClass Itinerary = itin;
  list<Register> Uses = [DSPCtrl];
  string Constraints = "$src = $rt";
}

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 1
//===----------------------------------------------------------------------===//

// Addition/subtraction
class ADDU_QB_DESC : ADDU_QB_DESC_BASE<"addu.qb", int_mips_addu_qb, NoItinerary,
                                       DSPRegs, DSPRegs>, IsCommutable;

class ADDU_S_QB_DESC : ADDU_QB_DESC_BASE<"addu_s.qb", int_mips_addu_s_qb,
                                         NoItinerary, DSPRegs, DSPRegs>,
                       IsCommutable;

class SUBU_QB_DESC : ADDU_QB_DESC_BASE<"subu.qb", int_mips_subu_qb, NoItinerary,
                                       DSPRegs, DSPRegs>;

class SUBU_S_QB_DESC : ADDU_QB_DESC_BASE<"subu_s.qb", int_mips_subu_s_qb,
                                         NoItinerary, DSPRegs, DSPRegs>;

class ADDQ_PH_DESC : ADDU_QB_DESC_BASE<"addq.ph", int_mips_addq_ph, NoItinerary,
                                       DSPRegs, DSPRegs>, IsCommutable;

class ADDQ_S_PH_DESC : ADDU_QB_DESC_BASE<"addq_s.ph", int_mips_addq_s_ph,
                                         NoItinerary, DSPRegs, DSPRegs>,
                       IsCommutable;

class SUBQ_PH_DESC : ADDU_QB_DESC_BASE<"subq.ph", int_mips_subq_ph, NoItinerary,
                                       DSPRegs, DSPRegs>;

class SUBQ_S_PH_DESC : ADDU_QB_DESC_BASE<"subq_s.ph", int_mips_subq_s_ph,
                                         NoItinerary, DSPRegs, DSPRegs>;

class ADDQ_S_W_DESC : ADDU_QB_DESC_BASE<"addq_s.w", int_mips_addq_s_w,
                                        NoItinerary, CPURegs, CPURegs>,
                      IsCommutable;

class SUBQ_S_W_DESC : ADDU_QB_DESC_BASE<"subq_s.w", int_mips_subq_s_w,
                                        NoItinerary, CPURegs, CPURegs>;

class ADDSC_DESC : ADDU_QB_DESC_BASE<"addsc", int_mips_addsc, NoItinerary,
                                     CPURegs, CPURegs>, IsCommutable;

class ADDWC_DESC : ADDU_QB_DESC_BASE<"addwc", int_mips_addwc, NoItinerary,
                                     CPURegs, CPURegs>,
                   IsCommutable, UseDSPCtrl;

class MODSUB_DESC : ADDU_QB_DESC_BASE<"modsub", int_mips_modsub, NoItinerary,
                                      CPURegs, CPURegs>, ClearDefs;

class RADDU_W_QB_DESC : RADDU_W_QB_DESC_BASE<"raddu.w.qb", int_mips_raddu_w_qb,
                                             NoItinerary, CPURegs, DSPRegs>,
                        ClearDefs;

// Absolute value
class ABSQ_S_PH_DESC : ABSQ_S_PH_R2_DESC_BASE<"absq_s.ph", int_mips_absq_s_ph,
                                              NoItinerary, DSPRegs>;

class ABSQ_S_W_DESC : ABSQ_S_PH_R2_DESC_BASE<"absq_s.w", int_mips_absq_s_w,
                                             NoItinerary, CPURegs>;

// Precision reduce/expand
class PRECRQ_QB_PH_DESC : CMP_EQ_QB_R3_DESC_BASE<"precrq.qb.ph",
                                                 int_mips_precrq_qb_ph,
                                                 NoItinerary, DSPRegs, DSPRegs>,
                          ClearDefs;

class PRECRQ_PH_W_DESC : CMP_EQ_QB_R3_DESC_BASE<"precrq.ph.w",
                                                int_mips_precrq_ph_w,
                                                NoItinerary, DSPRegs, CPURegs>,
                         ClearDefs;

class PRECRQ_RS_PH_W_DESC : CMP_EQ_QB_R3_DESC_BASE<"precrq_rs.ph.w",
                                                   int_mips_precrq_rs_ph_w,
                                                   NoItinerary, DSPRegs,
                                                   CPURegs>;

class PRECRQU_S_QB_PH_DESC : CMP_EQ_QB_R3_DESC_BASE<"precrqu_s.qb.ph",
                                                    int_mips_precrqu_s_qb_ph,
                                                    NoItinerary, DSPRegs,
                                                    DSPRegs>;

class PRECEQ_W_PHL_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceq.w.phl",
                                                 int_mips_preceq_w_phl,
                                                 NoItinerary, CPURegs, DSPRegs>,
                          ClearDefs;

class PRECEQ_W_PHR_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceq.w.phr",
                                                 int_mips_preceq_w_phr,
                                                 NoItinerary, CPURegs, DSPRegs>,
                          ClearDefs;

class PRECEQU_PH_QBL_DESC : ABSQ_S_PH_R2_DESC_BASE<"precequ.ph.qbl",
                                                   int_mips_precequ_ph_qbl,
                                                   NoItinerary, DSPRegs>,
                            ClearDefs;

class PRECEQU_PH_QBR_DESC : ABSQ_S_PH_R2_DESC_BASE<"precequ.ph.qbr",
                                                   int_mips_precequ_ph_qbr,
                                                   NoItinerary, DSPRegs>,
                            ClearDefs;

class PRECEQU_PH_QBLA_DESC : ABSQ_S_PH_R2_DESC_BASE<"precequ.ph.qbla",
                                                    int_mips_precequ_ph_qbla,
                                                    NoItinerary, DSPRegs>,
                             ClearDefs;

class PRECEQU_PH_QBRA_DESC : ABSQ_S_PH_R2_DESC_BASE<"precequ.ph.qbra",
                                                    int_mips_precequ_ph_qbra,
                                                    NoItinerary, DSPRegs>,
                             ClearDefs;

class PRECEU_PH_QBL_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceu.ph.qbl",
                                                  int_mips_preceu_ph_qbl,
                                                  NoItinerary, DSPRegs>,
                           ClearDefs;

class PRECEU_PH_QBR_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceu.ph.qbr",
                                                  int_mips_preceu_ph_qbr,
                                                  NoItinerary, DSPRegs>,
                           ClearDefs;

class PRECEU_PH_QBLA_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceu.ph.qbla",
                                                   int_mips_preceu_ph_qbla,
                                                   NoItinerary, DSPRegs>,
                            ClearDefs;

class PRECEU_PH_QBRA_DESC : ABSQ_S_PH_R2_DESC_BASE<"preceu.ph.qbra",
                                                   int_mips_preceu_ph_qbra,
                                                   NoItinerary, DSPRegs>,
                            ClearDefs;

// Shift
class SHLL_QB_DESC : SHLL_QB_R2_DESC_BASE<"shll.qb", int_mips_shll_qb, immZExt3,
                                          NoItinerary, DSPRegs>;

class SHLLV_QB_DESC : SHLL_QB_R3_DESC_BASE<"shllv.qb", int_mips_shll_qb,
                                           NoItinerary, DSPRegs>;

class SHRL_QB_DESC : SHLL_QB_R2_DESC_BASE<"shrl.qb", int_mips_shrl_qb, immZExt3,
                                          NoItinerary, DSPRegs>, ClearDefs;

class SHRLV_QB_DESC : SHLL_QB_R3_DESC_BASE<"shrlv.qb", int_mips_shrl_qb,
                                           NoItinerary, DSPRegs>, ClearDefs;

class SHLL_PH_DESC : SHLL_QB_R2_DESC_BASE<"shll.ph", int_mips_shll_ph, immZExt4,
                                          NoItinerary, DSPRegs>;

class SHLLV_PH_DESC : SHLL_QB_R3_DESC_BASE<"shllv.ph", int_mips_shll_ph,
                                           NoItinerary, DSPRegs>;

class SHLL_S_PH_DESC : SHLL_QB_R2_DESC_BASE<"shll_s.ph", int_mips_shll_s_ph,
                                            immZExt4, NoItinerary, DSPRegs>;

class SHLLV_S_PH_DESC : SHLL_QB_R3_DESC_BASE<"shllv_s.ph", int_mips_shll_s_ph,
                                             NoItinerary, DSPRegs>;

class SHRA_PH_DESC : SHLL_QB_R2_DESC_BASE<"shra.ph", int_mips_shra_ph, immZExt4,
                                          NoItinerary, DSPRegs>, ClearDefs;

class SHRAV_PH_DESC : SHLL_QB_R3_DESC_BASE<"shrav.ph", int_mips_shra_ph,
                                           NoItinerary, DSPRegs>, ClearDefs;

class SHRA_R_PH_DESC : SHLL_QB_R2_DESC_BASE<"shra_r.ph", int_mips_shra_r_ph,
                                            immZExt4, NoItinerary, DSPRegs>,
                       ClearDefs;

class SHRAV_R_PH_DESC : SHLL_QB_R3_DESC_BASE<"shrav_r.ph", int_mips_shra_r_ph,
                                             NoItinerary, DSPRegs>, ClearDefs;

class SHLL_S_W_DESC : SHLL_QB_R2_DESC_BASE<"shll_s.w", int_mips_shll_s_w,
                                           immZExt5, NoItinerary, CPURegs>;

class SHLLV_S_W_DESC : SHLL_QB_R3_DESC_BASE<"shllv_s.w", int_mips_shll_s_w,
                                            NoItinerary, CPURegs>;

class SHRA_R_W_DESC : SHLL_QB_R2_DESC_BASE<"shra_r.w", int_mips_shra_r_w,
                                           immZExt5, NoItinerary, CPURegs>,
                      ClearDefs;

class SHRAV_R_W_DESC : SHLL_QB_R3_DESC_BASE<"shrav_r.w", int_mips_shra_r_w,
                                            NoItinerary, CPURegs>;

// Multiplication
class MULEU_S_PH_QBL_DESC : ADDU_QB_DESC_BASE<"muleu_s.ph.qbl",
                                              int_mips_muleu_s_ph_qbl,
                                              NoItinerary, DSPRegs, DSPRegs>;

class MULEU_S_PH_QBR_DESC : ADDU_QB_DESC_BASE<"muleu_s.ph.qbr",
                                              int_mips_muleu_s_ph_qbr,
                                              NoItinerary, DSPRegs, DSPRegs>;

class MULEQ_S_W_PHL_DESC : ADDU_QB_DESC_BASE<"muleq_s.w.phl",
                                             int_mips_muleq_s_w_phl,
                                             NoItinerary, CPURegs, DSPRegs>,
                           IsCommutable;

class MULEQ_S_W_PHR_DESC : ADDU_QB_DESC_BASE<"muleq_s.w.phr",
                                             int_mips_muleq_s_w_phr,
                                             NoItinerary, CPURegs, DSPRegs>,
                           IsCommutable;

class MULQ_RS_PH_DESC : ADDU_QB_DESC_BASE<"mulq_rs.ph", int_mips_mulq_rs_ph,
                                          NoItinerary, DSPRegs, DSPRegs>,
                        IsCommutable;

class MULSAQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"mulsaq_s.w.ph">;

class MAQ_S_W_PHL_DESC : DPA_W_PH_DESC_BASE<"maq_s.w.phl">;

class MAQ_S_W_PHR_DESC : DPA_W_PH_DESC_BASE<"maq_s.w.phr">;

class MAQ_SA_W_PHL_DESC : DPA_W_PH_DESC_BASE<"maq_sa.w.phl">;

class MAQ_SA_W_PHR_DESC : DPA_W_PH_DESC_BASE<"maq_sa.w.phr">;

// Dot product with accumulate/subtract
class DPAU_H_QBL_DESC : DPA_W_PH_DESC_BASE<"dpau.h.qbl">;

class DPAU_H_QBR_DESC : DPA_W_PH_DESC_BASE<"dpau.h.qbr">;

class DPSU_H_QBL_DESC : DPA_W_PH_DESC_BASE<"dpsu.h.qbl">;

class DPSU_H_QBR_DESC : DPA_W_PH_DESC_BASE<"dpsu.h.qbr">;

class DPAQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaq_s.w.ph">;

class DPSQ_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsq_s.w.ph">;

class DPAQ_SA_L_W_DESC : DPA_W_PH_DESC_BASE<"dpaq_sa.l.w">;

class DPSQ_SA_L_W_DESC : DPA_W_PH_DESC_BASE<"dpsq_sa.l.w">;

class MULT_DSP_DESC : MULT_DESC_BASE<"mult">;

class MULTU_DSP_DESC : MULT_DESC_BASE<"multu">;

class MADD_DSP_DESC : MULT_DESC_BASE<"madd">;

class MADDU_DSP_DESC : MULT_DESC_BASE<"maddu">;

class MSUB_DSP_DESC : MULT_DESC_BASE<"msub">;

class MSUBU_DSP_DESC : MULT_DESC_BASE<"msubu">;

// Comparison
class CMPU_EQ_QB_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmpu.eq.qb",
                                               int_mips_cmpu_eq_qb, NoItinerary,
                                               DSPRegs>, IsCommutable;

class CMPU_LT_QB_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmpu.lt.qb",
                                               int_mips_cmpu_lt_qb, NoItinerary,
                                               DSPRegs>, IsCommutable;

class CMPU_LE_QB_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmpu.le.qb",
                                               int_mips_cmpu_le_qb, NoItinerary,
                                               DSPRegs>, IsCommutable;

class CMPGU_EQ_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgu.eq.qb",
                                                int_mips_cmpgu_eq_qb,
                                                NoItinerary, CPURegs, DSPRegs>,
                         IsCommutable;

class CMPGU_LT_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgu.lt.qb",
                                                int_mips_cmpgu_lt_qb,
                                                NoItinerary, CPURegs, DSPRegs>,
                         IsCommutable;

class CMPGU_LE_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgu.le.qb",
                                                int_mips_cmpgu_le_qb,
                                                NoItinerary, CPURegs, DSPRegs>,
                         IsCommutable;

class CMP_EQ_PH_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmp.eq.ph", int_mips_cmp_eq_ph,
                                              NoItinerary, DSPRegs>,
                       IsCommutable;

class CMP_LT_PH_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmp.lt.ph", int_mips_cmp_lt_ph,
                                              NoItinerary, DSPRegs>,
                       IsCommutable;

class CMP_LE_PH_DESC : CMP_EQ_QB_R2_DESC_BASE<"cmp.le.ph", int_mips_cmp_le_ph,
                                              NoItinerary, DSPRegs>,
                       IsCommutable;

// Misc
class BITREV_DESC : ABSQ_S_PH_R2_DESC_BASE<"bitrev", int_mips_bitrev,
                                           NoItinerary, CPURegs>, ClearDefs;

class PACKRL_PH_DESC : CMP_EQ_QB_R3_DESC_BASE<"packrl.ph", int_mips_packrl_ph,
                                              NoItinerary, DSPRegs, DSPRegs>,
                       ClearDefs;

class REPL_QB_DESC : REPL_DESC_BASE<"repl.qb", int_mips_repl_qb, immZExt8,
                                    NoItinerary, DSPRegs>, ClearDefs;

class REPL_PH_DESC : REPL_DESC_BASE<"repl.ph", int_mips_repl_ph, immZExt10,
                                    NoItinerary, DSPRegs>, ClearDefs;

class REPLV_QB_DESC : ABSQ_S_PH_R2_DESC_BASE<"replv.qb", int_mips_repl_qb,
                                             NoItinerary, DSPRegs, CPURegs>,
                      ClearDefs;

class REPLV_PH_DESC : ABSQ_S_PH_R2_DESC_BASE<"replv.ph", int_mips_repl_ph,
                                             NoItinerary, DSPRegs, CPURegs>,
                      ClearDefs;

class PICK_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"pick.qb", int_mips_pick_qb,
                                            NoItinerary, DSPRegs, DSPRegs>,
                     ClearDefs, UseDSPCtrl;

class PICK_PH_DESC : CMP_EQ_QB_R3_DESC_BASE<"pick.ph", int_mips_pick_ph,
                                            NoItinerary, DSPRegs, DSPRegs>,
                     ClearDefs, UseDSPCtrl;

class LWX_DESC : LX_DESC_BASE<"lwx", int_mips_lwx, NoItinerary>, ClearDefs;

class LHX_DESC : LX_DESC_BASE<"lhx", int_mips_lhx, NoItinerary>, ClearDefs;

class LBUX_DESC : LX_DESC_BASE<"lbux", int_mips_lbux, NoItinerary>, ClearDefs;

class BPOSGE32_DESC : BPOSGE32_DESC_BASE<"bposge32", NoItinerary>;

// Extr
class EXTP_DESC : EXTR_W_TY1_R1_DESC_BASE<"extp", MipsEXTP, NoItinerary>;

class EXTPV_DESC : EXTR_W_TY1_R2_DESC_BASE<"extpv", MipsEXTP, NoItinerary>;

class EXTPDP_DESC : EXTR_W_TY1_R1_DESC_BASE<"extpdp", MipsEXTPDP, NoItinerary>;

class EXTPDPV_DESC : EXTR_W_TY1_R2_DESC_BASE<"extpdpv", MipsEXTPDP,
                                             NoItinerary>;

class EXTR_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr.w", MipsEXTR_W, NoItinerary>;

class EXTRV_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv.w", MipsEXTR_W,
                                             NoItinerary>;

class EXTR_R_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_r.w", MipsEXTR_R_W,
                                              NoItinerary>;

class EXTRV_R_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_r.w", MipsEXTR_R_W,
                                               NoItinerary>;

class EXTR_RS_W_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_rs.w", MipsEXTR_RS_W,
                                               NoItinerary>;

class EXTRV_RS_W_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_rs.w", MipsEXTR_RS_W,
                                                NoItinerary>;

class EXTR_S_H_DESC : EXTR_W_TY1_R1_DESC_BASE<"extr_s.h", MipsEXTR_S_H,
                                              NoItinerary>;

class EXTRV_S_H_DESC : EXTR_W_TY1_R2_DESC_BASE<"extrv_s.h", MipsEXTR_S_H,
                                               NoItinerary>;

class SHILO_DESC : SHILO_R1_DESC_BASE<"shilo">;

class SHILOV_DESC : SHILO_R2_DESC_BASE<"shilov">;

class MTHLIP_DESC : MTHLIP_DESC_BASE<"mthlip">;

class RDDSP_DESC : RDDSP_DESC_BASE<"rddsp", int_mips_rddsp, NoItinerary>;

class WRDSP_DESC : WRDSP_DESC_BASE<"wrdsp", int_mips_wrdsp, NoItinerary>;

class INSV_DESC : INSV_DESC_BASE<"insv", int_mips_insv, NoItinerary>;

//===----------------------------------------------------------------------===//
// MIPS DSP Rev 2
// Addition/subtraction
class ADDU_PH_DESC : ADDU_QB_DESC_BASE<"addu.ph", int_mips_addu_ph, NoItinerary,
                                       DSPRegs, DSPRegs>, IsCommutable;

class ADDU_S_PH_DESC : ADDU_QB_DESC_BASE<"addu_s.ph", int_mips_addu_s_ph,
                                         NoItinerary, DSPRegs, DSPRegs>,
                       IsCommutable;

class SUBU_PH_DESC : ADDU_QB_DESC_BASE<"subu.ph", int_mips_subu_ph, NoItinerary,
                                       DSPRegs, DSPRegs>;

class SUBU_S_PH_DESC : ADDU_QB_DESC_BASE<"subu_s.ph", int_mips_subu_s_ph,
                                         NoItinerary, DSPRegs, DSPRegs>;

class ADDUH_QB_DESC : ADDUH_QB_DESC_BASE<"adduh.qb", int_mips_adduh_qb,
                                         NoItinerary, DSPRegs>,
                      ClearDefs, IsCommutable;

class ADDUH_R_QB_DESC : ADDUH_QB_DESC_BASE<"adduh_r.qb", int_mips_adduh_r_qb,
                                           NoItinerary, DSPRegs>,
                        ClearDefs, IsCommutable;

class SUBUH_QB_DESC : ADDUH_QB_DESC_BASE<"subuh.qb", int_mips_subuh_qb,
                                         NoItinerary, DSPRegs>, ClearDefs;

class SUBUH_R_QB_DESC : ADDUH_QB_DESC_BASE<"subuh_r.qb", int_mips_subuh_r_qb,
                                           NoItinerary, DSPRegs>, ClearDefs;

class ADDQH_PH_DESC : ADDUH_QB_DESC_BASE<"addqh.ph", int_mips_addqh_ph,
                                         NoItinerary, DSPRegs>,
                      ClearDefs, IsCommutable;

class ADDQH_R_PH_DESC : ADDUH_QB_DESC_BASE<"addqh_r.ph", int_mips_addqh_r_ph,
                                           NoItinerary, DSPRegs>,
                        ClearDefs, IsCommutable;

class SUBQH_PH_DESC : ADDUH_QB_DESC_BASE<"subqh.ph", int_mips_subqh_ph,
                                         NoItinerary, DSPRegs>, ClearDefs;

class SUBQH_R_PH_DESC : ADDUH_QB_DESC_BASE<"subqh_r.ph", int_mips_subqh_r_ph,
                                           NoItinerary, DSPRegs>, ClearDefs;

class ADDQH_W_DESC : ADDUH_QB_DESC_BASE<"addqh.w", int_mips_addqh_w,
                                        NoItinerary, CPURegs>,
                     ClearDefs, IsCommutable;

class ADDQH_R_W_DESC : ADDUH_QB_DESC_BASE<"addqh_r.w", int_mips_addqh_r_w,
                                          NoItinerary, CPURegs>,
                       ClearDefs, IsCommutable;

class SUBQH_W_DESC : ADDUH_QB_DESC_BASE<"subqh.w", int_mips_subqh_w,
                                        NoItinerary, CPURegs>, ClearDefs;

class SUBQH_R_W_DESC : ADDUH_QB_DESC_BASE<"subqh_r.w", int_mips_subqh_r_w,
                                          NoItinerary, CPURegs>, ClearDefs;

// Comparison
class CMPGDU_EQ_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgdu.eq.qb",
                                                 int_mips_cmpgdu_eq_qb,
                                                 NoItinerary, CPURegs, DSPRegs>,
                          IsCommutable;

class CMPGDU_LT_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgdu.lt.qb",
                                                 int_mips_cmpgdu_lt_qb,
                                                 NoItinerary, CPURegs, DSPRegs>,
                          IsCommutable;

class CMPGDU_LE_QB_DESC : CMP_EQ_QB_R3_DESC_BASE<"cmpgdu.le.qb",
                                                 int_mips_cmpgdu_le_qb,
                                                 NoItinerary, CPURegs, DSPRegs>,
                          IsCommutable;

// Absolute
class ABSQ_S_QB_DESC : ABSQ_S_PH_R2_DESC_BASE<"absq_s.qb", int_mips_absq_s_qb,
                                              NoItinerary, DSPRegs>;

// Multiplication
class MUL_PH_DESC : ADDUH_QB_DESC_BASE<"mul.ph", int_mips_mul_ph, NoItinerary,
                                       DSPRegs>, IsCommutable;

class MUL_S_PH_DESC : ADDUH_QB_DESC_BASE<"mul_s.ph", int_mips_mul_s_ph,
                                         NoItinerary, DSPRegs>, IsCommutable;

class MULQ_S_W_DESC : ADDUH_QB_DESC_BASE<"mulq_s.w", int_mips_mulq_s_w,
                                         NoItinerary, CPURegs>, IsCommutable;

class MULQ_RS_W_DESC : ADDUH_QB_DESC_BASE<"mulq_rs.w", int_mips_mulq_rs_w,
                                          NoItinerary, CPURegs>, IsCommutable;

class MULQ_S_PH_DESC : ADDU_QB_DESC_BASE<"mulq_s.ph", int_mips_mulq_s_ph,
                                         NoItinerary, DSPRegs, DSPRegs>,
                       IsCommutable;

// Dot product with accumulate/subtract
class DPA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpa.w.ph">;

class DPS_W_PH_DESC : DPA_W_PH_DESC_BASE<"dps.w.ph">;

class DPAQX_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaqx_s.w.ph">;

class DPAQX_SA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpaqx_sa.w.ph">;

class DPAX_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpax.w.ph">;

class DPSX_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsx.w.ph">;

class DPSQX_S_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsqx_s.w.ph">;

class DPSQX_SA_W_PH_DESC : DPA_W_PH_DESC_BASE<"dpsqx_sa.w.ph">;

class MULSA_W_PH_DESC : DPA_W_PH_DESC_BASE<"mulsa.w.ph">;

// Precision reduce/expand
class PRECR_QB_PH_DESC : CMP_EQ_QB_R3_DESC_BASE<"precr.qb.ph",
                                                int_mips_precr_qb_ph,
                                                NoItinerary, DSPRegs, DSPRegs>;

class PRECR_SRA_PH_W_DESC : PRECR_SRA_PH_W_DESC_BASE<"precr_sra.ph.w",
                                                     int_mips_precr_sra_ph_w,
                                                     NoItinerary, DSPRegs,
                                                     CPURegs>, ClearDefs;

class PRECR_SRA_R_PH_W_DESC : PRECR_SRA_PH_W_DESC_BASE<"precr_sra_r.ph.w",
                                                      int_mips_precr_sra_r_ph_w,
                                                       NoItinerary, DSPRegs,
                                                       CPURegs>, ClearDefs;

// Shift
class SHRA_QB_DESC : SHLL_QB_R2_DESC_BASE<"shra.qb", int_mips_shra_qb, immZExt3,
                                          NoItinerary, DSPRegs>, ClearDefs;

class SHRAV_QB_DESC : SHLL_QB_R3_DESC_BASE<"shrav.qb", int_mips_shra_qb,
                                           NoItinerary, DSPRegs>, ClearDefs;

class SHRA_R_QB_DESC : SHLL_QB_R2_DESC_BASE<"shra_r.qb", int_mips_shra_r_qb,
                                            immZExt3, NoItinerary, DSPRegs>,
                       ClearDefs;

class SHRAV_R_QB_DESC : SHLL_QB_R3_DESC_BASE<"shrav_r.qb", int_mips_shra_r_qb,
                                             NoItinerary, DSPRegs>, ClearDefs;

class SHRL_PH_DESC : SHLL_QB_R2_DESC_BASE<"shrl.ph", int_mips_shrl_ph, immZExt4,
                                          NoItinerary, DSPRegs>, ClearDefs;

class SHRLV_PH_DESC : SHLL_QB_R3_DESC_BASE<"shrlv.ph", int_mips_shrl_ph,
                                           NoItinerary, DSPRegs>, ClearDefs;

// Misc
class APPEND_DESC : APPEND_DESC_BASE<"append", int_mips_append, immZExt5,
                                     NoItinerary>, ClearDefs;

class BALIGN_DESC : APPEND_DESC_BASE<"balign", int_mips_balign, immZExt2,
                                     NoItinerary>, ClearDefs;

class PREPEND_DESC : APPEND_DESC_BASE<"prepend", int_mips_prepend, immZExt5,
                                      NoItinerary>, ClearDefs;

// Pseudos.
def BPOSGE32_PSEUDO : BPOSGE32_PSEUDO_DESC_BASE<int_mips_bposge32, NoItinerary>;

// Instruction defs.
// MIPS DSP Rev 1
def ADDU_QB : ADDU_QB_ENC, ADDU_QB_DESC;
def ADDU_S_QB : ADDU_S_QB_ENC, ADDU_S_QB_DESC;
def SUBU_QB : SUBU_QB_ENC, SUBU_QB_DESC;
def SUBU_S_QB : SUBU_S_QB_ENC, SUBU_S_QB_DESC;
def ADDQ_PH : ADDQ_PH_ENC, ADDQ_PH_DESC;
def ADDQ_S_PH : ADDQ_S_PH_ENC, ADDQ_S_PH_DESC;
def SUBQ_PH : SUBQ_PH_ENC, SUBQ_PH_DESC;
def SUBQ_S_PH : SUBQ_S_PH_ENC, SUBQ_S_PH_DESC;
def ADDQ_S_W : ADDQ_S_W_ENC, ADDQ_S_W_DESC;
def SUBQ_S_W : SUBQ_S_W_ENC, SUBQ_S_W_DESC;
def ADDSC : ADDSC_ENC, ADDSC_DESC;
def ADDWC : ADDWC_ENC, ADDWC_DESC;
def MODSUB : MODSUB_ENC, MODSUB_DESC;
def RADDU_W_QB : RADDU_W_QB_ENC, RADDU_W_QB_DESC;
def ABSQ_S_PH : ABSQ_S_PH_ENC, ABSQ_S_PH_DESC;
def ABSQ_S_W : ABSQ_S_W_ENC, ABSQ_S_W_DESC;
def PRECRQ_QB_PH : PRECRQ_QB_PH_ENC, PRECRQ_QB_PH_DESC;
def PRECRQ_PH_W : PRECRQ_PH_W_ENC, PRECRQ_PH_W_DESC;
def PRECRQ_RS_PH_W : PRECRQ_RS_PH_W_ENC, PRECRQ_RS_PH_W_DESC;
def PRECRQU_S_QB_PH : PRECRQU_S_QB_PH_ENC, PRECRQU_S_QB_PH_DESC;
def PRECEQ_W_PHL : PRECEQ_W_PHL_ENC, PRECEQ_W_PHL_DESC;
def PRECEQ_W_PHR : PRECEQ_W_PHR_ENC, PRECEQ_W_PHR_DESC;
def PRECEQU_PH_QBL : PRECEQU_PH_QBL_ENC, PRECEQU_PH_QBL_DESC;
def PRECEQU_PH_QBR : PRECEQU_PH_QBR_ENC, PRECEQU_PH_QBR_DESC;
def PRECEQU_PH_QBLA : PRECEQU_PH_QBLA_ENC, PRECEQU_PH_QBLA_DESC;
def PRECEQU_PH_QBRA : PRECEQU_PH_QBRA_ENC, PRECEQU_PH_QBRA_DESC;
def PRECEU_PH_QBL : PRECEU_PH_QBL_ENC, PRECEU_PH_QBL_DESC;
def PRECEU_PH_QBR : PRECEU_PH_QBR_ENC, PRECEU_PH_QBR_DESC;
def PRECEU_PH_QBLA : PRECEU_PH_QBLA_ENC, PRECEU_PH_QBLA_DESC;
def PRECEU_PH_QBRA : PRECEU_PH_QBRA_ENC, PRECEU_PH_QBRA_DESC;
def SHLL_QB : SHLL_QB_ENC, SHLL_QB_DESC;
def SHLLV_QB : SHLLV_QB_ENC, SHLLV_QB_DESC;
def SHRL_QB : SHRL_QB_ENC, SHRL_QB_DESC;
def SHRLV_QB : SHRLV_QB_ENC, SHRLV_QB_DESC;
def SHLL_PH : SHLL_PH_ENC, SHLL_PH_DESC;
def SHLLV_PH : SHLLV_PH_ENC, SHLLV_PH_DESC;
def SHLL_S_PH : SHLL_S_PH_ENC, SHLL_S_PH_DESC;
def SHLLV_S_PH : SHLLV_S_PH_ENC, SHLLV_S_PH_DESC;
def SHRA_PH : SHRA_PH_ENC, SHRA_PH_DESC;
def SHRAV_PH : SHRAV_PH_ENC, SHRAV_PH_DESC;
def SHRA_R_PH : SHRA_R_PH_ENC, SHRA_R_PH_DESC;
def SHRAV_R_PH : SHRAV_R_PH_ENC, SHRAV_R_PH_DESC;
def SHLL_S_W : SHLL_S_W_ENC, SHLL_S_W_DESC;
def SHLLV_S_W : SHLLV_S_W_ENC, SHLLV_S_W_DESC;
def SHRA_R_W : SHRA_R_W_ENC, SHRA_R_W_DESC;
def SHRAV_R_W : SHRAV_R_W_ENC, SHRAV_R_W_DESC;
def MULEU_S_PH_QBL : MULEU_S_PH_QBL_ENC, MULEU_S_PH_QBL_DESC;
def MULEU_S_PH_QBR : MULEU_S_PH_QBR_ENC, MULEU_S_PH_QBR_DESC;
def MULEQ_S_W_PHL : MULEQ_S_W_PHL_ENC, MULEQ_S_W_PHL_DESC;
def MULEQ_S_W_PHR : MULEQ_S_W_PHR_ENC, MULEQ_S_W_PHR_DESC;
def MULQ_RS_PH : MULQ_RS_PH_ENC, MULQ_RS_PH_DESC;
def MULSAQ_S_W_PH : MULSAQ_S_W_PH_ENC, MULSAQ_S_W_PH_DESC;
def MAQ_S_W_PHL : MAQ_S_W_PHL_ENC, MAQ_S_W_PHL_DESC;
def MAQ_S_W_PHR : MAQ_S_W_PHR_ENC, MAQ_S_W_PHR_DESC;
def MAQ_SA_W_PHL : MAQ_SA_W_PHL_ENC, MAQ_SA_W_PHL_DESC;
def MAQ_SA_W_PHR : MAQ_SA_W_PHR_ENC, MAQ_SA_W_PHR_DESC;
def DPAU_H_QBL : DPAU_H_QBL_ENC, DPAU_H_QBL_DESC;
def DPAU_H_QBR : DPAU_H_QBR_ENC, DPAU_H_QBR_DESC;
def DPSU_H_QBL : DPSU_H_QBL_ENC, DPSU_H_QBL_DESC;
def DPSU_H_QBR : DPSU_H_QBR_ENC, DPSU_H_QBR_DESC;
def DPAQ_S_W_PH : DPAQ_S_W_PH_ENC, DPAQ_S_W_PH_DESC;
def DPSQ_S_W_PH : DPSQ_S_W_PH_ENC, DPSQ_S_W_PH_DESC;
def DPAQ_SA_L_W : DPAQ_SA_L_W_ENC, DPAQ_SA_L_W_DESC;
def DPSQ_SA_L_W : DPSQ_SA_L_W_ENC, DPSQ_SA_L_W_DESC;
def MULT_DSP : MULT_DSP_ENC, MULT_DSP_DESC;
def MULTU_DSP : MULTU_DSP_ENC, MULTU_DSP_DESC;
def MADD_DSP : MADD_DSP_ENC, MADD_DSP_DESC;
def MADDU_DSP : MADDU_DSP_ENC, MADDU_DSP_DESC;
def MSUB_DSP : MSUB_DSP_ENC, MSUB_DSP_DESC;
def MSUBU_DSP : MSUBU_DSP_ENC, MSUBU_DSP_DESC;
def CMPU_EQ_QB : CMPU_EQ_QB_ENC, CMPU_EQ_QB_DESC;
def CMPU_LT_QB : CMPU_LT_QB_ENC, CMPU_LT_QB_DESC;
def CMPU_LE_QB : CMPU_LE_QB_ENC, CMPU_LE_QB_DESC;
def CMPGU_EQ_QB : CMPGU_EQ_QB_ENC, CMPGU_EQ_QB_DESC;
def CMPGU_LT_QB : CMPGU_LT_QB_ENC, CMPGU_LT_QB_DESC;
def CMPGU_LE_QB : CMPGU_LE_QB_ENC, CMPGU_LE_QB_DESC;
def CMP_EQ_PH : CMP_EQ_PH_ENC, CMP_EQ_PH_DESC;
def CMP_LT_PH : CMP_LT_PH_ENC, CMP_LT_PH_DESC;
def CMP_LE_PH : CMP_LE_PH_ENC, CMP_LE_PH_DESC;
def BITREV : BITREV_ENC, BITREV_DESC;
def PACKRL_PH : PACKRL_PH_ENC, PACKRL_PH_DESC;
def REPL_QB : REPL_QB_ENC, REPL_QB_DESC;
def REPL_PH : REPL_PH_ENC, REPL_PH_DESC;
def REPLV_QB : REPLV_QB_ENC, REPLV_QB_DESC;
def REPLV_PH : REPLV_PH_ENC, REPLV_PH_DESC;
def PICK_QB : PICK_QB_ENC, PICK_QB_DESC;
def PICK_PH : PICK_PH_ENC, PICK_PH_DESC;
def LWX : LWX_ENC, LWX_DESC;
def LHX : LHX_ENC, LHX_DESC;
def LBUX : LBUX_ENC, LBUX_DESC;
def BPOSGE32 : BPOSGE32_ENC, BPOSGE32_DESC;
def INSV : INSV_ENC, INSV_DESC;
def EXTP : EXTP_ENC, EXTP_DESC;
def EXTPV : EXTPV_ENC, EXTPV_DESC;
def EXTPDP : EXTPDP_ENC, EXTPDP_DESC;
def EXTPDPV : EXTPDPV_ENC, EXTPDPV_DESC;
def EXTR_W : EXTR_W_ENC, EXTR_W_DESC;
def EXTRV_W : EXTRV_W_ENC, EXTRV_W_DESC;
def EXTR_R_W : EXTR_R_W_ENC, EXTR_R_W_DESC;
def EXTRV_R_W : EXTRV_R_W_ENC, EXTRV_R_W_DESC;
def EXTR_RS_W : EXTR_RS_W_ENC, EXTR_RS_W_DESC;
def EXTRV_RS_W : EXTRV_RS_W_ENC, EXTRV_RS_W_DESC;
def EXTR_S_H : EXTR_S_H_ENC, EXTR_S_H_DESC;
def EXTRV_S_H : EXTRV_S_H_ENC, EXTRV_S_H_DESC;
def SHILO : SHILO_ENC, SHILO_DESC;
def SHILOV : SHILOV_ENC, SHILOV_DESC;
def MTHLIP : MTHLIP_ENC, MTHLIP_DESC;
def RDDSP : RDDSP_ENC, RDDSP_DESC;
def WRDSP : WRDSP_ENC, WRDSP_DESC;

// MIPS DSP Rev 2
let Predicates = [HasDSPR2] in {

def ADDU_PH : ADDU_PH_ENC, ADDU_PH_DESC;
def ADDU_S_PH : ADDU_S_PH_ENC, ADDU_S_PH_DESC;
def SUBU_PH : SUBU_PH_ENC, SUBU_PH_DESC;
def SUBU_S_PH : SUBU_S_PH_ENC, SUBU_S_PH_DESC;
def CMPGDU_EQ_QB : CMPGDU_EQ_QB_ENC, CMPGDU_EQ_QB_DESC;
def CMPGDU_LT_QB : CMPGDU_LT_QB_ENC, CMPGDU_LT_QB_DESC;
def CMPGDU_LE_QB : CMPGDU_LE_QB_ENC, CMPGDU_LE_QB_DESC;
def ABSQ_S_QB : ABSQ_S_QB_ENC, ABSQ_S_QB_DESC;
def ADDUH_QB : ADDUH_QB_ENC, ADDUH_QB_DESC;
def ADDUH_R_QB : ADDUH_R_QB_ENC, ADDUH_R_QB_DESC;
def SUBUH_QB : SUBUH_QB_ENC, SUBUH_QB_DESC;
def SUBUH_R_QB : SUBUH_R_QB_ENC, SUBUH_R_QB_DESC;
def ADDQH_PH : ADDQH_PH_ENC, ADDQH_PH_DESC;
def ADDQH_R_PH : ADDQH_R_PH_ENC, ADDQH_R_PH_DESC;
def SUBQH_PH : SUBQH_PH_ENC, SUBQH_PH_DESC;
def SUBQH_R_PH : SUBQH_R_PH_ENC, SUBQH_R_PH_DESC;
def ADDQH_W : ADDQH_W_ENC, ADDQH_W_DESC;
def ADDQH_R_W : ADDQH_R_W_ENC, ADDQH_R_W_DESC;
def SUBQH_W : SUBQH_W_ENC, SUBQH_W_DESC;
def SUBQH_R_W : SUBQH_R_W_ENC, SUBQH_R_W_DESC;
def MUL_PH : MUL_PH_ENC, MUL_PH_DESC;
def MUL_S_PH : MUL_S_PH_ENC, MUL_S_PH_DESC;
def MULQ_S_W : MULQ_S_W_ENC, MULQ_S_W_DESC;
def MULQ_RS_W : MULQ_RS_W_ENC, MULQ_RS_W_DESC;
def MULQ_S_PH : MULQ_S_PH_ENC, MULQ_S_PH_DESC;
def DPA_W_PH : DPA_W_PH_ENC, DPA_W_PH_DESC;
def DPS_W_PH : DPS_W_PH_ENC, DPS_W_PH_DESC;
def DPAQX_S_W_PH : DPAQX_S_W_PH_ENC, DPAQX_S_W_PH_DESC;
def DPAQX_SA_W_PH : DPAQX_SA_W_PH_ENC, DPAQX_SA_W_PH_DESC;
def DPAX_W_PH : DPAX_W_PH_ENC, DPAX_W_PH_DESC;
def DPSX_W_PH : DPSX_W_PH_ENC, DPSX_W_PH_DESC;
def DPSQX_S_W_PH : DPSQX_S_W_PH_ENC, DPSQX_S_W_PH_DESC;
def DPSQX_SA_W_PH : DPSQX_SA_W_PH_ENC, DPSQX_SA_W_PH_DESC;
def MULSA_W_PH : MULSA_W_PH_ENC, MULSA_W_PH_DESC;
def PRECR_QB_PH : PRECR_QB_PH_ENC, PRECR_QB_PH_DESC;
def PRECR_SRA_PH_W : PRECR_SRA_PH_W_ENC, PRECR_SRA_PH_W_DESC;
def PRECR_SRA_R_PH_W : PRECR_SRA_R_PH_W_ENC, PRECR_SRA_R_PH_W_DESC;
def SHRA_QB : SHRA_QB_ENC, SHRA_QB_DESC;
def SHRAV_QB : SHRAV_QB_ENC, SHRAV_QB_DESC;
def SHRA_R_QB : SHRA_R_QB_ENC, SHRA_R_QB_DESC;
def SHRAV_R_QB : SHRAV_R_QB_ENC, SHRAV_R_QB_DESC;
def SHRL_PH : SHRL_PH_ENC, SHRL_PH_DESC;
def SHRLV_PH : SHRLV_PH_ENC, SHRLV_PH_DESC;
def APPEND : APPEND_ENC, APPEND_DESC;
def BALIGN : BALIGN_ENC, BALIGN_DESC;
def PREPEND : PREPEND_ENC, PREPEND_DESC;

}

// Pseudos.
def MULSAQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMULSAQ_S_W_PH, NoItinerary,
                                                MULSAQ_S_W_PH>;
def MAQ_S_W_PHL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_S_W_PHL, NoItinerary,
                                              MAQ_S_W_PHL>;
def MAQ_S_W_PHR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_S_W_PHR, NoItinerary,
                                              MAQ_S_W_PHR>;
def MAQ_SA_W_PHL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_SA_W_PHL, NoItinerary,
                                               MAQ_SA_W_PHL>;
def MAQ_SA_W_PHR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMAQ_SA_W_PHR, NoItinerary,
                                               MAQ_SA_W_PHR>;
def DPAU_H_QBL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAU_H_QBL, NoItinerary,
                                             DPAU_H_QBL>;
def DPAU_H_QBR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAU_H_QBR, NoItinerary,
                                             DPAU_H_QBR>;
def DPSU_H_QBL_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSU_H_QBL, NoItinerary,
                                             DPSU_H_QBL>;
def DPSU_H_QBR_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSU_H_QBR, NoItinerary,
                                             DPSU_H_QBR>;
def DPAQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQ_S_W_PH, NoItinerary,
                                              DPAQ_S_W_PH>;
def DPSQ_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQ_S_W_PH, NoItinerary,
                                              DPSQ_S_W_PH>;
def DPAQ_SA_L_W_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQ_SA_L_W, NoItinerary,
                                              DPAQ_SA_L_W>;
def DPSQ_SA_L_W_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQ_SA_L_W, NoItinerary,
                                              DPSQ_SA_L_W>;

def MULT_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMULT, NoItinerary, MULT_DSP>,
                      IsCommutable;
def MULTU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMULTU, NoItinerary, MULTU_DSP>,
                       IsCommutable;
def MADD_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMADD_DSP, NoItinerary, MADD_DSP>,
                      IsCommutable, UseAC;
def MADDU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMADDU_DSP, NoItinerary, MADDU_DSP>,
                       IsCommutable, UseAC;
def MSUB_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMSUB_DSP, NoItinerary, MSUB_DSP>,
                      UseAC;
def MSUBU_DSP_PSEUDO : MULT_PSEUDO_BASE<MipsMSUBU_DSP, NoItinerary, MSUBU_DSP>,
                       UseAC;

def SHILO_PSEUDO : SHILO_R1_PSEUDO_BASE<MipsSHILO, NoItinerary, SHILO>;
def SHILOV_PSEUDO : SHILO_R2_PSEUDO_BASE<MipsSHILO, NoItinerary, SHILOV>;
def MTHLIP_PSEUDO : SHILO_R2_PSEUDO_BASE<MipsMTHLIP, NoItinerary, MTHLIP>;

let Predicates = [HasDSPR2] in {

def DPA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPA_W_PH, NoItinerary, DPA_W_PH>;
def DPS_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPS_W_PH, NoItinerary, DPS_W_PH>;
def DPAQX_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQX_S_W_PH, NoItinerary,
                                               DPAQX_S_W_PH>;
def DPAQX_SA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAQX_SA_W_PH, NoItinerary,
                                                DPAQX_SA_W_PH>;
def DPAX_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPAX_W_PH, NoItinerary,
                                            DPAX_W_PH>;
def DPSX_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSX_W_PH, NoItinerary,
                                            DPSX_W_PH>;
def DPSQX_S_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQX_S_W_PH, NoItinerary,
                                               DPSQX_S_W_PH>;
def DPSQX_SA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsDPSQX_SA_W_PH, NoItinerary,
                                                DPSQX_SA_W_PH>;
def MULSA_W_PH_PSEUDO : DPA_W_PH_PSEUDO_BASE<MipsMULSA_W_PH, NoItinerary,
                                             MULSA_W_PH>;

}

// Patterns.
class DSPPat<dag pattern, dag result, Predicate pred = HasDSP> :
  Pat<pattern, result>, Requires<[pred]>;

class BitconvertPat<ValueType DstVT, ValueType SrcVT, RegisterClass DstRC,
                    RegisterClass SrcRC> :
   DSPPat<(DstVT (bitconvert (SrcVT SrcRC:$src))),
          (COPY_TO_REGCLASS SrcRC:$src, DstRC)>;

def : BitconvertPat<i32, v2i16, CPURegs, DSPRegs>;
def : BitconvertPat<i32, v4i8, CPURegs, DSPRegs>;
def : BitconvertPat<v2i16, i32, DSPRegs, CPURegs>;
def : BitconvertPat<v4i8, i32, DSPRegs, CPURegs>;

def : DSPPat<(v2i16 (load addr:$a)),
             (v2i16 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(v4i8 (load addr:$a)),
             (v4i8 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(store (v2i16 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;
def : DSPPat<(store (v4i8 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;

// Extr patterns.
class EXTR_W_TY1_R2_Pat<SDPatternOperator OpNode, Instruction Instr> :
  DSPPat<(i32 (OpNode CPURegs:$rs)), (Instr AC0, CPURegs:$rs)>;

class EXTR_W_TY1_R1_Pat<SDPatternOperator OpNode, Instruction Instr> :
  DSPPat<(i32 (OpNode immZExt5:$shift)), (Instr AC0, immZExt5:$shift)>;

def : EXTR_W_TY1_R1_Pat<MipsEXTP, EXTP>;
def : EXTR_W_TY1_R2_Pat<MipsEXTP, EXTPV>;
def : EXTR_W_TY1_R1_Pat<MipsEXTPDP, EXTPDP>;
def : EXTR_W_TY1_R2_Pat<MipsEXTPDP, EXTPDPV>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_W, EXTR_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_W, EXTRV_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_R_W, EXTR_R_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_R_W, EXTRV_R_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_RS_W, EXTR_RS_W>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_RS_W, EXTRV_RS_W>;
def : EXTR_W_TY1_R1_Pat<MipsEXTR_S_H, EXTR_S_H>;
def : EXTR_W_TY1_R2_Pat<MipsEXTR_S_H, EXTRV_S_H>;
