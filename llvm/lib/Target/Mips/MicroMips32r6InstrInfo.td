//=- MicroMips32r6InstrInfo.td - MicroMips r6 Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes microMIPSr6 instructions.
//
//===----------------------------------------------------------------------===//

def brtarget26_mm : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget26OpValueMM";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget26MM";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//
class ADD_MMR6_ENC : ARITH_FM_MMR6<"add", 0x110>;
class ADDIU_MMR6_ENC : ADDI_FM_MMR6<"addiu", 0xc>;
class ADDU_MMR6_ENC : ARITH_FM_MMR6<"addu", 0x150>;
class ADDIUPC_MMR6_ENC : PCREL19_FM_MMR6<0b00>;
class ALUIPC_MMR6_ENC : PCREL16_FM_MMR6<0b11111>;
class AND_MMR6_ENC : ARITH_FM_MMR6<"and", 0x250>;
class ANDI_MMR6_ENC : ADDI_FM_MMR6<"andi", 0x34>;
class AUIPC_MMR6_ENC  : PCREL16_FM_MMR6<0b11110>;
class ALIGN_MMR6_ENC : POOL32A_ALIGN_FM_MMR6<0b011111>;
class AUI_MMR6_ENC : AUI_FM_MMR6;
class BALC_MMR6_ENC  : BRANCH_OFF26_FM<0b101101>;
class BC_MMR6_ENC : BRANCH_OFF26_FM<0b100101>;
class BC16_MMR6_ENC : BC16_FM_MM16R6;
class BEQZC16_MMR6_ENC : BEQZC_BNEZC_FM_MM16R6<0x23>;
class BNEZC16_MMR6_ENC : BEQZC_BNEZC_FM_MM16R6<0x2b>;
class BITSWAP_MMR6_ENC : POOL32A_BITSWAP_FM_MMR6<0b101100>;
class BRK_MMR6_ENC : BREAK_MMR6_ENC<"break">;
class BEQZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b011101>;
class BNEZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b011111>;
class BGTZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b111000>;
class BLTZALC_MMR6_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM_MMR6<0b111000>;
class BGEZALC_MMR6_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM_MMR6<0b110000>;
class BLEZALC_MMR6_ENC : CMP_BRANCH_1R_RT_OFF16_FM_MMR6<0b110000>;
class CACHE_MMR6_ENC : CACHE_PREF_FM_MMR6<0b001000, 0b0110>;
class CLO_MMR6_ENC : POOL32A_2R_FM_MMR6<0b0100101100>;
class CLZ_MMR6_ENC : SPECIAL_2R_FM_MMR6<0b010000>;
class DIV_MMR6_ENC : ARITH_FM_MMR6<"div", 0x118>;
class DIVU_MMR6_ENC : ARITH_FM_MMR6<"divu", 0x198>;
class EHB_MMR6_ENC : BARRIER_MMR6_ENC<"ehb", 0x3>;
class EI_MMR6_ENC : POOL32A_EIDI_MMR6_ENC<"ei", 0x15d>;
class DI_MMR6_ENC : POOL32A_EIDI_MMR6_ENC<"di", 0b0100011101>;
class ERET_MMR6_ENC : POOL32A_ERET_FM_MMR6<"eret", 0x3cd>;
class DERET_MMR6_ENC : POOL32A_ERET_FM_MMR6<"eret", 0b1110001101>;
class ERETNC_MMR6_ENC : ERETNC_FM_MMR6<"eretnc">;
class JALRC16_MMR6_ENC : POOL16C_JALRC_FM_MM16R6<0xb>;
class JIALC_MMR6_ENC : JMP_IDX_COMPACT_FM<0b100000>;
class JIC_MMR6_ENC   : JMP_IDX_COMPACT_FM<0b101000>;
class JRC16_MMR6_ENC: POOL16C_JALRC_FM_MM16R6<0x3>;
class JRCADDIUSP_MMR6_ENC : POOL16C_JRCADDIUSP_FM_MM16R6<0x13>;
class LSA_MMR6_ENC : POOL32A_LSA_FM<0b001111>;
class LWPC_MMR6_ENC  : PCREL19_FM_MMR6<0b01>;
class LWM16_MMR6_ENC : POOL16C_LWM_SWM_FM_MM16R6<0x2>;
class MFC0_MMR6_ENC : POOL32A_MFTC0_FM_MMR6<"mfc0", 0b00011, 0b111100>;
class MFC1_MMR6_ENC : POOL32F_MFTC1_FM_MMR6<"mfc1", 0b10000000>;
class MFC2_MMR6_ENC : POOL32A_MFTC2_FM_MMR6<"mfc2", 0b0100110100>;
class MFHC0_MMR6_ENC : POOL32A_MFTC0_FM_MMR6<"mfhc0", 0b00011, 0b110100>;
class MFHC1_MMR6_ENC : POOL32F_MFTC1_FM_MMR6<"mfhc1", 0b11000000>;
class MFHC2_MMR6_ENC : POOL32A_MFTC2_FM_MMR6<"mfhc2", 0b1000110100>;
class MOD_MMR6_ENC : ARITH_FM_MMR6<"mod", 0x158>;
class MODU_MMR6_ENC : ARITH_FM_MMR6<"modu", 0x1d8>;
class MUL_MMR6_ENC : ARITH_FM_MMR6<"mul", 0x18>;
class MUH_MMR6_ENC : ARITH_FM_MMR6<"muh", 0x58>;
class MULU_MMR6_ENC : ARITH_FM_MMR6<"mulu", 0x98>;
class MUHU_MMR6_ENC : ARITH_FM_MMR6<"muhu", 0xd8>;
class MTC0_MMR6_ENC : POOL32A_MFTC0_FM_MMR6<"mtc0", 0b01011, 0b111100>;
class MTC1_MMR6_ENC : POOL32F_MFTC1_FM_MMR6<"mtc1", 0b10100000>;
class MTC2_MMR6_ENC : POOL32A_MFTC2_FM_MMR6<"mtc2", 0b0101110100>;
class MTHC0_MMR6_ENC : POOL32A_MFTC0_FM_MMR6<"mthc0", 0b01011, 0b110100>;
class MTHC1_MMR6_ENC : POOL32F_MFTC1_FM_MMR6<"mthc1", 0b11100000>;
class MTHC2_MMR6_ENC : POOL32A_MFTC2_FM_MMR6<"mthc2", 0b1001110100>;
class NOR_MMR6_ENC : ARITH_FM_MMR6<"nor", 0x2d0>;
class OR_MMR6_ENC : ARITH_FM_MMR6<"or", 0x290>;
class ORI_MMR6_ENC : ADDI_FM_MMR6<"ori", 0x14>;
class PREF_MMR6_ENC : CACHE_PREF_FM_MMR6<0b011000, 0b0010>;
class SB16_MMR6_ENC : LOAD_STORE_FM_MM16<0x22>;
class SEB_MMR6_ENC : SIGN_EXTEND_FM_MMR6<"seb", 0b0010101100>;
class SEH_MMR6_ENC : SIGN_EXTEND_FM_MMR6<"seh", 0b0011101100>;
class SELEQZ_MMR6_ENC : POOL32A_FM_MMR6<0b0101000000>;
class SELNEZ_MMR6_ENC : POOL32A_FM_MMR6<0b0110000000>;
class SH16_MMR6_ENC : LOAD_STORE_FM_MM16<0x2a>;
class SLL_MMR6_ENC : SHIFT_MMR6_ENC<"sll", 0x00, 0b0>;
class SUB_MMR6_ENC : ARITH_FM_MMR6<"sub", 0x190>;
class SUBU_MMR6_ENC : ARITH_FM_MMR6<"subu", 0x1d0>;
class SW_MMR6_ENC : SW32_FM_MMR6<"sw", 0x3e>;
class SWE_MMR6_ENC : POOL32C_SWE_FM_MMR6<"swe", 0x18, 0xa, 0x7>;
class SW16_MMR6_ENC : LOAD_STORE_FM_MM16<0x3a>;
class SWM16_MMR6_ENC : POOL16C_LWM_SWM_FM_MM16R6<0xa>;
class SWSP_MMR6_ENC : LOAD_STORE_SP_FM_MM16<0x32>;
class PREFE_MMR6_ENC : POOL32C_ST_EVA_FM_MMR6<0b011000, 0b010>;
class CACHEE_MMR6_ENC : POOL32C_ST_EVA_FM_MMR6<0b011000, 0b011>;
class WRPGPR_MMR6_ENC : POOL32A_WRPGPR_WSBH_FM_MMR6<0x3c5>;
class WSBH_MMR6_ENC : POOL32A_WRPGPR_WSBH_FM_MMR6<0x1ec>;
class LB_MMR6_ENC : LB32_FM_MMR6;
class LBU_MMR6_ENC : LBU32_FM_MMR6;
class LBE_MMR6_ENC : POOL32C_LB_LBU_FM_MMR6<0b100>;
class LBUE_MMR6_ENC : POOL32C_LB_LBU_FM_MMR6<0b000>;
class PAUSE_MMR6_ENC : POOL32A_PAUSE_FM_MMR6<"pause", 0b00101>;
class RDHWR_MMR6_ENC : POOL32A_RDHWR_FM_MMR6;
class WAIT_MMR6_ENC : WAIT_FM_MM, MMR6Arch<"wait">;
class SSNOP_MMR6_ENC : BARRIER_FM_MM<0x1>, MMR6Arch<"ssnop">;
class SYNC_MMR6_ENC : POOL32A_SYNC_FM_MMR6;
class SYNCI_MMR6_ENC : POOL32I_SYNCI_FM_MMR6, MMR6Arch<"synci">;
class RDPGPR_MMR6_ENC : POOL32A_RDPGPR_FM_MMR6<0b1110000101>;
class SDBBP_MMR6_ENC : SDBBP_FM_MM, MMR6Arch<"sdbbp">;
class XOR_MMR6_ENC : ARITH_FM_MMR6<"xor", 0x310>;
class XORI_MMR6_ENC : ADDI_FM_MMR6<"xori", 0x1c>;
class ABS_S_MMR6_ENC : POOL32F_ABS_FM_MMR6<"abs.s", 0, 0b0001101>;
class ABS_D_MMR6_ENC : POOL32F_ABS_FM_MMR6<"abs.d", 1, 0b0001101>;
class FLOOR_L_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"floor.l.s", 0, 0b00001100>;
class FLOOR_L_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"floor.l.d", 1, 0b00001100>;
class FLOOR_W_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"floor.w.s", 0, 0b00101100>;
class FLOOR_W_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"floor.w.d", 1, 0b00101100>;
class CEIL_L_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"ceil.l.s", 0, 0b01001100>;
class CEIL_L_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"ceil.l.d", 1, 0b01001100>;
class CEIL_W_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"ceil.w.s", 0, 0b01101100>;
class CEIL_W_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"ceil.w.d", 1, 0b01101100>;
class TRUNC_L_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"trunc.l.s", 0, 0b10001100>;
class TRUNC_L_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"trunc.l.d", 1, 0b10001100>;
class TRUNC_W_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"trunc.w.s", 0, 0b10101100>;
class TRUNC_W_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"trunc.w.d", 1, 0b10101100>;
class SQRT_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"sqrt.s", 0, 0b00101000>;
class SQRT_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"sqrt.d", 1, 0b00101000>;
class RSQRT_S_MMR6_ENC : POOL32F_MATH_FM_MMR6<"rsqrt.s", 0, 0b00001000>;
class RSQRT_D_MMR6_ENC : POOL32F_MATH_FM_MMR6<"rsqrt.d", 1, 0b00001000>;
class SB_MMR6_ENC : SB32_SH32_STORE_FM_MMR6<0b000110>;
class SBE_MMR6_ENC : POOL32C_STORE_EVA_FM_MMR6<0b100>;
class SCE_MMR6_ENC : POOL32C_STORE_EVA_FM_MMR6<0b110>;
class SH_MMR6_ENC : SB32_SH32_STORE_FM_MMR6<0b001110>;
class SHE_MMR6_ENC : POOL32C_STORE_EVA_FM_MMR6<0b101>;
class LLE_MMR6_ENC : LOAD_WORD_EVA_FM_MMR6<0b110>;
class LWE_MMR6_ENC : LOAD_WORD_EVA_FM_MMR6<0b111>;
class LW_MMR6_ENC : LOAD_WORD_FM_MMR6;
class LUI_MMR6_ENC : LOAD_UPPER_IMM_FM_MMR6;
class RECIP_S_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"recip.s", 0, 0b01001000>;
class RECIP_D_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"recip.d", 1, 0b01001000>;
class RINT_S_MMR6_ENC : POOL32F_RINT_FM_MMR6<"rint.s", 0>;
class RINT_D_MMR6_ENC : POOL32F_RINT_FM_MMR6<"rint.d", 1>;
class ROUND_L_S_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"round.l.s", 0,
                                                       0b11001100>;
class ROUND_L_D_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"round.l.d", 1,
                                                       0b11001100>;
class ROUND_W_S_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"round.w.s", 0,
                                                       0b11101100>;
class ROUND_W_D_MMR6_ENC : POOL32F_RECIP_ROUND_FM_MMR6<"round.w.d", 1,
                                                       0b11101100>;
class SEL_S_MMR6_ENC : POOL32F_SEL_FM_MMR6<"sel.s", 0, 0b010111000>;
class SEL_D_MMR6_ENC : POOL32F_SEL_FM_MMR6<"sel.d", 1, 0b010111000>;
class SELEQZ_S_MMR6_ENC : POOL32F_SEL_FM_MMR6<"seleqz.s", 0, 0b000111000>;
class SELEQZ_D_MMR6_ENC : POOL32F_SEL_FM_MMR6<"seleqz.d", 1, 0b000111000>;
class SELENZ_S_MMR6_ENC : POOL32F_SEL_FM_MMR6<"selenz.s", 0, 0b001111000>;
class SELENZ_D_MMR6_ENC : POOL32F_SEL_FM_MMR6<"selenz.d", 1, 0b001111000>;
class CLASS_S_MMR6_ENC : POOL32F_CLASS_FM_MMR6<"class.s", 0, 0b001100000>;
class CLASS_D_MMR6_ENC : POOL32F_CLASS_FM_MMR6<"class.d", 1, 0b001100000>;

class ADDU16_MMR6_ENC : POOL16A_ADDU16_FM_MMR6;
class AND16_MMR6_ENC : POOL16C_AND16_FM_MMR6;
class ANDI16_MMR6_ENC : ANDI_FM_MM16<0b001011>, MicroMipsR6Inst16;
class NOT16_MMR6_ENC : POOL16C_NOT16_FM_MMR6;
class OR16_MMR6_ENC : POOL16C_OR16_XOR16_FM_MMR6<0b1001>;
class SLL16_MMR6_ENC : SHIFT_FM_MM16<0>, MicroMipsR6Inst16;
class SRL16_MMR6_ENC : SHIFT_FM_MM16<1>, MicroMipsR6Inst16;
class BREAK16_MMR6_ENC : POOL16C_BREAKPOINT_FM_MMR6<0b011011>;
class LI16_MMR6_ENC : LI_FM_MM16;
class MOVE16_MMR6_ENC : MOVE_FM_MM16<0b000011>;
class SDBBP16_MMR6_ENC : POOL16C_BREAKPOINT_FM_MMR6<0b111011>;
class SUBU16_MMR6_ENC : POOL16A_SUBU16_FM_MMR6;
class XOR16_MMR6_ENC : POOL16C_OR16_XOR16_FM_MMR6<0b1000>;
class TLBINV_MMR6_ENC : POOL32A_TLBINV_FM_MMR6<"tlbinv", 0x10d>;
class TLBINVF_MMR6_ENC : POOL32A_TLBINV_FM_MMR6<"tlbinvf", 0x14d>;

class CMP_CBR_RT_Z_MMR6_DESC_BASE<string instr_asm, DAGOperand opnd,
                                  RegisterOperand GPROpnd>
    : BRANCH_DESC_BASE, MMR6Arch<instr_asm> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $offset");
  list<Register> Defs = [AT];
}

class BEQZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"beqzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bgezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGTZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bgtzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"blezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLTZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bltzalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BNEZALC_MMR6_DESC : CMP_CBR_RT_Z_MMR6_DESC_BASE<"bnezalc", brtarget_mm,
                                                      GPR32Opnd> {
  list<Register> Defs = [RA];
}

/// Floating Point Instructions
class FADD_S_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"add.s", 0, 0b00110000>;
class FADD_D_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"add.d", 1, 0b00110000>;
class FSUB_S_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"sub.s", 0, 0b01110000>;
class FSUB_D_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"sub.d", 1, 0b01110000>;
class FMUL_S_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"mul.s", 0, 0b10110000>;
class FMUL_D_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"mul.d", 1, 0b10110000>;
class FDIV_S_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"div.s", 0, 0b11110000>;
class FDIV_D_MMR6_ENC : POOL32F_ARITH_FM_MMR6<"div.d", 1, 0b11110000>;
class MADDF_S_MMR6_ENC : POOL32F_ARITHF_FM_MMR6<"maddf.s", 0, 0b110111000>;
class MADDF_D_MMR6_ENC : POOL32F_ARITHF_FM_MMR6<"maddf.d", 1, 0b110111000>;
class MSUBF_S_MMR6_ENC : POOL32F_ARITHF_FM_MMR6<"msubf.s", 0, 0b111111000>;
class MSUBF_D_MMR6_ENC : POOL32F_ARITHF_FM_MMR6<"msubf.d", 1, 0b111111000>;
class FMOV_S_MMR6_ENC : POOL32F_MOV_NEG_FM_MMR6<"mov.s", 0, 0b0000001>;
class FMOV_D_MMR6_ENC : POOL32F_MOV_NEG_FM_MMR6<"mov.d", 1, 0b0000001>;
class FNEG_S_MMR6_ENC : POOL32F_MOV_NEG_FM_MMR6<"neg.s", 0, 0b0101101>;
class FNEG_D_MMR6_ENC : POOL32F_MOV_NEG_FM_MMR6<"neg.d", 1, 0b0101101>;
class MAX_S_MMR6_ENC : POOL32F_MINMAX_FM<"max.s", 0, 0b000001011>;
class MAX_D_MMR6_ENC : POOL32F_MINMAX_FM<"max.d", 1, 0b000001011>;
class MAXA_S_MMR6_ENC : POOL32F_MINMAX_FM<"maxa.s", 0, 0b000101011>;
class MAXA_D_MMR6_ENC : POOL32F_MINMAX_FM<"maxa.d", 1, 0b000101011>;
class MIN_S_MMR6_ENC : POOL32F_MINMAX_FM<"min.s", 0, 0b000000011>;
class MIN_D_MMR6_ENC : POOL32F_MINMAX_FM<"min.d", 1, 0b000000011>;
class MINA_S_MMR6_ENC : POOL32F_MINMAX_FM<"mina.s", 0, 0b000100011>;
class MINA_D_MMR6_ENC : POOL32F_MINMAX_FM<"mina.d", 1, 0b000100011>;

class CVT_L_S_MMR6_ENC : POOL32F_CVT_LW_FM<"cvt.l.s", 0, 0b00000100>;
class CVT_L_D_MMR6_ENC : POOL32F_CVT_LW_FM<"cvt.l.d", 1, 0b00000100>;
class CVT_W_S_MMR6_ENC : POOL32F_CVT_LW_FM<"cvt.w.s", 0, 0b00100100>;
class CVT_W_D_MMR6_ENC : POOL32F_CVT_LW_FM<"cvt.w.d", 1, 0b00100100>;
class CVT_D_S_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.d.s", 0, 0b1001101>;
class CVT_D_W_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.d.w", 1, 0b1001101>;
class CVT_D_L_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.d.l", 2, 0b1001101>;
class CVT_S_D_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.s.d", 0, 0b1101101>;
class CVT_S_W_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.s.w", 1, 0b1101101>;
class CVT_S_L_MMR6_ENC : POOL32F_CVT_DS_FM<"cvt.s.l", 2, 0b1101101>;

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class ADD_MMR6_DESC : ArithLogicR<"add", GPR32Opnd>;
class ADDIU_MMR6_DESC : ArithLogicI<"addiu", simm16, GPR32Opnd>;
class ADDU_MMR6_DESC : ArithLogicR<"addu", GPR32Opnd>;
class MUL_MMR6_DESC : ArithLogicR<"mul", GPR32Opnd>;
class MUH_MMR6_DESC : ArithLogicR<"muh", GPR32Opnd>;
class MULU_MMR6_DESC : ArithLogicR<"mulu", GPR32Opnd>;
class MUHU_MMR6_DESC : ArithLogicR<"muhu", GPR32Opnd>;

class BC_MMR6_DESC_BASE<string instr_asm, DAGOperand opnd>
    : BRANCH_DESC_BASE, MMR6Arch<instr_asm> {
  dag InOperandList = (ins opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  bit isBarrier = 1;
}

class BALC_MMR6_DESC : BC_MMR6_DESC_BASE<"balc", brtarget26_mm> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}
class BC_MMR6_DESC : BC_MMR6_DESC_BASE<"bc", brtarget26_mm>;

class BC16_MMR6_DESC : MicroMipsInst16<(outs), (ins brtarget10_mm:$offset),
                                       !strconcat("bc16", "\t$offset"), [],
                                       II_BC, FrmI>,
                       MMR6Arch<"bc16">, MicroMipsR6Inst16 {
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
  let hasDelaySlot = 0;
  let AdditionalPredicates = [RelocPIC];
  let Defs = [AT];
}

class BEQZC_BNEZC_MM16R6_DESC_BASE<string instr_asm>
    : CBranchZeroMM<instr_asm, brtarget7_mm, GPRMM16Opnd>, MMR6Arch<instr_asm> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 0;
  let Defs = [AT];
}
class BEQZC16_MMR6_DESC : BEQZC_BNEZC_MM16R6_DESC_BASE<"beqzc16">;
class BNEZC16_MMR6_DESC : BEQZC_BNEZC_MM16R6_DESC_BASE<"bnezc16">;

class SUB_MMR6_DESC : ArithLogicR<"sub", GPR32Opnd>;
class SUBU_MMR6_DESC : ArithLogicR<"subu", GPR32Opnd>;

class BITSWAP_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt");
  list<dag> Pattern = [];
}

class BITSWAP_MMR6_DESC : BITSWAP_MMR6_DESC_BASE<"bitswap", GPR32Opnd>;

class BRK_MMR6_DESC : BRK_FT<"break">;

class CACHE_HINT_MMR6_DESC<string instr_asm, Operand MemOpnd,
                           RegisterOperand GPROpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins MemOpnd:$addr, uimm5:$hint);
  string AsmString = !strconcat(instr_asm, "\t$hint, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeCacheOpMM";
}

class CACHE_MMR6_DESC : CACHE_HINT_MMR6_DESC<"cache", mem_mm_12, GPR32Opnd>;
class PREF_MMR6_DESC : CACHE_HINT_MMR6_DESC<"pref", mem_mm_12, GPR32Opnd>;

class PREFE_CACHEE_MMR6_DESC_BASE<string instr_asm, Operand MemOpnd,
                                  RegisterOperand GPROpnd> :
                                  CACHE_HINT_MMR6_DESC<instr_asm, MemOpnd,
                                  GPROpnd> {
  string DecoderMethod = "DecodePrefeOpMM";
}

class PREFE_MMR6_DESC : PREFE_CACHEE_MMR6_DESC_BASE<"prefe", mem_mm_9, GPR32Opnd>;
class CACHEE_MMR6_DESC : PREFE_CACHEE_MMR6_DESC_BASE<"cachee", mem_mm_9, GPR32Opnd>;

class LB_LBU_MMR6_DESC_BASE<string instr_asm, Operand MemOpnd,
                            RegisterOperand GPROpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  string DecoderMethod = "DecodeLoadByte15";
  bit mayLoad = 1;
}
class LB_MMR6_DESC : LB_LBU_MMR6_DESC_BASE<"lb", mem_mm_16, GPR32Opnd>;
class LBU_MMR6_DESC : LB_LBU_MMR6_DESC_BASE<"lbu", mem_mm_16, GPR32Opnd>;

class LBE_LBUE_MMR6_DESC_BASE<string instr_asm, Operand MemOpnd,
                              RegisterOperand GPROpnd>
    : LB_LBU_MMR6_DESC_BASE<instr_asm, MemOpnd, GPROpnd> {
  let DecoderMethod = "DecodeLoadByte9";
}
class LBE_MMR6_DESC : LBE_LBUE_MMR6_DESC_BASE<"lbe", mem_mm_9, GPR32Opnd>;
class LBUE_MMR6_DESC : LBE_LBUE_MMR6_DESC_BASE<"lbue", mem_mm_9, GPR32Opnd>;

class CLO_CLZ_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs");
}

class CLO_MMR6_DESC : CLO_CLZ_MMR6_DESC_BASE<"clo", GPR32Opnd>;
class CLZ_MMR6_DESC : CLO_CLZ_MMR6_DESC_BASE<"clz", GPR32Opnd>;

class EHB_MMR6_DESC : Barrier<"ehb">;
class EI_MMR6_DESC : DEI_FT<"ei", GPR32Opnd>;
class DI_MMR6_DESC : DEI_FT<"di", GPR32Opnd>;

class ERET_MMR6_DESC : ER_FT<"eret">;
class DERET_MMR6_DESC : ER_FT<"deret">;
class ERETNC_MMR6_DESC : ER_FT<"eretnc">;

class JALRC16_MMR6_DESC_BASE<string opstr, RegisterOperand RO>
    : MicroMipsInst16<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"),
                      [(MipsJmpLink RO:$rs)], II_JALR, FrmR>,
      MMR6Arch<opstr>, MicroMipsR6Inst16 {
  let isCall = 1;
  let hasDelaySlot = 0;
  let Defs = [RA];
}
class JALRC16_MMR6_DESC : JALRC16_MMR6_DESC_BASE<"jalr", GPR32Opnd>;

class JMP_MMR6_IDX_COMPACT_DESC_BASE<string opstr, DAGOperand opnd,
                                     RegisterOperand GPROpnd>
    : MMR6Arch<opstr> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  string AsmString = !strconcat(opstr, "\t$rt, $offset");
  list<dag> Pattern = [];
  bit isTerminator = 1;
  bit hasDelaySlot = 0;
}

class JIALC_MMR6_DESC : JMP_MMR6_IDX_COMPACT_DESC_BASE<"jialc", calloffset16,
                                                       GPR32Opnd> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class JIC_MMR6_DESC : JMP_MMR6_IDX_COMPACT_DESC_BASE<"jic", jmpoffset16,
                                                     GPR32Opnd> {
  bit isBarrier = 1;
  list<Register> Defs = [AT];
}

class JRC16_MMR6_DESC_BASE<string opstr, RegisterOperand RO>
    : MicroMipsInst16<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"),
                      [], II_JR, FrmR>,
      MMR6Arch<opstr>, MicroMipsR6Inst16 {
  let hasDelaySlot = 0;
  let isBranch = 1;
  let isIndirectBranch = 1;
}
class JRC16_MMR6_DESC : JRC16_MMR6_DESC_BASE<"jrc16", GPR32Opnd>;

class JRCADDIUSP_MMR6_DESC
    : MicroMipsInst16<(outs), (ins uimm5_lsl2:$imm), "jrcaddiusp\t$imm",
                      [], II_JRADDIUSP, FrmR>,
      MMR6Arch<"jrcaddiusp">, MicroMipsR6Inst16 {
  let hasDelaySlot = 0;
  let isTerminator = 1;
  let isBarrier = 1;
  let isBranch = 1;
  let isIndirectBranch = 1;
}

class ALIGN_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd>  : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$bp);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $bp");
  list<dag> Pattern = [];
}

class ALIGN_MMR6_DESC : ALIGN_MMR6_DESC_BASE<"align", GPR32Opnd, uimm2>;

class AUI_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $imm");
  list<dag> Pattern = [];
}

class AUI_MMR6_DESC : AUI_MMR6_DESC_BASE<"aui", GPR32Opnd>;

class SEB_MMR6_DESC : SignExtInReg<"seb", i8, GPR32Opnd, II_SEB>;
class SEH_MMR6_DESC : SignExtInReg<"seh", i16, GPR32Opnd, II_SEH>;
class ALUIPC_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  list<dag> Pattern = [];
}

class ALUIPC_MMR6_DESC : ALUIPC_MMR6_DESC_BASE<"aluipc", GPR32Opnd>;
class AUIPC_MMR6_DESC : ALUIPC_MMR6_DESC_BASE<"auipc", GPR32Opnd>;

class LSA_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                         Operand ImmOpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$imm2);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $rd, $imm2");
  list<dag> Pattern = [];
}

class LSA_MMR6_DESC : LSA_MMR6_DESC_BASE<"lsa", GPR32Opnd, uimm2_plus1>;

class PCREL_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                           Operand ImmOpnd> : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins ImmOpnd:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rt, $imm");
  list<dag> Pattern = [];
}

class ADDIUPC_MMR6_DESC : PCREL_MMR6_DESC_BASE<"addiupc", GPR32Opnd, simm19_lsl2>;
class LWPC_MMR6_DESC: PCREL_MMR6_DESC_BASE<"lwpc", GPR32Opnd, simm19_lsl2>;

class SELEQNE_Z_MMR6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MMR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [];
}

class SELEQZ_MMR6_DESC : SELEQNE_Z_MMR6_DESC_BASE<"seleqz", GPR32Opnd>;
class SELNEZ_MMR6_DESC : SELEQNE_Z_MMR6_DESC_BASE<"selnez", GPR32Opnd>;
class PAUSE_MMR6_DESC : Barrier<"pause">;
class RDHWR_MMR6_DESC : MMR6Arch<"rdhwr">, MipsR6Inst {
  dag OutOperandList = (outs GPR32Opnd:$rt);
  dag InOperandList = (ins HWRegsOpnd:$rs, uimm3:$sel);
  string AsmString = !strconcat("rdhwr", "\t$rt, $rs, $sel");
  list<dag> Pattern = [];
  InstrItinClass Itinerary = II_RDHWR;
  Format Form = FrmR;
}

class WAIT_MMR6_DESC : WaitMM<"wait">;
class SSNOP_MMR6_DESC : Barrier<"ssnop">;
class SLL_MMR6_DESC : shift_rotate_imm<"sll", uimm5, GPR32Opnd, II_SLL>;
class DIV_MMR6_DESC : ArithLogicR<"div", GPR32Opnd>;
class DIVU_MMR6_DESC : ArithLogicR<"divu", GPR32Opnd>;
class MOD_MMR6_DESC : ArithLogicR<"mod", GPR32Opnd>;
class MODU_MMR6_DESC : ArithLogicR<"modu", GPR32Opnd>;
class AND_MMR6_DESC : ArithLogicR<"and", GPR32Opnd>;
class ANDI_MMR6_DESC : ArithLogicI<"andi", uimm16, GPR32Opnd>;
class NOR_MMR6_DESC : ArithLogicR<"nor", GPR32Opnd>;
class OR_MMR6_DESC : ArithLogicR<"or", GPR32Opnd>;
class ORI_MMR6_DESC : ArithLogicI<"ori", uimm16, GPR32Opnd>;
class XOR_MMR6_DESC : ArithLogicR<"xor", GPR32Opnd>;
class XORI_MMR6_DESC : ArithLogicI<"xori", uimm16, GPR32Opnd>;

class SWE_MMR6_DESC_BASE<string opstr, DAGOperand RO, DAGOperand MO,
                  SDPatternOperator OpNode = null_frag,
                  InstrItinClass Itin = NoItinerary,
                  ComplexPattern Addr = addr> :
  InstSE<(outs), (ins RO:$rt, MO:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, Addr:$addr)], Itin, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  let mayStore = 1;
}
class SW_MMR6_DESC : Store<"sw", GPR32Opnd>;
class SWE_MMR6_DESC : SWE_MMR6_DESC_BASE<"swe", GPR32Opnd, mem_simm9>;

class WRPGPR_WSBH_MMR6_DESC_BASE<string instr_asm, RegisterOperand RO>
    : MMR6Arch<instr_asm> {
  dag InOperandList = (ins RO:$rs);
  dag OutOperandList = (outs RO:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs");
  list<dag> Pattern = [];
  Format f = FrmR;
  string BaseOpcode = instr_asm;
  bit hasSideEffects = 0;
}
class WRPGPR_MMR6_DESC : WRPGPR_WSBH_MMR6_DESC_BASE<"wrpgpr", GPR32Opnd>;
class WSBH_MMR6_DESC : WRPGPR_WSBH_MMR6_DESC_BASE<"wsbh", GPR32Opnd>;

class MTC0_MMR6_DESC_BASE<string opstr, RegisterOperand DstRC,
                         RegisterOperand SrcRC> {
  dag InOperandList = (ins SrcRC:$rt, uimm3:$sel);
  dag OutOperandList = (outs DstRC:$rs);
  string AsmString = !strconcat(opstr, "\t$rt, $rs, $sel");
  list<dag> Pattern = [];
  Format f = FrmFR;
  string BaseOpcode = opstr;
}
class MTC1_MMR6_DESC_BASE<
      string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
      InstrItinClass Itin = NoItinerary, SDPatternOperator OpNode = null_frag>
      : MipsR6Inst {
  dag InOperandList = (ins SrcRC:$rt);
  dag OutOperandList = (outs DstRC:$fs);
  string AsmString = !strconcat(opstr, "\t$rt, $fs");
  list<dag> Pattern = [(set DstRC:$fs, (OpNode SrcRC:$rt))];
  Format f = FrmFR;
  InstrItinClass Itinerary = Itin;
  string BaseOpcode = opstr;
}
class MTC1_64_MMR6_DESC_BASE<
      string opstr, RegisterOperand DstRC, RegisterOperand SrcRC,
      InstrItinClass Itin = NoItinerary> : MipsR6Inst {
  dag InOperandList = (ins DstRC:$fs_in, SrcRC:$rt);
  dag OutOperandList = (outs DstRC:$fs);
  string AsmString = !strconcat(opstr, "\t$rt, $fs");
  list<dag> Pattern = [];
  Format f = FrmFR;
  InstrItinClass Itinerary = Itin;
  string BaseOpcode = opstr;
  // $fs_in is part of a white lie to work around a widespread bug in the FPU
  // implementation. See expandBuildPairF64 for details.
  let Constraints = "$fs = $fs_in";
}
class MTC2_MMR6_DESC_BASE<string opstr, RegisterOperand DstRC,
                         RegisterOperand SrcRC> {
  dag InOperandList = (ins SrcRC:$rt);
  dag OutOperandList = (outs DstRC:$impl);
  string AsmString = !strconcat(opstr, "\t$rt, $impl");
  list<dag> Pattern = [];
  Format f = FrmFR;
  string BaseOpcode = opstr;
}

class MTC0_MMR6_DESC : MTC0_MMR6_DESC_BASE<"mtc0", COP0Opnd, GPR32Opnd>;
class MTC1_MMR6_DESC : MTC1_MMR6_DESC_BASE<"mtc1", FGR32Opnd, GPR32Opnd,
                                           II_MTC1, bitconvert>, HARDFLOAT;
class MTC2_MMR6_DESC : MTC2_MMR6_DESC_BASE<"mtc2", COP2Opnd, GPR32Opnd>;
class MTHC0_MMR6_DESC : MTC0_MMR6_DESC_BASE<"mthc0", COP0Opnd, GPR32Opnd>;
class MTHC1_D32_MMR6_DESC : MTC1_64_MMR6_DESC_BASE<"mthc1", AFGR64Opnd, GPR32Opnd>,
                            HARDFLOAT, FGR_32;
class MTHC1_D64_MMR6_DESC : MTC1_64_MMR6_DESC_BASE<"mthc1", FGR64Opnd, GPR32Opnd>,
                            HARDFLOAT, FGR_64;
class MTHC2_MMR6_DESC : MTC2_MMR6_DESC_BASE<"mthc2", COP2Opnd, GPR32Opnd>;

class MFC0_MMR6_DESC_BASE<string opstr, RegisterOperand DstRC,
                          RegisterOperand SrcRC> {
  dag InOperandList = (ins SrcRC:$rs, uimm3:$sel);
  dag OutOperandList = (outs DstRC:$rt);
  string AsmString = !strconcat(opstr, "\t$rt, $rs, $sel");
  list<dag> Pattern = [];
  Format f = FrmFR;
  string BaseOpcode = opstr;
}
class MFC1_MMR6_DESC_BASE<string opstr, RegisterOperand DstRC,
                          RegisterOperand SrcRC,
                          InstrItinClass Itin = NoItinerary,
                          SDPatternOperator OpNode = null_frag> : MipsR6Inst {
  dag InOperandList = (ins SrcRC:$fs);
  dag OutOperandList = (outs DstRC:$rt);
  string AsmString = !strconcat(opstr, "\t$rt, $fs");
  list<dag> Pattern = [(set DstRC:$rt, (OpNode SrcRC:$fs))];
  Format f = FrmFR;
  InstrItinClass Itinerary = Itin;
  string BaseOpcode = opstr;
}
class MFC2_MMR6_DESC_BASE<string opstr, RegisterOperand DstRC,
                          RegisterOperand SrcRC> {
  dag InOperandList = (ins SrcRC:$impl);
  dag OutOperandList = (outs DstRC:$rt);
  string AsmString = !strconcat(opstr, "\t$rt, $impl");
  list<dag> Pattern = [];
  Format f = FrmFR;
  string BaseOpcode = opstr;
}
class MFC0_MMR6_DESC : MFC0_MMR6_DESC_BASE<"mfc0", GPR32Opnd, COP0Opnd>;
class MFC1_MMR6_DESC : MFC1_MMR6_DESC_BASE<"mfc1", GPR32Opnd, FGR32Opnd,
                                           II_MFC1, bitconvert>, HARDFLOAT;
class MFC2_MMR6_DESC : MFC2_MMR6_DESC_BASE<"mfc2", GPR32Opnd, COP2Opnd>;
class MFHC0_MMR6_DESC : MFC0_MMR6_DESC_BASE<"mfhc0", GPR32Opnd, COP0Opnd>;
class MFHC1_D32_MMR6_DESC : MFC1_MMR6_DESC_BASE<"mfhc1", GPR32Opnd, AFGR64Opnd,
                                                II_MFHC1>, HARDFLOAT, FGR_32;
class MFHC1_D64_MMR6_DESC : MFC1_MMR6_DESC_BASE<"mfhc1", GPR32Opnd, FGR64Opnd,
                                                II_MFHC1>, HARDFLOAT, FGR_64;
class MFHC2_MMR6_DESC : MFC2_MMR6_DESC_BASE<"mfhc2", GPR32Opnd, COP2Opnd>;

/// Floating Point Instructions
class FARITH_MMR6_DESC_BASE<string instr_asm, RegisterOperand RC,
                            InstrItinClass Itin, bit isComm,
                            SDPatternOperator OpNode = null_frag> : HARDFLOAT {
  dag OutOperandList = (outs RC:$fd);
  dag InOperandList = (ins RC:$ft, RC:$fs);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [(set RC:$fd, (OpNode RC:$fs, RC:$ft))];
  InstrItinClass Itinerary = Itin;
  bit isCommutable = isComm;
}
class FADD_S_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"add.s", FGR32Opnd, II_ADD_S, 1, fadd>;
class FADD_D_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"add.d", AFGR64Opnd, II_ADD_D, 1, fadd>;
class FSUB_S_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"sub.s", FGR32Opnd, II_SUB_S, 0, fsub>;
class FSUB_D_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"sub.d", AFGR64Opnd, II_SUB_D, 0, fsub>;
class FMUL_S_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"mul.s", FGR32Opnd, II_MUL_S, 1, fmul>;
class FMUL_D_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"mul.d", AFGR64Opnd, II_MUL_D, 1, fmul>;
class FDIV_S_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"div.s", FGR32Opnd, II_DIV_S, 0, fdiv>;
class FDIV_D_MMR6_DESC
  : FARITH_MMR6_DESC_BASE<"div.d", AFGR64Opnd, II_DIV_D, 0, fdiv>;
class MADDF_S_MMR6_DESC : COP1_4R_DESC_BASE<"maddf.s", FGR32Opnd>, HARDFLOAT;
class MADDF_D_MMR6_DESC : COP1_4R_DESC_BASE<"maddf.d", FGR64Opnd>, HARDFLOAT;
class MSUBF_S_MMR6_DESC : COP1_4R_DESC_BASE<"msubf.s", FGR32Opnd>, HARDFLOAT;
class MSUBF_D_MMR6_DESC : COP1_4R_DESC_BASE<"msubf.d", FGR64Opnd>, HARDFLOAT;

class FMOV_FNEG_MMR6_DESC_BASE<string instr_asm, RegisterOperand DstRC,
                               RegisterOperand SrcRC, InstrItinClass Itin,
                               SDPatternOperator OpNode = null_frag>
                               : HARDFLOAT, NeverHasSideEffects {
  dag OutOperandList = (outs DstRC:$ft);
  dag InOperandList = (ins SrcRC:$fs);
  string AsmString = !strconcat(instr_asm, "\t$ft, $fs");
  list<dag> Pattern = [(set DstRC:$ft, (OpNode SrcRC:$fs))];
  InstrItinClass Itinerary = Itin;
  Format Form = FrmFR;
}
class FMOV_S_MMR6_DESC
  : FMOV_FNEG_MMR6_DESC_BASE<"mov.s", FGR32Opnd, FGR32Opnd, II_MOV_S>;
class FMOV_D_MMR6_DESC
  : FMOV_FNEG_MMR6_DESC_BASE<"mov.d", AFGR64Opnd, AFGR64Opnd, II_MOV_D>;
class FNEG_S_MMR6_DESC
  : FMOV_FNEG_MMR6_DESC_BASE<"neg.s", FGR32Opnd, FGR32Opnd, II_NEG, fneg>;
class FNEG_D_MMR6_DESC
  : FMOV_FNEG_MMR6_DESC_BASE<"neg.d", AFGR64Opnd, AFGR64Opnd, II_NEG, fneg>;

class MAX_S_MMR6_DESC : MAX_MIN_DESC_BASE<"max.s", FGR32Opnd>, HARDFLOAT;
class MAX_D_MMR6_DESC : MAX_MIN_DESC_BASE<"max.d", FGR64Opnd>, HARDFLOAT;
class MIN_S_MMR6_DESC : MAX_MIN_DESC_BASE<"min.s", FGR32Opnd>, HARDFLOAT;
class MIN_D_MMR6_DESC : MAX_MIN_DESC_BASE<"min.d", FGR64Opnd>, HARDFLOAT;

class MAXA_S_MMR6_DESC : MAX_MIN_DESC_BASE<"maxa.s", FGR32Opnd>, HARDFLOAT;
class MAXA_D_MMR6_DESC : MAX_MIN_DESC_BASE<"maxa.d", FGR64Opnd>, HARDFLOAT;
class MINA_S_MMR6_DESC : MAX_MIN_DESC_BASE<"mina.s", FGR32Opnd>, HARDFLOAT;
class MINA_D_MMR6_DESC : MAX_MIN_DESC_BASE<"mina.d", FGR64Opnd>, HARDFLOAT;

class CVT_MMR6_DESC_BASE<
    string instr_asm, RegisterOperand DstRC, RegisterOperand SrcRC,
    InstrItinClass Itin, SDPatternOperator OpNode = null_frag>
    : HARDFLOAT, NeverHasSideEffects {
  dag OutOperandList = (outs DstRC:$ft);
  dag InOperandList = (ins SrcRC:$fs);
  string AsmString = !strconcat(instr_asm, "\t$ft, $fs");
  list<dag> Pattern = [(set DstRC:$ft, (OpNode SrcRC:$fs))];
  InstrItinClass Itinerary = Itin;
  Format Form = FrmFR;
}

class CVT_L_S_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.l.s", FGR64Opnd, FGR32Opnd,
                                             II_CVT>;
class CVT_L_D_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.l.d", FGR64Opnd, FGR64Opnd,
                                             II_CVT>;
class CVT_W_S_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.w.s", FGR32Opnd, FGR32Opnd,
                                             II_CVT>;
class CVT_W_D_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.w.d", FGR32Opnd, AFGR64Opnd,
                                             II_CVT>;
class CVT_D_S_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.d.s", FGR32Opnd, AFGR64Opnd,
                                             II_CVT>;
class CVT_D_W_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.d.w", FGR32Opnd, AFGR64Opnd,
                                             II_CVT>;
class CVT_D_L_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.d.l", FGR64Opnd, FGR64Opnd,
                                             II_CVT>, FGR_64;
class CVT_S_D_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.s.d", AFGR64Opnd, FGR32Opnd,
                                             II_CVT>;
class CVT_S_W_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.s.w", FGR32Opnd, FGR32Opnd,
                                             II_CVT>;
class CVT_S_L_MMR6_DESC : CVT_MMR6_DESC_BASE<"cvt.s.l", FGR64Opnd, FGR32Opnd,
                                             II_CVT>, FGR_64;

multiclass CMP_CC_MMR6<bits<6> format, string Typestr,
                       RegisterOperand FGROpnd> {
  def CMP_AF_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.af.", Typestr), format, FIELD_CMP_COND_AF>,
      CMP_CONDN_DESC_BASE<"af", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_UN_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.un.", Typestr), format, FIELD_CMP_COND_UN>,
      CMP_CONDN_DESC_BASE<"un", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_EQ_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.eq.", Typestr), format, FIELD_CMP_COND_EQ>,
      CMP_CONDN_DESC_BASE<"eq", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_UEQ_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.ueq.", Typestr), format, FIELD_CMP_COND_UEQ>,
      CMP_CONDN_DESC_BASE<"ueq", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_LT_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.lt.", Typestr), format, FIELD_CMP_COND_LT>,
      CMP_CONDN_DESC_BASE<"lt", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_ULT_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.ult.", Typestr), format, FIELD_CMP_COND_ULT>,
      CMP_CONDN_DESC_BASE<"ult", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_LE_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.le.", Typestr), format, FIELD_CMP_COND_LE>,
      CMP_CONDN_DESC_BASE<"le", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_ULE_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.ule.", Typestr), format, FIELD_CMP_COND_ULE>,
      CMP_CONDN_DESC_BASE<"ule", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SAF_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.saf.", Typestr), format, FIELD_CMP_COND_SAF>,
      CMP_CONDN_DESC_BASE<"saf", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SUN_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.sun.", Typestr), format, FIELD_CMP_COND_SUN>,
      CMP_CONDN_DESC_BASE<"sun", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SEQ_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.seq.", Typestr), format, FIELD_CMP_COND_SEQ>,
      CMP_CONDN_DESC_BASE<"seq", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SUEQ_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.sueq.", Typestr), format, FIELD_CMP_COND_SUEQ>,
      CMP_CONDN_DESC_BASE<"sueq", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SLT_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.slt.", Typestr), format, FIELD_CMP_COND_SLT>,
      CMP_CONDN_DESC_BASE<"slt", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SULT_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.sult.", Typestr), format, FIELD_CMP_COND_SULT>,
      CMP_CONDN_DESC_BASE<"sult", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SLE_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.sle.", Typestr), format, FIELD_CMP_COND_SLE>,
      CMP_CONDN_DESC_BASE<"sle", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
  def CMP_SULE_#NAME : POOL32F_CMP_FM<
      !strconcat("cmp.sule.", Typestr), format, FIELD_CMP_COND_SULE>,
      CMP_CONDN_DESC_BASE<"sule", Typestr, FGROpnd>, HARDFLOAT, R6MMR6Rel,
      ISA_MICROMIPS32R6;
}

class ABSS_FT_MMR6_DESC_BASE<string instr_asm, RegisterOperand DstRC,
                             RegisterOperand SrcRC, InstrItinClass Itin,
                             SDPatternOperator OpNode = null_frag>
    : HARDFLOAT, NeverHasSideEffects {
  dag OutOperandList = (outs DstRC:$ft);
  dag InOperandList  = (ins SrcRC:$fs);
  string AsmString   = !strconcat(instr_asm, "\t$ft, $fs");
  list<dag>  Pattern = [(set DstRC:$ft, (OpNode SrcRC:$fs))];
  InstrItinClass Itinerary = Itin;
  Format Form = FrmFR;
  list<Predicate> EncodingPredicates = [HasStdEnc];
}

class ABS_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"abs.s", FGR32Opnd, FGR32Opnd,
                                                II_ABS, fabs>;
class ABS_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"abs.d", AFGR64Opnd, AFGR64Opnd,
                                                II_ABS, fabs>;
class FLOOR_L_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"floor.l.s", FGR64Opnd,
                                                    FGR32Opnd, II_FLOOR>;
class FLOOR_L_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"floor.l.d", FGR64Opnd,
                                                    FGR64Opnd, II_FLOOR>;
class FLOOR_W_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"floor.w.s", FGR32Opnd,
                                                    FGR32Opnd, II_FLOOR>;
class FLOOR_W_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"floor.w.d", FGR32Opnd,
                                                    AFGR64Opnd, II_FLOOR>;
class CEIL_L_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"ceil.l.s", FGR64Opnd,
                                                   FGR32Opnd, II_CEIL>;
class CEIL_L_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"ceil.l.d", FGR64Opnd,
                                                   FGR64Opnd, II_CEIL>;
class CEIL_W_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"ceil.w.s", FGR32Opnd,
                                                   FGR32Opnd, II_CEIL>;
class CEIL_W_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"ceil.w.d", FGR32Opnd,
                                                   AFGR64Opnd, II_CEIL>;
class TRUNC_L_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"trunc.l.s", FGR64Opnd,
                                                    FGR32Opnd, II_TRUNC>;
class TRUNC_L_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"trunc.l.d", FGR64Opnd,
                                                    FGR64Opnd, II_TRUNC>;
class TRUNC_W_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"trunc.w.s", FGR32Opnd,
                                                    FGR32Opnd, II_TRUNC>;
class TRUNC_W_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"trunc.w.d", FGR32Opnd,
                                                    AFGR64Opnd, II_TRUNC>;
class SQRT_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"sqrt.s", FGR32Opnd, FGR32Opnd,
                                                 II_SQRT_S, fsqrt>;
class SQRT_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"sqrt.d", AFGR64Opnd, AFGR64Opnd,
                                                 II_SQRT_D, fsqrt>;
class RSQRT_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"rsqrt.s", FGR32Opnd,
                                                  FGR32Opnd, II_TRUNC>;
class RSQRT_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"rsqrt.d", FGR32Opnd,
                                                  AFGR64Opnd, II_TRUNC>;
class RECIP_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"recip.s", FGR32Opnd,
                                                 FGR32Opnd, II_ROUND>;
class RECIP_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"recip.d", FGR32Opnd, FGR32Opnd,
                                                 II_ROUND>;
class ROUND_L_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"round.l.s", FGR64Opnd,
                                                   FGR32Opnd, II_ROUND>;
class ROUND_L_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"round.l.d", FGR64Opnd,
                                                   FGR64Opnd, II_ROUND>;
class ROUND_W_S_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"round.w.s", FGR32Opnd,
                                                   FGR32Opnd, II_ROUND>;
class ROUND_W_D_MMR6_DESC : ABSS_FT_MMR6_DESC_BASE<"round.w.d", FGR64Opnd,
                                                   FGR64Opnd, II_ROUND>;

class SEL_S_MMR6_DESC : COP1_SEL_DESC_BASE<"sel.s", FGR32Opnd>;
class SEL_D_MMR6_DESC : COP1_SEL_DESC_BASE<"sel.d", FGR64Opnd> {
  // We must insert a SUBREG_TO_REG around $fd_in
  bit usesCustomInserter = 1;
}

class SELEQZ_S_MMR6_DESC : SELEQNEZ_DESC_BASE<"seleqz.s", FGR32Opnd>;
class SELEQZ_D_MMR6_DESC : SELEQNEZ_DESC_BASE<"seleqz.d", FGR64Opnd>;
class SELENZ_S_MMR6_DESC : SELEQNEZ_DESC_BASE<"selnez.s", FGR32Opnd>;
class SELENZ_D_MMR6_DESC : SELEQNEZ_DESC_BASE<"selnez.d", FGR64Opnd>;
class RINT_S_MMR6_DESC : CLASS_RINT_DESC_BASE<"rint.s", FGR32Opnd>;
class RINT_D_MMR6_DESC : CLASS_RINT_DESC_BASE<"rint.d", FGR64Opnd>;
class CLASS_S_MMR6_DESC  : CLASS_RINT_DESC_BASE<"class.s", FGR32Opnd>;
class CLASS_D_MMR6_DESC  : CLASS_RINT_DESC_BASE<"class.d", FGR64Opnd>;

class STORE_MMR6_DESC_BASE<string opstr, DAGOperand RO>
    : Store<opstr, RO>, MMR6Arch<opstr> {
  let DecoderMethod = "DecodeMemMMImm16";
}
class SB_MMR6_DESC : STORE_MMR6_DESC_BASE<"sb", GPR32Opnd>;

class STORE_EVA_MMR6_DESC_BASE<string instr_asm, RegisterOperand RO>
    : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RO:$rt, mem_mm_9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  string DecoderMethod = "DecodeStoreEvaOpMM";
  bit mayStore = 1;
}
class SBE_MMR6_DESC : STORE_EVA_MMR6_DESC_BASE<"sbe", GPR32Opnd>;
class SCE_MMR6_DESC : STORE_EVA_MMR6_DESC_BASE<"sce", GPR32Opnd>;
class SH_MMR6_DESC : STORE_MMR6_DESC_BASE<"sh", GPR32Opnd>;
class SHE_MMR6_DESC : STORE_EVA_MMR6_DESC_BASE<"she", GPR32Opnd>;
class LOAD_WORD_EVA_MMR6_DESC_BASE<string instr_asm, RegisterOperand RO> :
            MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs RO:$rt);
  dag InOperandList = (ins mem_mm_12:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  string DecoderMethod = "DecodeMemMMImm9";
  bit mayLoad = 1;
}
class LLE_MMR6_DESC : LOAD_WORD_EVA_MMR6_DESC_BASE<"lle", GPR32Opnd>;
class LWE_MMR6_DESC : LOAD_WORD_EVA_MMR6_DESC_BASE<"lwe", GPR32Opnd>;
class ADDU16_MMR6_DESC : ArithRMM16<"addu16", GPRMM16Opnd, 1, II_ADDU, add>,
      MMR6Arch<"addu16">;
class AND16_MMR6_DESC : LogicRMM16<"and16", GPRMM16Opnd, II_AND, and>,
      MMR6Arch<"and16">;
class ANDI16_MMR6_DESC : AndImmMM16<"andi16", GPRMM16Opnd, II_AND>,
      MMR6Arch<"andi16">;
class NOT16_MMR6_DESC : NotMM16<"not16", GPRMM16Opnd>, MMR6Arch<"not16">;
class OR16_MMR6_DESC : LogicRMM16<"or16", GPRMM16Opnd, II_OR, or>,
      MMR6Arch<"or16">;
class SLL16_MMR6_DESC : ShiftIMM16<"sll16", uimm3_shift, GPRMM16Opnd, II_SLL>,
      MMR6Arch<"sll16">;
class SRL16_MMR6_DESC : ShiftIMM16<"srl16", uimm3_shift, GPRMM16Opnd, II_SRL>,
      MMR6Arch<"srl16">;
class BREAK16_MMR6_DESC : BrkSdbbp16MM<"break16">, MMR6Arch<"srl16">,
      MicroMipsR6Inst16;
class LI16_MMR6_DESC : LoadImmMM16<"li16", li16_imm, GPRMM16Opnd>,
      MMR6Arch<"srl16">, MicroMipsR6Inst16, IsAsCheapAsAMove;
class MOVE16_MMR6_DESC : MoveMM16<"move16", GPR32Opnd>, MMR6Arch<"srl16">,
      MicroMipsR6Inst16;
class SDBBP16_MMR6_DESC : BrkSdbbp16MM<"sdbbp16">, MMR6Arch<"sdbbp16">,
      MicroMipsR6Inst16;
class SUBU16_MMR6_DESC : ArithRMM16<"subu16", GPRMM16Opnd, 0, II_SUBU, sub>,
      MMR6Arch<"sdbbp16">, MicroMipsR6Inst16;
class XOR16_MMR6_DESC : LogicRMM16<"xor16", GPRMM16Opnd, II_XOR, xor>,
      MMR6Arch<"sdbbp16">;

class LW_MMR6_DESC : MMR6Arch<"lw">, MipsR6Inst {
  dag OutOperandList = (outs GPR32Opnd:$rt);
  dag InOperandList = (ins mem:$addr);
  string AsmString = "lw\t$rt, $addr";
  let DecoderMethod = "DecodeMemMMImm16";
  let canFoldAsLoad = 1;
  let mayLoad = 1;
  list<dag> Pattern = [(set GPR32Opnd:$rt, (load addrDefault:$addr))];
  InstrItinClass Itinerary = II_LW;
}

class LUI_MMR6_DESC : IsAsCheapAsAMove, MMR6Arch<"lui">, MipsR6Inst{
  dag OutOperandList = (outs GPR32Opnd:$rt);
  dag InOperandList = (ins uimm16:$imm16);
  string AsmString = "lui\t$rt, $imm16";
  list<dag> Pattern = [];
  bit hasSideEffects = 0;
  bit isReMaterializable = 1;
  InstrItinClass Itinerary = II_LUI;
  Format Form = FrmI;
}

class SYNC_MMR6_DESC : MMR6Arch<"sync">, MipsR6Inst {
  dag OutOperandList = (outs);
  dag InOperandList = (ins i32imm:$stype);
  string AsmString = !strconcat("sync", "\t$stype");
  list<dag> Pattern = [(MipsSync imm:$stype)];
  InstrItinClass Itinerary = NoItinerary;
  bit HasSideEffects = 1;
}

class SYNCI_MMR6_DESC : SYNCI_FT<"synci"> {
  let DecoderMethod = "DecodeSynciR6";
}

class RDPGPR_MMR6_DESC : MMR6Arch<"rdpgpr">, MipsR6Inst {
  dag OutOperandList = (outs GPR32Opnd:$rt);
  dag InOperandList = (ins GPR32Opnd:$rd);
  string AsmString = !strconcat("rdpgpr", "\t$rt, $rd");
}

class SDBBP_MMR6_DESC : MipsR6Inst {
  dag OutOperandList = (outs);
  dag InOperandList = (ins uimm20:$code_);
  string AsmString = !strconcat("sdbbp", "\t$code_");
  list<dag> Pattern = [];
}

class LWM16_MMR6_DESC
    : MicroMipsInst16<(outs reglist16:$rt), (ins mem_mm_4sp:$addr),
                      !strconcat("lwm16", "\t$rt, $addr"), [],
                      NoItinerary, FrmI>,
      MMR6Arch<"lwm16">, MicroMipsR6Inst16 {
  let DecoderMethod = "DecodeMemMMReglistImm4Lsl2";
  let mayLoad = 1;
  InstrItinClass Itin = NoItinerary;
  ComplexPattern Addr = addr;
}

class SWM16_MMR6_DESC
    : MicroMipsInst16<(outs), (ins reglist16:$rt, mem_mm_4sp:$addr),
                      !strconcat("swm16", "\t$rt, $addr"), [],
                      NoItinerary, FrmI>,
      MMR6Arch<"swm16">, MicroMipsR6Inst16 {
  let DecoderMethod = "DecodeMemMMReglistImm4Lsl2";
  let mayStore = 1;
  InstrItinClass Itin = NoItinerary;
  ComplexPattern Addr = addr;
}

class SB16_MMR6_DESC_BASE<string opstr, DAGOperand RTOpnd, DAGOperand RO,
                          SDPatternOperator OpNode, InstrItinClass Itin,
                          Operand MemOpnd>
    : MicroMipsInst16<(outs), (ins RTOpnd:$rt, MemOpnd:$addr),
                      !strconcat(opstr, "\t$rt, $addr"), [], Itin, FrmI>,
      MMR6Arch<opstr>, MicroMipsR6Inst16 {
  let DecoderMethod = "DecodeMemMMImm4";
  let mayStore = 1;
}
class SB16_MMR6_DESC : SB16_MMR6_DESC_BASE<"sb16", GPRMM16OpndZero, GPRMM16Opnd,
                                           truncstorei8, II_SB, mem_mm_4>;
class SH16_MMR6_DESC : SB16_MMR6_DESC_BASE<"sh16", GPRMM16OpndZero, GPRMM16Opnd,
                                           truncstorei16, II_SH, mem_mm_4_lsl1>;
class SW16_MMR6_DESC : SB16_MMR6_DESC_BASE<"sw16", GPRMM16OpndZero, GPRMM16Opnd,
                                           store, II_SW, mem_mm_4_lsl2>;

class SWSP_MMR6_DESC
    : MicroMipsInst16<(outs), (ins GPR32Opnd:$rt, mem_mm_sp_imm5_lsl2:$offset),
                      !strconcat("sw", "\t$rt, $offset"), [], II_SW, FrmI>,
      MMR6Arch<"sw">, MicroMipsR6Inst16 {
  let DecoderMethod = "DecodeMemMMSPImm5Lsl2";
  let mayStore = 1;
}

class TLBINV_MMR6_DESC_BASE<string opstr> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  string AsmString = opstr;
  list<dag> Pattern = [];
}

class TLBINV_MMR6_DESC : TLBINV_MMR6_DESC_BASE<"tlbinv">;
class TLBINVF_MMR6_DESC : TLBINV_MMR6_DESC_BASE<"tlbinvf">;

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

let DecoderNamespace = "MicroMipsR6" in {
def ADD_MMR6 : StdMMR6Rel, ADD_MMR6_DESC, ADD_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDIU_MMR6 : StdMMR6Rel, ADDIU_MMR6_DESC, ADDIU_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDU_MMR6 : StdMMR6Rel, ADDU_MMR6_DESC, ADDU_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDIUPC_MMR6 : R6MMR6Rel, ADDIUPC_MMR6_ENC, ADDIUPC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def ALUIPC_MMR6 : R6MMR6Rel, ALUIPC_MMR6_ENC, ALUIPC_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def AND_MMR6 : StdMMR6Rel, AND_MMR6_DESC, AND_MMR6_ENC, ISA_MICROMIPS32R6;
def ANDI_MMR6 : StdMMR6Rel, ANDI_MMR6_DESC, ANDI_MMR6_ENC, ISA_MICROMIPS32R6;
def AUIPC_MMR6 : R6MMR6Rel, AUIPC_MMR6_ENC, AUIPC_MMR6_DESC, ISA_MICROMIPS32R6;
def ALIGN_MMR6 : R6MMR6Rel, ALIGN_MMR6_ENC, ALIGN_MMR6_DESC, ISA_MICROMIPS32R6;
def AUI_MMR6 : R6MMR6Rel, AUI_MMR6_ENC, AUI_MMR6_DESC, ISA_MICROMIPS32R6;
def BALC_MMR6 : R6MMR6Rel, BALC_MMR6_ENC, BALC_MMR6_DESC, ISA_MICROMIPS32R6;
def BC_MMR6 : R6MMR6Rel, BC_MMR6_ENC, BC_MMR6_DESC, ISA_MICROMIPS32R6;
def BC16_MMR6 : StdMMR6Rel, BC16_MMR6_DESC, BC16_MMR6_ENC, ISA_MICROMIPS32R6;
def BEQZC16_MMR6 : StdMMR6Rel, BEQZC16_MMR6_DESC, BEQZC16_MMR6_ENC,
                   ISA_MICROMIPS32R6;
def BNEZC16_MMR6 : StdMMR6Rel, BNEZC16_MMR6_DESC, BNEZC16_MMR6_ENC,
                   ISA_MICROMIPS32R6;
def BITSWAP_MMR6 : R6MMR6Rel, BITSWAP_MMR6_ENC, BITSWAP_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BEQZALC_MMR6 : R6MMR6Rel, BEQZALC_MMR6_ENC, BEQZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BGEZALC_MMR6 : R6MMR6Rel, BGEZALC_MMR6_ENC, BGEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BGTZALC_MMR6 : R6MMR6Rel, BGTZALC_MMR6_ENC, BGTZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BLEZALC_MMR6 : R6MMR6Rel, BLEZALC_MMR6_ENC, BLEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BLTZALC_MMR6 : R6MMR6Rel, BLTZALC_MMR6_ENC, BLTZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BNEZALC_MMR6 : R6MMR6Rel, BNEZALC_MMR6_ENC, BNEZALC_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def BREAK_MMR6 : StdMMR6Rel, BRK_MMR6_DESC, BRK_MMR6_ENC, ISA_MICROMIPS32R6;
def CACHE_MMR6 : R6MMR6Rel, CACHE_MMR6_ENC, CACHE_MMR6_DESC, ISA_MICROMIPS32R6;
def CLO_MMR6 : R6MMR6Rel, CLO_MMR6_ENC, CLO_MMR6_DESC, ISA_MICROMIPS32R6;
def CLZ_MMR6 : R6MMR6Rel, CLZ_MMR6_ENC, CLZ_MMR6_DESC, ISA_MICROMIPS32R6;
def DIV_MMR6 : R6MMR6Rel, DIV_MMR6_DESC, DIV_MMR6_ENC, ISA_MICROMIPS32R6;
def DIVU_MMR6 : R6MMR6Rel, DIVU_MMR6_DESC, DIVU_MMR6_ENC, ISA_MICROMIPS32R6;
def EHB_MMR6 : StdMMR6Rel, EHB_MMR6_DESC, EHB_MMR6_ENC, ISA_MICROMIPS32R6;
def EI_MMR6 : StdMMR6Rel, EI_MMR6_DESC, EI_MMR6_ENC, ISA_MICROMIPS32R6;
def DI_MMR6 : StdMMR6Rel, DI_MMR6_DESC, DI_MMR6_ENC, ISA_MICROMIPS32R6;
def ERET_MMR6 : StdMMR6Rel, ERET_MMR6_DESC, ERET_MMR6_ENC, ISA_MICROMIPS32R6;
def DERET_MMR6 : StdMMR6Rel, DERET_MMR6_DESC, DERET_MMR6_ENC, ISA_MICROMIPS32R6;
def ERETNC_MMR6 : R6MMR6Rel, ERETNC_MMR6_DESC, ERETNC_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def JALRC16_MMR6 : R6MMR6Rel, JALRC16_MMR6_DESC, JALRC16_MMR6_ENC,
                   ISA_MICROMIPS32R6;
def JIALC_MMR6 : R6MMR6Rel, JIALC_MMR6_ENC, JIALC_MMR6_DESC, ISA_MICROMIPS32R6;
def JIC_MMR6 : R6MMR6Rel, JIC_MMR6_ENC, JIC_MMR6_DESC, ISA_MICROMIPS32R6;
def JRC16_MMR6 : R6MMR6Rel, JRC16_MMR6_DESC, JRC16_MMR6_ENC, ISA_MICROMIPS32R6;
def JRCADDIUSP_MMR6 : R6MMR6Rel, JRCADDIUSP_MMR6_DESC, JRCADDIUSP_MMR6_ENC,
                      ISA_MICROMIPS32R6;
def LSA_MMR6 : R6MMR6Rel, LSA_MMR6_ENC, LSA_MMR6_DESC, ISA_MICROMIPS32R6;
def LWPC_MMR6 : R6MMR6Rel, LWPC_MMR6_ENC, LWPC_MMR6_DESC, ISA_MICROMIPS32R6;
def LWM16_MMR6 : StdMMR6Rel, LWM16_MMR6_DESC, LWM16_MMR6_ENC, ISA_MICROMIPS32R6;
def MTC0_MMR6 : StdMMR6Rel, MTC0_MMR6_ENC, MTC0_MMR6_DESC, ISA_MICROMIPS32R6;
def MTC1_MMR6 : StdMMR6Rel, MTC1_MMR6_DESC, MTC1_MMR6_ENC, ISA_MICROMIPS32R6;
def MTC2_MMR6 : StdMMR6Rel, MTC2_MMR6_ENC, MTC2_MMR6_DESC, ISA_MICROMIPS32R6;
def MTHC0_MMR6 : R6MMR6Rel, MTHC0_MMR6_ENC, MTHC0_MMR6_DESC, ISA_MICROMIPS32R6;
def MTHC1_D32_MMR6 : StdMMR6Rel, MTHC1_D32_MMR6_DESC, MTHC1_MMR6_ENC, ISA_MICROMIPS32R6;
let DecoderNamespace = "MicroMips32r6FP64" in {
  def MTHC1_D64_MMR6 : R6MMR6Rel, MTHC1_D64_MMR6_DESC, MTHC1_MMR6_ENC,
                       ISA_MICROMIPS32R6;
}
def MTHC2_MMR6 : StdMMR6Rel, MTHC2_MMR6_ENC, MTHC2_MMR6_DESC, ISA_MICROMIPS32R6;
def MFC0_MMR6 : StdMMR6Rel, MFC0_MMR6_ENC, MFC0_MMR6_DESC, ISA_MICROMIPS32R6;
def MFC1_MMR6 : StdMMR6Rel, MFC1_MMR6_DESC, MFC1_MMR6_ENC, ISA_MICROMIPS32R6;
def MFC2_MMR6 : StdMMR6Rel, MFC2_MMR6_ENC, MFC2_MMR6_DESC, ISA_MICROMIPS32R6;
def MFHC0_MMR6 : R6MMR6Rel, MFHC0_MMR6_ENC, MFHC0_MMR6_DESC, ISA_MICROMIPS32R6;
def MFHC1_D32_MMR6 : StdMMR6Rel, MFHC1_D32_MMR6_DESC, MFHC1_MMR6_ENC,
                     ISA_MICROMIPS32R6;
let DecoderNamespace = "MicroMips32r6FP64" in {
  def MFHC1_D64_MMR6 : StdMMR6Rel, MFHC1_D64_MMR6_DESC, MFHC1_MMR6_ENC,
                       ISA_MICROMIPS32R6;
}
def MFHC2_MMR6 : StdMMR6Rel, MFHC2_MMR6_ENC, MFHC2_MMR6_DESC, ISA_MICROMIPS32R6;
def MOD_MMR6 : R6MMR6Rel, MOD_MMR6_DESC, MOD_MMR6_ENC, ISA_MICROMIPS32R6;
def MODU_MMR6 : R6MMR6Rel, MODU_MMR6_DESC, MODU_MMR6_ENC, ISA_MICROMIPS32R6;
def MUL_MMR6 : R6MMR6Rel, MUL_MMR6_DESC, MUL_MMR6_ENC, ISA_MICROMIPS32R6;
def MUH_MMR6 : R6MMR6Rel, MUH_MMR6_DESC, MUH_MMR6_ENC, ISA_MICROMIPS32R6;
def MULU_MMR6 : R6MMR6Rel, MULU_MMR6_DESC, MULU_MMR6_ENC, ISA_MICROMIPS32R6;
def MUHU_MMR6 : R6MMR6Rel, MUHU_MMR6_DESC, MUHU_MMR6_ENC, ISA_MICROMIPS32R6;
def NOR_MMR6 : StdMMR6Rel, NOR_MMR6_DESC, NOR_MMR6_ENC, ISA_MICROMIPS32R6;
def OR_MMR6 : StdMMR6Rel, OR_MMR6_DESC, OR_MMR6_ENC, ISA_MICROMIPS32R6;
def ORI_MMR6 : StdMMR6Rel, ORI_MMR6_DESC, ORI_MMR6_ENC, ISA_MICROMIPS32R6;
def PREF_MMR6 : R6MMR6Rel, PREF_MMR6_ENC, PREF_MMR6_DESC, ISA_MICROMIPS32R6;
def SB16_MMR6 : StdMMR6Rel, SB16_MMR6_DESC, SB16_MMR6_ENC, ISA_MICROMIPS32R6;
def SEB_MMR6 : StdMMR6Rel, SEB_MMR6_DESC, SEB_MMR6_ENC, ISA_MICROMIPS32R6;
def SEH_MMR6 : StdMMR6Rel, SEH_MMR6_DESC, SEH_MMR6_ENC, ISA_MICROMIPS32R6;
def SELEQZ_MMR6 : R6MMR6Rel, SELEQZ_MMR6_ENC, SELEQZ_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def SELNEZ_MMR6 : R6MMR6Rel, SELNEZ_MMR6_ENC, SELNEZ_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def SH16_MMR6 : StdMMR6Rel, SH16_MMR6_DESC, SH16_MMR6_ENC, ISA_MICROMIPS32R6;
def SLL_MMR6 : StdMMR6Rel, SLL_MMR6_DESC, SLL_MMR6_ENC, ISA_MICROMIPS32R6;
def SUB_MMR6 : StdMMR6Rel, SUB_MMR6_DESC, SUB_MMR6_ENC, ISA_MICROMIPS32R6;
def SUBU_MMR6 : StdMMR6Rel, SUBU_MMR6_DESC, SUBU_MMR6_ENC, ISA_MICROMIPS32R6;
def SW16_MMR6 : StdMMR6Rel, SW16_MMR6_DESC, SW16_MMR6_ENC, ISA_MICROMIPS32R6;
def SWM16_MMR6 : StdMMR6Rel, SWM16_MMR6_DESC, SWM16_MMR6_ENC, ISA_MICROMIPS32R6;
def SWSP_MMR6 : StdMMR6Rel, SWSP_MMR6_DESC, SWSP_MMR6_ENC, ISA_MICROMIPS32R6;
def PREFE_MMR6 : StdMMR6Rel, PREFE_MMR6_ENC, PREFE_MMR6_DESC, ISA_MICROMIPS32R6;
def CACHEE_MMR6 : StdMMR6Rel, CACHEE_MMR6_ENC, CACHEE_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def WRPGPR_MMR6 : StdMMR6Rel, WRPGPR_MMR6_ENC, WRPGPR_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def WSBH_MMR6 : StdMMR6Rel, WSBH_MMR6_ENC, WSBH_MMR6_DESC, ISA_MICROMIPS32R6;
def LB_MMR6 : R6MMR6Rel, LB_MMR6_ENC, LB_MMR6_DESC, ISA_MICROMIPS32R6;
def LBU_MMR6 : R6MMR6Rel, LBU_MMR6_ENC, LBU_MMR6_DESC, ISA_MICROMIPS32R6;
def LBE_MMR6 : R6MMR6Rel, LBE_MMR6_ENC, LBE_MMR6_DESC, ISA_MICROMIPS32R6;
def LBUE_MMR6 : R6MMR6Rel, LBUE_MMR6_ENC, LBUE_MMR6_DESC, ISA_MICROMIPS32R6;
def PAUSE_MMR6 : StdMMR6Rel, PAUSE_MMR6_DESC, PAUSE_MMR6_ENC, ISA_MICROMIPS32R6;
def RDHWR_MMR6 : R6MMR6Rel, RDHWR_MMR6_DESC, RDHWR_MMR6_ENC, ISA_MICROMIPS32R6;
def WAIT_MMR6 : StdMMR6Rel, WAIT_MMR6_DESC, WAIT_MMR6_ENC, ISA_MICROMIPS32R6;
def SSNOP_MMR6 : StdMMR6Rel, SSNOP_MMR6_DESC, SSNOP_MMR6_ENC, ISA_MICROMIPS32R6;
def SYNC_MMR6 : StdMMR6Rel, SYNC_MMR6_DESC, SYNC_MMR6_ENC, ISA_MICROMIPS32R6;
def SYNCI_MMR6 : StdMMR6Rel, SYNCI_MMR6_DESC, SYNCI_MMR6_ENC, ISA_MICROMIPS32R6;
def RDPGPR_MMR6 : R6MMR6Rel, RDPGPR_MMR6_DESC, RDPGPR_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def SDBBP_MMR6 : R6MMR6Rel, SDBBP_MMR6_DESC, SDBBP_MMR6_ENC, ISA_MICROMIPS32R6;
def XOR_MMR6 : StdMMR6Rel, XOR_MMR6_DESC, XOR_MMR6_ENC, ISA_MICROMIPS32R6;
def XORI_MMR6 : StdMMR6Rel, XORI_MMR6_DESC, XORI_MMR6_ENC, ISA_MICROMIPS32R6;
let DecoderMethod = "DecodeMemMMImm16" in {
  def SW_MMR6 : StdMMR6Rel, SW_MMR6_DESC, SW_MMR6_ENC, ISA_MICROMIPS32R6;
}
let DecoderMethod = "DecodeMemMMImm9" in {
  def SWE_MMR6 : StdMMR6Rel, SWE_MMR6_DESC, SWE_MMR6_ENC, ISA_MICROMIPS32R6;
}
/// Floating Point Instructions
def FADD_S_MMR6 : StdMMR6Rel, FADD_S_MMR6_ENC, FADD_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FADD_D_MMR6 : StdMMR6Rel, FADD_D_MMR6_ENC, FADD_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FSUB_S_MMR6 : StdMMR6Rel, FSUB_S_MMR6_ENC, FSUB_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FSUB_D_MMR6 : StdMMR6Rel, FSUB_D_MMR6_ENC, FSUB_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FMUL_S_MMR6 : StdMMR6Rel, FMUL_S_MMR6_ENC, FMUL_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FMUL_D_MMR6 : StdMMR6Rel, FMUL_D_MMR6_ENC, FMUL_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FDIV_S_MMR6 : StdMMR6Rel, FDIV_S_MMR6_ENC, FDIV_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FDIV_D_MMR6 : StdMMR6Rel, FDIV_D_MMR6_ENC, FDIV_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def MADDF_S_MMR6 : R6MMR6Rel, MADDF_S_MMR6_ENC, MADDF_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def MADDF_D_MMR6 : R6MMR6Rel, MADDF_D_MMR6_ENC, MADDF_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def MSUBF_S_MMR6 : R6MMR6Rel, MSUBF_S_MMR6_ENC, MSUBF_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def MSUBF_D_MMR6 : R6MMR6Rel, MSUBF_D_MMR6_ENC, MSUBF_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def FMOV_S_MMR6 : StdMMR6Rel, FMOV_S_MMR6_ENC, FMOV_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FMOV_D_MMR6 : StdMMR6Rel, FMOV_D_MMR6_ENC, FMOV_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FNEG_S_MMR6 : StdMMR6Rel, FNEG_S_MMR6_ENC, FNEG_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def FNEG_D_MMR6 : StdMMR6Rel, FNEG_D_MMR6_ENC, FNEG_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def MAX_S_MMR6 : R6MMR6Rel, MAX_S_MMR6_ENC, MAX_S_MMR6_DESC, ISA_MICROMIPS32R6;
def MAX_D_MMR6 : R6MMR6Rel, MAX_D_MMR6_ENC, MAX_D_MMR6_DESC, ISA_MICROMIPS32R6;
def MIN_S_MMR6 : R6MMR6Rel, MIN_S_MMR6_ENC, MIN_S_MMR6_DESC, ISA_MICROMIPS32R6;
def MIN_D_MMR6 : R6MMR6Rel, MIN_D_MMR6_ENC, MIN_D_MMR6_DESC, ISA_MICROMIPS32R6;
def MAXA_S_MMR6 : R6MMR6Rel, MAXA_S_MMR6_ENC, MAXA_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def MAXA_D_MMR6 : R6MMR6Rel, MAXA_D_MMR6_ENC, MAXA_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def MINA_S_MMR6 : R6MMR6Rel, MINA_S_MMR6_ENC, MINA_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def MINA_D_MMR6 : R6MMR6Rel, MINA_D_MMR6_ENC, MINA_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def CVT_L_S_MMR6 : StdMMR6Rel, CVT_L_S_MMR6_ENC, CVT_L_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_L_D_MMR6 : StdMMR6Rel, CVT_L_D_MMR6_ENC, CVT_L_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_W_S_MMR6 : StdMMR6Rel, CVT_W_S_MMR6_ENC, CVT_W_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_W_D_MMR6 : StdMMR6Rel, CVT_W_D_MMR6_ENC, CVT_W_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_D_S_MMR6 : StdMMR6Rel, CVT_D_S_MMR6_ENC, CVT_D_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_D_W_MMR6 : StdMMR6Rel, CVT_D_W_MMR6_ENC, CVT_D_W_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_D_L_MMR6 : StdMMR6Rel, CVT_D_L_MMR6_ENC, CVT_D_L_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_S_D_MMR6 : StdMMR6Rel, CVT_S_D_MMR6_ENC, CVT_S_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_S_W_MMR6 : StdMMR6Rel, CVT_S_W_MMR6_ENC, CVT_S_W_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CVT_S_L_MMR6 : StdMMR6Rel, CVT_S_L_MMR6_ENC, CVT_S_L_MMR6_DESC,
                   ISA_MICROMIPS32R6;
defm S_MMR6 : CMP_CC_MMR6<0b000101, "s", FGR32Opnd>;
defm D_MMR6 : CMP_CC_MMR6<0b010101, "d", FGR64Opnd>;
def ABS_S_MMR6 : StdMMR6Rel, ABS_S_MMR6_ENC, ABS_S_MMR6_DESC, ISA_MICROMIPS32R6;
def ABS_D_MMR6 : StdMMR6Rel, ABS_D_MMR6_ENC, ABS_D_MMR6_DESC, ISA_MICROMIPS32R6;
def FLOOR_L_S_MMR6 : StdMMR6Rel, FLOOR_L_S_MMR6_ENC, FLOOR_L_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def FLOOR_L_D_MMR6 : StdMMR6Rel, FLOOR_L_D_MMR6_ENC, FLOOR_L_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def FLOOR_W_S_MMR6 : StdMMR6Rel, FLOOR_W_S_MMR6_ENC, FLOOR_W_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def FLOOR_W_D_MMR6 : StdMMR6Rel, FLOOR_W_D_MMR6_ENC, FLOOR_W_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def CEIL_L_S_MMR6 : StdMMR6Rel, CEIL_L_S_MMR6_ENC, CEIL_L_S_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def CEIL_L_D_MMR6 : StdMMR6Rel, CEIL_L_D_MMR6_ENC, CEIL_L_D_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def CEIL_W_S_MMR6 : StdMMR6Rel, CEIL_W_S_MMR6_ENC, CEIL_W_S_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def CEIL_W_D_MMR6 : StdMMR6Rel, CEIL_W_D_MMR6_ENC, CEIL_W_D_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def TRUNC_L_S_MMR6 : StdMMR6Rel, TRUNC_L_S_MMR6_ENC, TRUNC_L_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def TRUNC_L_D_MMR6 : StdMMR6Rel, TRUNC_L_D_MMR6_ENC, TRUNC_L_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def TRUNC_W_S_MMR6 : StdMMR6Rel, TRUNC_W_S_MMR6_ENC, TRUNC_W_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def TRUNC_W_D_MMR6 : StdMMR6Rel, TRUNC_W_D_MMR6_ENC, TRUNC_W_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def SQRT_S_MMR6 : StdMMR6Rel, SQRT_S_MMR6_ENC, SQRT_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def SQRT_D_MMR6 : StdMMR6Rel, SQRT_D_MMR6_ENC, SQRT_D_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def RSQRT_S_MMR6 : StdMMR6Rel, RSQRT_S_MMR6_ENC, RSQRT_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def RSQRT_D_MMR6 : StdMMR6Rel, RSQRT_D_MMR6_ENC, RSQRT_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def SB_MMR6 : StdMMR6Rel, SB_MMR6_DESC, SB_MMR6_ENC, ISA_MICROMIPS32R6;
def SBE_MMR6 : StdMMR6Rel, SBE_MMR6_DESC, SBE_MMR6_ENC, ISA_MICROMIPS32R6;
def SCE_MMR6 : StdMMR6Rel, SCE_MMR6_DESC, SCE_MMR6_ENC, ISA_MICROMIPS32R6;
def SH_MMR6 : StdMMR6Rel, SH_MMR6_DESC, SH_MMR6_ENC, ISA_MICROMIPS32R6;
def SHE_MMR6 : StdMMR6Rel, SHE_MMR6_DESC, SHE_MMR6_ENC, ISA_MICROMIPS32R6;
def LLE_MMR6 : StdMMR6Rel, LLE_MMR6_DESC, LLE_MMR6_ENC, ISA_MICROMIPS32R6;
def LWE_MMR6 : StdMMR6Rel, LWE_MMR6_DESC, LWE_MMR6_ENC, ISA_MICROMIPS32R6;
def LW_MMR6 : StdMMR6Rel, LW_MMR6_DESC, LW_MMR6_ENC, ISA_MICROMIPS32R6;
def LUI_MMR6 : R6MMR6Rel, LUI_MMR6_DESC, LUI_MMR6_ENC, ISA_MICROMIPS32R6;
def ADDU16_MMR6 : StdMMR6Rel, ADDU16_MMR6_DESC, ADDU16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def AND16_MMR6 : StdMMR6Rel, AND16_MMR6_DESC, AND16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def ANDI16_MMR6 : StdMMR6Rel, ANDI16_MMR6_DESC, ANDI16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def NOT16_MMR6 : StdMMR6Rel, NOT16_MMR6_DESC, NOT16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def OR16_MMR6 : StdMMR6Rel, OR16_MMR6_DESC, OR16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def SLL16_MMR6 : StdMMR6Rel, SLL16_MMR6_DESC, SLL16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def SRL16_MMR6 : StdMMR6Rel, SRL16_MMR6_DESC, SRL16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def BREAK16_MMR6 : StdMMR6Rel, BREAK16_MMR6_DESC, BREAK16_MMR6_ENC,
                   ISA_MICROMIPS32R6;
def LI16_MMR6 : StdMMR6Rel, LI16_MMR6_DESC, LI16_MMR6_ENC,
                ISA_MICROMIPS32R6;
def MOVE16_MMR6 : StdMMR6Rel, MOVE16_MMR6_DESC, MOVE16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def SDBBP16_MMR6 : StdMMR6Rel, SDBBP16_MMR6_DESC, SDBBP16_MMR6_ENC,
                   ISA_MICROMIPS32R6;
def SUBU16_MMR6 : StdMMR6Rel, SUBU16_MMR6_DESC, SUBU16_MMR6_ENC,
                  ISA_MICROMIPS32R6;
def XOR16_MMR6 : StdMMR6Rel, XOR16_MMR6_DESC, XOR16_MMR6_ENC,
                 ISA_MICROMIPS32R6;
def RECIP_S_MMR6 : StdMMR6Rel, RECIP_S_MMR6_ENC, RECIP_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def RECIP_D_MMR6 : StdMMR6Rel, RECIP_D_MMR6_ENC, RECIP_D_MMR6_DESC, ISA_MICROMIPS32R6;
def RINT_S_MMR6 : StdMMR6Rel, RINT_S_MMR6_ENC, RINT_S_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def RINT_D_MMR6 : StdMMR6Rel, RINT_D_MMR6_ENC, RINT_D_MMR6_DESC, ISA_MICROMIPS32R6;
def ROUND_L_S_MMR6 : StdMMR6Rel, ROUND_L_S_MMR6_ENC, ROUND_L_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def ROUND_L_D_MMR6 : StdMMR6Rel, ROUND_L_D_MMR6_ENC, ROUND_L_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def ROUND_W_S_MMR6 : StdMMR6Rel, ROUND_W_S_MMR6_ENC, ROUND_W_S_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def ROUND_W_D_MMR6 : StdMMR6Rel, ROUND_W_D_MMR6_ENC, ROUND_W_D_MMR6_DESC,
                     ISA_MICROMIPS32R6;
def SEL_S_MMR6 : StdMMR6Rel, SEL_S_MMR6_ENC, SEL_S_MMR6_DESC, ISA_MICROMIPS32R6;
def SEL_D_MMR6 : StdMMR6Rel, SEL_D_MMR6_ENC, SEL_D_MMR6_DESC, ISA_MICROMIPS32R6;
def SELEQZ_S_MMR6 : StdMMR6Rel, SELEQZ_S_MMR6_ENC, SELEQZ_S_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def SELEQZ_D_MMR6 : StdMMR6Rel, SELEQZ_D_MMR6_ENC, SELEQZ_D_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def SELENZ_S_MMR6 : StdMMR6Rel, SELENZ_S_MMR6_ENC, SELENZ_S_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def SELENZ_D_MMR6 : StdMMR6Rel, SELENZ_D_MMR6_ENC, SELENZ_D_MMR6_DESC,
                    ISA_MICROMIPS32R6;
def CLASS_S_MMR6 : StdMMR6Rel, CLASS_S_MMR6_ENC, CLASS_S_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def CLASS_D_MMR6 : StdMMR6Rel, CLASS_D_MMR6_ENC, CLASS_D_MMR6_DESC,
                   ISA_MICROMIPS32R6;
def TLBINV_MMR6 : StdMMR6Rel, TLBINV_MMR6_ENC, TLBINV_MMR6_DESC,
                  ISA_MICROMIPS32R6;
def TLBINVF_MMR6 : StdMMR6Rel, TLBINVF_MMR6_ENC, TLBINVF_MMR6_DESC,
                   ISA_MICROMIPS32R6;
}

//===----------------------------------------------------------------------===//
//
// MicroMips instruction aliases
//
//===----------------------------------------------------------------------===//

def : MipsInstAlias<"ei", (EI_MMR6 ZERO), 1>, ISA_MICROMIPS32R6;
def : MipsInstAlias<"di", (DI_MMR6 ZERO), 1>, ISA_MICROMIPS32R6;
def : MipsInstAlias<"nop", (SLL_MMR6 ZERO, ZERO, 0), 1>, ISA_MICROMIPS32R6;
def B_MMR6_Pseudo : MipsAsmPseudoInst<(outs), (ins brtarget_mm:$offset),
                                      !strconcat("b", "\t$offset")> {
  string DecoderNamespace = "MicroMipsR6";
}
def : MipsInstAlias<"sync", (SYNC_MMR6 0), 1>, ISA_MICROMIPS32R6;
def : MipsInstAlias<"sdbbp", (SDBBP_MMR6 0), 1>, ISA_MICROMIPS32R6;
def : MipsInstAlias<"rdhwr $rt, $rs",
                    (RDHWR_MMR6 GPR32Opnd:$rt, HWRegsOpnd:$rs, 0), 1>,
                    ISA_MICROMIPS32R6;
def : MipsInstAlias<"mtc0 $rt, $rs",
                    (MTC0_MMR6 COP0Opnd:$rs, GPR32Opnd:$rt, 0), 0>,
                    ISA_MICROMIPS32R6;
def : MipsInstAlias<"mthc0 $rt, $rs",
                    (MTHC0_MMR6 COP0Opnd:$rs, GPR32Opnd:$rt, 0), 0>,
                    ISA_MICROMIPS32R6;
def : MipsInstAlias<"mfc0 $rt, $rs",
                    (MFC0_MMR6 GPR32Opnd:$rt, COP0Opnd:$rs, 0), 0>,
                    ISA_MICROMIPS32R6;
def : MipsInstAlias<"mfhc0 $rt, $rs",
                    (MFHC0_MMR6 GPR32Opnd:$rt, COP0Opnd:$rs, 0), 0>,
                    ISA_MICROMIPS32R6;

//===----------------------------------------------------------------------===//
//
// MicroMips arbitrary patterns that map to one or more instructions
//
//===----------------------------------------------------------------------===//

def : MipsPat<(store GPRMM16:$src, addrimm4lsl2:$addr),
              (SW16_MMR6 GPRMM16:$src, addrimm4lsl2:$addr)>, ISA_MICROMIPS32R6;
