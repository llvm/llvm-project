//=- Mips32r6InstrInfo.td - Mips32r6 Instruction Information -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips32r6 instructions.
//
//===----------------------------------------------------------------------===//

include "Mips32r6InstrFormats.td"

// Notes about removals/changes from MIPS32r6:
// Reencoded: jr -> jalr
// Reencoded: jr.hb -> jalr.hb

def brtarget21 : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget21OpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget21";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def brtarget26 : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget26OpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget26";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def jmpoffset16 : Operand<OtherVT> {
  let EncoderMethod = "getJumpOffset16OpValue";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def calloffset16 : Operand<iPTR> {
  let EncoderMethod = "getJumpOffset16OpValue";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

//===----------------------------------------------------------------------===//
//
// Instruction Encodings
//
//===----------------------------------------------------------------------===//

class ADDIUPC_ENC : PCREL19_FM<OPCODE2_ADDIUPC>;
class ALIGN_ENC  : SPECIAL3_ALIGN_FM<OPCODE6_ALIGN>;
class ALUIPC_ENC : PCREL16_FM<OPCODE5_ALUIPC>;
class AUI_ENC    : AUI_FM;
class AUIPC_ENC  : PCREL16_FM<OPCODE5_AUIPC>;

class BAL_ENC   : BAL_FM;
class BALC_ENC  : BRANCH_OFF26_FM<0b111010>;
class BC_ENC    : BRANCH_OFF26_FM<0b110010>;
class BEQC_ENC  : CMP_BRANCH_2R_OFF16_FM<OPGROUP_ADDI>,
                  DecodeDisambiguates<"AddiGroupBranch">;
class BEQZALC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_ADDI>,
                    DecodeDisambiguatedBy<"DaddiGroupBranch">;
class BNEC_ENC  : CMP_BRANCH_2R_OFF16_FM<OPGROUP_DADDI>,
                  DecodeDisambiguates<"DaddiGroupBranch">;
class BNEZALC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_DADDI>,
                    DecodeDisambiguatedBy<"DaddiGroupBranch">;

class BLTZC_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM<OPGROUP_BGTZL>,
                  DecodeDisambiguates<"BgtzlGroupBranch">;
class BGEC_ENC  : CMP_BRANCH_2R_OFF16_FM<OPGROUP_BLEZL>,
                  DecodeDisambiguatedBy<"BlezlGroupBranch">;
class BGEUC_ENC : CMP_BRANCH_2R_OFF16_FM<OPGROUP_BLEZ>,
                  DecodeDisambiguatedBy<"BlezGroupBranch">;
class BGEZC_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM<OPGROUP_BLEZL>,
                  DecodeDisambiguates<"BlezlGroupBranch">;
class BGTZALC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_BGTZ>,
                    DecodeDisambiguatedBy<"BgtzGroupBranch">;

class BLTC_ENC : CMP_BRANCH_2R_OFF16_FM<OPGROUP_BGTZL>,
                 DecodeDisambiguatedBy<"BgtzlGroupBranch">;
class BLTUC_ENC : CMP_BRANCH_2R_OFF16_FM<OPGROUP_BGTZ>,
                  DecodeDisambiguatedBy<"BgtzGroupBranch">;

class BLEZC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_BLEZL>,
                  DecodeDisambiguatedBy<"BlezlGroupBranch">;
class BLTZALC_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM<OPGROUP_BGTZ>,
                    DecodeDisambiguates<"BgtzGroupBranch">;
class BGTZC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_BGTZL>,
                  DecodeDisambiguatedBy<"BgtzlGroupBranch">;

class BEQZC_ENC : CMP_BRANCH_OFF21_FM<0b110110>;
class BGEZALC_ENC : CMP_BRANCH_1R_BOTH_OFF16_FM<OPGROUP_BLEZ>,
                    DecodeDisambiguates<"BlezGroupBranch">;
class BNEZC_ENC : CMP_BRANCH_OFF21_FM<0b111110>;

class BC1EQZ_ENC : COP1_BCCZ_FM<OPCODE5_BC1EQZ>;
class BC1NEZ_ENC : COP1_BCCZ_FM<OPCODE5_BC1NEZ>;
class BC2EQZ_ENC : COP2_BCCZ_FM<OPCODE5_BC2EQZ>;
class BC2NEZ_ENC : COP2_BCCZ_FM<OPCODE5_BC2NEZ>;

class JIALC_ENC : JMP_IDX_COMPACT_FM<0b111110>;
class JIC_ENC   : JMP_IDX_COMPACT_FM<0b110110>;
class JR_HB_R6_ENC : JR_HB_R6_FM<OPCODE6_JALR>;
class BITSWAP_ENC : SPECIAL3_2R_FM<OPCODE6_BITSWAP>;
class BLEZALC_ENC : CMP_BRANCH_1R_RT_OFF16_FM<OPGROUP_BLEZ>,
                    DecodeDisambiguatedBy<"BlezGroupBranch">;
class BNVC_ENC   : CMP_BRANCH_2R_OFF16_FM<OPGROUP_DADDI>,
                   DecodeDisambiguatedBy<"DaddiGroupBranch">;
class BOVC_ENC   : CMP_BRANCH_2R_OFF16_FM<OPGROUP_ADDI>,
                   DecodeDisambiguatedBy<"AddiGroupBranch">;
class DIV_ENC    : SPECIAL_3R_FM<0b00010, 0b011010>;
class DIVU_ENC   : SPECIAL_3R_FM<0b00010, 0b011011>;
class MOD_ENC    : SPECIAL_3R_FM<0b00011, 0b011010>;
class MODU_ENC   : SPECIAL_3R_FM<0b00011, 0b011011>;
class MUH_ENC    : SPECIAL_3R_FM<0b00011, 0b011000>;
class MUHU_ENC   : SPECIAL_3R_FM<0b00011, 0b011001>;
class MUL_R6_ENC : SPECIAL_3R_FM<0b00010, 0b011000>;
class MULU_ENC   : SPECIAL_3R_FM<0b00010, 0b011001>;

class MADDF_S_ENC  : COP1_3R_FM<0b011000, FIELD_FMT_S>;
class MADDF_D_ENC  : COP1_3R_FM<0b011000, FIELD_FMT_D>;
class MSUBF_S_ENC  : COP1_3R_FM<0b011001, FIELD_FMT_S>;
class MSUBF_D_ENC  : COP1_3R_FM<0b011001, FIELD_FMT_D>;

class SEL_D_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_D>;
class SEL_S_ENC  : COP1_3R_FM<0b010000, FIELD_FMT_S>;

class SELEQZ_ENC : SPECIAL_3R_FM<0b00000, 0b110101>;
class SELNEZ_ENC : SPECIAL_3R_FM<0b00000, 0b110111>;

class LWPC_ENC   : PCREL19_FM<OPCODE2_LWPC>;
class LWUPC_ENC  : PCREL19_FM<OPCODE2_LWUPC>;

class MAX_S_ENC : COP1_3R_FM<0b011101, FIELD_FMT_S>;
class MAX_D_ENC : COP1_3R_FM<0b011101, FIELD_FMT_D>;
class MIN_S_ENC : COP1_3R_FM<0b011100, FIELD_FMT_S>;
class MIN_D_ENC : COP1_3R_FM<0b011100, FIELD_FMT_D>;

class MAXA_S_ENC : COP1_3R_FM<0b011111, FIELD_FMT_S>;
class MAXA_D_ENC : COP1_3R_FM<0b011111, FIELD_FMT_D>;
class MINA_S_ENC : COP1_3R_FM<0b011110, FIELD_FMT_S>;
class MINA_D_ENC : COP1_3R_FM<0b011110, FIELD_FMT_D>;

class SELEQZ_S_ENC : COP1_3R_FM<0b010100, FIELD_FMT_S>;
class SELEQZ_D_ENC : COP1_3R_FM<0b010100, FIELD_FMT_D>;
class SELNEZ_S_ENC : COP1_3R_FM<0b010111, FIELD_FMT_S>;
class SELNEZ_D_ENC : COP1_3R_FM<0b010111, FIELD_FMT_D>;

class RINT_S_ENC : COP1_2R_FM<0b011010, FIELD_FMT_S>;
class RINT_D_ENC : COP1_2R_FM<0b011010, FIELD_FMT_D>;
class CLASS_S_ENC : COP1_2R_FM<0b011011, FIELD_FMT_S>;
class CLASS_D_ENC : COP1_2R_FM<0b011011, FIELD_FMT_D>;

class CACHE_ENC : SPECIAL3_MEM_FM<OPCODE6_CACHE>;
class PREF_ENC : SPECIAL3_MEM_FM<OPCODE6_PREF>;

class LDC2_R6_ENC : COP2LDST_FM<OPCODE5_LDC2>;
class LWC2_R6_ENC : COP2LDST_FM<OPCODE5_LWC2>;
class SDC2_R6_ENC : COP2LDST_FM<OPCODE5_SDC2>;
class SWC2_R6_ENC : COP2LDST_FM<OPCODE5_SWC2>;

class LSA_R6_ENC : SPECIAL_LSA_FM<OPCODE6_LSA>;

class LL_R6_ENC : SPECIAL3_LL_SC_FM<OPCODE6_LL>;
class SC_R6_ENC : SPECIAL3_LL_SC_FM<OPCODE6_SC>;

class CLO_R6_ENC : SPECIAL_2R_FM<OPCODE6_CLO>;
class CLZ_R6_ENC : SPECIAL_2R_FM<OPCODE6_CLZ>;

class SDBBP_R6_ENC : SPECIAL_SDBBP_FM;

//===----------------------------------------------------------------------===//
//
// Instruction Multiclasses
//
//===----------------------------------------------------------------------===//

class CMP_CONDN_DESC_BASE<string CondStr, string Typestr,
                          RegisterOperand FGROpnd,
                          SDPatternOperator Op = null_frag> {
  dag OutOperandList = (outs FGRCCOpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat("cmp.", CondStr, ".", Typestr, "\t$fd, $fs, $ft");
  list<dag> Pattern = [(set FGRCCOpnd:$fd, (Op FGROpnd:$fs, FGROpnd:$ft))];
}

multiclass CMP_CC_M <FIELD_CMP_FORMAT Format, string Typestr,
                     RegisterOperand FGROpnd>{
  let AdditionalPredicates = [NotInMicroMips] in {
    def CMP_F_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_AF>,
                      CMP_CONDN_DESC_BASE<"af", Typestr, FGROpnd>,
                      ISA_MIPS32R6, HARDFLOAT;
    def CMP_UN_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_UN>,
                       CMP_CONDN_DESC_BASE<"un", Typestr, FGROpnd, setuo>,
                       ISA_MIPS32R6, HARDFLOAT;
    def CMP_EQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_EQ>,
                       CMP_CONDN_DESC_BASE<"eq", Typestr, FGROpnd, setoeq>,
                       ISA_MIPS32R6, HARDFLOAT;
    def CMP_UEQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_UEQ>,
                        CMP_CONDN_DESC_BASE<"ueq", Typestr, FGROpnd, setueq>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_LT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_LT>,
                       CMP_CONDN_DESC_BASE<"lt", Typestr, FGROpnd, setolt>,
                       ISA_MIPS32R6, HARDFLOAT;
    def CMP_ULT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_ULT>,
                        CMP_CONDN_DESC_BASE<"ult", Typestr, FGROpnd, setult>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_LE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_LE>,
                       CMP_CONDN_DESC_BASE<"le", Typestr, FGROpnd, setole>,
                       ISA_MIPS32R6, HARDFLOAT;
    def CMP_ULE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_ULE>,
                        CMP_CONDN_DESC_BASE<"ule", Typestr, FGROpnd, setule>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SAF_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SAF>,
                        CMP_CONDN_DESC_BASE<"saf", Typestr, FGROpnd>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SUN_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SUN>,
                        CMP_CONDN_DESC_BASE<"sun", Typestr, FGROpnd>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SEQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SEQ>,
                        CMP_CONDN_DESC_BASE<"seq", Typestr, FGROpnd>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SUEQ_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SUEQ>,
                         CMP_CONDN_DESC_BASE<"sueq", Typestr, FGROpnd>,
                         ISA_MIPS32R6, HARDFLOAT;
    def CMP_SLT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SLT>,
                        CMP_CONDN_DESC_BASE<"slt", Typestr, FGROpnd>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SULT_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SULT>,
                         CMP_CONDN_DESC_BASE<"sult", Typestr, FGROpnd>,
                         ISA_MIPS32R6, HARDFLOAT;
    def CMP_SLE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SLE>,
                        CMP_CONDN_DESC_BASE<"sle", Typestr, FGROpnd>,
                        ISA_MIPS32R6, HARDFLOAT;
    def CMP_SULE_#NAME : COP1_CMP_CONDN_FM<Format, FIELD_CMP_COND_SULE>,
                         CMP_CONDN_DESC_BASE<"sule", Typestr, FGROpnd>,
                         ISA_MIPS32R6, HARDFLOAT;
  }
}

//===----------------------------------------------------------------------===//
//
// Instruction Descriptions
//
//===----------------------------------------------------------------------===//

class PCREL_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd> : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins ImmOpnd:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $imm");
  list<dag> Pattern = [];
}

class ADDIUPC_DESC : PCREL_DESC_BASE<"addiupc", GPR32Opnd, simm19_lsl2>;
class LWPC_DESC: PCREL_DESC_BASE<"lwpc", GPR32Opnd, simm19_lsl2>;
class LWUPC_DESC: PCREL_DESC_BASE<"lwupc", GPR32Opnd, simm19_lsl2>;

class ALIGN_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                      Operand ImmOpnd>  : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$bp);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $bp");
  list<dag> Pattern = [];
}

class ALIGN_DESC : ALIGN_DESC_BASE<"align", GPR32Opnd, uimm2>;

class ALUIPC_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $imm");
  list<dag> Pattern = [];
}

class ALUIPC_DESC : ALUIPC_DESC_BASE<"aluipc", GPR32Opnd>;
class AUIPC_DESC : ALUIPC_DESC_BASE<"auipc", GPR32Opnd>;

class AUI_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rs);
  dag InOperandList = (ins GPROpnd:$rt, simm16:$imm);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt, $imm");
  list<dag> Pattern = [];
}

class AUI_DESC : AUI_DESC_BASE<"aui", GPR32Opnd>;

class BRANCH_DESC_BASE {
  bit isBranch = 1;
  bit isTerminator = 1;
  bit hasDelaySlot = 0;
}

class BC_DESC_BASE<string instr_asm, DAGOperand opnd> : BRANCH_DESC_BASE,
    MipsR6Arch<instr_asm> {
  dag InOperandList = (ins opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$offset");
  bit isBarrier = 1;
}

class CMP_BC_DESC_BASE<string instr_asm, DAGOperand opnd,
                       RegisterOperand GPROpnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $rt, $offset");
  list<Register> Defs = [AT];
}

class CMP_CBR_EQNE_Z_DESC_BASE<string instr_asm, DAGOperand opnd,
                               RegisterOperand GPROpnd> : BRANCH_DESC_BASE {
  dag InOperandList = (ins GPROpnd:$rs, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rs, $offset");
  list<Register> Defs = [AT];
}

class CMP_CBR_RT_Z_DESC_BASE<string instr_asm, DAGOperand opnd,
                             RegisterOperand GPROpnd>
    : BRANCH_DESC_BASE, MipsR6Arch<instr_asm> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  dag OutOperandList = (outs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $offset");
  list<Register> Defs = [AT];
}

class BAL_DESC : BC_DESC_BASE<"bal", brtarget> {
  bit isCall = 1;
  bit hasDelaySlot = 1;
  list<Register> Defs = [RA];
}

class BALC_DESC : BC_DESC_BASE<"balc", brtarget26> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class BC_DESC : BC_DESC_BASE<"bc", brtarget26>;
class BGEC_DESC : CMP_BC_DESC_BASE<"bgec", brtarget, GPR32Opnd>;
class BGEUC_DESC : CMP_BC_DESC_BASE<"bgeuc", brtarget, GPR32Opnd>;
class BEQC_DESC : CMP_BC_DESC_BASE<"beqc", brtarget, GPR32Opnd>;
class BNEC_DESC : CMP_BC_DESC_BASE<"bnec", brtarget, GPR32Opnd>;

class BLTC_DESC : CMP_BC_DESC_BASE<"bltc", brtarget, GPR32Opnd>;
class BLTUC_DESC : CMP_BC_DESC_BASE<"bltuc", brtarget, GPR32Opnd>;

class BLTZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bltzc", brtarget, GPR32Opnd>;
class BGEZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgezc", brtarget, GPR32Opnd>;

class BLEZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"blezc", brtarget, GPR32Opnd>;
class BGTZC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgtzc", brtarget, GPR32Opnd>;

class BEQZC_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"beqzc", brtarget21, GPR32Opnd>;
class BNEZC_DESC : CMP_CBR_EQNE_Z_DESC_BASE<"bnezc", brtarget21, GPR32Opnd>;

class COP1_BCCZ_DESC_BASE<string instr_asm> : BRANCH_DESC_BASE {
  dag InOperandList = (ins FGR64Opnd:$ft, brtarget:$offset);
  dag OutOperandList = (outs);
  string AsmString = instr_asm;
  bit hasDelaySlot = 1;
}

class BC1EQZ_DESC : COP1_BCCZ_DESC_BASE<"bc1eqz $ft, $offset">;
class BC1NEZ_DESC : COP1_BCCZ_DESC_BASE<"bc1nez $ft, $offset">;

class COP2_BCCZ_DESC_BASE<string instr_asm> : BRANCH_DESC_BASE {
  dag InOperandList = (ins COP2Opnd:$ct, brtarget:$offset);
  dag OutOperandList = (outs);
  string AsmString = instr_asm;
  bit hasDelaySlot = 1;
}

class BC2EQZ_DESC : COP2_BCCZ_DESC_BASE<"bc2eqz $ct, $offset">;
class BC2NEZ_DESC : COP2_BCCZ_DESC_BASE<"bc2nez $ct, $offset">;

class BOVC_DESC   : CMP_BC_DESC_BASE<"bovc", brtarget, GPR32Opnd>;
class BNVC_DESC   : CMP_BC_DESC_BASE<"bnvc", brtarget, GPR32Opnd>;

class JMP_IDX_COMPACT_DESC_BASE<string opstr, DAGOperand opnd,
                                RegisterOperand GPROpnd>
    : MipsR6Arch<opstr> {
  dag InOperandList = (ins GPROpnd:$rt, opnd:$offset);
  string AsmString = !strconcat(opstr, "\t$rt, $offset");
  list<dag> Pattern = [];
  bit isTerminator = 1;
  bit hasDelaySlot = 0;
}

class JIALC_DESC : JMP_IDX_COMPACT_DESC_BASE<"jialc", calloffset16,
                                             GPR32Opnd> {
  bit isCall = 1;
  list<Register> Defs = [RA];
}

class JIC_DESC : JMP_IDX_COMPACT_DESC_BASE<"jic", jmpoffset16, GPR32Opnd> {
  bit isBarrier = 1;
  list<Register> Defs = [AT];
}

class JR_HB_R6_DESC : JR_HB_DESC_BASE<"jr.hb", GPR32Opnd> {
  bit isBranch = 1;
  bit isIndirectBranch = 1;
  bit hasDelaySlot = 1;
  bit isTerminator=1;
  bit isBarrier=1;
}

class BITSWAP_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt");
  list<dag> Pattern = [];
}

class BITSWAP_DESC : BITSWAP_DESC_BASE<"bitswap", GPR32Opnd>;

class DIVMOD_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                       SDPatternOperator Op=null_frag>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set GPROpnd:$rd, (Op GPROpnd:$rs, GPROpnd:$rt))];

  // This instruction doesn't trap division by zero itself. We must insert
  // teq instructions as well.
  bit usesCustomInserter = 1;
}

class DIV_DESC  : DIVMOD_DESC_BASE<"div", GPR32Opnd, sdiv>;
class DIVU_DESC : DIVMOD_DESC_BASE<"divu", GPR32Opnd, udiv>;
class MOD_DESC  : DIVMOD_DESC_BASE<"mod", GPR32Opnd, srem>;
class MODU_DESC : DIVMOD_DESC_BASE<"modu", GPR32Opnd, urem>;

class BEQZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"beqzalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGEZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgezalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BGTZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bgtzalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLEZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"blezalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BLTZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bltzalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class BNEZALC_DESC : CMP_CBR_RT_Z_DESC_BASE<"bnezalc", brtarget, GPR32Opnd> {
  list<Register> Defs = [RA];
}

class MUL_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                       SDPatternOperator Op=null_frag> : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set GPROpnd:$rd, (Op GPROpnd:$rs, GPROpnd:$rt))];
}

class MUH_DESC    : MUL_R6_DESC_BASE<"muh", GPR32Opnd, mulhs>;
class MUHU_DESC   : MUL_R6_DESC_BASE<"muhu", GPR32Opnd, mulhu>;
class MUL_R6_DESC : MUL_R6_DESC_BASE<"mul", GPR32Opnd, mul>;
class MULU_DESC   : MUL_R6_DESC_BASE<"mulu", GPR32Opnd>;

class COP1_SEL_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGRCCOpnd:$fd_in, FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [(set FGROpnd:$fd, (select FGRCCOpnd:$fd_in,
                                                 FGROpnd:$ft,
                                                 FGROpnd:$fs))];
  string Constraints = "$fd_in = $fd";
}

class SEL_D_DESC : COP1_SEL_DESC_BASE<"sel.d", FGR64Opnd> {
  // We must insert a SUBREG_TO_REG around $fd_in
  bit usesCustomInserter = 1;
}
class SEL_S_DESC : COP1_SEL_DESC_BASE<"sel.s", FGR32Opnd>;

class SELEQNE_Z_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [];
}

class SELEQZ_DESC : SELEQNE_Z_DESC_BASE<"seleqz", GPR32Opnd>;
class SELNEZ_DESC : SELEQNE_Z_DESC_BASE<"selnez", GPR32Opnd>;

class COP1_4R_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fd_in, FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
  string Constraints = "$fd_in = $fd";
}

class MADDF_S_DESC  : COP1_4R_DESC_BASE<"maddf.s", FGR32Opnd>;
class MADDF_D_DESC  : COP1_4R_DESC_BASE<"maddf.d", FGR64Opnd>;
class MSUBF_S_DESC  : COP1_4R_DESC_BASE<"msubf.s", FGR32Opnd>;
class MSUBF_D_DESC  : COP1_4R_DESC_BASE<"msubf.d", FGR64Opnd>;

class MAX_MIN_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
}

class MAX_S_DESC : MAX_MIN_DESC_BASE<"max.s", FGR32Opnd>;
class MAX_D_DESC : MAX_MIN_DESC_BASE<"max.d", FGR64Opnd>;
class MIN_S_DESC : MAX_MIN_DESC_BASE<"min.s", FGR32Opnd>;
class MIN_D_DESC : MAX_MIN_DESC_BASE<"min.d", FGR64Opnd>;

class MAXA_S_DESC : MAX_MIN_DESC_BASE<"maxa.s", FGR32Opnd>;
class MAXA_D_DESC : MAX_MIN_DESC_BASE<"maxa.d", FGR64Opnd>;
class MINA_S_DESC : MAX_MIN_DESC_BASE<"mina.s", FGR32Opnd>;
class MINA_D_DESC : MAX_MIN_DESC_BASE<"mina.d", FGR64Opnd>;

class SELEQNEZ_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs, FGROpnd:$ft);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs, $ft");
  list<dag> Pattern = [];
}

class SELEQZ_S_DESC : SELEQNEZ_DESC_BASE<"seleqz.s", FGR32Opnd>;
class SELEQZ_D_DESC : SELEQNEZ_DESC_BASE<"seleqz.d", FGR64Opnd>;
class SELNEZ_S_DESC : SELEQNEZ_DESC_BASE<"selnez.s", FGR32Opnd>;
class SELNEZ_D_DESC : SELEQNEZ_DESC_BASE<"selnez.d", FGR64Opnd>;

class CLASS_RINT_DESC_BASE<string instr_asm, RegisterOperand FGROpnd> {
  dag OutOperandList = (outs FGROpnd:$fd);
  dag InOperandList = (ins FGROpnd:$fs);
  string AsmString = !strconcat(instr_asm, "\t$fd, $fs");
  list<dag> Pattern = [];
}

class RINT_S_DESC : CLASS_RINT_DESC_BASE<"rint.s", FGR32Opnd>;
class RINT_D_DESC : CLASS_RINT_DESC_BASE<"rint.d", FGR64Opnd>;
class CLASS_S_DESC : CLASS_RINT_DESC_BASE<"class.s", FGR32Opnd>;
class CLASS_D_DESC : CLASS_RINT_DESC_BASE<"class.d", FGR64Opnd>;

class CACHE_HINT_DESC<string instr_asm, Operand MemOpnd,
                      RegisterOperand GPROpnd> : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins MemOpnd:$addr, uimm5:$hint);
  string AsmString = !strconcat(instr_asm, "\t$hint, $addr");
  list<dag> Pattern = [];
  string DecoderMethod = "DecodeCacheeOp_CacheOpR6";
}

class CACHE_DESC : CACHE_HINT_DESC<"cache", mem_simm9, GPR32Opnd>;
class PREF_DESC : CACHE_HINT_DESC<"pref", mem_simm9, GPR32Opnd>;

class COP2LD_DESC_BASE<string instr_asm, RegisterOperand COPOpnd> {
  dag OutOperandList = (outs COPOpnd:$rt);
  dag InOperandList = (ins mem_simm11:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayLoad = 1;
  string DecoderMethod = "DecodeFMemCop2R6";
}

class LDC2_R6_DESC : COP2LD_DESC_BASE<"ldc2", COP2Opnd>;
class LWC2_R6_DESC : COP2LD_DESC_BASE<"lwc2", COP2Opnd>;

class COP2ST_DESC_BASE<string instr_asm, RegisterOperand COPOpnd> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins COPOpnd:$rt, mem_simm11:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayStore = 1;
  string DecoderMethod = "DecodeFMemCop2R6";
}

class SDC2_R6_DESC : COP2ST_DESC_BASE<"sdc2", COP2Opnd>;
class SWC2_R6_DESC : COP2ST_DESC_BASE<"swc2", COP2Opnd>;

class LSA_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd,
                       Operand ImmOpnd> : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt, ImmOpnd:$imm2);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt, $imm2");
  list<dag> Pattern = [];
}

class LSA_R6_DESC : LSA_R6_DESC_BASE<"lsa", GPR32Opnd, uimm2_plus1>;

class LL_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayLoad = 1;
}

class LL_R6_DESC : LL_R6_DESC_BASE<"ll", GPR32Opnd>;

class SC_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$dst);
  dag InOperandList = (ins GPROpnd:$rt, mem_simm9:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [];
  bit mayStore = 1;
  string Constraints = "$rt = $dst";
}

class SC_R6_DESC : SC_R6_DESC_BASE<"sc", GPR32Opnd>;

class CLO_CLZ_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd>
    : MipsR6Arch<instr_asm> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs");
}

class CLO_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> :
    CLO_CLZ_R6_DESC_BASE<instr_asm, GPROpnd> {
  list<dag> Pattern = [(set GPROpnd:$rd, (ctlz (not GPROpnd:$rs)))];
}

class CLZ_R6_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> :
    CLO_CLZ_R6_DESC_BASE<instr_asm, GPROpnd> {
  list<dag> Pattern = [(set GPROpnd:$rd, (ctlz GPROpnd:$rs))];
}

class CLO_R6_DESC : CLO_R6_DESC_BASE<"clo", GPR32Opnd>;
class CLZ_R6_DESC : CLZ_R6_DESC_BASE<"clz", GPR32Opnd>;

class SDBBP_R6_DESC {
  dag OutOperandList = (outs);
  dag InOperandList = (ins uimm20:$code_);
  string AsmString = "sdbbp\t$code_";
  list<dag> Pattern = [];
}

//===----------------------------------------------------------------------===//
//
// Instruction Definitions
//
//===----------------------------------------------------------------------===//

def ADDIUPC : R6MMR6Rel, ADDIUPC_ENC, ADDIUPC_DESC, ISA_MIPS32R6;
def ALIGN : R6MMR6Rel, ALIGN_ENC, ALIGN_DESC, ISA_MIPS32R6;
def ALUIPC : R6MMR6Rel, ALUIPC_ENC, ALUIPC_DESC, ISA_MIPS32R6;
def AUI : R6MMR6Rel, AUI_ENC, AUI_DESC, ISA_MIPS32R6;
def AUIPC : R6MMR6Rel, AUIPC_ENC, AUIPC_DESC, ISA_MIPS32R6;
def BAL : BAL_ENC, BAL_DESC, ISA_MIPS32R6;
def BALC : R6MMR6Rel, BALC_ENC, BALC_DESC, ISA_MIPS32R6;
def BC1EQZ : BC1EQZ_ENC, BC1EQZ_DESC, ISA_MIPS32R6, HARDFLOAT;
def BC1NEZ : BC1NEZ_ENC, BC1NEZ_DESC, ISA_MIPS32R6, HARDFLOAT;
def BC2EQZ : BC2EQZ_ENC, BC2EQZ_DESC, ISA_MIPS32R6;
def BC2NEZ : BC2NEZ_ENC, BC2NEZ_DESC, ISA_MIPS32R6;
def BC : R6MMR6Rel, BC_ENC, BC_DESC, ISA_MIPS32R6;
def BEQC : BEQC_ENC, BEQC_DESC, ISA_MIPS32R6;
def BEQZALC : R6MMR6Rel, BEQZALC_ENC, BEQZALC_DESC, ISA_MIPS32R6;
def BEQZC : BEQZC_ENC, BEQZC_DESC, ISA_MIPS32R6;
def BGEC : BGEC_ENC, BGEC_DESC, ISA_MIPS32R6;
def BGEUC : BGEUC_ENC, BGEUC_DESC, ISA_MIPS32R6;
def BGEZALC : R6MMR6Rel, BGEZALC_ENC, BGEZALC_DESC, ISA_MIPS32R6;
def BGEZC : BGEZC_ENC, BGEZC_DESC, ISA_MIPS32R6;
def BGTZALC : R6MMR6Rel, BGTZALC_ENC, BGTZALC_DESC, ISA_MIPS32R6;
def BGTZC : BGTZC_ENC, BGTZC_DESC, ISA_MIPS32R6;
def BITSWAP : R6MMR6Rel, BITSWAP_ENC, BITSWAP_DESC, ISA_MIPS32R6;
def BLEZALC : R6MMR6Rel, BLEZALC_ENC, BLEZALC_DESC, ISA_MIPS32R6;
def BLEZC : BLEZC_ENC, BLEZC_DESC, ISA_MIPS32R6;
def BLTC : BLTC_ENC, BLTC_DESC, ISA_MIPS32R6;
def BLTUC : BLTUC_ENC, BLTUC_DESC, ISA_MIPS32R6;
def BLTZALC : R6MMR6Rel, BLTZALC_ENC, BLTZALC_DESC, ISA_MIPS32R6;
def BLTZC : BLTZC_ENC, BLTZC_DESC, ISA_MIPS32R6;
def BNEC : BNEC_ENC, BNEC_DESC, ISA_MIPS32R6;
def BNEZALC : R6MMR6Rel, BNEZALC_ENC, BNEZALC_DESC, ISA_MIPS32R6;
def BNEZC : BNEZC_ENC, BNEZC_DESC, ISA_MIPS32R6;
def BNVC : BNVC_ENC, BNVC_DESC, ISA_MIPS32R6;
def BOVC : BOVC_ENC, BOVC_DESC, ISA_MIPS32R6;
def CACHE_R6 : R6MMR6Rel, CACHE_ENC, CACHE_DESC, ISA_MIPS32R6;
def CLASS_D : CLASS_D_ENC, CLASS_D_DESC, ISA_MIPS32R6, HARDFLOAT;
def CLASS_S : CLASS_S_ENC, CLASS_S_DESC, ISA_MIPS32R6, HARDFLOAT;
def CLO_R6 : R6MMR6Rel, CLO_R6_ENC, CLO_R6_DESC, ISA_MIPS32R6;
def CLZ_R6 : R6MMR6Rel, CLZ_R6_ENC, CLZ_R6_DESC, ISA_MIPS32R6;
defm S : CMP_CC_M<FIELD_CMP_FORMAT_S, "s", FGR32Opnd>;
defm D : CMP_CC_M<FIELD_CMP_FORMAT_D, "d", FGR64Opnd>;
def DIV : R6MMR6Rel, DIV_ENC, DIV_DESC, ISA_MIPS32R6;
def DIVU : R6MMR6Rel, DIVU_ENC, DIVU_DESC, ISA_MIPS32R6;
def JIALC : R6MMR6Rel, JIALC_ENC, JIALC_DESC, ISA_MIPS32R6;
def JIC : R6MMR6Rel, JIC_ENC, JIC_DESC, ISA_MIPS32R6;
def JR_HB_R6 : JR_HB_R6_ENC, JR_HB_R6_DESC, ISA_MIPS32R6;
def LDC2_R6 : LDC2_R6_ENC, LDC2_R6_DESC, ISA_MIPS32R6;
def LL_R6 : LL_R6_ENC, LL_R6_DESC, ISA_MIPS32R6;
def LSA_R6 : R6MMR6Rel, LSA_R6_ENC, LSA_R6_DESC, ISA_MIPS32R6;
def LWC2_R6 : LWC2_R6_ENC, LWC2_R6_DESC, ISA_MIPS32R6;
def LWPC : R6MMR6Rel, LWPC_ENC, LWPC_DESC, ISA_MIPS32R6;
def LWUPC : LWUPC_ENC, LWUPC_DESC, ISA_MIPS32R6;
let AdditionalPredicates = [NotInMicroMips] in {
  def MADDF_S : MADDF_S_ENC, MADDF_S_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MADDF_D : MADDF_D_ENC, MADDF_D_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MAXA_D  : MAXA_D_ENC, MAXA_D_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MAXA_S  : MAXA_S_ENC, MAXA_S_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MAX_D   : MAX_D_ENC, MAX_D_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MAX_S   : MAX_S_ENC, MAX_S_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MINA_D  : MINA_D_ENC, MINA_D_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MINA_S  : MINA_S_ENC, MINA_S_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MIN_D   : MIN_D_ENC, MIN_D_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MIN_S   : MIN_S_ENC, MIN_S_DESC, ISA_MIPS32R6, HARDFLOAT;
}
def MOD : R6MMR6Rel, MOD_ENC, MOD_DESC, ISA_MIPS32R6;
def MODU : R6MMR6Rel, MODU_ENC, MODU_DESC, ISA_MIPS32R6;
let AdditionalPredicates = [NotInMicroMips] in {
  def MSUBF_S : MSUBF_S_ENC, MSUBF_S_DESC, ISA_MIPS32R6, HARDFLOAT;
  def MSUBF_D : MSUBF_D_ENC, MSUBF_D_DESC, ISA_MIPS32R6, HARDFLOAT;
}
def MUH    : R6MMR6Rel, MUH_ENC, MUH_DESC, ISA_MIPS32R6;
def MUHU   : R6MMR6Rel, MUHU_ENC, MUHU_DESC, ISA_MIPS32R6;
def MUL_R6 : R6MMR6Rel, MUL_R6_ENC, MUL_R6_DESC, ISA_MIPS32R6;
def MULU   : R6MMR6Rel, MULU_ENC, MULU_DESC, ISA_MIPS32R6;
def NAL; // BAL with rd=0
def PREF_R6 : R6MMR6Rel, PREF_ENC, PREF_DESC, ISA_MIPS32R6;
def RINT_D : RINT_D_ENC, RINT_D_DESC, ISA_MIPS32R6, HARDFLOAT;
def RINT_S : RINT_S_ENC, RINT_S_DESC, ISA_MIPS32R6, HARDFLOAT;
def SC_R6 : SC_R6_ENC, SC_R6_DESC, ISA_MIPS32R6;
let AdditionalPredicates = [NotInMicroMips] in {
def SDBBP_R6 : SDBBP_R6_ENC, SDBBP_R6_DESC, ISA_MIPS32R6;
}
def SDC2_R6 : SDC2_R6_ENC, SDC2_R6_DESC, ISA_MIPS32R6;
def SELEQZ : R6MMR6Rel, SELEQZ_ENC, SELEQZ_DESC, ISA_MIPS32R6, GPR_32;
def SELEQZ_D : SELEQZ_D_ENC, SELEQZ_D_DESC, ISA_MIPS32R6, HARDFLOAT;
def SELEQZ_S : SELEQZ_S_ENC, SELEQZ_S_DESC, ISA_MIPS32R6, HARDFLOAT;
def SELNEZ : R6MMR6Rel, SELNEZ_ENC, SELNEZ_DESC, ISA_MIPS32R6, GPR_32;
def SELNEZ_D : SELNEZ_D_ENC, SELNEZ_D_DESC, ISA_MIPS32R6, HARDFLOAT;
def SELNEZ_S : SELNEZ_S_ENC, SELNEZ_S_DESC, ISA_MIPS32R6, HARDFLOAT;
def SEL_D    : SEL_D_ENC, SEL_D_DESC, ISA_MIPS32R6, HARDFLOAT;
def SEL_S    : SEL_S_ENC, SEL_S_DESC, ISA_MIPS32R6, HARDFLOAT;
def SWC2_R6 : SWC2_R6_ENC, SWC2_R6_DESC, ISA_MIPS32R6;

//===----------------------------------------------------------------------===//
//
// Instruction Aliases
//
//===----------------------------------------------------------------------===//

let AdditionalPredicates = [NotInMicroMips] in {
def : MipsInstAlias<"sdbbp", (SDBBP_R6 0)>, ISA_MIPS32R6;
}
def : MipsInstAlias<"jr $rs", (JALR ZERO, GPR32Opnd:$rs), 1>, ISA_MIPS32R6;

//===----------------------------------------------------------------------===//
//
// Patterns and Pseudo Instructions
//
//===----------------------------------------------------------------------===//

// f32 comparisons supported via another comparison
def : MipsPat<(setone f32:$lhs, f32:$rhs),
              (NOR (CMP_UEQ_S f32:$lhs, f32:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(seto f32:$lhs, f32:$rhs),
              (NOR (CMP_UN_S f32:$lhs, f32:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(setune f32:$lhs, f32:$rhs),
              (NOR (CMP_EQ_S f32:$lhs, f32:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(seteq f32:$lhs, f32:$rhs), (CMP_EQ_S f32:$lhs, f32:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setgt f32:$lhs, f32:$rhs), (CMP_LE_S f32:$rhs, f32:$lhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setge f32:$lhs, f32:$rhs), (CMP_LT_S f32:$rhs, f32:$lhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setlt f32:$lhs, f32:$rhs), (CMP_LT_S f32:$lhs, f32:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setle f32:$lhs, f32:$rhs), (CMP_LE_S f32:$lhs, f32:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setne f32:$lhs, f32:$rhs),
              (NOR (CMP_EQ_S f32:$lhs, f32:$rhs), ZERO)>, ISA_MIPS32R6;

// f64 comparisons supported via another comparison
def : MipsPat<(setone f64:$lhs, f64:$rhs),
              (NOR (CMP_UEQ_D f64:$lhs, f64:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(seto f64:$lhs, f64:$rhs),
              (NOR (CMP_UN_D f64:$lhs, f64:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(setune f64:$lhs, f64:$rhs),
              (NOR (CMP_EQ_D f64:$lhs, f64:$rhs), ZERO)>, ISA_MIPS32R6;
def : MipsPat<(seteq f64:$lhs, f64:$rhs), (CMP_EQ_D f64:$lhs, f64:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setgt f64:$lhs, f64:$rhs), (CMP_LE_D f64:$rhs, f64:$lhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setge f64:$lhs, f64:$rhs), (CMP_LT_D f64:$rhs, f64:$lhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setlt f64:$lhs, f64:$rhs), (CMP_LT_D f64:$lhs, f64:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setle f64:$lhs, f64:$rhs), (CMP_LE_D f64:$lhs, f64:$rhs)>,
      ISA_MIPS32R6;
def : MipsPat<(setne f64:$lhs, f64:$rhs),
              (NOR (CMP_EQ_D f64:$lhs, f64:$rhs), ZERO)>, ISA_MIPS32R6;

// i32 selects
def : MipsPat<(select i32:$cond, i32:$t, i32:$f),
              (OR (SELNEZ i32:$t, i32:$cond), (SELEQZ i32:$f, i32:$cond))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), i32:$t, i32:$f),
              (OR (SELEQZ i32:$t, i32:$cond), (SELNEZ i32:$f, i32:$cond))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), i32:$t, i32:$f),
              (OR (SELNEZ i32:$t, i32:$cond), (SELEQZ i32:$f, i32:$cond))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immZExt16:$imm)), i32:$t, i32:$f),
              (OR (SELEQZ i32:$t, (XORi i32:$cond, immZExt16:$imm)),
                  (SELNEZ i32:$f, (XORi i32:$cond, immZExt16:$imm)))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (setne i32:$cond, immZExt16:$imm)), i32:$t, i32:$f),
              (OR (SELNEZ i32:$t, (XORi i32:$cond, immZExt16:$imm)),
                  (SELEQZ i32:$f, (XORi i32:$cond, immZExt16:$imm)))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (setgt i32:$cond, immSExt16Plus1:$imm)), i32:$t,
                      i32:$f),
              (OR (SELEQZ i32:$t, (SLTi i32:$cond, (Plus1 imm:$imm))),
                  (SELNEZ i32:$f, (SLTi i32:$cond, (Plus1 imm:$imm))))>,
              ISA_MIPS32R6;
def : MipsPat<(select (i32 (setugt i32:$cond, immSExt16Plus1:$imm)),
                      i32:$t, i32:$f),
              (OR (SELEQZ i32:$t, (SLTiu i32:$cond, (Plus1 imm:$imm))),
                  (SELNEZ i32:$f, (SLTiu i32:$cond, (Plus1 imm:$imm))))>,
              ISA_MIPS32R6;

def : MipsPat<(select i32:$cond, i32:$t, immz),
              (SELNEZ i32:$t, i32:$cond)>, ISA_MIPS32R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), i32:$t, immz),
              (SELNEZ i32:$t, i32:$cond)>, ISA_MIPS32R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), i32:$t, immz),
              (SELEQZ i32:$t, i32:$cond)>, ISA_MIPS32R6;
def : MipsPat<(select i32:$cond, immz, i32:$f),
              (SELEQZ i32:$f, i32:$cond)>, ISA_MIPS32R6;
def : MipsPat<(select (i32 (setne i32:$cond, immz)), immz, i32:$f),
              (SELEQZ i32:$f, i32:$cond)>, ISA_MIPS32R6;
def : MipsPat<(select (i32 (seteq i32:$cond, immz)), immz, i32:$f),
              (SELNEZ i32:$f, i32:$cond)>, ISA_MIPS32R6;
