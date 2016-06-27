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
class DADD_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dadd", 0b100010000>;
class DADDIU_MM64R6_ENC : DADDIU_FM_MMR6<"daddiu">;
class DADDU_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"daddu", 0b101010000>;
class LDPC_MMR646_ENC : PCREL18_FM_MMR6<0b110>;
class DSUB_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dsub", 0b110010000>;
class DSUBU_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dsubu", 0b111010000>;
class DMUL_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dmul", 0b000011000>;
class DMUH_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dmuh", 0b001011000>;
class DMULU_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dmulu", 0b010011000>;
class DMUHU_MM64R6_ENC : POOL32S_ARITH_FM_MMR6<"dmuhu", 0b011011000>;
class DSBH_MM64R6_ENC : POOL32S_2R_FM_MMR6<"dsbh", 0b0111101100>;
class DSHD_MM64R6_ENC : POOL32S_2R_FM_MMR6<"dshd", 0b1111101100>;
class DSLL_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsll", 0b000000000>;
class DSLL32_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsll32", 0b000001000>;
class DSLLV_MM64R6_ENC : POOL32S_3R_FM_MMR6<"dsllv", 0b000010000>;
class DSRAV_MM64R6_ENC : POOL32S_3R_FM_MMR6<"dsrav", 0b010010000>;
class DSRA_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsra", 0b010000000>;
class DSRA32_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsra32", 0b010000100>;
class DCLO_MM64R6_ENC : POOL32S_2R_FM_MMR6<"dclo", 0b0100101100>;
class DCLZ_MM64R6_ENC : POOL32S_2R_FM_MMR6<"dclz", 0b0101101100>;
class DROTR_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"drotr", 0b011000000>;
class DROTR32_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"drotr32", 0b011001000>;
class DROTRV_MM64R6_ENC : POOL32S_3R_FM_MMR6<"drotrv", 0b011010000>;
class LD_MM64R6_ENC : LD_SD_32_2R_OFFSET16_FM_MMR6<"ld", 0b110111>;
class LLD_MM64R6_ENC : POOL32C_2R_OFFSET12_FM_MMR6<"lld", 0b0111>;
class LWU_MM64R6_ENC : POOL32C_2R_OFFSET12_FM_MMR6<"lwu", 0b1110>;
class SD_MM64R6_ENC : LD_SD_32_2R_OFFSET16_FM_MMR6<"sd", 0b110110>;
class DSRL_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsrl", 0b001000000>;
class DSRL32_MM64R6_ENC : POOL32S_2RSA5B0_FM_MMR6<"dsrl32", 0b001001000>;
class DSRLV_MM64R6_ENC : POOL32S_3R_FM_MMR6<"dsrlv", 0b001010000>;

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

class DDIV_MM64R6_DESC : DIVMOD_MMR6_DESC_BASE<"ddiv", GPR64Opnd, sdiv>;
class DMOD_MM64R6_DESC : DIVMOD_MMR6_DESC_BASE<"dmod", GPR64Opnd, srem>;
class DDIVU_MM64R6_DESC : DIVMOD_MMR6_DESC_BASE<"ddivu", GPR64Opnd, udiv>;
class DMODU_MM64R6_DESC : DIVMOD_MMR6_DESC_BASE<"dmodu", GPR64Opnd, urem>;

class DCLO_MM64R6_DESC {
  dag OutOperandList = (outs GPR64Opnd:$rt);
  dag InOperandList = (ins GPR64Opnd:$rs);
  string AsmString = !strconcat("dclo", "\t$rt, $rs");
  list<dag> Pattern = [(set GPR64Opnd:$rt, (ctlz (not GPR64Opnd:$rs)))];
  InstrItinClass Itinerary = II_CLO;
  Format Form = FrmR;
  string BaseOpcode = "dclo";
}

class DCLZ_MM64R6_DESC {
  dag OutOperandList = (outs GPR64Opnd:$rt);
  dag InOperandList = (ins GPR64Opnd:$rs);
  string AsmString = !strconcat("dclz", "\t$rt, $rs");
  list<dag> Pattern = [(set GPR64Opnd:$rt, (ctlz GPR64Opnd:$rs))];
  InstrItinClass Itinerary = II_CLZ;
  Format Form = FrmR;
  string BaseOpcode = "dclz";
}

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

class DADD_MM64R6_DESC : ArithLogicR<"dadd", GPR64Opnd, 1, II_DADD>;
class DADDIU_MM64R6_DESC : ArithLogicI<"daddiu", simm16_64, GPR64Opnd,
                                       II_DADDIU, immSExt16, add>,
                           IsAsCheapAsAMove;
class DADDU_MM64R6_DESC : ArithLogicR<"daddu", GPR64Opnd, 1, II_DADDU, add>;

class DSUB_DESC_BASE<string instr_asm, RegisterOperand RO,
                     InstrItinClass Itin = NoItinerary,
                     SDPatternOperator OpNode = null_frag>
                     : MipsR6Inst {
  dag OutOperandList = (outs RO:$rd);
  dag InOperandList = (ins RO:$rs, RO:$rt);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs, $rt");
  list<dag> Pattern = [(set RO:$rd, (OpNode RO:$rs, RO:$rt))];
  InstrItinClass Itinerary = Itin;
  Format Form = FrmR;
  string BaseOpcode = instr_asm;
  let isCommutable = 0;
  let isReMaterializable = 1;
  let TwoOperandAliasConstraint = "$rd = $rs";
}
class DSUB_MM64R6_DESC : DSUB_DESC_BASE<"dsub", GPR64Opnd, II_DSUB>;
class DSUBU_MM64R6_DESC : DSUB_DESC_BASE<"dsubu", GPR64Opnd, II_DSUBU, sub>;

class LDPC_MM64R6_DESC : PCREL_MMR6_DESC_BASE<"ldpc", GPR64Opnd, simm18_lsl3>;

class MUL_MM64R6_DESC_BASE<string opstr, RegisterOperand GPROpnd,
                           InstrItinClass Itin = NoItinerary,
                           SDPatternOperator Op = null_frag> : MipsR6Inst {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs, GPROpnd:$rt);
  string AsmString = !strconcat(opstr, "\t$rd, $rs, $rt");
  InstrItinClass Itinerary = Itin;
  list<dag> Pattern = [(set GPROpnd:$rd, (Op GPROpnd:$rs, GPROpnd:$rt))];
}

class DMUL_MM64R6_DESC : MUL_MM64R6_DESC_BASE<"dmul", GPR64Opnd, II_DMUL, mul>;
class DMUH_MM64R6_DESC : MUL_MM64R6_DESC_BASE<"dmuh", GPR64Opnd, II_DMUH,
                                              mulhs>;
class DMULU_MM64R6_DESC : MUL_MM64R6_DESC_BASE<"dmulu", GPR64Opnd, II_DMULU>;
class DMUHU_MM64R6_DESC : MUL_MM64R6_DESC_BASE<"dmuhu", GPR64Opnd, II_DMUHU,
                                               mulhu>;

class DSBH_DSHD_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rt);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs");
  bit hasSideEffects = 0;
  list<dag> Pattern = [];
  InstrItinClass Itinerary = NoItinerary;
  Format Form = FrmR;
  string BaseOpcode = instr_asm;
}

class DSBH_MM64R6_DESC : DSBH_DSHD_DESC_BASE<"dsbh", GPR64Opnd>;
class DSHD_MM64R6_DESC : DSBH_DSHD_DESC_BASE<"dshd", GPR64Opnd>;

class SHIFT_ROTATE_IMM_MM64R6<string instr_asm, Operand ImmOpnd,
                              InstrItinClass itin,
                              SDPatternOperator OpNode = null_frag,
                              SDPatternOperator PO = null_frag> {
  dag OutOperandList = (outs GPR64Opnd:$rt);
  dag InOperandList = (ins GPR64Opnd:$rs, ImmOpnd:$sa);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $sa");
  list<dag> Pattern = [(set GPR64Opnd:$rt, (OpNode GPR64Opnd:$rs, PO:$sa))];
  InstrItinClass Itinerary = itin;
  Format Form = FrmR;
  string TwoOperandAliasConstraint = "$rs = $rt";
  string BaseOpcode = instr_asm;
}

class SHIFT_ROTATE_REG_MM64R6<string instr_asm, InstrItinClass itin,
                              SDPatternOperator OpNode = null_frag> {
  dag OutOperandList = (outs GPR64Opnd:$rd);
  dag InOperandList = (ins GPR64Opnd:$rt, GPR32Opnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rt, $rs");
  list<dag> Pattern = [(set GPR64Opnd:$rd,
                       (OpNode GPR64Opnd:$rt, GPR32Opnd:$rs))];
  InstrItinClass Itinerary = itin;
  Format Form = FrmR;
  string BaseOpcode = instr_asm;
}

class DSLL_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsll", uimm6, II_DSLL, shl,
                                                 immZExt6>;
class DSLL32_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsll32", uimm5, II_DSLL32>;
class DSLLV_MM64R6_DESC : SHIFT_ROTATE_REG_MM64R6<"dsllv", II_DSLLV, shl>;
class DSRAV_MM64R6_DESC : SHIFT_ROTATE_REG_MM64R6<"dsrav", II_DSRAV, sra>;
class DSRA_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsra", uimm6, II_DSRA, sra,
                                                 immZExt6>;
class DSRA32_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsra32", uimm5, II_DSRA32>;
class DROTR_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"drotr", uimm6, II_DROTR,
                                                  rotr, immZExt6>;
class DROTR32_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"drotr32", uimm5,
                                                    II_DROTR32>;
class DROTRV_MM64R6_DESC : SHIFT_ROTATE_REG_MM64R6<"drotrv", II_DROTRV, rotr>;
class DSRL_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsrl", uimm6, II_DSRL, srl,
                                                 immZExt6>;
class DSRL32_MM64R6_DESC : SHIFT_ROTATE_IMM_MM64R6<"dsrl32", uimm5, II_DSRL32>;
class DSRLV_MM64R6_DESC : SHIFT_ROTATE_REG_MM64R6<"dsrlv", II_DSRLV, srl>;

class Load_MM64R6<string instr_asm, Operand MemOpnd, InstrItinClass itin,
                  SDPatternOperator OpNode = null_frag> {
  dag OutOperandList = (outs GPR64Opnd:$rt);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$rt, $addr");
  list<dag> Pattern = [(set GPR64Opnd:$rt, (OpNode addr:$addr))];
  InstrItinClass Itinerary = itin;
  Format Form = FrmI;
  bit mayLoad = 1;
  bit canFoldAsLoad = 1;
  string BaseOpcode = instr_asm;
}

class LD_MM64R6_DESC : Load_MM64R6<"ld", mem_simm16, II_LD, load> {
  string DecoderMethod = "DecodeMemMMImm16";
}
class LWU_MM64R6_DESC : Load_MM64R6<"lwu", mem_simm12, II_LWU, zextloadi32>{
  string DecoderMethod = "DecodeMemMMImm12";
}

class LLD_MM64R6_DESC {
  dag OutOperandList = (outs GPR64Opnd:$rt);
  dag InOperandList = (ins mem_simm12:$addr);
  string AsmString = "lld\t$rt, $addr";
  list<dag> Pattern = [];
  bit mayLoad = 1;
  InstrItinClass Itinerary = II_LLD;
  string BaseOpcode = "lld";
  string DecoderMethod = "DecodeMemMMImm12";
}

class SD_MM64R6_DESC {
  dag OutOperandList = (outs);
  dag InOperandList = (ins GPR64Opnd:$rt, mem_simm16:$addr);
  string AsmString = "sd\t$rt, $addr";
  list<dag> Pattern = [(store GPR64Opnd:$rt, addr:$addr)];
  InstrItinClass Itinerary = II_SD;
  Format Form = FrmI;
  bit mayStore = 1;
  string BaseOpcode = "sd";
  string DecoderMethod = "DecodeMemMMImm16";
}

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
  def DADD_MM64R6: StdMMR6Rel, DADD_MM64R6_DESC, DADD_MM64R6_ENC,
                   ISA_MICROMIPS64R6;
  def DADDIU_MM64R6: StdMMR6Rel, DADDIU_MM64R6_DESC, DADDIU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DADDU_MM64R6: StdMMR6Rel, DADDU_MM64R6_DESC, DADDU_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def LDPC_MM64R6 :  R6MMR6Rel, LDPC_MMR646_ENC, LDPC_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DSUB_MM64R6 : StdMMR6Rel, DSUB_MM64R6_DESC, DSUB_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DSUBU_MM64R6 : StdMMR6Rel, DSUBU_MM64R6_DESC, DSUBU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DMUL_MM64R6 : R6MMR6Rel, DMUL_MM64R6_DESC, DMUL_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DMUH_MM64R6 : R6MMR6Rel, DMUH_MM64R6_DESC, DMUH_MM64R6_ENC,
                    ISA_MICROMIPS64R6;
  def DMULU_MM64R6 : R6MMR6Rel, DMULU_MM64R6_DESC, DMULU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DMUHU_MM64R6 : R6MMR6Rel, DMUHU_MM64R6_DESC, DMUHU_MM64R6_ENC,
                     ISA_MICROMIPS64R6;
  def DSBH_MM64R6 : R6MMR6Rel, DSBH_MM64R6_ENC, DSBH_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSHD_MM64R6 : R6MMR6Rel, DSHD_MM64R6_ENC, DSHD_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSLL_MM64R6 : StdMMR6Rel, DSLL_MM64R6_ENC, DSLL_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSLL32_MM64R6 : StdMMR6Rel, DSLL32_MM64R6_ENC, DSLL32_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSLLV_MM64R6 : StdMMR6Rel, DSLLV_MM64R6_ENC, DSLLV_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DSRAV_MM64R6 : StdMMR6Rel, DSRAV_MM64R6_ENC, DSRAV_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSRA_MM64R6 : StdMMR6Rel, DSRA_MM64R6_ENC, DSRA_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSRA32_MM64R6 : StdMMR6Rel, DSRA32_MM64R6_ENC, DSRA32_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DCLO_MM64R6 : StdMMR6Rel, R6MMR6Rel, DCLO_MM64R6_ENC, DCLO_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DCLZ_MM64R6 : StdMMR6Rel, R6MMR6Rel, DCLZ_MM64R6_ENC, DCLZ_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DROTR_MM64R6 : StdMMR6Rel, DROTR_MM64R6_ENC, DROTR_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
  def DROTR32_MM64R6 : StdMMR6Rel, DROTR32_MM64R6_ENC, DROTR32_MM64R6_DESC,
                       ISA_MICROMIPS64R6;
  def DROTRV_MM64R6 : StdMMR6Rel, DROTRV_MM64R6_ENC, DROTRV_MM64R6_DESC,
                      ISA_MICROMIPS64R6;
  def LD_MM64R6 : StdMMR6Rel, LD_MM64R6_ENC, LD_MM64R6_DESC,
                  ISA_MICROMIPS64R6;
  def LLD_MM64R6 : StdMMR6Rel, R6MMR6Rel, LLD_MM64R6_ENC, LLD_MM64R6_DESC,
                   ISA_MICROMIPS64R6;
  def LWU_MM64R6 : StdMMR6Rel, LWU_MM64R6_ENC, LWU_MM64R6_DESC,
                   ISA_MICROMIPS64R6;
  def SD_MM64R6 : StdMMR6Rel, SD_MM64R6_ENC, SD_MM64R6_DESC,
                  ISA_MICROMIPS64R6;
  def DSRL_MM64R6 : StdMMR6Rel, DSRL_MM64R6_ENC, DSRL_MM64R6_DESC,
                    ISA_MICROMIPS64R6;
  def DSRL32_MM64R6 : StdMMR6Rel, DSRL32_MM64R6_ENC, DSRL32_MM64R6_DESC,
                      ISA_MICROMIPS64R6;
  def DSRLV_MM64R6 : StdMMR6Rel, DSRLV_MM64R6_ENC, DSRLV_MM64R6_DESC,
                     ISA_MICROMIPS64R6;
}

//===----------------------------------------------------------------------===//
//
// Arbitrary patterns that map to one or more instructions
//
//===----------------------------------------------------------------------===//

def : MipsPat<(MipsLo tglobaladdr:$in),
              (DADDIU_MM64R6 ZERO_64, tglobaladdr:$in)>, ISA_MICROMIPS64R6;
def : MipsPat<(MipsLo tblockaddress:$in),
              (DADDIU_MM64R6 ZERO_64, tblockaddress:$in)>, ISA_MICROMIPS64R6;
def : MipsPat<(MipsLo tjumptable:$in),
              (DADDIU_MM64R6 ZERO_64, tjumptable:$in)>, ISA_MICROMIPS64R6;
def : MipsPat<(MipsLo tconstpool:$in),
              (DADDIU_MM64R6 ZERO_64, tconstpool:$in)>, ISA_MICROMIPS64R6;
def : MipsPat<(MipsLo tglobaltlsaddr:$in),
              (DADDIU_MM64R6 ZERO_64, tglobaltlsaddr:$in)>, ISA_MICROMIPS64R6;
def : MipsPat<(MipsLo texternalsym:$in),
              (DADDIU_MM64R6 ZERO_64, texternalsym:$in)>, ISA_MICROMIPS64R6;

def : MipsPat<(add GPR64:$hi, (MipsLo tglobaladdr:$lo)),
              (DADDIU_MM64R6 GPR64:$hi, tglobaladdr:$lo)>, ISA_MICROMIPS64R6;
def : MipsPat<(add GPR64:$hi, (MipsLo tblockaddress:$lo)),
              (DADDIU_MM64R6 GPR64:$hi, tblockaddress:$lo)>, ISA_MICROMIPS64R6;
def : MipsPat<(add GPR64:$hi, (MipsLo tjumptable:$lo)),
              (DADDIU_MM64R6 GPR64:$hi, tjumptable:$lo)>, ISA_MICROMIPS64R6;
def : MipsPat<(add GPR64:$hi, (MipsLo tconstpool:$lo)),
              (DADDIU_MM64R6 GPR64:$hi, tconstpool:$lo)>, ISA_MICROMIPS64R6;
def : MipsPat<(add GPR64:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (DADDIU_MM64R6 GPR64:$hi, tglobaltlsaddr:$lo)>, ISA_MICROMIPS64R6;

def : MipsPat<(addc GPR64:$lhs, GPR64:$rhs),
              (DADDU_MM64R6 GPR64:$lhs, GPR64:$rhs)>, ISA_MICROMIPS64R6;
def : MipsPat<(addc GPR64:$lhs, immSExt16:$imm),
              (DADDIU_MM64R6 GPR64:$lhs, imm:$imm)>, ISA_MICROMIPS64R6;


def : MipsPat<(rotr GPR64:$rt, (i32 (trunc GPR64:$rs))),
              (DROTRV_MM64R6 GPR64:$rt, (EXTRACT_SUBREG GPR64:$rs, sub_32))>,
              ISA_MICROMIPS64R6;


def : WrapperPat<tglobaladdr, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;
def : WrapperPat<tconstpool, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;
def : WrapperPat<texternalsym, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;
def : WrapperPat<tblockaddress, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;
def : WrapperPat<tjumptable, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;
def : WrapperPat<tglobaltlsaddr, DADDIU_MM64R6, GPR64>, ISA_MICROMIPS64R6;

// Carry pattern
def : MipsPat<(subc GPR64:$lhs, GPR64:$rhs),
              (DSUBU_MM64R6 GPR64:$lhs, GPR64:$rhs)>, ISA_MICROMIPS64R6;

def : MipsPat<(atomic_load_64 addr:$a), (LD_MM64R6 addr:$a)>, ISA_MICROMIPS64R6;

//===----------------------------------------------------------------------===//
//
// Instruction aliases
//
//===----------------------------------------------------------------------===//

def : MipsInstAlias<"dmtc0 $rt, $rd",
                    (DMTC0_MM64R6 COP0Opnd:$rd, GPR64Opnd:$rt, 0), 0>;
def : MipsInstAlias<"dmfc0 $rt, $rd",
                    (DMFC0_MM64R6 GPR64Opnd:$rt, COP0Opnd:$rd, 0), 0>,
                    ISA_MICROMIPS64R6;
def : MipsInstAlias<"daddu $rs, $rt, $imm",
                    (DADDIU_MM64R6 GPR64Opnd:$rs,
                                   GPR64Opnd:$rt,
                                   simm16_64:$imm),
                    0>, ISA_MICROMIPS64R6;
def : MipsInstAlias<"daddu $rs, $imm",
                    (DADDIU_MM64R6 GPR64Opnd:$rs,
                                   GPR64Opnd:$rs,
                                   simm16_64:$imm),
                    0>, ISA_MICROMIPS64R6;
def : MipsInstAlias<"dsubu $rt, $rs, $imm",
                    (DADDIU_MM64R6 GPR64Opnd:$rt,
                                   GPR64Opnd:$rs,
                                   InvertedImOperand64:$imm),
                    0>, ISA_MICROMIPS64R6;
def : MipsInstAlias<"dsubu $rs, $imm",
                    (DADDIU_MM64R6 GPR64Opnd:$rs,
                                   GPR64Opnd:$rs,
                                   InvertedImOperand64:$imm),
                    0>, ISA_MICROMIPS64R6;
def : MipsInstAlias<"dneg $rt, $rs",
                    (DSUB_MM64R6 GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rs), 1>,
                    ISA_MICROMIPS64R6;
def : MipsInstAlias<"dneg $rt",
                    (DSUB_MM64R6 GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rt), 0>,
                    ISA_MICROMIPS64R6;
def : MipsInstAlias<"dnegu $rt, $rs",
                    (DSUBU_MM64R6 GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rs), 1>,
                    ISA_MICROMIPS64R6;
def : MipsInstAlias<"dnegu $rt",
                    (DSUBU_MM64R6 GPR64Opnd:$rt, ZERO_64, GPR64Opnd:$rt), 0>,
                    ISA_MICROMIPS64R6;
