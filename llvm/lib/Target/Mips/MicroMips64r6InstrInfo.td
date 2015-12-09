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
    SDPatternOperator Op = null_frag> : MMR6Arch<instr_asm>, MipsR6Inst {
  dag OutOperandList = (outs RO:$rt);
  dag InOperandList = (ins RO:$rs, PosOpnd:$pos, size_ext:$size);
  string AsmString = !strconcat(instr_asm, "\t$rt, $rs, $pos, $size");
  list<dag> Pattern = [(set RO:$rt, (Op RO:$rs, imm:$pos, imm:$size))];
  InstrItinClass Itinerary = II_EXT;
  Format Form = FrmR;
  string BaseOpcode = instr_asm;
}
class DEXT_MMR6_DESC : EXTBITS_DESC_BASE<"dext", GPR64Opnd, uimm6,
                                                 MipsExt>;
class DEXTM_MMR6_DESC : EXTBITS_DESC_BASE<"dextm", GPR64Opnd, uimm5,
                                                  MipsExt>;
class DEXTU_MMR6_DESC : EXTBITS_DESC_BASE<"dextu", GPR64Opnd, uimm5_plus32,
                                                  MipsExt>;

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
}
