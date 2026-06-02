//===-- EZHInstrInfo.td - EZH Instruction defs ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "EZHInstrFormats.td"

//===----------------------------------------------------------------------===//
// Instruction definitions.
//===----------------------------------------------------------------------===//

def calltarget : Operand<i32>, ImmLeaf<i32, [{ return (Imm & 3) == 0; }]> {
  let PrintMethod = "printBranchTarget";
  let EncoderMethod = "getCallTargetOpValue";
  let DecoderMethod = "decodeSimm30";
}

def brtarget : Operand<OtherVT> {
  let PrintMethod = "printBranchTarget";
  let EncoderMethod = "getBranchTargetOpValue";
  let DecoderMethod = "decodeSimm21";
}

def imm20 : Operand<i32>, ImmLeaf<i32, [{ return isUInt<20>(Imm) && (Imm & 3) == 0; }]> {
  let PrintMethod = "printImmOperand";
}


def shifted_imms11 : Operand<i32>, ImmLeaf<i32, [{ return isInt<11>(Imm); }]> {
  let PrintMethod = "printShiftedImmOperand";
  let DecoderMethod = "decodeSimm11";
}

def imms11 : Operand<i32>, ImmLeaf<i32, [{ return isInt<11>(Imm); }]> {
  let PrintMethod = "printImmOperand";
  let DecoderMethod = "decodeSimm11";
}

def imms12_logical : Operand<i32>, ImmLeaf<i32, [{ return isInt<12>(Imm); }]> {
  let PrintMethod = "printImmOperand";
  let DecoderMethod = "decodeSimm12";
}

def imms12 : Operand<i32>, ImmLeaf<i32, [{ return isInt<12>(Imm); }]> {
  let PrintMethod = "printImmOperand";
  let DecoderMethod = "decodeSimm12";
}


def imms8_word : PatLeaf<(imm), [{
  return isInt<8>(N->getSExtValue() >> 2) && ((N->getZExtValue() & 3) == 0);
}]>;


def imms8_word_offset : Operand<i32>, ImmLeaf<i32, [{ return isInt<8>(Imm >> 2) && ((Imm & 3) == 0); }]> {
  let PrintMethod = "printWordOffset";
  let EncoderMethod = "getWordOffsetOpValue";
  let DecoderMethod = "decodeWordOffset";
}

def imms8 : Operand<i32>, ImmLeaf<i32, [{ return isInt<8>(Imm); }]> {
  let PrintMethod = "printImmOperand";
  let DecoderMethod = "decodeSimm8";
}

def imm5 : Operand<i32>, ImmLeaf<i32, [{ return isInt<5>(Imm) || isUInt<5>(Imm); }]> {
  let PrintMethod = "printImmOperand";
}

def imm16 : Operand<i32>, ImmLeaf<i32, [{ return isUInt<16>(Imm); }]> {
  let PrintMethod = "printImmOperand";
  let DecoderMethod = "decodeSimm16";
}

def imm24 : Operand<i32>, ImmLeaf<i32, [{ return isUInt<24>(Imm); }]> {
  let PrintMethod = "printImmOperand";
}

def imm1 : Operand<i32>, ImmLeaf<i32, [{ return isInt<1>(Imm) || isUInt<1>(Imm); }]> {
  let PrintMethod = "printImmOperand";
}

def SDT_EZHCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                           SDTCisVT<1, i32>]>;
def SDT_EZHCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                         SDTCisVT<1, i32>]>;
def SDT_EZHCall         : SDTypeProfile<0, -1, [SDTCisVT<0, i32>]>;

def EZH_call            : SDNode<"EZHISD::CALL", SDT_EZHCall,
                                  [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue, SDNPVariadic]>;
def EZH_adjcallstackdown : SDNode<"ISD::CALLSEQ_START", SDT_EZHCallSeqStart,
                                  [SDNPHasChain, SDNPOutGlue]>;
def EZH_adjcallstackup   : SDNode<"ISD::CALLSEQ_END",   SDT_EZHCallSeqEnd,
                                  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def EZH_ret_glue_node   : SDNode<"EZHISD::RET_GLUE_INTERNAL", SDTNone,
                                  [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// Condition Code Classes
class EZHCondCode<bits<4> val, string suffix> {
  bits<4> Value = val;
  string Suffix = suffix;
}

def Cond_eu : EZHCondCode<0x0, "">;     // Always / True (EU flag)
def Cond_ze : EZHCondCode<0x1, "_ze">;  // Zero (ZE flag)
def Cond_nz : EZHCondCode<0x2, "_nz">;  // Not Zero (NZ flag)
def Cond_po : EZHCondCode<0x3, "_po">;  // Positive (PO flag)
def Cond_ne : EZHCondCode<0x4, "_ne">;  // Negative (NE flag)
def Cond_az : EZHCondCode<0x5, "_az">;  // Above Zero (AZ flag)
def Cond_zb : EZHCondCode<0x6, "_zb">;  // Zero or Below (ZB flag)
def Cond_ca : EZHCondCode<0x7, "_ca">;  // Carry (CA flag)
def Cond_nc : EZHCondCode<0x8, "_nc">;  // No Carry (NC flag)
def Cond_cz : EZHCondCode<0x9, "_cz">;  // Carry or Zero (CZ flag)
def Cond_spo: EZHCondCode<0xA, "_spo">; // Shift-Only-when-Positive / Unconditionally Scheduled branch delay slot (SPO/UNS)
def Cond_sne: EZHCondCode<0xB, "_sne">; // Shift-Only-when-Negative / Scheduled branch delay slot when Not Zero (SNE/NZS)
def Cond_nbs: EZHCondCode<0xC, "_nbs">; // Not Boolean-expression set (NBS flag)
def Cond_nex: EZHCondCode<0xD, "_nex">; // External flag is not set (NEX flag)
def Cond_bs : EZHCondCode<0xE, "_bs">;  // Boolean-expression set (BS flag)
def Cond_ex : EZHCondCode<0xF, "_ex">;  // External flag is set (EX flag)

defvar CondCodes = [Cond_eu, Cond_ze, Cond_nz, Cond_po, Cond_ne, Cond_az, Cond_zb, Cond_ca, Cond_nc, Cond_cz, Cond_spo, Cond_sne, Cond_nbs, Cond_nex, Cond_bs, Cond_ex];





class SetFlag<bits<1> val, string suffix> {
  bits<1> Value = val;
  string Suffix = suffix;
}

def Flag_none : SetFlag<0, "">;
def Flag_s : SetFlag<1, "s">;

defvar SetFlags = [Flag_none, Flag_s];

multiclass ALUShiftedOps<bits<5> op, string opc_str, string sf_suffix, string cc_suffix, SDPatternOperator op_node = null_frag> {
  // Normal
  def _LSL_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_lsl"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (shl (op_node i32:$Rs1, i32:$Rs2), imm5:$ShAmt))], [])>;
  def _LSR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 2, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_lsr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (srl (op_node i32:$Rs1, i32:$Rs2), imm5:$ShAmt))], [])>;
  def _ASR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 1, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_asr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (sra (op_node i32:$Rs1, i32:$Rs2), imm5:$ShAmt))], [])>;
  def _ROR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 3, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_ror"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (rotr (op_node i32:$Rs1, i32:$Rs2), imm5:$ShAmt))], [])>;
  def _FLSL_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 4, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_flsl"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def _FLSR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 6, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_flsr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def _FASR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 5, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_fasr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def _FROR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 7, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"_fror"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def _F_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 4, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2), "e_"#opc_str#"_f"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2", []> { let ShAmt = 0; }

  // Inverted
  def N_LSL_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 0, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_lsl"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (shl (not (op_node i32:$Rs1, i32:$Rs2)), imm5:$ShAmt))], [])>;
  def N_LSR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 2, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_lsr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (srl (not (op_node i32:$Rs1, i32:$Rs2)), imm5:$ShAmt))], [])>;
  def N_ASR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 1, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_asr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (sra (not (op_node i32:$Rs1, i32:$Rs2)), imm5:$ShAmt))], [])>;
  def N_ROR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 3, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_ror"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt",
      !if(!and(!empty(cc_suffix), !empty(sf_suffix)), [(set i32:$Rd, (rotr (not (op_node i32:$Rs1, i32:$Rs2)), imm5:$ShAmt))], [])>;
  def N_FLSL_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 4, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_flsl"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def N_FLSR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 6, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_flsr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def N_FASR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 5, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_fasr"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def N_FROR_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 7, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_"#opc_str#"n_fror"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def N_F_#sf_suffix#_#cc_suffix : EZHInstALUShifted<op, 4, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2), "e_"#opc_str#"n_f"#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2", []> { let ShAmt = 0; }
}




multiclass FlipALUOps<bits<3> op, string opc_str, string sf_suffix, string cc_suffix> {
  def FBIT_LSR_#opc_str#_#sf_suffix#_#cc_suffix : EZHInstFlipALU<op, 0, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_fbit_lsr_"#opc_str#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def FBIT_ASR_#opc_str#_#sf_suffix#_#cc_suffix : EZHInstFlipALU<op, 1, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_fbit_asr_"#opc_str#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def FEND_LSR_#opc_str#_#sf_suffix#_#cc_suffix : EZHInstFlipALU<op, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_fend_lsr_"#opc_str#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
  def FEND_ASR_#opc_str#_#sf_suffix#_#cc_suffix : EZHInstFlipALU<op, 1, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), "e_fend_asr_"#opc_str#sf_suffix#cc_suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []>;
}

foreach cc = CondCodes in {

  defvar is_predicated = !not(!empty(cc.Suffix));
  defvar predicated_constraints = !if(is_predicated, "$Rd = $src_in", "");

  defvar ins_GPRAll = !con((ins GPRAll:$Rs1), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_2 = !con((ins GPRAll:$Rs1, GPRAll:$Rs2), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_imm = !con((ins imms11:$Imm), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_simm = !con((ins shifted_imms11:$Imm, imm5:$ShAmt), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_imm12 = !con((ins GPRAll:$Rs1, imms12:$Imm), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_logical_imm12 = !con((ins GPRAll:$Rs1, imms12_logical:$Imm), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRSP_imm12 = !con((ins GPRSP:$Rs1, imms12:$Imm), !if(is_predicated, (ins GPRSP:$src_in), (ins)));

  defvar ins_GPRAll_shift = !con((ins GPRAll:$Rs, imm5:$ShAmt), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_flip = !con((ins GPRAll:$Rs2, imm5:$ShAmt), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_shift_alu = !con((ins GPRAll:$Rs1, GPRAll:$Rs2, imm5:$ShAmt), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_reg_shift = !con((ins GPRAll:$Rs2, GPRAll:$Rshift), !if(is_predicated, (ins GPRAll:$src_in), (ins)));
  defvar ins_GPRAll_reg_shift_alu = !con((ins GPRAll:$Rs1, GPRAll:$Rs2, GPRAll:$Rshift), !if(is_predicated, (ins GPRAll:$src_in), (ins)));

  foreach sf = SetFlags in {
  let Cond = cc.Value, S = sf.Value in {

// MOV
def MOVrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x00, 0, (outs GPRAll:$Rd), ins_GPRAll,
                       "e_mov"#sf.Suffix#cc.Suffix#" $Rd, $Rs1", []> { 
                         let Rs2 = 0; 
                         let Constraints = predicated_constraints;
                       }
def MVNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x00, 0, (outs GPRAll:$Rd), ins_GPRAll,
                       "e_mvn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1", !if(!empty(cc.Suffix), [(set i32:$Rd, (not i32:$Rs1))], [])> { 
                         let Rs2 = 0; 
                         let Inst{31} = 1; 
                         let Constraints = predicated_constraints;
                       }
def MOVri_#sf.Suffix#_#cc.Suffix : EZHInstALUI<0x00, 0, (outs GPRAll:$Rd), ins_GPRAll_imm,
                       "e_load_imm"#sf.Suffix#cc.Suffix#" $Rd, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (i32 imm:$Imm))], [])> { 
                         let Rs1 = 0; 
                         let Constraints = predicated_constraints;
                       }
let AsmMatchConverter = "cvtShiftedImm" in {
def MOVSri_#sf.Suffix#_#cc.Suffix : EZHInstALUIShifted<0x00, 0, (outs GPRAll:$Rd), ins_GPRAll_simm,
                       "e_load_simm"#sf.Suffix#cc.Suffix#" $Rd, $Imm", []> {
                         let Constraints = predicated_constraints;
                       }
}
def MOVNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI<0x00, 0, (outs GPRAll:$Rd), ins_GPRAll_imm,
                       "e_load_immn"#sf.Suffix#cc.Suffix#" $Rd, $Imm", []> { 
                         let Rs1 = 0; 
                         let Inst{31} = 1; 
                         let Constraints = predicated_constraints;
                       }
let AsmMatchConverter = "cvtShiftedImm" in {
def MOVNSri_#sf.Suffix#_#cc.Suffix : EZHInstALUIShifted<0x00, 1, (outs GPRAll:$Rd), ins_GPRAll_simm,
                       "e_load_simmn"#sf.Suffix#cc.Suffix#" $Rd, $Imm", []> {
                         let Constraints = predicated_constraints;
                       }
}

// ADD
def ADDrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x06, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ADDNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x06, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_addn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (add i32:$Rs1, i32:$Rs2)))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ADDri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x06, 0, (outs GPRSP:$Rd), ins_GPRSP_imm12,
                         "e_add_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def ADDNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x06, 1, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_addn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (add i32:$Rs1, (i32 imm:$Imm))))], [])> {
                           let Constraints = predicated_constraints;
                         }

// SUB
def SUBrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x08, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def SUBNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x08, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_subn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (sub i32:$Rs1, i32:$Rs2)))], [])> {
                         let Constraints = predicated_constraints;
                       }
def SUBri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x08, 0, (outs GPRSP:$Rd), ins_GPRSP_imm12,
                         "e_sub_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def SUBNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x08, 1, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_subn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (sub i32:$Rs1, (i32 imm:$Imm))))], [])> {
                           let Constraints = predicated_constraints;
                         }

// ADC
def ADCrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x09, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (adde i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ADCNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x09, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_adcn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", []> {
                         let Constraints = predicated_constraints;
                       }
def ADCri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x09, 0, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_adc_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (adde i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def ADCNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x09, 1, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_adcn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", []> {
                           let Constraints = predicated_constraints;
                         }

// SBC
def SBCrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0A, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (sube i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def SBCNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0A, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_sbcn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", []> {
                         let Constraints = predicated_constraints;
                       }
def SBCri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0A, 0, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_sbc_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (sube i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def SBCNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0A, 1, (outs GPRAll:$Rd), ins_GPRAll_imm12,
                         "e_sbcn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", []> {
                           let Constraints = predicated_constraints;
                         }

// OR
def ORrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0C, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ORNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0C, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_orn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (or i32:$Rs1, i32:$Rs2)))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ORri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0C, 0, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_or_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def ORNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0C, 1, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_orn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (or i32:$Rs1, (i32 imm:$Imm))))], [])> {
                           let Constraints = predicated_constraints;
                         }

// AND
def ANDrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0D, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ANDNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0D, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_andn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (and i32:$Rs1, i32:$Rs2)))], [])> {
                         let Constraints = predicated_constraints;
                       }
def ANDri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0D, 0, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_and_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def ANDNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0D, 1, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_andn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (and i32:$Rs1, (i32 imm:$Imm))))], [])> {
                           let Constraints = predicated_constraints;
                         }

// XOR
def XORrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0E, 0, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, i32:$Rs2))], [])> {
                         let Constraints = predicated_constraints;
                       }
def XORNrr_#sf.Suffix#_#cc.Suffix : EZHInstALU<0x0E, 1, (outs GPRAll:$Rd), ins_GPRAll_2,
                       "e_xorn"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (xor i32:$Rs1, i32:$Rs2)))], [])> {
                         let Constraints = predicated_constraints;
                       }
def XORri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0E, 0, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_xor_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (i32 imm:$Imm)))], [])> {
                           let Constraints = predicated_constraints;
                         }
def XORNri_#sf.Suffix#_#cc.Suffix : EZHInstALUI12<0x0E, 1, (outs GPRAll:$Rd), ins_GPRAll_logical_imm12,
                         "e_xorn_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Imm", !if(!empty(cc.Suffix), [(set i32:$Rd, (not (xor i32:$Rs1, (i32 imm:$Imm))))], [])> {
                           let Constraints = predicated_constraints;
                         }


// Shifted immediates
defm ADD : ALUShiftedOps<0x06, "add", sf.Suffix, cc.Suffix, add>;
defm SUB : ALUShiftedOps<0x08, "sub", sf.Suffix, cc.Suffix, sub>;
defm ADC : ALUShiftedOps<0x09, "adc", sf.Suffix, cc.Suffix>;
defm SBC : ALUShiftedOps<0x0A, "sbc", sf.Suffix, cc.Suffix>;
defm OR  : ALUShiftedOps<0x0C, "or",  sf.Suffix, cc.Suffix, or>;
defm AND : ALUShiftedOps<0x0D, "and", sf.Suffix, cc.Suffix, and>;
defm XOR : ALUShiftedOps<0x0E, "xor", sf.Suffix, cc.Suffix, xor>;

def FBIT_LSR_#sf.Suffix#_#cc.Suffix : EZHInstFlipALU<0, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_flip, "e_fbit_lsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $ShAmt", []> { 
  let Rs1 = 0; 
  let Constraints = predicated_constraints;
}
def FEND_LSR_#sf.Suffix#_#cc.Suffix : EZHInstFlipALU<0, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_flip, "e_fend_lsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $ShAmt", []> { 
  let Rs1 = 0; 
  let Constraints = predicated_constraints;
}
def FBIT_ASR_#sf.Suffix#_#cc.Suffix : EZHInstFlipALU<0, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_flip, "e_fbit_asr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $ShAmt", []> { 
  let Rs1 = 0; 
  let Constraints = predicated_constraints;
}
def FEND_ASR_#sf.Suffix#_#cc.Suffix : EZHInstFlipALU<0, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_flip, "e_fend_asr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $ShAmt", []> { 
  let Rs1 = 0; 
  let Constraints = predicated_constraints;
}

defm FBIT_AND : FlipALUOps<1, "and", sf.Suffix, cc.Suffix>;
defm FBIT_OR  : FlipALUOps<2, "or",  sf.Suffix, cc.Suffix>;
defm FBIT_XOR : FlipALUOps<3, "xor", sf.Suffix, cc.Suffix>;
defm FBIT_ADD : FlipALUOps<4, "add", sf.Suffix, cc.Suffix>;
defm FBIT_SUB : FlipALUOps<5, "sub", sf.Suffix, cc.Suffix>;
defm FBIT_ADC : FlipALUOps<6, "adc", sf.Suffix, cc.Suffix>;
defm FBIT_SBC : FlipALUOps<7, "sbc", sf.Suffix, cc.Suffix>;

def ANDOR_#sf.Suffix#_#cc.Suffix : EZHInstAndOr<(outs GPRAll:$Rd), !con((ins GPRAll:$Rs1, GPRAll:$Rs2, GPRAll:$Rs3), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_andor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rs3", []> {
  let Constraints = predicated_constraints;
}

} // let Cond, S
} // foreach sf

let Cond = cc.Value, S = 0 in {

// Shifts
foreach sf = SetFlags in {
let S = sf.Value in {
def LSLi_#sf.Suffix#_#cc.Suffix : EZHInstShift<0, (outs GPRAll:$Rd), ins_GPRAll_shift,
                        "e_lsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (shl i32:$Rs, (i32 imm:$ShAmt)))], [])> {
                          let Constraints = predicated_constraints;
                        }
def LSRi_#sf.Suffix#_#cc.Suffix : EZHInstShift<2, (outs GPRAll:$Rd), ins_GPRAll_shift,
                        "e_lsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (srl i32:$Rs, (i32 imm:$ShAmt)))], [])> {
                          let Constraints = predicated_constraints;
                        }
def ASRi_#sf.Suffix#_#cc.Suffix : EZHInstShift<1, (outs GPRAll:$Rd), ins_GPRAll_shift,
                        "e_asr"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (sra i32:$Rs, (i32 imm:$ShAmt)))], [])> {
                          let Constraints = predicated_constraints;
                        }
def RORi_#sf.Suffix#_#cc.Suffix : EZHInstShift<3, (outs GPRAll:$Rd), ins_GPRAll_shift,
                        "e_ror"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (rotr i32:$Rs, (i32 imm:$ShAmt)))], [])> {
                          let Constraints = predicated_constraints;
                        }

// ShiftALU operations
def LSL_AND_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<1, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (shl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def LSR_AND_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<1, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (srl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ASR_AND_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<1, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (sra i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ROR_AND_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<1, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (rotr i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }

def LSL_OR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<2, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (shl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def LSR_OR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<2, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (srl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ASR_OR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<2, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (sra i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ROR_OR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<2, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (rotr i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }

def LSL_XOR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<3, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (shl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def LSR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<3, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (srl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ASR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<3, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (sra i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ROR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<3, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (rotr i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }

def LSL_ADD_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<4, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (shl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def LSR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<4, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (srl i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ASR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<4, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (sra i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }
def ROR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<4, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (rotr i32:$Rs2, (i32 imm:$ShAmt))))], [])> {
      let Constraints = predicated_constraints;
    }

def LSL_SUB_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<5, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (shl i32:$Rs2, (i32 imm:$ShAmt)), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def LSR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<5, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (srl i32:$Rs2, (i32 imm:$ShAmt)), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def ASR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<5, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (sra i32:$Rs2, (i32 imm:$ShAmt)), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def ROR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<5, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (rotr i32:$Rs2, (i32 imm:$ShAmt)), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }

def LSL_ADC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<6, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def LSR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<6, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def ASR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<6, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def ROR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<6, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }

def LSL_SBC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<7, 0, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsl_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def LSR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<7, 1, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_lsr_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def ASR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<7, 2, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_asr_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }
def ROR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstShiftALU<7, 3, (outs GPRAll:$Rd), ins_GPRAll_shift_alu,
    "e_ror_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $ShAmt", []> {
      let Constraints = predicated_constraints;
    }

} // let S
} // foreach sf

// Load/Store
def LDR#cc.Suffix : EZHInstMem<0x01, 0, 1, 0, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_ldr"#cc.Suffix#" $Rd, $Rn, $Offset", !if(!empty(cc.Suffix), [(set i32:$Rd, (load (add i32:$Rn, imms8_word:$Offset)))], [])> {}
def LDR_PRE#cc.Suffix : EZHInstMem<0x01, 0, 1, 0, 0, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_ldr_pre"#cc.Suffix#" $Rd, $Rn, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }
def LDR_POST#cc.Suffix : EZHInstMem<0x01, 0, 1, 0, 1, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_ldr_post"#cc.Suffix#" $Rn, $Rd, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }

def STR#cc.Suffix : EZHInstMem<0x02, 1, 1, 0, 0, 0, (outs), (ins GPRAll:$Rd, GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_str"#cc.Suffix#" $Rn, $Rd, $Offset", !if(!empty(cc.Suffix), [(store i32:$Rd, (add i32:$Rn, imms8_word:$Offset))], [])> {}
def STR_PRE#cc.Suffix : EZHInstMem<0x02, 1, 1, 0, 0, 1, (outs GPRAll:$Rn_wb), (ins GPRAll:$Rd, GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_str_pre"#cc.Suffix#" $Rn, $Rd, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }
def STR_POST#cc.Suffix : EZHInstMem<0x02, 1, 1, 0, 1, 1, (outs GPRAll:$Rn_wb), (ins GPRAll:$Rd, GPRAll:$Rn, imms8_word_offset:$Offset),
                     "e_str_post"#cc.Suffix#" $Rn, $Rd, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }

def LDRB#cc.Suffix : EZHInstMem<0x01, 0, 0, 0, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rn, imms8:$Offset),
                      "e_ldrb"#cc.Suffix#" $Rd, $Rn, $Offset", !if(!empty(cc.Suffix), [(set i32:$Rd, (zextloadi8 (add i32:$Rn, imms8:$Offset)))], [])> {}
def LDRB_PRE#cc.Suffix : EZHInstMem<0x01, 0, 0, 0, 0, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8:$Offset),
                      "e_ldrb_pre"#cc.Suffix#" $Rd, $Rn, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }
def LDRB_POST#cc.Suffix : EZHInstMem<0x01, 0, 0, 0, 1, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8:$Offset),
                      "e_ldrb_post"#cc.Suffix#" $Rd, $Rn, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }

def LDRBS#cc.Suffix : EZHInstMem<0x01, 0, 0, 1, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rn, imms8:$Offset),
                       "e_ldrbs"#cc.Suffix#" $Rd, $Rn, $Offset", !if(!empty(cc.Suffix), [(set i32:$Rd, (sextloadi8 (add i32:$Rn, imms8:$Offset)))], [])> {}
def LDRBS_PRE#cc.Suffix : EZHInstMem<0x01, 0, 0, 1, 0, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8:$Offset),
                       "e_ldrbs_pre"#cc.Suffix#" $Rd, $Rn, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }
def LDRBS_POST#cc.Suffix : EZHInstMem<0x01, 0, 0, 1, 1, 1, (outs GPRAll:$Rd, GPRAll:$Rn_wb), (ins GPRAll:$Rn, imms8:$Offset),
                       "e_ldrbs_post"#cc.Suffix#" $Rd, $Rn, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }

def STRB#cc.Suffix : EZHInstMem<0x02, 1, 0, 0, 0, 0, (outs), (ins GPRAll:$Rd, GPRAll:$Rn, imms8:$Offset),
                      "e_strb"#cc.Suffix#" $Rn, $Rd, $Offset", !if(!empty(cc.Suffix), [(truncstorei8 i32:$Rd, (add i32:$Rn, imms8:$Offset))], [])> {}
def STRB_PRE#cc.Suffix : EZHInstMem<0x02, 1, 0, 0, 0, 1, (outs GPRAll:$Rn_wb), (ins GPRAll:$Rd, GPRAll:$Rn, imms8:$Offset),
                      "e_strb_pre"#cc.Suffix#" $Rn, $Rd, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }
def STRB_POST#cc.Suffix : EZHInstMem<0x02, 1, 0, 0, 1, 1, (outs GPRAll:$Rn_wb), (ins GPRAll:$Rd, GPRAll:$Rn, imms8:$Offset),
                      "e_strb_post"#cc.Suffix#" $Rn, $Rd, $Offset", []> { let Constraints = "$Rn = $Rn_wb"; }

// RegShiftALU operations
foreach sf = SetFlags in {
let S = sf.Value in {

// Shift only (alu_op = 0, is_post = 0)
def RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<0, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift,
    "e_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (shl i32:$Rs2, i32:$Rshift))], [])> { 
      let Rs1 = 0; 
      let Constraints = predicated_constraints;
    }
def RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<0, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift,
    "e_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (srl i32:$Rs2, i32:$Rshift))], [])> { 
      let Rs1 = 0; 
      let Constraints = predicated_constraints;
    }
def RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<0, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift,
    "e_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (sra i32:$Rs2, i32:$Rshift))], [])> { 
      let Rs1 = 0; 
      let Constraints = predicated_constraints;
    }
def RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<0, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift,
    "e_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (rotr i32:$Rs2, i32:$Rshift))], [])> { 
      let Rs1 = 0; 
      let Constraints = predicated_constraints;
    }

// AND (alu_op = 1)
// is_post = 0 (Shift then ALU)
def RLSL_AND_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (shl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RLSR_AND_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (srl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RASR_AND_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (sra i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RROR_AND_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_and"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (and i32:$Rs1, (rotr i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def AND_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_and_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def AND_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_and_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def AND_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_and_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def AND_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<1, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_and_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

// OR (alu_op = 2)
// is_post = 0 (Shift then ALU)
def RLSL_OR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (shl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RLSR_OR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (srl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RASR_OR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (sra i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RROR_OR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_or"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (or i32:$Rs1, (rotr i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def OR_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_or_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def OR_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_or_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def OR_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_or_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def OR_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<2, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_or_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

// XOR (alu_op = 3)
// is_post = 0 (Shift then ALU)
def RLSL_XOR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (shl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RLSR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (srl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RASR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (sra i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RROR_XOR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_xor"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (xor i32:$Rs1, (rotr i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def XOR_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_xor_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def XOR_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_xor_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def XOR_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_xor_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def XOR_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<3, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_xor_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

// ADD (alu_op = 4)
// is_post = 0 (Shift then ALU)
def RLSL_ADD_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (shl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RLSR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (srl i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RASR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (sra i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
def RROR_ADD_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_add"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (add i32:$Rs1, (rotr i32:$Rs2, i32:$Rshift)))], [])> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def ADD_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_add_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADD_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_add_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADD_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_add_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADD_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<4, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_add_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

// SUB (alu_op = 5)
// is_post = 0 (Shift then ALU)
def RLSL_SUB_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (shl i32:$Rs2, i32:$Rshift), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def RLSR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (srl i32:$Rs2, i32:$Rshift), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def RASR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (sub (sra i32:$Rs2, i32:$Rshift), i32:$Rs1))], [])> {
      let Constraints = predicated_constraints;
    }
def RROR_SUB_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_sub"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def SUB_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sub_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SUB_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sub_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SUB_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sub_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SUB_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<5, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sub_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", !if(!empty(cc.Suffix), [(set i32:$Rd, (rotr (sub i32:$Rs2, i32:$Rs1), i32:$Rshift))], [])> {
      let Constraints = predicated_constraints;
    }

// ADC (alu_op = 6)
// is_post = 0 (Shift then ALU)
def RLSL_ADC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RLSR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RASR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RROR_ADC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_adc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def ADC_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_adc_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADC_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_adc_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADC_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_adc_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def ADC_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<6, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_adc_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

// SBC (alu_op = 7)
// is_post = 0 (Shift then ALU)
def RLSL_SBC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 0, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsl_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RLSR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 1, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rlsr_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RASR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 2, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rasr_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def RROR_SBC_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 3, 0, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_rror_sbc"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
// is_post = 1 (ALU then Shift)
def SBC_RLSL_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 0, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sbc_rlsl"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SBC_RLSR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 1, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sbc_rlsr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SBC_RASR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 2, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sbc_rasr"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }
def SBC_RROR_#sf.Suffix#_#cc.Suffix : EZHInstRegShiftALU<7, 3, 1, (outs GPRAll:$Rd), ins_GPRAll_reg_shift_alu,
    "e_sbc_rror"#sf.Suffix#cc.Suffix#" $Rd, $Rs1, $Rs2, $Rshift", []> {
      let Constraints = predicated_constraints;
    }

}
}

// Tight Loop
def TIGHT_LOOP#cc.Suffix : EZHInstTightLoop<(outs), (ins GPRAll:$Rend, GPRAll:$Rcount), "e_tight_loop"#cc.Suffix#" $Rend, $Rcount", []>;

// Reg Offset Load/Store
def LDR_REG#cc.Suffix : EZHInstMemReg<0x1D, 0, 1, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rn, GPRAll:$Rm), "e_ldr_reg"#cc.Suffix#" $Rd, $Rn, $Rm", []>;
def LDR_REGB#cc.Suffix : EZHInstMemReg<0x1D, 0, 0, 0, (outs GPRAll:$Rd), (ins GPRAll:$Rn, GPRAll:$Rm), "e_ldr_regb"#cc.Suffix#" $Rd, $Rn, $Rm", []>;
def LDR_REGBS#cc.Suffix : EZHInstMemReg<0x1D, 0, 0, 1, (outs GPRAll:$Rd), (ins GPRAll:$Rn, GPRAll:$Rm), "e_ldr_regbs"#cc.Suffix#" $Rd, $Rn, $Rm", []>;

def STR_REG#cc.Suffix : EZHInstMemReg<0x1D, 1, 1, 0, (outs), (ins GPRAll:$Rd, GPRAll:$Rn, GPRAll:$Rm), "e_str_reg"#cc.Suffix#" $Rn, $Rd, $Rm", []>;
def STR_REGB#cc.Suffix : EZHInstMemReg<0x1D, 1, 0, 0, (outs), (ins GPRAll:$Rd, GPRAll:$Rn, GPRAll:$Rm), "e_str_regb"#cc.Suffix#" $Rn, $Rd, $Rm", []>;

// Push / Pop (Upward Stack)
def PUSH#cc.Suffix : EZHInstMem<0x02, 1, 1, 0, 1, 1, (outs), (ins GPRAll:$Rd), "e_push"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 1; // +1
}
def POP#cc.Suffix : EZHInstMem<0x01, 0, 1, 0, 0, 1, (outs GPRAll:$Rd), (ins), "e_pop"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 0xFF; // -1
}
def PUSHB#cc.Suffix : EZHInstMem<0x02, 1, 0, 0, 1, 1, (outs), (ins GPRAll:$Rd), "e_pushb"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 1; // +1
}
def POPB#cc.Suffix : EZHInstMem<0x01, 0, 0, 0, 0, 1, (outs GPRAll:$Rd), (ins), "e_popb"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 0xFF; // -1
}
def POPBS#cc.Suffix : EZHInstMem<0x01, 0, 0, 1, 0, 1, (outs GPRAll:$Rd), (ins), "e_popbs"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 0xFF; // -1
}

// Push / Pop Down (Downward Stack)
def PUSHD#cc.Suffix : EZHInstMem<0x02, 1, 1, 0, 0, 1, (outs), (ins GPRAll:$Rd), "e_pushd"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 0xFF; // -1 (Pre-decrement)
}
def POPD#cc.Suffix : EZHInstMem<0x01, 0, 1, 0, 1, 1, (outs GPRAll:$Rd), (ins), "e_popd"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 1; // +1 (Post-increment)
}
def PUSHDB#cc.Suffix : EZHInstMem<0x02, 1, 0, 0, 0, 1, (outs), (ins GPRAll:$Rd), "e_pushdb"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 0xFF; // -1
}
def POPDB#cc.Suffix : EZHInstMem<0x01, 0, 0, 0, 1, 1, (outs GPRAll:$Rd), (ins), "e_popdb"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 1; // +1
}
def POPDBS#cc.Suffix : EZHInstMem<0x01, 0, 0, 1, 1, 1, (outs GPRAll:$Rd), (ins), "e_popdbs"#cc.Suffix#" $Rd", []> {
  let Rn = 12; // SP
  let Offset = 1; // +1
}

def PER_READ#cc.Suffix : EZHInstPerRead<0x04, (outs GPRAll:$Rd), (ins imm20:$addr), "e_per_read"#cc.Suffix#" $Rd, $addr", []>;
def PER_WRITE#cc.Suffix : EZHInstPerWrite<0x05, (outs), (ins GPRAll:$Rs, imm20:$addr), "e_per_write"#cc.Suffix#" $Rs, $addr", []>;

} // let Cond

// Bit manipulation
let Cond = cc.Value in {
foreach sf = SetFlags in {
let S = sf.Value in {
def BTST_#sf.Suffix#_#cc.Suffix : EZHInstBitReg<1, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, GPRAll:$Rbit), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_btst"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Rbit", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}
def BSET_#sf.Suffix#_#cc.Suffix : EZHInstBitReg<2, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, GPRAll:$Rbit), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_bset"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Rbit", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}
def BCLR_#sf.Suffix#_#cc.Suffix : EZHInstBitReg<1, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, GPRAll:$Rbit), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_bclr"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Rbit", []> { 
  let S_bit = 0; 
  let Constraints = predicated_constraints;
}
def BTOG_#sf.Suffix#_#cc.Suffix : EZHInstBitReg<3, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, GPRAll:$Rbit), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_btog"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Rbit", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}

def BTST_IMM_#sf.Suffix#_#cc.Suffix : EZHInstBit<1, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, imm5:$Bit5), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_btst_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Bit5", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}
def BSET_IMM_#sf.Suffix#_#cc.Suffix : EZHInstBit<2, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, imm5:$Bit5), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_bset_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Bit5", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}
def BCLR_IMM_#sf.Suffix#_#cc.Suffix : EZHInstBit<1, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, imm5:$Bit5), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_bclr_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Bit5", []> { 
  let S_bit = 0; 
  let Constraints = predicated_constraints;
}
def BTOG_IMM_#sf.Suffix#_#cc.Suffix : EZHInstBit<3, (outs GPRAll:$Rd), !con((ins GPRAll:$Rs, imm5:$Bit5), !if(is_predicated, (ins GPRAll:$src_in), (ins))), "e_btog_imm"#sf.Suffix#cc.Suffix#" $Rd, $Rs, $Bit5", []> { 
  let S_bit = 1; 
  let Constraints = predicated_constraints;
}
}


}
} // let Cond
} // foreach cc = CondCodes

// Unconditional instructions (no space for Cond field)
let Cond = 0, S = 0 in {
def NOP : EZHInstSingle<0x12, (outs), (ins), "e_nop", []> { let Inst{31-5} = 0; }
def INT_TRIGGER : EZHInstSingle<0x14, (outs), (ins imm24:$Imm), "e_int_trigger $Imm", []> { bits<24> Imm; let Inst{31-8} = Imm; let Inst{7-5} = 0; }
def MODIFY_GPO_BYTE : EZHInstModifyGPOByte<(outs), (ins imms8:$AndMask, imms8:$OrMask, imms8:$XorMask),
                        "e_modify_gpo_byte $AndMask, $OrMask, $XorMask", []>;
}

// Return
let isReturn = 1, isTerminator = 1, isBarrier = 1 in {
  def RET : EZHInst<(outs), (ins variable_ops), "e_goto ra", [(EZH_ret_glue_node)]> {
    let Inst{4-0}   = 0x15; // Opcode for E_GOTO
    let Inst{9-5}   = 0;
    let Inst{13-10} = 1;
    let Inst{17-14} = 0x0f; // RA
    let Inst{31-18} = 0;
  }
}

class EZHPseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  let Namespace = "EZH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}



def LOAD_CONSTANT : EZHPseudo<(outs GPRAll:$Rd), (ins i32imm:$imm), "", []>;

let isNotDuplicable = 1, isBarrier = 1, hasSideEffects = 1 in
def CONSTPOOL_ENTRY : EZHPseudo<(outs), (ins unknown:$sym, unknown:$val), "$sym:\n\t.long $val", []>;



let isCall = 1, Defs = [R0, R1, R2, R3, RA], isCodeGenOnly = 1 in {
def CALL_INDIRECT : EZHInstGotoReg<1, (outs), (ins tcGPR:$raddr), "e_goto_regl $raddr", [(EZH_call i32:$raddr)]>;
}


def EZHcmp : SDNode<"EZHISD::CMP", SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>, [SDNPOutGlue]>;
def EZHbrcc : SDNode<"EZHISD::BR_CC", SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>, [SDNPHasChain, SDNPInGlue]>;
def EZHselectcc : SDNode<"EZHISD::SELECT_CC", SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisVT<3, i32>]>, [SDNPInGlue]>;
def EZHbtog : SDNode<"EZHISD::BTOG", SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisVT<2, i32>]>>;
def EZHbrjt : SDNode<"EZHISD::BR_JT", SDTypeProfile<0, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i32>]>, [SDNPHasChain]>;
def EZHsjlj_setjmp : SDNode<"EZHISD::EH_SJLJ_SETJMP", SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>, [SDNPHasChain, SDNPOptInGlue]>;
def EZHsjlj_longjmp : SDNode<"EZHISD::EH_SJLJ_LONGJMP", SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>, [SDNPHasChain, SDNPOptInGlue]>;
def EZHsjlj_setup_dispatch : SDNode<"EZHISD::EH_SJLJ_SETUP_DISPATCH", SDTypeProfile<0, 0, []>, [SDNPHasChain, SDNPOptInGlue]>;

let Defs = [RA], isTerminator = 1, isBarrier = 1, isBranch = 1, isIndirectBranch = 1 in
def PseudoBR_JT : EZHPseudo<(outs), (ins GPRAll:$table, GPRAll:$index, i32imm:$jti), "# PseudoBR_JT $table, $index, $jti", [(EZHbrjt i32:$table, i32:$index, tjumptable:$jti)]>;

def : Pat<(EZHbtog i32:$Rs, (i32 imm:$Bit)), (BTOG_IMM__ GPRAll:$Rs, imm:$Bit)>;

let usesCustomInserter = 1, hasSideEffects = 1 in {
  def PseudoCMP : EZHPseudo<(outs), (ins GPRAll:$lhs, GPRAll:$rhs), "", [(EZHcmp i32:$lhs, i32:$rhs)]>;
  def PseudoCMPi : EZHPseudo<(outs), (ins GPRAll:$lhs, imms12:$rhs), "", [(EZHcmp i32:$lhs, imms12:$rhs)]>;
}

let usesCustomInserter = 1, isTerminator = 1, isBranch = 1 in {
  def PseudoBR_CC : EZHPseudo<(outs), (ins brtarget:$dest, i32imm:$cc), "", [(EZHbrcc bb:$dest, timm:$cc)]>;
}

let usesCustomInserter = 1 in {
  def PseudoSELECT_CC : EZHPseudo<(outs GPRAll:$dst), (ins GPRAll:$truev, GPRAll:$falsev, i32imm:$cc), "", [(set i32:$dst, (EZHselectcc i32:$truev, i32:$falsev, timm:$cc))]>;
}

// Pseudo instructions
let isPseudo = 1, hasSideEffects = 1 in {
  def ADJCALLSTACKDOWN : EZHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                 "",
                                 [(EZH_adjcallstackdown timm:$amt1, timm:$amt2)]>;
  def ADJCALLSTACKUP : EZHPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                              "",
                              [(EZH_adjcallstackup timm:$amt1, timm:$amt2)]>;
  
  let usesCustomInserter = 1 in {
    let Defs = [R0, R1, R2, R3, R4, R5, R6, R7, RA], isBarrier = 1 in {
      def EH_SjLj_SetJmp : EZHPseudo<(outs GPRAll:$dst), (ins GPRAll:$buf), "", 
                           [(set i32:$dst, (EZHsjlj_setjmp i32:$buf))]>;
    }
    let isTerminator = 1, isBarrier = 1 in {
      def EH_SjLj_LongJmp : EZHPseudo<(outs), (ins GPRAll:$buf), "", 
                            [(EZHsjlj_longjmp i32:$buf)]>;
    }
    let isBarrier = 1, hasSideEffects = 1 in {
      def EH_SjLj_Setup_Dispatch : EZHPseudo<(outs), (ins), "",
                            [(EZHsjlj_setup_dispatch)]>;
    }
  }
}


// Pseudo instructions

foreach cc = CondCodes in {
let Cond = cc.Value, S = 0 in {

let isBranch = 1, isTerminator = 1, isBarrier = !if(!empty(cc.Suffix), 1, 0) in {
  let S = 1 in {
    def GOTO#cc.Suffix : EZHInstGoto<0, (outs), (ins brtarget:$addr), "e_goto"#cc.Suffix#" $addr", !if(!empty(cc.Suffix), [(br bb:$addr)], [])>;
    def GOTOL#cc.Suffix : EZHInstGoto<1, (outs), (ins brtarget:$addr), "e_gotol"#cc.Suffix#" $addr", []> { let Defs = [RA]; }
  }
  let S = 0 in {
    def GOTO_REG#cc.Suffix : EZHInstGotoReg<0, (outs), (ins GPRAll:$raddr), "e_goto_reg"#cc.Suffix#" $raddr", !if(!empty(cc.Suffix), [(brind i32:$raddr)], [])>;
    def GOTO_REGL#cc.Suffix : EZHInstGotoReg<1, (outs), (ins GPRAll:$raddr), "e_goto_regl"#cc.Suffix#" $raddr", []> { let Defs = [RA]; }
  }
}

let isCall = 1, Uses = [SP], Defs = [R0, R1, R2, R3, RA], isCodeGenOnly = 1 in {
  let S = 1 in {
      }
  let S = 0 in {
      }
}
// Hold instructions
let isTerminator = 1, isBarrier = 1, hasSideEffects = 1 in {
  let S = 0 in {
    def HOLD#cc.Suffix : EZHInst<(outs), (ins), "e_hold"#cc.Suffix, []> {
      let Opcode = 0x1C;
      let Inst{13-10} = 13; // PC
      let Inst{14} = 0;
      let Inst{15} = 1;
      let Inst{31-16} = 0;
    }
    
    def VECTORED_HOLD#cc.Suffix : EZHInstVectoredHold<0, 0, (outs), (ins GPRAll:$table), "e_vectored_hold"#cc.Suffix#" $table", []>;
    def VECTORED_HOLD_NRA#cc.Suffix : EZHInstVectoredHold<1, 0, (outs), (ins GPRAll:$table), "e_vectored_hold_nra"#cc.Suffix#" $table", []>;
    def VECTORED_HOLD_LV#cc.Suffix : EZHInstVectoredHold<0, 1, (outs), (ins GPRAll:$table), "e_vectored_hold_lv"#cc.Suffix#" $table", []>;
    def VECTORED_HOLD_LV_NRA#cc.Suffix : EZHInstVectoredHold<1, 1, (outs), (ins GPRAll:$table), "e_vectored_hold_lv_nra"#cc.Suffix#" $table", []>;
  }
  let S = 1 in {
    def ACC_VECTORED_HOLD#cc.Suffix : EZHInstAccVectoredHold<0, 0, (outs), (ins GPRAll:$table, imms8:$vectors), "e_acc_vectored_hold"#cc.Suffix#" $table, $vectors", []>;
    def ACC_VECTORED_HOLD_NRA#cc.Suffix : EZHInstAccVectoredHold<1, 0, (outs), (ins GPRAll:$table, imms8:$vectors), "e_acc_vectored_hold_nra"#cc.Suffix#" $table, $vectors", []>;
    def ACC_VECTORED_HOLD_LV#cc.Suffix : EZHInstAccVectoredHold<0, 1, (outs), (ins GPRAll:$table, imms8:$vectors), "e_acc_vectored_hold_lv"#cc.Suffix#" $table, $vectors", []>;
    def ACC_VECTORED_HOLD_LV_NRA#cc.Suffix : EZHInstAccVectoredHold<1, 1, (outs), (ins GPRAll:$table, imms8:$vectors), "e_acc_vectored_hold_lv_nra"#cc.Suffix#" $table, $vectors", []>;
  }
}

} // let Cond
} // foreach cc

// Special Beats

let Cond = 1, S = 0 in
def HEART_RYTHM_IMM : EZHInstSingle<0x12, (outs), (ins imm16:$Imm), "e_heart_rythm_imm $Imm", []> {
  bits<16> Imm;
  let Inst{31-16} = Imm;
  let Inst{15-6} = 0;
  let Inst{5} = 1;
}

def HEART_RYTHM : EZHInstSingle<0x12, (outs), (ins GPRAll:$Rs), "e_heart_rythm $Rs", []> {
  bits<4> Rs;
  let Inst{31-18} = 0;
  let Inst{17-14} = Rs;
  let Inst{13-10} = 0;
  let Inst{9} = 1;
  let Inst{8-6} = 0;
  let Inst{5} = 1;
}

def WAIT_FOR_BEAT : EZHInstSingle<0x12, (outs), (ins), "e_wait_for_beat", []> {
  let Inst{31-7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 1;
  let Inst{4-0} = 0x12;
}

def SYNCH_ALL_TO_BEAT : EZHInstSingle<0x12, (outs), (ins imm1:$Imm), "e_synch_all_to_beat $Imm", []> {
  bits<1> Imm;
  let Inst{31} = Imm;
  let Inst{30-7} = 0;
  let Inst{6} = 1;
  let Inst{5} = 0;
  let Inst{4-0} = 0x12;
}

// Call instructions
let isCall = 1, Defs = [R0, R1, R2, R3, RA] in {
def CALL : EZHInstCall<0x03, (outs), (ins calltarget:$addr), "e_gosub $addr", [(EZH_call tglobaladdr:$addr)]>;
let isCodeGenOnly = 1 in
def CALLExt : EZHInstCall<0x03, (outs), (ins calltarget:$addr), "e_gosub $addr", [(EZH_call texternalsym:$addr)]>;
}

// Memory Patterns (Words and Bytes)
// 16-bit access is handled by the Legalizer's Expand mechanism into bytes.
def : Pat<(i32 (load GPRAll:$Rn)), (LDR GPRAll:$Rn, 0)>;
def : Pat<(store i32:$Rd, GPRAll:$Rn), (STR GPRAll:$Rd, GPRAll:$Rn, 0)>;

def : Pat<(i32 (zextloadi8 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;
def : Pat<(i32 (sextloadi8 GPRAll:$Rn)), (LDRBS GPRAll:$Rn, 0)>;
def : Pat<(i32 (extloadi8 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;

// Final Memory Patterns and Indexed Addressing
def : Pat<(i32 (load GPRAll:$Rn)), (LDR GPRAll:$Rn, 0)>;
def : Pat<(store i32:$Rd, GPRAll:$Rn), (STR GPRAll:$Rd, GPRAll:$Rn, 0)>;


// Memory Patterns (Words and Bytes)
def : Pat<(i32 (load GPRAll:$Rn)), (LDR GPRAll:$Rn, 0)>;
def : Pat<(store i32:$Rd, GPRAll:$Rn), (STR GPRAll:$Rd, GPRAll:$Rn, 0)>;

def : Pat<(i32 (zextloadi8 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;
def : Pat<(i32 (sextloadi8 GPRAll:$Rn)), (LDRBS GPRAll:$Rn, 0)>;
def : Pat<(i32 (extloadi8 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;
def : Pat<(truncstorei8 GPRAll:$Rd, GPRAll:$Rn), (STRB GPRAll:$Rd, GPRAll:$Rn, 0)>;

def : Pat<(i32 (extloadi1 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;
def : Pat<(i32 (zextloadi1 GPRAll:$Rn)), (LDRB GPRAll:$Rn, 0)>;

// Variable shift patterns
def : Pat<(shl GPRAll:$Rs2, GPRAll:$Rshift), (RLSL__ GPRAll:$Rs2, GPRAll:$Rshift)>;
def : Pat<(srl GPRAll:$Rs2, GPRAll:$Rshift), (RLSR__ GPRAll:$Rs2, GPRAll:$Rshift)>;
def : Pat<(sra GPRAll:$Rs2, GPRAll:$Rshift), (RASR__ GPRAll:$Rs2, GPRAll:$Rshift)>;

// Patterns for Opcode 6/8 shifted instructions (Shift the Result)
def : Pat<(shl (add i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (ADD_LSL__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(srl (add i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (ADD_LSR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(sra (add i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (ADD_ASR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(rotr (add i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (ADD_ROR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;

def : Pat<(shl (sub i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (SUB_LSL__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(srl (sub i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (SUB_LSR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(sra (sub i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (SUB_ASR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;
def : Pat<(rotr (sub i32:$Rs1, i32:$Rs2), (i32 imm:$ShAmt)), (SUB_ROR__ GPRAll:$Rs1, GPRAll:$Rs2, imm:$ShAmt)>;

//===----------------------------------------------------------------------===//
// Instruction Selection Patterns (ISel lowering)
//===----------------------------------------------------------------------===//

// 1. Negate-Result ALU
def : Pat<(not (add i32:$Rs1, i32:$Rs2)), (ADDNrr__ GPRAll:$Rs1, GPRAll:$Rs2)>;
def : Pat<(not (sub i32:$Rs1, i32:$Rs2)), (SUBNrr__ GPRAll:$Rs1, GPRAll:$Rs2)>;
def : Pat<(not (and i32:$Rs1, i32:$Rs2)), (ANDNrr__ GPRAll:$Rs1, GPRAll:$Rs2)>;
def : Pat<(not (or i32:$Rs1, i32:$Rs2)), (ORNrr__ GPRAll:$Rs1, GPRAll:$Rs2)>;
def : Pat<(not (xor i32:$Rs1, i32:$Rs2)), (XORNrr__ GPRAll:$Rs1, GPRAll:$Rs2)>;

// 2. Combined AND-then-OR
def : Pat<(or (and i32:$Rs1, i32:$Rs2), i32:$Rs3), (ANDOR__ GPRAll:$Rs1, GPRAll:$Rs2, GPRAll:$Rs3)>;

// 3. Dynamic Bit Manipulation
def : Pat<(or i32:$Rs, (shl 1, i32:$Rbit)), (BSET__ GPRAll:$Rs, GPRAll:$Rbit)>;
def : Pat<(and i32:$Rs, (not (shl 1, i32:$Rbit))), (BCLR__ GPRAll:$Rs, GPRAll:$Rbit)>;
def : Pat<(xor i32:$Rs, (shl 1, i32:$Rbit)), (BTOG__ GPRAll:$Rs, GPRAll:$Rbit)>;
def : Pat<(and i32:$Rs, (shl 1, i32:$Rbit)), (BTST__ GPRAll:$Rs, GPRAll:$Rbit)>;

// 4. Immediate Bit Manipulation (Power of 2 Matching)
def Log2 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(Log2_32(N->getZExtValue()), SDLoc(N), MVT::i32);
}]>;

def Log2Inv : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(Log2_32(~N->getZExtValue()), SDLoc(N), MVT::i32);
}]>;

def power_of_2 : PatLeaf<(imm), [{
  return isPowerOf2_32(N->getZExtValue());
}], Log2>;

def power_of_2_inv : PatLeaf<(imm), [{
  return isPowerOf2_32(~N->getZExtValue());
}], Log2Inv>;

def : Pat<(or i32:$Rs, power_of_2:$Bit), (BSET_IMM__ GPRAll:$Rs, power_of_2:$Bit)>;
def : Pat<(and i32:$Rs, power_of_2_inv:$Bit), (BCLR_IMM__ GPRAll:$Rs, power_of_2_inv:$Bit)>;
def : Pat<(xor i32:$Rs, power_of_2:$Bit), (BTOG_IMM__ GPRAll:$Rs, power_of_2:$Bit)>;

def : Pat<(bswap i32:$Rs2), (FEND_LSR__ GPRAll:$Rs2, 0)>;
def : Pat<(bitreverse i32:$Rs2), (FBIT_LSR__ GPRAll:$Rs2, 0)>;
