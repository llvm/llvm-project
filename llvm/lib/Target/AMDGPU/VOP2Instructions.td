//===-- VOP2Instructions.td - Vector Instruction Defintions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOP2 Classes
//===----------------------------------------------------------------------===//

class VOP2e <bits<6> op, VOPProfile P> : Enc32 {
  bits<8> vdst;
  bits<9> src0;
  bits<8> src1;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; //encoding
}

class VOP2_MADKe <bits<6> op, VOPProfile P> : Enc64 {
  bits<8>  vdst;
  bits<9>  src0;
  bits<8>  src1;
  bits<32> imm;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
  let Inst{63-32} = imm;
}

class VOP2_SDWAe <bits<6> op, VOPProfile P> : VOP_SDWAe <P> {
  bits<8> vdst;
  bits<8> src1;

  let Inst{8-0}   = 0xf9; // sdwa
  let Inst{16-9}  = !if(P.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
}

class VOP2_Pseudo <string opName, VOPProfile P, list<dag> pattern=[], string suffix = "_e32"> :
  InstSI <P.Outs32, P.Ins32, "", pattern>,
  VOP <opName>,
  SIMCInstr <opName#suffix, SIEncodingFamily.NONE>,
  MnemonicAlias<opName#suffix, opName> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = P.Asm32;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SubtargetPredicate = isGCN;

  let VOP2 = 1;
  let VALU = 1;
  let Uses = [EXEC];

  let AsmVariantName = AMDGPUAsmVariants.Default;

  VOPProfile Pfl = P;
}

class VOP2_Real <VOP2_Pseudo ps, int EncodingFamily> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
}

class VOP2_SDWA_Pseudo <string OpName, VOPProfile P, list<dag> pattern=[]> :
  VOP_SDWA_Pseudo <OpName, P, pattern> {
  let AsmMatchConverter = "cvtSdwaVOP2";
}

class getVOP2Pat64 <SDPatternOperator node, VOPProfile P> : LetDummies {
  list<dag> ret = !if(P.HasModifiers,
    [(set P.DstVT:$vdst,
      (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
            (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
    [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]);
}

multiclass VOP2Inst <string opName,
                     VOPProfile P,
                     SDPatternOperator node = null_frag,
                     string revOp = opName> {

  def _e32 : VOP2_Pseudo <opName, P>,
             Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
             Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _sdwa : VOP2_SDWA_Pseudo <opName, P>;
}

// TODO: add SDWA pseudo instructions for VOP2bInst and VOP2eInst
multiclass VOP2bInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {

  let SchedRW = [Write32Bit, WriteSALU] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]), Defs = [VCC] in {
      def _e32 : VOP2_Pseudo <opName, P>,
                 Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;

      def _sdwa : VOP2_SDWA_Pseudo <opName, P>;
    }

    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
  }
}

multiclass VOP2eInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {

  let SchedRW = [Write32Bit] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]) in {
      def _e32 : VOP2_Pseudo <opName, P>,
                 Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;
    }

    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
  }
}

class VOP_MADAK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = (ins VCSrc_f32:$src0, VGPR_32:$src1, ImmOpType:$imm);
  field string Asm32 = "$vdst, $src0, $src1, $imm";
  field bit HasExt = 0;
}

def VOP_MADAK_F16 : VOP_MADAK <f16>;
def VOP_MADAK_F32 : VOP_MADAK <f32>;

class VOP_MADMK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = (ins VCSrc_f32:$src0, ImmOpType:$imm, VGPR_32:$src1);
  field string Asm32 = "$vdst, $src0, $imm, $src1";
  field bit HasExt = 0;
}

def VOP_MADMK_F16 : VOP_MADMK <f16>;
def VOP_MADMK_F32 : VOP_MADMK <f32>;

class VOP_MAC <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1, VGPR_32:$src2);
  let Ins64 = getIns64<Src0RC64, Src1RC64, RegisterOperand<VGPR_32>, 3,
                       HasModifiers, HasOMod, Src0Mod, Src1Mod, Src2Mod>.ret;
  let InsDPP = (ins Src0ModDPP:$src0_modifiers, Src0DPP:$src0,
                    Src1ModDPP:$src1_modifiers, Src1DPP:$src1,
                    VGPR_32:$src2, // stub argument
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsSDWA = (ins Src0ModSDWA:$src0_modifiers, Src0SDWA:$src0,
                     Src1ModSDWA:$src1_modifiers, Src1SDWA:$src1,
                     VGPR_32:$src2, // stub argument
                     clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);
  let Asm32 = getAsm32<1, 2, vt>.ret;
  let Asm64 = getAsm64<1, 2, HasModifiers, HasOMod, vt>.ret;
  let AsmDPP = getAsmDPP<1, 2, HasModifiers, vt>.ret;
  let AsmSDWA = getAsmSDWA<1, 2, HasModifiers, vt>.ret;
  let HasSrc2 = 0;
  let HasSrc2Mods = 0;
  let HasExt = 1;
}

def VOP_MAC_F16 : VOP_MAC <f16> {
  // FIXME: Move 'Asm64' definition to VOP_MAC, and use 'vt'. Currently it gives
  // 'not a string initializer' error.
  let Asm64 = getAsm64<1, 2, HasModifiers, HasOMod, f16>.ret;
}

def VOP_MAC_F32 : VOP_MAC <f32> {
  // FIXME: Move 'Asm64' definition to VOP_MAC, and use 'vt'. Currently it gives
  // 'not a string initializer' error.
  let Asm64 = getAsm64<1, 2, HasModifiers, HasOMod, f32>.ret;
}

// Write out to vcc or arbitrary SGPR.
def VOP2b_I32_I1_I32_I32 : VOPProfile<[i32, i32, i32, untyped]> {
  let Asm32 = "$vdst, vcc, $src0, $src1";
  let Asm64 = "$vdst, $sdst, $src0, $src1";
  let AsmSDWA = "$vdst, vcc, $src0_modifiers, $src1_modifiers$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmDPP = "$vdst, vcc, $src0, $src1 $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);
}

// Write out to vcc or arbitrary SGPR and read in from vcc or
// arbitrary SGPR.
def VOP2b_I32_I1_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1]> {
  // We use VCSrc_b32 to exclude literal constants, even though the
  // encoding normally allows them since the implicit VCC use means
  // using one would always violate the constant bus
  // restriction. SGPRs are still allowed because it should
  // technically be possible to use VCC again as src0.
  let Src0RC32 = VCSrc_b32;
  let Asm32 = "$vdst, vcc, $src0, $src1, vcc";
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2";
  let AsmSDWA = "$vdst, vcc, $src0_modifiers, $src1_modifiers, vcc $clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmDPP = "$vdst, vcc, $src0, $src1, vcc $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, SReg_64:$sdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);

  let InsSDWA = (ins Src0Mod:$src0_modifiers, Src0SDWA:$src0,
                     Src1Mod:$src1_modifiers, Src1SDWA:$src1,
                     clampmod:$clamp, dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);

  let InsDPP = (ins Src0Mod:$src0_modifiers, Src0DPP:$src0,
                    Src1Mod:$src1_modifiers, Src1DPP:$src1,
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let HasExt = 1;
}

// Read in from vcc or arbitrary SGPR
def VOP2e_I32_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1]> {
  let Src0RC32 = VCSrc_b32; // See comment in def VOP2b_I32_I1_I32_I32_I1 above.
  let Asm32 = "$vdst, $src0, $src1, vcc";
  let Asm64 = "$vdst, $src0, $src1, $src2";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);
}

def VOP_READLANE : VOPProfile<[i32, i32, i32]> {
  let Outs32 = (outs SReg_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins VGPR_32:$src0, SCSrc_b32:$src1);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;
}

def VOP_WRITELANE : VOPProfile<[i32, i32, i32]> {
  let Outs32 = (outs VGPR_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins SReg_32:$src0, SCSrc_b32:$src1);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;
}

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

let SubtargetPredicate = isGCN in {

defm V_CNDMASK_B32 : VOP2eInst <"v_cndmask_b32", VOP2e_I32_I32_I32_I1>;
def V_MADMK_F32 : VOP2_Pseudo <"v_madmk_f32", VOP_MADMK_F32>;

let isCommutable = 1 in {
defm V_ADD_F32 : VOP2Inst <"v_add_f32", VOP_F32_F32_F32, fadd>;
defm V_SUB_F32 : VOP2Inst <"v_sub_f32", VOP_F32_F32_F32, fsub>;
defm V_SUBREV_F32 : VOP2Inst <"v_subrev_f32", VOP_F32_F32_F32, null_frag, "v_sub_f32">;
defm V_MUL_LEGACY_F32 : VOP2Inst <"v_mul_legacy_f32", VOP_F32_F32_F32, AMDGPUfmul_legacy>;
defm V_MUL_F32 : VOP2Inst <"v_mul_f32", VOP_F32_F32_F32, fmul>;
defm V_MUL_I32_I24 : VOP2Inst <"v_mul_i32_i24", VOP_I32_I32_I32, AMDGPUmul_i24>;
defm V_MUL_HI_I32_I24 : VOP2Inst <"v_mul_hi_i32_i24", VOP_I32_I32_I32, AMDGPUmulhi_i24>;
defm V_MUL_U32_U24 : VOP2Inst <"v_mul_u32_u24", VOP_I32_I32_I32, AMDGPUmul_u24>;
defm V_MUL_HI_U32_U24 : VOP2Inst <"v_mul_hi_u32_u24", VOP_I32_I32_I32, AMDGPUmulhi_u24>;
defm V_MIN_F32 : VOP2Inst <"v_min_f32", VOP_F32_F32_F32, fminnum>;
defm V_MAX_F32 : VOP2Inst <"v_max_f32", VOP_F32_F32_F32, fmaxnum>;
defm V_MIN_I32 : VOP2Inst <"v_min_i32", VOP_I32_I32_I32>;
defm V_MAX_I32 : VOP2Inst <"v_max_i32", VOP_I32_I32_I32>;
defm V_MIN_U32 : VOP2Inst <"v_min_u32", VOP_I32_I32_I32>;
defm V_MAX_U32 : VOP2Inst <"v_max_u32", VOP_I32_I32_I32>;
defm V_LSHRREV_B32 : VOP2Inst <"v_lshrrev_b32", VOP_I32_I32_I32, null_frag, "v_lshr_b32">;
defm V_ASHRREV_I32 : VOP2Inst <"v_ashrrev_i32", VOP_I32_I32_I32, null_frag, "v_ashr_i32">;
defm V_LSHLREV_B32 : VOP2Inst <"v_lshlrev_b32", VOP_I32_I32_I32, null_frag, "v_lshl_b32">;
defm V_AND_B32 : VOP2Inst <"v_and_b32", VOP_I32_I32_I32>;
defm V_OR_B32 : VOP2Inst <"v_or_b32", VOP_I32_I32_I32>;
defm V_XOR_B32 : VOP2Inst <"v_xor_b32", VOP_I32_I32_I32>;

let Constraints = "$vdst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F32 : VOP2Inst <"v_mac_f32", VOP_MAC_F32>;
}

def V_MADAK_F32 : VOP2_Pseudo <"v_madak_f32", VOP_MADAK_F32>;

// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.

// V_ADD_I32, V_SUB_I32, and V_SUBREV_I32 where renamed to *_U32 in VI,
// but the VI instructions behave the same as the SI versions.
defm V_ADD_I32 : VOP2bInst <"v_add_i32", VOP2b_I32_I1_I32_I32>;
defm V_SUB_I32 : VOP2bInst <"v_sub_i32", VOP2b_I32_I1_I32_I32>;
defm V_SUBREV_I32 : VOP2bInst <"v_subrev_i32", VOP2b_I32_I1_I32_I32, null_frag, "v_sub_i32">;
defm V_ADDC_U32 : VOP2bInst <"v_addc_u32", VOP2b_I32_I1_I32_I32_I1>;
defm V_SUBB_U32 : VOP2bInst <"v_subb_u32", VOP2b_I32_I1_I32_I32_I1>;
defm V_SUBBREV_U32 : VOP2bInst <"v_subbrev_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32">;
} // End isCommutable = 1

// These are special and do not read the exec mask.
let isConvergent = 1, Uses = []<Register> in {
def V_READLANE_B32 : VOP2_Pseudo<"v_readlane_b32", VOP_READLANE,
  [(set i32:$vdst, (int_amdgcn_readlane i32:$src0, i32:$src1))], "">;

def V_WRITELANE_B32 : VOP2_Pseudo<"v_writelane_b32", VOP_WRITELANE, [], "">;
} // End isConvergent = 1

defm V_BFM_B32 : VOP2Inst <"v_bfm_b32", VOP_I32_I32_I32>;
defm V_BCNT_U32_B32 : VOP2Inst <"v_bcnt_u32_b32", VOP_I32_I32_I32>;
defm V_MBCNT_LO_U32_B32 : VOP2Inst <"v_mbcnt_lo_u32_b32", VOP_I32_I32_I32, int_amdgcn_mbcnt_lo>;
defm V_MBCNT_HI_U32_B32 : VOP2Inst <"v_mbcnt_hi_u32_b32", VOP_I32_I32_I32, int_amdgcn_mbcnt_hi>;
defm V_LDEXP_F32 : VOP2Inst <"v_ldexp_f32", VOP_F32_F32_I32, AMDGPUldexp>;
defm V_CVT_PKACCUM_U8_F32 : VOP2Inst <"v_cvt_pkaccum_u8_f32", VOP_I32_F32_I32>; // TODO: set "Uses = dst"
defm V_CVT_PKNORM_I16_F32 : VOP2Inst <"v_cvt_pknorm_i16_f32", VOP_I32_F32_F32>;
defm V_CVT_PKNORM_U16_F32 : VOP2Inst <"v_cvt_pknorm_u16_f32", VOP_I32_F32_F32>;
defm V_CVT_PKRTZ_F16_F32 : VOP2Inst <"v_cvt_pkrtz_f16_f32", VOP_I32_F32_F32, AMDGPUpkrtz_f16_f32>;
defm V_CVT_PK_U16_U32 : VOP2Inst <"v_cvt_pk_u16_u32", VOP_I32_I32_I32>;
defm V_CVT_PK_I16_I32 : VOP2Inst <"v_cvt_pk_i16_i32", VOP_I32_I32_I32>;

} // End SubtargetPredicate = isGCN


// These instructions only exist on SI and CI
let SubtargetPredicate = isSICI in {

defm V_MIN_LEGACY_F32 : VOP2Inst <"v_min_legacy_f32", VOP_F32_F32_F32, AMDGPUfmin_legacy>;
defm V_MAX_LEGACY_F32 : VOP2Inst <"v_max_legacy_f32", VOP_F32_F32_F32, AMDGPUfmax_legacy>;

let isCommutable = 1 in {
defm V_MAC_LEGACY_F32 : VOP2Inst <"v_mac_legacy_f32", VOP_F32_F32_F32>;
defm V_LSHR_B32 : VOP2Inst <"v_lshr_b32", VOP_I32_I32_I32>;
defm V_ASHR_I32 : VOP2Inst <"v_ashr_i32", VOP_I32_I32_I32>;
defm V_LSHL_B32 : VOP2Inst <"v_lshl_b32", VOP_I32_I32_I32>;
} // End isCommutable = 1

} // End let SubtargetPredicate = SICI

let SubtargetPredicate = isVI in {

def V_MADMK_F16 : VOP2_Pseudo <"v_madmk_f16", VOP_MADMK_F16>;
defm V_LSHLREV_B16 : VOP2Inst <"v_lshlrev_b16", VOP_I16_I16_I16>;
defm V_LSHRREV_B16 : VOP2Inst <"v_lshrrev_b16", VOP_I16_I16_I16>;
defm V_ASHRREV_I16 : VOP2Inst <"v_ashrrev_i16", VOP_I16_I16_I16>;
defm V_LDEXP_F16 : VOP2Inst <"v_ldexp_f16", VOP_F16_F16_I32, AMDGPUldexp>;

let isCommutable = 1 in {
defm V_ADD_F16 : VOP2Inst <"v_add_f16", VOP_F16_F16_F16, fadd>;
defm V_SUB_F16 : VOP2Inst <"v_sub_f16", VOP_F16_F16_F16, fsub>;
defm V_SUBREV_F16 : VOP2Inst <"v_subrev_f16", VOP_F16_F16_F16, null_frag, "v_sub_f16">;
defm V_MUL_F16 : VOP2Inst <"v_mul_f16", VOP_F16_F16_F16, fmul>;
def V_MADAK_F16 : VOP2_Pseudo <"v_madak_f16", VOP_MADAK_F16>;
defm V_ADD_U16 : VOP2Inst <"v_add_u16", VOP_I16_I16_I16>;
defm V_SUB_U16 : VOP2Inst <"v_sub_u16" , VOP_I16_I16_I16>;
defm V_SUBREV_U16 : VOP2Inst <"v_subrev_u16", VOP_I16_I16_I16, null_frag, "v_sub_u16">;
defm V_MUL_LO_U16 : VOP2Inst <"v_mul_lo_u16", VOP_I16_I16_I16>;
defm V_MAX_F16 : VOP2Inst <"v_max_f16", VOP_F16_F16_F16, fmaxnum>;
defm V_MIN_F16 : VOP2Inst <"v_min_f16", VOP_F16_F16_F16, fminnum>;
defm V_MAX_U16 : VOP2Inst <"v_max_u16", VOP_I16_I16_I16>;
defm V_MAX_I16 : VOP2Inst <"v_max_i16", VOP_I16_I16_I16>;
defm V_MIN_U16 : VOP2Inst <"v_min_u16", VOP_I16_I16_I16>;
defm V_MIN_I16 : VOP2Inst <"v_min_i16", VOP_I16_I16_I16>;

let Constraints = "$vdst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F16 : VOP2Inst <"v_mac_f16", VOP_MAC_F16>;
}
} // End isCommutable = 1

} // End SubtargetPredicate = isVI

// Note: 16-bit instructions produce a 0 result in the high 16-bits.
multiclass Arithmetic_i16_Pats <SDPatternOperator op, Instruction inst> {

def : Pat<
  (op i16:$src0, i16:$src1),
  (inst $src0, $src1)
>;

def : Pat<
  (i32 (zext (op i16:$src0, i16:$src1))),
  (inst $src0, $src1)
>;

def : Pat<
  (i64 (zext (op i16:$src0, i16:$src1))),
   (REG_SEQUENCE VReg_64,
     (inst $src0, $src1), sub0,
     (V_MOV_B32_e32 (i32 0)), sub1)
>;

}

multiclass Bits_OpsRev_i16_Pats <SDPatternOperator op, Instruction inst> {

def : Pat<
  (op i16:$src0, i16:$src1),
  (inst $src1, $src0)
>;

def : Pat<
  (i32 (zext (op i16:$src0, i16:$src1))),
  (inst $src1, $src0)
>;


def : Pat<
  (i64 (zext (op i16:$src0, i16:$src1))),
   (REG_SEQUENCE VReg_64,
     (inst $src1, $src0), sub0,
     (V_MOV_B32_e32 (i32 0)), sub1)
>;
}

class ZExt_i16_i1_Pat <SDNode ext> : Pat <
  (i16 (ext i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 1), $src)
>;

let Predicates = [isVI] in {

defm : Arithmetic_i16_Pats<add, V_ADD_U16_e64>;
defm : Arithmetic_i16_Pats<mul, V_MUL_LO_U16_e64>;
defm : Arithmetic_i16_Pats<sub, V_SUB_U16_e64>;
defm : Arithmetic_i16_Pats<smin, V_MIN_I16_e64>;
defm : Arithmetic_i16_Pats<smax, V_MAX_I16_e64>;
defm : Arithmetic_i16_Pats<umin, V_MIN_U16_e64>;
defm : Arithmetic_i16_Pats<umax, V_MAX_U16_e64>;

def : Pat <
  (and i16:$src0, i16:$src1),
  (V_AND_B32_e64 $src0, $src1)
>;

def : Pat <
  (or i16:$src0, i16:$src1),
  (V_OR_B32_e64 $src0, $src1)
>;

def : Pat <
  (xor i16:$src0, i16:$src1),
  (V_XOR_B32_e64 $src0, $src1)
>;

defm : Bits_OpsRev_i16_Pats<shl, V_LSHLREV_B16_e64>;
defm : Bits_OpsRev_i16_Pats<srl, V_LSHRREV_B16_e64>;
defm : Bits_OpsRev_i16_Pats<sra, V_ASHRREV_I16_e64>;

def : ZExt_i16_i1_Pat<zext>;
def : ZExt_i16_i1_Pat<anyext>;

def : Pat <
  (i16 (sext i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0), (i32 -1), $src)
>;

// Undo sub x, c -> add x, -c canonicalization since c is more likely
// an inline immediate than -c.
// TODO: Also do for 64-bit.
def : Pat<
  (add i16:$src0, (i16 NegSubInlineConst16:$src1)),
  (V_SUB_U16_e64 $src0, NegSubInlineConst16:$src1)
>;

} // End Predicates = [isVI]

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

let AssemblerPredicates = [isSICI], DecoderNamespace = "SICI" in {

multiclass VOP2_Real_si <bits<6> op> {
  def _si :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.SI>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
}

multiclass VOP2_Real_MADK_si <bits<6> op> {
  def _si : VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.SI>,
            VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
}

multiclass VOP2_Real_e32_si <bits<6> op> {
  def _e32_si :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.SI>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>;
}

multiclass VOP2_Real_e32e64_si <bits<6> op> : VOP2_Real_e32_si<op> {
  def _e64_si :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
    VOP3e_si <{1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass VOP2be_Real_e32e64_si <bits<6> op> : VOP2_Real_e32_si<op> {
  def _e64_si :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.SI>,
    VOP3be_si <{1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

} // End AssemblerPredicates = [isSICI], DecoderNamespace = "SICI"

defm V_CNDMASK_B32        : VOP2_Real_e32e64_si <0x0>;
defm V_ADD_F32            : VOP2_Real_e32e64_si <0x3>;
defm V_SUB_F32            : VOP2_Real_e32e64_si <0x4>;
defm V_SUBREV_F32         : VOP2_Real_e32e64_si <0x5>;
defm V_MUL_LEGACY_F32     : VOP2_Real_e32e64_si <0x7>;
defm V_MUL_F32            : VOP2_Real_e32e64_si <0x8>;
defm V_MUL_I32_I24        : VOP2_Real_e32e64_si <0x9>;
defm V_MUL_HI_I32_I24     : VOP2_Real_e32e64_si <0xa>;
defm V_MUL_U32_U24        : VOP2_Real_e32e64_si <0xb>;
defm V_MUL_HI_U32_U24     : VOP2_Real_e32e64_si <0xc>;
defm V_MIN_F32            : VOP2_Real_e32e64_si <0xf>;
defm V_MAX_F32            : VOP2_Real_e32e64_si <0x10>;
defm V_MIN_I32            : VOP2_Real_e32e64_si <0x11>;
defm V_MAX_I32            : VOP2_Real_e32e64_si <0x12>;
defm V_MIN_U32            : VOP2_Real_e32e64_si <0x13>;
defm V_MAX_U32            : VOP2_Real_e32e64_si <0x14>;
defm V_LSHRREV_B32        : VOP2_Real_e32e64_si <0x16>;
defm V_ASHRREV_I32        : VOP2_Real_e32e64_si <0x18>;
defm V_LSHLREV_B32        : VOP2_Real_e32e64_si <0x1a>;
defm V_AND_B32            : VOP2_Real_e32e64_si <0x1b>;
defm V_OR_B32             : VOP2_Real_e32e64_si <0x1c>;
defm V_XOR_B32            : VOP2_Real_e32e64_si <0x1d>;
defm V_MAC_F32            : VOP2_Real_e32e64_si <0x1f>;
defm V_MADMK_F32          : VOP2_Real_MADK_si <0x20>;
defm V_MADAK_F32          : VOP2_Real_MADK_si <0x21>;
defm V_ADD_I32            : VOP2be_Real_e32e64_si <0x25>;
defm V_SUB_I32            : VOP2be_Real_e32e64_si <0x26>;
defm V_SUBREV_I32         : VOP2be_Real_e32e64_si <0x27>;
defm V_ADDC_U32           : VOP2be_Real_e32e64_si <0x28>;
defm V_SUBB_U32           : VOP2be_Real_e32e64_si <0x29>;
defm V_SUBBREV_U32        : VOP2be_Real_e32e64_si <0x2a>;

defm V_READLANE_B32       : VOP2_Real_si <0x01>;
defm V_WRITELANE_B32      : VOP2_Real_si <0x02>;

defm V_MAC_LEGACY_F32     : VOP2_Real_e32e64_si <0x6>;
defm V_MIN_LEGACY_F32     : VOP2_Real_e32e64_si <0xd>;
defm V_MAX_LEGACY_F32     : VOP2_Real_e32e64_si <0xe>;
defm V_LSHR_B32           : VOP2_Real_e32e64_si <0x15>;
defm V_ASHR_I32           : VOP2_Real_e32e64_si <0x17>;
defm V_LSHL_B32           : VOP2_Real_e32e64_si <0x19>;

defm V_BFM_B32            : VOP2_Real_e32e64_si <0x1e>;
defm V_BCNT_U32_B32       : VOP2_Real_e32e64_si <0x22>;
defm V_MBCNT_LO_U32_B32   : VOP2_Real_e32e64_si <0x23>;
defm V_MBCNT_HI_U32_B32   : VOP2_Real_e32e64_si <0x24>;
defm V_LDEXP_F32          : VOP2_Real_e32e64_si <0x2b>;
defm V_CVT_PKACCUM_U8_F32 : VOP2_Real_e32e64_si <0x2c>;
defm V_CVT_PKNORM_I16_F32 : VOP2_Real_e32e64_si <0x2d>;
defm V_CVT_PKNORM_U16_F32 : VOP2_Real_e32e64_si <0x2e>;
defm V_CVT_PKRTZ_F16_F32  : VOP2_Real_e32e64_si <0x2f>;
defm V_CVT_PK_U16_U32     : VOP2_Real_e32e64_si <0x30>;
defm V_CVT_PK_I16_I32     : VOP2_Real_e32e64_si <0x31>;


//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class VOP2_DPP <bits<6> op, VOP2_Pseudo ps, VOPProfile P = ps.Pfl> :
  VOP_DPP <ps.OpName, P> {
  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let SchedRW = ps.SchedRW;
  let hasSideEffects = ps.hasSideEffects;
  let Constraints = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  bits<8> vdst;
  bits<8> src1;
  let Inst{8-0}   = 0xfa; //dpp
  let Inst{16-9}  = !if(P.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; //encoding
}

let AssemblerPredicates = [isVI], DecoderNamespace = "VI" in {

multiclass VOP32_Real_vi <bits<10> op> {
  def _vi :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.VI>,
    VOP3e_vi<op, !cast<VOP2_Pseudo>(NAME).Pfl>;
}

multiclass VOP2_Real_MADK_vi <bits<6> op> {
  def _vi : VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
}

multiclass VOP2_Real_e32_vi <bits<6> op> {
  def _e32_vi :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.VI>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>;
}

multiclass VOP2_Real_e64_vi <bits<10> op> {
  def _e64_vi :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
    VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass Base_VOP2be_Real_e32e64_vi <bits<6> op> : VOP2_Real_e32_vi<op> {
  def _e64_vi :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
    VOP3be_vi <{0, 1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass Base_VOP2_Real_e32e64_vi <bits<6> op> :
  VOP2_Real_e32_vi<op>,
  VOP2_Real_e64_vi<{0, 1, 0, 0, op{5-0}}>;

} // End AssemblerPredicates = [isVI], DecoderNamespace = "VI"

multiclass VOP2_SDWA_Real <bits<6> op> {
  def _sdwa_vi :
    VOP_SDWA_Real <!cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa")>,
    VOP2_SDWAe <op{5-0}, !cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa").Pfl>;
}

multiclass VOP2be_Real_e32e64_vi <bits<6> op> :
  Base_VOP2be_Real_e32e64_vi<op>, VOP2_SDWA_Real<op> {
  // For now left dpp only for asm/dasm
  // TODO: add corresponding pseudo
  def _dpp : VOP2_DPP<op, !cast<VOP2_Pseudo>(NAME#"_e32")>;
}

multiclass VOP2_Real_e32e64_vi <bits<6> op> :
  Base_VOP2_Real_e32e64_vi<op>, VOP2_SDWA_Real<op> {
  // For now left dpp only for asm/dasm
  // TODO: add corresponding pseudo
  def _dpp : VOP2_DPP<op, !cast<VOP2_Pseudo>(NAME#"_e32")>;
}

defm V_CNDMASK_B32        : Base_VOP2_Real_e32e64_vi <0x0>;
defm V_ADD_F32            : VOP2_Real_e32e64_vi <0x1>;
defm V_SUB_F32            : VOP2_Real_e32e64_vi <0x2>;
defm V_SUBREV_F32         : VOP2_Real_e32e64_vi <0x3>;
defm V_MUL_LEGACY_F32     : VOP2_Real_e32e64_vi <0x4>;
defm V_MUL_F32            : VOP2_Real_e32e64_vi <0x5>;
defm V_MUL_I32_I24        : VOP2_Real_e32e64_vi <0x6>;
defm V_MUL_HI_I32_I24     : VOP2_Real_e32e64_vi <0x7>;
defm V_MUL_U32_U24        : VOP2_Real_e32e64_vi <0x8>;
defm V_MUL_HI_U32_U24     : VOP2_Real_e32e64_vi <0x9>;
defm V_MIN_F32            : VOP2_Real_e32e64_vi <0xa>;
defm V_MAX_F32            : VOP2_Real_e32e64_vi <0xb>;
defm V_MIN_I32            : VOP2_Real_e32e64_vi <0xc>;
defm V_MAX_I32            : VOP2_Real_e32e64_vi <0xd>;
defm V_MIN_U32            : VOP2_Real_e32e64_vi <0xe>;
defm V_MAX_U32            : VOP2_Real_e32e64_vi <0xf>;
defm V_LSHRREV_B32        : VOP2_Real_e32e64_vi <0x10>;
defm V_ASHRREV_I32        : VOP2_Real_e32e64_vi <0x11>;
defm V_LSHLREV_B32        : VOP2_Real_e32e64_vi <0x12>;
defm V_AND_B32            : VOP2_Real_e32e64_vi <0x13>;
defm V_OR_B32             : VOP2_Real_e32e64_vi <0x14>;
defm V_XOR_B32            : VOP2_Real_e32e64_vi <0x15>;
defm V_MAC_F32            : VOP2_Real_e32e64_vi <0x16>;
defm V_MADMK_F32          : VOP2_Real_MADK_vi <0x17>;
defm V_MADAK_F32          : VOP2_Real_MADK_vi <0x18>;
defm V_ADD_I32            : VOP2be_Real_e32e64_vi <0x19>;
defm V_SUB_I32            : VOP2be_Real_e32e64_vi <0x1a>;
defm V_SUBREV_I32         : VOP2be_Real_e32e64_vi <0x1b>;
defm V_ADDC_U32           : VOP2be_Real_e32e64_vi <0x1c>;
defm V_SUBB_U32           : VOP2be_Real_e32e64_vi <0x1d>;
defm V_SUBBREV_U32        : VOP2be_Real_e32e64_vi <0x1e>;

defm V_READLANE_B32       : VOP32_Real_vi <0x289>;
defm V_WRITELANE_B32      : VOP32_Real_vi <0x28a>;

defm V_BFM_B32            : VOP2_Real_e64_vi <0x293>;
defm V_BCNT_U32_B32       : VOP2_Real_e64_vi <0x28b>;
defm V_MBCNT_LO_U32_B32   : VOP2_Real_e64_vi <0x28c>;
defm V_MBCNT_HI_U32_B32   : VOP2_Real_e64_vi <0x28d>;
defm V_LDEXP_F32          : VOP2_Real_e64_vi <0x288>;
defm V_CVT_PKACCUM_U8_F32 : VOP2_Real_e64_vi <0x1f0>;
defm V_CVT_PKNORM_I16_F32 : VOP2_Real_e64_vi <0x294>;
defm V_CVT_PKNORM_U16_F32 : VOP2_Real_e64_vi <0x295>;
defm V_CVT_PKRTZ_F16_F32  : VOP2_Real_e64_vi <0x296>;
defm V_CVT_PK_U16_U32     : VOP2_Real_e64_vi <0x297>;
defm V_CVT_PK_I16_I32     : VOP2_Real_e64_vi <0x298>;

defm V_ADD_F16            : VOP2_Real_e32e64_vi <0x1f>;
defm V_SUB_F16            : VOP2_Real_e32e64_vi <0x20>;
defm V_SUBREV_F16         : VOP2_Real_e32e64_vi <0x21>;
defm V_MUL_F16            : VOP2_Real_e32e64_vi <0x22>;
defm V_MAC_F16            : VOP2_Real_e32e64_vi <0x23>;
defm V_MADMK_F16          : VOP2_Real_MADK_vi <0x24>;
defm V_MADAK_F16          : VOP2_Real_MADK_vi <0x25>;
defm V_ADD_U16            : VOP2_Real_e32e64_vi <0x26>;
defm V_SUB_U16            : VOP2_Real_e32e64_vi <0x27>;
defm V_SUBREV_U16         : VOP2_Real_e32e64_vi <0x28>;
defm V_MUL_LO_U16         : VOP2_Real_e32e64_vi <0x29>;
defm V_LSHLREV_B16        : VOP2_Real_e32e64_vi <0x2a>;
defm V_LSHRREV_B16        : VOP2_Real_e32e64_vi <0x2b>;
defm V_ASHRREV_I16        : VOP2_Real_e32e64_vi <0x2c>;
defm V_MAX_F16            : VOP2_Real_e32e64_vi <0x2d>;
defm V_MIN_F16            : VOP2_Real_e32e64_vi <0x2e>;
defm V_MAX_U16            : VOP2_Real_e32e64_vi <0x2f>;
defm V_MAX_I16            : VOP2_Real_e32e64_vi <0x30>;
defm V_MIN_U16            : VOP2_Real_e32e64_vi <0x31>;
defm V_MIN_I16            : VOP2_Real_e32e64_vi <0x32>;
defm V_LDEXP_F16          : VOP2_Real_e32e64_vi <0x33>;

let SubtargetPredicate = isVI in {

// Aliases to simplify matching of floating-point instructions that
// are VOP2 on SI and VOP3 on VI.
class SI2_VI3Alias <string name, Instruction inst> : InstAlias <
  name#" $dst, $src0, $src1",
  (inst VGPR_32:$dst, 0, VCSrc_f32:$src0, 0, VCSrc_f32:$src1, 0, 0)
>, PredicateControl {
  let UseInstAsmMatchConverter = 0;
  let AsmVariantName = AMDGPUAsmVariants.VOP3;
}

def : SI2_VI3Alias <"v_ldexp_f32", V_LDEXP_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkaccum_u8_f32", V_CVT_PKACCUM_U8_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_i16_f32", V_CVT_PKNORM_I16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_u16_f32", V_CVT_PKNORM_U16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkrtz_f16_f32", V_CVT_PKRTZ_F16_F32_e64_vi>;

} // End SubtargetPredicate = isVI
