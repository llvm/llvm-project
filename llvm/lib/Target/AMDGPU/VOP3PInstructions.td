//===-- VOP3PInstructions.td - Vector Instruction Definitions -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOP3P Classes
//===----------------------------------------------------------------------===//

class VOP3PInst<string OpName, VOPProfile P,
                SDPatternOperator node = null_frag,
                bit HasExplicitClamp = 0> :
  VOP3P_Pseudo<OpName, P,
    !if(P.HasModifiers, getVOP3PModPat<P, node, HasExplicitClamp>.ret, getVOP3Pat<P, node>.ret)
>;

// Non-packed instructions that use the VOP3P encoding.
// VOP3 neg/abs and VOP3P opsel/opsel_hi modifiers are allowed.
class VOP3_VOP3PInst<string OpName, VOPProfile P, bit UseTiedOutput = 0,
                     SDPatternOperator node = null_frag> :
  VOP3P_Pseudo<OpName, P> {
  // These operands are only sort of f16 operands. Depending on
  // op_sel_hi, these may be interpreted as f32. The inline immediate
  // values are really f16 converted to f32, so we treat these as f16
  // operands.
  let InOperandList =
    !con(
      !con(
        (ins FP16InputMods:$src0_modifiers, VCSrc_f16:$src0,
             FP16InputMods:$src1_modifiers, VCSrc_f16:$src1,
             FP16InputMods:$src2_modifiers, VCSrc_f16:$src2),
         // FIXME: clampmod0 misbehaves with the non-default vdst_in
         // following it. For now workaround this by requiring clamp
         // in tied patterns. This should use undef_tied_input, but it
         // seems underdeveloped and doesn't apply the right register
         // class constraints.
         !if(UseTiedOutput, (ins clampmod:$clamp, VGPR_32:$vdst_in),
                            (ins clampmod0:$clamp))),
         (ins op_sel:$op_sel, op_sel_hi:$op_sel_hi));

  let Constraints = !if(UseTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(UseTiedOutput, "$vdst_in", "");
  let AsmOperands =
    " $vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$op_sel$op_sel_hi$clamp";
}

let isCommutable = 1 in {
def V_PK_MAD_I16 : VOP3PInst<"v_pk_mad_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16_V2I16>>;
def V_PK_MAD_U16 : VOP3PInst<"v_pk_mad_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16_V2I16>>;

let FPDPRounding = 1 in {
def V_PK_FMA_F16 : VOP3PInst<"v_pk_fma_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16_V2F16>, any_fma>;
def V_PK_ADD_F16 : VOP3PInst<"v_pk_add_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, any_fadd>;
def V_PK_MUL_F16 : VOP3PInst<"v_pk_mul_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, any_fmul>;
} // End FPDPRounding = 1
def V_PK_MAX_F16 : VOP3PInst<"v_pk_max_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fmaxnum_like>;
def V_PK_MIN_F16 : VOP3PInst<"v_pk_min_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fminnum_like>;

def V_PK_ADD_U16 : VOP3PInst<"v_pk_add_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, add>;
def V_PK_ADD_I16 : VOP3PInst<"v_pk_add_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>>;
def V_PK_MUL_LO_U16 : VOP3PInst<"v_pk_mul_lo_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, mul>;

def V_PK_MIN_I16 : VOP3PInst<"v_pk_min_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smin>;
def V_PK_MIN_U16 : VOP3PInst<"v_pk_min_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umin>;
def V_PK_MAX_I16 : VOP3PInst<"v_pk_max_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smax>;
def V_PK_MAX_U16 : VOP3PInst<"v_pk_max_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umax>;
}

def V_PK_SUB_U16 : VOP3PInst<"v_pk_sub_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>>;
def V_PK_SUB_I16 : VOP3PInst<"v_pk_sub_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, sub>;

def V_PK_LSHLREV_B16 : VOP3PInst<"v_pk_lshlrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshl_rev>;
def V_PK_ASHRREV_I16 : VOP3PInst<"v_pk_ashrrev_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, ashr_rev>;
def V_PK_LSHRREV_B16 : VOP3PInst<"v_pk_lshrrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshr_rev>;


let SubtargetPredicate = HasVOP3PInsts in {

// Undo sub x, c -> add x, -c canonicalization since c is more likely
// an inline immediate than -c.
// The constant will be emitted as a mov, and folded later.
// TODO: We could directly encode the immediate now
def : GCNPat<
  (add (v2i16 (VOP3PMods v2i16:$src0, i32:$src0_modifiers)), NegSubInlineConstV216:$src1),
  (V_PK_SUB_U16 $src0_modifiers, $src0, SRCMODS.OP_SEL_1, NegSubInlineConstV216:$src1)
>;

// Integer operations with clamp bit set.
class VOP3PSatPat<SDPatternOperator pat, Instruction inst> : GCNPat<
  (pat (v2i16 (VOP3PMods v2i16:$src0, i32:$src0_modifiers)),
       (v2i16 (VOP3PMods v2i16:$src1, i32:$src1_modifiers))),
  (inst $src0_modifiers, $src0, $src1_modifiers, $src1, DSTCLAMP.ENABLE)
>;

def : VOP3PSatPat<uaddsat, V_PK_ADD_U16>;
def : VOP3PSatPat<saddsat, V_PK_ADD_I16>;
def : VOP3PSatPat<usubsat, V_PK_SUB_U16>;
def : VOP3PSatPat<ssubsat, V_PK_SUB_I16>;
} // End SubtargetPredicate = HasVOP3PInsts

multiclass MadFmaMixPats<SDPatternOperator fma_like,
                         Instruction mix_inst,
                         Instruction mixlo_inst,
                         Instruction mixhi_inst> {
  def : GCNPat <
    (f16 (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                            (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                            (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers))))),
    (mixlo_inst $src0_modifiers, $src0,
                $src1_modifiers, $src1,
                $src2_modifiers, $src2,
                DSTCLAMP.NONE,
                (i32 (IMPLICIT_DEF)))
  >;

  // FIXME: Special case handling for maxhi (especially for clamp)
  // because dealing with the write to high half of the register is
  // difficult.
  def : GCNPat <
    (build_vector f16:$elt0, (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                                                (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                                                (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers))))),
    (v2f16 (mixhi_inst $src0_modifiers, $src0,
                       $src1_modifiers, $src1,
                       $src2_modifiers, $src2,
                       DSTCLAMP.NONE,
                       $elt0))
  >;

  def : GCNPat <
    (build_vector
      f16:$elt0,
      (AMDGPUclamp (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                                      (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                                      (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers)))))),
    (v2f16 (mixhi_inst $src0_modifiers, $src0,
                       $src1_modifiers, $src1,
                       $src2_modifiers, $src2,
                       DSTCLAMP.ENABLE,
                       $elt0))
  >;

  def : GCNPat <
    (AMDGPUclamp (build_vector
      (fpround (fma_like (f32 (VOP3PMadMixMods f16:$lo_src0, i32:$lo_src0_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$lo_src1, i32:$lo_src1_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$lo_src2, i32:$lo_src2_modifiers)))),
      (fpround (fma_like (f32 (VOP3PMadMixMods f16:$hi_src0, i32:$hi_src0_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$hi_src1, i32:$hi_src1_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$hi_src2, i32:$hi_src2_modifiers)))))),
    (v2f16 (mixhi_inst $hi_src0_modifiers, $hi_src0,
                       $hi_src1_modifiers, $hi_src1,
                       $hi_src2_modifiers, $hi_src2,
                       DSTCLAMP.ENABLE,
                       (mixlo_inst $lo_src0_modifiers, $lo_src0,
                                   $lo_src1_modifiers, $lo_src1,
                                   $lo_src2_modifiers, $lo_src2,
                                   DSTCLAMP.ENABLE,
                                   (i32 (IMPLICIT_DEF)))))
  >;
}

let SubtargetPredicate = HasMadMixInsts in {

// These are VOP3a-like opcodes which accept no omod.
// Size of src arguments (16/32) is controlled by op_sel.
// For 16-bit src arguments their location (hi/lo) are controlled by op_sel_hi.
let isCommutable = 1, mayRaiseFPException = 0 in {
def V_MAD_MIX_F32 : VOP3_VOP3PInst<"v_mad_mix_f32", VOP3_Profile<VOP_F32_F16_F16_F16, VOP3_OPSEL>>;

let FPDPRounding = 1 in {
// Clamp modifier is applied after conversion to f16.
def V_MAD_MIXLO_F16 : VOP3_VOP3PInst<"v_mad_mixlo_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, 1>;

let ClampLo = 0, ClampHi = 1 in {
def V_MAD_MIXHI_F16 : VOP3_VOP3PInst<"v_mad_mixhi_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, 1>;
}
} // End FPDPRounding = 1
}

defm : MadFmaMixPats<fmad, V_MAD_MIX_F32, V_MAD_MIXLO_F16, V_MAD_MIXHI_F16>;
} // End SubtargetPredicate = HasMadMixInsts


// Essentially the same as the mad_mix versions
let SubtargetPredicate = HasFmaMixInsts in {
let isCommutable = 1 in {
def V_FMA_MIX_F32 : VOP3_VOP3PInst<"v_fma_mix_f32", VOP3_Profile<VOP_F32_F16_F16_F16, VOP3_OPSEL>>;

let FPDPRounding = 1 in {
// Clamp modifier is applied after conversion to f16.
def V_FMA_MIXLO_F16 : VOP3_VOP3PInst<"v_fma_mixlo_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, 1>;

let ClampLo = 0, ClampHi = 1 in {
def V_FMA_MIXHI_F16 : VOP3_VOP3PInst<"v_fma_mixhi_f16", VOP3_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL>, 1>;
}
} // End FPDPRounding = 1
}

defm : MadFmaMixPats<fma, V_FMA_MIX_F32, V_FMA_MIXLO_F16, V_FMA_MIXHI_F16>;
}

// Defines patterns that extract signed 4bit from each Idx[0].
foreach Idx = [[0,28],[4,24],[8,20],[12,16],[16,12],[20,8],[24,4]] in
  def ExtractSigned4bit_#Idx[0] : PatFrag<(ops node:$src),
                                          (sra (shl node:$src, (i32 Idx[1])), (i32 28))>;

// Defines code pattern that extracts U(unsigned/signed) 4/8bit from FromBitIndex.
class Extract<int FromBitIndex, int BitMask, bit U>: PatFrag<
  (ops node:$src),
  !if (!or (!and (!eq (BitMask, 255), !eq (FromBitIndex, 24)), !eq (FromBitIndex, 28)), // last element
       !if (U, (srl node:$src, (i32 FromBitIndex)), (sra node:$src, (i32 FromBitIndex))),
       !if (!eq (FromBitIndex, 0), // first element
            !if (U, (and node:$src, (i32 BitMask)),
                 !if (!eq (BitMask, 15), (!cast<PatFrag>("ExtractSigned4bit_"#FromBitIndex) node:$src),
                                         (sext_inreg node:$src, i8))),
            !if (U, (and (srl node:$src, (i32 FromBitIndex)), (i32 BitMask)),
                 !if (!eq (BitMask, 15), (!cast<PatFrag>("ExtractSigned4bit_"#FromBitIndex) node:$src),
                      (sext_inreg (srl node:$src, (i32 FromBitIndex)), i8)))))>;


foreach Type = ["I", "U"] in
  foreach Index = 0-3 in {
    // Defines patterns that extract each Index'ed 8bit from an unsigned
    // 32bit scalar value;
    def Type#Index#"_8bit" : Extract<!shl(Index, 3), 255, !if (!eq (Type, "U"), 1, 0)>;

    // Defines multiplication patterns where the multiplication is happening on each
    // Index'ed 8bit of a 32bit scalar value.

    def Mul#Type#_Elt#Index : PatFrag<
      (ops node:$src0, node:$src1),
      (!cast<HasOneUseBinOp>(!if (!eq (Type, "I"), AMDGPUmul_i24_oneuse, AMDGPUmul_u24_oneuse))
                            (!cast<Extract>(Type#Index#"_8bit") node:$src0),
                            (!cast<Extract>(Type#Index#"_8bit") node:$src1))>;
  }

// Different variants of dot8 patterns cause a huge increase in the compile time.
// Define non-associative/commutative add/mul to prevent permutation in the dot8
// pattern.
def NonACAdd        : SDNode<"ISD::ADD"       , SDTIntBinOp>;
def NonACAdd_oneuse : HasOneUseBinOp<NonACAdd>;

def NonACAMDGPUmul_u24        : SDNode<"AMDGPUISD::MUL_U24"       , SDTIntBinOp>;
def NonACAMDGPUmul_u24_oneuse : HasOneUseBinOp<NonACAMDGPUmul_u24>;

def NonACAMDGPUmul_i24        : SDNode<"AMDGPUISD::MUL_I24"       , SDTIntBinOp>;
def NonACAMDGPUmul_i24_oneuse : HasOneUseBinOp<NonACAMDGPUmul_i24>;

foreach Type = ["I", "U"] in
  foreach Index = 0-7 in {
    // Defines patterns that extract each Index'ed 4bit from an unsigned
    // 32bit scalar value;
    def Type#Index#"_4bit" : Extract<!shl(Index, 2), 15, !if (!eq (Type, "U"), 1, 0)>;

    // Defines multiplication patterns where the multiplication is happening on each
    // Index'ed 8bit of a 32bit scalar value.
    def Mul#Type#Index#"_4bit" : PatFrag<
      (ops node:$src0, node:$src1),
      (!cast<HasOneUseBinOp>(!if (!eq (Type, "I"), NonACAMDGPUmul_i24_oneuse, NonACAMDGPUmul_u24_oneuse))
                             (!cast<Extract>(Type#Index#"_4bit") node:$src0),
                             (!cast<Extract>(Type#Index#"_4bit") node:$src1))>;
  }

class UDot2Pat<Instruction Inst> : GCNPat <
  (add (add_oneuse (AMDGPUmul_u24_oneuse (srl i32:$src0, (i32 16)),
                                         (srl i32:$src1, (i32 16))), i32:$src2),
       (AMDGPUmul_u24_oneuse (and i32:$src0, (i32 65535)),
                             (and i32:$src1, (i32 65535)))
   ),
  (Inst (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))> {
  let SubtargetPredicate = !cast<VOP_Pseudo>(Inst).SubtargetPredicate;
}

class SDot2Pat<Instruction Inst> : GCNPat <
  (add (add_oneuse (AMDGPUmul_i24_oneuse (sra i32:$src0, (i32 16)),
                                         (sra i32:$src1, (i32 16))), i32:$src2),
       (AMDGPUmul_i24_oneuse (sext_inreg i32:$src0, i16),
                             (sext_inreg i32:$src1, i16))),
  (Inst (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))> {
  let SubtargetPredicate = !cast<VOP_Pseudo>(Inst).SubtargetPredicate;
}

let IsDOT = 1 in {
let SubtargetPredicate = HasDot2Insts in {

def V_DOT2_F32_F16 : VOP3PInst<"v_dot2_f32_f16",
  VOP3_Profile<VOP_F32_V2F16_V2F16_F32>,
  AMDGPUfdot2, 1/*ExplicitClamp*/>;
def V_DOT2_I32_I16 : VOP3PInst<"v_dot2_i32_i16",
  VOP3_Profile<VOP_I32_V2I16_V2I16_I32>, int_amdgcn_sdot2, 1>;
def V_DOT2_U32_U16 : VOP3PInst<"v_dot2_u32_u16",
  VOP3_Profile<VOP_I32_V2I16_V2I16_I32>, int_amdgcn_udot2, 1>;
def V_DOT4_U32_U8  : VOP3PInst<"v_dot4_u32_u8",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_udot4, 1>;
def V_DOT8_U32_U4  : VOP3PInst<"v_dot8_u32_u4",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_udot8, 1>;

} // End SubtargetPredicate = HasDot2Insts

let SubtargetPredicate = HasDot1Insts in {

def V_DOT4_I32_I8  : VOP3PInst<"v_dot4_i32_i8",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_sdot4, 1>;
def V_DOT8_I32_I4  : VOP3PInst<"v_dot8_i32_i4",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_sdot8, 1>;

} // End SubtargetPredicate = HasDot1Insts
} // End let IsDOT = 1

def : UDot2Pat<V_DOT2_U32_U16>;
def : SDot2Pat<V_DOT2_I32_I16>;

foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT4_"#Type#"32_"#Type#8).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((i32 i32:$src2), [0, 1, 2, 3], lhs, y,
                      (add_oneuse lhs, (!cast<PatFrag>("Mul"#Type#"_Elt"#y) i32:$src0, i32:$src1)))),
    (!cast<VOP3PInst>("V_DOT4_"#Type#"32_"#Type#8) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT8_"#Type#"32_"#Type#4).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((add_oneuse i32:$src2, (!cast<PatFrag>("Mul"#Type#"0_4bit") i32:$src0, i32:$src1)),
                      [1, 2, 3, 4, 5, 6, 7], lhs, y,
                      (NonACAdd_oneuse lhs, (!cast<PatFrag>("Mul"#Type#y#"_4bit") i32:$src0, i32:$src1)))),
    (!cast<VOP3PInst>("V_DOT8_"#Type#"32_"#Type#4) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

// Different variants of dot8 code-gen dag patterns are not generated through table-gen due to a huge increase
// in the compile time. Directly handle the pattern generated by the FE here.
foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT8_"#Type#"32_"#Type#4).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((add_oneuse i32:$src2, (!cast<PatFrag>("Mul"#Type#"0_4bit") i32:$src0, i32:$src1)),
                      [7, 1, 2, 3, 4, 5, 6], lhs, y,
                      (NonACAdd_oneuse lhs, (!cast<PatFrag>("Mul"#Type#y#"_4bit") i32:$src0, i32:$src1)))),
    (!cast<VOP3PInst>("V_DOT8_"#Type#"32_"#Type#4) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

def ADst_32   : VOPDstOperand<AGPR_32>;
def ADst_128  : VOPDstOperand<AReg_128>;
def ADst_512  : VOPDstOperand<AReg_512>;
def ADst_1024 : VOPDstOperand<AReg_1024>;

def VOPProfileAccRead : VOP3_Profile<VOP_I32_I32, VOP3_MAI> {
  let Src0RC64 = ARegSrc_32;
}

def VOPProfileAccWrite : VOP3_Profile<VOP_I32_I32, VOP3_MAI> {
  let DstRC = ADst_32;
  let Src0RC64 = VISrc_b32;
}

class VOPProfileMAI<VOPProfile P, RegisterOperand _SrcRC, RegisterOperand _DstRC,
                    RegisterOperand SrcABRC = AVSrc_32>
  : VOP3_Profile<P, VOP3_MAI> {
  let DstRC = _DstRC;
  let Src0RC64 = SrcABRC;
  let Src1RC64 = SrcABRC;
  let Src2RC64 = _SrcRC;
  let HasOpSel = 0;
  let HasClamp = 0;
  let HasModifiers = 0;
  let Asm64 = " $vdst, $src0, $src1, $src2$cbsz$abid$blgp";
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, cbsz:$cbsz, abid:$abid, blgp:$blgp);
}

def VOPProfileMAI_F32_F32_X4    : VOPProfileMAI<VOP_V4F32_F32_F32_V4F32,       AISrc_128_f32,  ADst_128>;
def VOPProfileMAI_F32_F32_X16   : VOPProfileMAI<VOP_V16F32_F32_F32_V16F32,     AISrc_512_f32,  ADst_512>;
def VOPProfileMAI_F32_F32_X32   : VOPProfileMAI<VOP_V32F32_F32_F32_V32F32,     AISrc_1024_f32, ADst_1024>;
def VOPProfileMAI_I32_I32_X4    : VOPProfileMAI<VOP_V4I32_I32_I32_V4I32,       AISrc_128_b32,  ADst_128>;
def VOPProfileMAI_I32_I32_X16   : VOPProfileMAI<VOP_V16I32_I32_I32_V16I32,     AISrc_512_b32,  ADst_512>;
def VOPProfileMAI_I32_I32_X32   : VOPProfileMAI<VOP_V32I32_I32_I32_V32I32,     AISrc_1024_b32, ADst_1024>;
def VOPProfileMAI_F32_V2I16_X4  : VOPProfileMAI<VOP_V4F32_V2I16_V2I16_V4F32,   AISrc_128_b32,  ADst_128>;
def VOPProfileMAI_F32_V2I16_X16 : VOPProfileMAI<VOP_V16F32_V2I16_V2I16_V16F32, AISrc_512_b32,  ADst_512>;
def VOPProfileMAI_F32_V2I16_X32 : VOPProfileMAI<VOP_V32F32_V2I16_V2I16_V32F32, AISrc_1024_b32, ADst_1024>;
def VOPProfileMAI_F32_V4F16_X4  : VOPProfileMAI<VOP_V4F32_V4F16_V4F16_V4F32,   AISrc_128_b32,  ADst_128,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X16 : VOPProfileMAI<VOP_V16F32_V4F16_V4F16_V16F32, AISrc_512_b32,  ADst_512,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X32 : VOPProfileMAI<VOP_V32F32_V4F16_V4F16_V32F32, AISrc_1024_b32, ADst_1024, AVSrc_64>;

let Predicates = [HasMAIInsts] in {

let isAsCheapAsAMove = 1, isReMaterializable = 1 in {
def V_ACCVGPR_READ_B32  : VOP3Inst<"v_accvgpr_read_b32",  VOPProfileAccRead>;
def V_ACCVGPR_WRITE_B32 : VOP3Inst<"v_accvgpr_write_b32", VOPProfileAccWrite> {
  let isMoveImm = 1;
}
}

// FP32 denorm mode is respected, rounding mode is not. Exceptions are not supported.
let isConvergent = 1, mayRaiseFPException = 0, ReadsModeReg = 1 in {
def V_MFMA_F32_4X4X1F32    : VOP3Inst<"v_mfma_f32_4x4x1f32",    VOPProfileMAI_F32_F32_X4,    int_amdgcn_mfma_f32_4x4x1f32>;
def V_MFMA_F32_4X4X4F16    : VOP3Inst<"v_mfma_f32_4x4x4f16",    VOPProfileMAI_F32_V4F16_X4,  int_amdgcn_mfma_f32_4x4x4f16>;
def V_MFMA_I32_4X4X4I8     : VOP3Inst<"v_mfma_i32_4x4x4i8",     VOPProfileMAI_I32_I32_X4,    int_amdgcn_mfma_i32_4x4x4i8>;
def V_MFMA_F32_4X4X2BF16   : VOP3Inst<"v_mfma_f32_4x4x2bf16",   VOPProfileMAI_F32_V2I16_X4,  int_amdgcn_mfma_f32_4x4x2bf16>;
def V_MFMA_F32_16X16X1F32  : VOP3Inst<"v_mfma_f32_16x16x1f32",  VOPProfileMAI_F32_F32_X16,   int_amdgcn_mfma_f32_16x16x1f32>;
def V_MFMA_F32_16X16X4F32  : VOP3Inst<"v_mfma_f32_16x16x4f32",  VOPProfileMAI_F32_F32_X4,    int_amdgcn_mfma_f32_16x16x4f32>;
def V_MFMA_F32_16X16X4F16  : VOP3Inst<"v_mfma_f32_16x16x4f16",  VOPProfileMAI_F32_V4F16_X16, int_amdgcn_mfma_f32_16x16x4f16>;
def V_MFMA_F32_16X16X16F16 : VOP3Inst<"v_mfma_f32_16x16x16f16", VOPProfileMAI_F32_V4F16_X4,  int_amdgcn_mfma_f32_16x16x16f16>;
def V_MFMA_I32_16X16X4I8   : VOP3Inst<"v_mfma_i32_16x16x4i8",   VOPProfileMAI_I32_I32_X16,   int_amdgcn_mfma_i32_16x16x4i8>;
def V_MFMA_I32_16X16X16I8  : VOP3Inst<"v_mfma_i32_16x16x16i8",  VOPProfileMAI_I32_I32_X4,    int_amdgcn_mfma_i32_16x16x16i8>;
def V_MFMA_F32_16X16X2BF16 : VOP3Inst<"v_mfma_f32_16x16x2bf16", VOPProfileMAI_F32_V2I16_X16, int_amdgcn_mfma_f32_16x16x2bf16>;
def V_MFMA_F32_16X16X8BF16 : VOP3Inst<"v_mfma_f32_16x16x8bf16", VOPProfileMAI_F32_V2I16_X4,  int_amdgcn_mfma_f32_16x16x8bf16>;
def V_MFMA_F32_32X32X1F32  : VOP3Inst<"v_mfma_f32_32x32x1f32",  VOPProfileMAI_F32_F32_X32,   int_amdgcn_mfma_f32_32x32x1f32>;
def V_MFMA_F32_32X32X2F32  : VOP3Inst<"v_mfma_f32_32x32x2f32",  VOPProfileMAI_F32_F32_X16,   int_amdgcn_mfma_f32_32x32x2f32>;
def V_MFMA_F32_32X32X4F16  : VOP3Inst<"v_mfma_f32_32x32x4f16",  VOPProfileMAI_F32_V4F16_X32, int_amdgcn_mfma_f32_32x32x4f16>;
def V_MFMA_F32_32X32X8F16  : VOP3Inst<"v_mfma_f32_32x32x8f16",  VOPProfileMAI_F32_V4F16_X16, int_amdgcn_mfma_f32_32x32x8f16>;
def V_MFMA_I32_32X32X4I8   : VOP3Inst<"v_mfma_i32_32x32x4i8",   VOPProfileMAI_I32_I32_X32,   int_amdgcn_mfma_i32_32x32x4i8>;
def V_MFMA_I32_32X32X8I8   : VOP3Inst<"v_mfma_i32_32x32x8i8",   VOPProfileMAI_I32_I32_X16,   int_amdgcn_mfma_i32_32x32x8i8>;
def V_MFMA_F32_32X32X2BF16 : VOP3Inst<"v_mfma_f32_32x32x2bf16", VOPProfileMAI_F32_V2I16_X32, int_amdgcn_mfma_f32_32x32x2bf16>;
def V_MFMA_F32_32X32X4BF16 : VOP3Inst<"v_mfma_f32_32x32x4bf16", VOPProfileMAI_F32_V2I16_X16, int_amdgcn_mfma_f32_32x32x4bf16>;
} // End isConvergent = 1, mayRaiseFPException = 0, ReadsModeReg = 1

} // End SubtargetPredicate = HasMAIInsts

def : MnemonicAlias<"v_accvgpr_read",  "v_accvgpr_read_b32">;
def : MnemonicAlias<"v_accvgpr_write", "v_accvgpr_write_b32">;

multiclass VOP3P_Real_vi<bits<10> op> {
  def _vi : VOP3P_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3Pe <op, !cast<VOP3_Pseudo>(NAME).Pfl> {
    let AssemblerPredicate = HasVOP3PInsts;
    let DecoderNamespace = "GFX8";
  }
}

multiclass VOP3P_Real_MAI<bits<10> op> {
  def _vi : VOP3P_Real<!cast<VOP3_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP3Pe_MAI <op, !cast<VOP3_Pseudo>(NAME).Pfl> {
    let AssemblerPredicate = HasMAIInsts;
    let DecoderNamespace = "GFX8";
  }
}

defm V_PK_MAD_I16 : VOP3P_Real_vi <0x380>;
defm V_PK_MUL_LO_U16 : VOP3P_Real_vi <0x381>;
defm V_PK_ADD_I16 : VOP3P_Real_vi <0x382>;
defm V_PK_SUB_I16 : VOP3P_Real_vi <0x383>;
defm V_PK_LSHLREV_B16 : VOP3P_Real_vi <0x384>;
defm V_PK_LSHRREV_B16 : VOP3P_Real_vi <0x385>;
defm V_PK_ASHRREV_I16 : VOP3P_Real_vi <0x386>;
defm V_PK_MAX_I16 : VOP3P_Real_vi <0x387>;
defm V_PK_MIN_I16 : VOP3P_Real_vi <0x388>;
defm V_PK_MAD_U16 : VOP3P_Real_vi <0x389>;

defm V_PK_ADD_U16 : VOP3P_Real_vi <0x38a>;
defm V_PK_SUB_U16 : VOP3P_Real_vi <0x38b>;
defm V_PK_MAX_U16 : VOP3P_Real_vi <0x38c>;
defm V_PK_MIN_U16 : VOP3P_Real_vi <0x38d>;
defm V_PK_FMA_F16 : VOP3P_Real_vi <0x38e>;
defm V_PK_ADD_F16 : VOP3P_Real_vi <0x38f>;
defm V_PK_MUL_F16 : VOP3P_Real_vi <0x390>;
defm V_PK_MIN_F16 : VOP3P_Real_vi <0x391>;
defm V_PK_MAX_F16 : VOP3P_Real_vi <0x392>;


let SubtargetPredicate = HasMadMixInsts in {
defm V_MAD_MIX_F32 : VOP3P_Real_vi <0x3a0>;
defm V_MAD_MIXLO_F16 : VOP3P_Real_vi <0x3a1>;
defm V_MAD_MIXHI_F16 : VOP3P_Real_vi <0x3a2>;
}

let SubtargetPredicate = HasFmaMixInsts in {
let DecoderNamespace = "GFX9_DL" in {
// The mad_mix instructions were renamed and their behaviors changed,
// but the opcode stayed the same so we need to put these in a
// different DecoderNamespace to avoid the ambiguity.
defm V_FMA_MIX_F32 : VOP3P_Real_vi <0x3a0>;
defm V_FMA_MIXLO_F16 : VOP3P_Real_vi <0x3a1>;
defm V_FMA_MIXHI_F16 : VOP3P_Real_vi <0x3a2>;
}
}


let SubtargetPredicate = HasDot2Insts in {

defm V_DOT2_F32_F16 : VOP3P_Real_vi <0x3a3>;
defm V_DOT2_I32_I16 : VOP3P_Real_vi <0x3a6>;
defm V_DOT2_U32_U16 : VOP3P_Real_vi <0x3a7>;
defm V_DOT4_U32_U8  : VOP3P_Real_vi <0x3a9>;
defm V_DOT8_U32_U4  : VOP3P_Real_vi <0x3ab>;

} // End SubtargetPredicate = HasDot2Insts

let SubtargetPredicate = HasDot1Insts in {

defm V_DOT4_I32_I8  : VOP3P_Real_vi <0x3a8>;
defm V_DOT8_I32_I4  : VOP3P_Real_vi <0x3aa>;

} // End SubtargetPredicate = HasDot1Insts

let SubtargetPredicate = HasMAIInsts in {

defm V_ACCVGPR_READ_B32  : VOP3P_Real_MAI <0x3d8>;
defm V_ACCVGPR_WRITE_B32 : VOP3P_Real_MAI <0x3d9>;
defm V_MFMA_F32_32X32X1F32  : VOP3P_Real_MAI <0x3c0>;
defm V_MFMA_F32_16X16X1F32  : VOP3P_Real_MAI <0x3c1>;
defm V_MFMA_F32_4X4X1F32    : VOP3P_Real_MAI <0x3c2>;
defm V_MFMA_F32_32X32X2F32  : VOP3P_Real_MAI <0x3c4>;
defm V_MFMA_F32_16X16X4F32  : VOP3P_Real_MAI <0x3c5>;
defm V_MFMA_F32_32X32X4F16  : VOP3P_Real_MAI <0x3c8>;
defm V_MFMA_F32_16X16X4F16  : VOP3P_Real_MAI <0x3c9>;
defm V_MFMA_F32_4X4X4F16    : VOP3P_Real_MAI <0x3ca>;
defm V_MFMA_F32_32X32X8F16  : VOP3P_Real_MAI <0x3cc>;
defm V_MFMA_F32_16X16X16F16 : VOP3P_Real_MAI <0x3cd>;
defm V_MFMA_I32_32X32X4I8   : VOP3P_Real_MAI <0x3d0>;
defm V_MFMA_I32_16X16X4I8   : VOP3P_Real_MAI <0x3d1>;
defm V_MFMA_I32_4X4X4I8     : VOP3P_Real_MAI <0x3d2>;
defm V_MFMA_I32_32X32X8I8   : VOP3P_Real_MAI <0x3d4>;
defm V_MFMA_I32_16X16X16I8  : VOP3P_Real_MAI <0x3d5>;
defm V_MFMA_F32_32X32X2BF16 : VOP3P_Real_MAI <0x3e8>;
defm V_MFMA_F32_16X16X2BF16 : VOP3P_Real_MAI <0x3e9>;
defm V_MFMA_F32_4X4X2BF16   : VOP3P_Real_MAI <0x3eb>;
defm V_MFMA_F32_32X32X4BF16 : VOP3P_Real_MAI <0x3ec>;
defm V_MFMA_F32_16X16X8BF16 : VOP3P_Real_MAI <0x3ed>;

} // End SubtargetPredicate = HasMAIInsts

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10" in {
  multiclass VOP3P_Real_gfx10<bits<10> op> {
    def _gfx10 : VOP3P_Real<!cast<VOP3P_Pseudo>(NAME), SIEncodingFamily.GFX10>,
                 VOP3Pe_gfx10 <op, !cast<VOP3P_Pseudo>(NAME).Pfl>;
  }
} // End AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10"

defm V_PK_MAD_I16     : VOP3P_Real_gfx10<0x000>;
defm V_PK_MUL_LO_U16  : VOP3P_Real_gfx10<0x001>;
defm V_PK_ADD_I16     : VOP3P_Real_gfx10<0x002>;
defm V_PK_SUB_I16     : VOP3P_Real_gfx10<0x003>;
defm V_PK_LSHLREV_B16 : VOP3P_Real_gfx10<0x004>;
defm V_PK_LSHRREV_B16 : VOP3P_Real_gfx10<0x005>;
defm V_PK_ASHRREV_I16 : VOP3P_Real_gfx10<0x006>;
defm V_PK_MAX_I16     : VOP3P_Real_gfx10<0x007>;
defm V_PK_MIN_I16     : VOP3P_Real_gfx10<0x008>;
defm V_PK_MAD_U16     : VOP3P_Real_gfx10<0x009>;
defm V_PK_ADD_U16     : VOP3P_Real_gfx10<0x00a>;
defm V_PK_SUB_U16     : VOP3P_Real_gfx10<0x00b>;
defm V_PK_MAX_U16     : VOP3P_Real_gfx10<0x00c>;
defm V_PK_MIN_U16     : VOP3P_Real_gfx10<0x00d>;
defm V_PK_FMA_F16     : VOP3P_Real_gfx10<0x00e>;
defm V_PK_ADD_F16     : VOP3P_Real_gfx10<0x00f>;
defm V_PK_MUL_F16     : VOP3P_Real_gfx10<0x010>;
defm V_PK_MIN_F16     : VOP3P_Real_gfx10<0x011>;
defm V_PK_MAX_F16     : VOP3P_Real_gfx10<0x012>;
defm V_FMA_MIX_F32    : VOP3P_Real_gfx10<0x020>;
defm V_FMA_MIXLO_F16  : VOP3P_Real_gfx10<0x021>;
defm V_FMA_MIXHI_F16  : VOP3P_Real_gfx10<0x022>;

let SubtargetPredicate = HasDot2Insts in {

defm V_DOT2_F32_F16 : VOP3P_Real_gfx10 <0x013>;
defm V_DOT2_I32_I16 : VOP3P_Real_gfx10 <0x014>;
defm V_DOT2_U32_U16 : VOP3P_Real_gfx10 <0x015>;
defm V_DOT4_U32_U8  : VOP3P_Real_gfx10 <0x017>;
defm V_DOT8_U32_U4  : VOP3P_Real_gfx10 <0x019>;

} // End SubtargetPredicate = HasDot2Insts

let SubtargetPredicate = HasDot1Insts in {

defm V_DOT4_I32_I8  : VOP3P_Real_gfx10 <0x016>;
defm V_DOT8_I32_I4  : VOP3P_Real_gfx10 <0x018>;

} // End SubtargetPredicate = HasDot1Insts
