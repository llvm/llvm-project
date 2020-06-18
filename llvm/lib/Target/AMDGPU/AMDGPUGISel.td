//===-- AMDGPUGIsel.td - AMDGPU GlobalISel Patterns---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This files contains patterns that should only be used by GlobalISel.  For
// example patterns for V_* instructions that have S_* equivalents.
// SelectionDAG does not support selecting V_* instructions.
//===----------------------------------------------------------------------===//

include "AMDGPU.td"
include "AMDGPUCombine.td"

def sd_vsrc0 : ComplexPattern<i32, 1, "">;
def gi_vsrc0 :
    GIComplexOperandMatcher<s32, "selectVSRC0">,
    GIComplexPatternEquiv<sd_vsrc0>;

def sd_vcsrc : ComplexPattern<i32, 1, "">;
def gi_vcsrc :
    GIComplexOperandMatcher<s32, "selectVCSRC">,
    GIComplexPatternEquiv<sd_vcsrc>;

def gi_vop3mods0 :
    GIComplexOperandMatcher<s32, "selectVOP3Mods0">,
    GIComplexPatternEquiv<VOP3Mods0>;

def gi_vop3mods :
    GIComplexOperandMatcher<s32, "selectVOP3Mods">,
    GIComplexPatternEquiv<VOP3Mods>;

def gi_vop3_no_mods :
    GIComplexOperandMatcher<s32, "selectVOP3NoMods">,
    GIComplexPatternEquiv<VOP3NoMods>;

def gi_vop3mods_nnan :
    GIComplexOperandMatcher<s32, "selectVOP3Mods_nnan">,
    GIComplexPatternEquiv<VOP3Mods_nnan>;

def gi_vop3omods :
    GIComplexOperandMatcher<s32, "selectVOP3OMods">,
    GIComplexPatternEquiv<VOP3OMods>;

def gi_vop3pmods :
    GIComplexOperandMatcher<s32, "selectVOP3PMods">,
    GIComplexPatternEquiv<VOP3PMods>;

def gi_vop3opselmods :
    GIComplexOperandMatcher<s32, "selectVOP3OpSelMods">,
    GIComplexPatternEquiv<VOP3OpSelMods>;

def gi_smrd_imm :
    GIComplexOperandMatcher<s64, "selectSmrdImm">,
    GIComplexPatternEquiv<SMRDImm>;

def gi_smrd_imm32 :
    GIComplexOperandMatcher<s64, "selectSmrdImm32">,
    GIComplexPatternEquiv<SMRDImm32>;

def gi_smrd_sgpr :
    GIComplexOperandMatcher<s64, "selectSmrdSgpr">,
    GIComplexPatternEquiv<SMRDSgpr>;

// FIXME: Why are the atomic versions separated?
def gi_flat_offset :
    GIComplexOperandMatcher<s64, "selectFlatOffset">,
    GIComplexPatternEquiv<FLATOffset>;
def gi_flat_offset_signed :
    GIComplexOperandMatcher<s64, "selectFlatOffsetSigned">,
    GIComplexPatternEquiv<FLATOffsetSigned>;
def gi_flat_atomic :
    GIComplexOperandMatcher<s64, "selectFlatOffset">,
    GIComplexPatternEquiv<FLATAtomic>;
def gi_flat_signed_atomic :
    GIComplexOperandMatcher<s64, "selectFlatOffsetSigned">,
    GIComplexPatternEquiv<FLATSignedAtomic>;

def gi_mubuf_scratch_offset :
    GIComplexOperandMatcher<s32, "selectMUBUFScratchOffset">,
    GIComplexPatternEquiv<MUBUFScratchOffset>;
def gi_mubuf_scratch_offen :
    GIComplexOperandMatcher<s32, "selectMUBUFScratchOffen">,
    GIComplexPatternEquiv<MUBUFScratchOffen>;

def gi_ds_1addr_1offset :
    GIComplexOperandMatcher<s32, "selectDS1Addr1Offset">,
    GIComplexPatternEquiv<DS1Addr1Offset>;

def gi_ds_64bit_4byte_aligned :
    GIComplexOperandMatcher<s64, "selectDS64Bit4ByteAligned">,
    GIComplexPatternEquiv<DS64Bit4ByteAligned>;

def gi_mubuf_addr64 :
    GIComplexOperandMatcher<s64, "selectMUBUFAddr64">,
    GIComplexPatternEquiv<MUBUFAddr64>;

def gi_mubuf_offset :
    GIComplexOperandMatcher<s64, "selectMUBUFOffset">,
    GIComplexPatternEquiv<MUBUFOffset>;

def gi_mubuf_addr64_atomic :
    GIComplexOperandMatcher<s64, "selectMUBUFAddr64Atomic">,
    GIComplexPatternEquiv<MUBUFAddr64Atomic>;

def gi_mubuf_offset_atomic :
    GIComplexOperandMatcher<s64, "selectMUBUFOffsetAtomic">,
    GIComplexPatternEquiv<MUBUFOffsetAtomic>;

def gi_smrd_buffer_imm :
    GIComplexOperandMatcher<s64, "selectSMRDBufferImm">,
    GIComplexPatternEquiv<SMRDBufferImm>;

def gi_smrd_buffer_imm32 :
    GIComplexOperandMatcher<s64, "selectSMRDBufferImm32">,
    GIComplexPatternEquiv<SMRDBufferImm32>;

// Separate load nodes are defined to glue m0 initialization in
// SelectionDAG. The GISel selector can just insert m0 initialization
// directly before before selecting a glue-less load, so hide this
// distinction.

def : GINodeEquiv<G_LOAD, AMDGPUld_glue> {
  let CheckMMOIsNonAtomic = 1;
}

def : GINodeEquiv<G_STORE, AMDGPUst_glue> {
  let CheckMMOIsNonAtomic = 1;
}

def : GINodeEquiv<G_LOAD, AMDGPUatomic_ld_glue> {
  bit CheckMMOIsAtomic = 1;
}



def : GINodeEquiv<G_ATOMIC_CMPXCHG, atomic_cmp_swap_glue>;
def : GINodeEquiv<G_ATOMICRMW_XCHG, atomic_swap_glue>;
def : GINodeEquiv<G_ATOMICRMW_ADD, atomic_load_add_glue>;
def : GINodeEquiv<G_ATOMICRMW_SUB, atomic_load_sub_glue>;
def : GINodeEquiv<G_ATOMICRMW_AND, atomic_load_and_glue>;
def : GINodeEquiv<G_ATOMICRMW_OR, atomic_load_or_glue>;
def : GINodeEquiv<G_ATOMICRMW_XOR, atomic_load_xor_glue>;
def : GINodeEquiv<G_ATOMICRMW_MIN, atomic_load_min_glue>;
def : GINodeEquiv<G_ATOMICRMW_MAX, atomic_load_max_glue>;
def : GINodeEquiv<G_ATOMICRMW_UMIN, atomic_load_umin_glue>;
def : GINodeEquiv<G_ATOMICRMW_UMAX, atomic_load_umax_glue>;
def : GINodeEquiv<G_ATOMICRMW_FADD, atomic_load_fadd_glue>;

def : GINodeEquiv<G_AMDGPU_FFBH_U32, AMDGPUffbh_u32_impl>;
def : GINodeEquiv<G_AMDGPU_FMIN_LEGACY, AMDGPUfmin_legacy>;
def : GINodeEquiv<G_AMDGPU_FMAX_LEGACY, AMDGPUfmax_legacy>;
def : GINodeEquiv<G_AMDGPU_RCP_IFLAG, AMDGPUrcp_iflag>;

def : GINodeEquiv<G_AMDGPU_CVT_F32_UBYTE0, AMDGPUcvt_f32_ubyte0>;
def : GINodeEquiv<G_AMDGPU_CVT_F32_UBYTE1, AMDGPUcvt_f32_ubyte1>;
def : GINodeEquiv<G_AMDGPU_CVT_F32_UBYTE2, AMDGPUcvt_f32_ubyte2>;
def : GINodeEquiv<G_AMDGPU_CVT_F32_UBYTE3, AMDGPUcvt_f32_ubyte3>;

def : GINodeEquiv<G_AMDGPU_ATOMIC_CMPXCHG, AMDGPUatomic_cmp_swap>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD, SIbuffer_load>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_USHORT, SIbuffer_load_ushort>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_UBYTE, SIbuffer_load_ubyte>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_SSHORT, SIbuffer_load_short>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_SBYTE, SIbuffer_load_byte>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_FORMAT, SIbuffer_load_format>;
def : GINodeEquiv<G_AMDGPU_BUFFER_LOAD_FORMAT_D16, SIbuffer_load_format_d16>;
def : GINodeEquiv<G_AMDGPU_TBUFFER_LOAD_FORMAT, SItbuffer_load>;
def : GINodeEquiv<G_AMDGPU_TBUFFER_LOAD_FORMAT_D16, SItbuffer_load_d16>;
def : GINodeEquiv<G_AMDGPU_BUFFER_STORE, SIbuffer_store>;
def : GINodeEquiv<G_AMDGPU_BUFFER_STORE_SHORT, SIbuffer_store_short>;
def : GINodeEquiv<G_AMDGPU_BUFFER_STORE_BYTE, SIbuffer_store_byte>;
def : GINodeEquiv<G_AMDGPU_BUFFER_STORE_FORMAT, SIbuffer_store_format>;
def : GINodeEquiv<G_AMDGPU_BUFFER_STORE_FORMAT_D16, SIbuffer_store_format_d16>;
def : GINodeEquiv<G_AMDGPU_TBUFFER_STORE_FORMAT, SItbuffer_store>;
def : GINodeEquiv<G_AMDGPU_TBUFFER_STORE_FORMAT_D16, SItbuffer_store_d16>;

// FIXME: Check MMO is atomic
def : GINodeEquiv<G_AMDGPU_ATOMIC_INC, SIatomic_inc>;
def : GINodeEquiv<G_AMDGPU_ATOMIC_DEC, SIatomic_dec>;
def : GINodeEquiv<G_AMDGPU_ATOMIC_INC, atomic_inc_glue>;
def : GINodeEquiv<G_AMDGPU_ATOMIC_DEC, atomic_dec_glue>;

def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_SWAP, SIbuffer_atomic_swap>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_ADD, SIbuffer_atomic_add>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_SUB, SIbuffer_atomic_sub>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_SMIN, SIbuffer_atomic_smin>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_UMIN, SIbuffer_atomic_umin>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_SMAX, SIbuffer_atomic_smax>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_UMAX, SIbuffer_atomic_umax>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_AND, SIbuffer_atomic_and>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_OR, SIbuffer_atomic_or>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_XOR, SIbuffer_atomic_xor>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_INC, SIbuffer_atomic_inc>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_DEC, SIbuffer_atomic_dec>;
def : GINodeEquiv<G_AMDGPU_BUFFER_ATOMIC_CMPSWAP, SIbuffer_atomic_cmpswap>;
def : GINodeEquiv<G_AMDGPU_S_BUFFER_LOAD, SIsbuffer_load>;

class GISelSop2Pat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt SReg_32:$src0), (src1_vt SReg_32:$src1))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

class GISelVop2Pat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt (sd_vsrc0 src0_vt:$src0)), (src1_vt VGPR_32:$src1))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

class GISelVop2CommutePat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src1_vt VGPR_32:$src1), (src0_vt (sd_vsrc0 src0_vt:$src0)))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

class GISelVop3Pat2 <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt (sd_vcsrc src0_vt:$src0)), (src1_vt (sd_vcsrc src1_vt:$src1)))),
  (inst src0_vt:$src0, src1_vt:$src1)
>;

class GISelVop3Pat2CommutePat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt>   : GCNPat <

  (dst_vt (node (src0_vt (sd_vcsrc src0_vt:$src0)), (src1_vt (sd_vcsrc src1_vt:$src1)))),
  (inst src0_vt:$src1, src1_vt:$src0)
>;

class GISelVop3Pat2ModsPat <
  SDPatternOperator node,
  Instruction inst,
  ValueType dst_vt,
  ValueType src0_vt = dst_vt, ValueType src1_vt = src0_vt> : GCNPat <

  (dst_vt (node (src0_vt (VOP3Mods0 src0_vt:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omods)),
                (src1_vt (VOP3Mods src1_vt:$src1, i32:$src1_modifiers)))),
  (inst i32:$src0_modifiers, src0_vt:$src0,
        i32:$src1_modifiers, src1_vt:$src1, $clamp, $omods)
>;

multiclass GISelVop2IntrPat <
  SDPatternOperator node, Instruction inst,
  ValueType dst_vt, ValueType src_vt = dst_vt> {

  def : GISelVop2Pat <node, inst, dst_vt, src_vt>;

  // FIXME: Intrinsics aren't marked as commutable, so we need to add an explicit
  // pattern to handle commuting.  This is another reason why legalizing to a
  // generic machine instruction may be better that matching the intrinsic
  // directly.
  def : GISelVop2CommutePat <node, inst, dst_vt, src_vt>;
}

// Since GlobalISel is more flexible then SelectionDAG, I think we can get
// away with adding patterns for integer types and not legalizing all
// loads and stores to vector types.  This should help simplify the load/store
// legalization.
foreach Ty = [i64, p0, p1, p4] in {
  defm : SMRD_Pattern <"S_LOAD_DWORDX2",  Ty>;
}

def gi_as_i32timm : GICustomOperandRenderer<"renderTruncTImm32">,
  GISDNodeXFormEquiv<as_i32timm>;

def gi_as_i16timm : GICustomOperandRenderer<"renderTruncTImm16">,
  GISDNodeXFormEquiv<as_i16timm>;

def gi_as_i8timm : GICustomOperandRenderer<"renderTruncTImm8">,
  GISDNodeXFormEquiv<as_i8timm>;

def gi_as_i1timm : GICustomOperandRenderer<"renderTruncTImm1">,
  GISDNodeXFormEquiv<as_i1timm>;

def gi_NegateImm : GICustomOperandRenderer<"renderNegateImm">,
  GISDNodeXFormEquiv<NegateImm>;

def gi_bitcast_fpimm_to_i32 : GICustomOperandRenderer<"renderBitcastImm">,
  GISDNodeXFormEquiv<bitcast_fpimm_to_i32>;

def gi_IMMPopCount : GICustomOperandRenderer<"renderPopcntImm">,
  GISDNodeXFormEquiv<IMMPopCount>;

def gi_extract_glc : GICustomOperandRenderer<"renderExtractGLC">,
  GISDNodeXFormEquiv<extract_glc>;

def gi_extract_slc : GICustomOperandRenderer<"renderExtractSLC">,
  GISDNodeXFormEquiv<extract_slc>;

def gi_extract_dlc : GICustomOperandRenderer<"renderExtractDLC">,
  GISDNodeXFormEquiv<extract_dlc>;

def gi_extract_swz : GICustomOperandRenderer<"renderExtractSWZ">,
  GISDNodeXFormEquiv<extract_swz>;
