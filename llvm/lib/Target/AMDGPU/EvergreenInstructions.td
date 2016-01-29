//===-- EvergreenInstructions.td - EG Instruction defs  ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TableGen definitions for instructions which are:
// - Available to Evergreen and newer VLIW4/VLIW5 GPUs
// - Available only on Evergreen family GPUs.
//
//===----------------------------------------------------------------------===//

def isEG : Predicate<
  "Subtarget->getGeneration() >= AMDGPUSubtarget::EVERGREEN && "
  "Subtarget->getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS && "
  "!Subtarget->hasCaymanISA()"
>;

def isEGorCayman : Predicate<
  "Subtarget->getGeneration() == AMDGPUSubtarget::EVERGREEN ||"
  "Subtarget->getGeneration() ==AMDGPUSubtarget::NORTHERN_ISLANDS"
>;

//===----------------------------------------------------------------------===//
// Evergreen / Cayman store instructions
//===----------------------------------------------------------------------===//

let Predicates = [isEGorCayman] in {

class CF_MEM_RAT_CACHELESS <bits<6> rat_inst, bits<4> rat_id, bits<4> mask, dag ins,
                           string name, list<dag> pattern>
    : EG_CF_RAT <0x57, rat_inst, rat_id, mask, (outs), ins,
                 "MEM_RAT_CACHELESS "#name, pattern>;

class CF_MEM_RAT <bits<6> rat_inst, bits<4> rat_id, dag ins, string name,
                  list<dag> pattern>
    : EG_CF_RAT <0x56, rat_inst, rat_id, 0xf /* mask */, (outs), ins,
                 "MEM_RAT "#name, pattern>;

class CF_MEM_RAT_STORE_TYPED<bits<1> has_eop>
    : CF_MEM_RAT <0x1, ?, (ins R600_Reg128:$rw_gpr, R600_Reg128:$index_gpr,
                           i32imm:$rat_id, InstFlag:$eop),
                  "STORE_TYPED RAT($rat_id) $rw_gpr, $index_gpr"
                               #!if(has_eop, ", $eop", ""),
                  [(int_r600_rat_store_typed R600_Reg128:$rw_gpr,
                                             R600_Reg128:$index_gpr,
                                             (i32 imm:$rat_id))]>;

def RAT_MSKOR : CF_MEM_RAT <0x11, 0,
  (ins R600_Reg128:$rw_gpr, R600_TReg32_X:$index_gpr),
  "MSKOR $rw_gpr.XW, $index_gpr",
  [(mskor_global v4i32:$rw_gpr, i32:$index_gpr)]
> {
  let eop = 0;
}

} // End let Predicates = [isEGorCayman]

//===----------------------------------------------------------------------===//
// Evergreen Only instructions
//===----------------------------------------------------------------------===//

let Predicates = [isEG] in {

def RECIP_IEEE_eg : RECIP_IEEE_Common<0x86>;
defm DIV_eg : DIV_Common<RECIP_IEEE_eg>;

def MULLO_INT_eg : MULLO_INT_Common<0x8F>;
def MULHI_INT_eg : MULHI_INT_Common<0x90>;
def MULLO_UINT_eg : MULLO_UINT_Common<0x91>;
def MULHI_UINT_eg : MULHI_UINT_Common<0x92>;
def RECIP_UINT_eg : RECIP_UINT_Common<0x94>;
def RECIPSQRT_CLAMPED_eg : RECIPSQRT_CLAMPED_Common<0x87>;
def EXP_IEEE_eg : EXP_IEEE_Common<0x81>;
def LOG_IEEE_eg : LOG_IEEE_Common<0x83>;
def RECIP_CLAMPED_eg : RECIP_CLAMPED_Common<0x84>;
def RECIPSQRT_IEEE_eg : RECIPSQRT_IEEE_Common<0x89>;
def : RsqPat<RECIPSQRT_IEEE_eg, f32>;
def SIN_eg : SIN_Common<0x8D>;
def COS_eg : COS_Common<0x8E>;

def : POW_Common <LOG_IEEE_eg, EXP_IEEE_eg, MUL>;
def : Pat<(fsqrt f32:$src), (MUL $src, (RECIPSQRT_CLAMPED_eg $src))>;

//===----------------------------------------------------------------------===//
// Memory read/write instructions
//===----------------------------------------------------------------------===//

let usesCustomInserter = 1 in {

// 32-bit store
def RAT_WRITE_CACHELESS_32_eg : CF_MEM_RAT_CACHELESS <0x2, 0, 0x1,
  (ins R600_TReg32_X:$rw_gpr, R600_TReg32_X:$index_gpr, InstFlag:$eop),
  "STORE_RAW $rw_gpr, $index_gpr, $eop",
  [(global_store i32:$rw_gpr, i32:$index_gpr)]
>;

// 64-bit store
def RAT_WRITE_CACHELESS_64_eg : CF_MEM_RAT_CACHELESS <0x2, 0, 0x3,
  (ins R600_Reg64:$rw_gpr, R600_TReg32_X:$index_gpr, InstFlag:$eop),
  "STORE_RAW $rw_gpr.XY, $index_gpr, $eop",
  [(global_store v2i32:$rw_gpr, i32:$index_gpr)]
>;

//128-bit store
def RAT_WRITE_CACHELESS_128_eg : CF_MEM_RAT_CACHELESS <0x2, 0, 0xf,
  (ins R600_Reg128:$rw_gpr, R600_TReg32_X:$index_gpr, InstFlag:$eop),
  "STORE_RAW $rw_gpr.XYZW, $index_gpr, $eop",
  [(global_store v4i32:$rw_gpr, i32:$index_gpr)]
>;

def RAT_STORE_TYPED_eg: CF_MEM_RAT_STORE_TYPED<1>;

} // End usesCustomInserter = 1

class VTX_READ_eg <string name, bits<8> buffer_id, dag outs, list<dag> pattern>
    : VTX_WORD0_eg, VTX_READ<name, buffer_id, outs, pattern> {

  // Static fields
  let VC_INST = 0;
  let FETCH_TYPE = 2;
  let FETCH_WHOLE_QUAD = 0;
  let BUFFER_ID = buffer_id;
  let SRC_REL = 0;
  // XXX: We can infer this field based on the SRC_GPR.  This would allow us
  // to store vertex addresses in any channel, not just X.
  let SRC_SEL_X = 0;

  let Inst{31-0} = Word0;
}

class VTX_READ_8_eg <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_eg <"VTX_READ_8 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {

  let MEGA_FETCH_COUNT = 1;
  let DST_SEL_X = 0;
  let DST_SEL_Y = 7;   // Masked
  let DST_SEL_Z = 7;   // Masked
  let DST_SEL_W = 7;   // Masked
  let DATA_FORMAT = 1; // FMT_8
}

class VTX_READ_16_eg <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_eg <"VTX_READ_16 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {
  let MEGA_FETCH_COUNT = 2;
  let DST_SEL_X = 0;
  let DST_SEL_Y = 7;   // Masked
  let DST_SEL_Z = 7;   // Masked
  let DST_SEL_W = 7;   // Masked
  let DATA_FORMAT = 5; // FMT_16

}

class VTX_READ_32_eg <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_eg <"VTX_READ_32 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {

  let MEGA_FETCH_COUNT = 4;
  let DST_SEL_X        = 0;
  let DST_SEL_Y        = 7;   // Masked
  let DST_SEL_Z        = 7;   // Masked
  let DST_SEL_W        = 7;   // Masked
  let DATA_FORMAT      = 0xD; // COLOR_32

  // This is not really necessary, but there were some GPU hangs that appeared
  // to be caused by ALU instructions in the next instruction group that wrote
  // to the $src_gpr registers of the VTX_READ.
  // e.g.
  // %T3_X<def> = VTX_READ_PARAM_32_eg %T2_X<kill>, 24
  // %T2_X<def> = MOV %ZERO
  //Adding this constraint prevents this from happening.
  let Constraints = "$src_gpr.ptr = $dst_gpr";
}

class VTX_READ_64_eg <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_eg <"VTX_READ_64 $dst_gpr.XY, $src_gpr", buffer_id,
                   (outs R600_Reg64:$dst_gpr), pattern> {

  let MEGA_FETCH_COUNT = 8;
  let DST_SEL_X        = 0;
  let DST_SEL_Y        = 1;
  let DST_SEL_Z        = 7;
  let DST_SEL_W        = 7;
  let DATA_FORMAT      = 0x1D; // COLOR_32_32
}

class VTX_READ_128_eg <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_eg <"VTX_READ_128 $dst_gpr.XYZW, $src_gpr", buffer_id,
                   (outs R600_Reg128:$dst_gpr), pattern> {

  let MEGA_FETCH_COUNT = 16;
  let DST_SEL_X        =  0;
  let DST_SEL_Y        =  1;
  let DST_SEL_Z        =  2;
  let DST_SEL_W        =  3;
  let DATA_FORMAT      =  0x22; // COLOR_32_32_32_32

  // XXX: Need to force VTX_READ_128 instructions to write to the same register
  // that holds its buffer address to avoid potential hangs.  We can't use
  // the same constraint as VTX_READ_32_eg, because the $src_gpr.ptr and $dst
  // registers are different sizes.
}

//===----------------------------------------------------------------------===//
// VTX Read from parameter memory space
//===----------------------------------------------------------------------===//

def VTX_READ_PARAM_8_eg : VTX_READ_8_eg <0,
  [(set i32:$dst_gpr, (load_param_exti8 ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_16_eg : VTX_READ_16_eg <0,
  [(set i32:$dst_gpr, (load_param_exti16 ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_32_eg : VTX_READ_32_eg <0,
  [(set i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_64_eg : VTX_READ_64_eg <0,
  [(set v2i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_128_eg : VTX_READ_128_eg <0,
  [(set v4i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

//===----------------------------------------------------------------------===//
// VTX Read from global memory space
//===----------------------------------------------------------------------===//

// 8-bit reads
def VTX_READ_GLOBAL_8_eg : VTX_READ_8_eg <1,
  [(set i32:$dst_gpr, (az_extloadi8_global ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_GLOBAL_16_eg : VTX_READ_16_eg <1,
  [(set i32:$dst_gpr, (az_extloadi16_global ADDRVTX_READ:$src_gpr))]
>;

// 32-bit reads
def VTX_READ_GLOBAL_32_eg : VTX_READ_32_eg <1,
  [(set i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

// 64-bit reads
def VTX_READ_GLOBAL_64_eg : VTX_READ_64_eg <1,
  [(set v2i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

// 128-bit reads
def VTX_READ_GLOBAL_128_eg : VTX_READ_128_eg <1,
  [(set v4i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

} // End Predicates = [isEG]

//===----------------------------------------------------------------------===//
// Evergreen / Cayman Instructions
//===----------------------------------------------------------------------===//

let Predicates = [isEGorCayman] in {

// Should be predicated on FeatureFP64
// def FMA_64 : R600_3OP <
//   0xA, "FMA_64",
//   [(set f64:$dst, (fma f64:$src0, f64:$src1, f64:$src2))]
// >;

// BFE_UINT - bit_extract, an optimization for mask and shift
// Src0 = Input
// Src1 = Offset
// Src2 = Width
//
// bit_extract = (Input << (32 - Offset - Width)) >> (32 - Width)
//
// Example Usage:
// (Offset, Width)
//
// (0, 8)  = (Input << 24) >> 24 = (Input &  0xff)       >> 0
// (8, 8)  = (Input << 16) >> 24 = (Input &  0xffff)     >> 8
// (16, 8) = (Input <<  8) >> 24 = (Input &  0xffffff)   >> 16
// (24, 8) = (Input <<  0) >> 24 = (Input &  0xffffffff) >> 24
def BFE_UINT_eg : R600_3OP <0x4, "BFE_UINT",
  [(set i32:$dst, (AMDGPUbfe_u32 i32:$src0, i32:$src1, i32:$src2))],
  VecALU
>;

def BFE_INT_eg : R600_3OP <0x5, "BFE_INT",
  [(set i32:$dst, (AMDGPUbfe_i32 i32:$src0, i32:$src1, i32:$src2))],
  VecALU
>;

def : BFEPattern <BFE_UINT_eg, MOV_IMM_I32>;

def BFI_INT_eg : R600_3OP <0x06, "BFI_INT",
  [(set i32:$dst, (AMDGPUbfi i32:$src0, i32:$src1, i32:$src2))],
  VecALU
>;

def : Pat<(i32 (sext_inreg i32:$src, i1)),
  (BFE_INT_eg i32:$src, (i32 ZERO), (i32 ONE_INT))>;
def : Pat<(i32 (sext_inreg i32:$src, i8)),
  (BFE_INT_eg i32:$src, (i32 ZERO), (MOV_IMM_I32 8))>;
def : Pat<(i32 (sext_inreg i32:$src, i16)),
  (BFE_INT_eg i32:$src, (i32 ZERO), (MOV_IMM_I32 16))>;

defm : BFIPatterns <BFI_INT_eg, MOV_IMM_I32, R600_Reg64>;

def BFM_INT_eg : R600_2OP <0xA0, "BFM_INT",
  [(set i32:$dst, (AMDGPUbfm i32:$src0, i32:$src1))],
  VecALU
>;

def MULADD_UINT24_eg : R600_3OP <0x10, "MULADD_UINT24",
  [(set i32:$dst, (AMDGPUmad_u24 i32:$src0, i32:$src1, i32:$src2))], VecALU
>;

def : UMad24Pat<MULADD_UINT24_eg>;

def BIT_ALIGN_INT_eg : R600_3OP <0xC, "BIT_ALIGN_INT", [], VecALU>;
def : ROTRPattern <BIT_ALIGN_INT_eg>;
def MULADD_eg : MULADD_Common<0x14>;
def MULADD_IEEE_eg : MULADD_IEEE_Common<0x18>;
def FMA_eg : FMA_Common<0x7>;
def ASHR_eg : ASHR_Common<0x15>;
def LSHR_eg : LSHR_Common<0x16>;
def LSHL_eg : LSHL_Common<0x17>;
def CNDE_eg : CNDE_Common<0x19>;
def CNDGT_eg : CNDGT_Common<0x1A>;
def CNDGE_eg : CNDGE_Common<0x1B>;
def MUL_LIT_eg : MUL_LIT_Common<0x1F>;
def LOG_CLAMPED_eg : LOG_CLAMPED_Common<0x82>;
def MUL_UINT24_eg : R600_2OP <0xB5, "MUL_UINT24",
  [(set i32:$dst, (AMDGPUmul_u24 i32:$src0, i32:$src1))], VecALU
>;
def DOT4_eg : DOT4_Common<0xBE>;
defm CUBE_eg : CUBE_Common<0xC0>;

def BCNT_INT : R600_1OP_Helper <0xAA, "BCNT_INT", ctpop, VecALU>;

def ADDC_UINT : R600_2OP_Helper <0x52, "ADDC_UINT", AMDGPUcarry>;
def SUBB_UINT : R600_2OP_Helper <0x53, "SUBB_UINT", AMDGPUborrow>;

def FFBH_UINT : R600_1OP_Helper <0xAB, "FFBH_UINT", AMDGPUffbh_u32, VecALU>;
def FFBL_INT : R600_1OP_Helper <0xAC, "FFBL_INT", cttz_zero_undef, VecALU>;

let hasSideEffects = 1 in {
  def MOVA_INT_eg : R600_1OP <0xCC, "MOVA_INT", [], VecALU>;
}

def FLT_TO_INT_eg : FLT_TO_INT_Common<0x50> {
  let Pattern = [];
  let Itinerary = AnyALU;
}

def INT_TO_FLT_eg : INT_TO_FLT_Common<0x9B>;

def FLT_TO_UINT_eg : FLT_TO_UINT_Common<0x9A> {
  let Pattern = [];
}

def UINT_TO_FLT_eg : UINT_TO_FLT_Common<0x9C>;

def GROUP_BARRIER : InstR600 <
    (outs), (ins), "  GROUP_BARRIER", [(int_AMDGPU_barrier_local), (int_AMDGPU_barrier_global)], AnyALU>,
    R600ALU_Word0,
    R600ALU_Word1_OP2 <0x54> {

  let dst = 0;
  let dst_rel = 0;
  let src0 = 0;
  let src0_rel = 0;
  let src0_neg = 0;
  let src0_abs = 0;
  let src1 = 0;
  let src1_rel = 0;
  let src1_neg = 0;
  let src1_abs = 0;
  let write = 0;
  let omod = 0;
  let clamp = 0;
  let last = 1;
  let bank_swizzle = 0;
  let pred_sel = 0;
  let update_exec_mask = 0;
  let update_pred = 0;

  let Inst{31-0}  = Word0;
  let Inst{63-32} = Word1;

  let ALUInst = 1;
}

def : Pat <
	(int_AMDGPU_barrier_global),
	(GROUP_BARRIER)
>;

//===----------------------------------------------------------------------===//
// LDS Instructions
//===----------------------------------------------------------------------===//
class R600_LDS  <bits<6> op, dag outs, dag ins, string asm,
                 list<dag> pattern = []> :

    InstR600 <outs, ins, asm, pattern, XALU>,
    R600_ALU_LDS_Word0,
    R600LDS_Word1 {

  bits<6>  offset = 0;
  let lds_op = op;

  let Word1{27} = offset{0};
  let Word1{12} = offset{1};
  let Word1{28} = offset{2};
  let Word1{31} = offset{3};
  let Word0{12} = offset{4};
  let Word0{25} = offset{5};


  let Inst{31-0}  = Word0;
  let Inst{63-32} = Word1;

  let ALUInst = 1;
  let HasNativeOperands = 1;
  let UseNamedOperandTable = 1;
}

class R600_LDS_1A <bits<6> lds_op, string name, list<dag> pattern> : R600_LDS <
  lds_op,
  (outs R600_Reg32:$dst),
  (ins R600_Reg32:$src0, REL:$src0_rel, SEL:$src0_sel,
       LAST:$last, R600_Pred:$pred_sel,
       BANK_SWIZZLE:$bank_swizzle),
  "  "#name#" $last OQAP, $src0$src0_rel $pred_sel",
  pattern
  > {

  let src1 = 0;
  let src1_rel = 0;
  let src2 = 0;
  let src2_rel = 0;

  let usesCustomInserter = 1;
  let LDS_1A = 1;
  let DisableEncoding = "$dst";
}

class R600_LDS_1A1D <bits<6> lds_op, dag outs, string name, list<dag> pattern,
                     string dst =""> :
    R600_LDS <
  lds_op, outs,
  (ins R600_Reg32:$src0, REL:$src0_rel, SEL:$src0_sel,
       R600_Reg32:$src1, REL:$src1_rel, SEL:$src1_sel,
       LAST:$last, R600_Pred:$pred_sel,
       BANK_SWIZZLE:$bank_swizzle),
  "  "#name#" $last "#dst#"$src0$src0_rel, $src1$src1_rel, $pred_sel",
  pattern
  > {

  field string BaseOp;

  let src2 = 0;
  let src2_rel = 0;
  let LDS_1A1D = 1;
}

class R600_LDS_1A1D_NORET <bits<6> lds_op, string name, list<dag> pattern> :
    R600_LDS_1A1D <lds_op, (outs), name, pattern> {
  let BaseOp = name;
}

class R600_LDS_1A1D_RET <bits<6> lds_op, string name, list<dag> pattern> :
    R600_LDS_1A1D <lds_op,  (outs R600_Reg32:$dst), name##"_RET", pattern, "OQAP, "> {

  let BaseOp = name;
  let usesCustomInserter = 1;
  let DisableEncoding = "$dst";
}

class R600_LDS_1A2D <bits<6> lds_op, dag outs, string name, list<dag> pattern,
                     string dst =""> :
    R600_LDS <
  lds_op, outs,
  (ins R600_Reg32:$src0, REL:$src0_rel, SEL:$src0_sel,
       R600_Reg32:$src1, REL:$src1_rel, SEL:$src1_sel,
       R600_Reg32:$src2, REL:$src2_rel, SEL:$src2_sel,
       LAST:$last, R600_Pred:$pred_sel, BANK_SWIZZLE:$bank_swizzle),
  "  "#name# "$last "#dst#"$src0$src0_rel, $src1$src1_rel, $src2$src2_rel, $pred_sel",
  pattern> {

  field string BaseOp;

  let LDS_1A1D = 0;
  let LDS_1A2D = 1;
}

class R600_LDS_1A2D_NORET <bits<6> lds_op, string name, list<dag> pattern> :
    R600_LDS_1A2D <lds_op, (outs), name, pattern> {
  let BaseOp = name;
}

class R600_LDS_1A2D_RET <bits<6> lds_op, string name, list<dag> pattern> :
    R600_LDS_1A2D <lds_op, (outs R600_Reg32:$dst), name, pattern> {

  let BaseOp = name;
  let usesCustomInserter = 1;
  let DisableEncoding = "$dst";
}

def LDS_ADD : R600_LDS_1A1D_NORET <0x0, "LDS_ADD", [] >;
def LDS_SUB : R600_LDS_1A1D_NORET <0x1, "LDS_SUB", [] >;
def LDS_AND : R600_LDS_1A1D_NORET <0x9, "LDS_AND", [] >;
def LDS_OR : R600_LDS_1A1D_NORET <0xa, "LDS_OR", [] >;
def LDS_XOR : R600_LDS_1A1D_NORET <0xb, "LDS_XOR", [] >;
def LDS_WRXCHG: R600_LDS_1A1D_NORET <0xd, "LDS_WRXCHG", [] >;
def LDS_CMPST: R600_LDS_1A2D_NORET <0x10, "LDS_CMPST", [] >;
def LDS_MIN_INT : R600_LDS_1A1D_NORET <0x5, "LDS_MIN_INT", [] >;
def LDS_MAX_INT : R600_LDS_1A1D_NORET <0x6, "LDS_MAX_INT", [] >;
def LDS_MIN_UINT : R600_LDS_1A1D_NORET <0x7, "LDS_MIN_UINT", [] >;
def LDS_MAX_UINT : R600_LDS_1A1D_NORET <0x8, "LDS_MAX_UINT", [] >;
def LDS_WRITE : R600_LDS_1A1D_NORET <0xD, "LDS_WRITE",
  [(local_store (i32 R600_Reg32:$src1), R600_Reg32:$src0)]
>;
def LDS_BYTE_WRITE : R600_LDS_1A1D_NORET<0x12, "LDS_BYTE_WRITE",
  [(truncstorei8_local i32:$src1, i32:$src0)]
>;
def LDS_SHORT_WRITE : R600_LDS_1A1D_NORET<0x13, "LDS_SHORT_WRITE",
  [(truncstorei16_local i32:$src1, i32:$src0)]
>;
def LDS_ADD_RET : R600_LDS_1A1D_RET <0x20, "LDS_ADD",
  [(set i32:$dst, (atomic_load_add_local i32:$src0, i32:$src1))]
>;
def LDS_SUB_RET : R600_LDS_1A1D_RET <0x21, "LDS_SUB",
  [(set i32:$dst, (atomic_load_sub_local i32:$src0, i32:$src1))]
>;
def LDS_AND_RET : R600_LDS_1A1D_RET <0x29, "LDS_AND",
  [(set i32:$dst, (atomic_load_and_local i32:$src0, i32:$src1))]
>;
def LDS_OR_RET : R600_LDS_1A1D_RET <0x2a, "LDS_OR",
  [(set i32:$dst, (atomic_load_or_local i32:$src0, i32:$src1))]
>;
def LDS_XOR_RET : R600_LDS_1A1D_RET <0x2b, "LDS_XOR",
  [(set i32:$dst, (atomic_load_xor_local i32:$src0, i32:$src1))]
>;
def LDS_MIN_INT_RET : R600_LDS_1A1D_RET <0x25, "LDS_MIN_INT",
  [(set i32:$dst, (atomic_load_min_local i32:$src0, i32:$src1))]
>;
def LDS_MAX_INT_RET : R600_LDS_1A1D_RET <0x26, "LDS_MAX_INT",
  [(set i32:$dst, (atomic_load_max_local i32:$src0, i32:$src1))]
>;
def LDS_MIN_UINT_RET : R600_LDS_1A1D_RET <0x27, "LDS_MIN_UINT",
  [(set i32:$dst, (atomic_load_umin_local i32:$src0, i32:$src1))]
>;
def LDS_MAX_UINT_RET : R600_LDS_1A1D_RET <0x28, "LDS_MAX_UINT",
  [(set i32:$dst, (atomic_load_umax_local i32:$src0, i32:$src1))]
>;
def LDS_WRXCHG_RET : R600_LDS_1A1D_RET <0x2d, "LDS_WRXCHG",
  [(set i32:$dst, (atomic_swap_local i32:$src0, i32:$src1))]
>;
def LDS_CMPST_RET : R600_LDS_1A2D_RET <0x30, "LDS_CMPST",
  [(set i32:$dst, (atomic_cmp_swap_32_local i32:$src0, i32:$src1, i32:$src2))]
>;
def LDS_READ_RET : R600_LDS_1A <0x32, "LDS_READ_RET",
  [(set (i32 R600_Reg32:$dst), (local_load R600_Reg32:$src0))]
>;
def LDS_BYTE_READ_RET : R600_LDS_1A <0x36, "LDS_BYTE_READ_RET",
  [(set i32:$dst, (sextloadi8_local i32:$src0))]
>;
def LDS_UBYTE_READ_RET : R600_LDS_1A <0x37, "LDS_UBYTE_READ_RET",
  [(set i32:$dst, (az_extloadi8_local i32:$src0))]
>;
def LDS_SHORT_READ_RET : R600_LDS_1A <0x38, "LDS_SHORT_READ_RET",
  [(set i32:$dst, (sextloadi16_local i32:$src0))]
>;
def LDS_USHORT_READ_RET : R600_LDS_1A <0x39, "LDS_USHORT_READ_RET",
  [(set i32:$dst, (az_extloadi16_local i32:$src0))]
>;

// TRUNC is used for the FLT_TO_INT instructions to work around a
// perceived problem where the rounding modes are applied differently
// depending on the instruction and the slot they are in.
// See:
// https://bugs.freedesktop.org/show_bug.cgi?id=50232
// Mesa commit: a1a0974401c467cb86ef818f22df67c21774a38c
//
// XXX: Lowering SELECT_CC will sometimes generate fp_to_[su]int nodes,
// which do not need to be truncated since the fp values are 0.0f or 1.0f.
// We should look into handling these cases separately.
def : Pat<(fp_to_sint f32:$src0), (FLT_TO_INT_eg (TRUNC $src0))>;

def : Pat<(fp_to_uint f32:$src0), (FLT_TO_UINT_eg (TRUNC $src0))>;

// SHA-256 Patterns
def : SHA256MaPattern <BFI_INT_eg, XOR_INT>;

def EG_ExportSwz : ExportSwzInst {
  let Word1{19-16} = 0; // BURST_COUNT
  let Word1{20} = 0; // VALID_PIXEL_MODE
  let Word1{21} = eop;
  let Word1{29-22} = inst;
  let Word1{30} = 0; // MARK
  let Word1{31} = 1; // BARRIER
}
defm : ExportPattern<EG_ExportSwz, 83>;

def EG_ExportBuf : ExportBufInst {
  let Word1{19-16} = 0; // BURST_COUNT
  let Word1{20} = 0; // VALID_PIXEL_MODE
  let Word1{21} = eop;
  let Word1{29-22} = inst;
  let Word1{30} = 0; // MARK
  let Word1{31} = 1; // BARRIER
}
defm : SteamOutputExportPattern<EG_ExportBuf, 0x40, 0x41, 0x42, 0x43>;

def CF_TC_EG : CF_CLAUSE_EG<1, (ins i32imm:$ADDR, i32imm:$COUNT),
  "TEX $COUNT @$ADDR"> {
  let POP_COUNT = 0;
}
def CF_VC_EG : CF_CLAUSE_EG<2, (ins i32imm:$ADDR, i32imm:$COUNT),
  "VTX $COUNT @$ADDR"> {
  let POP_COUNT = 0;
}
def WHILE_LOOP_EG : CF_CLAUSE_EG<6, (ins i32imm:$ADDR),
  "LOOP_START_DX10 @$ADDR"> {
  let POP_COUNT = 0;
  let COUNT = 0;
}
def END_LOOP_EG : CF_CLAUSE_EG<5, (ins i32imm:$ADDR), "END_LOOP @$ADDR"> {
  let POP_COUNT = 0;
  let COUNT = 0;
}
def LOOP_BREAK_EG : CF_CLAUSE_EG<9, (ins i32imm:$ADDR),
  "LOOP_BREAK @$ADDR"> {
  let POP_COUNT = 0;
  let COUNT = 0;
}
def CF_CONTINUE_EG : CF_CLAUSE_EG<8, (ins i32imm:$ADDR),
  "CONTINUE @$ADDR"> {
  let POP_COUNT = 0;
  let COUNT = 0;
}
def CF_JUMP_EG : CF_CLAUSE_EG<10, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "JUMP @$ADDR POP:$POP_COUNT"> {
  let COUNT = 0;
}
def CF_PUSH_EG : CF_CLAUSE_EG<11, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
                              "PUSH @$ADDR POP:$POP_COUNT"> {
  let COUNT = 0;
}
def CF_ELSE_EG : CF_CLAUSE_EG<13, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "ELSE @$ADDR POP:$POP_COUNT"> {
  let COUNT = 0;
}
def CF_CALL_FS_EG : CF_CLAUSE_EG<19, (ins), "CALL_FS"> {
  let ADDR = 0;
  let COUNT = 0;
  let POP_COUNT = 0;
}
def POP_EG : CF_CLAUSE_EG<14, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "POP @$ADDR POP:$POP_COUNT"> {
  let COUNT = 0;
}
def CF_END_EG :  CF_CLAUSE_EG<0, (ins), "CF_END"> {
  let COUNT = 0;
  let POP_COUNT = 0;
  let ADDR = 0;
  let END_OF_PROGRAM = 1;
}

} // End Predicates = [isEGorCayman]
