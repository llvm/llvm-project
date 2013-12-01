//===-- R600Instructions.td - R600 Instruction defs  -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// R600 Tablegen instruction definitions
//
//===----------------------------------------------------------------------===//

include "R600Intrinsics.td"
include "R600InstrFormats.td"

class InstR600ISA <dag outs, dag ins, string asm, list<dag> pattern> :
    InstR600 <outs, ins, asm, pattern, NullALU> {

  let Namespace = "AMDGPU";
}

def MEMxi : Operand<iPTR> {
  let MIOperandInfo = (ops R600_TReg32_X:$ptr, i32imm:$index);
  let PrintMethod = "printMemOperand";
}

def MEMrr : Operand<iPTR> {
  let MIOperandInfo = (ops R600_Reg32:$ptr, R600_Reg32:$index);
}

// Operands for non-registers

class InstFlag<string PM = "printOperand", int Default = 0>
    : OperandWithDefaultOps <i32, (ops (i32 Default))> {
  let PrintMethod = PM;
}

// src_sel for ALU src operands, see also ALU_CONST, ALU_PARAM registers
def SEL : OperandWithDefaultOps <i32, (ops (i32 -1))> {
  let PrintMethod = "printSel";
}
def BANK_SWIZZLE : OperandWithDefaultOps <i32, (ops (i32 0))> {
  let PrintMethod = "printBankSwizzle";
}

def LITERAL : InstFlag<"printLiteral">;

def WRITE : InstFlag <"printWrite", 1>;
def OMOD : InstFlag <"printOMOD">;
def REL : InstFlag <"printRel">;
def CLAMP : InstFlag <"printClamp">;
def NEG : InstFlag <"printNeg">;
def ABS : InstFlag <"printAbs">;
def UEM : InstFlag <"printUpdateExecMask">;
def UP : InstFlag <"printUpdatePred">;

// XXX: The r600g finalizer in Mesa expects last to be one in most cases.
// Once we start using the packetizer in this backend we should have this
// default to 0.
def LAST : InstFlag<"printLast", 1>;
def RSel : Operand<i32> {
  let PrintMethod = "printRSel";
}
def CT: Operand<i32> {
  let PrintMethod = "printCT";
}

def FRAMEri : Operand<iPTR> {
  let MIOperandInfo = (ops R600_Reg32:$ptr, i32imm:$index);
}

def ADDRParam : ComplexPattern<i32, 2, "SelectADDRParam", [], []>;
def ADDRDWord : ComplexPattern<i32, 1, "SelectADDRDWord", [], []>;
def ADDRVTX_READ : ComplexPattern<i32, 2, "SelectADDRVTX_READ", [], []>;
def ADDRGA_CONST_OFFSET : ComplexPattern<i32, 1, "SelectGlobalValueConstantOffset", [], []>;
def ADDRGA_VAR_OFFSET : ComplexPattern<i32, 2, "SelectGlobalValueVariableOffset", [], []>;


def R600_Pred : PredicateOperand<i32, (ops R600_Predicate),
                                     (ops PRED_SEL_OFF)>;


let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {

// Class for instructions with only one source register.
// If you add new ins to this instruction, make sure they are listed before
// $literal, because the backend currently assumes that the last operand is
// a literal.  Also be sure to update the enum R600Op1OperandIndex::ROI in
// R600Defines.h, R600InstrInfo::buildDefaultInstruction(),
// and R600InstrInfo::getOperandIdx().
class R600_1OP <bits<11> inst, string opName, list<dag> pattern,
                InstrItinClass itin = AnyALU> :
    InstR600 <(outs R600_Reg32:$dst),
              (ins WRITE:$write, OMOD:$omod, REL:$dst_rel, CLAMP:$clamp,
                   R600_Reg32:$src0, NEG:$src0_neg, REL:$src0_rel, ABS:$src0_abs, SEL:$src0_sel,
                   LAST:$last, R600_Pred:$pred_sel, LITERAL:$literal,
                   BANK_SWIZZLE:$bank_swizzle),
              !strconcat("  ", opName,
                   "$clamp $last $dst$write$dst_rel$omod, "
                   "$src0_neg$src0_abs$src0$src0_abs$src0_rel, "
                   "$pred_sel $bank_swizzle"),
              pattern,
              itin>,
    R600ALU_Word0,
    R600ALU_Word1_OP2 <inst> {

  let src1 = 0;
  let src1_rel = 0;
  let src1_neg = 0;
  let src1_abs = 0;
  let update_exec_mask = 0;
  let update_pred = 0;
  let HasNativeOperands = 1;
  let Op1 = 1;
  let ALUInst = 1;
  let DisableEncoding = "$literal";
  let UseNamedOperandTable = 1;

  let Inst{31-0}  = Word0;
  let Inst{63-32} = Word1;
}

class R600_1OP_Helper <bits<11> inst, string opName, SDPatternOperator node,
                    InstrItinClass itin = AnyALU> :
    R600_1OP <inst, opName,
              [(set R600_Reg32:$dst, (node R600_Reg32:$src0))]
>;

// If you add or change the operands for R600_2OP instructions, you must
// also update the R600Op2OperandIndex::ROI enum in R600Defines.h,
// R600InstrInfo::buildDefaultInstruction(), and R600InstrInfo::getOperandIdx().
class R600_2OP <bits<11> inst, string opName, list<dag> pattern,
                InstrItinClass itin = AnyALU> :
  InstR600 <(outs R600_Reg32:$dst),
          (ins UEM:$update_exec_mask, UP:$update_pred, WRITE:$write,
               OMOD:$omod, REL:$dst_rel, CLAMP:$clamp,
               R600_Reg32:$src0, NEG:$src0_neg, REL:$src0_rel, ABS:$src0_abs, SEL:$src0_sel,
               R600_Reg32:$src1, NEG:$src1_neg, REL:$src1_rel, ABS:$src1_abs, SEL:$src1_sel,
               LAST:$last, R600_Pred:$pred_sel, LITERAL:$literal,
               BANK_SWIZZLE:$bank_swizzle),
          !strconcat("  ", opName,
                "$clamp $last $update_exec_mask$update_pred$dst$write$dst_rel$omod, "
                "$src0_neg$src0_abs$src0$src0_abs$src0_rel, "
                "$src1_neg$src1_abs$src1$src1_abs$src1_rel, "
                "$pred_sel $bank_swizzle"),
          pattern,
          itin>,
    R600ALU_Word0,
    R600ALU_Word1_OP2 <inst> {

  let HasNativeOperands = 1;
  let Op2 = 1;
  let ALUInst = 1;
  let DisableEncoding = "$literal";
  let UseNamedOperandTable = 1;

  let Inst{31-0}  = Word0;
  let Inst{63-32} = Word1;
}

class R600_2OP_Helper <bits<11> inst, string opName, SDPatternOperator node,
                       InstrItinClass itim = AnyALU> :
    R600_2OP <inst, opName,
              [(set R600_Reg32:$dst, (node R600_Reg32:$src0,
                                           R600_Reg32:$src1))]
>;

// If you add our change the operands for R600_3OP instructions, you must
// also update the R600Op3OperandIndex::ROI enum in R600Defines.h,
// R600InstrInfo::buildDefaultInstruction(), and
// R600InstrInfo::getOperandIdx().
class R600_3OP <bits<5> inst, string opName, list<dag> pattern,
                InstrItinClass itin = AnyALU> :
  InstR600 <(outs R600_Reg32:$dst),
          (ins REL:$dst_rel, CLAMP:$clamp,
               R600_Reg32:$src0, NEG:$src0_neg, REL:$src0_rel, SEL:$src0_sel,
               R600_Reg32:$src1, NEG:$src1_neg, REL:$src1_rel, SEL:$src1_sel,
               R600_Reg32:$src2, NEG:$src2_neg, REL:$src2_rel, SEL:$src2_sel,
               LAST:$last, R600_Pred:$pred_sel, LITERAL:$literal,
               BANK_SWIZZLE:$bank_swizzle),
          !strconcat("  ", opName, "$clamp $last $dst$dst_rel, "
                             "$src0_neg$src0$src0_rel, "
                             "$src1_neg$src1$src1_rel, "
                             "$src2_neg$src2$src2_rel, "
                             "$pred_sel"
                             "$bank_swizzle"),
          pattern,
          itin>,
    R600ALU_Word0,
    R600ALU_Word1_OP3<inst>{

  let HasNativeOperands = 1;
  let DisableEncoding = "$literal";
  let Op3 = 1;
  let UseNamedOperandTable = 1;
  let ALUInst = 1;

  let Inst{31-0}  = Word0;
  let Inst{63-32} = Word1;
}

class R600_REDUCTION <bits<11> inst, dag ins, string asm, list<dag> pattern,
                      InstrItinClass itin = VecALU> :
  InstR600 <(outs R600_Reg32:$dst),
          ins,
          asm,
          pattern,
          itin>;



} // End mayLoad = 1, mayStore = 0, hasSideEffects = 0

def TEX_SHADOW : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return (TType >= 6 && TType <= 8) || (TType >= 11 && TType <= 13);
  }]
>;

def TEX_RECT : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return TType == 5;
  }]
>;

def TEX_ARRAY : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return TType == 9 || TType == 10 || TType == 16;
  }]
>;

def TEX_SHADOW_ARRAY : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return TType == 11 || TType == 12 || TType == 17;
  }]
>;

def TEX_MSAA : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return TType == 14;
  }]
>;

def TEX_ARRAY_MSAA : PatLeaf<
  (imm),
  [{uint32_t TType = (uint32_t)N->getZExtValue();
    return TType == 15;
  }]
>;

class EG_CF_RAT <bits <8> cfinst, bits <6> ratinst, bits<4> ratid, bits<4> mask,
                 dag outs, dag ins, string asm, list<dag> pattern> :
    InstR600ISA <outs, ins, asm, pattern>,
    CF_ALLOC_EXPORT_WORD0_RAT, CF_ALLOC_EXPORT_WORD1_BUF  {

  let rat_id = ratid;
  let rat_inst = ratinst;
  let rim         = 0;
  // XXX: Have a separate instruction for non-indexed writes.
  let type        = 1;
  let rw_rel      = 0;
  let elem_size   = 0;

  let array_size  = 0;
  let comp_mask   = mask;
  let burst_count = 0;
  let vpm         = 0;
  let cf_inst = cfinst;
  let mark        = 0;
  let barrier     = 1;

  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
  let IsExport = 1;

}

class VTX_READ <string name, bits<8> buffer_id, dag outs, list<dag> pattern>
    : InstR600ISA <outs, (ins MEMxi:$src_gpr), name, pattern>,
      VTX_WORD1_GPR {

  // Static fields
  let DST_REL = 0;
  // The docs say that if this bit is set, then DATA_FORMAT, NUM_FORMAT_ALL,
  // FORMAT_COMP_ALL, SRF_MODE_ALL, and ENDIAN_SWAP fields will be ignored,
  // however, based on my testing if USE_CONST_FIELDS is set, then all
  // these fields need to be set to 0.
  let USE_CONST_FIELDS = 0;
  let NUM_FORMAT_ALL = 1;
  let FORMAT_COMP_ALL = 0;
  let SRF_MODE_ALL = 0;

  let Inst{63-32} = Word1;
  // LLVM can only encode 64-bit instructions, so these fields are manually
  // encoded in R600CodeEmitter
  //
  // bits<16> OFFSET;
  // bits<2>  ENDIAN_SWAP = 0;
  // bits<1>  CONST_BUF_NO_STRIDE = 0;
  // bits<1>  MEGA_FETCH = 0;
  // bits<1>  ALT_CONST = 0;
  // bits<2>  BUFFER_INDEX_MODE = 0;

  // VTX_WORD2 (LLVM can only encode 64-bit instructions, so WORD2 encoding
  // is done in R600CodeEmitter
  //
  // Inst{79-64} = OFFSET;
  // Inst{81-80} = ENDIAN_SWAP;
  // Inst{82}    = CONST_BUF_NO_STRIDE;
  // Inst{83}    = MEGA_FETCH;
  // Inst{84}    = ALT_CONST;
  // Inst{86-85} = BUFFER_INDEX_MODE;
  // Inst{95-86} = 0; Reserved

  // VTX_WORD3 (Padding)
  //
  // Inst{127-96} = 0;

  let VTXInst = 1;
}

class LoadParamFrag <PatFrag load_type> : PatFrag <
  (ops node:$ptr), (load_type node:$ptr),
  [{ return isConstantLoad(dyn_cast<LoadSDNode>(N), 0); }]
>;

def load_param : LoadParamFrag<load>;
def load_param_exti8 : LoadParamFrag<az_extloadi8>;
def load_param_exti16 : LoadParamFrag<az_extloadi16>;

def isR600 : Predicate<"Subtarget.getGeneration() <= AMDGPUSubtarget::R700">;
def isR700 : Predicate<"Subtarget.getGeneration() == AMDGPUSubtarget::R700">;
def isEG : Predicate<
  "Subtarget.getGeneration() >= AMDGPUSubtarget::EVERGREEN && "
  "Subtarget.getGeneration() < AMDGPUSubtarget::SOUTHERN_ISLANDS && "
  "!Subtarget.hasCaymanISA()">;

def isCayman : Predicate<"Subtarget.hasCaymanISA()">;
def isEGorCayman : Predicate<"Subtarget.getGeneration() == "
                             "AMDGPUSubtarget::EVERGREEN"
                            "|| Subtarget.getGeneration() =="
                            "AMDGPUSubtarget::NORTHERN_ISLANDS">;

def isR600toCayman : Predicate<
                     "Subtarget.getGeneration() <= AMDGPUSubtarget::NORTHERN_ISLANDS">;

//===----------------------------------------------------------------------===//
// R600 SDNodes
//===----------------------------------------------------------------------===//

def INTERP_PAIR_XY :  AMDGPUShaderInst <
  (outs R600_TReg32_X:$dst0, R600_TReg32_Y:$dst1),
  (ins i32imm:$src0, R600_TReg32_Y:$src1, R600_TReg32_X:$src2),
  "INTERP_PAIR_XY $src0 $src1 $src2 : $dst0 dst1",
  []>;

def INTERP_PAIR_ZW :  AMDGPUShaderInst <
  (outs R600_TReg32_Z:$dst0, R600_TReg32_W:$dst1),
  (ins i32imm:$src0, R600_TReg32_Y:$src1, R600_TReg32_X:$src2),
  "INTERP_PAIR_ZW $src0 $src1 $src2 : $dst0 dst1",
  []>;

def CONST_ADDRESS: SDNode<"AMDGPUISD::CONST_ADDRESS",
  SDTypeProfile<1, -1, [SDTCisInt<0>, SDTCisPtrTy<1>]>,
  [SDNPVariadic]
>;

def DOT4 : SDNode<"AMDGPUISD::DOT4",
  SDTypeProfile<1, 8, [SDTCisFP<0>, SDTCisVT<1, f32>, SDTCisVT<2, f32>,
      SDTCisVT<3, f32>, SDTCisVT<4, f32>, SDTCisVT<5, f32>,
      SDTCisVT<6, f32>, SDTCisVT<7, f32>, SDTCisVT<8, f32>]>,
  []
>;

def COS_HW : SDNode<"AMDGPUISD::COS_HW",
  SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisFP<1>]>
>;

def SIN_HW : SDNode<"AMDGPUISD::SIN_HW",
  SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisFP<1>]>
>;

def TEXTURE_FETCH_Type : SDTypeProfile<1, 19, [SDTCisFP<0>]>;

def TEXTURE_FETCH: SDNode<"AMDGPUISD::TEXTURE_FETCH", TEXTURE_FETCH_Type, []>;

multiclass TexPattern<bits<32> TextureOp, Instruction inst, ValueType vt = v4f32> {
def : Pat<(TEXTURE_FETCH (i32 TextureOp), vt:$SRC_GPR,
          (i32 imm:$srcx), (i32 imm:$srcy), (i32 imm:$srcz), (i32 imm:$srcw),
          (i32 imm:$offsetx), (i32 imm:$offsety), (i32 imm:$offsetz),
          (i32 imm:$DST_SEL_X), (i32 imm:$DST_SEL_Y), (i32 imm:$DST_SEL_Z),
          (i32 imm:$DST_SEL_W),
          (i32 imm:$RESOURCE_ID), (i32 imm:$SAMPLER_ID),
          (i32 imm:$COORD_TYPE_X), (i32 imm:$COORD_TYPE_Y), (i32 imm:$COORD_TYPE_Z),
          (i32 imm:$COORD_TYPE_W)),
          (inst R600_Reg128:$SRC_GPR,
          imm:$srcx, imm:$srcy, imm:$srcz, imm:$srcw,
          imm:$offsetx, imm:$offsety, imm:$offsetz,
          imm:$DST_SEL_X, imm:$DST_SEL_Y, imm:$DST_SEL_Z,
          imm:$DST_SEL_W,
          imm:$RESOURCE_ID, imm:$SAMPLER_ID,
          imm:$COORD_TYPE_X, imm:$COORD_TYPE_Y, imm:$COORD_TYPE_Z,
          imm:$COORD_TYPE_W)>;
}

//===----------------------------------------------------------------------===//
// Interpolation Instructions
//===----------------------------------------------------------------------===//

def INTERP_VEC_LOAD :  AMDGPUShaderInst <
  (outs R600_Reg128:$dst),
  (ins i32imm:$src0),
  "INTERP_LOAD $src0 : $dst",
  [(set R600_Reg128:$dst, (int_R600_interp_const imm:$src0))]>;

def INTERP_XY : R600_2OP <0xD6, "INTERP_XY", []> {
  let bank_swizzle = 5;
}

def INTERP_ZW : R600_2OP <0xD7, "INTERP_ZW", []> {
  let bank_swizzle = 5;
}

def INTERP_LOAD_P0 : R600_1OP <0xE0, "INTERP_LOAD_P0", []>;

//===----------------------------------------------------------------------===//
// Export Instructions
//===----------------------------------------------------------------------===//

def ExportType : SDTypeProfile<0, 7, [SDTCisFP<0>, SDTCisInt<1>]>;

def EXPORT: SDNode<"AMDGPUISD::EXPORT", ExportType,
  [SDNPHasChain, SDNPSideEffect]>;

class ExportWord0 {
  field bits<32> Word0;

  bits<13> arraybase;
  bits<2> type;
  bits<7> gpr;
  bits<2> elem_size;

  let Word0{12-0} = arraybase;
  let Word0{14-13} = type;
  let Word0{21-15} = gpr;
  let Word0{22} = 0; // RW_REL
  let Word0{29-23} = 0; // INDEX_GPR
  let Word0{31-30} = elem_size;
}

class ExportSwzWord1 {
  field bits<32> Word1;

  bits<3> sw_x;
  bits<3> sw_y;
  bits<3> sw_z;
  bits<3> sw_w;
  bits<1> eop;
  bits<8> inst;

  let Word1{2-0} = sw_x;
  let Word1{5-3} = sw_y;
  let Word1{8-6} = sw_z;
  let Word1{11-9} = sw_w;
}

class ExportBufWord1 {
  field bits<32> Word1;

  bits<12> arraySize;
  bits<4> compMask;
  bits<1> eop;
  bits<8> inst;

  let Word1{11-0} = arraySize;
  let Word1{15-12} = compMask;
}

multiclass ExportPattern<Instruction ExportInst, bits<8> cf_inst> {
  def : Pat<(int_R600_store_pixel_depth R600_Reg32:$reg),
    (ExportInst
        (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), R600_Reg32:$reg, sub0),
        0, 61, 0, 7, 7, 7, cf_inst, 0)
  >;

  def : Pat<(int_R600_store_pixel_stencil R600_Reg32:$reg),
    (ExportInst
        (INSERT_SUBREG (v4f32 (IMPLICIT_DEF)), R600_Reg32:$reg, sub0),
        0, 61, 7, 0, 7, 7, cf_inst, 0)
  >;

  def : Pat<(int_R600_store_dummy (i32 imm:$type)),
    (ExportInst
        (v4f32 (IMPLICIT_DEF)), imm:$type, 0, 7, 7, 7, 7, cf_inst, 0)
  >;

  def : Pat<(int_R600_store_dummy 1),
    (ExportInst
        (v4f32 (IMPLICIT_DEF)), 1, 60, 7, 7, 7, 7, cf_inst, 0)
  >;

  def : Pat<(EXPORT (v4f32 R600_Reg128:$src), (i32 imm:$base), (i32 imm:$type),
    (i32 imm:$swz_x), (i32 imm:$swz_y), (i32 imm:$swz_z), (i32 imm:$swz_w)),
        (ExportInst R600_Reg128:$src, imm:$type, imm:$base,
        imm:$swz_x, imm:$swz_y, imm:$swz_z, imm:$swz_w, cf_inst, 0)
  >;

}

multiclass SteamOutputExportPattern<Instruction ExportInst,
    bits<8> buf0inst, bits<8> buf1inst, bits<8> buf2inst, bits<8> buf3inst> {
// Stream0
  def : Pat<(int_R600_store_stream_output (v4f32 R600_Reg128:$src),
      (i32 imm:$arraybase), (i32 0), (i32 imm:$mask)),
      (ExportInst R600_Reg128:$src, 0, imm:$arraybase,
      4095, imm:$mask, buf0inst, 0)>;
// Stream1
  def : Pat<(int_R600_store_stream_output (v4f32 R600_Reg128:$src),
      (i32 imm:$arraybase), (i32 1), (i32 imm:$mask)),
      (ExportInst R600_Reg128:$src, 0, imm:$arraybase,
      4095, imm:$mask, buf1inst, 0)>;
// Stream2
  def : Pat<(int_R600_store_stream_output (v4f32 R600_Reg128:$src),
      (i32 imm:$arraybase), (i32 2), (i32 imm:$mask)),
      (ExportInst R600_Reg128:$src, 0, imm:$arraybase,
      4095, imm:$mask, buf2inst, 0)>;
// Stream3
  def : Pat<(int_R600_store_stream_output (v4f32 R600_Reg128:$src),
      (i32 imm:$arraybase), (i32 3), (i32 imm:$mask)),
      (ExportInst R600_Reg128:$src, 0, imm:$arraybase,
      4095, imm:$mask, buf3inst, 0)>;
}

// Export Instructions should not be duplicated by TailDuplication pass
// (which assumes that duplicable instruction are affected by exec mask)
let usesCustomInserter = 1, isNotDuplicable = 1 in {

class ExportSwzInst : InstR600ISA<(
    outs),
    (ins R600_Reg128:$gpr, i32imm:$type, i32imm:$arraybase,
    RSel:$sw_x, RSel:$sw_y, RSel:$sw_z, RSel:$sw_w, i32imm:$inst,
    i32imm:$eop),
    !strconcat("EXPORT", " $gpr.$sw_x$sw_y$sw_z$sw_w"),
    []>, ExportWord0, ExportSwzWord1 {
  let elem_size = 3;
  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
  let IsExport = 1;
}

} // End usesCustomInserter = 1

class ExportBufInst : InstR600ISA<(
    outs),
    (ins R600_Reg128:$gpr, i32imm:$type, i32imm:$arraybase,
    i32imm:$arraySize, i32imm:$compMask, i32imm:$inst, i32imm:$eop),
    !strconcat("EXPORT", " $gpr"),
    []>, ExportWord0, ExportBufWord1 {
  let elem_size = 0;
  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
  let IsExport = 1;
}

//===----------------------------------------------------------------------===//
// Control Flow Instructions
//===----------------------------------------------------------------------===//


def KCACHE : InstFlag<"printKCache">;

class ALU_CLAUSE<bits<4> inst, string OpName> : AMDGPUInst <(outs),
(ins i32imm:$ADDR, i32imm:$KCACHE_BANK0, i32imm:$KCACHE_BANK1,
KCACHE:$KCACHE_MODE0, KCACHE:$KCACHE_MODE1,
i32imm:$KCACHE_ADDR0, i32imm:$KCACHE_ADDR1,
i32imm:$COUNT, i32imm:$Enabled),
!strconcat(OpName, " $COUNT, @$ADDR, "
"KC0[$KCACHE_MODE0], KC1[$KCACHE_MODE1]"),
[] >, CF_ALU_WORD0, CF_ALU_WORD1 {
  field bits<64> Inst;

  let CF_INST = inst;
  let ALT_CONST = 0;
  let WHOLE_QUAD_MODE = 0;
  let BARRIER = 1;
  let UseNamedOperandTable = 1;

  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
}

class CF_WORD0_R600 {
  field bits<32> Word0;

  bits<32> ADDR;

  let Word0 = ADDR;
}

class CF_CLAUSE_R600 <bits<7> inst, dag ins, string AsmPrint> : AMDGPUInst <(outs),
ins, AsmPrint, [] >, CF_WORD0_R600, CF_WORD1_R600 {
  field bits<64> Inst;
  bits<4> CNT;

  let CF_INST = inst;
  let BARRIER = 1;
  let CF_CONST = 0;
  let VALID_PIXEL_MODE = 0;
  let COND = 0;
  let COUNT = CNT{2-0};
  let CALL_COUNT = 0;
  let COUNT_3 = CNT{3};
  let END_OF_PROGRAM = 0;
  let WHOLE_QUAD_MODE = 0;

  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
}

class CF_CLAUSE_EG <bits<8> inst, dag ins, string AsmPrint> : AMDGPUInst <(outs),
ins, AsmPrint, [] >, CF_WORD0_EG, CF_WORD1_EG {
  field bits<64> Inst;

  let CF_INST = inst;
  let BARRIER = 1;
  let JUMPTABLE_SEL = 0;
  let CF_CONST = 0;
  let VALID_PIXEL_MODE = 0;
  let COND = 0;
  let END_OF_PROGRAM = 0;

  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;
}

def CF_ALU : ALU_CLAUSE<8, "ALU">;
def CF_ALU_PUSH_BEFORE : ALU_CLAUSE<9, "ALU_PUSH_BEFORE">;
def CF_ALU_POP_AFTER : ALU_CLAUSE<10, "ALU_POP_AFTER">;

def FETCH_CLAUSE : AMDGPUInst <(outs),
(ins i32imm:$addr), "Fetch clause starting at $addr:", [] > {
  field bits<8> Inst;
  bits<8> num;
  let Inst = num;
}

def ALU_CLAUSE : AMDGPUInst <(outs),
(ins i32imm:$addr), "ALU clause starting at $addr:", [] > {
  field bits<8> Inst;
  bits<8> num;
  let Inst = num;
}

def LITERALS : AMDGPUInst <(outs),
(ins LITERAL:$literal1, LITERAL:$literal2), "$literal1, $literal2", [] > {
  field bits<64> Inst;
  bits<32> literal1;
  bits<32> literal2;

  let Inst{31-0} = literal1;
  let Inst{63-32} = literal2;
}

def PAD : AMDGPUInst <(outs), (ins), "PAD", [] > {
  field bits<64> Inst;
}

let Predicates = [isR600toCayman] in {

//===----------------------------------------------------------------------===//
// Common Instructions R600, R700, Evergreen, Cayman
//===----------------------------------------------------------------------===//

def ADD : R600_2OP_Helper <0x0, "ADD", fadd>;
// Non-IEEE MUL: 0 * anything = 0
def MUL : R600_2OP_Helper <0x1, "MUL NON-IEEE", int_AMDGPU_mul>;
def MUL_IEEE : R600_2OP_Helper <0x2, "MUL_IEEE", fmul>;
def MAX : R600_2OP_Helper <0x3, "MAX", AMDGPUfmax>;
def MIN : R600_2OP_Helper <0x4, "MIN", AMDGPUfmin>;

// For the SET* instructions there is a naming conflict in TargetSelectionDAG.td,
// so some of the instruction names don't match the asm string.
// XXX: Use the defs in TargetSelectionDAG.td instead of intrinsics.
def SETE : R600_2OP <
  0x08, "SETE",
  [(set f32:$dst, (selectcc f32:$src0, f32:$src1, FP_ONE, FP_ZERO, COND_OEQ))]
>;

def SGT : R600_2OP <
  0x09, "SETGT",
  [(set f32:$dst, (selectcc f32:$src0, f32:$src1, FP_ONE, FP_ZERO, COND_OGT))]
>;

def SGE : R600_2OP <
  0xA, "SETGE",
  [(set f32:$dst, (selectcc f32:$src0, f32:$src1, FP_ONE, FP_ZERO, COND_OGE))]
>;

def SNE : R600_2OP <
  0xB, "SETNE",
  [(set f32:$dst, (selectcc f32:$src0, f32:$src1, FP_ONE, FP_ZERO, COND_UNE))]
>;

def SETE_DX10 : R600_2OP <
  0xC, "SETE_DX10",
  [(set i32:$dst, (selectcc f32:$src0, f32:$src1, -1, 0, COND_OEQ))]
>;

def SETGT_DX10 : R600_2OP <
  0xD, "SETGT_DX10",
  [(set i32:$dst, (selectcc f32:$src0, f32:$src1, -1, 0, COND_OGT))]
>;

def SETGE_DX10 : R600_2OP <
  0xE, "SETGE_DX10",
  [(set i32:$dst, (selectcc f32:$src0, f32:$src1, -1, 0, COND_OGE))]
>;

def SETNE_DX10 : R600_2OP <
  0xF, "SETNE_DX10",
  [(set i32:$dst, (selectcc f32:$src0, f32:$src1, -1, 0, COND_UNE))]
>;

def FRACT : R600_1OP_Helper <0x10, "FRACT", AMDGPUfract>;
def TRUNC : R600_1OP_Helper <0x11, "TRUNC", int_AMDGPU_trunc>;
def CEIL : R600_1OP_Helper <0x12, "CEIL", fceil>;
def RNDNE : R600_1OP_Helper <0x13, "RNDNE", frint>;
def FLOOR : R600_1OP_Helper <0x14, "FLOOR", ffloor>;

def MOV : R600_1OP <0x19, "MOV", []>;

let isPseudo = 1, isCodeGenOnly = 1, usesCustomInserter = 1 in {

class MOV_IMM <ValueType vt, Operand immType> : AMDGPUInst <
  (outs R600_Reg32:$dst),
  (ins immType:$imm),
  "",
  []
>;

} // end let isPseudo = 1, isCodeGenOnly = 1, usesCustomInserter = 1

def MOV_IMM_I32 : MOV_IMM<i32, i32imm>;
def : Pat <
  (imm:$val),
  (MOV_IMM_I32 imm:$val)
>;

def MOV_IMM_F32 : MOV_IMM<f32, f32imm>;
def : Pat <
  (fpimm:$val),
  (MOV_IMM_F32  fpimm:$val)
>;

def PRED_SETE : R600_2OP <0x20, "PRED_SETE", []>;
def PRED_SETGT : R600_2OP <0x21, "PRED_SETGT", []>;
def PRED_SETGE : R600_2OP <0x22, "PRED_SETGE", []>;
def PRED_SETNE : R600_2OP <0x23, "PRED_SETNE", []>;

let hasSideEffects = 1 in {

def KILLGT : R600_2OP <0x2D, "KILLGT", []>;

} // end hasSideEffects

def AND_INT : R600_2OP_Helper <0x30, "AND_INT", and>;
def OR_INT : R600_2OP_Helper <0x31, "OR_INT", or>;
def XOR_INT : R600_2OP_Helper <0x32, "XOR_INT", xor>;
def NOT_INT : R600_1OP_Helper <0x33, "NOT_INT", not>;
def ADD_INT : R600_2OP_Helper <0x34, "ADD_INT", add>;
def SUB_INT : R600_2OP_Helper <0x35, "SUB_INT", sub>;
def MAX_INT : R600_2OP_Helper <0x36, "MAX_INT", AMDGPUsmax>;
def MIN_INT : R600_2OP_Helper <0x37, "MIN_INT", AMDGPUsmin>;
def MAX_UINT : R600_2OP_Helper <0x38, "MAX_UINT", AMDGPUumax>;
def MIN_UINT : R600_2OP_Helper <0x39, "MIN_UINT", AMDGPUumin>;

def SETE_INT : R600_2OP <
  0x3A, "SETE_INT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETEQ))]
>;

def SETGT_INT : R600_2OP <
  0x3B, "SETGT_INT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETGT))]
>;

def SETGE_INT : R600_2OP <
  0x3C, "SETGE_INT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETGE))]
>;

def SETNE_INT : R600_2OP <
  0x3D, "SETNE_INT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETNE))]
>;

def SETGT_UINT : R600_2OP <
  0x3E, "SETGT_UINT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETUGT))]
>;

def SETGE_UINT : R600_2OP <
  0x3F, "SETGE_UINT",
  [(set i32:$dst, (selectcc i32:$src0, i32:$src1, -1, 0, SETUGE))]
>;

def PRED_SETE_INT : R600_2OP <0x42, "PRED_SETE_INT", []>;
def PRED_SETGT_INT : R600_2OP <0x43, "PRED_SETGE_INT", []>;
def PRED_SETGE_INT : R600_2OP <0x44, "PRED_SETGE_INT", []>;
def PRED_SETNE_INT : R600_2OP <0x45, "PRED_SETNE_INT", []>;

def CNDE_INT : R600_3OP <
  0x1C, "CNDE_INT",
  [(set i32:$dst, (selectcc i32:$src0, 0, i32:$src1, i32:$src2, COND_EQ))]
>;

def CNDGE_INT : R600_3OP <
  0x1E, "CNDGE_INT",
  [(set i32:$dst, (selectcc i32:$src0, 0, i32:$src1, i32:$src2, COND_SGE))]
>;

def CNDGT_INT : R600_3OP <
  0x1D, "CNDGT_INT",
  [(set i32:$dst, (selectcc i32:$src0, 0, i32:$src1, i32:$src2, COND_SGT))]
>;

//===----------------------------------------------------------------------===//
// Texture instructions
//===----------------------------------------------------------------------===//

let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {

class R600_TEX <bits<11> inst, string opName> :
  InstR600 <(outs R600_Reg128:$DST_GPR),
          (ins R600_Reg128:$SRC_GPR,
          RSel:$srcx, RSel:$srcy, RSel:$srcz, RSel:$srcw,
          i32imm:$offsetx, i32imm:$offsety, i32imm:$offsetz,
          RSel:$DST_SEL_X, RSel:$DST_SEL_Y, RSel:$DST_SEL_Z, RSel:$DST_SEL_W,
          i32imm:$RESOURCE_ID, i32imm:$SAMPLER_ID,
          CT:$COORD_TYPE_X, CT:$COORD_TYPE_Y, CT:$COORD_TYPE_Z,
          CT:$COORD_TYPE_W),
          !strconcat(opName,
          " $DST_GPR.$DST_SEL_X$DST_SEL_Y$DST_SEL_Z$DST_SEL_W, "
          "$SRC_GPR.$srcx$srcy$srcz$srcw "
          "RID:$RESOURCE_ID SID:$SAMPLER_ID "
          "CT:$COORD_TYPE_X$COORD_TYPE_Y$COORD_TYPE_Z$COORD_TYPE_W"),
          [],
          NullALU>, TEX_WORD0, TEX_WORD1, TEX_WORD2 {
  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;

  let TEX_INST = inst{4-0};
  let SRC_REL = 0;
  let DST_REL = 0;
  let LOD_BIAS = 0;

  let INST_MOD = 0;
  let FETCH_WHOLE_QUAD = 0;
  let ALT_CONST = 0;
  let SAMPLER_INDEX_MODE = 0;
  let RESOURCE_INDEX_MODE = 0;

  let TEXInst = 1;
}

} // End mayLoad = 0, mayStore = 0, hasSideEffects = 0



def TEX_SAMPLE : R600_TEX <0x10, "TEX_SAMPLE">;
def TEX_SAMPLE_C : R600_TEX <0x18, "TEX_SAMPLE_C">;
def TEX_SAMPLE_L : R600_TEX <0x11, "TEX_SAMPLE_L">;
def TEX_SAMPLE_C_L : R600_TEX <0x19, "TEX_SAMPLE_C_L">;
def TEX_SAMPLE_LB : R600_TEX <0x12, "TEX_SAMPLE_LB">;
def TEX_SAMPLE_C_LB : R600_TEX <0x1A, "TEX_SAMPLE_C_LB">;
def TEX_LD : R600_TEX <0x03, "TEX_LD">;
def TEX_LDPTR : R600_TEX <0x03, "TEX_LDPTR"> {
  let INST_MOD = 1;
}
def TEX_GET_TEXTURE_RESINFO : R600_TEX <0x04, "TEX_GET_TEXTURE_RESINFO">;
def TEX_GET_GRADIENTS_H : R600_TEX <0x07, "TEX_GET_GRADIENTS_H">;
def TEX_GET_GRADIENTS_V : R600_TEX <0x08, "TEX_GET_GRADIENTS_V">;
def TEX_SET_GRADIENTS_H : R600_TEX <0x0B, "TEX_SET_GRADIENTS_H">;
def TEX_SET_GRADIENTS_V : R600_TEX <0x0C, "TEX_SET_GRADIENTS_V">;
def TEX_SAMPLE_G : R600_TEX <0x14, "TEX_SAMPLE_G">;
def TEX_SAMPLE_C_G : R600_TEX <0x1C, "TEX_SAMPLE_C_G">;

defm : TexPattern<0, TEX_SAMPLE>;
defm : TexPattern<1, TEX_SAMPLE_C>;
defm : TexPattern<2, TEX_SAMPLE_L>;
defm : TexPattern<3, TEX_SAMPLE_C_L>;
defm : TexPattern<4, TEX_SAMPLE_LB>;
defm : TexPattern<5, TEX_SAMPLE_C_LB>;
defm : TexPattern<6, TEX_LD, v4i32>;
defm : TexPattern<7, TEX_GET_TEXTURE_RESINFO, v4i32>;
defm : TexPattern<8, TEX_GET_GRADIENTS_H>;
defm : TexPattern<9, TEX_GET_GRADIENTS_V>;
defm : TexPattern<10, TEX_LDPTR, v4i32>;

//===----------------------------------------------------------------------===//
// Helper classes for common instructions
//===----------------------------------------------------------------------===//

class MUL_LIT_Common <bits<5> inst> : R600_3OP <
  inst, "MUL_LIT",
  []
>;

class MULADD_Common <bits<5> inst> : R600_3OP <
  inst, "MULADD",
  []
>;

class MULADD_IEEE_Common <bits<5> inst> : R600_3OP <
  inst, "MULADD_IEEE",
  [(set f32:$dst, (fadd (fmul f32:$src0, f32:$src1), f32:$src2))]
>;

class CNDE_Common <bits<5> inst> : R600_3OP <
  inst, "CNDE",
  [(set f32:$dst, (selectcc f32:$src0, FP_ZERO, f32:$src1, f32:$src2, COND_OEQ))]
>;

class CNDGT_Common <bits<5> inst> : R600_3OP <
  inst, "CNDGT",
  [(set f32:$dst, (selectcc f32:$src0, FP_ZERO, f32:$src1, f32:$src2, COND_OGT))]
> {
  let Itinerary = VecALU;
}

class CNDGE_Common <bits<5> inst> : R600_3OP <
  inst, "CNDGE",
  [(set f32:$dst, (selectcc f32:$src0, FP_ZERO, f32:$src1, f32:$src2, COND_OGE))]
> {
  let Itinerary = VecALU;
}


let isCodeGenOnly = 1, isPseudo = 1, Namespace = "AMDGPU"  in {
class R600_VEC2OP<list<dag> pattern> : InstR600 <(outs R600_Reg32:$dst), (ins
// Slot X
   UEM:$update_exec_mask_X, UP:$update_pred_X, WRITE:$write_X,
   OMOD:$omod_X, REL:$dst_rel_X, CLAMP:$clamp_X,
   R600_TReg32_X:$src0_X, NEG:$src0_neg_X, REL:$src0_rel_X, ABS:$src0_abs_X, SEL:$src0_sel_X,
   R600_TReg32_X:$src1_X, NEG:$src1_neg_X, REL:$src1_rel_X, ABS:$src1_abs_X, SEL:$src1_sel_X,
   R600_Pred:$pred_sel_X,
// Slot Y
   UEM:$update_exec_mask_Y, UP:$update_pred_Y, WRITE:$write_Y,
   OMOD:$omod_Y, REL:$dst_rel_Y, CLAMP:$clamp_Y,
   R600_TReg32_Y:$src0_Y, NEG:$src0_neg_Y, REL:$src0_rel_Y, ABS:$src0_abs_Y, SEL:$src0_sel_Y,
   R600_TReg32_Y:$src1_Y, NEG:$src1_neg_Y, REL:$src1_rel_Y, ABS:$src1_abs_Y, SEL:$src1_sel_Y,
   R600_Pred:$pred_sel_Y,
// Slot Z
   UEM:$update_exec_mask_Z, UP:$update_pred_Z, WRITE:$write_Z,
   OMOD:$omod_Z, REL:$dst_rel_Z, CLAMP:$clamp_Z,
   R600_TReg32_Z:$src0_Z, NEG:$src0_neg_Z, REL:$src0_rel_Z, ABS:$src0_abs_Z, SEL:$src0_sel_Z,
   R600_TReg32_Z:$src1_Z, NEG:$src1_neg_Z, REL:$src1_rel_Z, ABS:$src1_abs_Z, SEL:$src1_sel_Z,
   R600_Pred:$pred_sel_Z,
// Slot W
   UEM:$update_exec_mask_W, UP:$update_pred_W, WRITE:$write_W,
   OMOD:$omod_W, REL:$dst_rel_W, CLAMP:$clamp_W,
   R600_TReg32_W:$src0_W, NEG:$src0_neg_W, REL:$src0_rel_W, ABS:$src0_abs_W, SEL:$src0_sel_W,
   R600_TReg32_W:$src1_W, NEG:$src1_neg_W, REL:$src1_rel_W, ABS:$src1_abs_W, SEL:$src1_sel_W,
   R600_Pred:$pred_sel_W,
   LITERAL:$literal0, LITERAL:$literal1),
  "",
  pattern,
  AnyALU> {

  let UseNamedOperandTable = 1;

}
}

def DOT_4 : R600_VEC2OP<[(set R600_Reg32:$dst, (DOT4
  R600_TReg32_X:$src0_X, R600_TReg32_X:$src1_X,
  R600_TReg32_Y:$src0_Y, R600_TReg32_Y:$src1_Y,
  R600_TReg32_Z:$src0_Z, R600_TReg32_Z:$src1_Z,
  R600_TReg32_W:$src0_W, R600_TReg32_W:$src1_W))]>;


class DOT4_Common <bits<11> inst> : R600_2OP <inst, "DOT4", []>;


let mayLoad = 0, mayStore = 0, hasSideEffects = 0 in {
multiclass CUBE_Common <bits<11> inst> {

  def _pseudo : InstR600 <
    (outs R600_Reg128:$dst),
    (ins R600_Reg128:$src0),
    "CUBE $dst $src0",
    [(set v4f32:$dst, (int_AMDGPU_cube v4f32:$src0))],
    VecALU
  > {
    let isPseudo = 1;
    let UseNamedOperandTable = 1;
  }

  def _real : R600_2OP <inst, "CUBE", []>;
}
} // End mayLoad = 0, mayStore = 0, hasSideEffects = 0

class EXP_IEEE_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "EXP_IEEE", fexp2
> {
  let Itinerary = TransALU;
}

class FLT_TO_INT_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "FLT_TO_INT", fp_to_sint
> {
  let Itinerary = TransALU;
}

class INT_TO_FLT_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "INT_TO_FLT", sint_to_fp
> {
  let Itinerary = TransALU;
}

class FLT_TO_UINT_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "FLT_TO_UINT", fp_to_uint
> {
  let Itinerary = TransALU;
}

class UINT_TO_FLT_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "UINT_TO_FLT", uint_to_fp
> {
  let Itinerary = TransALU;
}

class LOG_CLAMPED_Common <bits<11> inst> : R600_1OP <
  inst, "LOG_CLAMPED", []
>;

class LOG_IEEE_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "LOG_IEEE", flog2
> {
  let Itinerary = TransALU;
}

class LSHL_Common <bits<11> inst> : R600_2OP_Helper <inst, "LSHL", shl>;
class LSHR_Common <bits<11> inst> : R600_2OP_Helper <inst, "LSHR", srl>;
class ASHR_Common <bits<11> inst> : R600_2OP_Helper <inst, "ASHR", sra>;
class MULHI_INT_Common <bits<11> inst> : R600_2OP_Helper <
  inst, "MULHI_INT", mulhs
> {
  let Itinerary = TransALU;
}
class MULHI_UINT_Common <bits<11> inst> : R600_2OP_Helper <
  inst, "MULHI", mulhu
> {
  let Itinerary = TransALU;
}
class MULLO_INT_Common <bits<11> inst> : R600_2OP_Helper <
  inst, "MULLO_INT", mul
> {
  let Itinerary = TransALU;
}
class MULLO_UINT_Common <bits<11> inst> : R600_2OP <inst, "MULLO_UINT", []> {
  let Itinerary = TransALU;
}

class RECIP_CLAMPED_Common <bits<11> inst> : R600_1OP <
  inst, "RECIP_CLAMPED", []
> {
  let Itinerary = TransALU;
}

class RECIP_IEEE_Common <bits<11> inst> : R600_1OP <
  inst, "RECIP_IEEE", [(set f32:$dst, (fdiv FP_ONE, f32:$src0))]
> {
  let Itinerary = TransALU;
}

class RECIP_UINT_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "RECIP_UINT", AMDGPUurecip
> {
  let Itinerary = TransALU;
}

class RECIPSQRT_CLAMPED_Common <bits<11> inst> : R600_1OP_Helper <
  inst, "RECIPSQRT_CLAMPED", int_AMDGPU_rsq
> {
  let Itinerary = TransALU;
}

class RECIPSQRT_IEEE_Common <bits<11> inst> : R600_1OP <
  inst, "RECIPSQRT_IEEE", []
> {
  let Itinerary = TransALU;
}

class SIN_Common <bits<11> inst> : R600_1OP <
  inst, "SIN", [(set f32:$dst, (SIN_HW f32:$src0))]>{
  let Trig = 1;
  let Itinerary = TransALU;
}

class COS_Common <bits<11> inst> : R600_1OP <
  inst, "COS", [(set f32:$dst, (COS_HW f32:$src0))]> {
  let Trig = 1;
  let Itinerary = TransALU;
}

def CLAMP_R600 :  CLAMP <R600_Reg32>;
def FABS_R600 : FABS<R600_Reg32>;
def FNEG_R600 : FNEG<R600_Reg32>;

//===----------------------------------------------------------------------===//
// Helper patterns for complex intrinsics
//===----------------------------------------------------------------------===//

multiclass DIV_Common <InstR600 recip_ieee> {
def : Pat<
  (int_AMDGPU_div f32:$src0, f32:$src1),
  (MUL_IEEE $src0, (recip_ieee $src1))
>;

def : Pat<
  (fdiv f32:$src0, f32:$src1),
  (MUL_IEEE $src0, (recip_ieee $src1))
>;
}

class TGSI_LIT_Z_Common <InstR600 mul_lit, InstR600 log_clamped, InstR600 exp_ieee>
  : Pat <
  (int_TGSI_lit_z f32:$src_x, f32:$src_y, f32:$src_w),
  (exp_ieee (mul_lit (log_clamped (MAX $src_y, (f32 ZERO))), $src_w, $src_x))
>;

// FROUND pattern
class FROUNDPat<Instruction CNDGE> : Pat <
  (AMDGPUround f32:$x),
  (CNDGE (ADD (FNEG_R600 (f32 HALF)), (FRACT $x)), (CEIL $x), (FLOOR $x))
>;


//===----------------------------------------------------------------------===//
// R600 / R700 Instructions
//===----------------------------------------------------------------------===//

let Predicates = [isR600] in {

  def MUL_LIT_r600 : MUL_LIT_Common<0x0C>;
  def MULADD_r600 : MULADD_Common<0x10>;
  def MULADD_IEEE_r600 : MULADD_IEEE_Common<0x14>;
  def CNDE_r600 : CNDE_Common<0x18>;
  def CNDGT_r600 : CNDGT_Common<0x19>;
  def CNDGE_r600 : CNDGE_Common<0x1A>;
  def DOT4_r600 : DOT4_Common<0x50>;
  defm CUBE_r600 : CUBE_Common<0x52>;
  def EXP_IEEE_r600 : EXP_IEEE_Common<0x61>;
  def LOG_CLAMPED_r600 : LOG_CLAMPED_Common<0x62>;
  def LOG_IEEE_r600 : LOG_IEEE_Common<0x63>;
  def RECIP_CLAMPED_r600 : RECIP_CLAMPED_Common<0x64>;
  def RECIP_IEEE_r600 : RECIP_IEEE_Common<0x66>;
  def RECIPSQRT_CLAMPED_r600 : RECIPSQRT_CLAMPED_Common<0x67>;
  def RECIPSQRT_IEEE_r600 : RECIPSQRT_IEEE_Common<0x69>;
  def FLT_TO_INT_r600 : FLT_TO_INT_Common<0x6b>;
  def INT_TO_FLT_r600 : INT_TO_FLT_Common<0x6c>;
  def FLT_TO_UINT_r600 : FLT_TO_UINT_Common<0x79>;
  def UINT_TO_FLT_r600 : UINT_TO_FLT_Common<0x6d>;
  def SIN_r600 : SIN_Common<0x6E>;
  def COS_r600 : COS_Common<0x6F>;
  def ASHR_r600 : ASHR_Common<0x70>;
  def LSHR_r600 : LSHR_Common<0x71>;
  def LSHL_r600 : LSHL_Common<0x72>;
  def MULLO_INT_r600 : MULLO_INT_Common<0x73>;
  def MULHI_INT_r600 : MULHI_INT_Common<0x74>;
  def MULLO_UINT_r600 : MULLO_UINT_Common<0x75>;
  def MULHI_UINT_r600 : MULHI_UINT_Common<0x76>;
  def RECIP_UINT_r600 : RECIP_UINT_Common <0x78>;

  defm DIV_r600 : DIV_Common<RECIP_IEEE_r600>;
  def : POW_Common <LOG_IEEE_r600, EXP_IEEE_r600, MUL>;
  def TGSI_LIT_Z_r600 : TGSI_LIT_Z_Common<MUL_LIT_r600, LOG_CLAMPED_r600, EXP_IEEE_r600>;

  def : Pat<(fsqrt f32:$src), (MUL $src, (RECIPSQRT_CLAMPED_r600 $src))>;
  def : FROUNDPat <CNDGE_r600>;

  def R600_ExportSwz : ExportSwzInst {
    let Word1{20-17} = 0; // BURST_COUNT
    let Word1{21} = eop;
    let Word1{22} = 0; // VALID_PIXEL_MODE
    let Word1{30-23} = inst;
    let Word1{31} = 1; // BARRIER
  }
  defm : ExportPattern<R600_ExportSwz, 39>;

  def R600_ExportBuf : ExportBufInst {
    let Word1{20-17} = 0; // BURST_COUNT
    let Word1{21} = eop;
    let Word1{22} = 0; // VALID_PIXEL_MODE
    let Word1{30-23} = inst;
    let Word1{31} = 1; // BARRIER
  }
  defm : SteamOutputExportPattern<R600_ExportBuf, 0x20, 0x21, 0x22, 0x23>;

  def CF_TC_R600 : CF_CLAUSE_R600<1, (ins i32imm:$ADDR, i32imm:$CNT),
  "TEX $CNT @$ADDR"> {
    let POP_COUNT = 0;
  }
  def CF_VC_R600 : CF_CLAUSE_R600<2, (ins i32imm:$ADDR, i32imm:$CNT),
  "VTX $CNT @$ADDR"> {
    let POP_COUNT = 0;
  }
  def WHILE_LOOP_R600 : CF_CLAUSE_R600<6, (ins i32imm:$ADDR),
  "LOOP_START_DX10 @$ADDR"> {
    let POP_COUNT = 0;
    let CNT = 0;
  }
  def END_LOOP_R600 : CF_CLAUSE_R600<5, (ins i32imm:$ADDR), "END_LOOP @$ADDR"> {
    let POP_COUNT = 0;
    let CNT = 0;
  }
  def LOOP_BREAK_R600 : CF_CLAUSE_R600<9, (ins i32imm:$ADDR),
  "LOOP_BREAK @$ADDR"> {
    let POP_COUNT = 0;
    let CNT = 0;
  }
  def CF_CONTINUE_R600 : CF_CLAUSE_R600<8, (ins i32imm:$ADDR),
  "CONTINUE @$ADDR"> {
    let POP_COUNT = 0;
    let CNT = 0;
  }
  def CF_JUMP_R600 : CF_CLAUSE_R600<10, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "JUMP @$ADDR POP:$POP_COUNT"> {
    let CNT = 0;
  }
  def CF_ELSE_R600 : CF_CLAUSE_R600<13, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "ELSE @$ADDR POP:$POP_COUNT"> {
    let CNT = 0;
  }
  def CF_CALL_FS_R600 : CF_CLAUSE_R600<19, (ins), "CALL_FS"> {
    let ADDR = 0;
    let CNT = 0;
    let POP_COUNT = 0;
  }
  def POP_R600 : CF_CLAUSE_R600<14, (ins i32imm:$ADDR, i32imm:$POP_COUNT),
  "POP @$ADDR POP:$POP_COUNT"> {
    let CNT = 0;
  }
  def CF_END_R600 : CF_CLAUSE_R600<0, (ins), "CF_END"> {
    let CNT = 0;
    let POP_COUNT = 0;
    let ADDR = 0;
    let END_OF_PROGRAM = 1;
  }

}

//===----------------------------------------------------------------------===//
// R700 Only instructions
//===----------------------------------------------------------------------===//

let Predicates = [isR700] in {
  def SIN_r700 : SIN_Common<0x6E>;
  def COS_r700 : COS_Common<0x6F>;
}

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

def RAT_MSKOR : CF_MEM_RAT <0x11, 0,
  (ins R600_Reg128:$rw_gpr, R600_TReg32_X:$index_gpr),
  "MSKOR $rw_gpr.XW, $index_gpr",
  [(mskor_global v4i32:$rw_gpr, i32:$index_gpr)]
> {
  let eop = 0;
}

} // End Predicates = [isEGorCayman]


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
  // (0, 8)           = (Input << 24) >> 24  = (Input &  0xff)       >> 0
  // (8, 8)           = (Input << 16) >> 24  = (Input &  0xffff)     >> 8
  // (16,8)           = (Input <<  8) >> 24  = (Input &  0xffffff)   >> 16
  // (24,8)           = (Input <<  0) >> 24  = (Input &  0xffffffff) >> 24
  def BFE_UINT_eg : R600_3OP <0x4, "BFE_UINT",
    [(set i32:$dst, (int_AMDIL_bit_extract_u32 i32:$src0, i32:$src1,
                                               i32:$src2))],
    VecALU
  >;
  def : BFEPattern <BFE_UINT_eg>;

  def BFI_INT_eg : R600_3OP <0x06, "BFI_INT", [], VecALU>;
  defm : BFIPatterns <BFI_INT_eg>;

  def MULADD_UINT24_eg : R600_3OP <0x10, "MULADD_UINT24",
    [(set i32:$dst, (add (mul U24:$src0, U24:$src1), i32:$src2))], VecALU
  >;
  def BIT_ALIGN_INT_eg : R600_3OP <0xC, "BIT_ALIGN_INT", [], VecALU>;
  def : ROTRPattern <BIT_ALIGN_INT_eg>;

  def MULADD_eg : MULADD_Common<0x14>;
  def MULADD_IEEE_eg : MULADD_IEEE_Common<0x18>;
  def ASHR_eg : ASHR_Common<0x15>;
  def LSHR_eg : LSHR_Common<0x16>;
  def LSHL_eg : LSHL_Common<0x17>;
  def CNDE_eg : CNDE_Common<0x19>;
  def CNDGT_eg : CNDGT_Common<0x1A>;
  def CNDGE_eg : CNDGE_Common<0x1B>;
  def MUL_LIT_eg : MUL_LIT_Common<0x1F>;
  def LOG_CLAMPED_eg : LOG_CLAMPED_Common<0x82>;
  def MUL_UINT24_eg : R600_2OP <0xB5, "MUL_UINT24",
    [(set i32:$dst, (mul U24:$src0, U24:$src1))], VecALU
  >;
  def DOT4_eg : DOT4_Common<0xBE>;
  defm CUBE_eg : CUBE_Common<0xC0>;

let hasSideEffects = 1 in {
  def MOVA_INT_eg : R600_1OP <0xCC, "MOVA_INT", []>;
}

  def TGSI_LIT_Z_eg : TGSI_LIT_Z_Common<MUL_LIT_eg, LOG_CLAMPED_eg, EXP_IEEE_eg>;

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
    (outs), (ins), "  GROUP_BARRIER", [(int_AMDGPU_barrier_local)], AnyALU>,
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

  let Defs = [OQAP];
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
  let Defs = [OQAP];
}

class R600_LDS_1A2D <bits<6> lds_op, string name, list<dag> pattern> :
    R600_LDS <
  lds_op,
  (outs),
  (ins R600_Reg32:$src0, REL:$src0_rel, SEL:$src0_sel,
       R600_Reg32:$src1, REL:$src1_rel, SEL:$src1_sel,
       R600_Reg32:$src2, REL:$src2_rel, SEL:$src2_sel,
       LAST:$last, R600_Pred:$pred_sel, BANK_SWIZZLE:$bank_swizzle),
  "  "#name# "$last $src0$src0_rel, $src1$src1_rel, $src2$src2_rel, $pred_sel",
  pattern> {
  let LDS_1A2D = 1;
}

def LDS_ADD : R600_LDS_1A1D_NORET <0x0, "LDS_ADD", [] >;
def LDS_SUB : R600_LDS_1A1D_NORET <0x1, "LDS_SUB", [] >;
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

  def : FROUNDPat <CNDGE_eg>;

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

//===----------------------------------------------------------------------===//
// Regist loads and stores - for indirect addressing
//===----------------------------------------------------------------------===//

defm R600_ : RegisterLoadStore <R600_Reg32, FRAMEri, ADDRIndirect>;

//===----------------------------------------------------------------------===//
// Cayman Instructions
//===----------------------------------------------------------------------===//

let Predicates = [isCayman] in {

def MULADD_INT24_cm : R600_3OP <0x08, "MULADD_INT24",
  [(set i32:$dst, (add (mul I24:$src0, I24:$src1), i32:$src2))], VecALU
>;
def MUL_INT24_cm : R600_2OP <0x5B, "MUL_INT24",
  [(set i32:$dst, (mul I24:$src0, I24:$src1))], VecALU
>;

let isVector = 1 in {

def RECIP_IEEE_cm : RECIP_IEEE_Common<0x86>;

def MULLO_INT_cm : MULLO_INT_Common<0x8F>;
def MULHI_INT_cm : MULHI_INT_Common<0x90>;
def MULLO_UINT_cm : MULLO_UINT_Common<0x91>;
def MULHI_UINT_cm : MULHI_UINT_Common<0x92>;
def RECIPSQRT_CLAMPED_cm : RECIPSQRT_CLAMPED_Common<0x87>;
def EXP_IEEE_cm : EXP_IEEE_Common<0x81>;
def LOG_IEEE_cm : LOG_IEEE_Common<0x83>;
def RECIP_CLAMPED_cm : RECIP_CLAMPED_Common<0x84>;
def RECIPSQRT_IEEE_cm : RECIPSQRT_IEEE_Common<0x89>;
def SIN_cm : SIN_Common<0x8D>;
def COS_cm : COS_Common<0x8E>;
} // End isVector = 1

def : POW_Common <LOG_IEEE_cm, EXP_IEEE_cm, MUL>;

defm DIV_cm : DIV_Common<RECIP_IEEE_cm>;

// RECIP_UINT emulation for Cayman
// The multiplication scales from [0,1] to the unsigned integer range
def : Pat <
  (AMDGPUurecip i32:$src0),
  (FLT_TO_UINT_eg (MUL_IEEE (RECIP_IEEE_cm (UINT_TO_FLT_eg $src0)),
                            (MOV_IMM_I32 CONST.FP_UINT_MAX_PLUS_1)))
>;

  def CF_END_CM : CF_CLAUSE_EG<32, (ins), "CF_END"> {
    let ADDR = 0;
    let POP_COUNT = 0;
    let COUNT = 0;
  }

def : Pat<(fsqrt f32:$src), (MUL R600_Reg32:$src, (RECIPSQRT_CLAMPED_cm $src))>;

class RAT_STORE_DWORD <RegisterClass rc, ValueType vt, bits<4> mask> :
  CF_MEM_RAT_CACHELESS <0x14, 0, mask,
                        (ins rc:$rw_gpr, R600_TReg32_X:$index_gpr),
                        "STORE_DWORD $rw_gpr, $index_gpr",
                        [(global_store vt:$rw_gpr, i32:$index_gpr)]> {
  let eop = 0; // This bit is not used on Cayman.
}

def RAT_STORE_DWORD32 : RAT_STORE_DWORD <R600_TReg32_X, i32, 0x1>;
def RAT_STORE_DWORD64 : RAT_STORE_DWORD <R600_Reg64, v2i32, 0x3>;
def RAT_STORE_DWORD128 : RAT_STORE_DWORD <R600_Reg128, v4i32, 0xf>;

class VTX_READ_cm <string name, bits<8> buffer_id, dag outs, list<dag> pattern>
    : VTX_WORD0_cm, VTX_READ<name, buffer_id, outs, pattern> {

  // Static fields
  let VC_INST = 0;
  let FETCH_TYPE = 2;
  let FETCH_WHOLE_QUAD = 0;
  let BUFFER_ID = buffer_id;
  let SRC_REL = 0;
  // XXX: We can infer this field based on the SRC_GPR.  This would allow us
  // to store vertex addresses in any channel, not just X.
  let SRC_SEL_X = 0;
  let SRC_SEL_Y = 0;
  let STRUCTURED_READ = 0;
  let LDS_REQ = 0;
  let COALESCED_READ = 0;

  let Inst{31-0} = Word0;
}

class VTX_READ_8_cm <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_cm <"VTX_READ_8 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {

  let DST_SEL_X = 0;
  let DST_SEL_Y = 7;   // Masked
  let DST_SEL_Z = 7;   // Masked
  let DST_SEL_W = 7;   // Masked
  let DATA_FORMAT = 1; // FMT_8
}

class VTX_READ_16_cm <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_cm <"VTX_READ_16 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {
  let DST_SEL_X = 0;
  let DST_SEL_Y = 7;   // Masked
  let DST_SEL_Z = 7;   // Masked
  let DST_SEL_W = 7;   // Masked
  let DATA_FORMAT = 5; // FMT_16

}

class VTX_READ_32_cm <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_cm <"VTX_READ_32 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_TReg32_X:$dst_gpr), pattern> {

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

class VTX_READ_64_cm <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_cm <"VTX_READ_64 $dst_gpr, $src_gpr", buffer_id,
                   (outs R600_Reg64:$dst_gpr), pattern> {

  let DST_SEL_X        = 0;
  let DST_SEL_Y        = 1;
  let DST_SEL_Z        = 7;
  let DST_SEL_W        = 7;
  let DATA_FORMAT      = 0x1D; // COLOR_32_32
}

class VTX_READ_128_cm <bits<8> buffer_id, list<dag> pattern>
    : VTX_READ_cm <"VTX_READ_128 $dst_gpr.XYZW, $src_gpr", buffer_id,
                   (outs R600_Reg128:$dst_gpr), pattern> {

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
def VTX_READ_PARAM_8_cm : VTX_READ_8_cm <0,
  [(set i32:$dst_gpr, (load_param_exti8 ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_16_cm : VTX_READ_16_cm <0,
  [(set i32:$dst_gpr, (load_param_exti16 ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_32_cm : VTX_READ_32_cm <0,
  [(set i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_64_cm : VTX_READ_64_cm <0,
  [(set v2i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_PARAM_128_cm : VTX_READ_128_cm <0,
  [(set v4i32:$dst_gpr, (load_param ADDRVTX_READ:$src_gpr))]
>;

//===----------------------------------------------------------------------===//
// VTX Read from global memory space
//===----------------------------------------------------------------------===//

// 8-bit reads
def VTX_READ_GLOBAL_8_cm : VTX_READ_8_cm <1,
  [(set i32:$dst_gpr, (az_extloadi8_global ADDRVTX_READ:$src_gpr))]
>;

def VTX_READ_GLOBAL_16_cm : VTX_READ_16_cm <1,
  [(set i32:$dst_gpr, (az_extloadi16_global ADDRVTX_READ:$src_gpr))]
>;

// 32-bit reads
def VTX_READ_GLOBAL_32_cm : VTX_READ_32_cm <1,
  [(set i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

// 64-bit reads
def VTX_READ_GLOBAL_64_cm : VTX_READ_64_cm <1,
  [(set v2i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

// 128-bit reads
def VTX_READ_GLOBAL_128_cm : VTX_READ_128_cm <1,
  [(set v4i32:$dst_gpr, (global_load ADDRVTX_READ:$src_gpr))]
>;

} // End isCayman

//===----------------------------------------------------------------------===//
// Branch Instructions
//===----------------------------------------------------------------------===//


def IF_PREDICATE_SET  : ILFormat<(outs), (ins GPRI32:$src),
  "IF_PREDICATE_SET $src", []>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

let isPseudo = 1 in {

def PRED_X : InstR600 <
  (outs R600_Predicate_Bit:$dst),
  (ins R600_Reg32:$src0, i32imm:$src1, i32imm:$flags),
  "", [], NullALU> {
  let FlagOperandIdx = 3;
}

let isTerminator = 1, isBranch = 1 in {
def JUMP_COND : InstR600 <
          (outs),
          (ins brtarget:$target, R600_Predicate_Bit:$p),
          "JUMP $target ($p)",
          [], AnyALU
  >;

def JUMP : InstR600 <
          (outs),
          (ins brtarget:$target),
          "JUMP $target",
          [], AnyALU
  >
{
  let isPredicable = 1;
  let isBarrier = 1;
}

}  // End isTerminator = 1, isBranch = 1

let usesCustomInserter = 1 in {

let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in {

def MASK_WRITE : AMDGPUShaderInst <
    (outs),
    (ins R600_Reg32:$src),
    "MASK_WRITE $src",
    []
>;

} // End mayLoad = 0, mayStore = 0, hasSideEffects = 1


def TXD: InstR600 <
  (outs R600_Reg128:$dst),
  (ins R600_Reg128:$src0, R600_Reg128:$src1, R600_Reg128:$src2,
       i32imm:$resourceId, i32imm:$samplerId, i32imm:$textureTarget),
  "TXD $dst, $src0, $src1, $src2, $resourceId, $samplerId, $textureTarget",
  [(set v4f32:$dst, (int_AMDGPU_txd v4f32:$src0, v4f32:$src1, v4f32:$src2,
                     imm:$resourceId, imm:$samplerId, imm:$textureTarget))],
  NullALU > {
  let TEXInst = 1;
}

def TXD_SHADOW: InstR600 <
  (outs R600_Reg128:$dst),
  (ins R600_Reg128:$src0, R600_Reg128:$src1, R600_Reg128:$src2,
       i32imm:$resourceId, i32imm:$samplerId, i32imm:$textureTarget),
  "TXD_SHADOW $dst, $src0, $src1, $src2, $resourceId, $samplerId, $textureTarget",
  [(set v4f32:$dst, (int_AMDGPU_txd v4f32:$src0, v4f32:$src1, v4f32:$src2,
        imm:$resourceId, imm:$samplerId, TEX_SHADOW:$textureTarget))],
   NullALU
> {
  let TEXInst = 1;
}
} // End isPseudo = 1
} // End usesCustomInserter = 1

//===---------------------------------------------------------------------===//
// Return instruction
//===---------------------------------------------------------------------===//
let isTerminator = 1, isReturn = 1, hasCtrlDep = 1,
    usesCustomInserter = 1 in {
  def RETURN          : ILFormat<(outs), (ins variable_ops),
      "RETURN", [(IL_retflag)]>;
}


//===----------------------------------------------------------------------===//
// Constant Buffer Addressing Support
//===----------------------------------------------------------------------===//

let usesCustomInserter = 1, isCodeGenOnly = 1, isPseudo = 1, Namespace = "AMDGPU"  in {
def CONST_COPY : Instruction {
  let OutOperandList = (outs R600_Reg32:$dst);
  let InOperandList = (ins i32imm:$src);
  let Pattern =
      [(set R600_Reg32:$dst, (CONST_ADDRESS ADDRGA_CONST_OFFSET:$src))];
  let AsmString = "CONST_COPY";
  let neverHasSideEffects = 1;
  let isAsCheapAsAMove = 1;
  let Itinerary = NullALU;
}
} // end usesCustomInserter = 1, isCodeGenOnly = 1, isPseudo = 1, Namespace = "AMDGPU"

def TEX_VTX_CONSTBUF :
  InstR600ISA <(outs R600_Reg128:$dst), (ins MEMxi:$ptr, i32imm:$BUFFER_ID), "VTX_READ_eg $dst, $ptr",
      [(set v4i32:$dst, (CONST_ADDRESS ADDRGA_VAR_OFFSET:$ptr, (i32 imm:$BUFFER_ID)))]>,
  VTX_WORD1_GPR, VTX_WORD0_eg {

  let VC_INST = 0;
  let FETCH_TYPE = 2;
  let FETCH_WHOLE_QUAD = 0;
  let SRC_REL = 0;
  let SRC_SEL_X = 0;
  let DST_REL = 0;
  let USE_CONST_FIELDS = 0;
  let NUM_FORMAT_ALL = 2;
  let FORMAT_COMP_ALL = 1;
  let SRF_MODE_ALL = 1;
  let MEGA_FETCH_COUNT = 16;
  let DST_SEL_X        = 0;
  let DST_SEL_Y        = 1;
  let DST_SEL_Z        = 2;
  let DST_SEL_W        = 3;
  let DATA_FORMAT      = 35;

  let Inst{31-0} = Word0;
  let Inst{63-32} = Word1;

// LLVM can only encode 64-bit instructions, so these fields are manually
// encoded in R600CodeEmitter
//
// bits<16> OFFSET;
// bits<2>  ENDIAN_SWAP = 0;
// bits<1>  CONST_BUF_NO_STRIDE = 0;
// bits<1>  MEGA_FETCH = 0;
// bits<1>  ALT_CONST = 0;
// bits<2>  BUFFER_INDEX_MODE = 0;



// VTX_WORD2 (LLVM can only encode 64-bit instructions, so WORD2 encoding
// is done in R600CodeEmitter
//
// Inst{79-64} = OFFSET;
// Inst{81-80} = ENDIAN_SWAP;
// Inst{82}    = CONST_BUF_NO_STRIDE;
// Inst{83}    = MEGA_FETCH;
// Inst{84}    = ALT_CONST;
// Inst{86-85} = BUFFER_INDEX_MODE;
// Inst{95-86} = 0; Reserved

// VTX_WORD3 (Padding)
//
// Inst{127-96} = 0;
  let VTXInst = 1;
}

def TEX_VTX_TEXBUF:
  InstR600ISA <(outs R600_Reg128:$dst), (ins MEMxi:$ptr, i32imm:$BUFFER_ID), "TEX_VTX_EXPLICIT_READ $dst, $ptr",
      [(set v4f32:$dst, (int_R600_load_texbuf ADDRGA_VAR_OFFSET:$ptr, imm:$BUFFER_ID))]>,
VTX_WORD1_GPR, VTX_WORD0_eg {

let VC_INST = 0;
let FETCH_TYPE = 2;
let FETCH_WHOLE_QUAD = 0;
let SRC_REL = 0;
let SRC_SEL_X = 0;
let DST_REL = 0;
let USE_CONST_FIELDS = 1;
let NUM_FORMAT_ALL = 0;
let FORMAT_COMP_ALL = 0;
let SRF_MODE_ALL = 1;
let MEGA_FETCH_COUNT = 16;
let DST_SEL_X        = 0;
let DST_SEL_Y        = 1;
let DST_SEL_Z        = 2;
let DST_SEL_W        = 3;
let DATA_FORMAT      = 0;

let Inst{31-0} = Word0;
let Inst{63-32} = Word1;

// LLVM can only encode 64-bit instructions, so these fields are manually
// encoded in R600CodeEmitter
//
// bits<16> OFFSET;
// bits<2>  ENDIAN_SWAP = 0;
// bits<1>  CONST_BUF_NO_STRIDE = 0;
// bits<1>  MEGA_FETCH = 0;
// bits<1>  ALT_CONST = 0;
// bits<2>  BUFFER_INDEX_MODE = 0;



// VTX_WORD2 (LLVM can only encode 64-bit instructions, so WORD2 encoding
// is done in R600CodeEmitter
//
// Inst{79-64} = OFFSET;
// Inst{81-80} = ENDIAN_SWAP;
// Inst{82}    = CONST_BUF_NO_STRIDE;
// Inst{83}    = MEGA_FETCH;
// Inst{84}    = ALT_CONST;
// Inst{86-85} = BUFFER_INDEX_MODE;
// Inst{95-86} = 0; Reserved

// VTX_WORD3 (Padding)
//
// Inst{127-96} = 0;
  let VTXInst = 1;
}



//===--------------------------------------------------------------------===//
// Instructions support
//===--------------------------------------------------------------------===//
//===---------------------------------------------------------------------===//
// Custom Inserter for Branches and returns, this eventually will be a
// seperate pass
//===---------------------------------------------------------------------===//
let isTerminator = 1, usesCustomInserter = 1, isBranch = 1, isBarrier = 1 in {
  def BRANCH : ILFormat<(outs), (ins brtarget:$target),
      "; Pseudo unconditional branch instruction",
      [(br bb:$target)]>;
  defm BRANCH_COND : BranchConditional<IL_brcond, R600_Reg32, R600_Reg32>;
}

//===---------------------------------------------------------------------===//
// Flow and Program control Instructions
//===---------------------------------------------------------------------===//
let isTerminator=1 in {
  def SWITCH      : ILFormat< (outs), (ins GPRI32:$src),
  !strconcat("SWITCH", " $src"), []>;
  def CASE        : ILFormat< (outs), (ins GPRI32:$src),
      !strconcat("CASE", " $src"), []>;
  def BREAK       : ILFormat< (outs), (ins),
      "BREAK", []>;
  def CONTINUE    : ILFormat< (outs), (ins),
      "CONTINUE", []>;
  def DEFAULT     : ILFormat< (outs), (ins),
      "DEFAULT", []>;
  def ELSE        : ILFormat< (outs), (ins),
      "ELSE", []>;
  def ENDSWITCH   : ILFormat< (outs), (ins),
      "ENDSWITCH", []>;
  def ENDMAIN     : ILFormat< (outs), (ins),
      "ENDMAIN", []>;
  def END         : ILFormat< (outs), (ins),
      "END", []>;
  def ENDFUNC     : ILFormat< (outs), (ins),
      "ENDFUNC", []>;
  def ENDIF       : ILFormat< (outs), (ins),
      "ENDIF", []>;
  def WHILELOOP   : ILFormat< (outs), (ins),
      "WHILE", []>;
  def ENDLOOP     : ILFormat< (outs), (ins),
      "ENDLOOP", []>;
  def FUNC        : ILFormat< (outs), (ins),
      "FUNC", []>;
  def RETDYN      : ILFormat< (outs), (ins),
      "RET_DYN", []>;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm IF_LOGICALNZ  : BranchInstr<"IF_LOGICALNZ">;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm IF_LOGICALZ   : BranchInstr<"IF_LOGICALZ">;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm BREAK_LOGICALNZ : BranchInstr<"BREAK_LOGICALNZ">;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm BREAK_LOGICALZ : BranchInstr<"BREAK_LOGICALZ">;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm CONTINUE_LOGICALNZ : BranchInstr<"CONTINUE_LOGICALNZ">;
  // This opcode has custom swizzle pattern encoded in Swizzle Encoder
  defm CONTINUE_LOGICALZ : BranchInstr<"CONTINUE_LOGICALZ">;
  defm IFC         : BranchInstr2<"IFC">;
  defm BREAKC      : BranchInstr2<"BREAKC">;
  defm CONTINUEC   : BranchInstr2<"CONTINUEC">;
}

//===----------------------------------------------------------------------===//
// ISel Patterns
//===----------------------------------------------------------------------===//

// CND*_INT Pattterns for f32 True / False values

class CND_INT_f32 <InstR600 cnd, CondCode cc> : Pat <
  (selectcc i32:$src0, 0, f32:$src1, f32:$src2, cc),
  (cnd $src0, $src1, $src2)
>;

def : CND_INT_f32 <CNDE_INT,  SETEQ>;
def : CND_INT_f32 <CNDGT_INT, SETGT>;
def : CND_INT_f32 <CNDGE_INT, SETGE>;

//CNDGE_INT extra pattern
def : Pat <
  (selectcc i32:$src0, -1, i32:$src1, i32:$src2, COND_SGT),
  (CNDGE_INT $src0, $src1, $src2)
>;

// KIL Patterns
def KILP : Pat <
  (int_AMDGPU_kilp),
  (MASK_WRITE (KILLGT (f32 ONE), (f32 ZERO)))
>;

def KIL : Pat <
  (int_AMDGPU_kill f32:$src0),
  (MASK_WRITE (KILLGT (f32 ZERO), $src0))
>;

def : Extract_Element <f32, v4f32, 0, sub0>;
def : Extract_Element <f32, v4f32, 1, sub1>;
def : Extract_Element <f32, v4f32, 2, sub2>;
def : Extract_Element <f32, v4f32, 3, sub3>;

def : Insert_Element <f32, v4f32, 0, sub0>;
def : Insert_Element <f32, v4f32, 1, sub1>;
def : Insert_Element <f32, v4f32, 2, sub2>;
def : Insert_Element <f32, v4f32, 3, sub3>;

def : Extract_Element <i32, v4i32, 0, sub0>;
def : Extract_Element <i32, v4i32, 1, sub1>;
def : Extract_Element <i32, v4i32, 2, sub2>;
def : Extract_Element <i32, v4i32, 3, sub3>;

def : Insert_Element <i32, v4i32, 0, sub0>;
def : Insert_Element <i32, v4i32, 1, sub1>;
def : Insert_Element <i32, v4i32, 2, sub2>;
def : Insert_Element <i32, v4i32, 3, sub3>;

def : Vector4_Build <v4f32, f32>;
def : Vector4_Build <v4i32, i32>;

def : Extract_Element <f32, v2f32, 0, sub0>;
def : Extract_Element <f32, v2f32, 1, sub1>;

def : Insert_Element <f32, v2f32, 0, sub0>;
def : Insert_Element <f32, v2f32, 1, sub1>;

def : Extract_Element <i32, v2i32, 0, sub0>;
def : Extract_Element <i32, v2i32, 1, sub1>;

def : Insert_Element <i32, v2i32, 0, sub0>;
def : Insert_Element <i32, v2i32, 1, sub1>;

// bitconvert patterns

def : BitConvert <i32, f32, R600_Reg32>;
def : BitConvert <f32, i32, R600_Reg32>;
def : BitConvert <v2f32, v2i32, R600_Reg64>;
def : BitConvert <v2i32, v2f32, R600_Reg64>;
def : BitConvert <v4f32, v4i32, R600_Reg128>;
def : BitConvert <v4i32, v4f32, R600_Reg128>;

// DWORDADDR pattern
def : DwordAddrPat  <i32, R600_Reg32>;

} // End isR600toCayman Predicate

def getLDSNoRetOp : InstrMapping {
  let FilterClass = "R600_LDS_1A1D";
  let RowFields = ["BaseOp"];
  let ColFields = ["DisableEncoding"];
  let KeyCol = ["$dst"];
  let ValueCols = [[""""]];
}
