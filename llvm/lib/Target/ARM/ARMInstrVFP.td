//===- ARMInstrVFP.td - VFP support for ARM -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM VFP instruction set.
//
//===----------------------------------------------------------------------===//

def SDT_FTOI :
SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDT_ITOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;
def SDT_CMPFP0 :
SDTypeProfile<0, 1, [SDTCisFP<0>]>;
def SDT_VMOVDRR :
SDTypeProfile<1, 2, [SDTCisVT<0, f64>, SDTCisVT<1, i32>,
                     SDTCisSameAs<1, 2>]>;

def arm_ftoui  : SDNode<"ARMISD::FTOUI",  SDT_FTOI>;
def arm_ftosi  : SDNode<"ARMISD::FTOSI",  SDT_FTOI>;
def arm_sitof  : SDNode<"ARMISD::SITOF",  SDT_ITOF>;
def arm_uitof  : SDNode<"ARMISD::UITOF",  SDT_ITOF>;
def arm_fmstat : SDNode<"ARMISD::FMSTAT", SDTNone, [SDNPInFlag,SDNPOutFlag]>;
def arm_cmpfp  : SDNode<"ARMISD::CMPFP",  SDT_ARMCmp, [SDNPOutFlag]>;
def arm_cmpfp0 : SDNode<"ARMISD::CMPFPw0",SDT_CMPFP0, [SDNPOutFlag]>;
def arm_fmdrr  : SDNode<"ARMISD::VMOVDRR",  SDT_VMOVDRR>;

//===----------------------------------------------------------------------===//
// Operand Definitions.
//


def vfp_f32imm : Operand<f32>,
                 PatLeaf<(f32 fpimm), [{
      return ARM::getVFPf32Imm(N->getValueAPF()) != -1;
    }]> {
  let PrintMethod = "printVFPf32ImmOperand";
}

def vfp_f64imm : Operand<f64>,
                 PatLeaf<(f64 fpimm), [{
      return ARM::getVFPf64Imm(N->getValueAPF()) != -1;
    }]> {
  let PrintMethod = "printVFPf64ImmOperand";
}


//===----------------------------------------------------------------------===//
//  Load / store Instructions.
//

let canFoldAsLoad = 1, isReMaterializable = 1 in {
def VLDRD : ADI5<0b1101, 0b01, (outs DPR:$dst), (ins addrmode5:$addr),
                 IIC_fpLoad64, "vldr", ".64\t$dst, $addr",
                 [(set DPR:$dst, (f64 (load addrmode5:$addr)))]>;

def VLDRS : ASI5<0b1101, 0b01, (outs SPR:$dst), (ins addrmode5:$addr),
                 IIC_fpLoad32, "vldr", ".32\t$dst, $addr",
                 [(set SPR:$dst, (load addrmode5:$addr))]>;
} // canFoldAsLoad

def VSTRD  : ADI5<0b1101, 0b00, (outs), (ins DPR:$src, addrmode5:$addr),
                 IIC_fpStore64, "vstr", ".64\t$src, $addr",
                 [(store (f64 DPR:$src), addrmode5:$addr)]>;

def VSTRS  : ASI5<0b1101, 0b00, (outs), (ins SPR:$src, addrmode5:$addr),
                 IIC_fpStore32, "vstr", ".32\t$src, $addr",
                 [(store SPR:$src, addrmode5:$addr)]>;

//===----------------------------------------------------------------------===//
//  Load / store multiple Instructions.
//

let mayLoad = 1, neverHasSideEffects = 1, hasExtraDefRegAllocReq = 1 in {
def VLDMD : AXDI4<(outs), (ins addrmode4:$addr, pred:$p, reglist:$dsts,
                           variable_ops), IndexModeNone, IIC_fpLoadm,
                  "vldm${addr:submode}${p}\t$addr, $dsts", "", []> {
  let Inst{20} = 1;
}

def VLDMS : AXSI4<(outs), (ins addrmode4:$addr, pred:$p, reglist:$dsts,
                           variable_ops), IndexModeNone, IIC_fpLoadm,
                  "vldm${addr:submode}${p}\t$addr, $dsts", "", []> {
  let Inst{20} = 1;
}

def VLDMD_UPD : AXDI4<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$dsts, variable_ops),
                      IndexModeUpd, IIC_fpLoadm,
                      "vldm${addr:submode}${p}\t$addr!, $dsts",
                      "$addr.addr = $wb", []> {
  let Inst{20} = 1;
}

def VLDMS_UPD : AXSI4<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$dsts, variable_ops),
                      IndexModeUpd, IIC_fpLoadm, 
                      "vldm${addr:submode}${p}\t$addr!, $dsts",
                      "$addr.addr = $wb", []> {
  let Inst{20} = 1;
}
} // mayLoad, neverHasSideEffects, hasExtraDefRegAllocReq

let mayStore = 1, neverHasSideEffects = 1, hasExtraSrcRegAllocReq = 1 in {
def VSTMD : AXDI4<(outs), (ins addrmode4:$addr, pred:$p, reglist:$srcs,
                           variable_ops), IndexModeNone, IIC_fpStorem,
                  "vstm${addr:submode}${p}\t$addr, $srcs", "", []> {
  let Inst{20} = 0;
}

def VSTMS : AXSI4<(outs), (ins addrmode4:$addr, pred:$p, reglist:$srcs,
                           variable_ops), IndexModeNone, IIC_fpStorem,
                  "vstm${addr:submode}${p}\t$addr, $srcs", "", []> {
  let Inst{20} = 0;
}

def VSTMD_UPD : AXDI4<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$srcs, variable_ops),
                      IndexModeUpd, IIC_fpStorem,
                      "vstm${addr:submode}${p}\t$addr!, $srcs",
                      "$addr.addr = $wb", []> {
  let Inst{20} = 0;
}

def VSTMS_UPD : AXSI4<(outs GPR:$wb), (ins addrmode4:$addr, pred:$p,
                                       reglist:$srcs, variable_ops),
                      IndexModeUpd, IIC_fpStorem,
                      "vstm${addr:submode}${p}\t$addr!, $srcs",
                      "$addr.addr = $wb", []> {
  let Inst{20} = 0;
}
} // mayStore, neverHasSideEffects, hasExtraSrcRegAllocReq

// FLDMX, FSTMX - mixing S/D registers for pre-armv6 cores

//===----------------------------------------------------------------------===//
// FP Binary Operations.
//

def VADDD  : ADbI<0b11100, 0b11, 0, 0, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpALU64, "vadd", ".f64\t$dst, $a, $b",
                 [(set DPR:$dst, (fadd DPR:$a, (f64 DPR:$b)))]>;

def VADDS  : ASbIn<0b11100, 0b11, 0, 0, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpALU32, "vadd", ".f32\t$dst, $a, $b",
                  [(set SPR:$dst, (fadd SPR:$a, SPR:$b))]>;

// These are encoded as unary instructions.
let Defs = [FPSCR] in {
def VCMPED : ADuI<0b11101, 0b11, 0b0100, 0b11, 0, (outs), (ins DPR:$a, DPR:$b),
                 IIC_fpCMP64, "vcmpe", ".f64\t$a, $b",
                 [(arm_cmpfp DPR:$a, (f64 DPR:$b))]>;

def VCMPD  : ADuI<0b11101, 0b11, 0b0100, 0b01, 0, (outs), (ins DPR:$a, DPR:$b),
                 IIC_fpCMP64, "vcmp", ".f64\t$a, $b",
                 [/* For disassembly only; pattern left blank */]>;

def VCMPES : ASuI<0b11101, 0b11, 0b0100, 0b11, 0, (outs), (ins SPR:$a, SPR:$b),
                 IIC_fpCMP32, "vcmpe", ".f32\t$a, $b",
                 [(arm_cmpfp SPR:$a, SPR:$b)]>;

def VCMPS  : ASuI<0b11101, 0b11, 0b0100, 0b01, 0, (outs), (ins SPR:$a, SPR:$b),
                 IIC_fpCMP32, "vcmp", ".f32\t$a, $b",
                 [/* For disassembly only; pattern left blank */]>;
}

def VDIVD  : ADbI<0b11101, 0b00, 0, 0, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpDIV64, "vdiv", ".f64\t$dst, $a, $b",
                 [(set DPR:$dst, (fdiv DPR:$a, (f64 DPR:$b)))]>;

def VDIVS  : ASbI<0b11101, 0b00, 0, 0, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                 IIC_fpDIV32, "vdiv", ".f32\t$dst, $a, $b",
                 [(set SPR:$dst, (fdiv SPR:$a, SPR:$b))]>;

def VMULD  : ADbI<0b11100, 0b10, 0, 0, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpMUL64, "vmul", ".f64\t$dst, $a, $b",
                 [(set DPR:$dst, (fmul DPR:$a, (f64 DPR:$b)))]>;

def VMULS  : ASbIn<0b11100, 0b10, 0, 0, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpMUL32, "vmul", ".f32\t$dst, $a, $b",
                  [(set SPR:$dst, (fmul SPR:$a, SPR:$b))]>;

def VNMULD  : ADbI<0b11100, 0b10, 1, 0, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                  IIC_fpMUL64, "vnmul", ".f64\t$dst, $a, $b",
                  [(set DPR:$dst, (fneg (fmul DPR:$a, (f64 DPR:$b))))]>;

def VNMULS  : ASbI<0b11100, 0b10, 1, 0, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpMUL32, "vnmul", ".f32\t$dst, $a, $b",
                  [(set SPR:$dst, (fneg (fmul SPR:$a, SPR:$b)))]>;

// Match reassociated forms only if not sign dependent rounding.
def : Pat<(fmul (fneg DPR:$a), (f64 DPR:$b)),
          (VNMULD DPR:$a, DPR:$b)>, Requires<[NoHonorSignDependentRounding]>;
def : Pat<(fmul (fneg SPR:$a), SPR:$b),
          (VNMULS SPR:$a, SPR:$b)>, Requires<[NoHonorSignDependentRounding]>;


def VSUBD  : ADbI<0b11100, 0b11, 1, 0, (outs DPR:$dst), (ins DPR:$a, DPR:$b),
                 IIC_fpALU64, "vsub", ".f64\t$dst, $a, $b",
                 [(set DPR:$dst, (fsub DPR:$a, (f64 DPR:$b)))]>;

def VSUBS  : ASbIn<0b11100, 0b11, 1, 0, (outs SPR:$dst), (ins SPR:$a, SPR:$b),
                  IIC_fpALU32, "vsub", ".f32\t$dst, $a, $b",
                  [(set SPR:$dst, (fsub SPR:$a, SPR:$b))]>;

//===----------------------------------------------------------------------===//
// FP Unary Operations.
//

def VABSD  : ADuI<0b11101, 0b11, 0b0000, 0b11, 0, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "vabs", ".f64\t$dst, $a",
                 [(set DPR:$dst, (fabs (f64 DPR:$a)))]>;

def VABSS  : ASuIn<0b11101, 0b11, 0b0000, 0b11, 0,(outs SPR:$dst), (ins SPR:$a),
                  IIC_fpUNA32, "vabs", ".f32\t$dst, $a",
                  [(set SPR:$dst, (fabs SPR:$a))]>;

let Defs = [FPSCR] in {
def VCMPEZD : ADuI<0b11101, 0b11, 0b0101, 0b11, 0, (outs), (ins DPR:$a),
                  IIC_fpCMP64, "vcmpe", ".f64\t$a, #0",
                  [(arm_cmpfp0 (f64 DPR:$a))]>;

def VCMPZD  : ADuI<0b11101, 0b11, 0b0101, 0b01, 0, (outs), (ins DPR:$a),
                  IIC_fpCMP64, "vcmp", ".f64\t$a, #0",
                  [/* For disassembly only; pattern left blank */]>;

def VCMPEZS : ASuI<0b11101, 0b11, 0b0101, 0b11, 0, (outs), (ins SPR:$a),
                  IIC_fpCMP32, "vcmpe", ".f32\t$a, #0",
                  [(arm_cmpfp0 SPR:$a)]>;

def VCMPZS  : ASuI<0b11101, 0b11, 0b0101, 0b01, 0, (outs), (ins SPR:$a),
                  IIC_fpCMP32, "vcmp", ".f32\t$a, #0",
                  [/* For disassembly only; pattern left blank */]>;
}

def VCVTDS : ASuI<0b11101, 0b11, 0b0111, 0b11, 0, (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTDS, "vcvt", ".f64.f32\t$dst, $a",
                 [(set DPR:$dst, (fextend SPR:$a))]>;

// Special case encoding: bits 11-8 is 0b1011.
def VCVTSD : VFPAI<(outs SPR:$dst), (ins DPR:$a), VFPUnaryFrm,
                   IIC_fpCVTSD, "vcvt", ".f32.f64\t$dst, $a",
                   [(set SPR:$dst, (fround DPR:$a))]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-16} = 0b110111;
  let Inst{11-8}  = 0b1011;
  let Inst{7-6}   = 0b11;
  let Inst{4}     = 0;
}

// Between half-precision and single-precision.  For disassembly only.

def VCVTBSH: ASuI<0b11101, 0b11, 0b0010, 0b01, 0, (outs SPR:$dst), (ins SPR:$a),
                 /* FIXME */ IIC_fpCVTSH, "vcvtb", ".f32.f16\t$dst, $a",
                 [/* For disassembly only; pattern left blank */]>;

def : ARMPat<(f32_to_f16 SPR:$a),
             (i32 (COPY_TO_REGCLASS (VCVTBSH SPR:$a), GPR))>;

def VCVTBHS: ASuI<0b11101, 0b11, 0b0011, 0b01, 0, (outs SPR:$dst), (ins SPR:$a),
                 /* FIXME */ IIC_fpCVTHS, "vcvtb", ".f16.f32\t$dst, $a",
                 [/* For disassembly only; pattern left blank */]>;

def : ARMPat<(f16_to_f32 GPR:$a),
             (VCVTBHS (COPY_TO_REGCLASS GPR:$a, SPR))>;

def VCVTTSH: ASuI<0b11101, 0b11, 0b0010, 0b11, 0, (outs SPR:$dst), (ins SPR:$a),
                 /* FIXME */ IIC_fpCVTSH, "vcvtt", ".f32.f16\t$dst, $a",
                 [/* For disassembly only; pattern left blank */]>;

def VCVTTHS: ASuI<0b11101, 0b11, 0b0011, 0b11, 0, (outs SPR:$dst), (ins SPR:$a),
                 /* FIXME */ IIC_fpCVTHS, "vcvtt", ".f16.f32\t$dst, $a",
                 [/* For disassembly only; pattern left blank */]>;

let neverHasSideEffects = 1 in {
def VMOVD: ADuI<0b11101, 0b11, 0b0000, 0b01, 0, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "vmov", ".f64\t$dst, $a", []>;

def VMOVS: ASuI<0b11101, 0b11, 0b0000, 0b01, 0, (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpUNA32, "vmov", ".f32\t$dst, $a", []>;
} // neverHasSideEffects

def VNEGD  : ADuI<0b11101, 0b11, 0b0001, 0b01, 0, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpUNA64, "vneg", ".f64\t$dst, $a",
                 [(set DPR:$dst, (fneg (f64 DPR:$a)))]>;

def VNEGS  : ASuIn<0b11101, 0b11, 0b0001, 0b01, 0,(outs SPR:$dst), (ins SPR:$a),
                  IIC_fpUNA32, "vneg", ".f32\t$dst, $a",
                  [(set SPR:$dst, (fneg SPR:$a))]>;

def VSQRTD : ADuI<0b11101, 0b11, 0b0001, 0b11, 0, (outs DPR:$dst), (ins DPR:$a),
                 IIC_fpSQRT64, "vsqrt", ".f64\t$dst, $a",
                 [(set DPR:$dst, (fsqrt (f64 DPR:$a)))]>;

def VSQRTS : ASuI<0b11101, 0b11, 0b0001, 0b11, 0, (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpSQRT32, "vsqrt", ".f32\t$dst, $a",
                 [(set SPR:$dst, (fsqrt SPR:$a))]>;

//===----------------------------------------------------------------------===//
// FP <-> GPR Copies.  Int <-> FP Conversions.
//

def VMOVRS : AVConv2I<0b11100001, 0b1010, (outs GPR:$dst), (ins SPR:$src),
                 IIC_fpMOVSI, "vmov", "\t$dst, $src",
                 [(set GPR:$dst, (bitconvert SPR:$src))]>;

def VMOVSR : AVConv4I<0b11100000, 0b1010, (outs SPR:$dst), (ins GPR:$src),
                 IIC_fpMOVIS, "vmov", "\t$dst, $src",
                 [(set SPR:$dst, (bitconvert GPR:$src))]>;

let neverHasSideEffects = 1 in {
def VMOVRRD  : AVConv3I<0b11000101, 0b1011,
                      (outs GPR:$wb, GPR:$dst2), (ins DPR:$src),
                 IIC_fpMOVDI, "vmov", "\t$wb, $dst2, $src",
                 [/* FIXME: Can't write pattern for multiple result instr*/]> {
  let Inst{7-6} = 0b00;
}

def VMOVRRS  : AVConv3I<0b11000101, 0b1010,
                      (outs GPR:$wb, GPR:$dst2), (ins SPR:$src1, SPR:$src2),
                 IIC_fpMOVDI, "vmov", "\t$wb, $dst2, $src1, $src2",
                 [/* For disassembly only; pattern left blank */]> {
  let Inst{7-6} = 0b00;
}
} // neverHasSideEffects

// FMDHR: GPR -> SPR
// FMDLR: GPR -> SPR

def VMOVDRR : AVConv5I<0b11000100, 0b1011,
                     (outs DPR:$dst), (ins GPR:$src1, GPR:$src2),
                IIC_fpMOVID, "vmov", "\t$dst, $src1, $src2",
                [(set DPR:$dst, (arm_fmdrr GPR:$src1, GPR:$src2))]> {
  let Inst{7-6} = 0b00;
}

let neverHasSideEffects = 1 in
def VMOVSRR : AVConv5I<0b11000100, 0b1010,
                     (outs SPR:$dst1, SPR:$dst2), (ins GPR:$src1, GPR:$src2),
                IIC_fpMOVID, "vmov", "\t$dst1, $dst2, $src1, $src2",
                [/* For disassembly only; pattern left blank */]> {
  let Inst{7-6} = 0b00;
}

// FMRDH: SPR -> GPR
// FMRDL: SPR -> GPR
// FMRRS: SPR -> GPR
// FMRX : SPR system reg -> GPR

// FMSRR: GPR -> SPR

// FMXR: GPR -> VFP Sstem reg


// Int to FP:

def VSITOD : AVConv1I<0b11101, 0b11, 0b1000, 0b1011,
                 (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTID, "vcvt", ".f64.s32\t$dst, $a",
                 [(set DPR:$dst, (f64 (arm_sitof SPR:$a)))]> {
  let Inst{7} = 1; // s32
}

def VSITOS : AVConv1In<0b11101, 0b11, 0b1000, 0b1010,
                 (outs SPR:$dst),(ins SPR:$a),
                 IIC_fpCVTIS, "vcvt", ".f32.s32\t$dst, $a",
                 [(set SPR:$dst, (arm_sitof SPR:$a))]> {
  let Inst{7} = 1; // s32
}

def VUITOD : AVConv1I<0b11101, 0b11, 0b1000, 0b1011,
                 (outs DPR:$dst), (ins SPR:$a),
                 IIC_fpCVTID, "vcvt", ".f64.u32\t$dst, $a",
                 [(set DPR:$dst, (f64 (arm_uitof SPR:$a)))]> {
  let Inst{7} = 0; // u32
}

def VUITOS : AVConv1In<0b11101, 0b11, 0b1000, 0b1010,
                 (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTIS, "vcvt", ".f32.u32\t$dst, $a",
                 [(set SPR:$dst, (arm_uitof SPR:$a))]> {
  let Inst{7} = 0; // u32
}

// FP to Int:
// Always set Z bit in the instruction, i.e. "round towards zero" variants.

def VTOSIZD : AVConv1I<0b11101, 0b11, 0b1101, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "vcvt", ".s32.f64\t$dst, $a",
                 [(set SPR:$dst, (arm_ftosi (f64 DPR:$a)))]> {
  let Inst{7} = 1; // Z bit
}

def VTOSIZS : AVConv1In<0b11101, 0b11, 0b1101, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "vcvt", ".s32.f32\t$dst, $a",
                 [(set SPR:$dst, (arm_ftosi SPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

def VTOUIZD : AVConv1I<0b11101, 0b11, 0b1100, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "vcvt", ".u32.f64\t$dst, $a",
                 [(set SPR:$dst, (arm_ftoui (f64 DPR:$a)))]> {
  let Inst{7} = 1; // Z bit
}

def VTOUIZS : AVConv1In<0b11101, 0b11, 0b1100, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "vcvt", ".u32.f32\t$dst, $a",
                 [(set SPR:$dst, (arm_ftoui SPR:$a))]> {
  let Inst{7} = 1; // Z bit
}

// And the Z bit '0' variants, i.e. use the rounding mode specified by FPSCR.
// For disassembly only.
let Uses = [FPSCR] in {
def VTOSIRD : AVConv1I<0b11101, 0b11, 0b1101, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "vcvtr", ".s32.f64\t$dst, $a",
                 [(set SPR:$dst, (int_arm_vcvtr (f64 DPR:$a)))]> {
  let Inst{7} = 0; // Z bit
}

def VTOSIRS : AVConv1In<0b11101, 0b11, 0b1101, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "vcvtr", ".s32.f32\t$dst, $a",
                 [(set SPR:$dst, (int_arm_vcvtr SPR:$a))]> {
  let Inst{7} = 0; // Z bit
}

def VTOUIRD : AVConv1I<0b11101, 0b11, 0b1100, 0b1011,
                       (outs SPR:$dst), (ins DPR:$a),
                 IIC_fpCVTDI, "vcvtr", ".u32.f64\t$dst, $a",
                 [(set SPR:$dst, (int_arm_vcvtru (f64 DPR:$a)))]> {
  let Inst{7} = 0; // Z bit
}

def VTOUIRS : AVConv1In<0b11101, 0b11, 0b1100, 0b1010,
                        (outs SPR:$dst), (ins SPR:$a),
                 IIC_fpCVTSI, "vcvtr", ".u32.f32\t$dst, $a",
                 [(set SPR:$dst, (int_arm_vcvtru SPR:$a))]> {
  let Inst{7} = 0; // Z bit
}
}

// Convert between floating-point and fixed-point
// Data type for fixed-point naming convention:
//   S16 (U=0, sx=0) -> SH
//   U16 (U=1, sx=0) -> UH
//   S32 (U=0, sx=1) -> SL
//   U32 (U=1, sx=1) -> UL

let Constraints = "$a = $dst" in {

// FP to Fixed-Point:

let isCodeGenOnly = 1 in {
def VTOSHS : AVConv1XI<0b11101, 0b11, 0b1110, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".s16.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOUHS : AVConv1XI<0b11101, 0b11, 0b1111, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".u16.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOSLS : AVConv1XI<0b11101, 0b11, 0b1110, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".s32.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOULS : AVConv1XI<0b11101, 0b11, 0b1111, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTSI, "vcvt", ".u32.f32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOSHD : AVConv1XI<0b11101, 0b11, 0b1110, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".s16.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOUHD : AVConv1XI<0b11101, 0b11, 0b1111, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".u16.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOSLD : AVConv1XI<0b11101, 0b11, 0b1110, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".s32.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VTOULD : AVConv1XI<0b11101, 0b11, 0b1111, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTDI, "vcvt", ".u32.f64\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;
}

// Fixed-Point to FP:

let isCodeGenOnly = 1 in {
def VSHTOS : AVConv1XI<0b11101, 0b11, 0b1010, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.s16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VUHTOS : AVConv1XI<0b11101, 0b11, 0b1011, 0b1010, 0,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.u16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VSLTOS : AVConv1XI<0b11101, 0b11, 0b1010, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.s32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VULTOS : AVConv1XI<0b11101, 0b11, 0b1011, 0b1010, 1,
                       (outs SPR:$dst), (ins SPR:$a, i32imm:$fbits),
                 IIC_fpCVTIS, "vcvt", ".f32.u32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VSHTOD : AVConv1XI<0b11101, 0b11, 0b1010, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.s16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VUHTOD : AVConv1XI<0b11101, 0b11, 0b1011, 0b1011, 0,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.u16\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VSLTOD : AVConv1XI<0b11101, 0b11, 0b1010, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.s32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;

def VULTOD : AVConv1XI<0b11101, 0b11, 0b1011, 0b1011, 1,
                       (outs DPR:$dst), (ins DPR:$a, i32imm:$fbits),
                 IIC_fpCVTID, "vcvt", ".f64.u32\t$dst, $a, $fbits",
                 [/* For disassembly only; pattern left blank */]>;
}

} // End of 'let Constraints = "$src = $dst" in'

//===----------------------------------------------------------------------===//
// FP FMA Operations.
//

def VMLAD : ADbI_vmlX<0b11100, 0b00, 0, 0,
                (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                IIC_fpMAC64, "vmla", ".f64\t$dst, $a, $b",
                [(set DPR:$dst, (fadd (fmul DPR:$a, DPR:$b),
                                      (f64 DPR:$dstin)))]>,
                RegConstraint<"$dstin = $dst">;

def VMLAS : ASbIn<0b11100, 0b00, 0, 0,
                 (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                 IIC_fpMAC32, "vmla", ".f32\t$dst, $a, $b",
                 [(set SPR:$dst, (fadd (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                 RegConstraint<"$dstin = $dst">;

def VNMLSD : ADbI_vmlX<0b11100, 0b01, 0, 0,
                (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                IIC_fpMAC64, "vnmls", ".f64\t$dst, $a, $b",
                [(set DPR:$dst, (fsub (fmul DPR:$a, DPR:$b),
                                (f64 DPR:$dstin)))]>,
                RegConstraint<"$dstin = $dst">;

def VNMLSS : ASbI<0b11100, 0b01, 0, 0,
                (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                IIC_fpMAC32, "vnmls", ".f32\t$dst, $a, $b",
                [(set SPR:$dst, (fsub (fmul SPR:$a, SPR:$b), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def VMLSD : ADbI_vmlX<0b11100, 0b00, 1, 0,
                 (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 IIC_fpMAC64, "vmls", ".f64\t$dst, $a, $b",
             [(set DPR:$dst, (fadd (fneg (fmul DPR:$a, DPR:$b)),
                             (f64 DPR:$dstin)))]>,
                RegConstraint<"$dstin = $dst">;

def VMLSS : ASbIn<0b11100, 0b00, 1, 0,
                  (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                  IIC_fpMAC32, "vmls", ".f32\t$dst, $a, $b",
             [(set SPR:$dst, (fadd (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

def : Pat<(fsub DPR:$dstin, (fmul DPR:$a, (f64 DPR:$b))),
          (VMLSD DPR:$dstin, DPR:$a, DPR:$b)>, Requires<[DontUseNEONForFP]>;
def : Pat<(fsub SPR:$dstin, (fmul SPR:$a, SPR:$b)),
          (VMLSS SPR:$dstin, SPR:$a, SPR:$b)>, Requires<[DontUseNEONForFP]>;

def VNMLAD : ADbI_vmlX<0b11100, 0b01, 1, 0,
                 (outs DPR:$dst), (ins DPR:$dstin, DPR:$a, DPR:$b),
                 IIC_fpMAC64, "vnmla", ".f64\t$dst, $a, $b",
             [(set DPR:$dst, (fsub (fneg (fmul DPR:$a, DPR:$b)),
                             (f64 DPR:$dstin)))]>,
                RegConstraint<"$dstin = $dst">;

def VNMLAS : ASbI<0b11100, 0b01, 1, 0,
                (outs SPR:$dst), (ins SPR:$dstin, SPR:$a, SPR:$b),
                IIC_fpMAC32, "vnmla", ".f32\t$dst, $a, $b",
             [(set SPR:$dst, (fsub (fneg (fmul SPR:$a, SPR:$b)), SPR:$dstin))]>,
                RegConstraint<"$dstin = $dst">;

//===----------------------------------------------------------------------===//
// FP Conditional moves.
//

let neverHasSideEffects = 1 in {
def VMOVDcc  : ADuI<0b11101, 0b11, 0b0000, 0b01, 0,
                    (outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    IIC_fpUNA64, "vmov", ".f64\t$dst, $true",
                [/*(set DPR:$dst, (ARMcmov DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def VMOVScc  : ASuI<0b11101, 0b11, 0b0000, 0b01, 0,
                    (outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    IIC_fpUNA32, "vmov", ".f32\t$dst, $true",
                [/*(set SPR:$dst, (ARMcmov SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def VNEGDcc  : ADuI<0b11101, 0b11, 0b0001, 0b01, 0,
                    (outs DPR:$dst), (ins DPR:$false, DPR:$true),
                    IIC_fpUNA64, "vneg", ".f64\t$dst, $true",
                [/*(set DPR:$dst, (ARMcneg DPR:$false, DPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;

def VNEGScc  : ASuI<0b11101, 0b11, 0b0001, 0b01, 0,
                    (outs SPR:$dst), (ins SPR:$false, SPR:$true),
                    IIC_fpUNA32, "vneg", ".f32\t$dst, $true",
                [/*(set SPR:$dst, (ARMcneg SPR:$false, SPR:$true, imm:$cc))*/]>,
                    RegConstraint<"$false = $dst">;
} // neverHasSideEffects

//===----------------------------------------------------------------------===//
// Misc.
//

// APSR is the application level alias of CPSR. This FPSCR N, Z, C, V flags
// to APSR.
let Defs = [CPSR], Uses = [FPSCR] in
def FMSTAT : VFPAI<(outs), (ins), VFPMiscFrm, IIC_fpSTAT, "vmrs",
                   "\tapsr_nzcv, fpscr",
             [(arm_fmstat)]> {
  let Inst{27-20} = 0b11101111;
  let Inst{19-16} = 0b0001;
  let Inst{15-12} = 0b1111;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{4}     = 1;
}

// FPSCR <-> GPR (for disassembly only)
let hasSideEffects = 1, Uses = [FPSCR] in
def VMRS : VFPAI<(outs GPR:$dst), (ins), VFPMiscFrm, IIC_fpSTAT,
                 "vmrs", "\t$dst, fpscr",
             [(set GPR:$dst, (int_arm_get_fpscr))]> {
  let Inst{27-20} = 0b11101111;
  let Inst{19-16} = 0b0001;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{4}     = 1;
}

let Defs = [FPSCR] in 
def VMSR : VFPAI<(outs), (ins GPR:$src), VFPMiscFrm, IIC_fpSTAT, 
                 "vmsr", "\tfpscr, $src",
             [(int_arm_set_fpscr GPR:$src)]> {
  let Inst{27-20} = 0b11101110;
  let Inst{19-16} = 0b0001;
  let Inst{11-8}  = 0b1010;
  let Inst{7}     = 0;
  let Inst{4}     = 1;
}

// Materialize FP immediates. VFP3 only.
let isReMaterializable = 1 in {
def FCONSTD : VFPAI<(outs DPR:$dst), (ins vfp_f64imm:$imm),
                    VFPMiscFrm, IIC_fpUNA64,
                    "vmov", ".f64\t$dst, $imm",
                    [(set DPR:$dst, vfp_f64imm:$imm)]>, Requires<[HasVFP3]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-20} = 0b11;
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 1;
  let Inst{7-4}   = 0b0000;
}

def FCONSTS : VFPAI<(outs SPR:$dst), (ins vfp_f32imm:$imm),
                    VFPMiscFrm, IIC_fpUNA32,
                    "vmov", ".f32\t$dst, $imm",
                    [(set SPR:$dst, vfp_f32imm:$imm)]>, Requires<[HasVFP3]> {
  let Inst{27-23} = 0b11101;
  let Inst{21-20} = 0b11;
  let Inst{11-9}  = 0b101;
  let Inst{8}     = 0;
  let Inst{7-4}   = 0b0000;
}
}
