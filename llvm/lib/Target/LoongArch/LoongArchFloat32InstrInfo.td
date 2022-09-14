//=-- LoongArchInstrInfoF.td - Single-Precision Float instr --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the baisc single-precision floating-point instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LoongArch specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_LoongArchMOVGR2FR_W_LA64
    : SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisVT<1, i64>]>;
def SDT_LoongArchMOVFR2GR_S_LA64
    : SDTypeProfile<1, 1, [SDTCisVT<0, i64>, SDTCisVT<1, f32>]>;
def SDT_LoongArchFTINT : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisFP<1>]>;

def loongarch_movgr2fr_w_la64
    : SDNode<"LoongArchISD::MOVGR2FR_W_LA64", SDT_LoongArchMOVGR2FR_W_LA64>;
def loongarch_movfr2gr_s_la64
    : SDNode<"LoongArchISD::MOVFR2GR_S_LA64", SDT_LoongArchMOVFR2GR_S_LA64>;
def loongarch_ftint : SDNode<"LoongArchISD::FTINT", SDT_LoongArchFTINT>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBasicF] in {

// Arithmetic Operation Instructions
def FADD_S : FP_ALU_3R<0b00000001000000001, "fadd.s", FPR32>;
def FSUB_S : FP_ALU_3R<0b00000001000000101, "fsub.s", FPR32>;
def FMUL_S : FP_ALU_3R<0b00000001000001001, "fmul.s", FPR32>;
def FDIV_S : FP_ALU_3R<0b00000001000001101, "fdiv.s", FPR32>;
def FMADD_S  : FP_ALU_4R<0b000010000001, "fmadd.s", FPR32>;
def FMSUB_S  : FP_ALU_4R<0b000010000101, "fmsub.s", FPR32>;
def FNMADD_S : FP_ALU_4R<0b000010001001, "fnmadd.s", FPR32>;
def FNMSUB_S : FP_ALU_4R<0b000010001101, "fnmsub.s", FPR32>;
def FMAX_S  : FP_ALU_3R<0b00000001000010001, "fmax.s", FPR32>;
def FMIN_S  : FP_ALU_3R<0b00000001000010101, "fmin.s", FPR32>;
def FMAXA_S : FP_ALU_3R<0b00000001000011001, "fmaxa.s", FPR32>;
def FMINA_S : FP_ALU_3R<0b00000001000011101, "fmina.s", FPR32>;
def FABS_S   : FP_ALU_2R<0b0000000100010100000001, "fabs.s", FPR32>;
def FNEG_S   : FP_ALU_2R<0b0000000100010100000101, "fneg.s", FPR32>;
def FSQRT_S  : FP_ALU_2R<0b0000000100010100010001, "fsqrt.s", FPR32>;
def FRECIP_S : FP_ALU_2R<0b0000000100010100010101, "frecip.s", FPR32>;
def FRSQRT_S : FP_ALU_2R<0b0000000100010100011001, "frsqrt.s", FPR32>;
def FSCALEB_S : FP_ALU_3R<0b00000001000100001, "fscaleb.s", FPR32>;
def FLOGB_S   : FP_ALU_2R<0b0000000100010100001001, "flogb.s", FPR32>;
def FCOPYSIGN_S : FP_ALU_3R<0b00000001000100101, "fcopysign.s", FPR32>;
def FCLASS_S  : FP_ALU_2R<0b0000000100010100001101, "fclass.s", FPR32>;


// Comparison Instructions
def FCMP_CAF_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CAF, "fcmp.caf.s", FPR32>;
def FCMP_CUN_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CUN, "fcmp.cun.s", FPR32>;
def FCMP_CEQ_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CEQ, "fcmp.ceq.s", FPR32>;
def FCMP_CUEQ_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CUEQ, "fcmp.cueq.s", FPR32>;
def FCMP_CLT_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CLT, "fcmp.clt.s", FPR32>;
def FCMP_CULT_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CULT, "fcmp.cult.s", FPR32>;
def FCMP_CLE_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CLE, "fcmp.cle.s", FPR32>;
def FCMP_CULE_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CULE, "fcmp.cule.s", FPR32>;
def FCMP_CNE_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CNE, "fcmp.cne.s", FPR32>;
def FCMP_COR_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_COR, "fcmp.cor.s", FPR32>;
def FCMP_CUNE_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_CUNE, "fcmp.cune.s", FPR32>;
def FCMP_SAF_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SAF, "fcmp.saf.s", FPR32>;
def FCMP_SUN_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SUN, "fcmp.sun.s", FPR32>;
def FCMP_SEQ_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SEQ, "fcmp.seq.s", FPR32>;
def FCMP_SUEQ_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SUEQ, "fcmp.sueq.s", FPR32>;
def FCMP_SLT_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SLT, "fcmp.slt.s", FPR32>;
def FCMP_SULT_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SULT, "fcmp.sult.s", FPR32>;
def FCMP_SLE_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SLE, "fcmp.sle.s", FPR32>;
def FCMP_SULE_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SULE, "fcmp.sule.s", FPR32>;
def FCMP_SNE_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SNE, "fcmp.sne.s", FPR32>;
def FCMP_SOR_S  : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SOR, "fcmp.sor.s", FPR32>;
def FCMP_SUNE_S : FP_CMP<FPCMP_OPC_S, FPCMP_COND_SUNE, "fcmp.sune.s", FPR32>;

// Conversion Instructions
def FFINT_S_W    : FP_CONV<0b0000000100011101000100, "ffint.s.w", FPR32, FPR32>;
def FTINT_W_S    : FP_CONV<0b0000000100011011000001, "ftint.w.s", FPR32, FPR32>;
def FTINTRM_W_S  : FP_CONV<0b0000000100011010000001, "ftintrm.w.s", FPR32,
                           FPR32>;
def FTINTRP_W_S  : FP_CONV<0b0000000100011010010001, "ftintrp.w.s", FPR32,
                           FPR32>;
def FTINTRZ_W_S  : FP_CONV<0b0000000100011010100001, "ftintrz.w.s", FPR32,
                           FPR32>;
def FTINTRNE_W_S : FP_CONV<0b0000000100011010110001, "ftintrne.w.s", FPR32,
                           FPR32>;
def FRINT_S      : FP_CONV<0b0000000100011110010001, "frint.s", FPR32, FPR32>;

// Move Instructions
def FSEL_S : FP_SEL<0b00001101000000, "fsel", FPR32>;
def FMOV_S     : FP_MOV<0b0000000100010100100101, "fmov.s", FPR32, FPR32>;
def MOVGR2FR_W : FP_MOV<0b0000000100010100101001, "movgr2fr.w", FPR32, GPR>;
def MOVFR2GR_S : FP_MOV<0b0000000100010100101101, "movfr2gr.s", GPR, FPR32>;
def MOVGR2FCSR : FP_MOV<0b0000000100010100110000, "movgr2fcsr", FCSR, GPR>;
def MOVFCSR2GR : FP_MOV<0b0000000100010100110010, "movfcsr2gr", GPR, FCSR>;
def MOVFR2CF_S : FP_MOV<0b0000000100010100110100, "movfr2cf", CFR, FPR32>;
def MOVCF2FR_S : FP_MOV<0b0000000100010100110101, "movcf2fr", FPR32, CFR>;
def MOVGR2CF   : FP_MOV<0b0000000100010100110110, "movgr2cf", CFR, GPR>;
def MOVCF2GR   : FP_MOV<0b0000000100010100110111, "movcf2gr", GPR, CFR>;

// Branch Instructions
def BCEQZ : FP_BRANCH<0b01001000, "bceqz">;
def BCNEZ : FP_BRANCH<0b01001001, "bcnez">;

// Common Memory Access Instructions
def FLD_S : FP_LOAD_2RI12<0b0010101100, "fld.s", FPR32>;
def FST_S : FP_STORE_2RI12<0b0010101101, "fst.s", FPR32>;
def FLDX_S : FP_LOAD_3R<0b00111000001100000, "fldx.s", FPR32>;
def FSTX_S : FP_STORE_3R<0b00111000001110000, "fstx.s", FPR32>;

// Bound Check Memory Access Instructions
def FLDGT_S : FP_LOAD_3R<0b00111000011101000, "fldgt.s", FPR32>;
def FLDLE_S : FP_LOAD_3R<0b00111000011101010, "fldle.s", FPR32>;
def FSTGT_S : FP_STORE_3R<0b00111000011101100, "fstgt.s", FPR32>;
def FSTLE_S : FP_STORE_3R<0b00111000011101110, "fstle.s", FPR32>;

} // Predicates = [HasBasicF]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

/// Generic pattern classes

class PatFpr<SDPatternOperator OpNode, LAInst Inst, RegisterClass RegTy>
    : Pat<(OpNode RegTy:$fj), (Inst $fj)>;
class PatFprFpr<SDPatternOperator OpNode, LAInst Inst, RegisterClass RegTy>
    : Pat<(OpNode RegTy:$fj, RegTy:$fk), (Inst $fj, $fk)>;

let Predicates = [HasBasicF] in {

/// Float arithmetic operations

def : PatFprFpr<fadd, FADD_S, FPR32>;
def : PatFprFpr<fsub, FSUB_S, FPR32>;
def : PatFprFpr<fmul, FMUL_S, FPR32>;
def : PatFprFpr<fdiv, FDIV_S, FPR32>;
def : PatFprFpr<fcopysign, FCOPYSIGN_S, FPR32>;
def : PatFpr<fneg, FNEG_S, FPR32>;
def : PatFpr<fabs, FABS_S, FPR32>;
def : PatFpr<fsqrt, FSQRT_S, FPR32>;

def : Pat<(fdiv fpimm1, (fsqrt FPR32:$fj)), (FRSQRT_S FPR32:$fj)>;

/// Setcc

// Match non-signaling comparison

// TODO: change setcc to any_fsetcc after call is supported because
// we need to call llvm.experimental.constrained.fcmp.f32 in testcase.
// See RISCV float-fcmp-strict.ll for reference.
class PatFPSetcc<CondCode cc, LAInst CmpInst, RegisterClass RegTy>
    : Pat<(setcc RegTy:$fj, RegTy:$fk, cc),
          (MOVCF2GR (CmpInst RegTy:$fj, RegTy:$fk))>;
// SETOGT/SETOGE/SETUGT/SETUGE will expand into SETOLT/SETOLE/SETULT/SETULE.
def : PatFPSetcc<SETOEQ, FCMP_CEQ_S,  FPR32>;
def : PatFPSetcc<SETOLT, FCMP_CLT_S,  FPR32>;
def : PatFPSetcc<SETOLE, FCMP_CLE_S,  FPR32>;
def : PatFPSetcc<SETONE, FCMP_CNE_S,  FPR32>;
def : PatFPSetcc<SETO,   FCMP_COR_S,  FPR32>;
def : PatFPSetcc<SETUEQ, FCMP_CUEQ_S, FPR32>;
def : PatFPSetcc<SETULT, FCMP_CULT_S, FPR32>;
def : PatFPSetcc<SETULE, FCMP_CULE_S, FPR32>;
def : PatFPSetcc<SETUNE, FCMP_CUNE_S, FPR32>;
def : PatFPSetcc<SETUO,  FCMP_CUN_S,  FPR32>;
def : PatFPSetcc<SETLT,  FCMP_CLT_S,  FPR32>;

multiclass PatFPBrcond<CondCode cc, LAInst CmpInst, RegisterClass RegTy> {
  def : Pat<(brcond (xor (GRLenVT (setcc RegTy:$fj, RegTy:$fk, cc)), -1),
                     bb:$imm21),
            (BCEQZ (CmpInst RegTy:$fj, RegTy:$fk), bb:$imm21)>;
  def : Pat<(brcond (GRLenVT (setcc RegTy:$fj, RegTy:$fk, cc)), bb:$imm21),
            (BCNEZ (CmpInst RegTy:$fj, RegTy:$fk), bb:$imm21)>;
}

defm : PatFPBrcond<SETOEQ, FCMP_CEQ_S, FPR32>;
defm : PatFPBrcond<SETOLT, FCMP_CLT_S, FPR32>;
defm : PatFPBrcond<SETOLE, FCMP_CLE_S, FPR32>;
defm : PatFPBrcond<SETONE, FCMP_CNE_S, FPR32>;
defm : PatFPBrcond<SETO,   FCMP_COR_S, FPR32>;
defm : PatFPBrcond<SETUEQ, FCMP_CUEQ_S, FPR32>;
defm : PatFPBrcond<SETULT, FCMP_CULT_S, FPR32>;
defm : PatFPBrcond<SETULE, FCMP_CULE_S, FPR32>;
defm : PatFPBrcond<SETUNE, FCMP_CUNE_S, FPR32>;
defm : PatFPBrcond<SETUO,  FCMP_CUN_S, FPR32>;
defm : PatFPBrcond<SETLT,  FCMP_CLT_S, FPR32>;

// TODO: Match signaling comparison strict_fsetccs with FCMP_S*_S instructions.

/// Select

def : Pat<(select GPR:$cc, FPR32:$fk, FPR32:$fj),
          (FSEL_S FPR32:$fj, FPR32:$fk, (MOVGR2CF GPR:$cc))>;

/// Selectcc

class PatFPSelectcc<CondCode cc, LAInst CmpInst, LAInst SelInst,
                    RegisterClass RegTy>
    : Pat<(select (GRLenVT (setcc RegTy:$a, RegTy:$b, cc)), RegTy:$t, RegTy:$f),
          (SelInst RegTy:$f, RegTy:$t, (CmpInst RegTy:$a, RegTy:$b))>;
def : PatFPSelectcc<SETOEQ, FCMP_CEQ_S,  FSEL_S, FPR32>;
def : PatFPSelectcc<SETOLT, FCMP_CLT_S,  FSEL_S, FPR32>;
def : PatFPSelectcc<SETOLE, FCMP_CLE_S,  FSEL_S, FPR32>;
def : PatFPSelectcc<SETONE, FCMP_CNE_S,  FSEL_S, FPR32>;
def : PatFPSelectcc<SETO,   FCMP_COR_S,  FSEL_S, FPR32>;
def : PatFPSelectcc<SETUEQ, FCMP_CUEQ_S, FSEL_S, FPR32>;
def : PatFPSelectcc<SETULT, FCMP_CULT_S, FSEL_S, FPR32>;
def : PatFPSelectcc<SETULE, FCMP_CULE_S, FSEL_S, FPR32>;
def : PatFPSelectcc<SETUNE, FCMP_CUNE_S, FSEL_S, FPR32>;
def : PatFPSelectcc<SETUO,  FCMP_CUN_S,  FSEL_S, FPR32>;

/// Loads

defm : LdPat<load, FLD_S, f32>;
def : RegRegLdPat<load, FLDX_S, f32>;

/// Stores

defm : StPat<store, FST_S, FPR32, f32>;
def : RegRegStPat<store, FSTX_S, FPR32, f32>;

/// Floating point constants

def : Pat<(f32 fpimm0), (MOVGR2FR_W R0)>;
def : Pat<(f32 fpimm0neg), (FNEG_S (MOVGR2FR_W R0))>;
def : Pat<(f32 fpimm1), (FFINT_S_W (MOVGR2FR_W (ADDI_W R0, 1)))>;

// FP Conversion
def : Pat<(loongarch_ftint FPR32:$src), (FTINTRZ_W_S FPR32:$src)>;

// FP reciprocal operation
def : Pat<(fdiv fpimm1, FPR32:$src), (FRECIP_S $src)>;

// fmadd.s: fj * fk + fa
def : Pat<(fma FPR32:$fj, FPR32:$fk, FPR32:$fa), (FMADD_S $fj, $fk, $fa)>;

// fmsub.s: fj * fk - fa
def : Pat<(fma FPR32:$fj, FPR32:$fk, (fneg FPR32:$fa)),
          (FMSUB_S FPR32:$fj, FPR32:$fk, FPR32:$fa)>;

// fnmadd.s: -(fj * fk + fa)
def : Pat<(fneg (fma FPR32:$fj, FPR32:$fk, FPR32:$fa)),
          (FNMADD_S FPR32:$fj, FPR32:$fk, FPR32:$fa)>;

// fnmadd.s: -fj * fk - fa (the nsz flag on the FMA)
def : Pat<(fma_nsz (fneg FPR32:$fj), FPR32:$fk, (fneg FPR32:$fa)),
          (FNMADD_S FPR32:$fj, FPR32:$fk, FPR32:$fa)>;

// fnmsub.s: -fj * fk + fa
def : Pat<(fma (fneg FPR32:$fj), FPR32:$fk, FPR32:$fa),
          (FNMSUB_S FPR32:$fj, FPR32:$fk, FPR32:$fa)>;
} // Predicates = [HasBasicF]

let Predicates = [HasBasicF, IsLA64] in {
// GPR -> FPR
def : Pat<(loongarch_movgr2fr_w_la64 GPR:$src), (MOVGR2FR_W GPR:$src)>;
// FPR -> GPR
def : Pat<(loongarch_movfr2gr_s_la64 FPR32:$src),
          (MOVFR2GR_S FPR32:$src)>;
// int -> f32
def : Pat<(f32 (sint_to_fp GPR:$src)), (FFINT_S_W (MOVGR2FR_W GPR:$src))>;
} // Predicates = [HasBasicF, IsLA64]

// FP Rounding
let Predicates = [HasBasicF, IsLA64] in {
def : PatFpr<frint, FRINT_S, FPR32>;
} // Predicates = [HasBasicF, IsLA64]

let Predicates = [HasBasicF, IsLA32] in {
// GPR -> FPR
def : Pat<(bitconvert (i32 GPR:$src)), (MOVGR2FR_W GPR:$src)>;
// FPR -> GPR
def : Pat<(i32 (bitconvert FPR32:$src)), (MOVFR2GR_S FPR32:$src)>;
// int -> f32
def : Pat<(f32 (sint_to_fp (i32 GPR:$src))), (FFINT_S_W (MOVGR2FR_W GPR:$src))>;
} // Predicates = [HasBasicF, IsLA32]
