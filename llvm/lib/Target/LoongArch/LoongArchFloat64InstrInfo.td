//=-- LoongArchInstrInfoD.td - Double-Precision Float instr -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the basic double-precision floating-point instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBasicD] in {

// Arithmetic Operation Instructions
def FADD_D : FP_ALU_3R<0b00000001000000010, "fadd.d", FPR64>;
def FSUB_D : FP_ALU_3R<0b00000001000000110, "fsub.d", FPR64>;
def FMUL_D : FP_ALU_3R<0b00000001000001010, "fmul.d", FPR64>;
def FDIV_D : FP_ALU_3R<0b00000001000001110, "fdiv.d", FPR64>;
def FMADD_D  : FP_ALU_4R<0b000010000010, "fmadd.d", FPR64>;
def FMSUB_D  : FP_ALU_4R<0b000010000110, "fmsub.d", FPR64>;
def FNMADD_D : FP_ALU_4R<0b000010001010, "fnmadd.d", FPR64>;
def FNMSUB_D : FP_ALU_4R<0b000010001110, "fnmsub.d", FPR64>;
def FMAX_D  : FP_ALU_3R<0b00000001000010010, "fmax.d", FPR64>;
def FMIN_D  : FP_ALU_3R<0b00000001000010110, "fmin.d", FPR64>;
def FMAXA_D : FP_ALU_3R<0b00000001000011010, "fmaxa.d", FPR64>;
def FMINA_D : FP_ALU_3R<0b00000001000011110, "fmina.d", FPR64>;
def FABS_D   : FP_ALU_2R<0b0000000100010100000010, "fabs.d", FPR64>;
def FNEG_D   : FP_ALU_2R<0b0000000100010100000110, "fneg.d", FPR64>;
def FSQRT_D  : FP_ALU_2R<0b0000000100010100010010, "fsqrt.d", FPR64>;
def FRECIP_D : FP_ALU_2R<0b0000000100010100010110, "frecip.d", FPR64>;
def FRSQRT_D : FP_ALU_2R<0b0000000100010100011010, "frsqrt.d", FPR64>;
def FSCALEB_D : FP_ALU_3R<0b00000001000100010, "fscaleb.d", FPR64>;
def FLOGB_D   : FP_ALU_2R<0b0000000100010100001010, "flogb.d", FPR64>;
def FCOPYSIGN_D : FP_ALU_3R<0b00000001000100110, "fcopysign.d", FPR64>;
def FCLASS_D  : FP_ALU_2R<0b0000000100010100001110, "fclass.d", FPR64>;

// Comparison Instructions
def FCMP_CAF_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CAF, "fcmp.caf.d", FPR64>;
def FCMP_CUN_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CUN, "fcmp.cun.d", FPR64>;
def FCMP_CEQ_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CEQ, "fcmp.ceq.d", FPR64>;
def FCMP_CUEQ_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CUEQ, "fcmp.cueq.d", FPR64>;
def FCMP_CLT_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CLT, "fcmp.clt.d", FPR64>;
def FCMP_CULT_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CULT, "fcmp.cult.d", FPR64>;
def FCMP_CLE_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CLE, "fcmp.cle.d", FPR64>;
def FCMP_CULE_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CULE, "fcmp.cule.d", FPR64>;
def FCMP_CNE_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CNE, "fcmp.cne.d", FPR64>;
def FCMP_COR_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_COR, "fcmp.cor.d", FPR64>;
def FCMP_CUNE_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_CUNE, "fcmp.cune.d", FPR64>;
def FCMP_SAF_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SAF, "fcmp.saf.d", FPR64>;
def FCMP_SUN_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SUN, "fcmp.sun.d", FPR64>;
def FCMP_SEQ_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SEQ, "fcmp.seq.d", FPR64>;
def FCMP_SUEQ_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SUEQ, "fcmp.sueq.d", FPR64>;
def FCMP_SLT_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SLT, "fcmp.slt.d", FPR64>;
def FCMP_SULT_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SULT, "fcmp.sult.d", FPR64>;
def FCMP_SLE_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SLE, "fcmp.sle.d", FPR64>;
def FCMP_SULE_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SULE, "fcmp.sule.d", FPR64>;
def FCMP_SNE_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SNE, "fcmp.sne.d", FPR64>;
def FCMP_SOR_D  : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SOR, "fcmp.sor.d", FPR64>;
def FCMP_SUNE_D : FP_CMP<FPCMP_OPC_D, FPCMP_COND_SUNE, "fcmp.sune.d", FPR64>;

// Conversion Instructions
def FFINT_S_L : FP_CONV<0b0000000100011101000110, "ffint.s.l", FPR32, FPR64>;
def FTINT_L_S : FP_CONV<0b0000000100011011001001, "ftint.l.s", FPR64, FPR32>;
def FTINTRM_L_S : FP_CONV<0b0000000100011010001001, "ftintrm.l.s", FPR64,
                          FPR32>;
def FTINTRP_L_S : FP_CONV<0b0000000100011010011001, "ftintrp.l.s", FPR64,
                          FPR32>;
def FTINTRZ_L_S : FP_CONV<0b0000000100011010101001, "ftintrz.l.s", FPR64,
                          FPR32>;
def FTINTRNE_L_S : FP_CONV<0b0000000100011010111001, "ftintrne.l.s", FPR64,
                           FPR32>;
def FCVT_S_D : FP_CONV<0b0000000100011001000110, "fcvt.s.d", FPR32, FPR64>;
def FCVT_D_S : FP_CONV<0b0000000100011001001001, "fcvt.d.s", FPR64, FPR32>;
def FFINT_D_W : FP_CONV<0b0000000100011101001000, "ffint.d.w", FPR64, FPR32>;
def FFINT_D_L : FP_CONV<0b0000000100011101001010, "ffint.d.l", FPR64, FPR64>;
def FTINT_W_D : FP_CONV<0b0000000100011011000010, "ftint.w.d", FPR32, FPR64>;
def FTINT_L_D : FP_CONV<0b0000000100011011001010, "ftint.l.d", FPR64, FPR64>;
def FTINTRM_W_D : FP_CONV<0b0000000100011010000010, "ftintrm.w.d", FPR32,
                          FPR64>;
def FTINTRM_L_D : FP_CONV<0b0000000100011010001010, "ftintrm.l.d", FPR64,
                          FPR64>;
def FTINTRP_W_D : FP_CONV<0b0000000100011010010010, "ftintrp.w.d", FPR32,
                          FPR64>;
def FTINTRP_L_D : FP_CONV<0b0000000100011010011010, "ftintrp.l.d", FPR64,
                          FPR64>;
def FTINTRZ_W_D : FP_CONV<0b0000000100011010100010, "ftintrz.w.d", FPR32,
                          FPR64>;
def FTINTRZ_L_D : FP_CONV<0b0000000100011010101010, "ftintrz.l.d", FPR64,
                          FPR64>;
def FTINTRNE_W_D : FP_CONV<0b0000000100011010110010, "ftintrne.w.d", FPR32,
                           FPR64>;
def FTINTRNE_L_D : FP_CONV<0b0000000100011010111010, "ftintrne.l.d", FPR64,
                           FPR64>;
def FRINT_D : FP_CONV<0b0000000100011110010010, "frint.d", FPR64, FPR64>;

// Move Instructions
def FMOV_D        : FP_MOV<0b0000000100010100100110, "fmov.d", FPR64, FPR64>;
def MOVFRH2GR_S   : FP_MOV<0b0000000100010100101111, "movfrh2gr.s", GPR, FPR64>;
let isCodeGenOnly = 1 in {
def MOVFR2GR_S_64 : FP_MOV<0b0000000100010100101101, "movfr2gr.s", GPR, FPR64>;
def FSEL_D : FP_SEL<0b00001101000000, "fsel", FPR64>;
} // isCodeGenOnly = 1
let Constraints = "$dst = $out" in {
def MOVGR2FRH_W : FPFmtMOV<0b0000000100010100101011, (outs FPR64:$out),
                           (ins FPR64:$dst, GPR:$src), "movgr2frh.w",
                           "$dst, $src">;
} // Constraints = "$dst = $out"

// Common Memory Access Instructions
def FLD_D : FP_LOAD_2RI12<0b0010101110, "fld.d", FPR64>;
def FST_D : FP_STORE_2RI12<0b0010101111, "fst.d", FPR64>;
def FLDX_D : FP_LOAD_3R<0b00111000001101000, "fldx.d", FPR64>;
def FSTX_D : FP_STORE_3R<0b00111000001111000, "fstx.d", FPR64>;

// Bound Check Memory Access Instructions
def FLDGT_D : FP_LOAD_3R<0b00111000011101001, "fldgt.d", FPR64>;
def FLDLE_D : FP_LOAD_3R<0b00111000011101011, "fldle.d", FPR64>;
def FSTGT_D : FP_STORE_3R<0b00111000011101101, "fstgt.d", FPR64>;
def FSTLE_D : FP_STORE_3R<0b00111000011101111, "fstle.d", FPR64>;

} // Predicates = [HasBasicD]

// Instructions only available on LA64
let Predicates = [HasBasicD, IsLA64] in {
def MOVGR2FR_D  : FP_MOV<0b0000000100010100101010, "movgr2fr.d", FPR64, GPR>;
def MOVFR2GR_D  : FP_MOV<0b0000000100010100101110, "movfr2gr.d", GPR, FPR64>;
} // Predicates = [HasBasicD, IsLA64]

// Instructions only available on LA32
let Predicates = [HasBasicD, IsLA32], isCodeGenOnly = 1 in {
def MOVGR2FR_W_64 : FP_MOV<0b0000000100010100101001, "movgr2fr.w", FPR64, GPR>;
} // Predicates = [HasBasicD, IsLA32], isCodeGenOnly = 1

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasBasicD] in {

/// Float arithmetic operations

def : PatFprFpr<fadd, FADD_D, FPR64>;
def : PatFprFpr<fsub, FSUB_D, FPR64>;
def : PatFprFpr<fmul, FMUL_D, FPR64>;
def : PatFprFpr<fdiv, FDIV_D, FPR64>;
def : PatFprFpr<fcopysign, FCOPYSIGN_D, FPR64>;
def : PatFprFpr<fmaxnum_ieee, FMAX_D, FPR64>;
def : PatFprFpr<fminnum_ieee, FMIN_D, FPR64>;
def : PatFpr<fneg, FNEG_D, FPR64>;
def : PatFpr<fabs, FABS_D, FPR64>;
def : PatFpr<fsqrt, FSQRT_D, FPR64>;

def : Pat<(fdiv fpimm1, (fsqrt FPR64:$fj)), (FRSQRT_D FPR64:$fj)>;

def : Pat<(fcopysign FPR64:$fj, FPR32:$fk),
          (FCOPYSIGN_D FPR64:$fj, (FCVT_D_S FPR32:$fk))>;
def : Pat<(fcopysign FPR32:$fj, FPR64:$fk),
          (FCOPYSIGN_S FPR32:$fj, (FCVT_S_D FPR64:$fk))>;

def : Pat<(fcanonicalize FPR64:$fj), (FMAX_D $fj, $fj)>;

/// Setcc

// Match non-signaling comparison

// SETOGT/SETOGE/SETUGT/SETUGE/SETGE/SETNE/SETGT will expand into
// SETOLT/SETOLE/SETULT/SETULE/SETLE/SETEQ/SETLT.
def : PatFPSetcc<SETOEQ, FCMP_CEQ_D,  FPR64>;
def : PatFPSetcc<SETEQ,  FCMP_CEQ_D,  FPR64>;
def : PatFPSetcc<SETOLT, FCMP_CLT_D,  FPR64>;
def : PatFPSetcc<SETOLE, FCMP_CLE_D,  FPR64>;
def : PatFPSetcc<SETLE,  FCMP_CLE_D,  FPR64>;
def : PatFPSetcc<SETONE, FCMP_CNE_D,  FPR64>;
def : PatFPSetcc<SETO,   FCMP_COR_D,  FPR64>;
def : PatFPSetcc<SETUEQ, FCMP_CUEQ_D, FPR64>;
def : PatFPSetcc<SETULT, FCMP_CULT_D, FPR64>;
def : PatFPSetcc<SETULE, FCMP_CULE_D, FPR64>;
def : PatFPSetcc<SETUNE, FCMP_CUNE_D, FPR64>;
def : PatFPSetcc<SETUO,  FCMP_CUN_D,  FPR64>;
def : PatFPSetcc<SETLT,  FCMP_CLT_D,  FPR64>;

defm : PatFPBrcond<SETOEQ, FCMP_CEQ_D, FPR64>;
defm : PatFPBrcond<SETOLT, FCMP_CLT_D, FPR64>;
defm : PatFPBrcond<SETOLE, FCMP_CLE_D, FPR64>;
defm : PatFPBrcond<SETONE, FCMP_CNE_D, FPR64>;
defm : PatFPBrcond<SETO,   FCMP_COR_D, FPR64>;
defm : PatFPBrcond<SETUEQ, FCMP_CUEQ_D, FPR64>;
defm : PatFPBrcond<SETULT, FCMP_CULT_D, FPR64>;
defm : PatFPBrcond<SETULE, FCMP_CULE_D, FPR64>;
defm : PatFPBrcond<SETUNE, FCMP_CUNE_D, FPR64>;
defm : PatFPBrcond<SETUO,  FCMP_CUN_D, FPR64>;
defm : PatFPBrcond<SETLT,  FCMP_CLT_D, FPR64>;

// Match signaling comparison

def : PatStrictFsetccs<SETOEQ, FCMP_SEQ_D,  FPR64>;
def : PatStrictFsetccs<SETOLT, FCMP_SLT_D,  FPR64>;
def : PatStrictFsetccs<SETOLE, FCMP_SLE_D,  FPR64>;
def : PatStrictFsetccs<SETONE, FCMP_SNE_D,  FPR64>;
def : PatStrictFsetccs<SETO,   FCMP_SOR_D,  FPR64>;
def : PatStrictFsetccs<SETUEQ, FCMP_SUEQ_D, FPR64>;
def : PatStrictFsetccs<SETULT, FCMP_SULT_D, FPR64>;
def : PatStrictFsetccs<SETULE, FCMP_SULE_D, FPR64>;
def : PatStrictFsetccs<SETUNE, FCMP_SUNE_D, FPR64>;
def : PatStrictFsetccs<SETUO,  FCMP_SUN_D,  FPR64>;
def : PatStrictFsetccs<SETLT,  FCMP_SLT_D,  FPR64>;

/// Select

def : Pat<(select CFR:$cc, FPR64:$fk, FPR64:$fj),
          (FSEL_D FPR64:$fj, FPR64:$fk, CFR:$cc)>;

/// Selectcc

def : PatFPSelectcc<SETOEQ, FCMP_CEQ_D,  FSEL_D, FPR64>;
def : PatFPSelectcc<SETOLT, FCMP_CLT_D,  FSEL_D, FPR64>;
def : PatFPSelectcc<SETOLE, FCMP_CLE_D,  FSEL_D, FPR64>;
def : PatFPSelectcc<SETONE, FCMP_CNE_D,  FSEL_D, FPR64>;
def : PatFPSelectcc<SETO,   FCMP_COR_D,  FSEL_D, FPR64>;
def : PatFPSelectcc<SETUEQ, FCMP_CUEQ_D, FSEL_D, FPR64>;
def : PatFPSelectcc<SETULT, FCMP_CULT_D, FSEL_D, FPR64>;
def : PatFPSelectcc<SETULE, FCMP_CULE_D, FSEL_D, FPR64>;
def : PatFPSelectcc<SETUNE, FCMP_CUNE_D, FSEL_D, FPR64>;
def : PatFPSelectcc<SETUO,  FCMP_CUN_D,  FSEL_D, FPR64>;

/// Loads

defm : LdPat<load, FLD_D, f64>;
def : RegRegLdPat<load, FLDX_D, f64>;

/// Stores

defm : StPat<store, FST_D, FPR64, f64>;
def : RegRegStPat<store, FSTX_D, FPR64, f64>;

/// FP conversion operations

def : Pat<(loongarch_ftint FPR64:$src), (FTINTRZ_W_D FPR64:$src)>;
def : Pat<(f64 (loongarch_ftint FPR64:$src)), (FTINTRZ_L_D FPR64:$src)>;
def : Pat<(loongarch_ftint FPR32:$src), (FTINTRZ_L_S FPR32:$src)>;

// f64 -> f32
def : Pat<(f32 (fpround FPR64:$src)), (FCVT_S_D FPR64:$src)>;
// f32 -> f64
def : Pat<(f64 (fpextend FPR32:$src)), (FCVT_D_S FPR32:$src)>;

// FP reciprocal operation
def : Pat<(fdiv fpimm1, FPR64:$src), (FRECIP_D $src)>;

// fmadd.d: fj * fk + fa
def : Pat<(fma FPR64:$fj, FPR64:$fk, FPR64:$fa), (FMADD_D $fj, $fk, $fa)>;

// fmsub.d: fj * fk - fa
def : Pat<(fma FPR64:$fj, FPR64:$fk, (fneg FPR64:$fa)),
          (FMSUB_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;

// fnmadd.d: -(fj * fk + fa)
def : Pat<(fneg (fma FPR64:$fj, FPR64:$fk, FPR64:$fa)),
          (FNMADD_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;

// fnmadd.d: -fj * fk - fa (the nsz flag on the FMA)
def : Pat<(fma_nsz (fneg FPR64:$fj), FPR64:$fk, (fneg FPR64:$fa)),
          (FNMADD_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;

// fnmsub.d: -(fj * fk - fa)
def : Pat<(fma (fneg FPR64:$fj), FPR64:$fk, FPR64:$fa),
          (FNMSUB_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;
} // Predicates = [HasBasicD]

/// Floating point constants

let Predicates = [HasBasicD, IsLA64] in {
def : Pat<(f64 fpimm0), (MOVGR2FR_D R0)>;
def : Pat<(f64 fpimm0neg), (FNEG_D (MOVGR2FR_D R0))>;
def : Pat<(f64 fpimm1), (FFINT_D_L (MOVGR2FR_D (ADDI_D R0, 1)))>;
} // Predicates = [HasBasicD, IsLA64]
let Predicates = [HasBasicD, IsLA32] in {
def : Pat<(f64 fpimm0), (MOVGR2FRH_W (MOVGR2FR_W_64 R0), R0)>;
def : Pat<(f64 fpimm0neg), (FNEG_D (MOVGR2FRH_W (MOVGR2FR_W_64 R0), R0))>;
def : Pat<(f64 fpimm1), (FCVT_D_S (FFINT_S_W (MOVGR2FR_W (ADDI_W R0, 1))))>;
} // Predicates = [HasBasicD, IsLA32]

/// Convert int to FP

let Predicates = [HasBasicD, IsLA64] in {
def : Pat<(f32 (sint_to_fp GPR:$src)), (FFINT_S_L (MOVGR2FR_D GPR:$src))>;
def : Pat<(f64 (sint_to_fp (i64 (sexti32 (i64 GPR:$src))))),
          (FFINT_D_W (MOVGR2FR_W GPR:$src))>;
def : Pat<(f64 (sint_to_fp GPR:$src)), (FFINT_D_L (MOVGR2FR_D GPR:$src))>;

def : Pat<(bitconvert GPR:$src), (MOVGR2FR_D GPR:$src)>;
} // Predicates = [HasBasicD, IsLA64]
let Predicates = [HasBasicD, IsLA32] in {
def : Pat<(f64 (sint_to_fp (i32 GPR:$src))), (FFINT_D_W (MOVGR2FR_W GPR:$src))>;
} // Predicates = [HasBasicD, IsLA32]

// Convert FP to int
let Predicates = [HasBasicD, IsLA64] in {
def : Pat<(bitconvert FPR64:$src), (MOVFR2GR_D FPR64:$src)>;
} // Predicates = [HasBasicD, IsLA64]

// FP Rounding
let Predicates = [HasBasicD, IsLA64] in {
def : PatFpr<frint, FRINT_D, FPR64>;
} // Predicates = [HasBasicD, IsLA64]
