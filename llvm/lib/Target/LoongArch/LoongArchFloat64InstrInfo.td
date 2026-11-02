// LoongArchFloat64InstrInfo.td - Double-Precision Float instr --*- tablegen -*-
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
// LoongArch specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_LoongArchMOVGR2FR_D
    : SDTypeProfile<1, 1, [SDTCisVT<0, f64>, SDTCisVT<1, i64>]>;
def SDT_LoongArchMOVGR2FR_D_LO_HI
    : SDTypeProfile<1, 2, [SDTCisVT<0, f64>, SDTCisVT<1, i32>,
                           SDTCisSameAs<1, 2>]>;

// FPR<->GPR transfer operations
def loongarch_movgr2fr_d
    : SDNode<"LoongArchISD::MOVGR2FR_D", SDT_LoongArchMOVGR2FR_D>;
def loongarch_movgr2fr_d_lo_hi
    : SDNode<"LoongArchISD::MOVGR2FR_D_LO_HI", SDT_LoongArchMOVGR2FR_D_LO_HI>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasBasicD] in {

// Arithmetic Operation Instructions
def FADD_D : FP_ALU_3R<0x01010000, FPR64>,
             Sched<[WriteFADD, ReadFADD, ReadFADD]>;
def FSUB_D : FP_ALU_3R<0x01030000, FPR64>,
             Sched<[WriteFADD, ReadFADD, ReadFADD]>;
def FMUL_D : FP_ALU_3R<0x01050000, FPR64>,
             Sched<[WriteFMUL, ReadFMUL, ReadFMUL]>;
def FDIV_D : FP_ALU_3R<0x01070000, FPR64>,
             Sched<[WriteFDIV, ReadFDIV, ReadFDIV]>;
def FMADD_D  : FP_ALU_4R<0x08200000, FPR64>,
               Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>;
def FMSUB_D  : FP_ALU_4R<0x08600000, FPR64>,
               Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>;
def FNMADD_D : FP_ALU_4R<0x08a00000, FPR64>,
               Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>;
def FNMSUB_D : FP_ALU_4R<0x08e00000, FPR64>,
               Sched<[WriteFMADD, ReadFMADD, ReadFMADD, ReadFMADD]>;
def FMAX_D  : FP_ALU_3R<0x01090000, FPR64>,
              Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>;
def FMIN_D  : FP_ALU_3R<0x010b0000, FPR64>,
              Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>;
def FMAXA_D : FP_ALU_3R<0x010d0000, FPR64>,
              Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>;
def FMINA_D : FP_ALU_3R<0x010f0000, FPR64>,
              Sched<[WriteFMinMax, ReadFMinMax, ReadFMinMax]>;
def FABS_D   : FP_ALU_2R<0x01140800, FPR64>,
               Sched<[WriteFABS, ReadFABS]>;
def FNEG_D   : FP_ALU_2R<0x01141800, FPR64>,
               Sched<[WriteFNEG, ReadFNEG]>;
def FSQRT_D  : FP_ALU_2R<0x01144800, FPR64>,
               Sched<[WriteFSQRT, ReadFSQRT]>;
def FRECIP_D : FP_ALU_2R<0x01145800, FPR64>,
               Sched<[WriteFRECIP, ReadFRECIP]>;
def FRSQRT_D : FP_ALU_2R<0x01146800, FPR64>,
               Sched<[WriteFRSQRT, ReadFRSQRT]>;
def FRECIPE_D : FP_ALU_2R<0x01147800, FPR64>,
                Sched<[WriteFRECIPE, ReadFRECIPE]>;
def FRSQRTE_D : FP_ALU_2R<0x01148800, FPR64>,
                Sched<[WriteFRSQRTE, ReadFRSQRTE]>;
def FSCALEB_D : FP_ALU_3R<0x01110000, FPR64>,
                Sched<[WriteFSCALEB, ReadFSCALEB, ReadFSCALEB]>;
def FLOGB_D   : FP_ALU_2R<0x01142800, FPR64>,
                Sched<[WriteFLOGB, ReadFLOGB]>;
def FCOPYSIGN_D : FP_ALU_3R<0x01130000, FPR64>,
                  Sched<[WriteFCOPYSIGN, ReadFCOPYSIGN, ReadFCOPYSIGN]>;
def FCLASS_D  : FP_ALU_2R<0x01143800, FPR64>,
                Sched<[WriteFCLASS, ReadFCLASS]>;

// Comparison Instructions
def FCMP_CAF_D  : FP_CMP<0x0c200000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CUN_D  : FP_CMP<0x0c240000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CEQ_D  : FP_CMP<0x0c220000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CUEQ_D : FP_CMP<0x0c260000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CLT_D  : FP_CMP<0x0c210000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CULT_D : FP_CMP<0x0c250000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CLE_D  : FP_CMP<0x0c230000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CULE_D : FP_CMP<0x0c270000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CNE_D  : FP_CMP<0x0c280000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_COR_D  : FP_CMP<0x0c2a0000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_CUNE_D : FP_CMP<0x0c2c0000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SAF_D  : FP_CMP<0x0c208000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SUN_D  : FP_CMP<0x0c248000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SEQ_D  : FP_CMP<0x0c228000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SUEQ_D : FP_CMP<0x0c268000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SLT_D  : FP_CMP<0x0c218000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SULT_D : FP_CMP<0x0c258000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SLE_D  : FP_CMP<0x0c238000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SULE_D : FP_CMP<0x0c278000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SNE_D  : FP_CMP<0x0c288000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SOR_D  : FP_CMP<0x0c2a8000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;
def FCMP_SUNE_D : FP_CMP<0x0c2c8000, FPR64>,
                  Sched<[WriteFCMP, ReadFCMP, ReadFCMP]>;

// Conversion Instructions
def FFINT_S_L : FP_CONV<0x011d1800, FPR32, FPR64>,
                Sched<[WriteFCvtStoL, ReadFCvtStoL]>;
def FTINT_L_S : FP_CONV<0x011b2400, FPR64, FPR32>,
                Sched<[WriteFCvtLtoS, ReadFCvtLtoS]>;
def FTINTRM_L_S : FP_CONV<0x011a2400, FPR64, FPR32>,
                  Sched<[WriteFCvtLtoS_RM, ReadFCvtLtoS_RM]>;
def FTINTRP_L_S : FP_CONV<0x011a6400, FPR64, FPR32>,
                  Sched<[WriteFCvtLtoS_RP, ReadFCvtLtoS_RP]>;
def FTINTRZ_L_S : FP_CONV<0x011aa400, FPR64, FPR32>,
                  Sched<[WriteFCvtLtoS_RZ, ReadFCvtLtoS_RZ]>;
def FTINTRNE_L_S : FP_CONV<0x011ae400, FPR64, FPR32>,
                   Sched<[WriteFCvtLtoS_RNE, ReadFCvtLtoS_RNE]>;
def FCVT_S_D : FP_CONV<0x01191800, FPR32, FPR64>,
               Sched<[WriteFCvtStoD, ReadFCvtStoD]>;
def FCVT_D_S : FP_CONV<0x01192400, FPR64, FPR32>,
               Sched<[WriteFCvtDtoS, ReadFCvtDtoS]>;
def FFINT_D_W : FP_CONV<0x011d2000, FPR64, FPR32>,
                Sched<[WriteFCvtDtoW, ReadFCvtDtoW]>;
def FFINT_D_L : FP_CONV<0x011d2800, FPR64, FPR64>,
                Sched<[WriteFCvtDtoL, ReadFCvtDtoL]>;
def FTINT_W_D : FP_CONV<0x011b0800, FPR32, FPR64>,
                Sched<[WriteFCvtWtoD, ReadFCvtWtoD]>;
def FTINT_L_D : FP_CONV<0x011b2800, FPR64, FPR64>,
                Sched<[WriteFCvtLtoD, ReadFCvtLtoD]>;
def FTINTRM_W_D : FP_CONV<0x011a0800, FPR32, FPR64>,
                  Sched<[WriteFCvtWtoD_RM, ReadFCvtWtoD_RM]>;
def FTINTRM_L_D : FP_CONV<0x011a2800, FPR64, FPR64>,
                  Sched<[WriteFCvtLtoD_RM, ReadFCvtLtoD_RM]>;
def FTINTRP_W_D : FP_CONV<0x011a4800, FPR32, FPR64>,
                  Sched<[WriteFCvtWtoD_RP, ReadFCvtWtoD_RP]>;
def FTINTRP_L_D : FP_CONV<0x011a6800, FPR64, FPR64>,
                  Sched<[WriteFCvtLtoD_RP, ReadFCvtLtoD_RP]>;
def FTINTRZ_W_D : FP_CONV<0x011a8800, FPR32, FPR64>,
                  Sched<[WriteFCvtWtoD_RZ, ReadFCvtWtoD_RZ]>;
def FTINTRZ_L_D : FP_CONV<0x011aa800, FPR64, FPR64>,
                  Sched<[WriteFCvtLtoD_RZ, ReadFCvtLtoD_RZ]>;
def FTINTRNE_W_D : FP_CONV<0x011ac800, FPR32, FPR64>,
                   Sched<[WriteFCvtWtoD_RNE, ReadFCvtWtoD_RNE]>;
def FTINTRNE_L_D : FP_CONV<0x011ae800, FPR64, FPR64>,
                   Sched<[WriteFCvtLtoD_RNE, ReadFCvtLtoD_RNE]>;
def FRINT_D : FP_CONV<0x011e4800, FPR64, FPR64>,
              Sched<[WriteFCvtFDtoID, ReadFCvtFDtoID]>;


// Move Instructions
def FMOV_D        : FP_MOV<0x01149800, FPR64, FPR64>, 
                    Sched<[WriteFMovD, ReadFMovD]>;
def MOVFRH2GR_S   : FP_MOV<0x0114bc00, GPR, FPR64>,
                    Sched<[WriteFMovFRHtoGRS, ReadFMovFRHtoGRS]>;
let isCodeGenOnly = 1 in {
def MOVFR2GR_S_64 : FP_MOV<0x0114b400, GPR, FPR64>,
                    Sched<[WriteFMovFRtoGRS, ReadFMovFRtoGRS]>;
def FSEL_xD : FP_SEL<0x0d000000, FPR64>,
              Sched<[WriteFSEL, ReadFSEL, ReadFSEL, ReadFSEL]>;
} // isCodeGenOnly = 1
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Constraints = "$dst = $out" in {
def MOVGR2FRH_W : FPFmtMOV<0x0114ac00, (outs FPR64:$out),
                           (ins FPR64:$dst, GPR:$src),
                           "$dst, $src">,
                  Sched<[WriteFMovGRtoFRHW, ReadFMovGRtoFRHW]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0, Constraints = "$dst = $out"

// Common Memory Access Instructions
def FLD_D : FP_LOAD_2RI12<0x2b800000, FPR64>,
            Sched<[WriteFLD, ReadFMemBase]>;
def FST_D : FP_STORE_2RI12<0x2bc00000, FPR64>,
            Sched<[WriteFST, ReadFSTData, ReadFMemBase]>;
def FLDX_D : FP_LOAD_3R<0x38340000, FPR64>,
             Sched<[WriteFLDX, ReadFMemBase, ReadFMemOffset]>;
def FSTX_D : FP_STORE_3R<0x383c0000, FPR64>,
             Sched<[WriteFSTX, ReadFSTData, ReadFMemBase, ReadFMemOffset]>;

// Bound Check Memory Access Instructions
def FLDGT_D : FP_LOAD_3R<0x38748000, FPR64>,
              Sched<[WriteFLDGT, ReadFMemBase, ReadFMemCmp]>;
def FLDLE_D : FP_LOAD_3R<0x38758000, FPR64>, 
              Sched<[WriteFLDLE, ReadFMemBase, ReadFMemCmp]>;
def FSTGT_D : FP_STORE_3R<0x38768000, FPR64>,
              Sched<[WriteFSTGT, ReadFSTData, ReadFMemBase, ReadFMemCmp]>;
def FSTLE_D : FP_STORE_3R<0x38778000, FPR64>,
              Sched<[WriteFSTLE, ReadFSTData, ReadFMemBase, ReadFMemCmp]>;

} // Predicates = [HasBasicD]

// Instructions only available on LA64
let Predicates = [HasBasicD, IsLA64] in {
def MOVGR2FR_D  : FP_MOV<0x0114a800, FPR64, GPR>,
                  Sched<[WriteFMovGRtoFRD, ReadFMovGRtoFRD]>;
def MOVFR2GR_D  : FP_MOV<0x0114b800, GPR, FPR64>,
                  Sched<[WriteFMovFRtoGRD, ReadFMovFRtoGRD]>;
} // Predicates = [HasBasicD, IsLA64]

// Instructions only available on LA32
let Predicates = [HasBasicD, IsLA32], isCodeGenOnly = 1 in {
def MOVGR2FR_W_64 : FP_MOV<0x0114a400, FPR64, GPR>,
                    Sched<[WriteFMovGRtoFRW, ReadFMovGRtoFRW]>;
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
def : PatFprFpr<fmaxnum, FMAX_D, FPR64>;
def : PatFprFpr<fminnum_ieee, FMIN_D, FPR64>;
def : PatFprFpr<fminnum, FMIN_D, FPR64>;
def : PatFpr<fneg, FNEG_D, FPR64>;
def : PatFpr<fabs, FABS_D, FPR64>;
def : PatFpr<fsqrt, FSQRT_D, FPR64>;
def : Pat<(fdiv fpimm1, (fsqrt FPR64:$fj)), (FRSQRT_D FPR64:$fj)>;
let Predicates = [IsLA32] in {
def : Pat<(fdiv (loongarch_movgr2fr_d_lo_hi (i32 0), (i32 1072693248)),
                (fsqrt FPR64:$fj)),
          (FRSQRT_D FPR64:$fj)>;
} // Predicates = [IsLA32]
def : Pat<(fcopysign FPR64:$fj, FPR32:$fk),
          (FCOPYSIGN_D FPR64:$fj, (FCVT_D_S FPR32:$fk))>;
def : Pat<(fcopysign FPR32:$fj, FPR64:$fk),
          (FCOPYSIGN_S FPR32:$fj, (FCVT_S_D FPR64:$fk))>;
def : Pat<(fcanonicalize FPR64:$fj), (FMAX_D $fj, $fj)>;
let Predicates = [IsLA32] in {
def : Pat<(is_fpclass FPR64:$fj, (i32 timm:$mask)),
          (SLTU R0, (ANDI (MOVFR2GR_S_64 (FCLASS_D FPR64:$fj)),
                          (to_fclass_mask timm:$mask)))>;
} // Predicates = [IsLA32]
let Predicates = [IsLA64] in {
def : Pat<(is_fpclass FPR64:$fj, (i32 timm:$mask)),
          (SLTU R0, (ANDI (MOVFR2GR_D (FCLASS_D FPR64:$fj)),
                          (to_fclass_mask timm:$mask)))>;
} // Predicates = [IsLA64]

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
defm : PatFPBrcond<SETEQ,  FCMP_CEQ_D, FPR64>;
defm : PatFPBrcond<SETOLT, FCMP_CLT_D, FPR64>;
defm : PatFPBrcond<SETOLE, FCMP_CLE_D, FPR64>;
defm : PatFPBrcond<SETLE,  FCMP_CLE_D, FPR64>;
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
          (FSEL_xD FPR64:$fj, FPR64:$fk, CFR:$cc)>;

/// Selectcc

def : PatFPSelectcc<SETOEQ, FCMP_CEQ_D,  FSEL_xD, FPR64>;
def : PatFPSelectcc<SETOLT, FCMP_CLT_D,  FSEL_xD, FPR64>;
def : PatFPSelectcc<SETOLE, FCMP_CLE_D,  FSEL_xD, FPR64>;
def : PatFPSelectcc<SETONE, FCMP_CNE_D,  FSEL_xD, FPR64>;
def : PatFPSelectcc<SETO,   FCMP_COR_D,  FSEL_xD, FPR64>;
def : PatFPSelectcc<SETUEQ, FCMP_CUEQ_D, FSEL_xD, FPR64>;
def : PatFPSelectcc<SETULT, FCMP_CULT_D, FSEL_xD, FPR64>;
def : PatFPSelectcc<SETULE, FCMP_CULE_D, FSEL_xD, FPR64>;
def : PatFPSelectcc<SETUNE, FCMP_CUNE_D, FSEL_xD, FPR64>;
def : PatFPSelectcc<SETUO,  FCMP_CUN_D,  FSEL_xD, FPR64>;

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
let Predicates = [IsLA32] in {
def : Pat<(fdiv (loongarch_movgr2fr_d_lo_hi (i32 0), (i32 1072693248)), FPR64:$src),
          (FRECIP_D FPR64:$src)>;
} // Predicates = [IsLA32]

let Predicates = [HasFrecipe] in {
// FP approximate reciprocal operation
def : Pat<(int_loongarch_frecipe_d FPR64:$src), (FRECIPE_D FPR64:$src)>;
def : Pat<(int_loongarch_frsqrte_d FPR64:$src), (FRSQRTE_D FPR64:$src)>;
def : Pat<(loongarch_frecipe FPR64:$src), (FRECIPE_D FPR64:$src)>;
def : Pat<(loongarch_frsqrte FPR64:$src), (FRSQRTE_D FPR64:$src)>;
}

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
def : Pat<(fneg (fma FPR64:$fj, FPR64:$fk, (fneg FPR64:$fa))),
          (FNMSUB_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;

// fnmsub.d: -fj * fk + fa (the nsz flag on the FMA)
def : Pat<(fma_nsz (fneg FPR64:$fj), FPR64:$fk, FPR64:$fa),
          (FNMSUB_D FPR64:$fj, FPR64:$fk, FPR64:$fa)>;

// ffint.d.l
def : Pat<(loongarch_sitof FPR64:$fj), (FFINT_D_L FPR64:$fj)>;
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
def : Pat<(loongarch_movgr2fr_d GPR:$src), (MOVGR2FR_D GPR:$src)>;
} // Predicates = [HasBasicD, IsLA64]
let Predicates = [HasBasicD, IsLA32] in {
def : Pat<(f64 (sint_to_fp (i32 GPR:$src))), (FFINT_D_W (MOVGR2FR_W GPR:$src))>;

def : Pat<(f64 (loongarch_movgr2fr_d_lo_hi (i32 GPR:$lo), (i32 GPR:$hi))),
          (MOVGR2FRH_W (MOVGR2FR_W_64 GPR:$lo), GPR:$hi)>;
} // Predicates = [HasBasicD, IsLA32]

// Convert FP to int
let Predicates = [HasBasicD, IsLA64] in {
def : Pat<(bitconvert FPR64:$src), (MOVFR2GR_D FPR64:$src)>;
} // Predicates = [HasBasicD, IsLA64]

// FP Rounding
let Predicates = [HasBasicD, IsLA64] in {
def : PatFpr<frint, FRINT_D, FPR64>;
} // Predicates = [HasBasicD, IsLA64]

/// Pseudo-instructions needed for the soft-float ABI with LA32D

let Predicates = [HasBasicD, IsLA32] in {
// Moves two GPRs to an FPR.
let usesCustomInserter = 1 in
def BuildPairF64Pseudo
    : Pseudo<(outs FPR64:$dst), (ins GPR:$src1, GPR:$src2),
             [(set FPR64:$dst, (loongarch_build_pair_f64 GPR:$src1, GPR:$src2))]>;

// Moves an FPR to two GPRs.
let usesCustomInserter = 1 in
def SplitPairF64Pseudo
    : Pseudo<(outs GPR:$dst1, GPR:$dst2), (ins FPR64:$src),
             [(set GPR:$dst1, GPR:$dst2, (loongarch_split_pair_f64 FPR64:$src))]>;
} // Predicates = [HasBasicD, IsLA32]
