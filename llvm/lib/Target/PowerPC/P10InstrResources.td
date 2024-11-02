//===--- P10InstrResources.td - P10 Scheduling Definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//
// This file defines the itinerary class data for the POWER10 processor.
//
//===----------------------------------------------------------------------===//
// 22 Cycles Binary Floating Point operations, 2 input operands
def : InstRW<[P10W_BF_22C, P10W_DISP_ANY, P10BF_Read, P10BF_Read],
      (instrs
    FDIVS,
    XSDIVSP
)>;

// 2-way crack instructions
// 22 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_BF_22C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FDIVS_rec
)>;

// 24 Cycles Binary Floating Point operations, 2 input operands
def : InstRW<[P10W_BF_24C, P10W_DISP_ANY, P10BF_Read, P10BF_Read],
      (instrs
    XVDIVSP
)>;

// 26 Cycles Binary Floating Point operations, 1 input operands
def : InstRW<[P10W_BF_26C, P10W_DISP_ANY, P10BF_Read],
      (instrs
    FSQRTS,
    XSSQRTSP
)>;

// 2-way crack instructions
// 26 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_BF_26C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FSQRTS_rec
)>;

// 27 Cycles Binary Floating Point operations, 1 input operands
def : InstRW<[P10W_BF_27C, P10W_DISP_ANY, P10BF_Read],
      (instrs
    XVSQRTSP
)>;

// 27 Cycles Binary Floating Point operations, 2 input operands
def : InstRW<[P10W_BF_27C, P10W_DISP_ANY, P10BF_Read, P10BF_Read],
      (instrs
    FDIV,
    XSDIVDP,
    XVDIVDP
)>;

// 2-way crack instructions
// 27 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_BF_27C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FDIV_rec
)>;

// 36 Cycles Binary Floating Point operations, 1 input operands
def : InstRW<[P10W_BF_36C, P10W_DISP_ANY, P10BF_Read],
      (instrs
    FSQRT,
    XSSQRTDP,
    XVSQRTDP
)>;

// 2-way crack instructions
// 36 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_BF_36C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FSQRT_rec
)>;

// 7 Cycles Binary Floating Point operations, 1 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_ANY, P10BF_Read],
      (instrs
    FCFID,
    FCFIDS,
    FCFIDU,
    FCFIDUS,
    FCTID,
    FCTIDU,
    FCTIDUZ,
    FCTIDZ,
    FCTIW,
    FCTIWU,
    FCTIWUZ,
    FCTIWZ,
    FRE,
    FRES,
    FRIMD, FRIMS,
    FRIND, FRINS,
    FRIPD, FRIPS,
    FRIZD, FRIZS,
    FRSP,
    FRSQRTE,
    FRSQRTES,
    VCFSX, VCFSX_0,
    VCFUX, VCFUX_0,
    VCTSXS, VCTSXS_0,
    VCTUXS, VCTUXS_0,
    VLOGEFP,
    VREFP,
    VRFIM,
    VRFIN,
    VRFIP,
    VRFIZ,
    VRSQRTEFP,
    XSCVDPHP,
    XSCVDPSP,
    XSCVDPSPN,
    XSCVDPSXDS, XSCVDPSXDSs,
    XSCVDPSXWS, XSCVDPSXWSs,
    XSCVDPUXDS, XSCVDPUXDSs,
    XSCVDPUXWS, XSCVDPUXWSs,
    XSCVSPDP,
    XSCVSXDDP,
    XSCVSXDSP,
    XSCVUXDDP,
    XSCVUXDSP,
    XSRDPI,
    XSRDPIC,
    XSRDPIM,
    XSRDPIP,
    XSRDPIZ,
    XSREDP,
    XSRESP,
    XSRSP,
    XSRSQRTEDP,
    XSRSQRTESP,
    XVCVDPSP,
    XVCVDPSXDS,
    XVCVDPSXWS,
    XVCVDPUXDS,
    XVCVDPUXWS,
    XVCVSPBF16,
    XVCVSPDP,
    XVCVSPHP,
    XVCVSPSXDS,
    XVCVSPSXWS,
    XVCVSPUXDS,
    XVCVSPUXWS,
    XVCVSXDDP,
    XVCVSXDSP,
    XVCVSXWDP,
    XVCVSXWSP,
    XVCVUXDDP,
    XVCVUXDSP,
    XVCVUXWDP,
    XVCVUXWSP,
    XVRDPI,
    XVRDPIC,
    XVRDPIM,
    XVRDPIP,
    XVRDPIZ,
    XVREDP,
    XVRESP,
    XVRSPI,
    XVRSPIC,
    XVRSPIM,
    XVRSPIP,
    XVRSPIZ,
    XVRSQRTEDP,
    XVRSQRTESP
)>;

// 7 Cycles Binary Floating Point operations, 2 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_ANY, P10BF_Read, P10BF_Read],
      (instrs
    FADD,
    FADDS,
    FMUL,
    FMULS,
    FSUB,
    FSUBS,
    VADDFP,
    VSUBFP,
    XSADDDP,
    XSADDSP,
    XSMULDP,
    XSMULSP,
    XSSUBDP,
    XSSUBSP,
    XVADDDP,
    XVADDSP,
    XVMULDP,
    XVMULSP,
    XVSUBDP,
    XVSUBSP
)>;

// 7 Cycles Binary Floating Point operations, 3 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_ANY, P10BF_Read, P10BF_Read, P10BF_Read],
      (instrs
    FMADD,
    FMADDS,
    FMSUB,
    FMSUBS,
    FNMADD,
    FNMADDS,
    FNMSUB,
    FNMSUBS,
    FSELD, FSELS,
    VMADDFP,
    VNMSUBFP,
    XSMADDADP,
    XSMADDASP,
    XSMADDMDP,
    XSMADDMSP,
    XSMSUBADP,
    XSMSUBASP,
    XSMSUBMDP,
    XSMSUBMSP,
    XSNMADDADP,
    XSNMADDASP,
    XSNMADDMDP,
    XSNMADDMSP,
    XSNMSUBADP,
    XSNMSUBASP,
    XSNMSUBMDP,
    XSNMSUBMSP,
    XVMADDADP,
    XVMADDASP,
    XVMADDMDP,
    XVMADDMSP,
    XVMSUBADP,
    XVMSUBASP,
    XVMSUBMDP,
    XVMSUBMSP,
    XVNMADDADP,
    XVNMADDASP,
    XVNMADDMDP,
    XVNMADDMSP,
    XVNMSUBADP,
    XVNMSUBASP,
    XVNMSUBMDP,
    XVNMSUBMSP
)>;

// 2-way crack instructions
// 7 Cycles Binary Floating Point operations, and 7 Cycles Binary Floating Point operations, 1 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_EVEN, P10W_BF_7C, P10W_DISP_ANY, P10BF_Read],
      (instrs
    VEXPTEFP
)>;

// 2-way crack instructions
// 7 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FADD_rec,
    FADDS_rec,
    FMUL_rec,
    FMULS_rec,
    FSUB_rec,
    FSUBS_rec
)>;

// 2-way crack instructions
// 7 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FCFID_rec,
    FCFIDS_rec,
    FCFIDU_rec,
    FCFIDUS_rec,
    FCTID_rec,
    FCTIDU_rec,
    FCTIDUZ_rec,
    FCTIDZ_rec,
    FCTIW_rec,
    FCTIWU_rec,
    FCTIWUZ_rec,
    FCTIWZ_rec,
    FRE_rec,
    FRES_rec,
    FRIMD_rec, FRIMS_rec,
    FRIND_rec, FRINS_rec,
    FRIPD_rec, FRIPS_rec,
    FRIZD_rec, FRIZS_rec,
    FRSP_rec,
    FRSQRTE_rec,
    FRSQRTES_rec
)>;

// 2-way crack instructions
// 7 Cycles Binary Floating Point operations, and 3 Cycles ALU operations, 3 input operands
def : InstRW<[P10W_BF_7C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    FMADD_rec,
    FMADDS_rec,
    FMSUB_rec,
    FMSUBS_rec,
    FNMADD_rec,
    FNMADDS_rec,
    FNMSUB_rec,
    FNMSUBS_rec,
    FSELD_rec, FSELS_rec
)>;

// 2 Cycles Branch operations, 0 input operands
def : InstRW<[P10W_BR_2C, P10W_DISP_ANY],
      (instrs
    BCLR, BCLRn, BDNZLR, BDNZLR8, BDNZLRm, BDNZLRp, BDZLR, BDZLR8, BDZLRm, BDZLRp, gBCLR,
    BCLRL, BCLRLn, BDNZLRL, BDNZLRLm, BDNZLRLp, BDZLRL, BDZLRLm, BDZLRLp, gBCLRL,
    BL, BL8, BL8_NOP, BL8_NOP_RM, BL8_NOP_TLS, BL8_NOTOC, BL8_NOTOC_RM, BL8_NOTOC_TLS, BL8_RM, BL8_TLS, BL8_TLS_, BLR, BLR8, BLRL, BL_NOP, BL_NOP_RM, BL_RM, BL_TLS
)>;

// 2 Cycles Branch operations, 1 input operands
def : InstRW<[P10W_BR_2C, P10W_DISP_ANY, P10BR_Read],
      (instrs
    B, BCC, BCCA, BCCCTR, BCCCTR8, BCCCTRL, BCCCTRL8, BCCL, BCCLA, BCCLR, BCCLRL, CTRL_DEP, TAILB, TAILB8,
    BA, TAILBA, TAILBA8,
    BC, BCTR, BCTR8, BCTRL, BCTRL8, BCTRL8_LDinto_toc, BCTRL8_LDinto_toc_RM, BCTRL8_RM, BCTRL_LWZinto_toc, BCTRL_LWZinto_toc_RM, BCTRL_RM, BCn, BDNZ, BDNZ8, BDNZm, BDNZp, BDZ, BDZ8, BDZm, BDZp, TAILBCTR, TAILBCTR8, gBC, gBCat,
    BCL, BCLalways, BCLn, BDNZL, BDNZLm, BDNZLp, BDZL, BDZLm, BDZLp, gBCL, gBCLat,
    BLA, BLA8, BLA8_NOP, BLA8_NOP_RM, BLA8_RM, BLA_RM
)>;

// 2 Cycles Branch operations, 3 input operands
def : InstRW<[P10W_BR_2C, P10W_DISP_ANY, P10BR_Read, P10BR_Read, P10BR_Read],
      (instrs
    BCCTR, BCCTR8, BCCTR8n, BCCTRn, gBCCTR,
    BCCTRL, BCCTRL8, BCCTRL8n, BCCTRLn, gBCCTRL
)>;

// 2 Cycles Branch operations, 4 input operands
def : InstRW<[P10W_BR_2C, P10W_DISP_ANY, P10BR_Read, P10BR_Read, P10BR_Read, P10BR_Read],
      (instrs
    BDNZA, BDNZAm, BDNZAp, BDZA, BDZAm, BDZAp, gBCA, gBCAat,
    BDNZLA, BDNZLAm, BDNZLAp, BDZLA, BDZLAm, BDZLAp, gBCLA, gBCLAat
)>;

// 7 Cycles Crypto operations, 1 input operands
def : InstRW<[P10W_CY_7C, P10W_DISP_ANY, P10CY_Read],
      (instrs
    VSBOX
)>;

// 7 Cycles Crypto operations, 2 input operands
def : InstRW<[P10W_CY_7C, P10W_DISP_ANY, P10CY_Read, P10CY_Read],
      (instrs
    CFUGED,
    CNTLZDM,
    CNTTZDM,
    PDEPD,
    PEXTD,
    VCFUGED,
    VCIPHER,
    VCIPHERLAST,
    VCLZDM,
    VCTZDM,
    VGNB,
    VNCIPHER,
    VNCIPHERLAST,
    VPDEPD,
    VPEXTD,
    VPMSUMB,
    VPMSUMD,
    VPMSUMH,
    VPMSUMW
)>;

// 13 Cycles Decimal Floating Point operations, 1 input operands
def : InstRW<[P10W_DF_13C, P10W_DISP_ANY, P10DF_Read],
      (instrs
    XSCVDPQP,
    XSCVQPDP,
    XSCVQPDPO,
    XSCVQPSDZ,
    XSCVQPSQZ,
    XSCVQPSWZ,
    XSCVQPUDZ,
    XSCVQPUQZ,
    XSCVQPUWZ,
    XSCVSDQP,
    XSCVSQQP,
    XSCVUDQP,
    XSCVUQQP
)>;

// 13 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_13C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    XSADDQP,
    XSADDQPO,
    XSSUBQP,
    XSSUBQPO
)>;

// 13 Cycles Decimal Floating Point operations, 3 input operands
def : InstRW<[P10W_DF_13C, P10W_DISP_ANY, P10DF_Read, P10DF_Read, P10DF_Read],
      (instrs
    BCDSR_rec,
    XSRQPI,
    XSRQPIX,
    XSRQPXP
)>;

// 2-way crack instructions
// 13 Cycles Decimal Floating Point operations, and 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_DF_13C, P10W_DISP_EVEN, P10W_ST_3C, P10W_DISP_ANY],
      (instrs
    HASHST, HASHST8,
    HASHSTP, HASHSTP8
)>;

// 24 Cycles Decimal Floating Point operations, 1 input operands
def : InstRW<[P10W_DF_24C, P10W_DISP_ANY, P10DF_Read],
      (instrs
    BCDCTSQ_rec
)>;

// 25 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_25C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    XSMULQP,
    XSMULQPO
)>;

// 25 Cycles Decimal Floating Point operations, 3 input operands
def : InstRW<[P10W_DF_25C, P10W_DISP_ANY, P10DF_Read, P10DF_Read, P10DF_Read],
      (instrs
    XSMADDQP,
    XSMADDQPO,
    XSMSUBQP,
    XSMSUBQPO,
    XSNMADDQP,
    XSNMADDQPO,
    XSNMSUBQP,
    XSNMSUBQPO
)>;

// 38 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_38C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    BCDCFSQ_rec
)>;

// 59 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_59C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    XSDIVQP,
    XSDIVQPO
)>;

// 61 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_61C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    VDIVESQ,
    VDIVEUQ,
    VDIVSQ,
    VDIVUQ
)>;

// 68 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_DF_68C, P10W_DISP_ANY, P10DF_Read, P10DF_Read],
      (instrs
    VMODSQ,
    VMODUQ
)>;

// 77 Cycles Decimal Floating Point operations, 1 input operands
def : InstRW<[P10W_DF_77C, P10W_DISP_ANY, P10DF_Read],
      (instrs
    XSSQRTQP,
    XSSQRTQPO
)>;

// 20 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_20C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    DIVW,
    DIVWO,
    DIVWU,
    DIVWUO,
    MODSW
)>;

// 2-way crack instructions
// 20 Cycles Scalar Fixed-Point Divide operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_DV_20C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    DIVW_rec,
    DIVWO_rec,
    DIVWU_rec,
    DIVWUO_rec
)>;

// 25 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_25C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    DIVD,
    DIVDO,
    DIVDU,
    DIVDUO,
    DIVWE,
    DIVWEO,
    DIVWEU,
    DIVWEUO
)>;

// 2-way crack instructions
// 25 Cycles Scalar Fixed-Point Divide operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_DV_25C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    DIVD_rec,
    DIVDO_rec,
    DIVDU_rec,
    DIVDUO_rec,
    DIVWE_rec,
    DIVWEO_rec,
    DIVWEU_rec,
    DIVWEUO_rec
)>;

// 27 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_27C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    MODSD,
    MODUD,
    MODUW
)>;

// 41 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_41C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    DIVDE,
    DIVDEO,
    DIVDEU,
    DIVDEUO
)>;

// 2-way crack instructions
// 41 Cycles Scalar Fixed-Point Divide operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_DV_41C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    DIVDE_rec,
    DIVDEO_rec,
    DIVDEU_rec,
    DIVDEUO_rec
)>;

// 43 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_43C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VDIVSD,
    VDIVUD
)>;

// 47 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_47C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VMODSD,
    VMODUD
)>;

// 54 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_54C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VDIVSW,
    VDIVUW
)>;

// 60 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_60C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VMODSW,
    VMODUW
)>;

// 75 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_75C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VDIVESD,
    VDIVEUD
)>;

// 83 Cycles Scalar Fixed-Point Divide operations, 2 input operands
def : InstRW<[P10W_DV_83C, P10W_DISP_ANY, P10DV_Read, P10DV_Read],
      (instrs
    VDIVESW,
    VDIVEUW
)>;

// 5 Cycles Fixed-Point and BCD operations, 1 input operands
def : InstRW<[P10W_DX_5C, P10W_DISP_ANY, P10DX_Read],
      (instrs
    BCDCTN_rec,
    VMUL10CUQ,
    VMUL10UQ,
    XSXSIGQP
)>;

// 5 Cycles Fixed-Point and BCD operations, 2 input operands
def : InstRW<[P10W_DX_5C, P10W_DISP_ANY, P10DX_Read, P10DX_Read],
      (instrs
    BCDCFN_rec,
    BCDCFZ_rec,
    BCDCPSGN_rec,
    BCDCTZ_rec,
    BCDSETSGN_rec,
    BCDUS_rec,
    BCDUTRUNC_rec,
    VADDCUQ,
    VADDUQM,
    VMUL10ECUQ,
    VMUL10EUQ,
    VSUBCUQ,
    VSUBUQM,
    XSCMPEQQP,
    XSCMPEXPQP,
    XSCMPGEQP,
    XSCMPGTQP,
    XSCMPOQP,
    XSCMPUQP,
    XSMAXCQP,
    XSMINCQP,
    XSTSTDCQP,
    XXGENPCVBM
)>;

// 5 Cycles Fixed-Point and BCD operations, 3 input operands
def : InstRW<[P10W_DX_5C, P10W_DISP_ANY, P10DX_Read, P10DX_Read, P10DX_Read],
      (instrs
    BCDADD_rec,
    BCDS_rec,
    BCDSUB_rec,
    BCDTRUNC_rec,
    VADDECUQ,
    VADDEUQM,
    VSUBECUQ,
    VSUBEUQM
)>;

// 4 Cycles ALU2 operations, 0 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY],
      (instrs
    TRAP, TW
)>;

// 4 Cycles ALU2 operations, 1 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY, P10F2_Read],
      (instrs
    CNTLZD,
    CNTLZD_rec,
    CNTLZW, CNTLZW8,
    CNTLZW8_rec, CNTLZW_rec,
    CNTTZD,
    CNTTZD_rec,
    CNTTZW, CNTTZW8,
    CNTTZW8_rec, CNTTZW_rec,
    FTSQRT,
    MTVSRBM,
    MTVSRBMI,
    MTVSRDM,
    MTVSRHM,
    MTVSRQM,
    MTVSRWM,
    POPCNTB, POPCNTB8,
    POPCNTD,
    POPCNTW,
    VCLZB,
    VCLZD,
    VCLZH,
    VCLZW,
    VCTZB,
    VCTZD,
    VCTZH,
    VCTZW,
    VEXPANDBM,
    VEXPANDDM,
    VEXPANDHM,
    VEXPANDQM,
    VEXPANDWM,
    VEXTRACTBM,
    VEXTRACTDM,
    VEXTRACTHM,
    VEXTRACTQM,
    VEXTRACTWM,
    VPOPCNTB,
    VPOPCNTD,
    VPOPCNTH,
    VPOPCNTW,
    VPRTYBD,
    VPRTYBW,
    XSCVHPDP,
    XSCVSPDPN,
    XSTSQRTDP,
    XVCVHPSP,
    XVTLSBB,
    XVTSQRTDP,
    XVTSQRTSP
)>;

// 4 Cycles ALU2 operations, 2 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY, P10F2_Read, P10F2_Read],
      (instrs
    CMPEQB,
    EXTSWSLI_32_64_rec, EXTSWSLI_rec,
    FCMPOD, FCMPOS,
    FCMPUD, FCMPUS,
    FTDIV,
    SLD_rec,
    SLW8_rec, SLW_rec,
    SRD_rec,
    SRW8_rec, SRW_rec,
    VABSDUB,
    VABSDUH,
    VABSDUW,
    VADDCUW,
    VADDSBS,
    VADDSHS,
    VADDSWS,
    VADDUBS,
    VADDUHS,
    VADDUWS,
    VAVGSB,
    VAVGSH,
    VAVGSW,
    VAVGUB,
    VAVGUH,
    VAVGUW,
    VCMPBFP,
    VCMPBFP_rec,
    VCMPEQFP,
    VCMPEQFP_rec,
    VCMPEQUB_rec,
    VCMPEQUD_rec,
    VCMPEQUH_rec,
    VCMPEQUQ,
    VCMPEQUQ_rec,
    VCMPEQUW_rec,
    VCMPGEFP,
    VCMPGEFP_rec,
    VCMPGTFP,
    VCMPGTFP_rec,
    VCMPGTSB_rec,
    VCMPGTSD_rec,
    VCMPGTSH_rec,
    VCMPGTSQ,
    VCMPGTSQ_rec,
    VCMPGTSW_rec,
    VCMPGTUB_rec,
    VCMPGTUD_rec,
    VCMPGTUH_rec,
    VCMPGTUQ,
    VCMPGTUQ_rec,
    VCMPGTUW_rec,
    VCMPNEB_rec,
    VCMPNEH_rec,
    VCMPNEW_rec,
    VCMPNEZB_rec,
    VCMPNEZH_rec,
    VCMPNEZW_rec,
    VCMPSQ,
    VCMPUQ,
    VCNTMBB,
    VCNTMBD,
    VCNTMBH,
    VCNTMBW,
    VMAXFP,
    VMINFP,
    VSUBCUW,
    VSUBSBS,
    VSUBSHS,
    VSUBSWS,
    VSUBUBS,
    VSUBUHS,
    VSUBUWS,
    XSCMPEQDP,
    XSCMPEXPDP,
    XSCMPGEDP,
    XSCMPGTDP,
    XSCMPODP,
    XSCMPUDP,
    XSMAXCDP,
    XSMAXDP,
    XSMAXJDP,
    XSMINCDP,
    XSMINDP,
    XSMINJDP,
    XSTDIVDP,
    XSTSTDCDP,
    XSTSTDCSP,
    XVCMPEQDP,
    XVCMPEQDP_rec,
    XVCMPEQSP,
    XVCMPEQSP_rec,
    XVCMPGEDP,
    XVCMPGEDP_rec,
    XVCMPGESP,
    XVCMPGESP_rec,
    XVCMPGTDP,
    XVCMPGTDP_rec,
    XVCMPGTSP,
    XVCMPGTSP_rec,
    XVMAXDP,
    XVMAXSP,
    XVMINDP,
    XVMINSP,
    XVTDIVDP,
    XVTDIVSP,
    XVTSTDCDP,
    XVTSTDCSP
)>;

// 4 Cycles ALU2 operations, 3 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY, P10F2_Read, P10F2_Read, P10F2_Read],
      (instrs
    CMPRB, CMPRB8,
    RLDCL_rec,
    RLDCR_rec,
    RLDIC_rec,
    RLDICL_32_rec, RLDICL_rec,
    RLDICR_rec,
    TD,
    TDI,
    TWI,
    VSHASIGMAD,
    VSHASIGMAW
)>;

// 4 Cycles ALU2 operations, 4 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY, P10F2_Read, P10F2_Read, P10F2_Read, P10F2_Read],
      (instrs
    RLDIMI_rec,
    RLWINM8_rec, RLWINM_rec,
    RLWNM8_rec, RLWNM_rec
)>;

// 4 Cycles ALU2 operations, 5 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_ANY, P10F2_Read, P10F2_Read, P10F2_Read, P10F2_Read, P10F2_Read],
      (instrs
    RLWIMI8_rec, RLWIMI_rec
)>;

// Single crack instructions
// 4 Cycles ALU2 operations, 2 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_EVEN, P10W_DISP_ANY, P10F2_Read, P10F2_Read],
      (instrs
    SRAD_rec,
    SRADI_rec,
    SRAW_rec,
    SRAWI_rec
)>;

// Single crack instructions
// 4 Cycles ALU2 operations, 3 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_EVEN, P10W_DISP_ANY, P10F2_Read, P10F2_Read, P10F2_Read],
      (instrs
    TABORTDC,
    TABORTDCI,
    TABORTWC,
    TABORTWCI
)>;

// 2-way crack instructions
// 4 Cycles ALU2 operations, and 4 Cycles Permute operations, 2 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_EVEN, P10W_PM_4C, P10W_DISP_ANY],
      (instrs
    VRLQ,
    VRLQNM,
    VSLQ,
    VSRAQ,
    VSRQ
)>;

// 2-way crack instructions
// 4 Cycles ALU2 operations, and 4 Cycles Permute operations, 3 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_EVEN, P10W_PM_4C, P10W_DISP_ANY],
      (instrs
    VRLQMI
)>;

// 2-way crack instructions
// 4 Cycles ALU2 operations, and 4 Cycles ALU2 operations, 0 input operands
def : InstRW<[P10W_F2_4C, P10W_DISP_PAIR, P10W_F2_4C],
      (instrs
    MFCR, MFCR8
)>;

// 2 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_FX_2C, P10W_DISP_ANY, P10FX_Read],
      (instrs
    MTCTR, MTCTR8, MTCTR8loop, MTCTRloop,
    MTLR, MTLR8
)>;

// 3 Cycles ALU operations, 0 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    CR6SET, CREQV, CRSET,
    DSS, DSSALL,
    MCRXRX,
    MFCTR, MFCTR8,
    MFLR, MFLR8,
    NOP, NOP_GT_PWR6, NOP_GT_PWR7, ORI, ORI8,
    VXOR, V_SET0, V_SET0B, V_SET0H,
    XXLEQV, XXLEQVOnes,
    XXLXOR, XXLXORdpz, XXLXORspz, XXLXORz
)>;

// 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY, P10FX_Read],
      (instrs
    ADDI, ADDI8, ADDIdtprelL32,  ADDItlsldLADDR32,  ADDItocL, LI, LI8,
    ADDIS, ADDIS8,  ADDISdtprelHA32, ADDIStocHA,  ADDIStocHA8, LIS, LIS8,
    ADDME, ADDME8,
    ADDME8O, ADDMEO,
    ADDZE, ADDZE8,
    ADDZE8O, ADDZEO,
    EXTSB, EXTSB8, EXTSB8_32_64,
    EXTSB8_rec, EXTSB_rec,
    EXTSH, EXTSH8, EXTSH8_32_64,
    EXTSH8_rec, EXTSH_rec,
    EXTSW, EXTSW_32, EXTSW_32_64,
    EXTSW_32_64_rec, EXTSW_rec,
    FABSD, FABSS,
    FMR,
    FNABSD, FNABSS,
    FNEGD, FNEGS,
    MCRF,
    MFOCRF, MFOCRF8,
    MFVRD, MFVSRD,
    MFVRWZ, MFVSRWZ,
    MTOCRF, MTOCRF8,
    MTVRD, MTVSRD,
    MTVRWA, MTVSRWA,
    MTVRWZ, MTVSRWZ,
    NEG, NEG8,
    NEG8_rec, NEG_rec,
    NEG8O, NEGO,
    SETB, SETB8,
    SETBC, SETBC8,
    SETBCR, SETBCR8,
    SETNBC, SETNBC8,
    SETNBCR, SETNBCR8,
    SUBFME, SUBFME8,
    SUBFME8O, SUBFMEO,
    SUBFZE, SUBFZE8,
    SUBFZE8O, SUBFZEO,
    VEXTSB2D, VEXTSB2Ds,
    VEXTSB2W, VEXTSB2Ws,
    VEXTSD2Q,
    VEXTSH2D, VEXTSH2Ds,
    VEXTSH2W, VEXTSH2Ws,
    VEXTSW2D, VEXTSW2Ds,
    VNEGD,
    VNEGW,
    WAIT,
    XSABSDP,
    XSABSQP,
    XSNABSDP, XSNABSDPs,
    XSNABSQP,
    XSNEGDP,
    XSNEGQP,
    XSXEXPDP,
    XSXEXPQP,
    XSXSIGDP,
    XVABSDP,
    XVABSSP,
    XVNABSDP,
    XVNABSSP,
    XVNEGDP,
    XVNEGSP,
    XVXEXPDP,
    XVXEXPSP,
    XVXSIGDP,
    XVXSIGSP
)>;

// 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read],
      (instrs
    ADD4, ADD4TLS, ADD8, ADD8TLS, ADD8TLS_,
    ADD4_rec, ADD8_rec,
    ADDE, ADDE8,
    ADDE8O, ADDEO,
    ADDIC, ADDIC8,
    ADD4O, ADD8O,
    AND, AND8,
    AND8_rec, AND_rec,
    ANDC, ANDC8,
    ANDC8_rec, ANDC_rec,
    ANDI8_rec, ANDI_rec,
    ANDIS8_rec, ANDIS_rec,
    CMPD, CMPW,
    CMPB, CMPB8,
    CMPDI, CMPWI,
    CMPLD, CMPLW,
    CMPLDI, CMPLWI,
    CRAND,
    CRANDC,
    CRNAND,
    CRNOR,
    CRNOT,
    CROR,
    CRORC,
    CR6UNSET, CRUNSET, CRXOR,
    EQV, EQV8,
    EQV8_rec, EQV_rec,
    EXTSWSLI, EXTSWSLI_32_64,
    FCPSGND, FCPSGNS,
    NAND, NAND8,
    NAND8_rec, NAND_rec,
    NOR, NOR8,
    NOR8_rec, NOR_rec,
    COPY, OR, OR8,
    OR8_rec, OR_rec,
    ORC, ORC8,
    ORC8_rec, ORC_rec,
    ORIS, ORIS8,
    SLD,
    SLW, SLW8,
    SRAD,
    SRADI, SRADI_32,
    SRAW,
    SRAWI,
    SRD,
    SRW, SRW8,
    SUBF, SUBF8,
    SUBF8_rec, SUBF_rec,
    SUBFE, SUBFE8,
    SUBFE8O, SUBFEO,
    SUBFIC, SUBFIC8,
    SUBF8O, SUBFO,
    VADDUBM,
    VADDUDM,
    VADDUHM,
    VADDUWM,
    VAND,
    VANDC,
    VCMPEQUB,
    VCMPEQUD,
    VCMPEQUH,
    VCMPEQUW,
    VCMPGTSB,
    VCMPGTSD,
    VCMPGTSH,
    VCMPGTSW,
    VCMPGTUB,
    VCMPGTUD,
    VCMPGTUH,
    VCMPGTUW,
    VCMPNEB,
    VCMPNEH,
    VCMPNEW,
    VCMPNEZB,
    VCMPNEZH,
    VCMPNEZW,
    VEQV,
    VMAXSB,
    VMAXSD,
    VMAXSH,
    VMAXSW,
    VMAXUB,
    VMAXUD,
    VMAXUH,
    VMAXUW,
    VMINSB,
    VMINSD,
    VMINSH,
    VMINSW,
    VMINUB,
    VMINUD,
    VMINUH,
    VMINUW,
    VMRGEW,
    VMRGOW,
    VNAND,
    VNOR,
    VOR,
    VORC,
    VRLB,
    VRLD,
    VRLDNM,
    VRLH,
    VRLW,
    VRLWNM,
    VSLB,
    VSLD,
    VSLH,
    VSLW,
    VSRAB,
    VSRAD,
    VSRAH,
    VSRAW,
    VSRB,
    VSRD,
    VSRH,
    VSRW,
    VSUBUBM,
    VSUBUDM,
    VSUBUHM,
    VSUBUWM,
    XOR, XOR8,
    XOR8_rec, XOR_rec,
    XORI, XORI8,
    XORIS, XORIS8,
    XSCPSGNDP,
    XSCPSGNQP,
    XSIEXPDP,
    XSIEXPQP,
    XVCPSGNDP,
    XVCPSGNSP,
    XVIEXPDP,
    XVIEXPSP,
    XXLAND,
    XXLANDC,
    XXLNAND,
    XXLNOR,
    XXLOR, XXLORf,
    XXLORC
)>;

// 3 Cycles ALU operations, 3 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read, P10FX_Read],
      (instrs
    ADDEX, ADDEX8,
    DST, DST64, DSTT, DSTT64,
    DSTST, DSTST64, DSTSTT, DSTSTT64,
    ISEL, ISEL8,
    RLDCL,
    RLDCR,
    RLDIC,
    RLDICL, RLDICL_32, RLDICL_32_64,
    RLDICR, RLDICR_32,
    VRLDMI,
    VRLWMI,
    VSEL,
    XXSEL
)>;

// 3 Cycles ALU operations, 4 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read, P10FX_Read, P10FX_Read],
      (instrs
    RLDIMI,
    RLWINM, RLWINM8,
    RLWNM, RLWNM8
)>;

// 3 Cycles ALU operations, 5 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read, P10FX_Read, P10FX_Read, P10FX_Read],
      (instrs
    RLWIMI, RLWIMI8
)>;

// Single crack instructions
// 3 Cycles ALU operations, 0 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_DISP_ANY],
      (instrs
    MFFS,
    MFFS_rec,
    MFFSL,
    MFVSCR,
    TRECHKPT
)>;

// Single crack instructions
// 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10FX_Read],
      (instrs
    ADDME8_rec, ADDME_rec,
    ADDME8O_rec, ADDMEO_rec,
    ADDZE8_rec, ADDZE_rec,
    ADDZE8O_rec, ADDZEO_rec,
    MCRFS,
    MFFSCDRN,
    MFFSCDRNI,
    MFFSCRN,
    MFFSCRNI,
    MTFSB0,
    MTVSCR,
    NEG8O_rec, NEGO_rec,
    SUBFME8_rec, SUBFME_rec,
    SUBFME8O_rec, SUBFMEO_rec,
    SUBFZE8_rec, SUBFZE_rec,
    SUBFZE8O_rec, SUBFZEO_rec,
    TABORT,
    TBEGIN,
    TRECLAIM,
    TSR
)>;

// Single crack instructions
// 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10FX_Read, P10FX_Read],
      (instrs
    ADDE8_rec, ADDE_rec,
    ADDE8O_rec, ADDEO_rec,
    ADDIC_rec,
    ADD4O_rec, ADD8O_rec,
    SUBFE8_rec, SUBFE_rec,
    SUBFE8O_rec, SUBFEO_rec,
    SUBF8O_rec, SUBFO_rec
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 0 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    HRFID,
    MFFSCE,
    RFID,
    STOP
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10FX_Read],
      (instrs
    FABSD_rec, FABSS_rec,
    FMR_rec,
    FNABSD_rec, FNABSS_rec,
    FNEGD_rec, FNEGS_rec,
    MTFSB1,
    RFEBB,
    SC
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read],
      (instrs
    ADDC, ADDC8,
    ADDC8_rec, ADDC_rec,
    ADDC8O, ADDCO,
    FCPSGND_rec, FCPSGNS_rec,
    MTFSF, MTFSFb,
    MTFSFI, MTFSFIb,
    SUBFC, SUBFC8,
    SUBFC8_rec, SUBFC_rec,
    SUBFC8O, SUBFCO
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 3 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read, P10FX_Read],
      (instrs
    MTFSFI_rec
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 4 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read, P10FX_Read, P10FX_Read],
      (instrs
    MTFSF_rec
)>;

// 4-way crack instructions
// 3 Cycles ALU operations, 3 Cycles ALU operations, 3 Cycles ALU operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10W_FX_3C, P10W_DISP_ANY, P10W_FX_3C, P10W_DISP_ANY, P10FX_Read, P10FX_Read],
      (instrs
    ADDC8O_rec, ADDCO_rec,
    SUBFC8O_rec, SUBFCO_rec
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 4 Cycles Permute operations, 1 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_EVEN, P10W_PM_4C, P10W_DISP_ANY],
      (instrs
    VSTRIBL_rec,
    VSTRIBR_rec,
    VSTRIHL_rec,
    VSTRIHR_rec
)>;

// 2-way crack instructions
// 3 Cycles ALU operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_FX_3C, P10W_DISP_PAIR, P10W_FX_3C, P10FX_Read, P10FX_Read],
      (instrs
    MTCRF, MTCRF8
)>;

// 6 Cycles Load operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_ANY, P10LD_Read],
      (instrs
    LBZ, LBZ8,
    LD,  LDtoc,  LDtocBA,  LDtocCPT,  LDtocJTI,  LDtocL, SPILLTOVSR_LD,
    LDBRX,
     DFLOADf32, DFLOADf64, LFD,
    LFDX,  XFLOADf32, XFLOADf64,
    LFIWAX, LIWAX,
    LFIWZX, LIWZX,
    LHA, LHA8,
    LHAX, LHAX8,
    LHBRX, LHBRX8,
    LHZ, LHZ8,
    LVEBX,
    LVEHX,
    LVEWX,
    LVX,
    LVXL,
    LWA, LWA_32,
    LWAX, LWAX_32,
    LWBRX, LWBRX8,
    LWZ, LWZ8,  LWZtoc, LWZtocL,
    LXSD,
    LXSDX,
    LXSIBZX,
    LXSIHZX,
    LXSIWAX,
    LXSIWZX,
    LXV,
    LXVB16X,
    LXVD2X,
    LXVDSX,
    LXVH8X,
    LXVRBX,
    LXVRDX,
    LXVRHX,
    LXVRWX,
    LXVW4X,
    LXVWSX,
    LXVX
)>;

// 6 Cycles Load operations, 2 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_ANY, P10LD_Read, P10LD_Read],
      (instrs
    DCBT,
    DCBTST,
    ICBT,
    LBZX, LBZX8, LBZXTLS, LBZXTLS_, LBZXTLS_32,
    LDX, LDXTLS, LDXTLS_, SPILLTOVSR_LDX,
    LHZX, LHZX8, LHZXTLS, LHZXTLS_, LHZXTLS_32,
    LWZX, LWZX8, LWZXTLS, LWZXTLS_, LWZXTLS_32,
    LXVL,
    LXVLL
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 13 Cycles Decimal Floating Point operations, 2 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_DF_13C, P10W_DISP_ANY],
      (instrs
    HASHCHK, HASHCHK8,
    HASHCHKP, HASHCHKP8
)>;

// Single crack instructions
// 6 Cycles Load operations, 0 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_DISP_ANY],
      (instrs
    SLBIA
)>;

// Single crack instructions
// 6 Cycles Load operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_DISP_ANY, P10LD_Read],
      (instrs
    DARN,
    LBARX, LBARXL,
    LDARX, LDARXL,
    LHARX, LHARXL,
    LWARX, LWARXL,
    SLBFEE_rec,
    SLBIE,
    SLBMFEE,
    SLBMFEV
)>;

// Single crack instructions
// 6 Cycles Load operations, 2 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_DISP_ANY, P10LD_Read, P10LD_Read],
      (instrs
    LBZCIX,
    LDCIX,
    LHZCIX,
    LWZCIX,
    MTSPR, MTSPR8, MTSR, MTUDSCR, MTVRSAVE, MTVRSAVEv
)>;

// Expand instructions
// 6 Cycles Load operations, 6 Cycles Load operations, 6 Cycles Load operations, and 6 Cycles Load operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_LD_6C, P10W_DISP_ANY, P10W_LD_6C, P10W_DISP_ANY, P10W_LD_6C, P10W_DISP_ANY, P10LD_Read],
      (instrs
    LMW
)>;

// Expand instructions
// 6 Cycles Load operations, 6 Cycles Load operations, 6 Cycles Load operations, and 6 Cycles Load operations, 2 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_LD_6C, P10W_DISP_ANY, P10W_LD_6C, P10W_DISP_ANY, P10W_LD_6C, P10W_DISP_ANY, P10LD_Read, P10LD_Read],
      (instrs
    LSWI
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 3 Cycles Simple Fixed-point (SFX) operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_EVEN, P10W_SX_3C, P10W_DISP_ANY],
      (instrs
    LBZU, LBZU8,
    LBZUX, LBZUX8,
    LDU,
    LDUX,
    LFDU,
    LFDUX,
    LHAU, LHAU8,
    LHAUX, LHAUX8,
    LHZU, LHZU8,
    LHZUX, LHZUX8,
    LWAUX,
    LWZU, LWZU8,
    LWZUX, LWZUX8
)>;

// 6 Cycles Load operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10LD_Read],
      (instrs
    PLBZ, PLBZ8, PLBZ8pc, PLBZpc,
    PLD, PLDpc,
    PLFD, PLFDpc,
    PLFS, PLFSpc,
    PLHA, PLHA8, PLHA8pc, PLHApc,
    PLHZ, PLHZ8, PLHZ8pc, PLHZpc,
    PLWA, PLWA8, PLWA8pc, PLWApc,
    PLWZ, PLWZ8, PLWZ8pc, PLWZpc,
    PLXSD, PLXSDpc,
    PLXSSP, PLXSSPpc,
    PLXV, PLXVpc,
    PLXVP, PLXVPpc
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 4 Cycles ALU2 operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10W_F2_4C],
      (instrs
    LFS,
    LFSX,
    LXSSP,
    LXSSPX
)>;

// 4-way crack instructions
// 6 Cycles Load operations, 4 Cycles ALU2 operations, 3 Cycles Simple Fixed-point (SFX) operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10W_F2_4C, P10W_SX_3C, P10W_DISP_ANY, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    LFSU,
    LFSUX
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 6 Cycles Load operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10W_LD_6C, P10W_DISP_PAIR, P10LD_Read],
      (instrs
    TLBIEL
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 6 Cycles Load operations, 2 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10W_LD_6C, P10W_DISP_PAIR, P10LD_Read, P10LD_Read],
      (instrs
    SLBMTE
)>;

// 2-way crack instructions
// 6 Cycles Load operations, and 3 Cycles Simple Fixed-point (SFX) operations, 1 input operands
def : InstRW<[P10W_LD_6C, P10W_DISP_PAIR, P10W_SX_3C],
      (instrs
    LXVP,
    LXVPX
)>;

// Single crack instructions
// 13 Cycles Unknown operations, 1 input operands
def : InstRW<[P10W_MFL_13C, P10W_DISP_EVEN, P10W_DISP_ANY],
      (instrs
    MFSPR, MFSPR8, MFSR, MFTB8, MFUDSCR, MFVRSAVE, MFVRSAVEv
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 0 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_ANY],
      (instrs
    XXSETACCZ
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 2 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_ANY, P10MM_Read, P10MM_Read],
      (instrs
    XVBF16GER2,
    XVF16GER2,
    XVF32GER,
    XVF64GER,
    XVI16GER2,
    XVI16GER2S,
    XVI4GER8,
    XVI8GER4
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 3 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_ANY, P10MM_Read, P10MM_Read, P10MM_Read],
      (instrs
    XVBF16GER2NN,
    XVBF16GER2NP,
    XVBF16GER2PN,
    XVBF16GER2PP,
    XVF16GER2NN,
    XVF16GER2NP,
    XVF16GER2PN,
    XVF16GER2PP,
    XVF32GERNN,
    XVF32GERNP,
    XVF32GERPN,
    XVF32GERPP,
    XVF64GERNN,
    XVF64GERNP,
    XVF64GERPN,
    XVF64GERPP,
    XVI16GER2PP,
    XVI16GER2SPP,
    XVI4GER8PP,
    XVI8GER4PP,
    XVI8GER4SPP
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 4 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_PAIR, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read],
      (instrs
    PMXVF32GER,
    PMXVF64GER
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 5 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_PAIR, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read],
      (instrs
    PMXVBF16GER2,
    PMXVF16GER2,
    PMXVF32GERNN,
    PMXVF32GERNP,
    PMXVF32GERPN,
    PMXVF32GERPP,
    PMXVF64GERNN,
    PMXVF64GERNP,
    PMXVF64GERPN,
    PMXVF64GERPP,
    PMXVI16GER2,
    PMXVI16GER2S,
    PMXVI4GER8,
    PMXVI8GER4
)>;

// 10 Cycles SIMD Matrix Multiply Engine operations, 6 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_PAIR, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read, P10MM_Read],
      (instrs
    PMXVBF16GER2NN,
    PMXVBF16GER2NP,
    PMXVBF16GER2PN,
    PMXVBF16GER2PP,
    PMXVF16GER2NN,
    PMXVF16GER2NP,
    PMXVF16GER2PN,
    PMXVF16GER2PP,
    PMXVI16GER2PP,
    PMXVI16GER2SPP,
    PMXVI4GER8PP,
    PMXVI8GER4PP,
    PMXVI8GER4SPP
)>;

// 2-way crack instructions
// 10 Cycles SIMD Matrix Multiply Engine operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_PAIR, P10W_FX_3C],
      (instrs
    XXMTACC
)>;

// 4-way crack instructions
// 10 Cycles SIMD Matrix Multiply Engine operations, 3 Cycles ALU operations, 10 Cycles SIMD Matrix Multiply Engine operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_MM_10C, P10W_DISP_PAIR, P10W_FX_3C, P10W_MM_10C, P10W_DISP_PAIR, P10W_FX_3C],
      (instrs
    XXMFACC
)>;

// 5 Cycles GPR Multiply operations, 2 input operands
def : InstRW<[P10W_MU_5C, P10W_DISP_ANY, P10MU_Read, P10MU_Read],
      (instrs
    MULHD,
    MULHDU,
    MULHW,
    MULHWU,
    MULLD,
    MULLDO,
    MULLI, MULLI8,
    MULLW,
    MULLWO,
    VMULHSD,
    VMULHUD,
    VMULLD
)>;

// 5 Cycles GPR Multiply operations, 3 input operands
def : InstRW<[P10W_MU_5C, P10W_DISP_ANY, P10MU_Read, P10MU_Read, P10MU_Read],
      (instrs
    MADDHD,
    MADDHDU,
    MADDLD, MADDLD8
)>;

// 2-way crack instructions
// 5 Cycles GPR Multiply operations, and 3 Cycles ALU operations, 2 input operands
def : InstRW<[P10W_MU_5C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    MULHD_rec,
    MULHDU_rec,
    MULHW_rec,
    MULHWU_rec,
    MULLD_rec,
    MULLDO_rec,
    MULLW_rec,
    MULLWO_rec
)>;

// 4 Cycles Permute operations, 0 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_ANY],
      (instrs
    VSPLTISW, V_SETALLONES, V_SETALLONESB, V_SETALLONESH
)>;

// 4 Cycles Permute operations, 1 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_ANY, P10PM_Read],
      (instrs
    LVSL,
    LVSR,
    LXVKQ,
    MFVSRLD,
    MTVSRWS,
    VCLZLSBB,
    VCTZLSBB,
    VGBBD,
    VPRTYBQ,
    VSPLTISB,
    VSPLTISH,
    VSTRIBL,
    VSTRIBR,
    VSTRIHL,
    VSTRIHR,
    VUPKHPX,
    VUPKHSB,
    VUPKHSH,
    VUPKHSW,
    VUPKLPX,
    VUPKLSB,
    VUPKLSH,
    VUPKLSW,
    XVCVBF16SPN,
    XXBRD,
    XXBRH,
    XXBRQ,
    XXBRW,
    XXSPLTIB
)>;

// 4 Cycles Permute operations, 2 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_ANY, P10PM_Read, P10PM_Read],
      (instrs
    BPERMD,
    MTVSRDD,
    VBPERMD,
    VBPERMQ,
    VCLRLB,
    VCLRRB,
    VEXTRACTD,
    VEXTRACTUB,
    VEXTRACTUH,
    VEXTRACTUW,
    VEXTUBLX,
    VEXTUBRX,
    VEXTUHLX,
    VEXTUHRX,
    VEXTUWLX,
    VEXTUWRX,
    VINSERTD,
    VINSERTW,
    VMRGHB,
    VMRGHH,
    VMRGHW,
    VMRGLB,
    VMRGLH,
    VMRGLW,
    VPKPX,
    VPKSDSS,
    VPKSDUS,
    VPKSHSS,
    VPKSHUS,
    VPKSWSS,
    VPKSWUS,
    VPKUDUM,
    VPKUDUS,
    VPKUHUM,
    VPKUHUS,
    VPKUWUM,
    VPKUWUS,
    VSL,
    VSLO,
    VSLV,
    VSPLTB, VSPLTBs,
    VSPLTH, VSPLTHs,
    VSPLTW,
    VSR,
    VSRO,
    VSRV,
    XXEXTRACTUW,
    XXGENPCVDM,
    XXGENPCVHM,
    XXGENPCVWM,
    XXMRGHW,
    XXMRGLW,
    XXPERM,
    XXPERMDI, XXPERMDIs,
    XXPERMR,
    XXSLDWI, XXSLDWIs,
    XXSPLTW, XXSPLTWs
)>;

// 4 Cycles Permute operations, 3 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_ANY, P10PM_Read, P10PM_Read, P10PM_Read],
      (instrs
    VEXTDDVLX,
    VEXTDDVRX,
    VEXTDUBVLX,
    VEXTDUBVRX,
    VEXTDUHVLX,
    VEXTDUHVRX,
    VEXTDUWVLX,
    VEXTDUWVRX,
    VINSBLX,
    VINSBRX,
    VINSBVLX,
    VINSBVRX,
    VINSD,
    VINSDLX,
    VINSDRX,
    VINSERTB,
    VINSERTH,
    VINSHLX,
    VINSHRX,
    VINSHVLX,
    VINSHVRX,
    VINSW,
    VINSWLX,
    VINSWRX,
    VINSWVLX,
    VINSWVRX,
    VPERM,
    VPERMR,
    VPERMXOR,
    VSLDBI,
    VSLDOI,
    VSRDBI,
    XXINSERTW
)>;

// 2-way crack instructions
// 4 Cycles Permute operations, and 7 Cycles VMX Multiply operations, 2 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_EVEN, P10W_vMU_7C, P10W_DISP_ANY],
      (instrs
    VSUMSWS
)>;

// 4 Cycles Permute operations, 1 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_PAIR, P10PM_Read],
      (instrs
    XXSPLTIDP,
    XXSPLTIW
)>;

// 4 Cycles Permute operations, 3 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_PAIR, P10PM_Read, P10PM_Read, P10PM_Read],
      (instrs
    XXBLENDVB,
    XXBLENDVD,
    XXBLENDVH,
    XXBLENDVW,
    XXSPLTI32DX
)>;

// 4 Cycles Permute operations, 4 input operands
def : InstRW<[P10W_PM_4C, P10W_DISP_PAIR, P10PM_Read, P10PM_Read, P10PM_Read, P10PM_Read],
      (instrs
    XXEVAL,
    XXPERMX
)>;

// 3 Cycles Store operations, 1 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_ANY, P10ST_Read],
      (instrs
    DCBST,
    DCBZ,
    ICBI
)>;

// 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_ANY, P10ST_Read, P10ST_Read],
      (instrs
    DCBF,
    PSTXVP, PSTXVPpc,
    STB, STB8,
    STBU, STBU8,
    STBUX, STBUX8,
    SPILLTOVSR_ST, STD,
    STDBRX,
    STDU,
    STDUX,
     DFSTOREf32, DFSTOREf64, STFD,
    STFDU,
    STFDUX,
    STFDX,
    STFIWX, STIWX,
    STFS,
    STFSU,
    STFSUX,
    STFSX,
    STH, STH8,
    STHBRX,
    STHU, STHU8,
    STHUX, STHUX8,
    STVEBX,
    STVEHX,
    STVEWX,
    STVX,
    STVXL,
    STW, STW8,
    STWBRX,
    STWU, STWU8,
    STWUX, STWUX8,
    STXSD,
    STXSDX,
    STXSIBX, STXSIBXv,
    STXSIHX, STXSIHXv,
    STXSIWX,
    STXSSP,
    STXSSPX,
    STXV,
    STXVB16X,
    STXVD2X,
    STXVH8X,
    STXVRBX,
    STXVRDX,
    STXVRHX,
    STXVRWX,
    STXVW4X,
    STXVX
)>;

// 3 Cycles Store operations, 3 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_ANY, P10ST_Read, P10ST_Read, P10ST_Read],
      (instrs
    CP_COPY, CP_COPY8,
    STBX, STBX8, STBXTLS, STBXTLS_, STBXTLS_32,
    SPILLTOVSR_STX, STDX, STDXTLS, STDXTLS_,
    STHX, STHX8, STHXTLS, STHXTLS_, STHXTLS_32,
    STWX, STWX8, STWXTLS, STWXTLS_, STWXTLS_32,
    STXVL,
    STXVLL
)>;

// Single crack instructions
// 3 Cycles Store operations, 0 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_DISP_ANY],
      (instrs
    EnforceIEIO,
    MSGSYNC,
    SLBSYNC,
    TCHECK,
    TLBSYNC
)>;

// Single crack instructions
// 3 Cycles Store operations, 1 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10ST_Read],
      (instrs
    TEND
)>;

// Single crack instructions
// 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10ST_Read, P10ST_Read],
      (instrs
    SLBIEG,
    STBCX,
    STDCX,
    STHCX,
    STWCX,
    TLBIE
)>;

// Single crack instructions
// 3 Cycles Store operations, 3 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10ST_Read, P10ST_Read, P10ST_Read],
      (instrs
    CP_PASTE8_rec, CP_PASTE_rec,
    STBCIX,
    STDCIX,
    STHCIX,
    STWCIX
)>;

// 2-way crack instructions
// 3 Cycles Store operations, and 3 Cycles ALU operations, 0 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    ISYNC
)>;

// 2-way crack instructions
// 3 Cycles Store operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    SYNC
)>;

// Expand instructions
// 3 Cycles Store operations, 3 Cycles ALU operations, 3 Cycles Store operations, 3 Cycles ALU operations, 3 Cycles Store operations, 3 Cycles ALU operations, 6 Cycles Load operations, and 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10W_FX_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10W_FX_3C, P10W_DISP_ANY, P10W_LD_6C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY],
      (instrs
    LDAT,
    LWAT
)>;

// 4-way crack instructions
// 3 Cycles Store operations, 3 Cycles ALU operations, 3 Cycles Store operations, and 3 Cycles Store operations, 3 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY],
      (instrs
    STDAT,
    STWAT
)>;

// Expand instructions
// 3 Cycles Store operations, 3 Cycles Store operations, 3 Cycles Store operations, and 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_ST_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10ST_Read, P10ST_Read],
      (instrs
    STMW
)>;

// Expand instructions
// 3 Cycles Store operations, 3 Cycles Store operations, 3 Cycles Store operations, and 3 Cycles Store operations, 3 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_EVEN, P10W_ST_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10W_ST_3C, P10W_DISP_ANY, P10ST_Read, P10ST_Read, P10ST_Read],
      (instrs
    STSWI
)>;

// 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_PAIR, P10ST_Read, P10ST_Read],
      (instrs
    PSTB, PSTB8, PSTB8pc, PSTBpc,
    PSTD, PSTDpc,
    PSTFD, PSTFDpc,
    PSTFS, PSTFSpc,
    PSTH, PSTH8, PSTH8pc, PSTHpc,
    PSTW, PSTW8, PSTW8pc, PSTWpc,
    PSTXSD, PSTXSDpc,
    PSTXSSP, PSTXSSPpc,
    PSTXV, PSTXVpc
)>;

// 2-way crack instructions
// 3 Cycles Store operations, and 3 Cycles Store operations, 2 input operands
def : InstRW<[P10W_ST_3C, P10W_DISP_PAIR, P10W_ST_3C, P10ST_Read, P10ST_Read],
      (instrs
    STXVP,
    STXVPX
)>;

// FIXME - Miss scheduling information from datasheet
// Temporary set it as 1 Cycles Simple Fixed-point (SFX) operations, 0 input operands
def : InstRW<[P10W_SX, P10W_DISP_ANY],
      (instrs
    ATTN,
    CP_ABORT,
    DCBA,
    DCBI,
    DCBZL,
    DCCCI,
    ICBLC,
    ICBLQ,
    ICBTLS,
    ICCCI,
    LA, LA8,
    MFDCR,
    MFPMR,
    MFSRIN,
    MSYNC,
    MTDCR,
    MTPMR,
    MTSRIN,
    NAP,
    TLBIA,
    TLBLD,
    TLBLI,
    TLBRE2,
    TLBSX2,
    TLBSX2D,
    TLBWE2
)>;

// Single crack instructions
// 3 Cycles Simple Fixed-point (SFX) operations, 0 input operands
def : InstRW<[P10W_SX_3C, P10W_DISP_EVEN, P10W_DISP_ANY],
      (instrs
    CLRBHRB,
    MFMSR
)>;

// Single crack instructions
// 3 Cycles Simple Fixed-point (SFX) operations, 1 input operands
def : InstRW<[P10W_SX_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10SX_Read],
      (instrs
    MFTB
)>;

// Single crack instructions
// 3 Cycles Simple Fixed-point (SFX) operations, 2 input operands
def : InstRW<[P10W_SX_3C, P10W_DISP_EVEN, P10W_DISP_ANY, P10SX_Read, P10SX_Read],
      (instrs
    MFBHRBE,
    MTMSR,
    MTMSRD
)>;

// 2-way crack instructions
// 3 Cycles Simple Fixed-point (SFX) operations, and 3 Cycles ALU operations, 1 input operands
def : InstRW<[P10W_SX_3C, P10W_DISP_EVEN, P10W_FX_3C, P10W_DISP_ANY],
      (instrs
    ADDPCIS
)>;

// 3 Cycles Simple Fixed-point (SFX) operations, 1 input operands
def : InstRW<[P10W_SX_3C, P10W_DISP_PAIR, P10SX_Read],
      (instrs
    PADDI, PADDI8, PADDI8pc, PADDIpc, PLI, PLI8
)>;

// 7 Cycles VMX Multiply operations, 2 input operands
def : InstRW<[P10W_vMU_7C, P10W_DISP_ANY, P10vMU_Read, P10vMU_Read],
      (instrs
    VMULESB,
    VMULESD,
    VMULESH,
    VMULESW,
    VMULEUB,
    VMULEUD,
    VMULEUH,
    VMULEUW,
    VMULHSW,
    VMULHUW,
    VMULOSB,
    VMULOSD,
    VMULOSH,
    VMULOSW,
    VMULOUB,
    VMULOUD,
    VMULOUH,
    VMULOUW,
    VMULUWM,
    VSUM2SWS,
    VSUM4SBS,
    VSUM4SHS,
    VSUM4UBS
)>;

// 7 Cycles VMX Multiply operations, 3 input operands
def : InstRW<[P10W_vMU_7C, P10W_DISP_ANY, P10vMU_Read, P10vMU_Read, P10vMU_Read],
      (instrs
    VMHADDSHS,
    VMHRADDSHS,
    VMLADDUHM,
    VMSUMCUD,
    VMSUMMBM,
    VMSUMSHM,
    VMSUMSHS,
    VMSUMUBM,
    VMSUMUDM,
    VMSUMUHM,
    VMSUMUHS
)>;

