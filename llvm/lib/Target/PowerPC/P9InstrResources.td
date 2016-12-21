//===- P9InstrResources.td - P9 Instruction Resource Defs  -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines resources required by some of P9 instruction. This is part
// P9 processor model used for instruction scheduling. Not every instruction
// is listed here. Instructions in this file belong to itinerary classes that
// have instructions with different resource requirements.
//
//===----------------------------------------------------------------------===//


def : InstRW<[P9_ALUE_2C, P9_ALUO_2C, IP_EXECE_1C, IP_EXECO_1C,
              DISP_1C, DISP_1C],
      (instrs
    VADDCUW,
    VADDUBM,
    VADDUDM,
    VADDUHM,
    VADDUWM,
    VAND,
    VANDC,
    VCMPEQUB,
    VCMPEQUBo,
    VCMPEQUD,
    VCMPEQUDo,
    VCMPEQUH,
    VCMPEQUHo,
    VCMPEQUW,
    VCMPEQUWo,
    VCMPGTSB,
    VCMPGTSBo,
    VCMPGTSD,
    VCMPGTSDo,
    VCMPGTSH,
    VCMPGTSHo,
    VCMPGTSW,
    VCMPGTSWo,
    VCMPGTUB,
    VCMPGTUBo,
    VCMPGTUD,
    VCMPGTUDo,
    VCMPGTUH,
    VCMPGTUHo,
    VCMPGTUW,
    VCMPGTUWo,
    VCMPNEB,
    VCMPNEBo,
    VCMPNEH,
    VCMPNEHo,
    VCMPNEW,
    VCMPNEWo,
    VCMPNEZB,
    VCMPNEZBo,
    VCMPNEZH,
    VCMPNEZHo,
    VCMPNEZW,
    VCMPNEZWo,
    VEQV,
    VEXTSB2D,
    VEXTSB2W,
    VEXTSH2D,
    VEXTSH2W,
    VEXTSW2D,
    VMRGEW,
    VMRGOW,
    VNAND,
    VNEGD,
    VNEGW,
    VNOR,
    VOR,
    VORC,
    VPOPCNTB,
    VPOPCNTH,
    VPOPCNTW,
    VSEL,
    VSUBCUW,
    VSUBUBM,
    VSUBUDM,
    VSUBUHM,
    VSUBUWM,
    VXOR,
    V_SET0B,
    V_SET0H,
    V_SET0,
    XVABSDP,
    XVABSSP,
    XVCPSGNDP,
    XVCPSGNSP,
    XVIEXPDP,
    XVNABSDP,
    XVNABSSP,
    XVNEGDP,
    XVNEGSP,
    XVXEXPDP,
    XXLAND,
    XXLANDC,
    XXLEQV,
    XXLNAND,
    XXLNOR,
    XXLOR,
    XXLORf,
    XXLORC,
    XXLXOR,
    XXSEL
)>;

def : InstRW<[P9_ALU_2C, IP_EXEC_1C, DISP_1C, DISP_1C],
      (instrs
    XSABSQP,
    XSCPSGNQP,
    XSIEXPQP,
    XSNABSQP,
    XSNEGQP,
    XSXEXPQP,
    XSABSDP,
    XSCPSGNDP,
    XSIEXPDP,
    XSNABSDP,
    XSNEGDP,
    XSXEXPDP
)>;

def : InstRW<[P9_ALUE_3C, P9_ALUO_3C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs

    VMINSB,
    VMINSD,
    VMINSH,
    VMINSW,
    VMINUB,
    VMINUD,
    VMINUH,
    VMINUW,
    VPOPCNTD,
    VPRTYBD,
    VPRTYBW,
    VRLB,
    VRLD,
    VRLDMI,
    VRLDNM,
    VRLH,
    VRLW,
    VRLWMI,
    VRLWNM,
    VSHASIGMAD,
    VSHASIGMAW,
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
    XSCVSPDPN,
    XSMAXCDP,
    XSMAXDP,
    XSMAXJDP,
    XSMINCDP,
    XSMINDP,
    XSMINJDP,
    XSTDIVDP,
    XSTSQRTDP,
    XSTSTDCDP,
    XSTSTDCSP,
    XSXSIGDP,
    XVCMPEQDP,
    XVCMPEQDPo,
    XVCMPEQSP,
    XVCMPEQSPo,
    XVCMPGEDP,
    XVCMPGEDPo,
    XVCMPGESP,
    XVCMPGESPo,
    XVCMPGTDP,
    XVCMPGTDPo,
    XVCMPGTSP,
    XVCMPGTSPo,
    XVIEXPSP,
    XVMAXDP,
    XVMAXSP,
    XVMINDP,
    XVMINSP,
    XVTDIVDP,
    XVTDIVSP,
    XVTSQRTDP,
    XVTSQRTSP,
    XVTSTDCDP,
    XVTSTDCSP,
    XVXEXPSP,
    XVXSIGDP,
    XVXSIGSP
)>;

def : InstRW<[P9_ALUE_4C, P9_ALUO_4C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    VABSDUB,
    VABSDUH,
    VABSDUW,
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
    VBPERMD,
    VCLZB,
    VCLZD,
    VCLZH,
    VCLZW,
    VCMPBFP,
    VCMPBFPo,
    VCMPGTFP,
    VCMPGTFPo,
    VCTZB,
    VCTZD,
    VCTZH,
    VCTZW,
    VMAXFP,
    VMAXSB,
    VMAXSD,
    VMAXSH,
    VMAXSW,
    VMAXUB,
    VMAXUD,
    VMAXUH,
    VMAXUW,
    VMINFP,
    VCMPEQFP,
    VCMPEQFPo,
    VCMPGEFP,
    VCMPGEFPo
)>;

def : InstRW<[P9_DPE_7C, P9_DPO_7C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    VADDFP,
    VCTSXS,
    VCTSXS_0,
    VCTUXS,
    VCTUXS_0,
    VEXPTEFP,
    VLOGEFP,
    VMADDFP,
    VMHADDSHS,
    VNMSUBFP,
    VREFP,
    VRFIM,
    VRFIN,
    VRFIP,
    VRFIZ,
    VRSQRTEFP,
    VSUBFP,
    XVADDDP,
    XVADDSP,
    XVCVDPSP,
    XVCVDPSXDS,
    XVCVDPSXWS,
    XVCVDPUXDS,
    XVCVDPUXWS,
    XVCVHPSP,
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
    XVMADDADP,
    XVMADDASP,
    XVMADDMDP,
    XVMADDMSP,
    XVMSUBADP,
    XVMSUBASP,
    XVMSUBMDP,
    XVMSUBMSP,
    XVMULDP,
    XVMULSP,
    XVNMADDADP,
    XVNMADDASP,
    XVNMADDMDP,
    XVNMADDMSP,
    XVNMSUBADP,
    XVNMSUBASP,
    XVNMSUBMDP,
    XVNMSUBMSP,
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
    XVRSQRTESP,
    XVSUBDP,
    XVSUBSP,
    VCFSX,
    VCFSX_0,
    VCFUX,
    VCFUX_0,
    VMHRADDSHS,
    VMLADDUHM,
    VMSUMMBM,
    VMSUMSHM,
    VMSUMSHS,
    VMSUMUBM,
    VMSUMUHM,
    VMSUMUHS,
    VMULESB,
    VMULESH,
    VMULESW,
    VMULEUB,
    VMULEUH,
    VMULEUW,
    VMULOSB,
    VMULOSH,
    VMULOSW,
    VMULOUB,
    VMULOUH,
    VMULOUW,
    VMULUWM,
    VSUM2SWS,
    VSUM4SBS,
    VSUM4SHS,
    VSUM4UBS,
    VSUMSWS
)>;

def : InstRW<[P9_DP_7C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    XSMADDADP,
    XSMADDASP,
    XSMADDMDP,
    XSMADDMSP,
    XSMSUBADP,
    XSMSUBASP,
    XSMSUBMDP,
    XSMSUBMSP,
    XSMULDP,
    XSMULSP,
    XSNMADDADP,
    XSNMADDASP,
    XSNMADDMDP,
    XSNMADDMSP,
    XSNMSUBADP,
    XSNMSUBASP,
    XSNMSUBMDP,
    XSNMSUBMSP
)>;


def : InstRW<[P9_DP_7C, IP_EXEC_1C, DISP_1C, DISP_1C],
      (instrs
    XSADDDP,
    XSADDSP,
    XSCVDPHP,
    XSCVDPSP,
    XSCVDPSXDS,
    XSCVDPSXWS,
    XSCVDPUXDS,
    XSCVDPUXWS,
    XSCVHPDP,
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
    //XSRSP,
    XSRSQRTEDP,
    XSRSQRTESP,
    XSSUBDP,
    XSSUBSP,
    XSCVDPSPN
)>;

def : InstRW<[P9_PM_3C, IP_EXECO_1C, IP_EXECE_1C, DISP_1C, DISP_1C],
      (instrs
    VBPERMQ,
    VCLZLSBB,
    VCTZLSBB,
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
    VGBBD,
    VINSERTB,
    VINSERTD,
    VINSERTH,
    VINSERTW,
    VMRGHB,
    VMRGHH,
    VMRGHW,
    VMRGLB,
    VMRGLH,
    VMRGLW,
    VPERM,
    VPERMR,
    VPERMXOR,
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
    VPRTYBQ,
    VSL,
    VSLDOI,
    VSLO,
    VSLV,
    VSPLTB,
    VSPLTH,
    VSPLTISB,
    VSPLTISH,
    VSPLTISW,
    VSPLTW,
    VSR,
    VSRO,
    VSRV,
    VUPKHPX,
    VUPKHSB,
    VUPKHSH,
    VUPKHSW,
    VUPKLPX,
    VUPKLSB,
    VUPKLSH,
    VUPKLSW,
    XXBRD,
    XXBRH,
    XXBRQ,
    XXBRW,
    XXEXTRACTUW,
    XXINSERTW,
    XXMRGHW,
    XXMRGLW,
    XXPERM,
    XXPERMR,
    XXSLDWI,
    XXSPLTIB,
    XXSPLTW,
    VADDCUQ,
    VADDECUQ,
    VADDEUQM,
    VADDUQM,
    VMUL10CUQ,
    VMUL10ECUQ,
    VMUL10EUQ,
    VMUL10UQ,
    VSUBCUQ,
    VSUBECUQ,
    VSUBEUQM,
    VSUBUQM,
    XSCMPEXPQP,
    XSCMPOQP,
    XSCMPUQP,
    XSTSTDCQP,
    XSXSIGQP
)>;

def : InstRW<[P9_DFU_12C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XSADDQP,
    XSADDQPO,
    XSCVDPQP,
    XSCVQPDP,
    XSCVQPDPO,
    XSCVQPSDZ,
    XSCVQPSWZ,
    XSCVQPUDZ,
    XSCVQPUWZ,
    XSCVSDQP,
    XSCVUDQP,
    XSRQPI,
    XSRQPXP,
    XSSUBQP,
    XSSUBQPO
)>;

def : InstRW<[P9_DFU_24C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XSMADDQP,
    XSMADDQPO,
    XSMSUBQP,
    XSMSUBQPO,
    XSMULQP,
    XSMULQPO,
    XSNMADDQP,
    XSNMADDQPO,
    XSNMSUBQP,
    XSNMSUBQPO
)>;

def : InstRW<[P9_DFU_58C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XSDIVQP,
    XSDIVQPO
)>;

def : InstRW<[P9_DFU_76C, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XSSQRTQP,
    XSSQRTQPO
)>;

// Load Operation in IIC_LdStLFD

def : InstRW<[P9_LS_5C, IP_AGEN_1C, DISP_1C, DISP_1C],
      (instrs
    LXSDX,
    LXVD2X,
    LXSIWZX,
    LXV,
    LXSD
)>;

def : InstRW<[P9_LS_5C, IP_AGEN_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    LFIWZX,
    LFDX,
    LFD
)>;

def : InstRW<[P9_LoadAndALUOp_7C, IP_AGEN_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    LXSSPX,
    LXSIWAX,
    LXSSP
)>;

def : InstRW<[P9_LoadAndALUOp_7C, IP_AGEN_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    LFIWAX,
    LFSX,
    LFS
)>;

def : InstRW<[P9_LoadAndPMOp_8C, IP_AGEN_1C, IP_EXEC_1C, DISP_1C, DISP_1C],
      (instrs
    LXVDSX,
    LXVW4X
)>;

// Store Operations in IIC_LdStSTFD.

def : InstRW<[P9_LS_1C, IP_EXEC_1C, IP_AGEN_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    STFS,
    STFD,
    STFIWX,
    STFSX,
    STFDX,
    STXSDX,
    STXSSPX,
    STXSIWX
)>;

def : InstRW<[P9_LS_1C, IP_EXEC_1C, IP_EXEC_1C, IP_AGEN_1C, DISP_1C, DISP_1C],
      (instrs
    STXVD2X,
    STXVW4X
)>;


// Divide Operations in IIC_IntDivW, IIC_IntDivD.

def : InstRW<[P9_DIV_16C_8, IP_EXECE_1C, DISP_1C, DISP_1C],
      (instrs
    DIVW,
    DIVWU
)>;

def : InstRW<[P9_DIV_24C_8, IP_EXECE_1C, DISP_1C, DISP_1C],
      (instrs
    DIVWE,
    DIVD,
    DIVWEU,
    DIVDU
)>;

def : InstRW<[P9_DIV_40C_8, IP_EXECE_1C, DISP_1C, DISP_1C],
      (instrs
    DIVDE,
    DIVDEU
)>;

def : InstRW<[P9_IntDivAndALUOp_26C_8, IP_EXECE_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    DIVWEo,
    DIVWEUo
)>;

def : InstRW<[P9_IntDivAndALUOp_42C_8, IP_EXECE_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    DIVDEo,
    DIVDEUo
)>;

// Rotate Operations in IIC_IntRotateD, IIC_IntRotateDI
def : InstRW<[P9_ALU_2C, IP_EXEC_1C, DISP_1C, DISP_1C],
      (instrs
    SLD,
    SRD,
    SRAD,
    SRADI,
    RLDIC
)>;

def : InstRW<[P9_ALU_2C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    RLDCL,
    RLDCR,
    RLDIMI,
    RLDICL,
    RLDICR,
    RLDICL_32_64
)>;

// CR access instructions in _BrMCR, IIC_BrMCRX.

def : InstRW<[P9_ALU_2C, P9_ALU_2C, IP_EXEC_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    MTOCRF,
    MTOCRF8,
    MTCRF,
    MTCRF8
)>;

def : InstRW<[P9_ALU_5C, IP_EXEC_1C, DISP_1C, DISP_1C],
      (instrs
    MCRF,
    MCRXRX
)>;

def : InstRW<[P9_ALU_5C, P9_ALU_5C, IP_EXEC_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    MCRFS
)>;

// FP Div instructions in IIC_FPDivD and IIC_FPDivS.

def : InstRW<[P9_DP_33C_8, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    FDIV,
    XSDIVDP
)>;

def : InstRW<[P9_DP_22C_5, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    FDIVS,
    XSDIVSP
)>;

def : InstRW<[P9_DP_24C_8, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XVDIVSP
)>;

def : InstRW<[P9_DP_33C_8, IP_EXECE_1C, IP_EXECO_1C, DISP_1C, DISP_1C],
      (instrs
    XVDIVDP
)>;

// FP Instructions in IIC_FPGeneral, IIC_FPFused

def : InstRW<[P9_DP_7C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    FRSP,
    FRIND,
    FRINS,
    FRIPD,
    FRIPS,
    FRIZD,
    FRIZS,
    FRIMD,
    FRIMS,
    FRE,
    FRES,
    FRSQRTE,
    FRSQRTES,
    FMADDS,
    FMADD,
    FMSUBS,
    FMSUB,
    FNMADDS,
    FNMADD,
    FNMSUBS,
    FNMSUB,
    FSELD,
    FSELS,
    FADDS,
    FMULS,
    FMUL,
    FSUBS,
    FCFID,
    FCTID,
    FCTIDZ,
    FCFIDU,
    FCFIDS,
    FCFIDUS,
    FCTIDUZ,
    FCTIWUZ,
    FCTIW,
    FCTIWZ
)>;

def : InstRW<[P9_DP_7C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    FMR,
    FABSD,
    FABSS,
    FNABSD,
    FNABSS,
    FNEGD,
    FNEGS,
    FCPSGND,
    FCPSGNS
)>;

def : InstRW<[P9_ALU_3C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    FCMPUS,
    FCMPUD
)>;

// Load instructions in IIC_LdStLFDU and IIC_LdStLFDUX.

def : InstRW<[P9_LoadAndALUOp_7C, P9_ALU_2C,
              IP_AGEN_1C, IP_EXEC_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    LFSU,
    LFSUX
)>;

def : InstRW<[P9_LS_5C, P9_ALU_2C, IP_AGEN_1C, IP_EXEC_1C,
              DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
      (instrs
    LFDU,
    LFDUX
)>;

