//===- PPCSchedule.td - PowerPC Scheduling Definitions -----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across PowerPC chips sets
//
def BPU    : FuncUnit; // Branch unit
def SLU    : FuncUnit; // Store/load unit
def SRU    : FuncUnit; // special register unit
def IU1    : FuncUnit; // integer unit 1 (simple)
def IU2    : FuncUnit; // integer unit 2 (complex)
def IU3    : FuncUnit; // integer unit 3 (7450 simple)
def IU4    : FuncUnit; // integer unit 4 (7450 simple)
def FPU1   : FuncUnit; // floating point unit 1
def FPU2   : FuncUnit; // floating point unit 2
def VPU    : FuncUnit; // vector permutation unit
def VIU1   : FuncUnit; // vector integer unit 1 (simple)
def VIU2   : FuncUnit; // vector integer unit 2 (complex)
def VFPU   : FuncUnit; // vector floating point unit


//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for PowerPC
//
def IntGeneral   : InstrItinClass;
def IntCompare   : InstrItinClass;
def IntDivD      : InstrItinClass;
def IntDivW      : InstrItinClass;
def IntMFFS      : InstrItinClass;
def IntMFVSCR    : InstrItinClass;
def IntMTFSB0    : InstrItinClass;
def IntMTSRD     : InstrItinClass;
def IntMulHD     : InstrItinClass;
def IntMulHW     : InstrItinClass;
def IntMulHWU    : InstrItinClass;
def IntMulLI     : InstrItinClass;
def IntRFID      : InstrItinClass;
def IntRotateD   : InstrItinClass;
def IntRotate    : InstrItinClass;
def IntShift     : InstrItinClass;
def IntTrapD     : InstrItinClass;
def IntTrapW     : InstrItinClass;
def BrB          : InstrItinClass;
def BrCR         : InstrItinClass;
def BrMCR        : InstrItinClass;
def BrMCRX       : InstrItinClass;
def LdStDCBA     : InstrItinClass;
def LdStDCBF     : InstrItinClass;
def LdStDCBI     : InstrItinClass;
def LdStGeneral  : InstrItinClass;
def LdStDSS      : InstrItinClass;
def LdStICBI     : InstrItinClass;
def LdStUX       : InstrItinClass;
def LdStLD       : InstrItinClass;
def LdStLDARX    : InstrItinClass;
def LdStLFD      : InstrItinClass;
def LdStLFDU     : InstrItinClass;
def LdStLHA      : InstrItinClass;
def LdStLMW      : InstrItinClass;
def LdStLVecX    : InstrItinClass;
def LdStLWA      : InstrItinClass;
def LdStLWARX    : InstrItinClass;
def LdStSLBIA    : InstrItinClass;
def LdStSLBIE    : InstrItinClass;
def LdStSTD      : InstrItinClass;
def LdStSTDCX    : InstrItinClass;
def LdStSTVEBX   : InstrItinClass;
def LdStSTWCX    : InstrItinClass;
def LdStSync     : InstrItinClass;
def SprISYNC     : InstrItinClass;
def SprMFSR      : InstrItinClass;
def SprMTMSR     : InstrItinClass;
def SprMTSR      : InstrItinClass;
def SprTLBSYNC   : InstrItinClass;
def SprMFCR      : InstrItinClass;
def SprMFMSR     : InstrItinClass;
def SprMFSPR     : InstrItinClass;
def SprMFTB      : InstrItinClass;
def SprMTSPR     : InstrItinClass;
def SprMTSRIN    : InstrItinClass;
def SprRFI       : InstrItinClass;
def SprSC        : InstrItinClass;
def FPGeneral    : InstrItinClass;
def FPCompare    : InstrItinClass;
def FPDivD       : InstrItinClass;
def FPDivS       : InstrItinClass;
def FPFused      : InstrItinClass;
def FPRes        : InstrItinClass;
def FPSqrt       : InstrItinClass;
def VecGeneral   : InstrItinClass;
def VecFP        : InstrItinClass;
def VecFPCompare : InstrItinClass;
def VecComplex   : InstrItinClass;
def VecPerm      : InstrItinClass;
def VecFPRound   : InstrItinClass;
def VecVSL       : InstrItinClass;
def VecVSR       : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

include "PPCScheduleG3.td"
include "PPCScheduleG4.td"
include "PPCScheduleG4Plus.td"
include "PPCScheduleG5.td"

//===----------------------------------------------------------------------===//
// Instruction to itinerary class map - When add new opcodes to the supported
// set, refer to the following table to determine which itinerary class the
// opcode belongs.
//
//    opcode     itinerary class
//    ======     ===============
//    add        IntGeneral
//    addc       IntGeneral
//    adde       IntGeneral
//    addi       IntGeneral
//    addic      IntGeneral
//    addic.     IntGeneral
//    addis      IntGeneral
//    addme      IntGeneral
//    addze      IntGeneral
//    and        IntGeneral
//    andc       IntGeneral
//    andi.      IntGeneral
//    andis.     IntGeneral
//    b          BrB
//    bc         BrB
//    bcctr      BrB
//    bclr       BrB
//    cmp        IntCompare
//    cmpi       IntCompare
//    cmpl       IntCompare
//    cmpli      IntCompare
//    cntlzd     IntRotateD
//    cntlzw     IntGeneral
//    crand      BrCR
//    crandc     BrCR
//    creqv      BrCR
//    crnand     BrCR
//    crnor      BrCR
//    cror       BrCR
//    crorc      BrCR
//    crxor      BrCR
//    dcba       LdStDCBA
//    dcbf       LdStDCBF
//    dcbi       LdStDCBI
//    dcbst      LdStDCBF
//    dcbt       LdStGeneral
//    dcbtst     LdStGeneral
//    dcbz       LdStDCBF
//    divd       IntDivD
//    divdu      IntDivD
//    divw       IntDivW
//    divwu      IntDivW
//    dss        LdStDSS
//    dst        LdStDSS
//    dstst      LdStDSS
//    eciwx      LdStGeneral
//    ecowx      LdStGeneral
//    eieio      LdStGeneral
//    eqv        IntGeneral
//    extsb      IntGeneral
//    extsh      IntGeneral
//    extsw      IntRotateD
//    fabs       FPGeneral
//    fadd       FPGeneral
//    fadds      FPGeneral
//    fcfid      FPGeneral
//    fcmpo      FPCompare
//    fcmpu      FPCompare
//    fctid      FPGeneral
//    fctidz     FPGeneral
//    fctiw      FPGeneral
//    fctiwz     FPGeneral
//    fdiv       FPDivD
//    fdivs      FPDivS
//    fmadd      FPFused
//    fmadds     FPGeneral
//    fmr        FPGeneral
//    fmsub      FPFused
//    fmsubs     FPGeneral
//    fmul       FPFused
//    fmuls      FPGeneral
//    fnabs      FPGeneral
//    fneg       FPGeneral
//    fnmadd     FPFused
//    fnmadds    FPGeneral
//    fnmsub     FPFused
//    fnmsubs    FPGeneral
//    fres       FPRes
//    frsp       FPGeneral
//    frsqrte    FPGeneral
//    fsel       FPGeneral
//    fsqrt      FPSqrt
//    fsqrts     FPSqrt
//    fsub       FPGeneral
//    fsubs      FPGeneral
//    icbi       LdStICBI
//    isync      SprISYNC
//    lbz        LdStGeneral
//    lbzu       LdStGeneral
//    lbzux      LdStUX
//    lbzx       LdStGeneral
//    ld         LdStLD
//    ldarx      LdStLDARX
//    ldu        LdStLD
//    ldux       LdStLD
//    ldx        LdStLD
//    lfd        LdStLFD
//    lfdu       LdStLFDU
//    lfdux      LdStLFDU
//    lfdx       LdStLFDU
//    lfs        LdStLFDU
//    lfsu       LdStLFDU
//    lfsux      LdStLFDU
//    lfsx       LdStLFDU
//    lha        LdStLHA
//    lhau       LdStLHA
//    lhaux      LdStLHA
//    lhax       LdStLHA
//    lhbrx      LdStGeneral
//    lhz        LdStGeneral
//    lhzu       LdStGeneral
//    lhzux      LdStUX
//    lhzx       LdStGeneral
//    lmw        LdStLMW
//    lswi       LdStLMW
//    lswx       LdStLMW
//    lvebx      LdStLVecX
//    lvehx      LdStLVecX
//    lvewx      LdStLVecX
//    lvsl       LdStLVecX
//    lvsr       LdStLVecX
//    lvx        LdStLVecX
//    lvxl       LdStLVecX
//    lwa        LdStLWA
//    lwarx      LdStLWARX
//    lwaux      LdStLHA
//    lwax       LdStLHA
//    lwbrx      LdStGeneral
//    lwz        LdStGeneral
//    lwzu       LdStGeneral
//    lwzux      LdStUX
//    lwzx       LdStGeneral
//    mcrf       BrMCR
//    mcrfs      FPGeneral
//    mcrxr      BrMCRX
//    mfcr       SprMFCR
//    mffs       IntMFFS
//    mfmsr      SprMFMSR
//    mfspr      SprMFSPR
//    mfsr       SprMFSR
//    mfsrin     SprMFSR
//    mftb       SprMFTB
//    mfvscr     IntMFVSCR
//    mtcrf      BrMCRX
//    mtfsb0     IntMTFSB0
//    mtfsb1     IntMTFSB0
//    mtfsf      IntMTFSB0
//    mtfsfi     IntMTFSB0
//    mtmsr      SprMTMSR
//    mtmsrd     LdStLD
//    mtspr      SprMTSPR
//    mtsr       SprMTSR
//    mtsrd      IntMTSRD
//    mtsrdin    IntMTSRD
//    mtsrin     SprMTSRIN
//    mtvscr     IntMFVSCR
//    mulhd      IntMulHD
//    mulhdu     IntMulHD
//    mulhw      IntMulHW
//    mulhwu     IntMulHWU
//    mulld      IntMulHD
//    mulli      IntMulLI
//    mullw      IntMulHW
//    nand       IntGeneral
//    neg        IntGeneral
//    nor        IntGeneral
//    or         IntGeneral
//    orc        IntGeneral
//    ori        IntGeneral
//    oris       IntGeneral
//    rfi        SprRFI
//    rfid       IntRFID
//    rldcl      IntRotateD
//    rldcr      IntRotateD
//    rldic      IntRotateD
//    rldicl     IntRotateD
//    rldicr     IntRotateD
//    rldimi     IntRotateD
//    rlwimi     IntRotate
//    rlwinm     IntGeneral
//    rlwnm      IntGeneral
//    sc         SprSC
//    slbia      LdStSLBIA
//    slbie      LdStSLBIE
//    sld        IntRotateD
//    slw        IntGeneral
//    srad       IntRotateD
//    sradi      IntRotateD
//    sraw       IntShift
//    srawi      IntShift
//    srd        IntRotateD
//    srw        IntGeneral
//    stb        LdStGeneral
//    stbu       LdStGeneral
//    stbux      LdStGeneral
//    stbx       LdStGeneral
//    std        LdStSTD
//    stdcx.     LdStSTDCX
//    stdu       LdStSTD
//    stdux      LdStSTD
//    stdx       LdStSTD
//    stfd       LdStUX
//    stfdu      LdStUX
//    stfdux     LdStUX
//    stfdx      LdStUX
//    stfiwx     LdStUX
//    stfs       LdStUX
//    stfsu      LdStUX
//    stfsux     LdStUX
//    stfsx      LdStUX
//    sth        LdStGeneral
//    sthbrx     LdStGeneral
//    sthu       LdStGeneral
//    sthux      LdStGeneral
//    sthx       LdStGeneral
//    stmw       LdStLMW
//    stswi      LdStLMW
//    stswx      LdStLMW
//    stvebx     LdStSTVEBX
//    stvehx     LdStSTVEBX
//    stvewx     LdStSTVEBX
//    stvx       LdStSTVEBX
//    stvxl      LdStSTVEBX
//    stw        LdStGeneral
//    stwbrx     LdStGeneral
//    stwcx.     LdStSTWCX
//    stwu       LdStGeneral
//    stwux      LdStGeneral
//    stwx       LdStGeneral
//    subf       IntGeneral
//    subfc      IntGeneral
//    subfe      IntGeneral
//    subfic     IntGeneral
//    subfme     IntGeneral
//    subfze     IntGeneral
//    sync       LdStSync
//    td         IntTrapD
//    tdi        IntTrapD
//    tlbia      LdStSLBIA
//    tlbie      LdStDCBF
//    tlbsync    SprTLBSYNC
//    tw         IntTrapW
//    twi        IntTrapW
//    vaddcuw    VecGeneral
//    vaddfp     VecFP
//    vaddsbs    VecGeneral
//    vaddshs    VecGeneral
//    vaddsws    VecGeneral
//    vaddubm    VecGeneral
//    vaddubs    VecGeneral
//    vadduhm    VecGeneral
//    vadduhs    VecGeneral
//    vadduwm    VecGeneral
//    vadduws    VecGeneral
//    vand       VecGeneral
//    vandc      VecGeneral
//    vavgsb     VecGeneral
//    vavgsh     VecGeneral
//    vavgsw     VecGeneral
//    vavgub     VecGeneral
//    vavguh     VecGeneral
//    vavguw     VecGeneral
//    vcfsx      VecFP
//    vcfux      VecFP
//    vcmpbfp    VecFPCompare
//    vcmpeqfp   VecFPCompare
//    vcmpequb   VecGeneral
//    vcmpequh   VecGeneral
//    vcmpequw   VecGeneral
//    vcmpgefp   VecFPCompare
//    vcmpgtfp   VecFPCompare
//    vcmpgtsb   VecGeneral
//    vcmpgtsh   VecGeneral
//    vcmpgtsw   VecGeneral
//    vcmpgtub   VecGeneral
//    vcmpgtuh   VecGeneral
//    vcmpgtuw   VecGeneral
//    vctsxs     VecFP
//    vctuxs     VecFP
//    vexptefp   VecFP
//    vlogefp    VecFP
//    vmaddfp    VecFP
//    vmaxfp     VecFPCompare
//    vmaxsb     VecGeneral
//    vmaxsh     VecGeneral
//    vmaxsw     VecGeneral
//    vmaxub     VecGeneral
//    vmaxuh     VecGeneral
//    vmaxuw     VecGeneral
//    vmhaddshs  VecComplex
//    vmhraddshs VecComplex
//    vminfp     VecFPCompare
//    vminsb     VecGeneral
//    vminsh     VecGeneral
//    vminsw     VecGeneral
//    vminub     VecGeneral
//    vminuh     VecGeneral
//    vminuw     VecGeneral
//    vmladduhm  VecComplex
//    vmrghb     VecPerm
//    vmrghh     VecPerm
//    vmrghw     VecPerm
//    vmrglb     VecPerm
//    vmrglh     VecPerm
//    vmrglw     VecPerm
//    vmsubfp    VecFP
//    vmsummbm   VecComplex
//    vmsumshm   VecComplex
//    vmsumshs   VecComplex
//    vmsumubm   VecComplex
//    vmsumuhm   VecComplex
//    vmsumuhs   VecComplex
//    vmulesb    VecComplex
//    vmulesh    VecComplex
//    vmuleub    VecComplex
//    vmuleuh    VecComplex
//    vmulosb    VecComplex
//    vmulosh    VecComplex
//    vmuloub    VecComplex
//    vmulouh    VecComplex
//    vnor       VecGeneral
//    vor        VecGeneral
//    vperm      VecPerm
//    vpkpx      VecPerm
//    vpkshss    VecPerm
//    vpkshus    VecPerm
//    vpkswss    VecPerm
//    vpkswus    VecPerm
//    vpkuhum    VecPerm
//    vpkuhus    VecPerm
//    vpkuwum    VecPerm
//    vpkuwus    VecPerm
//    vrefp      VecFPRound
//    vrfim      VecFPRound
//    vrfin      VecFPRound
//    vrfip      VecFPRound
//    vrfiz      VecFPRound
//    vrlb       VecGeneral
//    vrlh       VecGeneral
//    vrlw       VecGeneral
//    vrsqrtefp  VecFP
//    vsel       VecGeneral
//    vsl        VecVSL
//    vslb       VecGeneral
//    vsldoi     VecPerm
//    vslh       VecGeneral
//    vslo       VecPerm
//    vslw       VecGeneral
//    vspltb     VecPerm
//    vsplth     VecPerm
//    vspltisb   VecPerm
//    vspltish   VecPerm
//    vspltisw   VecPerm
//    vspltw     VecPerm
//    vsr        VecVSR
//    vsrab      VecGeneral
//    vsrah      VecGeneral
//    vsraw      VecGeneral
//    vsrb       VecGeneral
//    vsrh       VecGeneral
//    vsro       VecPerm
//    vsrw       VecGeneral
//    vsubcuw    VecGeneral
//    vsubfp     VecFP
//    vsubsbs    VecGeneral
//    vsubshs    VecGeneral
//    vsubsws    VecGeneral
//    vsububm    VecGeneral
//    vsububs    VecGeneral
//    vsubuhm    VecGeneral
//    vsubuhs    VecGeneral
//    vsubuwm    VecGeneral
//    vsubuws    VecGeneral
//    vsum2sws   VecComplex
//    vsum4sbs   VecComplex
//    vsum4shs   VecComplex
//    vsum4ubs   VecComplex
//    vsumsws    VecComplex
//    vupkhpx    VecPerm
//    vupkhsb    VecPerm
//    vupkhsh    VecPerm
//    vupklpx    VecPerm
//    vupklsb    VecPerm
//    vupklsh    VecPerm
//    vxor       VecGeneral
//    xor        IntGeneral
//    xori       IntGeneral
//    xoris      IntGeneral
//
