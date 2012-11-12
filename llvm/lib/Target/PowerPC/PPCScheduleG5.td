//===-- PPCScheduleG5.td - PPC G5 Scheduling Definitions ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G5 (970) processor.
//
//===----------------------------------------------------------------------===//

def G5Itineraries : ProcessorItineraries<
  [IU1, IU2, SLU, BPU, FPU1, FPU2, VFPU, VIU1, VIU2, VPU], [], [
  InstrItinData<IntSimple   , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IntGeneral  , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IntCompare  , [InstrStage<3, [IU1, IU2]>]>,
  InstrItinData<IntDivD     , [InstrStage<68, [IU1]>]>,
  InstrItinData<IntDivW     , [InstrStage<36, [IU1]>]>,
  InstrItinData<IntMFFS     , [InstrStage<6, [IU2]>]>,
  InstrItinData<IntMFVSCR   , [InstrStage<1, [VFPU]>]>,
  InstrItinData<IntMTFSB0   , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<IntMulHD    , [InstrStage<7, [IU1, IU2]>]>,
  InstrItinData<IntMulHW    , [InstrStage<5, [IU1, IU2]>]>,
  InstrItinData<IntMulHWU   , [InstrStage<5, [IU1, IU2]>]>,
  InstrItinData<IntMulLI    , [InstrStage<4, [IU1, IU2]>]>,
  InstrItinData<IntRFID     , [InstrStage<1, [IU2]>]>,
  InstrItinData<IntRotateD  , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IntRotateDI , [InstrStage<2, [IU1, IU2]>]>,  
  InstrItinData<IntRotate   , [InstrStage<4, [IU1, IU2]>]>,
  InstrItinData<IntShift    , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<IntTrapD    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntTrapW    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<BrCR        , [InstrStage<4, [BPU]>]>,
  InstrItinData<BrMCR       , [InstrStage<2, [BPU]>]>,
  InstrItinData<BrMCRX      , [InstrStage<3, [BPU]>]>,
  InstrItinData<LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLoad    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLoadUpd , [InstrStage<3, [SLU]>]>,  
  InstrItinData<LdStStore   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStStoreUpd, [InstrStage<3, [SLU]>]>,  
  InstrItinData<LdStDSS     , [InstrStage<10, [SLU]>]>,
  InstrItinData<LdStICBI    , [InstrStage<40, [SLU]>]>,
  InstrItinData<LdStSTFD    , [InstrStage<4, [SLU]>]>,
  InstrItinData<LdStSTFDU   , [InstrStage<4, [SLU]>]>,  
  InstrItinData<LdStLD      , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLDU     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLDARX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<LdStLFD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLFDU    , [InstrStage<5, [SLU]>]>,
  InstrItinData<LdStLHA     , [InstrStage<5, [SLU]>]>,
  InstrItinData<LdStLHAU    , [InstrStage<5, [SLU]>]>,  
  InstrItinData<LdStLMW     , [InstrStage<64, [SLU]>]>,
  InstrItinData<LdStLVecX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLWA     , [InstrStage<5, [SLU]>]>,
  InstrItinData<LdStLWARX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<LdStSLBIA   , [InstrStage<40, [SLU]>]>, // needs work
  InstrItinData<LdStSLBIE   , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStSTD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTDU    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTDCX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<LdStSTVEBX  , [InstrStage<5, [SLU]>]>,
  InstrItinData<LdStSTWCX   , [InstrStage<11, [SLU]>]>,
  InstrItinData<LdStSync    , [InstrStage<35, [SLU]>]>,
  InstrItinData<SprISYNC    , [InstrStage<40, [SLU]>]>, // needs work
  InstrItinData<SprMFSR     , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprMTMSR    , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprMTSR     , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprTLBSYNC  , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprMFCR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprMFMSR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<SprMFSPR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<SprMFTB     , [InstrStage<10, [IU2]>]>,
  InstrItinData<SprMTSPR    , [InstrStage<8, [IU2]>]>,
  InstrItinData<SprSC       , [InstrStage<1, [IU2]>]>,
  InstrItinData<FPGeneral   , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<FPAddSub    , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<FPCompare   , [InstrStage<8, [FPU1, FPU2]>]>,
  InstrItinData<FPDivD      , [InstrStage<33, [FPU1, FPU2]>]>,
  InstrItinData<FPDivS      , [InstrStage<33, [FPU1, FPU2]>]>,
  InstrItinData<FPFused     , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<FPRes       , [InstrStage<6, [FPU1, FPU2]>]>,
  InstrItinData<FPSqrt      , [InstrStage<40, [FPU1, FPU2]>]>,
  InstrItinData<VecGeneral  , [InstrStage<2, [VIU1]>]>,
  InstrItinData<VecFP       , [InstrStage<8, [VFPU]>]>,
  InstrItinData<VecFPCompare, [InstrStage<2, [VFPU]>]>,
  InstrItinData<VecComplex  , [InstrStage<5, [VIU2]>]>,
  InstrItinData<VecPerm     , [InstrStage<3, [VPU]>]>,
  InstrItinData<VecFPRound  , [InstrStage<8, [VFPU]>]>,
  InstrItinData<VecVSL      , [InstrStage<2, [VIU1]>]>,
  InstrItinData<VecVSR      , [InstrStage<3, [VPU]>]>
]>;
