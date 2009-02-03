//===- PPCScheduleG4Plus.td - PPC G4+ Scheduling Defs. -----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G4+ (7450) processor.
//
//===----------------------------------------------------------------------===//

def G4PlusItineraries : ProcessorItineraries<[
  InstrItinData<IntGeneral  , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IntCompare  , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IntDivW     , [InstrStage<23, [IU2]>]>,
  InstrItinData<IntMFFS     , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IntMFVSCR   , [InstrStage<2, [VFPU]>]>,
  InstrItinData<IntMTFSB0   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<IntMulHW    , [InstrStage<4, [IU2]>]>,
  InstrItinData<IntMulHWU   , [InstrStage<4, [IU2]>]>,
  InstrItinData<IntMulLI    , [InstrStage<3, [IU2]>]>,
  InstrItinData<IntRotate   , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IntShift    , [InstrStage<2, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<IntTrapW    , [InstrStage<2, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<BrCR        , [InstrStage<2, [IU2]>]>,
  InstrItinData<BrMCR       , [InstrStage<2, [IU2]>]>,
  InstrItinData<BrMCRX      , [InstrStage<2, [IU2]>]>,
  InstrItinData<LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStDCBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStGeneral , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStDSS     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStICBI    , [InstrStage<3, [IU2]>]>,
  InstrItinData<LdStUX      , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLFD     , [InstrStage<4, [SLU]>]>,
  InstrItinData<LdStLFDU    , [InstrStage<4, [SLU]>]>,
  InstrItinData<LdStLHA     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLMW     , [InstrStage<37, [SLU]>]>,
  InstrItinData<LdStLVecX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLWA     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStLWARX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTD     , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTDCX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTVEBX  , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTWCX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSync    , [InstrStage<35, [SLU]>]>,
  InstrItinData<SprISYNC    , [InstrStage<0, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<SprMFSR     , [InstrStage<4, [IU2]>]>,
  InstrItinData<SprMTMSR    , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprMTSR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprTLBSYNC  , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprMFCR     , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprMFMSR    , [InstrStage<3, [IU2]>]>,
  InstrItinData<SprMFSPR    , [InstrStage<4, [IU2]>]>,
  InstrItinData<SprMFTB     , [InstrStage<5, [IU2]>]>,
  InstrItinData<SprMTSPR    , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprMTSRIN   , [InstrStage<2, [IU2]>]>,
  InstrItinData<SprRFI      , [InstrStage<1, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<SprSC       , [InstrStage<0, [IU1, IU2, IU3, IU4]>]>,
  InstrItinData<FPGeneral   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<FPCompare   , [InstrStage<5, [FPU1]>]>,
  InstrItinData<FPDivD      , [InstrStage<35, [FPU1]>]>,
  InstrItinData<FPDivS      , [InstrStage<21, [FPU1]>]>,
  InstrItinData<FPFused     , [InstrStage<5, [FPU1]>]>,
  InstrItinData<FPRes       , [InstrStage<14, [FPU1]>]>,
  InstrItinData<VecGeneral  , [InstrStage<1, [VIU1]>]>,
  InstrItinData<VecFP       , [InstrStage<4, [VFPU]>]>,
  InstrItinData<VecFPCompare, [InstrStage<2, [VFPU]>]>,
  InstrItinData<VecComplex  , [InstrStage<4, [VIU2]>]>,
  InstrItinData<VecPerm     , [InstrStage<2, [VPU]>]>,
  InstrItinData<VecFPRound  , [InstrStage<4, [VIU1]>]>,
  InstrItinData<VecVSL      , [InstrStage<2, [VPU]>]>,
  InstrItinData<VecVSR      , [InstrStage<2, [VPU]>]>
]>;
