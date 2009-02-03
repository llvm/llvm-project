//===- PPCScheduleG3.td - PPC G3 Scheduling Definitions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the G3 (750) processor.
//
//===----------------------------------------------------------------------===//


def G3Itineraries : ProcessorItineraries<[
  InstrItinData<IntGeneral  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntCompare  , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntDivW     , [InstrStage<19, [IU1]>]>,
  InstrItinData<IntMFFS     , [InstrStage<1, [FPU1]>]>,
  InstrItinData<IntMTFSB0   , [InstrStage<3, [FPU1]>]>,
  InstrItinData<IntMulHW    , [InstrStage<5, [IU1]>]>,
  InstrItinData<IntMulHWU   , [InstrStage<6, [IU1]>]>,
  InstrItinData<IntMulLI    , [InstrStage<3, [IU1]>]>,
  InstrItinData<IntRotate   , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntShift    , [InstrStage<1, [IU1, IU2]>]>,
  InstrItinData<IntTrapW    , [InstrStage<2, [IU1, IU2]>]>,
  InstrItinData<BrB         , [InstrStage<1, [BPU]>]>,
  InstrItinData<BrCR        , [InstrStage<1, [SRU]>]>,
  InstrItinData<BrMCR       , [InstrStage<1, [SRU]>]>,
  InstrItinData<BrMCRX      , [InstrStage<1, [SRU]>]>,
  InstrItinData<LdStDCBA    , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStDCBF    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStDCBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStGeneral , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStICBI    , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStUX      , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLFD     , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLFDU    , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLHA     , [InstrStage<2, [SLU]>]>,
  InstrItinData<LdStLMW     , [InstrStage<34, [SLU]>]>,
  InstrItinData<LdStLWARX   , [InstrStage<3, [SLU]>]>,
  InstrItinData<LdStSTWCX   , [InstrStage<8, [SLU]>]>,
  InstrItinData<LdStSync    , [InstrStage<3, [SLU]>]>,
  InstrItinData<SprISYNC    , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprMFSR     , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMTMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMTSR     , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprTLBSYNC  , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMFCR     , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMFMSR    , [InstrStage<1, [SRU]>]>,
  InstrItinData<SprMFSPR    , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMFTB     , [InstrStage<3, [SRU]>]>,
  InstrItinData<SprMTSPR    , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprMTSRIN   , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprRFI      , [InstrStage<2, [SRU]>]>,
  InstrItinData<SprSC       , [InstrStage<2, [SRU]>]>,
  InstrItinData<FPGeneral   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<FPCompare   , [InstrStage<1, [FPU1]>]>,
  InstrItinData<FPDivD      , [InstrStage<31, [FPU1]>]>,
  InstrItinData<FPDivS      , [InstrStage<17, [FPU1]>]>,
  InstrItinData<FPFused     , [InstrStage<2, [FPU1]>]>,
  InstrItinData<FPRes       , [InstrStage<10, [FPU1]>]>
]>;
