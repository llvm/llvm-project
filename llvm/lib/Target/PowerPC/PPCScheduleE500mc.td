//===-- PPCScheduleE500mc.td - e500mc Scheduling Defs ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the Freescale e500mc 32-bit 
// Power processor.
// 
// All information is derived from the "e500mc Core Reference Manual",
// Freescale Document Number E500MCRM, Rev. 1, 03/2012.
//
//===----------------------------------------------------------------------===//
// Relevant functional units in the Freescale e500mc core:
//
//  * Decode & Dispatch
//    Can dispatch up to 2 instructions per clock cycle to either the GPR Issue
//    queues (GIQx), FP Issue Queue (FIQ), or Branch issue queue (BIQ).
def DIS0 : FuncUnit; // Dispatch stage - insn 1
def DIS1 : FuncUnit; // Dispatch stage - insn 2

//  * Execute
//    6 pipelined execution units: SFX0, SFX1, BU, FPU, LSU, CFX.
//    Some instructions can only execute in SFX0 but not SFX1.
//    The CFX has a bypass path, allowing non-divide instructions to execute 
//    while a divide instruction is executed.
def SFX0  : FuncUnit; // Simple unit 0
def SFX1  : FuncUnit; // Simple unit 1
def BU    : FuncUnit; // Branch unit
def CFX_DivBypass 
          : FuncUnit; // CFX divide bypass path
def CFX_0 : FuncUnit; // CFX pipeline
def LSU_0 : FuncUnit; // LSU pipeline
def FPU_0 : FuncUnit; // FPU pipeline

def PPCE500mcItineraries : ProcessorItineraries<
  [DIS0, DIS1, SFX0, SFX1, BU, CFX_DivBypass, CFX_0, LSU_0, FPU_0],
  [CR_Bypass, GPR_Bypass, FPR_Bypass], [
  InstrItinData<IntSimple   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1, 1], // Latency = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntGeneral  , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1, 1], // Latency = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntCompare  , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [5, 1, 1], // Latency = 1 or 2
                              [CR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntDivW     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [CFX_0], 0>,
                               InstrStage<14, [CFX_DivBypass]>],
                              [17, 1, 1], // Latency=4..35, Repeat= 4..35
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntMFFS     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<8, [FPU_0]>],
                              [11], // Latency = 8
                              [FPR_Bypass]>,
  InstrItinData<IntMTFSB0   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<8, [FPU_0]>],
                              [11, 1, 1], // Latency = 8
                              [NoBypass, NoBypass, NoBypass]>,
  InstrItinData<IntMulHW    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [CFX_0]>],
                              [7, 1, 1], // Latency = 4, Repeat rate = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntMulHWU   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [CFX_0]>],
                              [7, 1, 1], // Latency = 4, Repeat rate = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntMulLI    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [CFX_0]>],
                              [7, 1, 1], // Latency = 4, Repeat rate = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntRotate   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1, 1], // Latency = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntShift    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1, 1], // Latency = 1
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<IntTrapW    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<2, [SFX0]>],
                              [5, 1], // Latency = 2, Repeat rate = 2
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<BrB         , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [BU]>],
                              [4, 1], // Latency = 1
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<BrCR        , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [BU]>],
                              [4, 1, 1], // Latency = 1
                              [CR_Bypass, CR_Bypass, CR_Bypass]>,
  InstrItinData<BrMCR       , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [BU]>],
                              [4, 1], // Latency = 1
                              [CR_Bypass, CR_Bypass]>,
  InstrItinData<BrMCRX      , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1, 1], // Latency = 1
                              [CR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStDCBA    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3, Repeat rate = 1
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStDCBF    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStDCBI    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStLoad    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStLoadUpd , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass],
                              2>, // 2 micro-ops                              
  InstrItinData<LdStStore   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<LdStStoreUpd, [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [NoBypass, GPR_Bypass],
                              2>, // 2 micro-ops                              
  InstrItinData<LdStICBI    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<LdStSTFD    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStSTFDU   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass],
                              2>, // 2 micro-ops                              
  InstrItinData<LdStLFD     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [7, 1, 1], // Latency = 4
                              [FPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStLFDU    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [7, 1, 1], // Latency = 4
                              [FPR_Bypass, GPR_Bypass, GPR_Bypass],
                              2>, // 2 micro-ops
  InstrItinData<LdStLHA     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStLHAU    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [GPR_Bypass, GPR_Bypass]>,                              
  InstrItinData<LdStLMW     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [7, 1], // Latency = r+3
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<LdStLWARX   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<3, [LSU_0]>],
                              [6, 1, 1], // Latency = 3, Repeat rate = 3
                              [GPR_Bypass, GPR_Bypass, GPR_Bypass]>,
  InstrItinData<LdStSTWCX   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>],
                              [6, 1], // Latency = 3
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<LdStSync    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0]>]>,
  InstrItinData<SprMFSR     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<4, [SFX0]>],
                              [7, 1],
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<SprMTMSR    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<2, [SFX0, SFX1]>],
                              [5, 1], // Latency = 2, Repeat rate = 4
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<SprMTSR     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0]>],
                              [5, 1],
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<SprTLBSYNC  , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [LSU_0], 0>]>,
  InstrItinData<SprMFCR     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<5, [SFX0]>],
                              [8, 1],
                              [GPR_Bypass, CR_Bypass]>,
  InstrItinData<SprMFMSR    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<4, [SFX0]>],
                              [7, 1], // Latency = 4, Repeat rate = 4
                              [GPR_Bypass, GPR_Bypass]>,
  InstrItinData<SprMFSPR    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1], // Latency = 1, Repeat rate = 1
                              [GPR_Bypass, CR_Bypass]>,
  InstrItinData<SprMFTB     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<4, [SFX0]>],
                              [7, 1], // Latency = 4, Repeat rate = 4
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<SprMTSPR    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0, SFX1]>],
                              [4, 1], // Latency = 1, Repeat rate = 1
                              [CR_Bypass, GPR_Bypass]>,
  InstrItinData<SprMTSRIN   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<1, [SFX0]>],
                              [4, 1],
                              [NoBypass, GPR_Bypass]>,
  InstrItinData<FPGeneral   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<2, [FPU_0]>],
                              [11, 1, 1], // Latency = 8, Repeat rate = 2 
                              [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<FPAddSub    , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<4, [FPU_0]>],
                              [13, 1, 1], // Latency = 10, Repeat rate = 4 
                              [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,                              
  InstrItinData<FPCompare   , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<2, [FPU_0]>],
                              [11, 1, 1], // Latency = 8, Repeat rate = 2
                              [CR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<FPDivD      , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<68, [FPU_0]>],
                              [71, 1, 1], // Latency = 68, Repeat rate = 68
                              [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<FPDivS      , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<38, [FPU_0]>],
                              [41, 1, 1], // Latency = 38, Repeat rate = 38
                              [FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<FPFused     , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<4, [FPU_0]>],
                              [13, 1, 1, 1], // Latency = 10, Repeat rate = 4
                              [FPR_Bypass, FPR_Bypass, FPR_Bypass, FPR_Bypass]>,
  InstrItinData<FPRes       , [InstrStage<1, [DIS0, DIS1], 0>,
                               InstrStage<38, [FPU_0]>],
                              [41, 1], // Latency = 38, Repeat rate = 38
                              [FPR_Bypass, FPR_Bypass]>
]>;

// ===---------------------------------------------------------------------===//
// e500mc machine model for scheduling and other instruction cost heuristics.

def PPCE500mcModel : SchedMachineModel {
  let IssueWidth = 2;  // 2 micro-ops are dispatched per cycle.
  let MinLatency = -1; // OperandCycles are interpreted as MinLatency.
  let LoadLatency = 5; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.

  let Itineraries = PPCE500mcItineraries;
}
