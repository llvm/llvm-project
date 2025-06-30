//=----- X86SchedLunarlakeP.td - X86 LunarlakeP Scheduling *- tablegen -----*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for LunarlakeP to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//
def LunarlakePModel : SchedMachineModel {
  // LunarlakeP can allocate 8 uops per cycle.
  // Max micro-ops that may be scheduled per cycle.
  // Based on Allocator Width
  let IssueWidth = 8;  
  // Max micro-ops that can be buffered. 
  // Based on size of ROB
  let MicroOpBufferSize = 792;
  // INT LOAD takes 4 cycles
  let LoadLatency = 4;
  let MispredictPenalty = 14;
  // Latency for microcoded instructions or instructions without latency info.
  int MaxLatency = 100;
  // Based on the LSD (loop-stream detector) queue size (ST).
  // LSD is 200 uops per logical processor in single threaded mode
  // For SMT 100 uops/thread, LionCove removed SMT in HW.
  let LoopMicroOpBufferSize = 200;
  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = LunarlakePModel in {

// LunarlakeP can issue micro-ops to 18 different ports in one cycle.
// Lion Cove architectural spec uses port naming that is not sequential
// for better comprehension we opt for sequential naming since this ports
// serve logical information for schedule only.
// 6 INT ALU Ports {P0 to P5}
def LNLPPort00 : ProcResource<1>;
def LNLPPort01 : ProcResource<1>;
def LNLPPort02 : ProcResource<1>;
def LNLPPort03 : ProcResource<1>;
def LNLPPort04 : ProcResource<1>;
def LNLPPort05 : ProcResource<1>;
// 4 VEC ALU Ports {V0 to V3}
def LNLPVPort00 : ProcResource<1>;
def LNLPVPort01 : ProcResource<1>;
def LNLPVPort02 : ProcResource<1>;
def LNLPVPort03 : ProcResource<1>;
// 2 Store Data Ports {P10 to P11}
def LNLPPort10 : ProcResource<1>;
def LNLPPort11 : ProcResource<1>;
// 6 MEM Ports 6 AGU shared with 3 LD, 3 ST
// AGU LD {P20 to P22}
def LNLPPort20 : ProcResource<1>;
def LNLPPort21 : ProcResource<1>;
def LNLPPort22 : ProcResource<1>;
// AGU ST {P25 to P27}
def LNLPPort25 : ProcResource<1>;
def LNLPPort26 : ProcResource<1>;
def LNLPPort27 : ProcResource<1>;

// Workaround to represent invalid ports. WriteRes shouldn't use this resource.
def LNLPPortInvalid :ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def LNLPVPort00_01          : ProcResGroup<[LNLPVPort00, LNLPVPort01]>;
def LNLPVPort02_03          : ProcResGroup<[LNLPVPort02, LNLPVPort03]>;
def LNLPPort00_02_04        : ProcResGroup<[LNLPPort00, LNLPPort02, LNLPPort04]>;
def LNLPPort01_03_05        : ProcResGroup<[LNLPPort01, LNLPPort03, LNLPPort05]>;
def LNLPPort20_21_22        : ProcResGroup<[LNLPPort20, LNLPPort21, LNLPPort22]>;
def LNLPPort25_26_27        : ProcResGroup<[LNLPPort25, LNLPPort26, LNLPPort27]>;

// INT EU has 112 reservation stations.
def LNLPPort00_01_02_03_04_05 : ProcResGroup<[LNLPPort00, LNLPPort01, LNLPPort02,
                                             LNLPPort03, LNLPPort04, LNLPPort05]>{ 
  let BufferSize = 110;  // Reduced from 128 in GLC
}

// VEC EU has 180 reservation stations.
def LNLPVPort00_01_02_03    : ProcResGroup<[LNLPVPort00, LNLPVPort01, LNLPVPort02,
                                          LNLPVPort03]>{
  let BufferSize = 180; // EU for INT and VEC are seperated 
                        // VEC QUEUE SIZE = 60 + VEC EU RS (60+60)
}
// STD has 48 reservation stations.
def LNLPPort10_11          : ProcResGroup<[LNLPPort10, LNLPPort11]> {
  let BufferSize = 48;
}

// MEM has 72 reservation stations.
def LNLPPort20_21_22_25_26_27 : ProcResGroup<[LNLPPort20, LNLPPort21, LNLPPort22,
                                            LNLPPort25, LNLPPort26, LNLPPort27]> {
  let BufferSize = 72;
}

def LNLPPortAny : ProcResGroup<[LNLPPort00, LNLPPort01, LNLPPort02, LNLPPort03,
                               LNLPPort04, LNLPPort05, LNLPVPort00, LNLPVPort01,
                               LNLPVPort02, LNLPVPort03, LNLPPort10, LNLPPort11,
                               LNLPPort20, LNLPPort21, LNLPPort22, LNLPPort25,
                               LNLPPort26, LNLPPort27]>;

// Integer loads are 4 cycles, so ReadAfterLd registers needn't be available
// until 4 cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 4>;

// TODO: 6 Cycle latency for Vec load comes from ADL
// Vector loads are 6 cycles, so ReadAfterVec*Ld registers needn't be available
// until 6 cycles after the memory operand.
def : ReadAdvance<ReadAfterVecLd, 6>;
def : ReadAdvance<ReadAfterVecXLd, 6>;
def : ReadAdvance<ReadAfterVecYLd, 6>;

def : ReadAdvance<ReadInt2Fpu, 0>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass LNLPWriteResPair<X86FoldableSchedWrite SchedRW,
                           list<ProcResourceKind> ExePorts,
                           int Lat, list<int> Res = [1], int UOps = 1,
                           int LoadLat = 4, int LoadUOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on port 20/21/22 and adds LoadLat cycles to
  // the latency (default = 4).
  def : WriteRes<SchedRW.Folded, !listconcat([LNLPPort20_21_22], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ReleaseAtCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, LoadUOps);
  }
}

defm : X86WriteResUnsupported<WriteBEXTRLd>;

//===----------------------------------------------------------------------===//
// The following definitons are infered by smg.
//===----------------------------------------------------------------------===//

def : WriteRes<WriteADC, [LNLPPort00_02_04]>;
defm : X86WriteRes<WriteADCLd, [LNLPPort00_01_02_03_04_05, LNLPPort00_02_04], 11, [1, 1], 2>;
def : WriteRes<WriteAESDecEnc, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 3;
}
defm : X86WriteRes<WriteAESDecEncLd, [LNLPVPort00_01, LNLPPort20_21_22], 11, [4, 7], 2>;
defm : X86WriteResPairUnsupported<WriteAESIMC>;
defm : X86WriteResPairUnsupported<WriteAESKeyGen>;
def  : WriteRes<WriteALU, [LNLPPort01_03_05]>;
def  : WriteRes<WriteALULd, [LNLPPort01_03_05, LNLPPort20_21_22]> {
  let Latency = 11;
}
defm : X86WriteRes<WriteBEXTR, [LNLPPort01_03_05], 6, [2], 2>;
def  : WriteRes<WriteBLS, [LNLPPort01_03_05]>;
defm : X86WriteRes<WriteBLSLd, [LNLPPort01_03_05, LNLPPort20_21_22], 5, [1, 4], 2>;
defm : LNLPWriteResPair<WriteBSF, [LNLPPort01_03_05], 3, [1]>;
defm : LNLPWriteResPair<WriteBSR, [LNLPPort01_03_05], 3, [1]>;
def  : WriteRes<WriteBSWAP32, [LNLPPort01_03_05]>;
defm : X86WriteRes<WriteBSWAP64, [LNLPPort01_03_05, LNLPPort01_03_05], 2, [1, 1], 2>;
defm : LNLPWriteResPair<WriteBZHI, [LNLPPort01_03_05], 3, [1]>;
def : WriteRes<WriteBitTestSet, [LNLPPort01_03_05]>;
def : WriteRes<WriteBitTestSetImmLd, [LNLPPort01_03_05]> {
  let Latency = 11;
}
defm : X86WriteRes<WriteBitTestSetRegLd, [LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort00_02_04, LNLPPort01_03_05], 17, [3, 2, 1, 2], 8>;
def : WriteRes<WriteBitTest, [LNLPPort01_03_05]>;
defm : X86WriteRes<WriteBitTestImmLd, [LNLPPort01_03_05, LNLPPort20_21_22], 6, [1, 1], 2>;
defm : X86WriteRes<WriteBitTestRegLd, [LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22], 11, [4, 2, 1, 2, 1], 10>;

def  : WriteRes<WriteBlend, [LNLPVPort00_01_02_03]>;
defm : X86WriteRes<WriteBlendLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 7, [1, 6], 2>;
def  : WriteRes<WriteBlendY, [LNLPVPort00_01_02_03]>;
defm : X86WriteRes<WriteBlendYLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 8, [1, 7], 2>;

defm : LNLPWriteResPair<WriteCLMul, [LNLPVPort02_03], 3, [1], 1, 7>;
def : WriteRes<WriteCMOV, [LNLPPort00_01_02_03_04_05]>;
def : WriteRes<WriteCMOVLd, [LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 4];
  let Latency = 5;
}
defm : X86WriteRes<WriteCMPXCHG, [LNLPPort00_01_02_03_04_05, LNLPPort00_02_04], 3, [3, 2], 5>;
defm : X86WriteRes<WriteCMPXCHGRMW, [LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort20_21_22], 12, [1, 2, 1, 1, 1], 6>;

def  : WriteRes<WriteCRC32, [LNLPPort01_03_05]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def  : WriteRes<WriteCRC32Ld, [LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 4];
  let Latency = 7;
}

defm : X86WriteRes<WriteCvtI2PD, [LNLPVPort00_01, LNLPVPort02_03], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2PDLd, [LNLPVPort00_01, LNLPPort20_21_22], 11, [1,1], 2>; 
defm : X86WriteRes<WriteCvtI2PDY, [LNLPVPort00_01, LNLPVPort02_03], 7, [4, 3], 2>;
def  : WriteRes<WriteCvtI2PDYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteCvtI2PDZ>;
def  : WriteRes<WriteCvtI2PS, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteCvtI2PSLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
defm : LNLPWriteResPair<WriteCvtI2PSY, [LNLPVPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteCvtI2PSZ>;
defm : X86WriteRes<WriteCvtI2SD, [LNLPPort01_03_05, LNLPVPort00_01], 8, [4, 4], 2>;
def  : WriteRes<WriteCvtI2SDLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
defm : X86WriteRes<WriteCvtI2SS, [LNLPPort01_03_05, LNLPVPort00_01], 8, [4, 4], 2>;
def  : WriteRes<WriteCvtI2SSLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}

defm : LNLPWriteResPair<WriteCvtPD2I, [LNLPVPort00_01, LNLPVPort02_03], 5, [1, 1], 2, 7>;
defm : LNLPWriteResPair<WriteCvtPD2IY, [LNLPVPort00_01, LNLPVPort02_03], 7, [1, 1], 2, 8>;
defm : X86WriteResPairUnsupported<WriteCvtPD2IZ>;
defm : LNLPWriteResPair<WriteCvtPD2PS, [LNLPVPort00_01, LNLPVPort02_03], 5, [1, 1], 2, 7>;
defm : LNLPWriteResPair<WriteCvtPD2PSY, [LNLPVPort00_01, LNLPVPort02_03], 7, [1, 1], 2, 8>;
defm : X86WriteResPairUnsupported<WriteCvtPD2PSZ>;
defm : X86WriteRes<WriteCvtPH2PS, [LNLPVPort00_01, LNLPVPort02_03], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPH2PSLd, [LNLPVPort00_01, LNLPPort20_21_22], 12, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPH2PSY, [LNLPVPort00_01, LNLPVPort02_03], 7, [4, 3], 2>;
def  : WriteRes<WriteCvtPH2PSYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteCvtPH2PSZ>;
def  : WriteRes<WriteCvtPS2I, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
defm : X86WriteRes<WriteCvtPS2ILd, [LNLPVPort00_01, LNLPPort20_21_22], 10, [4, 6], 2>;
def   : WriteRes<WriteCvtPS2IY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteCvtPS2IYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteCvtPS2IZ>;
defm : X86WriteRes<WriteCvtPS2PD, [LNLPVPort00_01, LNLPVPort02_03], 5, [4, 1], 2>;
def  : WriteRes<WriteCvtPS2PDLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
defm : X86WriteRes<WriteCvtPS2PDY, [LNLPVPort00_01, LNLPVPort02_03], 7, [4, 3], 2>;
def  : WriteRes<WriteCvtPS2PDYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteCvtPS2PDZ>;
defm : X86WriteRes<WriteCvtPS2PH, [LNLPVPort00_01, LNLPVPort02_03], 5, [4, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PHSt, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 5, [4, 1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PHY, [LNLPVPort00_01, LNLPVPort02_03], 7, [4, 3], 2>;
defm : X86WriteRes<WriteCvtPS2PHYSt, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 5, [4, 1, 1], 3>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZ>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZSt>;
defm : X86WriteRes<WriteCvtSD2I, [LNLPVPort00_01], 7, [7], 2>;
defm : X86WriteRes<WriteCvtSD2ILd, [LNLPVPort00_01, LNLPPort20_21_22], 11, [7, 4], 3>;
defm : X86WriteRes<WriteCvtSD2SS, [LNLPVPort00_01, LNLPVPort02_03], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtSD2SSLd, [LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22], 11, [4, 1, 6], 3>;
defm : X86WriteRes<WriteCvtSS2I, [LNLPVPort00_01], 7, [7], 2>;
defm : X86WriteRes<WriteCvtSS2ILd, [LNLPVPort00_01, LNLPPort20_21_22], 11, [7, 4], 2>;
defm : X86WriteRes<WriteCvtSS2SD, [LNLPVPort00_01, LNLPVPort02_03], 5, [4, 1], 2>;
def  : WriteRes<WriteCvtSS2SDLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
defm : X86WriteRes<WriteDPPD, [LNLPVPort00_01, LNLPVPort02_03], 8, [2, 1], 3>;
defm : X86WriteRes<WriteDPPDLd, [LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22], 13, [2, 1, 1], 4>;
//defm : X86WriteRes<WriteDPPS, [LNLPVPort00_01, LNLPVPort02_03], 99, [7, 5], 5>;
// FIXME: Incompleted schedwrite.
//defm : X86WriteResUnsupported<WriteDPPSLd>;
defm : LNLPWriteResPair<WriteDPPS, [LNLPVPort00_01, LNLPVPort02_03, LNLPVPort02_03, LNLPVPort02_03, LNLPVPort00_01], 11, [2, 1, 1, 1, 1], 5, 7>;
//defm : X86WriteRes<WriteDPPSY, [LNLPVPort00_01, LNLPVPort02_03], 12, [7, 5], 5>;
// FIXME: Incompleted schedwrite.
//defm : X86WriteResUnsupported<WriteDPPSYLd>;
defm : LNLPWriteResPair<WriteDPPSY, [LNLPVPort00_01, LNLPVPort02_03, LNLPVPort02_03, LNLPVPort02_03, LNLPVPort00_01], 12, [2, 1, 1, 1, 1], 5, 7>;

defm : LNLPWriteResPair<WriteDiv16, [LNLPPort00_01_02_03_04_05, LNLPPort01], 16, [1, 3], 4, 4>;
defm : LNLPWriteResPair<WriteDiv32, [LNLPPort00_01_02_03_04_05, LNLPPort01], 15, [1, 3], 4, 4>;
defm : LNLPWriteResPair<WriteDiv64, [LNLPPort01], 18, [3], 3>;
defm : X86WriteRes<WriteDiv8, [LNLPPort01], 17, [3], 3>;
defm : X86WriteRes<WriteDiv8Ld, [LNLPPort01], 22, [3], 3>;
defm : X86WriteRes<WriteEMMS, [LNLPPort01, LNLPVPort00_01, LNLPVPort02_03], 10, [1, 8, 1], 10>;
def  : WriteRes<WriteFAdd, [LNLPVPort02_03]> {
  let Latency = 2;
}
// FIXME: Latency
defm : X86WriteRes<WriteFAddLd, [LNLPVPort02_03, LNLPPort20_21_22], 10, [1,1], 2>;  // 8
defm : LNLPWriteResPair<WriteFAdd64, [LNLPVPort02_03], 3, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFAdd64X, [LNLPVPort02_03], 3, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFAdd64Y, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFAdd64Z>;
defm : LNLPWriteResPair<WriteFAddX, [LNLPVPort02_03], 3, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFAddY, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFAddZ>;
def  : WriteRes<WriteFBlend, [LNLPVPort00_01_02_03]>;
defm : X86WriteRes<WriteFBlendLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 7, [1, 6], 2>;
def  : WriteRes<WriteFBlendY, [LNLPVPort00_01_02_03]>;
defm : X86WriteRes<WriteFBlendYLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 8, [1, 7], 2>;
def : WriteRes<WriteFCMOV, [LNLPVPort00_01]> {
  let Latency = 3;
}
def : WriteRes<WriteFCmp, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmpLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFCmp64, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmp64Ld, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFCmp64X, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmp64XLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFCmp64Y, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmp64YLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteFCmp64Z>;
def : WriteRes<WriteFCmpX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmpXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFCmpY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFCmpYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteFCmpZ>;
def : WriteRes<WriteFCom, [LNLPVPort00_01]>;
defm : X86WriteRes<WriteFComLd, [LNLPVPort00_01, LNLPPort20_21_22], 8, [1, 1], 2>;
def  : WriteRes<WriteFComX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
defm : X86WriteRes<WriteFComXLd, [LNLPVPort00_01, LNLPPort20_21_22], 9, [3, 6], 2>;
def  : WriteRes<WriteFDiv, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [7];
  let Latency = 7;
}
def : WriteRes<WriteFDivLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7, 6];
  let Latency = 13;
}
def : WriteRes<WriteFDiv64, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFDiv64Ld, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 6];
  let Latency = 16;
}
def : WriteRes<WriteFDiv64X, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFDiv64XLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 6];
  let Latency = 16;
}
def : WriteRes<WriteFDiv64Y, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFDiv64YLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 7];
  let Latency = 17;
}
defm : X86WriteResPairUnsupported<WriteFDiv64Z>;
def : WriteRes<WriteFDivX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [7];
  let Latency = 7;
}
def : WriteRes<WriteFDivXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7, 6];
  let Latency = 13;
}
def : WriteRes<WriteFDivY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [7];
  let Latency = 7;
}
def : WriteRes<WriteFDivYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7, 7];
  let Latency = 14;
}
defm : X86WriteResPairUnsupported<WriteFDivZ>;
defm : X86WriteRes<WriteFHAdd, [LNLPVPort02_03], 4, [5], 3>;
defm : X86WriteRes<WriteFHAddLd, [LNLPVPort02_03, LNLPPort20_21_22], 10, [5, 6], 4>;
defm : X86WriteRes<WriteFHAddY, [LNLPVPort02_03], 4, [5], 3>;
defm : X86WriteRes<WriteFHAddYLd, [LNLPVPort02_03, LNLPPort20_21_22], 11, [5, 7], 4>;
def : WriteRes<WriteFLD0, [LNLPVPort02_03]>;
defm : X86WriteRes<WriteFLD1, [LNLPVPort02_03], 1, [2], 2>;
defm : X86WriteRes<WriteFLDC, [LNLPVPort02_03], 1, [2], 2>;

def : WriteRes<WriteFLoad, [LNLPPort20_21_22]> {
  let Latency = 7;
}
def : WriteRes<WriteFLoadX, [LNLPPort20_21_22]> {
  let ReleaseAtCycles = [6];
  let Latency = 7;
}
def : WriteRes<WriteFLoadY, [LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7];
  let Latency = 8;
}
def : WriteRes<WriteFLogic, [LNLPVPort00_01_02_03]>;
def : WriteRes<WriteFLogicLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
}
def : WriteRes<WriteFLogicY, [LNLPVPort00_01_02_03]>;
def : WriteRes<WriteFLogicYLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
defm : X86WriteResPairUnsupported<WriteFLogicZ>;
def : WriteRes<WriteFMA, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFMALd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFMAX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFMAXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFMAY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFMAYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteFMAZ>;
def : WriteRes<WriteFMOVMSK, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}

defm : X86WriteRes<WriteFMaskedLoad, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 8, [1, 1], 2>;
defm : X86WriteRes<WriteFMaskedLoadY, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 9, [1, 1], 2>;
defm : X86WriteRes<WriteFMaskedStore32, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore32Y, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore64, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore64Y, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMoveX, [], 1, [], 0>;
defm : X86WriteRes<WriteFMoveY, [], 1, [], 0>;
defm : X86WriteResUnsupported<WriteFMoveZ>;
def : WriteRes<WriteFMul, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMulLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
}
def : WriteRes<WriteFMul64, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMul64Ld, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
}
def : WriteRes<WriteFMul64X, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMul64XLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
}
def : WriteRes<WriteFMul64Y, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMul64YLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
}
defm : X86WriteResPairUnsupported<WriteFMul64Z>;
def : WriteRes<WriteFMulX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMulXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
}
def : WriteRes<WriteFMulY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteFMulYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
}
defm : X86WriteResPairUnsupported<WriteFMulZ>;
def : WriteRes<WriteFRcp, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFRcpLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFRcpX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFRcpXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteFRcpY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteFRcpYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteFRcpZ>;
defm : X86WriteRes<WriteFRnd, [LNLPVPort00_01], 8, [8], 2>;
defm : X86WriteRes<WriteFRndLd, [LNLPVPort00_01, LNLPPort20_21_22], 14, [8, 6], 3>;
defm : X86WriteRes<WriteFRndY, [LNLPVPort00_01], 8, [8], 2>;
defm : X86WriteRes<WriteFRndYLd, [LNLPVPort00_01, LNLPPort20_21_22], 15, [8, 7], 3>;
defm : X86WriteResPairUnsupported<WriteFRndZ>;
def  : WriteRes<WriteFRsqrt, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def  : WriteRes<WriteFRsqrtLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def  : WriteRes<WriteFRsqrtX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def  : WriteRes<WriteFRsqrtXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def  : WriteRes<WriteFRsqrtY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def  : WriteRes<WriteFRsqrtYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteFRsqrtZ>;

defm : LNLPWriteResPair<WriteFShuffle, [LNLPVPort02_03], 1, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFShuffle256, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : LNLPWriteResPair<WriteFShuffleY, [LNLPVPort02_03], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFShuffleZ>;
def : WriteRes<WriteFSign, [LNLPVPort00_01]>;
def : WriteRes<WriteFSqrt, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFSqrtLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 6];
  let Latency = 16;
}
def : WriteRes<WriteFSqrt64, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [15];
  let Latency = 15;
}
def : WriteRes<WriteFSqrt64Ld, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [15, 6];
  let Latency = 21;
}
def : WriteRes<WriteFSqrt64X, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [15];
  let Latency = 15;
}
def : WriteRes<WriteFSqrt64XLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [15, 6];
  let Latency = 21;
}
def : WriteRes<WriteFSqrt64Y, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [15];
  let Latency = 15;
}
def : WriteRes<WriteFSqrt64YLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [15, 7];
  let Latency = 22;
}
defm : X86WriteResPairUnsupported<WriteFSqrt64Z>;
def : WriteRes<WriteFSqrt80, [LNLPPortInvalid, LNLPVPort00_01]> {
  let ReleaseAtCycles = [7, 1];
  let Latency = 21;
}
def : WriteRes<WriteFSqrtX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFSqrtXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 6];
  let Latency = 16;
}
def : WriteRes<WriteFSqrtY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [10];
  let Latency = 10;
}
def : WriteRes<WriteFSqrtYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [10, 7];
  let Latency = 17;
}
defm : X86WriteResPairUnsupported<WriteFSqrtZ>;
defm : X86WriteRes<WriteFStore, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : X86WriteResUnsupported<WriteFStoreNT>;
defm : X86WriteRes<WriteFStoreNTX, [LNLPPort10_11, LNLPPort25_26_27], 518, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreNTY, [LNLPPort10_11, LNLPPort25_26_27], 542, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreX, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreY, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : LNLPWriteResPair<WriteFTest, [LNLPVPort00_01], 3, [1]>;
defm : LNLPWriteResPair<WriteFTestY, [LNLPVPort00_01], 5, [1], 1, 6>;
defm : LNLPWriteResPair<WriteFVarBlend, [LNLPVPort00_01_02_03], 1, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFVarBlendY, [LNLPVPort00_01_02_03], 3, [3], 3, 7>;
defm : X86WriteResPairUnsupported<WriteFVarBlendZ>;
defm : LNLPWriteResPair<WriteFVarShuffle, [LNLPVPort02_03], 1, [1], 1, 7>;
defm : LNLPWriteResPair<WriteFVarShuffle256, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : LNLPWriteResPair<WriteFVarShuffleY, [LNLPVPort02_03], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFVarShuffleZ>;
defm : X86WriteRes<WriteFence, [LNLPPort10_11, LNLPPort25_26_27], 2, [1, 1], 2>;
defm : LNLPWriteResPair<WriteIDiv16, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05], 16, [1, 3], 4, 4>;
defm : LNLPWriteResPair<WriteIDiv32, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05], 15, [1, 3], 4, 4>;
defm : LNLPWriteResPair<WriteIDiv64, [LNLPPort01_03_05], 18, [3], 3>;
defm : X86WriteRes<WriteIDiv8, [LNLPPort01_03_05], 17, [3], 3>;
defm : X86WriteRes<WriteIDiv8Ld, [LNLPPort01_03_05], 22, [3], 3>;
defm : LNLPWriteResPair<WriteIMul16, [LNLPPort01_03_05, LNLPPort00_02_04], 5, [2, 1], 4>;
defm : LNLPWriteResPair<WriteIMul16Imm, [LNLPPort01_03_05, LNLPPort00_01_02_03_04_05], 4, [1, 1], 2>;
defm : LNLPWriteResPair<WriteIMul16Reg, [LNLPPort01_03_05], 3, [1]>;
defm : LNLPWriteResPair<WriteIMul32, [LNLPPort01_03_05, LNLPPort00_02_04], 4, [1, 1], 3>;
defm : LNLPWriteResPair<WriteIMul32Imm, [LNLPPort01_03_05, LNLPPort00_01_02_03_04_05], 2, [1, 1]>;
defm : LNLPWriteResPair<WriteIMul32Reg, [LNLPPort01_03_05], 3, [1]>;
defm : LNLPWriteResPair<WriteIMul64, [LNLPPort01_03_05, LNLPPort00_02_04], 4, [1, 1], 2>;
defm : LNLPWriteResPair<WriteIMul64Imm, [LNLPPort01_03_05, LNLPPort00_01_02_03_04_05], 2, [1, 1]>;
defm : LNLPWriteResPair<WriteIMul64Reg, [LNLPPort01_03_05], 3, [1]>;
defm : LNLPWriteResPair<WriteIMul8, [LNLPPort01_03_05], 3, [1]>;
def : WriteRes<WriteIMulH, []> {
  let Latency = 3;  // 4
}
def : WriteRes<WriteIMulHLd, []> {
  let Latency = 3;
}
def  : WriteRes<WriteJump, [LNLPPort00_02_04]>;
defm : X86WriteRes<WriteJumpLd, [LNLPPort00_02_04, LNLPPort20_21_22], 6, [1, 1], 2>;
defm : X86WriteResUnsupported<WriteLAHFSAHF>;
defm : X86WriteRes<WriteLDMXCSR, [LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPVPort02_03, LNLPPort25_26_27], 7, [1, 1, 1, 1], 4>;
def  : WriteRes<WriteLEA, [LNLPPort01]>;
defm : LNLPWriteResPair<WriteLZCNT, [LNLPPort01], 3, [1]>;
def  : WriteRes<WriteLoad, [LNLPPort20_21_22]> {
  let Latency = 4;
}
def  : WriteRes<WriteMMXMOVMSK, [LNLPVPort00_01]> {
  let Latency = 3 ;
}
defm : LNLPWriteResPair<WriteMPSAD, [LNLPVPort02_03, LNLPVPort02_03], 4, [1, 1], 2, 7>;
defm : LNLPWriteResPair<WriteMPSADY, [LNLPVPort02_03, LNLPVPort02_03], 4, [1, 1], 2, 8>;
defm : LNLPWriteResPair<WriteMULX32, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05], 4, [1, 1, 1], 2>;
defm : LNLPWriteResPair<WriteMULX64, [LNLPPort01_03_05, LNLPPort01_03_05], 4, [1, 1]>;
// FIXME: Incompleted schedwrite.
def  : WriteRes<WriteMicrocoded, [LNLPPort00_01_02_03_04_05]> {
  let Latency = LunarlakePModel.MaxLatency;
}
def  : WriteRes<WriteMove, [LNLPPort00_01_02_03_04_05]>;
defm : X86WriteRes<WriteNop, [], 1, [], 0>;
defm : X86WriteRes<WritePCmpEStrI, [LNLPPort00, LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort01_03_05], 16, [3, 2, 1, 1, 1], 8>;
defm : X86WriteRes<WritePCmpEStrILd, [LNLPPort00, LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort01_03_05], 31, [3, 1, 1, 1, 1, 1], 8>;
defm : X86WriteRes<WritePCmpEStrM, [LNLPPort00, LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05, LNLPPort01_03_05], 16, [3, 3, 1, 1, 1], 9>;
defm : X86WriteRes<WritePCmpEStrMLd, [LNLPPort00, LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort01_03_05], 17, [3, 2, 1, 1, 1, 1], 9>;
defm : LNLPWriteResPair<WritePCmpIStrI, [LNLPPort00], 11, [3], 3, 20>;
defm : LNLPWriteResPair<WritePCmpIStrM, [LNLPPort00], 11, [3], 3>;
defm : LNLPWriteResPair<WritePHAdd, [LNLPVPort00_01_02_03, LNLPVPort02_03], 3, [1, 2], 3, 8>;
defm : LNLPWriteResPair<WritePHAddX, [LNLPVPort00_01_02_03, LNLPVPort02_03], 2, [1, 2], 3, 7>;
defm : LNLPWriteResPair<WritePHAddY, [LNLPVPort00_01_02_03, LNLPVPort02_03], 2, [1, 2], 3, 8>;
def : WriteRes<WritePHMINPOS, [LNLPVPort00_01]> {
  let Latency = 5;
}
def : WriteRes<WritePHMINPOSLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let Latency = 11;
}

// FIXME : uops info is incorrect 
defm : LNLPWriteResPair<WritePMULLD, [LNLPVPort00_01_02_03], 10, [2], 2, 8>;
defm : LNLPWriteResPair<WritePMULLDY, [LNLPVPort00_01_02_03], 10, [2], 2, 8>; 
defm : X86WriteResPairUnsupported<WritePMULLDZ>;
defm : LNLPWriteResPair<WritePOPCNT, [LNLPPort01_03_05], 3, [1]>;
defm : LNLPWriteResPair<WritePSADBW, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : LNLPWriteResPair<WritePSADBWX, [LNLPVPort02_03], 3, [1], 1, 7>;
defm : LNLPWriteResPair<WritePSADBWY, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WritePSADBWZ>;
defm : X86WriteRes<WriteRMW, [LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27], 1, [1, 1, 1], 3>;
defm : X86WriteRes<WriteRotate, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05], 2, [1, 2], 3>;
defm : X86WriteRes<WriteRotateLd, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05], 12, [1, 2], 3>;
defm : X86WriteRes<WriteRotateCL, [LNLPPort01_03_05], 2, [2], 2>;
defm : X86WriteRes<WriteRotateCLLd, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05], 19, [2, 3, 2], 7>;
defm : X86WriteRes<WriteSETCC, [LNLPPort01_03_05], 2, [2], 2>;
defm : X86WriteRes<WriteSETCCStore, [LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27], 13, [2, 1, 1], 4>;
defm : X86WriteRes<WriteSHDmrcl, [LNLPPort00_01_02_03_04_05,LNLPPort01_03_05 , LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1, 1, 1, 1, 1], 6>;
defm : X86WriteRes<WriteSHDmri, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1, 1, 1, 1], 5>;
defm : X86WriteRes<WriteSHDrrcl, [LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort01_03_05], 5, [1, 1, 1], 3>;
def  : WriteRes<WriteSHDrri, [LNLPPort01_03_05]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteSTMXCSR, [LNLPPort00_02_04, LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], LunarlakePModel.MaxLatency, [1, 6, 1, 1], 4>;
def  : WriteRes<WriteShift, [LNLPPort01_03_05]>;
defm : X86WriteRes<WriteShiftLd, [LNLPPort01_03_05, LNLPPort20_21_22], 5, [1, 4], 2>;
defm : X86WriteRes<WriteShiftCL, [LNLPPort00_02_04], 2, [2], 2>;
defm : X86WriteRes<WriteShiftCLLd, [LNLPPort00_02_04], 12, [2], 2>;
defm : LNLPWriteResPair<WriteShuffle, [LNLPVPort02_03], 1, [1], 1, 8>;
defm : LNLPWriteResPair<WriteShuffle256, [LNLPVPort02_03], 3, [1], 1, 8>;
defm : LNLPWriteResPair<WriteShuffleX, [LNLPVPort02_03], 1, [1], 1, 7>;
defm : LNLPWriteResPair<WriteShuffleY, [LNLPVPort02_03], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteShuffleZ>;
defm : X86WriteRes<WriteStore, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : X86WriteRes<WriteStoreNT, [LNLPPort10_11, LNLPPort25_26_27], 512, [1, 1], 2>;
def : WriteRes<WriteSystem, [LNLPPort00_01_02_03_04_05]> {
  let Latency = LunarlakePModel.MaxLatency;
}
def : WriteRes<WriteTZCNT, [LNLPPort01_03_05]> {
  let Latency = 3;
}
def : WriteRes<WriteTZCNTLd, [LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 4];
  let Latency = 7;
}
def : WriteRes<WriteVPMOV256, [LNLPVPort02_03]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteVPMOV256Ld, [LNLPVPort02_03, LNLPPort20_21_22], 10, [3, 7], 2>;
defm : X86WriteRes<WriteVarBlend, [LNLPVPort00_01_02_03], 3, [3], 3>;
defm : X86WriteRes<WriteVarBlendLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 8, [3, 6], 4>;
defm : X86WriteRes<WriteVarBlendY, [LNLPVPort00_01_02_03], 3, [3], 3>;
defm : X86WriteRes<WriteVarBlendYLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 9, [3, 7], 4>;
defm : X86WriteResPairUnsupported<WriteVarBlendZ>;
defm : LNLPWriteResPair<WriteVarShuffle, [LNLPVPort02_03, LNLPVPort00_01], 3, [1, 1], 2, 8>;
def : WriteRes<WriteVarShuffle256, [LNLPVPort02_03]> {
  let Latency = 3;  
}
def : WriteRes<WriteVarShuffle256Ld, [LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
}
def : WriteRes<WriteVarShuffleX, [LNLPVPort02_03]>;
defm : X86WriteRes<WriteVarShuffleXLd, [LNLPVPort02_03, LNLPPort20_21_22], 7, [1, 6], 2>;
def : WriteRes<WriteVarShuffleY, [LNLPVPort02_03]>;
defm : X86WriteRes<WriteVarShuffleYLd, [LNLPVPort02_03, LNLPPort20_21_22], 8, [1, 7], 2>;
defm : X86WriteResPairUnsupported<WriteVarShuffleZ>;

def : WriteRes<WriteVarVecShift, [LNLPVPort00_01]>;
def : WriteRes<WriteVarVecShiftLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
}
def : WriteRes<WriteVarVecShiftY, [LNLPVPort00_01]>;
def : WriteRes<WriteVarVecShiftYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
defm : X86WriteResPairUnsupported<WriteVarVecShiftZ>;
defm : LNLPWriteResPair<WriteVecALU, [LNLPVPort00], 1, [1], 1, 8>;  // 4 ports ?
def : WriteRes<WriteVecALUX, [LNLPVPort00_01]>;
def : WriteRes<WriteVecALUXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
}
def : WriteRes<WriteVecALUY, [LNLPVPort00_01]>;
def : WriteRes<WriteVecALUYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
defm : X86WriteResPairUnsupported<WriteVecALUZ>;
defm : X86WriteRes<WriteVecExtract, [LNLPVPort00_01, LNLPVPort02_03], 4, [3, 1], 2>;
defm : X86WriteRes<WriteVecExtractSt, [LNLPVPort02_03, LNLPPort10_11, LNLPPort25_26_27], 2, [1, 1, 1], 2>;
defm : LNLPWriteResPair<WriteVecIMul, [LNLPVPort00_01], 5, [1], 1, 8>;
def : WriteRes<WriteVecIMulX, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteVecIMulXLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : WriteRes<WriteVecIMulY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
}
def : WriteRes<WriteVecIMulYLd, [LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
}
defm : X86WriteResPairUnsupported<WriteVecIMulZ>;
defm : X86WriteRes<WriteVecInsert, [LNLPPort01_03_05, LNLPVPort02_03], 5, [4, 1], 2>;
defm : X86WriteRes<WriteVecInsertLd, [LNLPVPort02_03, LNLPPort20_21_22], 7, [1, 6], 2>;
def  : WriteRes<WriteVecLoad, [LNLPPort20_21_22]> {
  let Latency = 6;
}
// FIXME: Incompleted schedwrite.
def : WriteRes<WriteVecLoadNT, [LNLPPort20_21_22]> {
  let Latency = 7;
}
// FIXME: Incompleted schedwrite.
def : WriteRes<WriteVecLoadNTY, [LNLPPort20_21_22]> {
  let Latency = 8;
}

def : WriteRes<WriteVecLoadX, [LNLPPort20_21_22]> {
  let Latency = 6;
}
def : WriteRes<WriteVecLoadY, [LNLPPort20_21_22]> {
  let Latency = 7;
}
defm : LNLPWriteResPair<WriteVecLogic, [LNLPVPort00_01_02_03], 1, [1], 1, 8>;
def : WriteRes<WriteVecLogicX, [LNLPVPort00_01_02_03]>;
def : WriteRes<WriteVecLogicXLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
}
def : WriteRes<WriteVecLogicY, [LNLPVPort00_01_02_03]>;
def : WriteRes<WriteVecLogicYLd, [LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
defm : X86WriteResPairUnsupported<WriteVecLogicZ>;
def : WriteRes<WriteVecMOVMSK, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : WriteRes<WriteVecMOVMSKY, [LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;  // Tool added Max
}
defm : X86WriteRes<WriteVecMaskedGatherWriteback, [], 5, [], 0>;
defm : X86WriteRes<WriteVecMaskedLoad, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 7, [1, 6], 2>;
defm : X86WriteRes<WriteVecMaskedLoadY, [LNLPVPort00_01_02_03, LNLPPort20_21_22], 8, [1, 7], 2>;
defm : X86WriteRes<WriteVecMaskedStore32, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;  // updated lat from 3 to 14
defm : X86WriteRes<WriteVecMaskedStore32Y, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteVecMaskedStore64, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteVecMaskedStore64Y, [LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27], 14, [1, 1, 1], 3>;
def : WriteRes<WriteVecMove, [LNLPVPort00_01_02_03]>;
def : WriteRes<WriteVecMoveFromGpr, [LNLPPort01_03_05]> {
  let Latency = 3;  // Originally 4
}
def : WriteRes<WriteVecMoveToGpr, [LNLPVPort00_01]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteVecMoveX, [LNLPVPort00_01_02_03], 1, [2], 2>;
defm : X86WriteRes<WriteVecMoveY, [], 1, [], 0>;
defm : X86WriteResUnsupported<WriteVecMoveZ>;
defm : LNLPWriteResPair<WriteVecShift, [LNLPVPort00_01], 1, [1], 1, 8>;
def  : WriteRes<WriteVecShiftImm, [LNLPVPort00_01]>;
def  : WriteRes<WriteVecShiftImmX, [LNLPVPort00_01]>;
defm : X86WriteResUnsupported<WriteVecShiftImmXLd>;
def  : WriteRes<WriteVecShiftImmY, [LNLPVPort00_01]>;
defm : X86WriteResUnsupported<WriteVecShiftImmYLd>;
defm : X86WriteResPairUnsupported<WriteVecShiftImmZ>;
defm : X86WriteRes<WriteVecShiftX, [LNLPVPort00_01, LNLPVPort02_03], 2, [1, 1], 2>;
defm : X86WriteRes<WriteVecShiftXLd, [LNLPVPort00_01, LNLPPort20_21_22], 8, [1, 1], 2>; // 7
defm : X86WriteRes<WriteVecShiftY, [LNLPVPort00_01, LNLPVPort02_03], 4, [1, 1], 2>;
defm : X86WriteRes<WriteVecShiftYLd, [LNLPVPort00_01, LNLPPort20_21_22], 9, [1, 7], 2>; // 8
defm : X86WriteResPairUnsupported<WriteVecShiftZ>;
defm : X86WriteRes<WriteVecStore, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>; 
defm : X86WriteRes<WriteVecStoreNT, [LNLPPort10_11, LNLPPort25_26_27], 511, [1, 1], 2>;  // historic value
defm : X86WriteRes<WriteVecStoreNTY, [LNLPPort10_11, LNLPPort25_26_27], 507, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreX, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreY, [LNLPPort10_11, LNLPPort25_26_27], 12, [1, 1], 2>;
defm : X86WriteRes<WriteVecTest, [LNLPVPort00_01, LNLPVPort02_03], 4, [1, 1], 2>;
defm : X86WriteRes<WriteVecTestLd, [LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22], 10, [3, 1, 6], 3>;
defm : X86WriteRes<WriteVecTestY, [LNLPVPort00_01, LNLPVPort02_03], LunarlakePModel.MaxLatency, [3, 1], 2>;
defm : X86WriteRes<WriteVecTestYLd, [LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22], 11, [3, 1, 6], 3>;
defm : X86WriteRes<WriteXCHG, [LNLPPort00_01_02_03_04_05], 2, [3], 3>;
def : WriteRes<WriteZero, []>;

// Manual Regressive SchedWriteRes and InstRW Definition. Suffix with "_X"
// All _X(N) prefix sequence are defs used from prev. generation to bypass incomplete data.
def LNLPWriteResGroupX0 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22, LNLPPort20_21_22, LNLPPort10_11]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX0], (instregex "^AA(D|N)D64mr$",
                                              "^A(X?)OR64mr$")>;
def LNLPWriteResGroupX1 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 1, 1, 1, 1];
  let Latency = 12;
  let NumMicroOps = 6;
}
def : InstRW<[LNLPWriteResGroupX1, ReadAfterLd, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(ADC|SBB)(16|32|64)mr$")>;
def LNLPWriteResGroupX2 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX2], (instregex "^JMP(16|32|64)m((_NT)?)$",
                                              "^RET(16|32)$",
                                              "^RORX(32|64)mi$")>;
def : InstRW<[LNLPWriteResGroupX2, ReadAfterLd, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(ADC|SBB)(8|16|32|64)rm$",
                                                                                                                                         "^AD(C|O)X(32|64)rm$")>;
def LNLPWriteResGroupX5 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX5], (instregex "^CMP(8|16|32)mi$",
                                              "^CMP(8|16|32|64)mi8$",
                                              "^MOV(8|16)rm$",
                                              "^POP(16|32)r((mr)?)$")>;
def : InstRW<[LNLPWriteResGroupX5], (instrs CMP64mi32,
                                           MOV8rm_NOREX,
                                           MOVZX16rm8)>;
def : InstRW<[LNLPWriteResGroupX5, ReadAfterLd], (instregex "^(ADD|CMP|SUB)(8|16|32|64)rm$",
                                                           "^AND(8|16|32)rm$",
                                                           "^(X?)OR(8|16|32)rm$")>;
def : InstRW<[LNLPWriteResGroupX5, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^CMP(8|16|32|64)mr$")>;
def LNLPWriteResGroupX6 : SchedWriteRes<[]> {
  let NumMicroOps = 0;
}
def : InstRW<[LNLPWriteResGroupX6], (instregex "^(ADD|SUB)64ri8$",
                                              "^(DE|IN)C64r$",
                                              "^MOV64rr((_REV)?)$")>;
def : InstRW<[LNLPWriteResGroupX6], (instrs CLC,
                                           JMP_2)>;
def LNLPWriteResGroupX7 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 13;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX7], (instregex "^A(D|N)D8mi(8?)$",
                                              "^(DE|IN)C8m$",
                                              "^N(EG|OT)8m$",
                                              "^(X?)OR8mi(8?)$",
                                              "^SUB8mi(8?)$")>;
def : InstRW<[LNLPWriteResGroupX7, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^A(D|N)D8mr$",
                                                                                                                            "^(X?)OR8mr$")>;
def : InstRW<[LNLPWriteResGroupX7, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instrs SUB8mr)>;
def LNLPWriteResGroupX8 : SchedWriteRes<[LNLPPort01_03_05]> {
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroupX8], (instregex "^(V?)(ADD|SUB)SSrr((_Int)?)$")>;
def LNLPWriteResGroupX9 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort01_03_05]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX9], (instregex "^ADD_F(32|64)m$",
                                              "^ILD_F(16|32|64)m$",
                                              "^SUB(R?)_F(32|64)m$")>;
def LNLPWriteResGroupX10 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX10], (instregex "^ADD_FI(16|32)m$",
                                               "^SUB(R?)_FI(16|32)m$")>;

def LNLPWriteResGroupX11 : SchedWriteRes<[LNLPPort00_01_02_03_04_05]> {
  let Latency = 2;
}
def : InstRW<[LNLPWriteResGroupX11], (instregex "^AND(8|16|32|64)r(r|i8)$",
                                               "^AND(8|16|32|64)rr_REV$",
                                               "^(AND|TEST)(32|64)i32$",
                                               "^(AND|TEST)(8|32)ri$",
                                               "^(AND|TEST)64ri32$",
                                               "^(AND|TEST)8i8$",
                                               "^(X?)OR(8|16|32|64)r(r|i8)$",
                                               "^(X?)OR(8|16|32|64)rr_REV$",
                                               "^(X?)OR(32|64)i32$",
                                               "^(X?)OR(8|32)ri$",
                                               "^(X?)OR64ri32$",
                                               "^(X?)OR8i8$",
                                               "^TEST(8|16|32|64)rr$")>;
def : InstRW<[LNLPWriteResGroupX11], (instrs XOR8rr_NOREX)>;
def LNLPWriteResGroupX12 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX12], (instregex "^TEST(8|16|32)mi$")>;
def : InstRW<[LNLPWriteResGroupX12], (instrs TEST64mi32)>;
def : InstRW<[LNLPWriteResGroupX12, ReadAfterLd], (instregex "^(X?)OR64rm$")>;
def : InstRW<[LNLPWriteResGroupX12, ReadAfterLd], (instrs AND64rm)>;
def : InstRW<[LNLPWriteResGroupX12, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^TEST(8|16|32|64)mr$")>;
def LNLPWriteResGroupX13 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort20_21_22]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX13, ReadAfterLd], (instregex "^ANDN(32|64)rm$")>;
def LNLPWriteResGroupX14 : SchedWriteRes<[LNLPPort01_03_05]> {
  let Latency = 2;
}
def : InstRW<[LNLPWriteResGroupX14], (instregex "^ANDN(32|64)rr$")>;
def LNLPWriteResGroupX15 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [5, 2, 1, 1];
  let Latency = 10;
  let NumMicroOps = 9;
}
def : InstRW<[LNLPWriteResGroupX15], (instrs BT64mr)>;
def LNLPWriteResGroupX16 : SchedWriteRes<[LNLPPort01]> {
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroupX16], (instregex "^BT((C|R|S)?)64rr$")>;
def LNLPWriteResGroupX17 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [4, 2, 1, 1, 1, 1];
  let Latency = 17;
  let NumMicroOps = 10;
}
def : InstRW<[LNLPWriteResGroupX17], (instregex "^BT(C|R|S)64mr$")>;
def LNLPWriteResGroupX18 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 7;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX18], (instregex "^CALL(16|32|64)m((_NT)?)$")>;

def LNLPWriteResGroupX19 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX19], (instregex "^CALL(16|32|64)r((_NT)?)$")>;

def LNLPWriteResGroupX20 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX20], (instrs CALL64pcrel32,
                                            MFENCE)>;
def LNLPWriteResGroupX21 : SchedWriteRes<[LNLPVPort02_03]>;
def : InstRW<[LNLPWriteResGroupX21], (instregex "^C(DQ|WD)E$",
                                               "^(V?)MOVS(H|L)DUPrr$",
                                               "^(V?)SHUFP(D|S)rri$",
                                               "^VMOVS(H|L)DUPYrr$",
                                               "^VSHUFP(D|S)Yrri$")>;
def : InstRW<[LNLPWriteResGroupX21], (instrs CBW)>;

def LNLPWriteResGroupX22 : SchedWriteRes<[LNLPPort00_02_04]>;
def : InstRW<[LNLPWriteResGroupX22], (instregex "^C(DQ|QO)$",
                                               "^(CL|ST)AC$")>;
def LNLPWriteResGroupX23 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX23], (instrs CLD)>;

def LNLPWriteResGroupX24 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX24], (instrs CLDEMOTE)>;

def LNLPWriteResGroupX25 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 2;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX25], (instrs CLFLUSH)>;

def LNLPWriteResGroupX26 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX26], (instrs CLFLUSHOPT)>;

def LNLPWriteResGroupX27 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort01]> {
  let ReleaseAtCycles = [2, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX27], (instrs CLI)>;

def LNLPWriteResGroupX28 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort01, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [6, 1, 3];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 10;
}
def : InstRW<[LNLPWriteResGroupX28], (instrs CLTS)>;

def LNLPWriteResGroupX29 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 5;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX29], (instregex "^MOV16o(16|32|64)a$")>;
def : InstRW<[LNLPWriteResGroupX29], (instrs CLWB)>;

def LNLPWriteResGroupX30 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [5, 2];
  let Latency = 6;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX30], (instregex "^CMPS(B|L|Q|W)$")>;

def LNLPWriteResGroupX31 : SchedWriteRes<[LNLPPort00, LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 7, 6, 2, 1, 1, 2, 1];
  let Latency = 32;
  let NumMicroOps = 22;
}
def : InstRW<[LNLPWriteResGroupX31], (instrs CMPXCHG16B)>;

def LNLPWriteResGroupX32 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [4, 7, 2, 1, 1, 1];
  let Latency = 25;
  let NumMicroOps = 16;
}
def : InstRW<[LNLPWriteResGroupX32], (instrs CMPXCHG8B)>;

def LNLPWriteResGroupX33 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [1, 2, 1, 1, 1];
  let Latency = 13;
  let NumMicroOps = 6;
}
def : InstRW<[LNLPWriteResGroupX33], (instrs CMPXCHG8rm)>;

def LNLPWriteResGroupX34 : SchedWriteRes<[LNLPPort00, LNLPVPort00_01, LNLPPort00_02_04, LNLPPort01, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 1, 10, 6, 1, 5, 1];
  let Latency = 18;
  let NumMicroOps = 26;
}
def : InstRW<[LNLPWriteResGroupX34], (instrs CPUID)>;

def LNLPWriteResGroupX35 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01, LNLPPort20_21_22]> {
  let Latency = 26;
  let NumMicroOps = 3;
}
def LNLPWriteResGroupX36 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22, LNLPPort01_03_05]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX36], (instrs CVTSI642SSrm)>;
def : InstRW<[LNLPWriteResGroupX36, ReadAfterVecLd], (instrs VCVTSI642SSrm)>;
def LNLPWriteResGroupX37 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX37, ReadInt2Fpu], (instrs CVTSI642SSrr)>;
def : InstRW<[LNLPWriteResGroupX37, ReadDefault, ReadInt2Fpu], (instrs VCVTSI642SSrr)>;
def LNLPWriteResGroupX38 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01, LNLPPort01_03_05]> {
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX38, ReadDefault], (instregex "^(V?)CVT(T?)SS2SI64rr$")>;
def LNLPWriteResGroupX39 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX39], (instregex "^J(E|R)CXZ$")>;
def : InstRW<[LNLPWriteResGroupX39], (instrs CWD)>;

def LNLPWriteResGroupX40 : SchedWriteRes<[LNLPPort00_01_02_03_04_05]>;
def : InstRW<[LNLPWriteResGroupX40], (instregex "^(LD|ST)_Frr$",
                                               "^MOV16s(m|r)$",
                                               "^MOV(32|64)sr$")>;
def : InstRW<[LNLPWriteResGroupX40], (instrs DEC16r_alt,
                                            SALC,
                                            ST_FPrr,
                                            SYSCALL)>;

def LNLPWriteResGroupX41 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 7;
}
def : InstRW<[LNLPWriteResGroupX41], (instrs DEC32r_alt)>;
def LNLPWriteResGroupX42 : SchedWriteRes<[LNLPPort00, LNLPPort20_21_22]> {
  let Latency = 27;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX42], (instregex "^DIVR_F(32|64)m$")>;
def LNLPWriteResGroupX43 : SchedWriteRes<[LNLPPort00, LNLPPort20_21_22, LNLPPort01_03_05]> {
  let Latency = 30;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX43], (instregex "^DIVR_FI(16|32)m$")>;

def LNLPWriteResGroupX44 : SchedWriteRes<[LNLPPort00]> {
  let Latency = 15;
}
def : InstRW<[LNLPWriteResGroupX44], (instregex "^DIVR_F(P?)rST0$")>;
def : InstRW<[LNLPWriteResGroupX44], (instrs DIVR_FST0r)>;
def LNLPWriteResGroupX45 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let Latency = 20;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX45, ReadAfterVecLd], (instregex "^(V?)DIVSDrm_Int$")>;
def LNLPWriteResGroupX46 : SchedWriteRes<[LNLPPort00, LNLPPort20_21_22]> {
  let Latency = 22;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX46], (instregex "^DIV_F(32|64)m$")>;
def LNLPWriteResGroupX47 : SchedWriteRes<[LNLPPort00, LNLPPort20_21_22, LNLPPort01_03_05]> {
  let Latency = 25;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX47], (instregex "^DIV_FI(16|32)m$")>;

def LNLPWriteResGroupX48 : SchedWriteRes<[LNLPPort00]> {
  let Latency = 20;
}
def : InstRW<[LNLPWriteResGroupX48], (instregex "^DIV_F(P?)rST0$")>;
def : InstRW<[LNLPWriteResGroupX48], (instrs DIV_FST0r)>;

def LNLPWriteResGroupX49 : SchedWriteRes<[LNLPPort00, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 21, 2, 14, 4, 9, 5];
  let Latency = 126;
  let NumMicroOps = 57;
}
def : InstRW<[LNLPWriteResGroupX49], (instrs ENTER)>;

def LNLPWriteResGroupX50 : SchedWriteRes<[LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX50], (instregex "^(V?)EXTRACTPSmri$")>;
def : InstRW<[LNLPWriteResGroupX50], (instrs SMSW16m)>;

def LNLPWriteResGroupX51 : SchedWriteRes<[LNLPVPort02_03, LNLPPort01_03_05]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX51], (instregex "^(V?)EXTRACTPSrri$")>;
def : InstRW<[LNLPWriteResGroupX51], (instrs MMX_PEXTRWrri)>;

def LNLPWriteResGroupX52 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22, LNLPPort20_21_22, LNLPPort10_11, LNLPVPort02_03]> {
  let Latency = 7;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX52], (instrs FARCALL64m)>;

def LNLPWriteResGroupX53 : SchedWriteRes<[LNLPPort20_21_22, LNLPVPort02_03]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX53], (instrs FARJMP64m,
                                            JMP64m_REX)>;

def LNLPWriteResGroupX54 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort10_11]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX54], (instregex "^(V?)MASKMOVDQU((64)?)$",
                                               "^ST_FP(32|64|80)m$")>;
def : InstRW<[LNLPWriteResGroupX54], (instrs FBSTPm,
                                            VMPTRSTm)>;

def LNLPWriteResGroupX55 : SchedWriteRes<[LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2];
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX55], (instrs FDECSTP)>;

def LNLPWriteResGroupX56 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 11;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX56], (instregex "^FICOM(P?)(16|32)m$")>;

def LNLPWriteResGroupX57 : SchedWriteRes<[LNLPPort01_03_05]>;
def : InstRW<[LNLPWriteResGroupX57], (instregex "^MMX_P(ADD|SUB)(B|D|Q|W)rr$")>;
def : InstRW<[LNLPWriteResGroupX57], (instrs FINCSTP,
                                            FNOP)>;

def LNLPWriteResGroupX58 : SchedWriteRes<[LNLPVPort02_03, LNLPPort01_03_05, LNLPPort20_21_22]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX58], (instrs FLDCW16m)>;

def LNLPWriteResGroupX59 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort00_02_04, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 39, 5, 10, 8];
  let Latency = 62;
  let NumMicroOps = 64;
}
def : InstRW<[LNLPWriteResGroupX59], (instrs FLDENVm)>;

def LNLPWriteResGroupX60 : SchedWriteRes<[LNLPVPort00_01_02_03]> {
  let ReleaseAtCycles = [4];
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX60], (instrs FNCLEX)>;

def LNLPWriteResGroupX61 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [6, 3, 6];
  let Latency = 75;
  let NumMicroOps = 15;
}
def : InstRW<[LNLPWriteResGroupX61], (instrs FNINIT)>;

def LNLPWriteResGroupX62 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort10_11, LNLPVPort02_03]> {
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX62], (instrs FNSTCW16m)>;

def LNLPWriteResGroupX63 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01_02_03]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX63], (instrs FNSTSW16r)>;

def LNLPWriteResGroupX64 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX64], (instrs FNSTSWm)>;

def LNLPWriteResGroupX65 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPVPort02_03]> {
  let ReleaseAtCycles = [9, 30, 21, 1, 11, 11, 16, 1];
  let Latency = 106;
  let NumMicroOps = 100;
}
def : InstRW<[LNLPWriteResGroupX65], (instrs FSTENVm)>;

def LNLPWriteResGroupX66 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort00_02_04, LNLPVPort02_03, LNLPPort20_21_22, LNLPVPort02_03]> {
  let ReleaseAtCycles = [4, 47, 1, 2, 1, 33, 2];
  let Latency = 63;
  let NumMicroOps = 90;
}
def : InstRW<[LNLPWriteResGroupX66], (instrs FXRSTOR)>;

def LNLPWriteResGroupX67 : SchedWriteRes<[LNLPVPort02_03, LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort00_02_04, LNLPVPort02_03, LNLPPort20_21_22, LNLPVPort02_03]> {
  let ReleaseAtCycles = [4, 45, 1, 2, 1, 31, 4];
  let Latency = 63;
  let NumMicroOps = 88;
}
def : InstRW<[LNLPWriteResGroupX67], (instrs FXRSTOR64)>;

def LNLPWriteResGroupX68 : SchedWriteRes<[LNLPVPort02_03, LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 5, 10, 10, 2, 38, 5, 38];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 110;
}
def : InstRW<[LNLPWriteResGroupX68], (instregex "^FXSAVE((64)?)$")>;

def LNLPWriteResGroupX69 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let Latency = 12;
  let NumMicroOps = 2;
}

def LNLPWriteResGroupX77 : SchedWriteRes<[LNLPPort00_02_04]> {
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX77], (instrs INC16r_alt)>;

def LNLPWriteResGroupX78 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 7;
}
def : InstRW<[LNLPWriteResGroupX78], (instrs INC32r_alt)>;

def LNLPWriteResGroupX85 : SchedWriteRes<[LNLPPort00_02_04]>;
def : InstRW<[LNLPWriteResGroupX85], (instrs JMP64r_REX)>;
def LNLPWriteResGroupX86 : SchedWriteRes<[]> {
  let Latency = 0;
  let NumMicroOps = 0;
}
def : InstRW<[LNLPWriteResGroupX86], (instregex "^JMP_(1|4)$")>;
def : InstRW<[LNLPWriteResGroupX86], (instrs VZEROUPPER)>;

def LNLPWriteResGroupX93 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX93], (instrs LEA16r)>;

def LNLPWriteResGroupX104 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [4, 6, 1];
  let Latency = 3;
  let NumMicroOps = 11;
}
def : InstRW<[LNLPWriteResGroupX104], (instrs LOOPE)>;

def LNLPWriteResGroupX105 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [4, 6, 1];
  let Latency = 2;
  let NumMicroOps = 11;
}
def : InstRW<[LNLPWriteResGroupX105], (instrs LOOPNE)>;

def LNLPWriteResGroupX115 : SchedWriteRes<[LNLPVPort00_01, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 1, 1];
  let Latency = 12;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX115], (instregex "^MMX_MASKMOVQ((64)?)$")>;
def LNLPWriteResGroupX118 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPVPort00_01]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX118], (instregex "^MMX_MOV(DQ|FR64)2Qrr$")>;
def LNLPWriteResGroupX122 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX122, ReadAfterVecLd], (instregex "^MMX_P(ADD|SUB)(B|D|Q|W)rm$")>;
def LNLPWriteResGroupX123 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 1, 2];
  let Latency = 11;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX123, ReadAfterVecLd], (instregex "^MMX_PH(ADD|SUB)SWrm$")>;
def LNLPWriteResGroupX124 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX124], (instregex "^MMX_PH(ADD|SUB)SWrr$")>;
def LNLPWriteResGroupX116 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 18;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX116], (instrs MMX_MOVD64mr)>;
def LNLPWriteResGroupX117 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 8;
}
def : InstRW<[LNLPWriteResGroupX117], (instregex "^MMX_MOV(D|Q)64rm$",
                                                "^VBROADCASTI128rm$")>;
def : InstRW<[LNLPWriteResGroupX117], (instrs MMX_MOVD64to64rm)>;
def LNLPWriteResGroupX120 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX120, ReadAfterVecLd], (instregex "^MMX_PACKSS(DW|WB)rm$")>;
def : InstRW<[LNLPWriteResGroupX120, ReadAfterVecLd], (instrs MMX_PACKUSWBrm)>;
def LNLPWriteResGroupX121 : SchedWriteRes<[LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2];
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX121], (instregex "^MMX_PACKSS(DW|WB)rr$")>;
def : InstRW<[LNLPWriteResGroupX121], (instrs MMX_PACKUSWBrr)>;
def : InstRW<[LNLPWriteResGroupX121, ReadDefault, ReadInt2Fpu], (instrs MMX_PINSRWrri)>;
def LNLPWriteResGroupX125 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort01_03_05]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX125, ReadAfterLd], (instrs MMX_PINSRWrmi)>;
def LNLPWriteResGroupX126 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX126], (instregex "^MOV16ao(16|32|64)$")>;
def LNLPWriteResGroupX127 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX127], (instregex "^PUSH(F|G)S(16|32)$")>;
def : InstRW<[LNLPWriteResGroupX127], (instrs MOV16ms,
                                             MOVBE32mr)>;
def LNLPWriteResGroupX128 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX128], (instregex "^MOV(16|32|64)rs$",
                                                "^S(TR|LDT)16r$")>;
def LNLPWriteResGroupX129 : SchedWriteRes<[LNLPPort20_21_22]>;
def : InstRW<[LNLPWriteResGroupX129], (instregex "^MOV32ao(16|32|64)$")>;
def : InstRW<[LNLPWriteResGroupX129], (instrs MOV64ao64)>;
def LNLPWriteResGroupX130 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX130], (instregex "^MOV(8|32)o(16|32)a$",
                                                "^MOV(8|32|64)o64a$")>;
def LNLPWriteResGroupX131 : SchedWriteRes<[LNLPPort00_01_02_03_04_05]> {
  let Latency = 0;
}
def : InstRW<[LNLPWriteResGroupX131], (instregex "^MOV32rr((_REV)?)$",
                                                "^MOVZX(32|64)rr8$")>;
def : InstRW<[LNLPWriteResGroupX131], (instrs MOVZX32rr8_NOREX)>;

def LNLPWriteResGroupX132 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 5;
}
def : InstRW<[LNLPWriteResGroupX132], (instrs MOV64ao32)>;
def LNLPWriteResGroupX134 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX134], (instrs MOV64o32a)>;
def LNLPWriteResGroupX135 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX135], (instrs MOV64rc)>;
def LNLPWriteResGroupX137 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX137], (instregex "^MOV8ao(16|32|64)$")>;
def LNLPWriteResGroupX138 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX138], (instregex "^MOV8m(i|r)$")>;
def : InstRW<[LNLPWriteResGroupX138], (instrs MOV8mr_NOREX)>;
def LNLPWriteResGroupX139 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX139], (instrs MOVBE16mr)>;
def LNLPWriteResGroupX142 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX142], (instrs MOVBE64mr,
                                             PUSHF16,
                                             SLDT16m,
                                             STRm)>;

def LNLPWriteResGroupX144 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX144], (instregex "^MOVDIR64B(16|32|64)$")>;
def LNLPWriteResGroupX147 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX147, ReadAfterVecXLd], (instregex "^(V?)MOVLP(D|S)rm$")>;
def LNLPWriteResGroupX148 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 512;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX148], (instrs MOVNTDQmr)>;
def LNLPWriteResGroupX149 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 518;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX149], (instrs MOVNTImr)>;
def LNLPWriteResGroupX150 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [4, 1, 1, 1];
  let Latency = 8;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX150], (instrs MOVSB)>;
def LNLPWriteResGroupX151 : SchedWriteRes<[LNLPVPort00_01_02_03]>;
def : InstRW<[LNLPWriteResGroupX151], (instrs VPBLENDDrri)>;
def LNLPWriteResGroupX152 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [4, 1, 1, 1];
  let Latency = 7;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX152], (instregex "^MOVS(L|Q|W)$")>;
def LNLPWriteResGroupX153 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 6;
}
def : InstRW<[LNLPWriteResGroupX153], (instregex "^MOVSX(16|32|64)rm(16|32)$",
                                                "^MOVSX(32|64)rm8$")>;
def : InstRW<[LNLPWriteResGroupX153], (instrs MOVSX32rm8_NOREX)>;
def LNLPWriteResGroupX154 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort20_21_22]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX154], (instrs MOVSX16rm8)>;
def LNLPWriteResGroupX156 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPVPort02_03]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX156], (instregex "^MUL_F(32|64)m$")>;
def LNLPWriteResGroupX157 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPVPort02_03, LNLPPort01_03_05]> {
  let Latency = 14;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX157], (instregex "^MUL_FI(16|32)m$")>;
def LNLPWriteResGroupX158 : SchedWriteRes<[LNLPVPort00_01_02_03]> {
  let Latency = 4;
}
def : InstRW<[LNLPWriteResGroupX158], (instregex "^MUL_F(P?)rST0$")>;
def : InstRW<[LNLPWriteResGroupX158], (instrs MUL_FST0r)>;
def LNLPWriteResGroupX155 : SchedWriteRes<[LNLPPort01_03_05]>;
def : InstRW<[LNLPWriteResGroupX155], (instregex "^MOVSX(16|32|64)rr(8|16|32)$")>;
def : InstRW<[LNLPWriteResGroupX155], (instrs MOVSX32rr8_NOREX)>;
def LNLPWriteResGroupX173 : SchedWriteRes<[LNLPPort01_03_05]>;
def : InstRW<[LNLPWriteResGroupX173], (instregex "^(V?)PALIGNRrri$",
                                                "^VPBROADCAST(B|D|Q|W)rr$")>;
def : InstRW<[LNLPWriteResGroupX173], (instrs VPALIGNRYrri)>;
def LNLPWriteResGroupX176 : SchedWriteRes<[LNLPVPort02_03, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX176], (instregex "^(V?)PEXTR(D|Q)mri$")>;
def LNLPWriteResGroupX179 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX179], (instregex "^POP(16|32|64)rmm$",
                                                "^PUSH(16|32)rmm$")>;
def LNLPWriteResGroupX180 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 5;
}
def : InstRW<[LNLPWriteResGroupX180], (instregex "^POPA(16|32)$",
                                                "^PREFETCHIT(0|1)$")>;
def : InstRW<[LNLPWriteResGroupX180], (instrs POPF32)>;
def LNLPWriteResGroupX181 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [6, 2, 1, 1];
  let Latency = 5;
  let NumMicroOps = 10;
}
def : InstRW<[LNLPWriteResGroupX181], (instrs POPF16)>;
def LNLPWriteResGroupX182 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 1, 1];
  let Latency = 5;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX182], (instrs POPF64)>;
def LNLPWriteResGroupX183 : SchedWriteRes<[LNLPPort20_21_22]> {
  let Latency = 0;
}
def : InstRW<[LNLPWriteResGroupX183], (instregex "^PREFETCHT(0|1|2)$")>;
def : InstRW<[LNLPWriteResGroupX183], (instrs PREFETCHNTA)>;
def LNLPWriteResGroupX187 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX187], (instregex "^PUSH64r((mr)?)$")>;
def LNLPWriteResGroupX188 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX188], (instrs PUSH64rmm)>;
def LNLPWriteResGroupX189 : SchedWriteRes<[LNLPPort20_21_22, LNLPPort10_11]>;
def : InstRW<[LNLPWriteResGroupX189], (instregex "^PUSHA(16|32)$",
                                                "^ST_F(32|64)m$")>;
def : InstRW<[LNLPWriteResGroupX189], (instrs PUSHF32)>;
def LNLPWriteResGroupX190 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX190], (instrs PUSHF64)>;
def LNLPWriteResGroupX191 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX191], (instregex "^PUSH(F|G)S64$")>;
def LNLPWriteResGroupX192 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2, 3, 2];
  let Latency = 8;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX192], (instregex "^RC(L|R)(16|32|64)rCL$")>;
def LNLPWriteResGroupX193 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX193, WriteRMW], (instregex "^RC(L|R)8m(1|i)$")>;
def LNLPWriteResGroupX194 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 5, 2];
  let Latency = 20;
  let NumMicroOps = 8;
}
def : InstRW<[LNLPWriteResGroupX194, WriteRMW], (instrs RCL8mCL)>;
def LNLPWriteResGroupX195 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2, 5, 2];
  let Latency = 7;
  let NumMicroOps = 9;
}
def : InstRW<[LNLPWriteResGroupX195], (instrs RCL8rCL)>;
def LNLPWriteResGroupX196 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2, 4, 3];
  let Latency = 20;
  let NumMicroOps = 9;
}
def : InstRW<[LNLPWriteResGroupX196, WriteRMW], (instrs RCR8mCL)>;
def LNLPWriteResGroupX197 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [3, 4, 3];
  let Latency = 9;
  let NumMicroOps = 10;
}
def : InstRW<[LNLPWriteResGroupX197], (instrs RCR8rCL)>;
def LNLPWriteResGroupX206 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [5, 6, 3, 1];
  let Latency = 18;
  let NumMicroOps = 15;
}
def : InstRW<[LNLPWriteResGroupX206], (instrs RDTSC)>;
def LNLPWriteResGroupX208 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX208], (instrs RET64)>;
def LNLPWriteResGroupX209 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 1];
  let Latency = 6;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX209], (instregex "^RETI(16|32|64)$")>;
def LNLPWriteResGroupX210 : SchedWriteRes<[]>;
def : InstRW<[LNLPWriteResGroupX210], (instrs REX64_PREFIX)>;
def LNLPWriteResGroupX211 : SchedWriteRes<[LNLPPort00_02_04]> {
  let ReleaseAtCycles = [2];
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX211, WriteRMW], (instregex "^RO(L|R)(16|32|64)m(1|i|CL)$")>;
def LNLPWriteResGroupX212 : SchedWriteRes<[LNLPPort00_02_04]> {
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX212], (instregex "^RO(L|R)(8|16|32|64)r(1|i)$")>;
def LNLPWriteResGroupX213 : SchedWriteRes<[LNLPPort00_02_04]> {
  let ReleaseAtCycles = [2];
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX213, WriteRMW], (instregex "^RO(L|R)8m(1|i)$",
                                                          "^(RO|SH)L8mCL$",
                                                          "^(RO|SA|SH)R8mCL$")>;
def LNLPWriteResGroupX214 : SchedWriteRes<[LNLPPort00_02_04]> {
  let ReleaseAtCycles = [2];
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX214], (instrs SAHF)>;
def LNLPWriteResGroupX215 : SchedWriteRes<[LNLPPort00_02_04]> {
  let Latency = 13;
}
def : InstRW<[LNLPWriteResGroupX215, WriteRMW], (instregex "^S(A|H)R8m(1|i)$",
                                                          "^SHL8m(1|i)$")>;
def LNLPWriteResGroupX216 : SchedWriteRes<[LNLPPort00_02_04, LNLPPort20_21_22]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX216, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^S(A|H)RX(32|64)rm$",
                                                                                                                              "^SHLX(32|64)rm$")>;
def LNLPWriteResGroupX217 : SchedWriteRes<[LNLPPort00_02_04]> {
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroupX217], (instregex "^S(A|H)RX(32|64)rr$",
                                                "^SHLX(32|64)rr$")>;
def LNLPWriteResGroupX218 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 2, 1, 1, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX218], (instrs SERIALIZE)>;
def LNLPWriteResGroupX219 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX219], (instrs SFENCE)>;
def LNLPWriteResGroupX220 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [1, 2, 2, 2];
  let Latency = 21;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX220], (instregex "^S(G|I)DT64m$")>;
def LNLPWriteResGroupX223 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPPort00_02_04, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 2, 1, 2, 1];
  let Latency = 13;
  let NumMicroOps = 8;
}
def : InstRW<[LNLPWriteResGroupX223, ReadAfterVecXLd], (instrs SHA1MSG2rm)>;
def LNLPWriteResGroupX224 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPPort00_02_04, LNLPVPort02_03]> {
  let ReleaseAtCycles = [2, 2, 1, 2];
  let Latency = 6;
  let NumMicroOps = 7;
}
def : InstRW<[LNLPWriteResGroupX224], (instrs SHA1MSG2rr)>;
def LNLPWriteResGroupX233 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 13;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX233], (instrs SHRD16mri8)>;
def LNLPWriteResGroupX234 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX234], (instregex "^SLDT(32|64)r$")>;
def LNLPWriteResGroupX235 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX235], (instrs SMSW16r)>;
def LNLPWriteResGroupX236 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX236], (instregex "^SMSW(32|64)r$")>;
def LNLPWriteResGroupX237 : SchedWriteRes<[LNLPPort00, LNLPPort20_21_22]> {
  let Latency = 24;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX237, ReadAfterVecLd], (instregex "^(V?)SQRTSDm_Int$")>;
def LNLPWriteResGroupX238 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX238], (instrs STD)>;
def LNLPWriteResGroupX239 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 4, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 6;
}
def : InstRW<[LNLPWriteResGroupX239], (instrs STI)>;
def LNLPWriteResGroupX240 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 1, 1];
  let Latency = 8;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX240], (instrs STOSB)>;
def LNLPWriteResGroupX241 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort10_11, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [2, 1, 1];
  let Latency = 7;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX241], (instregex "^STOS(L|Q|W)$")>;
def LNLPWriteResGroupX242 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort01_03_05]> {
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX242], (instregex "^STR(32|64)r$")>;
def LNLPWriteResGroupX243 : SchedWriteRes<[LNLPPort01_03_05]> {
  let Latency = 2;
}
def : InstRW<[LNLPWriteResGroupX243], (instregex "^(TST|XAM)_F$")>;
def : InstRW<[LNLPWriteResGroupX243], (instrs UCOM_FPPr)>;
def LNLPWriteResGroupX244 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 1];
  let Latency = 9;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroupX244, ReadAfterVecXLd, ReadAfterVecXLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^VBLENDVP(D|S)rmr$")>;
def : InstRW<[LNLPWriteResGroupX244, ReadAfterVecXLd, ReadAfterVecXLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instrs VPBLENDVBrmr)>;
def LNLPWriteResGroupX245 : SchedWriteRes<[LNLPPort01_03_05]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX245], (instregex "^VBLENDVP(D|S)rrr$")>;
def : InstRW<[LNLPWriteResGroupX245], (instrs VPBLENDVBrrr)>;
def LNLPWriteResGroupX250 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 2, 4];
  let Latency = 29;
  let NumMicroOps = 8;
}
def : InstRW<[LNLPWriteResGroupX250, WriteVecMaskedGatherWriteback], (instregex "^VGATHER(D|Q)PDYrm$",
                                                                               "^VPGATHER(D|Q)QYrm$")>;
def : InstRW<[LNLPWriteResGroupX250, WriteVecMaskedGatherWriteback], (instrs VGATHERQPSYrm,
                                                                            VPGATHERQDYrm)>;
def LNLPWriteResGroupX251 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 1, 2];
  let Latency = 20;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX251, WriteVecMaskedGatherWriteback], (instregex "^VGATHER(D|Q)PDrm$",
                                                                               "^VPGATHER(D|Q)Qrm$")>;
def : InstRW<[LNLPWriteResGroupX251, WriteVecMaskedGatherWriteback], (instrs VGATHERQPSrm,
                                                                            VPGATHERQDrm)>;
def LNLPWriteResGroupX252 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 2, 8];
  let Latency = 30;
  let NumMicroOps = 12;
}
def : InstRW<[LNLPWriteResGroupX252, WriteVecMaskedGatherWriteback], (instrs VGATHERDPSYrm,
                                                                            VPGATHERDDYrm)>;
def LNLPWriteResGroupX253 : SchedWriteRes<[LNLPVPort00_01, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 2, 4];
  let Latency = 28;
  let NumMicroOps = 8;
}
def : InstRW<[LNLPWriteResGroupX253, WriteVecMaskedGatherWriteback], (instrs VGATHERDPSrm,
                                                                            VPGATHERDDrm)>;
def LNLPWriteResGroupX254 : SchedWriteRes<[LNLPVPort02_03, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 5;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX254], (instregex "^VH(ADD|SUB)P(D|S)rr$")>;
def LNLPWriteResGroupX256 : SchedWriteRes<[LNLPVPort00_01, LNLPPort00_02_04, LNLPPort20_21_22]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX256], (instrs VLDMXCSR)>;
def LNLPWriteResGroupX257 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [8, 1, 1, 1, 1, 1, 2, 3];
  let Latency = 40;
  let NumMicroOps = 18;
}
def : InstRW<[LNLPWriteResGroupX257], (instrs VMCLEARm)>;
def LNLPWriteResGroupX259 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 521;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX259], (instrs VMOVNTDQmr)>;
def LNLPWriteResGroupX260 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 473;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX260], (instrs VMOVNTPDmr)>;
def LNLPWriteResGroupX261 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 494;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX261], (instrs VMOVNTPSYmr)>;
def LNLPWriteResGroupX262 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = 470;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX262], (instrs VMOVNTPSmr)>;
def LNLPWriteResGroupX264 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX264, ReadAfterVecYLd], (instregex "^VSHUFP(D|S)Yrmi$")>;
def LNLPWriteResGroupX267 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [1, 2, 3, 3, 1];
  let Latency = 16;
  let NumMicroOps = 10;
}
def : InstRW<[LNLPWriteResGroupX267], (instrs VZEROALL)>;
def LNLPWriteResGroupX268 : SchedWriteRes<[LNLPVPort00_01_02_03]> {
  let ReleaseAtCycles = [2];
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX268], (instrs WAIT)>;
def LNLPWriteResGroupX269 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [8, 6, 19, 63, 21, 15, 1, 10, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 144;
}
def : InstRW<[LNLPWriteResGroupX269], (instrs WRMSR)>;
def LNLPWriteResGroupX270 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [2, 1, 4, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 8;
}
def : InstRW<[LNLPWriteResGroupX270], (instrs WRPKRUr)>;
def LNLPWriteResGroupX271 : SchedWriteRes<[LNLPPort00_01_02_03_04_05]> {
  let ReleaseAtCycles = [2];
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX271, WriteRMW], (instregex "^XADD(16|32|64)rm$")>;
def LNLPWriteResGroupX272 : SchedWriteRes<[LNLPPort00_01_02_03_04_05]> {
  let ReleaseAtCycles = [2];
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroupX272, WriteRMW], (instrs XADD8rm)>;
def LNLPWriteResGroupX273 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [4, 1];
  let Latency = 39;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX273, WriteRMW], (instregex "^XCHG(16|32)rm$")>;
def LNLPWriteResGroupX274 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [5, 1];
  let Latency = 39;
  let NumMicroOps = 6;
}
def : InstRW<[LNLPWriteResGroupX274, WriteRMW], (instrs XCHG64rm)>;
def LNLPWriteResGroupX275 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [4, 1];
  let Latency = 40;
  let NumMicroOps = 5;
}
def : InstRW<[LNLPWriteResGroupX275, WriteRMW], (instrs XCHG8rm)>;
def LNLPWriteResGroupX276 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort01_03_05, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [2, 4, 2, 1, 2, 4];
  let Latency = 17;
  let NumMicroOps = 15;
}
def : InstRW<[LNLPWriteResGroupX276], (instrs XCH_F)>;
def LNLPWriteResGroupX277 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [7, 3, 8, 5];
  let Latency = 4;
  let NumMicroOps = 23;
}
def : InstRW<[LNLPWriteResGroupX277], (instrs XGETBV)>;
def LNLPWriteResGroupX278 : SchedWriteRes<[LNLPPort00_01_02_03_04_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 1];
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroupX278], (instrs XLAT)>;
def LNLPWriteResGroupX279 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort00_02_04]> {
  let ReleaseAtCycles = [21, 1, 1, 8];
  let Latency = 37;
  let NumMicroOps = 31;
}
def : InstRW<[LNLPWriteResGroupX279], (instregex "^XRSTOR((S|64)?)$")>;
def : InstRW<[LNLPWriteResGroupX279], (instrs XRSTORS64)>;
def LNLPWriteResGroupX280 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [14, 25, 44, 21, 21, 4, 1, 9, 1];
  let Latency = 42;
  let NumMicroOps = 140;
}
def : InstRW<[LNLPWriteResGroupX280], (instrs XSAVE)>;
def LNLPWriteResGroupX281 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [14, 25, 44, 21, 21, 4, 1, 9, 1];
  let Latency = 41;
  let NumMicroOps = 140;
}
def : InstRW<[LNLPWriteResGroupX281], (instrs XSAVE64)>;
def LNLPWriteResGroupX282 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [1, 19, 36, 52, 23, 4, 2, 12, 2];
  let Latency = 42;
  let NumMicroOps = 151;
}
def : InstRW<[LNLPWriteResGroupX282], (instrs XSAVEC)>;
def LNLPWriteResGroupX283 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [1, 19, 36, 53, 23, 4, 2, 12, 2];
  let Latency = 42;
  let NumMicroOps = 152;
}
def : InstRW<[LNLPWriteResGroupX283], (instrs XSAVEC64)>;
def LNLPWriteResGroupX284 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [25, 35, 52, 27, 4, 1, 10, 1];
  let Latency = 46;
  let NumMicroOps = 155;
}
def : InstRW<[LNLPWriteResGroupX284], (instrs XSAVEOPT)>;
def LNLPWriteResGroupX285 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [25, 35, 53, 27, 4, 1, 10, 1];
  let Latency = 46;
  let NumMicroOps = 156;
}
def : InstRW<[LNLPWriteResGroupX285], (instrs XSAVEOPT64)>;
def LNLPWriteResGroupX286 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [23, 32, 53, 29, 30, 4, 2, 9, 2];
  let Latency = 42;
  let NumMicroOps = 184;
}
def : InstRW<[LNLPWriteResGroupX286], (instrs XSAVES)>;
def LNLPWriteResGroupX287 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPVPort02_03, LNLPPort20_21_22, LNLPPort10_11, LNLPPort01_03_05, LNLPPort25_26_27]> {
  let ReleaseAtCycles = [23, 33, 53, 29, 32, 4, 2, 8, 2];
  let Latency = 42;
  let NumMicroOps = 186;
}
def : InstRW<[LNLPWriteResGroupX287], (instrs XSAVES64)>;
def LNLPWriteResGroupX288 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort00_01_02_03_04_05, LNLPVPort00_01_02_03, LNLPPort00_02_04, LNLPPort01_03_05, LNLPPort01_03_05, LNLPPort01_03_05]> {
  let ReleaseAtCycles = [4, 23, 2, 14, 8, 1, 2];
  let Latency = 5;
  let NumMicroOps = 54;
}
def : InstRW<[LNLPWriteResGroupX288], (instrs XSETBV)>;

// SchedWriteRes and InstRW definition
// Following defs are based on data shared by arch team
def LNLPWriteResGroup0 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup0, ReadAfterVecXLd], (instregex "^(V?)CMPP(D|S)rmi$",
                                                               "^GF2P8AFFINE((INV)?)QBrmi$")>;
def : InstRW<[LNLPWriteResGroup0, ReadAfterVecXLd], (instrs GF2P8MULBrm)>;
def : InstRW<[LNLPWriteResGroup0, ReadAfterVecLd], (instregex "^(V?)CMPS(D|S)rmi((_Int)?)$")>;
def LNLPWriteResGroup1 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 1, 6];
  let Latency = 11;
  let NumMicroOps = 2;
}
def LNLPWriteResGroup2 : SchedWriteRes<[LNLPPort01_03_05, LNLPVPort00_01, LNLPVPort02_03]> {
  let ReleaseAtCycles = [4, 4, 1];
  let Latency = 9;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup2, ReadDefault, ReadInt2Fpu], (instregex "^(V?)CVTSI642SSrr_Int$")>;
def LNLPWriteResGroup3 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7, 4];
  let Latency = 11;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup3], (instregex "^CVT(T?)SS2SI((64)?)rm_Int$",
                                              "^CVT(T?)SS2SIrm$",
                                              "^VCVTTSD2SI((64)?)rm_Int$")>;
def LNLPWriteResGroup4 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03]> {
  let ReleaseAtCycles = [7, 1];
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup4], (instregex "^(V?)CVT(T?)SS2SI64rr_Int$")>;
def LNLPWriteResGroup5 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [4];
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroup5], (instregex "^GF2P8AFFINE((INV)?)QBrri$")>;
def : InstRW<[LNLPWriteResGroup5], (instrs GF2P8MULBrr)>;
def LNLPWriteResGroup6 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 1, 8];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup6], (instregex "^MMX_CVT(T?)PD2PIrm$")>;
def LNLPWriteResGroup7 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03]> {
  let ReleaseAtCycles = [4, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup7], (instregex "^MMX_CVT(T?)PD2PIrr$")>;
def : InstRW<[LNLPWriteResGroup7], (instrs MMX_CVTPI2PDrr)>;
def LNLPWriteResGroup8 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [5];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup8], (instrs MMX_CVTPI2PSrr)>;
def LNLPWriteResGroup9 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 8];
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup9], (instregex "^MMX_CVT(T?)PS2PIrm$")>;
def LNLPWriteResGroup10 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03]> {
  let ReleaseAtCycles = [4, 1];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup10], (instrs MMX_CVTPS2PIrr)>;
def LNLPWriteResGroup11 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03]> {
  let ReleaseAtCycles = [4, 1];
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup11], (instrs MMX_CVTTPS2PIrr)>;
def LNLPWriteResGroup12 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup12], (instregex "^MMX_MOVQ2(DQ|FR64)rr$")>;
def LNLPWriteResGroup13 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [2, 4];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup13], (instregex "^MOVBE(16|32|64)rm$")>;
def LNLPWriteResGroup14 : SchedWriteRes<[LNLPPort20_21_22]> {
  let ReleaseAtCycles = [6];
  let Latency = 6;
}
def : InstRW<[LNLPWriteResGroup14], (instregex "^(V?)MOV(D|SH|SL)DUPrm$",
                                               "^VPBROADCAST(D|Q|W)rm$")>;
def : InstRW<[LNLPWriteResGroup14], (instrs VBROADCASTSSrm)>;
def LNLPWriteResGroup15 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup15], (instregex "^MOVDIRI(32|64)$")>;
def LNLPWriteResGroup16 : SchedWriteRes<[LNLPVPort00_01_02_03]>;
def : InstRW<[LNLPWriteResGroup16], (instregex "^(V?)MOVS(D|S)rr((_REV)?)$",
                                               "^(V?)P(ADD|SUB)(B|D|Q|W)rr$",
                                               "^VP(ADD|SUB)(B|D|Q|W)Yrr$")>;
def LNLPWriteResGroup17 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;  // originally 10
}
def : InstRW<[LNLPWriteResGroup17, ReadAfterVecXLd], (instregex "^(V?)PACK(S|U)S(DW|WB)rm$",
                                                                "^(V?)PCMPGTQrm$")>;
def LNLPWriteResGroup18 : SchedWriteRes<[LNLPVPort02_03]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroup18], (instregex "^(V?)PACK(S|U)S(DW|WB)rr$",
                                               "^(V?)PACKUSWBrr$",
                                               "^(V?)PCMPGTQrr$",
                                               "^VPACK(S|U)S(DW|WB)Yrr$",
                                               "^VSHA512MSG(1|2)rr$")>;
def : InstRW<[LNLPWriteResGroup18], (instrs VPCMPGTQYrr)>;
def LNLPWriteResGroup19 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
}
def : InstRW<[LNLPWriteResGroup19, ReadAfterVecXLd], (instregex "^(V?)P(ADD|SUB)(B|D|Q|W)rm$")>;
def : InstRW<[LNLPWriteResGroup19, ReadAfterVecXLd], (instrs VPBLENDDrmi)>;
def LNLPWriteResGroup20 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 6];
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup20], (instregex "^(V?)PSHUF(D|HW|LW)mi$",
                                               "^VPERMILP(D|S)mi$")>;
def : InstRW<[LNLPWriteResGroup20, ReadAfterVecXLd], (instregex "^(V?)PALIGNRrmi$",
                                                                "^(V?)SHUFP(D|S)rmi$")>;
def : InstRW<[LNLPWriteResGroup20, ReadAfterVecXLd], (instrs VINSERTPSrmi,
                                                             VPBLENDWrmi)>;
def LNLPWriteResGroup21 : SchedWriteRes<[LNLPPort01_03_05, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 4];
  let Latency = 7;
}
def : InstRW<[LNLPWriteResGroup21, ReadAfterLd], (instregex "^P(DEP|EXT)(32|64)rm$")>;
def LNLPWriteResGroup22 : SchedWriteRes<[LNLPPort01_03_05]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroup22], (instregex "^P(DEP|EXT)(32|64)rr$")>;
def LNLPWriteResGroup23 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 2, 1];
  let Latency = 9; // originally LunarlakePModel.MaxLatency;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroup23, ReadAfterVecXLd], (instregex "^(V?)PH(ADD|SUB)SWrm$")>;
def LNLPWriteResGroup24 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03]> {
  let ReleaseAtCycles = [1, 2];
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup24], (instregex "^(V?)PH(ADD|SUB)SWrr$",
                                               "^VPH(ADD|SUB)SWYrr$")>;
def LNLPWriteResGroup25 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
}
def : InstRW<[LNLPWriteResGroup25, ReadAfterVecXLd], (instregex "^(V?)PMULUDQrm$")>;
def : InstRW<[LNLPWriteResGroup25, ReadAfterVecXLd], (instrs VPMULDQrm)>;
def LNLPWriteResGroup26 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 3;
}
def : InstRW<[LNLPWriteResGroup26], (instregex "^(V?)PMULUDQrr$",
                                               "^VPMUL(U?)DQYrr$")>;
def : InstRW<[LNLPWriteResGroup26], (instrs VMOVSDto64Zrr,
                                            VPMULDQrr)>;

def LNLPWriteResGroup27 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 6];
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup27, ReadAfterVecXLd], (instrs SHA1MSG1rm)>;
def LNLPWriteResGroup28 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPVPort02_03]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup28], (instrs SHA1MSG1rr)>;
def LNLPWriteResGroup29 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 1, 1, 6];
  let Latency = 7;  // Originally 8
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroup29, ReadAfterVecXLd], (instrs SHA1NEXTErm)>;
def LNLPWriteResGroup30 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort00_01_02_03, LNLPVPort02_03]> {
  let Latency = 3;  // originally LunarlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup30], (instrs SHA1NEXTErr)>;
def LNLPWriteResGroup31 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup31, ReadAfterVecXLd], (instrs SHA1RNDS4rmi)>;
def LNLPWriteResGroup32 : SchedWriteRes<[LNLPVPort02_03]> {
  let ReleaseAtCycles = [3];
  let Latency = 2;
}
def : InstRW<[LNLPWriteResGroup32], (instrs SHA1RNDS4rri,
                                            SHA256RNDS2rr)>;
def LNLPWriteResGroup33 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 12 ; // orignally  99; diff lat for 1 & 2 version
}
def : InstRW<[LNLPWriteResGroup33, ReadAfterVecXLd], (instregex "^SHA256MSG(1|2)rm$")>;

def LNLPWriteResGroup34 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 5;  //  orignally 99; Diff lat for 1 & 2 version
}
def : InstRW<[LNLPWriteResGroup34], (instregex "^SHA256MSG(1|2)rr$")>;
def LNLPWriteResGroup35 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup35, ReadAfterVecXLd], (instrs SHA256RNDS2rm)>;
def LNLPWriteResGroup36 : SchedWriteRes<[LNLPPort20_21_22]> {
  let ReleaseAtCycles = [7];
  let Latency = 7;
}
def : InstRW<[LNLPWriteResGroup36], (instregex "^VBROADCASTS(D|S)Yrm$",
                                               "^VMOV(D|SH|SL)DUPYrm$",
                                               "^VPBROADCAST(D|Q|W)Yrm$")>;
def : InstRW<[LNLPWriteResGroup36], (instrs VBROADCASTF128rm)>;
def LNLPWriteResGroup37 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 7];
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup37, ReadAfterVecYLd], (instregex "^VCMPP(D|S)Yrmi$")>;
def LNLPWriteResGroup38 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 6];
  let Latency = 10;
}
def : InstRW<[LNLPWriteResGroup38], (instregex "^VCVT(T?)PS2DQrm$")>;
def LNLPWriteResGroup39 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [4, 1, 6];
  let Latency = 11;
  let NumMicroOps = 3;
}
def : InstRW<[LNLPWriteResGroup39, ReadAfterVecLd], (instrs VCVTSI642SSrm_Int)>;
def LNLPWriteResGroup40 : SchedWriteRes<[LNLPPort10_11, LNLPPort25_26_27]> {
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup40], (instregex "^VEXTRACT(F|I)128mri$")>;
def LNLPWriteResGroup41 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup41, ReadAfterVecYLd], (instregex "^VINSERT(F|I)128rmi$")>;
def LNLPWriteResGroup42 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = 4;
}
def : InstRW<[LNLPWriteResGroup42], (instregex "^VMOVMSKP(D|S)Yrr$")>;
def LNLPWriteResGroup43 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
}
def : InstRW<[LNLPWriteResGroup43, ReadAfterVecYLd], (instregex "^VPACK(S|U)S(DW|WB)Yrm$")>;
def : InstRW<[LNLPWriteResGroup43, ReadAfterVecYLd], (instrs VPCMPGTQYrm)>;
def LNLPWriteResGroup44 : SchedWriteRes<[LNLPVPort00_01_02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
def : InstRW<[LNLPWriteResGroup44, ReadAfterVecYLd], (instregex "^VP(ADD|SUB)(B|D|Q|W)Yrm$")>;
def LNLPWriteResGroup45 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup45], (instregex "^VPSHUF(D|HW|LW)Ymi$")>;
def : InstRW<[LNLPWriteResGroup45, ReadAfterVecYLd], (instrs VPALIGNRYrmi,
                                                             VPBLENDWYrmi)>;
def LNLPWriteResGroup46 : SchedWriteRes<[LNLPVPort02_03]>;
def : InstRW<[LNLPWriteResGroup46], (instregex "^VPBLENDW(Y?)rri$")>;
def LNLPWriteResGroup47 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 7];
  let Latency = 8;
}
def : InstRW<[LNLPWriteResGroup47], (instrs VPBROADCASTBYrm)>;
def : InstRW<[LNLPWriteResGroup47, ReadAfterVecYLd], (instregex "^VUNPCK(H|L)P(D|S)Yrm$")>;
def LNLPWriteResGroup48 : SchedWriteRes<[LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup48, ReadAfterVecXLd], (instrs VPCLMULQDQYrmi)>;
def LNLPWriteResGroup49 : SchedWriteRes<[LNLPVPort00_01, LNLPVPort02_03, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [1, 2, 7];
  let Latency = 9;
  let NumMicroOps = 4;
}
def : InstRW<[LNLPWriteResGroup49, ReadAfterVecYLd], (instregex "^VPH(ADD|SUB)SWYrm$")>;
def LNLPWriteResGroup50 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 7];
  let Latency = 10;
}
def : InstRW<[LNLPWriteResGroup50, ReadAfterVecYLd], (instregex "^VPMUL(U?)DQYrm$")>;
def LNLPWriteResGroup51 : SchedWriteRes<[LNLPVPort02_03]> {
  let ReleaseAtCycles = [3];
  let Latency = LunarlakePModel.MaxLatency;
}
def : InstRW<[LNLPWriteResGroup51], (instrs VSHA512RNDS2rr)>;
def LNLPWriteResGroup52 : SchedWriteRes<[LNLPVPort00_01, LNLPPort20_21_22]> {
  let ReleaseAtCycles = [3, 6];
  let Latency = LunarlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[LNLPWriteResGroup52], (instregex "^VSM3MSG(1|2)rm$")>;
def LNLPWriteResGroup53 : SchedWriteRes<[LNLPVPort00_01]> {
  let ReleaseAtCycles = [3];
  let Latency = LunarlakePModel.MaxLatency;
}
def : InstRW<[LNLPWriteResGroup53], (instregex "^VSM3MSG(1|2)rr$")>;
def : InstRW<[LNLPWriteResGroup53], (instrs VTESTPSYrr)>;
}
