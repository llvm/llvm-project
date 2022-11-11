//===- X86SchedAlderlakeP.td - X86 Alderlake-P Scheduling ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Alderlake-P core to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def AlderlakePModel : SchedMachineModel {
  // Alderlake-P core can allocate 6 uops per cycle.
  let IssueWidth = 6; // Based on allocator width.
  let MicroOpBufferSize = 512; // Based on the reorder buffer.
  let LoadLatency = 5;
  let MispredictPenalty = 14;

  // Latency for microcoded instructions or instructions without latency info.
  int MaxLatency = 100;

  // Based on the LSD (loop-stream detector) queue size (ST).
  let LoopMicroOpBufferSize = 72;

  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = AlderlakePModel in {

// Alderlake-P core can issue micro-ops to 12 different ports in one cycle.
def ADLPPort00 : ProcResource<1>;
def ADLPPort01 : ProcResource<1>;
def ADLPPort02 : ProcResource<1>;
def ADLPPort03 : ProcResource<1>;
def ADLPPort04 : ProcResource<1>;
def ADLPPort05 : ProcResource<1>;
def ADLPPort06 : ProcResource<1>;
def ADLPPort07 : ProcResource<1>;
def ADLPPort08 : ProcResource<1>;
def ADLPPort09 : ProcResource<1>;
def ADLPPort10 : ProcResource<1>;
def ADLPPort11 : ProcResource<1>;

// Workaround to represent invalid ports. WriteRes shouldn't use this resource.
def ADLPPortInvalid : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def ADLPPort00_01          : ProcResGroup<[ADLPPort00, ADLPPort01]>;
def ADLPPort00_01_05       : ProcResGroup<[ADLPPort00, ADLPPort01, ADLPPort05]>;
def ADLPPort00_01_05_06    : ProcResGroup<[ADLPPort00, ADLPPort01, ADLPPort05, ADLPPort06]>;
def ADLPPort00_05          : ProcResGroup<[ADLPPort00, ADLPPort05]>;
def ADLPPort00_05_06       : ProcResGroup<[ADLPPort00, ADLPPort05, ADLPPort06]>;
def ADLPPort00_06          : ProcResGroup<[ADLPPort00, ADLPPort06]>;
def ADLPPort01_05          : ProcResGroup<[ADLPPort01, ADLPPort05]>;
def ADLPPort01_05_10       : ProcResGroup<[ADLPPort01, ADLPPort05, ADLPPort10]>;
def ADLPPort02_03          : ProcResGroup<[ADLPPort02, ADLPPort03]>;
def ADLPPort02_03_07       : ProcResGroup<[ADLPPort02, ADLPPort03, ADLPPort07]>;
def ADLPPort02_03_11       : ProcResGroup<[ADLPPort02, ADLPPort03, ADLPPort11]>;
def ADLPPort07_08          : ProcResGroup<[ADLPPort07, ADLPPort08]>;

// EU has 112 reservation stations.
def ADLPPort00_01_05_06_10 : ProcResGroup<[ADLPPort00, ADLPPort01, ADLPPort05,
                                           ADLPPort06, ADLPPort10]> {
  let BufferSize = 112;
}

// STD has 48 reservation stations.
def ADLPPort04_09          : ProcResGroup<[ADLPPort04, ADLPPort09]> {
  let BufferSize = 48;
}

// MEM has 72 reservation stations.
def ADLPPort02_03_07_08_11 : ProcResGroup<[ADLPPort02, ADLPPort03, ADLPPort07,
                                           ADLPPort08, ADLPPort11]> {
  let BufferSize = 72;
}

// Integer loads are 5 cycles, so ReadAfterLd registers needn't be available
// until 5 cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

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
multiclass ADLPWriteResPair<X86FoldableSchedWrite SchedRW,
                            list<ProcResourceKind> ExePorts,
                            int Lat, list<int> Res = [1], int UOps = 1,
                            int LoadLat = 5> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on port 2/3/11 and adds LoadLat cycles to
  // the latency (default = 5).
  def : WriteRes<SchedRW.Folded, !listconcat([ADLPPort02_03_11], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

//===----------------------------------------------------------------------===//
// The following definitons are infered by smg.
//===----------------------------------------------------------------------===//

// Infered SchedWrite definition.
def : WriteRes<WriteADC, [ADLPPort00_06]>;
defm : X86WriteRes<WriteADCLd, [ADLPPort00_01_05_06_10, ADLPPort00_06], 11, [1, 1], 2>;
defm : ADLPWriteResPair<WriteAESDecEnc, [ADLPPort00_01], 5, [1], 1, 7>;
defm : ADLPWriteResPair<WriteAESIMC, [ADLPPort00_01], 8, [2], 2, 7>;
defm : X86WriteRes<WriteAESKeyGen, [ADLPPort00, ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01_05, ADLPPort05], 7, [4, 1, 1, 2, 3, 3], 14>;
defm : X86WriteRes<WriteAESKeyGenLd, [ADLPPort00, ADLPPort00_01, ADLPPort00_06, ADLPPort01_05, ADLPPort02_03_11, ADLPPort05], 12, [4, 1, 2, 3, 1, 3], 14>;
def : WriteRes<WriteALU, [ADLPPort00_01_05_06_10]>;
def : WriteRes<WriteALULd, [ADLPPort00_01_05_06_10]> {
  let Latency = 11;
}
defm : ADLPWriteResPair<WriteBEXTR, [ADLPPort00_06, ADLPPort01], 6, [1, 1], 2>;
defm : ADLPWriteResPair<WriteBLS, [ADLPPort01_05_10], 2, [1]>;
defm : ADLPWriteResPair<WriteBSF, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteBSR, [ADLPPort01], 3, [1]>;
def : WriteRes<WriteBSWAP32, [ADLPPort01]>;
defm : X86WriteRes<WriteBSWAP64, [ADLPPort00_06, ADLPPort01], 2, [1, 1], 2>;
defm : ADLPWriteResPair<WriteBZHI, [ADLPPort01], 3, [1]>;
def : WriteRes<WriteBitTest, [ADLPPort01]>;
defm : X86WriteRes<WriteBitTestImmLd, [ADLPPort01, ADLPPort02_03_11], 6, [1, 1], 2>;
defm : X86WriteRes<WriteBitTestRegLd, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11], 11, [4, 2, 1, 2, 1], 10>;
def : WriteRes<WriteBitTestSet, [ADLPPort01]>;
def : WriteRes<WriteBitTestSetImmLd, [ADLPPort01]> {
  let Latency = 11;
}
defm : X86WriteRes<WriteBitTestSetRegLd, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10], 17, [3, 2, 1, 2], 8>;
defm : ADLPWriteResPair<WriteBlend, [ADLPPort01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteBlendY, [ADLPPort00_01_05], 1, [1], 1, 8>;
defm : ADLPWriteResPair<WriteCLMul, [ADLPPort05], 3, [1], 1, 7>;
defm : ADLPWriteResPair<WriteCMOV, [ADLPPort00_06], 1, [1], 1, 6>;
defm : X86WriteRes<WriteCMPXCHG, [ADLPPort00_01_05_06_10, ADLPPort00_06], 3, [3, 2], 5>;
defm : X86WriteRes<WriteCMPXCHGRMW, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08], 12, [1, 2, 1, 1, 1], 6>;
defm : ADLPWriteResPair<WriteCRC32, [ADLPPort01], 3, [1]>;
defm : X86WriteRes<WriteCvtI2PD, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2PDLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2PDY, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2PDYLd, [ADLPPort00_01, ADLPPort02_03_11], 12, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtI2PDZ>;
defm : ADLPWriteResPair<WriteCvtI2PS, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteCvtI2PSY, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteCvtI2PSZ>;
defm : X86WriteRes<WriteCvtI2SD, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2SDLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2SS, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2>;
defm : X86WriteRes<WriteCvtI2SSLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : ADLPWriteResPair<WriteCvtPD2I, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2, 7>;
defm : ADLPWriteResPair<WriteCvtPD2IY, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2, 8>;
defm : X86WriteResPairUnsupported<WriteCvtPD2IZ>;
defm : ADLPWriteResPair<WriteCvtPD2PS, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2, 7>;
defm : ADLPWriteResPair<WriteCvtPD2PSY, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2, 8>;
defm : X86WriteResPairUnsupported<WriteCvtPD2PSZ>;
defm : X86WriteRes<WriteCvtPH2PS, [ADLPPort00_01, ADLPPort05], 6, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPH2PSLd, [ADLPPort00_01, ADLPPort02_03_11], 12, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPH2PSY, [ADLPPort00_01, ADLPPort05], 8, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPH2PSYLd, [ADLPPort00_01, ADLPPort02_03_11], 12, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPH2PSZ>;
defm : ADLPWriteResPair<WriteCvtPS2I, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteCvtPS2IY, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteCvtPS2IZ>;
defm : X86WriteRes<WriteCvtPS2PD, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PDLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PDY, [ADLPPort00_01, ADLPPort05], 7, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PDYLd, [ADLPPort00_01, ADLPPort02_03_11], 12, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPS2PDZ>;
defm : X86WriteRes<WriteCvtPS2PH, [ADLPPort00_01, ADLPPort05], 6, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PHSt, [ADLPPort00_01, ADLPPort04_09, ADLPPort07_08], 12, [1, 1, 1], 3>;
defm : X86WriteRes<WriteCvtPS2PHY, [ADLPPort00_01, ADLPPort05], 8, [1, 1], 2>;
defm : X86WriteRes<WriteCvtPS2PHYSt, [ADLPPort00_01, ADLPPort04_09, ADLPPort07_08], 12, [1, 1, 1], 3>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZ>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZSt>;
defm : ADLPWriteResPair<WriteCvtSD2I, [ADLPPort00, ADLPPort00_01], 7, [1, 1], 2>;
defm : ADLPWriteResPair<WriteCvtSD2SS, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2, 7>;
defm : ADLPWriteResPair<WriteCvtSS2I, [ADLPPort00, ADLPPort00_01], 7, [1, 1], 2>;
defm : X86WriteRes<WriteCvtSS2SD, [ADLPPort00_01, ADLPPort05], 5, [1, 1], 2>;
defm : X86WriteRes<WriteCvtSS2SDLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : ADLPWriteResPair<WriteDPPD, [ADLPPort00_01, ADLPPort01_05], 9, [2, 1], 3, 7>;
defm : ADLPWriteResPair<WriteDPPS, [ADLPPort00_01, ADLPPort00_06, ADLPPort01_05, ADLPPort05], 14, [2, 1, 2, 1], 6, 7>;
defm : ADLPWriteResPair<WriteDPPSY, [ADLPPort00_01, ADLPPort00_06, ADLPPort01_05, ADLPPort05], 14, [2, 1, 2, 1], 6, 8>;
defm : ADLPWriteResPair<WriteDiv16, [ADLPPort00_01_05_06_10, ADLPPort01], 16, [1, 3], 4, 4>;
defm : ADLPWriteResPair<WriteDiv32, [ADLPPort00_01_05_06_10, ADLPPort01], 15, [1, 3], 4, 4>;
defm : ADLPWriteResPair<WriteDiv64, [ADLPPort01], 18, [3], 3>;
defm : X86WriteRes<WriteDiv8, [ADLPPort01], 17, [3], 3>;
defm : X86WriteRes<WriteDiv8Ld, [ADLPPort01], 22, [3], 3>;
defm : X86WriteRes<WriteEMMS, [ADLPPort00, ADLPPort00_05, ADLPPort00_06], 10, [1, 8, 1], 10>;
def : WriteRes<WriteFAdd, [ADLPPort05]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteFAddLd, [ADLPPort02_03, ADLPPort05], 10, [1, 1], 2>;
defm : ADLPWriteResPair<WriteFAdd64, [ADLPPort01_05], 3, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFAdd64X, [ADLPPort01_05], 3, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFAdd64Y, [ADLPPort01_05], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFAdd64Z>;
defm : ADLPWriteResPair<WriteFAddX, [ADLPPort01_05], 3, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFAddY, [ADLPPort01_05], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFAddZ>;
defm : ADLPWriteResPair<WriteFBlend, [ADLPPort00_01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFBlendY, [ADLPPort00_01_05], 1, [1], 1, 8>;
def : WriteRes<WriteFCMOV, [ADLPPort01]> {
  let Latency = 3;
}
defm : ADLPWriteResPair<WriteFCmp, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFCmp64, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFCmp64X, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFCmp64Y, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFCmp64Z>;
defm : ADLPWriteResPair<WriteFCmpX, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFCmpY, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFCmpZ>;
def : WriteRes<WriteFCom, [ADLPPort05]>;
defm : X86WriteRes<WriteFComLd, [ADLPPort02_03, ADLPPort05], 8, [1, 1], 2>;
defm : ADLPWriteResPair<WriteFComX, [ADLPPort00], 3, [1]>;
def : WriteRes<WriteFDiv, [ADLPPort00]> {
  let Latency = 15;
}
defm : X86WriteRes<WriteFDivLd, [ADLPPort00, ADLPPort02_03], 27, [1, 1], 2>;
defm : ADLPWriteResPair<WriteFDiv64, [ADLPPort00], 14, [1], 1, 6>;
defm : ADLPWriteResPair<WriteFDiv64X, [ADLPPort00], 14, [1], 1, 6>;
defm : ADLPWriteResPair<WriteFDiv64Y, [ADLPPort00], 14, [1], 1, 7>;
defm : X86WriteResPairUnsupported<WriteFDiv64Z>;
defm : ADLPWriteResPair<WriteFDivX, [ADLPPort00], 11, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFDivY, [ADLPPort00], 11, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFDivZ>;
defm : ADLPWriteResPair<WriteFHAdd, [ADLPPort01_05, ADLPPort05], 6, [1, 2], 3, 6>;
defm : ADLPWriteResPair<WriteFHAddY, [ADLPPort01_05, ADLPPort05], 5, [1, 2], 3, 8>;
def : WriteRes<WriteFLD0, [ADLPPort00_05]>;
defm : X86WriteRes<WriteFLD1, [ADLPPort00_05], 1, [2], 2>;
defm : X86WriteRes<WriteFLDC, [ADLPPort00_05], 1, [2], 2>;
def : WriteRes<WriteFLoad, [ADLPPort02_03_11]> {
  let Latency = 7;
}
def : WriteRes<WriteFLoadX, [ADLPPort02_03_11]> {
  let Latency = 7;
}
def : WriteRes<WriteFLoadY, [ADLPPort02_03_11]> {
  let Latency = 8;
}
defm : ADLPWriteResPair<WriteFLogic, [ADLPPort00_01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFLogicY, [ADLPPort00_01_05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFLogicZ>;
defm : ADLPWriteResPair<WriteFMA, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFMAX, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFMAY, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFMAZ>;
def : WriteRes<WriteFMOVMSK, [ADLPPort00]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteFMaskedLoad, [ADLPPort00_01_05, ADLPPort02_03_11], 8, [1, 1], 2>;
defm : X86WriteRes<WriteFMaskedLoadY, [ADLPPort00_01_05, ADLPPort02_03_11], 9, [1, 1], 2>;
defm : X86WriteRes<WriteFMaskedStore32, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore32Y, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore64, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMaskedStore64Y, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteFMoveX, [], 1, [], 0>;
defm : X86WriteRes<WriteFMoveY, [], 1, [], 0>;
defm : X86WriteResUnsupported<WriteFMoveZ>;
def : WriteRes<WriteFMul, [ADLPPort00]> {
  let Latency = 4;
}
defm : X86WriteRes<WriteFMulLd, [ADLPPort00_01, ADLPPort02_03_11], 11, [1, 1], 2>;
defm : ADLPWriteResPair<WriteFMul64, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFMul64X, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFMul64Y, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFMul64Z>;
defm : ADLPWriteResPair<WriteFMulX, [ADLPPort00_01], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFMulY, [ADLPPort00_01], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFMulZ>;
defm : ADLPWriteResPair<WriteFRcp, [ADLPPort00], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFRcpX, [ADLPPort00], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFRcpY, [ADLPPort00], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFRcpZ>;
defm : ADLPWriteResPair<WriteFRnd, [ADLPPort00_01], 8, [2], 2, 7>;
defm : ADLPWriteResPair<WriteFRndY, [ADLPPort00_01], 8, [2], 2, 8>;
defm : X86WriteResPairUnsupported<WriteFRndZ>;
defm : ADLPWriteResPair<WriteFRsqrt, [ADLPPort00], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFRsqrtX, [ADLPPort00], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFRsqrtY, [ADLPPort00], 4, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFRsqrtZ>;
defm : ADLPWriteResPair<WriteFShuffle, [ADLPPort05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFShuffle256, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WriteFShuffleY, [ADLPPort05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFShuffleZ>;
def : WriteRes<WriteFSign, [ADLPPort00]>;
defm : ADLPWriteResPair<WriteFSqrt, [ADLPPort00], 12, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFSqrt64, [ADLPPort00], 18, [1], 1, 6>;
defm : ADLPWriteResPair<WriteFSqrt64X, [ADLPPort00], 18, [1], 1, 6>;
defm : ADLPWriteResPair<WriteFSqrt64Y, [ADLPPort00], 18, [1], 1, 7>;
defm : X86WriteResPairUnsupported<WriteFSqrt64Z>;
def : WriteRes<WriteFSqrt80, [ADLPPortInvalid, ADLPPort00]> {
  let ResourceCycles = [7, 1];
  let Latency = 21;
}
defm : ADLPWriteResPair<WriteFSqrtX, [ADLPPort00], 12, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFSqrtY, [ADLPPort00], 12, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFSqrtZ>;
defm : X86WriteRes<WriteFStore, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : X86WriteResUnsupported<WriteFStoreNT>;
defm : X86WriteRes<WriteFStoreNTX, [ADLPPort04_09, ADLPPort07_08], 518, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreNTY, [ADLPPort04_09, ADLPPort07_08], 542, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreX, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : X86WriteRes<WriteFStoreY, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : ADLPWriteResPair<WriteFTest, [ADLPPort00], 3, [1]>;
defm : ADLPWriteResPair<WriteFTestY, [ADLPPort00], 5, [1], 1, 6>;
defm : ADLPWriteResPair<WriteFVarBlend, [ADLPPort00_01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFVarBlendY, [ADLPPort00_01_05], 3, [3], 3, 7>;
defm : X86WriteResPairUnsupported<WriteFVarBlendZ>;
defm : ADLPWriteResPair<WriteFVarShuffle, [ADLPPort05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteFVarShuffle256, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WriteFVarShuffleY, [ADLPPort05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteFVarShuffleZ>;
def : WriteRes<WriteFence, [ADLPPort00_06]> {
  let Latency = 2;
}
defm : ADLPWriteResPair<WriteIDiv16, [ADLPPort00_01_05_06_10, ADLPPort01], 16, [1, 3], 4, 4>;
defm : ADLPWriteResPair<WriteIDiv32, [ADLPPort00_01_05_06_10, ADLPPort01], 15, [1, 3], 4, 4>;
defm : ADLPWriteResPair<WriteIDiv64, [ADLPPort01], 18, [3], 3>;
defm : X86WriteRes<WriteIDiv8, [ADLPPort01], 17, [3], 3>;
defm : X86WriteRes<WriteIDiv8Ld, [ADLPPort01], 22, [3], 3>;
defm : ADLPWriteResPair<WriteIMul16, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01], 5, [2, 1, 1], 4>;
defm : ADLPWriteResPair<WriteIMul16Imm, [ADLPPort00_01_05_06_10, ADLPPort01], 4, [1, 1], 2>;
defm : ADLPWriteResPair<WriteIMul16Reg, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteIMul32, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01], 4, [1, 1, 1], 3>;
defm : ADLPWriteResPair<WriteIMul32Imm, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteIMul32Reg, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteIMul64, [ADLPPort01, ADLPPort05], 4, [1, 1], 2>;
defm : ADLPWriteResPair<WriteIMul64Imm, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteIMul64Reg, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteIMul8, [ADLPPort01], 3, [1]>;
def : WriteRes<WriteIMulH, []> {
  let Latency = 3;
}
def : WriteRes<WriteIMulHLd, []> {
  let Latency = 3;
}
def : WriteRes<WriteJump, [ADLPPort00_06]>;
defm : X86WriteRes<WriteJumpLd, [ADLPPort00_06, ADLPPort02_03], 6, [1, 1], 2>;
def : WriteRes<WriteLAHFSAHF, [ADLPPort00_06]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteLDMXCSR, [ADLPPort00, ADLPPort00_01_05, ADLPPort00_06, ADLPPort02_03_11], 7, [1, 1, 1, 1], 4>;
def : WriteRes<WriteLEA, [ADLPPort01]>;
defm : ADLPWriteResPair<WriteLZCNT, [ADLPPort01], 3, [1]>;
def : WriteRes<WriteLoad, [ADLPPort02_03_11]> {
  let Latency = 5;
}
def : WriteRes<WriteMMXMOVMSK, [ADLPPort00]> {
  let Latency = 3;
}
defm : ADLPWriteResPair<WriteMPSAD, [ADLPPort01_05, ADLPPort05], 4, [1, 1], 2, 7>;
defm : ADLPWriteResPair<WriteMPSADY, [ADLPPort01_05, ADLPPort05], 4, [1, 1], 2, 8>;
defm : ADLPWriteResPair<WriteMULX32, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01], 4, [1, 1, 1], 2>;
defm : ADLPWriteResPair<WriteMULX64, [ADLPPort01, ADLPPort05], 4, [1, 1]>;
def : WriteRes<WriteMicrocoded, [ADLPPort00_01_05_06]> {
  let Latency = AlderlakePModel.MaxLatency;
}
def : WriteRes<WriteMove, [ADLPPort00_01_05_06_10]>;
defm : X86WriteRes<WriteNop, [], 1, [], 0>;
defm : X86WriteRes<WritePCmpEStrI, [ADLPPort00, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01, ADLPPort05], 16, [3, 2, 1, 1, 1], 8>;
defm : X86WriteRes<WritePCmpEStrILd, [ADLPPort00, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05], 31, [3, 1, 1, 1, 1, 1], 8>;
defm : X86WriteRes<WritePCmpEStrM, [ADLPPort00, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01, ADLPPort05], 16, [3, 3, 1, 1, 1], 9>;
defm : X86WriteRes<WritePCmpEStrMLd, [ADLPPort00, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05], 17, [3, 2, 1, 1, 1, 1], 9>;
defm : ADLPWriteResPair<WritePCmpIStrI, [ADLPPort00], 11, [3], 3, 20>;
defm : ADLPWriteResPair<WritePCmpIStrM, [ADLPPort00], 11, [3], 3>;
defm : ADLPWriteResPair<WritePHAdd, [ADLPPort00_05, ADLPPort05], 3, [1, 2], 3, 8>;
defm : ADLPWriteResPair<WritePHAddX, [ADLPPort00_01_05, ADLPPort01_05], 2, [1, 2], 3, 7>;
defm : ADLPWriteResPair<WritePHAddY, [ADLPPort00_01_05, ADLPPort01_05], 2, [1, 2], 3, 8>;
defm : ADLPWriteResPair<WritePHMINPOS, [ADLPPort00], 4, [1], 1, 7>;
defm : ADLPWriteResPair<WritePMULLD, [ADLPPort00_01], 10, [2], 2, 8>;
defm : ADLPWriteResPair<WritePMULLDY, [ADLPPort00_01], 10, [2], 2, 8>;
defm : X86WriteResPairUnsupported<WritePMULLDZ>;
defm : ADLPWriteResPair<WritePOPCNT, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WritePSADBW, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WritePSADBWX, [ADLPPort05], 3, [1], 1, 7>;
defm : ADLPWriteResPair<WritePSADBWY, [ADLPPort05], 3, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WritePSADBWZ>;
defm : X86WriteRes<WriteRMW, [ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08], 1, [1, 1, 1], 3>;
defm : X86WriteRes<WriteRotate, [ADLPPort00_01_05_06_10, ADLPPort00_06], 2, [1, 2], 3>;
defm : X86WriteRes<WriteRotateLd, [ADLPPort00_01_05_06_10, ADLPPort00_06], 12, [1, 2], 3>;
defm : X86WriteRes<WriteRotateCL, [ADLPPort00_06], 2, [2], 2>;
defm : X86WriteRes<WriteRotateCLLd, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01], 19, [2, 3, 2], 7>;
defm : X86WriteRes<WriteSETCC, [ADLPPort00_06], 2, [2], 2>;
defm : X86WriteRes<WriteSETCCStore, [ADLPPort00_06, ADLPPort04_09, ADLPPort07_08], 13, [2, 1, 1], 4>;
defm : X86WriteRes<WriteSHDmrcl, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08], 12, [1, 1, 1, 1, 1, 1], 6>;
defm : X86WriteRes<WriteSHDmri, [ADLPPort00_01_05_06_10, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08], 12, [1, 1, 1, 1, 1], 5>;
defm : X86WriteRes<WriteSHDrrcl, [ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01], 5, [1, 1, 1], 3>;
def : WriteRes<WriteSHDrri, [ADLPPort01]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteSTMXCSR, [ADLPPort00, ADLPPort00_06, ADLPPort04_09, ADLPPort07_08], 12, [1, 1, 1, 1], 4>;
def : WriteRes<WriteShift, [ADLPPort00_06]>;
def : WriteRes<WriteShiftLd, [ADLPPort00_06]> {
  let Latency = 12;
}
defm : X86WriteRes<WriteShiftCL, [ADLPPort00_06], 2, [2], 2>;
defm : X86WriteRes<WriteShiftCLLd, [ADLPPort00_06], 12, [2], 2>;
defm : ADLPWriteResPair<WriteShuffle, [ADLPPort05], 1, [1], 1, 8>;
defm : ADLPWriteResPair<WriteShuffle256, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WriteShuffleX, [ADLPPort01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteShuffleY, [ADLPPort01_05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteShuffleZ>;
defm : X86WriteRes<WriteStore, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : X86WriteRes<WriteStoreNT, [ADLPPort04_09, ADLPPort07_08], 512, [1, 1], 2>;
def : WriteRes<WriteSystem, [ADLPPort00_01_05_06]> {
  let Latency = AlderlakePModel.MaxLatency;
}
defm : ADLPWriteResPair<WriteTZCNT, [ADLPPort01], 3, [1]>;
defm : ADLPWriteResPair<WriteVPMOV256, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVarBlend, [ADLPPort00_01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteVarBlendY, [ADLPPort00_01_05], 3, [3], 3, 7>;
defm : X86WriteResPairUnsupported<WriteVarBlendZ>;
defm : ADLPWriteResPair<WriteVarShuffle, [ADLPPort00, ADLPPort05], 3, [1, 1], 2, 8>;
defm : ADLPWriteResPair<WriteVarShuffle256, [ADLPPort05], 3, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVarShuffleX, [ADLPPort01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteVarShuffleY, [ADLPPort01_05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteVarShuffleZ>;
defm : ADLPWriteResPair<WriteVarVecShift, [ADLPPort00_01], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteVarVecShiftY, [ADLPPort00_01], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteVarVecShiftZ>;
defm : ADLPWriteResPair<WriteVecALU, [ADLPPort00], 1, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVecALUX, [ADLPPort00_01], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteVecALUY, [ADLPPort00_01], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteVecALUZ>;
defm : X86WriteRes<WriteVecExtract, [ADLPPort00, ADLPPort01_05], 4, [1, 1], 2>;
defm : X86WriteRes<WriteVecExtractSt, [ADLPPort01_05, ADLPPort04_09, ADLPPort07_08], 19, [1, 1, 1], 3>;
defm : ADLPWriteResPair<WriteVecIMul, [ADLPPort00], 5, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVecIMulX, [ADLPPort00_01], 5, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVecIMulY, [ADLPPort00_01], 5, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteVecIMulZ>;
defm : X86WriteRes<WriteVecInsert, [ADLPPort01_05, ADLPPort05], 4, [1, 1], 2>;
defm : X86WriteRes<WriteVecInsertLd, [ADLPPort01_05, ADLPPort02_03_11], 8, [1, 1], 2>;
def : WriteRes<WriteVecLoad, [ADLPPort02_03_11]> {
  let Latency = 7;
}
def : WriteRes<WriteVecLoadNT, [ADLPPort02_03_11]> {
  let Latency = 7;
}
def : WriteRes<WriteVecLoadNTY, [ADLPPort02_03_11]> {
  let Latency = 8;
}
def : WriteRes<WriteVecLoadX, [ADLPPort02_03_11]> {
  let Latency = 7;
}
def : WriteRes<WriteVecLoadY, [ADLPPort02_03_11]> {
  let Latency = 8;
}
defm : ADLPWriteResPair<WriteVecLogic, [ADLPPort00_05], 1, [1], 1, 8>;
defm : ADLPWriteResPair<WriteVecLogicX, [ADLPPort00_01_05], 1, [1], 1, 7>;
defm : ADLPWriteResPair<WriteVecLogicY, [ADLPPort00_01_05], 1, [1], 1, 8>;
defm : X86WriteResPairUnsupported<WriteVecLogicZ>;
def : WriteRes<WriteVecMOVMSK, [ADLPPort00]> {
  let Latency = 3;
}
def : WriteRes<WriteVecMOVMSKY, [ADLPPort00]> {
  let Latency = 4;
}
defm : X86WriteRes<WriteVecMaskedGatherWriteback, [], 5, [], 0>;
defm : X86WriteRes<WriteVecMaskedLoad, [ADLPPort00_01_05, ADLPPort02_03_11], 8, [1, 1], 2>;
defm : X86WriteRes<WriteVecMaskedLoadY, [ADLPPort00_01_05, ADLPPort02_03_11], 9, [1, 1], 2>;
defm : X86WriteRes<WriteVecMaskedStore32, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteVecMaskedStore32Y, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteVecMaskedStore64, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
defm : X86WriteRes<WriteVecMaskedStore64Y, [ADLPPort00, ADLPPort04_09, ADLPPort07_08], 14, [1, 1, 1], 3>;
def : WriteRes<WriteVecMove, [ADLPPort00_05]>;
def : WriteRes<WriteVecMoveFromGpr, [ADLPPort05]> {
  let Latency = 3;
}
def : WriteRes<WriteVecMoveToGpr, [ADLPPort00]> {
  let Latency = 3;
}
defm : X86WriteRes<WriteVecMoveX, [], 1, [], 0>;
defm : X86WriteRes<WriteVecMoveY, [], 1, [], 0>;
defm : X86WriteResUnsupported<WriteVecMoveZ>;
defm : ADLPWriteResPair<WriteVecShift, [ADLPPort00], 1, [1], 1, 8>;
def : WriteRes<WriteVecShiftImm, [ADLPPort00]>;
def : WriteRes<WriteVecShiftImmX, [ADLPPort00_01]>;
defm : X86WriteResUnsupported<WriteVecShiftImmXLd>;
def : WriteRes<WriteVecShiftImmY, [ADLPPort00_01]>;
defm : X86WriteResUnsupported<WriteVecShiftImmYLd>;
defm : X86WriteResPairUnsupported<WriteVecShiftImmZ>;
defm : X86WriteRes<WriteVecShiftX, [ADLPPort00_01, ADLPPort01_05], 2, [1, 1], 2>;
defm : X86WriteRes<WriteVecShiftXLd, [ADLPPort00_01, ADLPPort02_03_11], 8, [1, 1], 2>;
defm : X86WriteRes<WriteVecShiftY, [ADLPPort00_01, ADLPPort05], 4, [1, 1], 2>;
defm : X86WriteRes<WriteVecShiftYLd, [ADLPPort00_01, ADLPPort02_03_11], 9, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteVecShiftZ>;
defm : X86WriteRes<WriteVecStore, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreNT, [ADLPPort04_09, ADLPPort07_08], 511, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreNTY, [ADLPPort04_09, ADLPPort07_08], 507, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreX, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : X86WriteRes<WriteVecStoreY, [ADLPPort04_09, ADLPPort07_08], 12, [1, 1], 2>;
defm : ADLPWriteResPair<WriteVecTest, [ADLPPort00, ADLPPort05], 4, [1, 1], 2>;
defm : ADLPWriteResPair<WriteVecTestY, [ADLPPort00, ADLPPort05], 6, [1, 1], 2, 6>;
defm : X86WriteRes<WriteXCHG, [ADLPPort00_01_05_06_10], 2, [3], 3>;
def : WriteRes<WriteZero, []>;

// Infered SchedWriteRes and InstRW definition.

def ADLPWriteResGroup0 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 1, 1, 1];
  let Latency = 12;
  let NumMicroOps = 6;
}
def : InstRW<[ADLPWriteResGroup0, ReadAfterLd, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(ADC|SBB)(16|32|64)mr$")>;

def ADLPWriteResGroup1 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup1], (instregex "^JMP(16|32|64)m$",
                                              "^RET(16|32)$",
                                              "^RORX(32|64)mi$")>;
def : InstRW<[ADLPWriteResGroup1, ReadAfterLd, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(ADC|SBB)(8|16|32|64)rm$",
                                                                                                                                         "^AD(C|O)X(32|64)rm$")>;

def ADLPWriteResGroup2 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 13;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup2], (instregex "^(ADC|SBB)8mi$")>;

def ADLPWriteResGroup3 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 1, 1, 1];
  let Latency = 13;
  let NumMicroOps = 6;
}
def : InstRW<[ADLPWriteResGroup3, ReadAfterLd, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(ADC|SBB)8mr$")>;

def ADLPWriteResGroup4 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup4], (instregex "^CMP(8|16|32)mi$",
                                              "^CMP(16|32|64)mi8$",
                                              "^CMP64mi32$",
                                              "^MOV(8|16)rm$",
                                              "^MOVZX16rm8$",
                                              "^POP(16|32)r$")>;
def : InstRW<[ADLPWriteResGroup4, ReadAfterLd], (instregex "^(ADD|CMP|SUB)(8|16|32|64)rm$",
                                                           "^(OR|AND|XOR)(8|16|32)rm$")>;
def : InstRW<[ADLPWriteResGroup4, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^CMP(8|16|32|64)mr$")>;

def ADLPWriteResGroup5 : SchedWriteRes<[]> {
  let NumMicroOps = 0;
}
def : InstRW<[ADLPWriteResGroup5], (instregex "^(ADD|SUB)64ri8$",
                                              "^CLC$",
                                              "^(DE|IN)C64r$",
                                              "^MOV64rr$",
                                              "^MOV64rr_REV$")>;

def ADLPWriteResGroup6 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 13;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup6], (instregex "^(OR|ADD|SUB|XOR)8mi$",
                                              "^AND8mi$",
                                              "^(DEC|NEG|NOT)8m$",
                                              "^INC8m$")>;
def : InstRW<[ADLPWriteResGroup6, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^(OR|ADD|SUB|XOR)8mr$",
                                                                                                                            "^AND8mr$")>;

def ADLPWriteResGroup7 : SchedWriteRes<[ADLPPort01_05, ADLPPort02_03_11]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup7, ReadAfterVecLd], (instregex "^(V?)(ADD|SUB)SSrm_Int$")>;

def ADLPWriteResGroup8 : SchedWriteRes<[ADLPPort01_05]> {
  let Latency = 3;
}
def : InstRW<[ADLPWriteResGroup8], (instregex "^(V?)(ADD|SUB)SSrr_Int$")>;

def ADLPWriteResGroup9 : SchedWriteRes<[ADLPPort02_03, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup9], (instregex "^(ADD|SUB)_FI(16|32)m$",
                                              "^SUBR_FI(16|32)m$")>;

def ADLPWriteResGroup10 : SchedWriteRes<[ADLPPort00_01_05_06_10]> {
  let Latency = 2;
}
def : InstRW<[ADLPWriteResGroup10], (instregex "^(OR|AND|XOR)(16|32|64)ri8$",
                                               "^(OR|AND|XOR)(8|16|32|64)rr$",
                                               "^(OR|AND|XOR)(32|64)i32$",
                                               "^(OR|AND|XOR)(8|32)ri$",
                                               "^(OR|AND|XOR)64ri32$",
                                               "^(OR|AND|XOR)8i8$",
                                               "^TEST(8|16|32|64)rr$",
                                               "^TEST(32|64)i32$",
                                               "^TEST(8|32)ri$",
                                               "^TEST64ri32$",
                                               "^TEST8i8$")>;

def ADLPWriteResGroup11 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup11], (instregex "^TEST(8|16|32)mi$",
                                               "^TEST64mi32$")>;
def : InstRW<[ADLPWriteResGroup11, ReadAfterLd], (instregex "^(OR|AND|XOR)64rm$")>;
def : InstRW<[ADLPWriteResGroup11, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^TEST(8|16|32|64)mr$")>;

def ADLPWriteResGroup12 : SchedWriteRes<[ADLPPort01_05_10, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup12, ReadAfterLd], (instregex "^ANDN(32|64)rm$")>;

def ADLPWriteResGroup13 : SchedWriteRes<[ADLPPort01_05_10]> {
  let Latency = 2;
}
def : InstRW<[ADLPWriteResGroup13], (instregex "^ANDN(32|64)rr$")>;

def ADLPWriteResGroup14 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [5, 2, 1, 1];
  let Latency = 10;
  let NumMicroOps = 9;
}
def : InstRW<[ADLPWriteResGroup14], (instregex "^BT64mr$")>;

def ADLPWriteResGroup15 : SchedWriteRes<[ADLPPort01]> {
  let Latency = 3;
}
def : InstRW<[ADLPWriteResGroup15], (instregex "^(B|PEX)T64rr$",
                                               "^BT(C|R|S)64rr$",
                                               "^PDEP(32|64)rr$",
                                               "^PEXT32rr$")>;

def ADLPWriteResGroup16 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [4, 2, 1, 1, 1, 1];
  let Latency = 17;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup16], (instregex "^BT(C|R|S)64mr$")>;

def ADLPWriteResGroup17 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 7;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup17], (instregex "^CALL(16|32|64)m$")>;

def ADLPWriteResGroup18 : SchedWriteRes<[ADLPPort00_06, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup18], (instregex "^CALL(16|32|64)r$")>;

def ADLPWriteResGroup19 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup19], (instregex "^CALL64pcrel32$",
                                               "^MFENCE$")>;

def ADLPWriteResGroup20 : SchedWriteRes<[ADLPPort01_05]>;
def : InstRW<[ADLPWriteResGroup20], (instregex "^C(BW|DQE|WDE)$",
                                               "^(V?)MOVS(H|L)DUPrr$",
                                               "^(V?)SHUFP(D|S)rri$",
                                               "^VMOVS(H|L)DUPYrr$",
                                               "^VPBLENDWYrri$",
                                               "^VSHUFP(D|S)Yrri$")>;

def ADLPWriteResGroup21 : SchedWriteRes<[ADLPPort00_06]>;
def : InstRW<[ADLPWriteResGroup21], (instregex "^C(DQ|QO|LAC)$",
                                               "^STAC$")>;

def ADLPWriteResGroup22 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup22], (instregex "^CLD$")>;

def ADLPWriteResGroup23 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup23], (instregex "^CLDEMOTE$")>;

def ADLPWriteResGroup24 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 2;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup24], (instregex "^CLFLUSH$")>;

def ADLPWriteResGroup25 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup25], (instregex "^CLFLUSHOPT$")>;

def ADLPWriteResGroup26 : SchedWriteRes<[ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [2, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup26], (instregex "^CLI$")>;

def ADLPWriteResGroup27 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort05]> {
  let ResourceCycles = [6, 1, 3];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup27], (instregex "^CLTS$")>;

def ADLPWriteResGroup28 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 5;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup28], (instregex "^CLWB$",
                                               "^MOV16o(16|32|64)a$")>;

def ADLPWriteResGroup29 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let ResourceCycles = [5, 2];
  let Latency = 6;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup29], (instregex "^CMPS(B|L|Q|W)$")>;

def ADLPWriteResGroup30 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01_05, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [2, 7, 6, 2, 1, 1, 2, 1];
  let Latency = 32;
  let NumMicroOps = 22;
}
def : InstRW<[ADLPWriteResGroup30], (instregex "^CMPXCHG16B$")>;

def ADLPWriteResGroup31 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [4, 7, 2, 1, 1, 1];
  let Latency = 25;
  let NumMicroOps = 16;
}
def : InstRW<[ADLPWriteResGroup31], (instregex "^CMPXCHG8B$")>;

def ADLPWriteResGroup32 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [1, 2, 1, 1, 1];
  let Latency = 13;
  let NumMicroOps = 6;
}
def : InstRW<[ADLPWriteResGroup32], (instregex "^CMPXCHG8rm$")>;

def ADLPWriteResGroup33 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_06, ADLPPort01, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 10, 6, 1, 5, 1];
  let Latency = 18;
  let NumMicroOps = 26;
}
def : InstRW<[ADLPWriteResGroup33], (instregex "^CPUID$")>;

def ADLPWriteResGroup34 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort02_03_11]> {
  let Latency = 26;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup34], (instregex "^(V?)CVT(T?)SD2SIrm_Int$")>;

def ADLPWriteResGroup35 : SchedWriteRes<[ADLPPort00_01, ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup35, ReadAfterVecLd], (instregex "^(V?)CVTSI642SSrm_Int$")>;

def ADLPWriteResGroup36 : SchedWriteRes<[ADLPPort00_01, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup36, ReadDefault, ReadInt2Fpu], (instregex "^(V?)CVTSI642SSrr_Int$")>;

def ADLPWriteResGroup37 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort05]> {
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup37], (instregex "^(V?)CVT(T?)SS2SI64rr_Int$")>;

def ADLPWriteResGroup38 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup38], (instregex "^CWD$",
                                               "^J(E|R)CXZ$")>;

def ADLPWriteResGroup39 : SchedWriteRes<[ADLPPort00_01_05_06]>;
def : InstRW<[ADLPWriteResGroup39], (instregex "^DEC16r_alt$",
                                               "^(LD|ST)_Frr$",
                                               "^MOV16s(m|r)$",
                                               "^MOV(32|64)sr$",
                                               "^SALC$",
                                               "^ST_FPrr$",
                                               "^SYSCALL$")>;

def ADLPWriteResGroup40 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 7;
}
def : InstRW<[ADLPWriteResGroup40], (instregex "^DEC32r_alt$")>;

def ADLPWriteResGroup41 : SchedWriteRes<[ADLPPort00, ADLPPort02_03, ADLPPort05]> {
  let Latency = 30;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup41], (instregex "^DIVR_FI(16|32)m$")>;

def ADLPWriteResGroup42 : SchedWriteRes<[ADLPPort00, ADLPPort02_03_11]> {
  let Latency = 18;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup42, ReadAfterVecLd], (instregex "^(V?)DIVSSrm_Int$")>;

def ADLPWriteResGroup43 : SchedWriteRes<[ADLPPort00]> {
  let Latency = 11;
}
def : InstRW<[ADLPWriteResGroup43], (instregex "^(V?)DIVSSrr_Int$")>;

def ADLPWriteResGroup44 : SchedWriteRes<[ADLPPort00, ADLPPort02_03]> {
  let Latency = 22;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup44], (instregex "^DIV_F(32|64)m$")>;

def ADLPWriteResGroup45 : SchedWriteRes<[ADLPPort00, ADLPPort02_03, ADLPPort05]> {
  let Latency = 25;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup45], (instregex "^DIV_FI(16|32)m$")>;

def ADLPWriteResGroup46 : SchedWriteRes<[ADLPPort00]> {
  let Latency = 20;
}
def : InstRW<[ADLPWriteResGroup46], (instregex "^DIV_F(P?)rST0$",
                                               "^DIV_FST0r$")>;

def ADLPWriteResGroup47 : SchedWriteRes<[ADLPPort00, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [2, 21, 2, 14, 4, 9, 5];
  let Latency = 126;
  let NumMicroOps = 57;
}
def : InstRW<[ADLPWriteResGroup47], (instregex "^ENTER$")>;

def ADLPWriteResGroup48 : SchedWriteRes<[ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup48], (instregex "^(V?)EXTRACTPSmr$",
                                               "^SMSW16m$")>;

def ADLPWriteResGroup49 : SchedWriteRes<[ADLPPort00, ADLPPort05]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup49], (instregex "^(V?)EXTRACTPSrr$",
                                               "^MMX_PEXTRWrr$")>;

def ADLPWriteResGroup50 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort02_03, ADLPPort02_03_07, ADLPPort04, ADLPPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup50], (instregex "^FARCALL64m$")>;

def ADLPWriteResGroup51 : SchedWriteRes<[ADLPPort02_03, ADLPPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup51], (instregex "^FARJMP64m$")>;

def ADLPWriteResGroup52 : SchedWriteRes<[ADLPPort02_03_07, ADLPPort04]> {
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup52], (instregex "^FBSTPm$",
                                               "^(V?)MASKMOVDQU((64)?)$",
                                               "^ST_FP(32|64|80)m$",
                                               "^VMPTRSTm$")>;

def ADLPWriteResGroup53 : SchedWriteRes<[ADLPPort00_05]> {
  let ResourceCycles = [2];
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup53], (instregex "^FDECSTP$")>;

def ADLPWriteResGroup54 : SchedWriteRes<[ADLPPort02_03, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 11;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup54], (instregex "^FICOM(P?)(16|32)m$")>;

def ADLPWriteResGroup55 : SchedWriteRes<[ADLPPort00_05]>;
def : InstRW<[ADLPWriteResGroup55], (instregex "^FINCSTP$",
                                               "^FNOP$",
                                               "^MMX_P(ADD|SUB)(B|D|Q|W)rr$")>;

def ADLPWriteResGroup56 : SchedWriteRes<[ADLPPort00, ADLPPort00_05, ADLPPort02_03]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup56], (instregex "^FLDCW16m$")>;

def ADLPWriteResGroup57 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06, ADLPPort00_05, ADLPPort00_06, ADLPPort02_03]> {
  let ResourceCycles = [2, 39, 5, 10, 8];
  let Latency = 62;
  let NumMicroOps = 64;
}
def : InstRW<[ADLPWriteResGroup57], (instregex "^FLDENVm$")>;

def ADLPWriteResGroup58 : SchedWriteRes<[ADLPPort00_01_05_06]> {
  let ResourceCycles = [4];
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup58], (instregex "^FNCLEX$")>;

def ADLPWriteResGroup59 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort00_05, ADLPPort05]> {
  let ResourceCycles = [6, 3, 6];
  let Latency = 75;
  let NumMicroOps = 15;
}
def : InstRW<[ADLPWriteResGroup59], (instregex "^FNINIT$")>;

def ADLPWriteResGroup60 : SchedWriteRes<[ADLPPort02_03_07, ADLPPort04, ADLPPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup60], (instregex "^FNSTCW16m$")>;

def ADLPWriteResGroup61 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup61], (instregex "^FNSTSW16r$")>;

def ADLPWriteResGroup62 : SchedWriteRes<[ADLPPort00, ADLPPort02_03_07, ADLPPort04]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup62], (instregex "^FNSTSWm$")>;

def ADLPWriteResGroup63 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06, ADLPPort00_06, ADLPPort01, ADLPPort02_03_07, ADLPPort04, ADLPPort05, ADLPPort06]> {
  let ResourceCycles = [9, 30, 21, 1, 11, 11, 16, 1];
  let Latency = 106;
  let NumMicroOps = 100;
}
def : InstRW<[ADLPWriteResGroup63], (instregex "^FSTENVm$")>;

def ADLPWriteResGroup64 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06, ADLPPort00_05, ADLPPort00_06, ADLPPort01_05, ADLPPort02_03, ADLPPort06]> {
  let ResourceCycles = [4, 47, 1, 2, 1, 33, 2];
  let Latency = 63;
  let NumMicroOps = 90;
}
def : InstRW<[ADLPWriteResGroup64], (instregex "^FXRSTOR$")>;

def ADLPWriteResGroup65 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06, ADLPPort00_05, ADLPPort00_06, ADLPPort01_05, ADLPPort02_03, ADLPPort06]> {
  let ResourceCycles = [4, 45, 1, 2, 1, 31, 4];
  let Latency = 63;
  let NumMicroOps = 88;
}
def : InstRW<[ADLPWriteResGroup65], (instregex "^FXRSTOR64$")>;

def ADLPWriteResGroup66 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [2, 5, 10, 10, 2, 38, 5, 38];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 110;
}
def : InstRW<[ADLPWriteResGroup66], (instregex "^FXSAVE((64)?)$")>;

def ADLPWriteResGroup67 : SchedWriteRes<[ADLPPort00_01, ADLPPort02_03_11]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup67, ReadAfterVecXLd], (instregex "^(V?)GF2P8AFFINE((INV)?)QBrmi$",
                                                                "^(V?)GF2P8MULBrm$",
                                                                "^VGF2P8AFFINE((INV)?)QBYrmi$",
                                                                "^VGF2P8MULBYrm$")>;

def ADLPWriteResGroup68 : SchedWriteRes<[ADLPPort00_01]> {
  let Latency = 5;
}
def : InstRW<[ADLPWriteResGroup68], (instregex "^(V?)GF2P8AFFINE((INV)?)QBrri$",
                                               "^(V?)GF2P8MULBrr$",
                                               "^VGF2P8AFFINE((INV)?)QBYrri$",
                                               "^VGF2P8MULBYrr$")>;

def ADLPWriteResGroup69 : SchedWriteRes<[ADLPPort02_03, ADLPPort05]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup69], (instregex "^ILD_F(16|32|64)m$")>;

def ADLPWriteResGroup70 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 5, 26, 19, 2, 7, 21];
  let Latency = 35;
  let NumMicroOps = 87;
}
def : InstRW<[ADLPWriteResGroup70], (instregex "^IN16ri$")>;

def ADLPWriteResGroup71 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 1, 4, 26, 19, 3, 7, 20];
  let Latency = 35;
  let NumMicroOps = 87;
}
def : InstRW<[ADLPWriteResGroup71], (instregex "^IN16rr$")>;

def ADLPWriteResGroup72 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 6, 28, 21, 2, 10, 20];
  let Latency = 35;
  let NumMicroOps = 94;
}
def : InstRW<[ADLPWriteResGroup72], (instregex "^IN32ri$")>;

def ADLPWriteResGroup73 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 9, 28, 21, 2, 11, 21];
  let NumMicroOps = 99;
}
def : InstRW<[ADLPWriteResGroup73], (instregex "^IN32rr$")>;

def ADLPWriteResGroup74 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 6, 25, 19, 2, 8, 20];
  let Latency = 35;
  let NumMicroOps = 87;
}
def : InstRW<[ADLPWriteResGroup74], (instregex "^IN8ri$")>;

def ADLPWriteResGroup75 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [7, 6, 25, 19, 2, 7, 20];
  let Latency = 35;
  let NumMicroOps = 86;
}
def : InstRW<[ADLPWriteResGroup75], (instregex "^IN8rr$")>;

def ADLPWriteResGroup76 : SchedWriteRes<[ADLPPort00_06]> {
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup76], (instregex "^INC16r_alt$")>;

def ADLPWriteResGroup77 : SchedWriteRes<[ADLPPort02_03_11]> {
  let Latency = 7;
}
def : InstRW<[ADLPWriteResGroup77], (instregex "^INC32r_alt$",
                                               "^(V?)MOV(D|SH|SL)DUPrm$",
                                               "^VBROADCASTSSrm$",
                                               "^VPBROADCAST(D|Q)rm$")>;

def ADLPWriteResGroup78 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [7, 6, 24, 17, 8, 1, 19, 1];
  let Latency = 20;
  let NumMicroOps = 83;
}
def : InstRW<[ADLPWriteResGroup78], (instregex "^INSB$")>;

def ADLPWriteResGroup79 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [7, 1, 5, 1, 27, 17, 11, 1, 21, 1];
  let Latency = 20;
  let NumMicroOps = 92;
}
def : InstRW<[ADLPWriteResGroup79], (instregex "^INSL$")>;

def ADLPWriteResGroup80 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [7, 1, 4, 1, 25, 17, 1, 9, 1, 19, 1];
  let Latency = 20;
  let NumMicroOps = 86;
}
def : InstRW<[ADLPWriteResGroup80], (instregex "^INSW$")>;

def ADLPWriteResGroup81 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [5, 4, 8, 6, 2, 5, 7, 5];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 42;
}
def : InstRW<[ADLPWriteResGroup81], (instregex "^INVLPG$")>;

def ADLPWriteResGroup82 : SchedWriteRes<[ADLPPort02_03_07, ADLPPort04, ADLPPort05]> {
  let Latency = 4;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup82], (instregex "^IST(T?)_FP(16|32|64)m$",
                                               "^IST_F(16|32)m$")>;

def ADLPWriteResGroup83 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort00_06]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup83], (instregex "^JCXZ$")>;

def ADLPWriteResGroup84 : SchedWriteRes<[]> {
  let Latency = 0;
  let NumMicroOps = 0;
}
def : InstRW<[ADLPWriteResGroup84], (instregex "^JMP_1$",
                                               "^VZEROUPPER$")>;

def ADLPWriteResGroup85 : SchedWriteRes<[ADLPPort00, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [8, 2, 14, 3, 1];
  let Latency = 198;
  let NumMicroOps = 81;
}
def : InstRW<[ADLPWriteResGroup85], (instregex "^LAR16rm$")>;

def ADLPWriteResGroup86 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 3, 1, 8, 5, 1, 2, 1];
  let Latency = 66;
  let NumMicroOps = 22;
}
def : InstRW<[ADLPWriteResGroup86], (instregex "^LAR16rr$")>;

def ADLPWriteResGroup87 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 2, 2, 9, 5, 3, 1];
  let Latency = 71;
  let NumMicroOps = 85;
}
def : InstRW<[ADLPWriteResGroup87], (instregex "^LAR32rm$")>;

def ADLPWriteResGroup88 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 3, 1, 8, 5, 1, 2, 1];
  let Latency = 65;
  let NumMicroOps = 22;
}
def : InstRW<[ADLPWriteResGroup88], (instregex "^LAR(32|64)rr$")>;

def ADLPWriteResGroup89 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 2, 2, 9, 5, 3, 1];
  let Latency = 71;
  let NumMicroOps = 87;
}
def : InstRW<[ADLPWriteResGroup89], (instregex "^LAR64rm$")>;

def ADLPWriteResGroup90 : SchedWriteRes<[ADLPPort02_03]> {
  let Latency = 7;
}
def : InstRW<[ADLPWriteResGroup90], (instregex "^LD_F(32|64|80)m$")>;

def ADLPWriteResGroup91 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup91], (instregex "^LEA16r$")>;

def ADLPWriteResGroup92 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let ResourceCycles = [3, 1];
  let Latency = 6;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup92], (instregex "^LEAVE$",
                                               "^(LOD|SCA)S(B|W)$",
                                               "^SCAS(L|Q)$")>;

def ADLPWriteResGroup93 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 1];
  let Latency = 6;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup93], (instregex "^LEAVE64$")>;

def ADLPWriteResGroup94 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [1, 2, 4, 3, 2, 1, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 14;
}
def : InstRW<[ADLPWriteResGroup94], (instregex "^LGDT64m$")>;

def ADLPWriteResGroup95 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [1, 1, 5, 3, 2, 1, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 14;
}
def : InstRW<[ADLPWriteResGroup95], (instregex "^LIDT64m$")>;

def ADLPWriteResGroup96 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [5, 3, 2, 1, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 12;
}
def : InstRW<[ADLPWriteResGroup96], (instregex "^LLDT16m$")>;

def ADLPWriteResGroup97 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [1, 4, 3, 1, 1, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 11;
}
def : InstRW<[ADLPWriteResGroup97], (instregex "^LLDT16r$")>;

def ADLPWriteResGroup98 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [1, 1, 2, 8, 3, 1, 2, 7, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 27;
}
def : InstRW<[ADLPWriteResGroup98], (instregex "^LMSW16m$")>;

def ADLPWriteResGroup99 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [5, 7, 1, 2, 5, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 22;
}
def : InstRW<[ADLPWriteResGroup99], (instregex "^LMSW16r$")>;

def ADLPWriteResGroup100 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 1];
  let Latency = 5;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup100], (instregex "^LODS(L|Q)$")>;

def ADLPWriteResGroup101 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [2, 4, 1];
  let Latency = 3;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup101], (instregex "^LOOP$")>;

def ADLPWriteResGroup102 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [4, 6, 1];
  let Latency = 3;
  let NumMicroOps = 11;
}
def : InstRW<[ADLPWriteResGroup102], (instregex "^LOOPE$")>;

def ADLPWriteResGroup103 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [4, 6, 1];
  let Latency = 2;
  let NumMicroOps = 11;
}
def : InstRW<[ADLPWriteResGroup103], (instregex "^LOOPNE$")>;

def ADLPWriteResGroup104 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort02_03, ADLPPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup104], (instregex "^LRET64$")>;

def ADLPWriteResGroup105 : SchedWriteRes<[ADLPPort00, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 5, 3, 3, 1];
  let Latency = 70;
  let NumMicroOps = 13;
}
def : InstRW<[ADLPWriteResGroup105], (instregex "^LSL(16|32|64)rm$")>;

def ADLPWriteResGroup106 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 4, 4, 3, 2, 1];
  let Latency = 63;
  let NumMicroOps = 15;
}
def : InstRW<[ADLPWriteResGroup106], (instregex "^LSL(16|32|64)rr$")>;

def ADLPWriteResGroup107 : SchedWriteRes<[ADLPPort00_01, ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 24;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup107], (instregex "^MMX_CVT(T?)PD2PIrm$")>;

def ADLPWriteResGroup108 : SchedWriteRes<[ADLPPort00_01, ADLPPort05]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup108], (instregex "^MMX_CVT(T?)PD2PIrr$")>;

def ADLPWriteResGroup109 : SchedWriteRes<[ADLPPort00_01, ADLPPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup109], (instregex "^MMX_CVTPI2PDrr$")>;

def ADLPWriteResGroup110 : SchedWriteRes<[ADLPPort00, ADLPPort00_01]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup110], (instregex "^MMX_CVTPI2PSrr$")>;

def ADLPWriteResGroup111 : SchedWriteRes<[ADLPPort00, ADLPPort02_03_11]> {
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup111], (instregex "^MMX_CVT(T?)PS2PIrm$")>;

def ADLPWriteResGroup112 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup112], (instregex "^MMX_CVT(T?)PS2PIrr$")>;

def ADLPWriteResGroup113 : SchedWriteRes<[ADLPPort00, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 1];
  let Latency = 12;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup113], (instregex "^MMX_MASKMOVQ((64)?)$")>;

def ADLPWriteResGroup114 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 18;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup114], (instregex "^MMX_MOVD64mr$")>;

def ADLPWriteResGroup115 : SchedWriteRes<[ADLPPort02_03_11]> {
  let Latency = 8;
}
def : InstRW<[ADLPWriteResGroup115], (instregex "^MMX_MOV(D|Q)64rm$",
                                                "^VBROADCAST(F|I)128$",
                                                "^VBROADCASTS(D|S)Yrm$",
                                                "^VMOV(D|SH|SL)DUPYrm$",
                                                "^VPBROADCAST(D|Q)Yrm$")>;

def ADLPWriteResGroup116 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort00_05]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup116], (instregex "^MMX_MOVDQ2Qrr$")>;

def ADLPWriteResGroup117 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup117], (instregex "^MMX_MOVQ2DQrr$")>;

def ADLPWriteResGroup118 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup118, ReadAfterVecLd], (instregex "^MMX_PACKSS(DW|WB)rm$",
                                                                "^MMX_PACKUSWBrm$")>;

def ADLPWriteResGroup119 : SchedWriteRes<[ADLPPort05]> {
  let ResourceCycles = [2];
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup119], (instregex "^MMX_PACKSS(DW|WB)rr$",
                                                "^MMX_PACKUSWBrr$")>;
def : InstRW<[ADLPWriteResGroup119, ReadDefault, ReadInt2Fpu], (instregex "^MMX_PINSRWrr$")>;

def ADLPWriteResGroup120 : SchedWriteRes<[ADLPPort00_05, ADLPPort02_03_11]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup120, ReadAfterVecLd], (instregex "^MMX_P(ADD|SUB)(B|D|Q|W)rm$")>;

def ADLPWriteResGroup121 : SchedWriteRes<[ADLPPort00, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 1, 2];
  let Latency = 11;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup121, ReadAfterVecLd], (instregex "^MMX_PH(ADD|SUB)SWrm$")>;

def ADLPWriteResGroup122 : SchedWriteRes<[ADLPPort00, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup122], (instregex "^MMX_PH(ADD|SUB)SWrr$")>;

def ADLPWriteResGroup123 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup123], (instregex "^VPBROADCAST(B|W)Yrm$")>;
def : InstRW<[ADLPWriteResGroup123, ReadAfterLd], (instregex "^MMX_PINSRWrm$")>;
def : InstRW<[ADLPWriteResGroup123, ReadAfterVecYLd], (instregex "^VPALIGNRYrmi$")>;

def ADLPWriteResGroup124 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup124], (instregex "^MOV16ao(16|32|64)$")>;

def ADLPWriteResGroup125 : SchedWriteRes<[ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup125], (instregex "^MOV16ms$",
                                                "^MOVBE32mr$",
                                                "^PUSH(F|G)S(16|32)$")>;

def ADLPWriteResGroup126 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01]> {
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup126], (instregex "^MOV(16|32|64)rs$",
                                                "^S(TR|LDT)16r$")>;

def ADLPWriteResGroup127 : SchedWriteRes<[ADLPPort02_03_11]>;
def : InstRW<[ADLPWriteResGroup127], (instregex "^MOV32ao(16|32|64)$",
                                                "^MOV64ao64$")>;

def ADLPWriteResGroup128 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup128], (instregex "^MOV(8|32)o(16|32)a$",
                                                "^MOV(8|32|64)o64a$")>;

def ADLPWriteResGroup129 : SchedWriteRes<[ADLPPort00_01_05_06_10]> {
  let Latency = 0;
}
def : InstRW<[ADLPWriteResGroup129], (instregex "^MOV32rr$",
                                                "^MOV32rr_REV$",
                                                "^MOVZX(32|64)rr8$")>;

def ADLPWriteResGroup130 : SchedWriteRes<[ADLPPort02_03_11]> {
  let Latency = 5;
}
def : InstRW<[ADLPWriteResGroup130], (instregex "^MOV64ao32$",
                                                "^MOVZX(32|64)rm(8|16)$")>;

def ADLPWriteResGroup131 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [1, 2, 4, 16, 7, 2, 2, 12, 2];
  let Latency = 217;
  let NumMicroOps = 48;
}
def : InstRW<[ADLPWriteResGroup131], (instregex "^MOV64dr$")>;

def ADLPWriteResGroup132 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup132], (instregex "^MOV64o32a$")>;

def ADLPWriteResGroup133 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort05]> {
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup133], (instregex "^MOV64rc$")>;

def ADLPWriteResGroup134 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort05]> {
  let ResourceCycles = [3, 4, 8, 4, 2, 3];
  let Latency = 181;
  let NumMicroOps = 24;
}
def : InstRW<[ADLPWriteResGroup134], (instregex "^MOV64rd$")>;

def ADLPWriteResGroup135 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup135], (instregex "^MOV8ao(16|32|64)$")>;

def ADLPWriteResGroup136 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup136], (instregex "^MOV8m(i|r)$")>;

def ADLPWriteResGroup137 : SchedWriteRes<[ADLPPort00_06, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup137], (instregex "^MOVBE16mr$")>;

def ADLPWriteResGroup138 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup138], (instregex "^MOVBE16rm$")>;

def ADLPWriteResGroup139 : SchedWriteRes<[ADLPPort01, ADLPPort02_03_11]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup139], (instregex "^MOVBE32rm$")>;

def ADLPWriteResGroup140 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup140], (instregex "^MOVBE64mr$",
                                                "^PUSHF16$",
                                                "^SLDT16m$",
                                                "^STRm$")>;

def ADLPWriteResGroup141 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup141], (instregex "^MOVBE64rm$")>;

def ADLPWriteResGroup142 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup142], (instregex "^MOVDIR64B(16|32|64)$")>;

def ADLPWriteResGroup143 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 511;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup143], (instregex "^MOVDIRI32$")>;

def ADLPWriteResGroup144 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 514;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup144], (instregex "^MOVDIRI64$")>;

def ADLPWriteResGroup145 : SchedWriteRes<[ADLPPort01_05, ADLPPort02_03_11]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup145, ReadAfterVecXLd], (instregex "^(V?)MOVLP(D|S)rm$",
                                                                 "^(V?)SHUFP(D|S)rmi$")>;

def ADLPWriteResGroup146 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 512;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup146], (instregex "^MOVNTDQmr$")>;

def ADLPWriteResGroup147 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 518;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup147], (instregex "^MOVNTImr$")>;

def ADLPWriteResGroup148 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [4, 1, 1, 1];
  let Latency = 8;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup148], (instregex "^MOVSB$")>;

def ADLPWriteResGroup149 : SchedWriteRes<[ADLPPort00_01_05]>;
def : InstRW<[ADLPWriteResGroup149], (instregex "^(V?)MOVS(D|S)rr$",
                                                "^(V?)MOVS(D|S)rr_REV$",
                                                "^(V?)P(ADD|SUB)(B|D|Q|W)rr$",
                                                "^VP(ADD|SUB)(B|D|Q|W)Yrr$",
                                                "^VPBLENDDrri$")>;

def ADLPWriteResGroup150 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [4, 1, 1, 1];
  let Latency = 7;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup150], (instregex "^MOVS(L|Q|W)$")>;

def ADLPWriteResGroup151 : SchedWriteRes<[ADLPPort01_05_10, ADLPPort02_03_11]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup151], (instregex "^MOVSX16rm8$")>;

def ADLPWriteResGroup152 : SchedWriteRes<[ADLPPort01_05_10]>;
def : InstRW<[ADLPWriteResGroup152], (instregex "^MOVSX(16|32|64)rr8$",
                                                "^MOVSX(32|64)rr16$",
                                                "^MOVSX64rr32$")>;

def ADLPWriteResGroup153 : SchedWriteRes<[ADLPPort02_03_11]> {
  let Latency = 6;
}
def : InstRW<[ADLPWriteResGroup153], (instregex "^MOVSX(32|64)rm(8|16)$",
                                                "^MOVSX64rm32$")>;

def ADLPWriteResGroup154 : SchedWriteRes<[ADLPPort00_01]> {
  let Latency = 4;
}
def : InstRW<[ADLPWriteResGroup154], (instregex "^(V?)MULSSrr_Int$")>;

def ADLPWriteResGroup155 : SchedWriteRes<[ADLPPort00, ADLPPort02_03]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup155], (instregex "^MUL_F(32|64)m$")>;

def ADLPWriteResGroup156 : SchedWriteRes<[ADLPPort00, ADLPPort02_03, ADLPPort05]> {
  let Latency = 14;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup156], (instregex "^MUL_FI(16|32)m$")>;

def ADLPWriteResGroup157 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort05, ADLPPort06]> {
  let ResourceCycles = [7, 1, 2];
  let Latency = 20;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup157], (instregex "^MWAITrr$")>;

def ADLPWriteResGroup158 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [6, 4, 1, 28, 15, 7, 1, 16, 1];
  let Latency = 35;
  let NumMicroOps = 79;
}
def : InstRW<[ADLPWriteResGroup158], (instregex "^OUT16ir$")>;

def ADLPWriteResGroup159 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [6, 6, 27, 15, 7, 1, 16, 1];
  let Latency = 35;
  let NumMicroOps = 79;
}
def : InstRW<[ADLPWriteResGroup159], (instregex "^OUT16rr$")>;

def ADLPWriteResGroup160 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [6, 4, 1, 30, 15, 9, 1, 18, 1];
  let Latency = 35;
  let NumMicroOps = 85;
}
def : InstRW<[ADLPWriteResGroup160], (instregex "^OUT32ir$")>;

def ADLPWriteResGroup161 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [6, 6, 29, 15, 9, 1, 18, 1];
  let Latency = 35;
  let NumMicroOps = 85;
}
def : InstRW<[ADLPWriteResGroup161], (instregex "^OUT32rr$")>;

def ADLPWriteResGroup162 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [5, 5, 1, 25, 15, 5, 1, 15, 1];
  let Latency = 35;
  let NumMicroOps = 73;
}
def : InstRW<[ADLPWriteResGroup162], (instregex "^OUT8ir$")>;

def ADLPWriteResGroup163 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [5, 5, 26, 15, 5, 1, 15, 1];
  let Latency = 35;
  let NumMicroOps = 73;
}
def : InstRW<[ADLPWriteResGroup163], (instregex "^OUT8rr$")>;

def ADLPWriteResGroup164 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [7, 6, 25, 16, 7, 1, 17, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 80;
}
def : InstRW<[ADLPWriteResGroup164], (instregex "^OUTSB$")>;

def ADLPWriteResGroup165 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [7, 6, 28, 16, 10, 1, 20, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 89;
}
def : InstRW<[ADLPWriteResGroup165], (instregex "^OUTSL$")>;

def ADLPWriteResGroup166 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [6, 1, 5, 27, 16, 8, 1, 18, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 83;
}
def : InstRW<[ADLPWriteResGroup166], (instregex "^OUTSW$")>;

def ADLPWriteResGroup167 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup167, ReadAfterVecXLd], (instregex "^(V?)PACK(S|U)S(DW|WB)rm$",
                                                                 "^(V?)PCMPGTQrm$")>;

def ADLPWriteResGroup168 : SchedWriteRes<[ADLPPort05]> {
  let Latency = 3;
}
def : InstRW<[ADLPWriteResGroup168], (instregex "^(V?)PACK(S|U)S(DW|WB)rr$",
                                                "^(V?)PCMPGTQrr$",
                                                "^VPACK(S|U)S(DW|WB)Yrr$",
                                                "^VPCMPGTQYrr$")>;

def ADLPWriteResGroup169 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort02_03_11]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup169, ReadAfterVecXLd], (instregex "^(V?)P(ADD|SUB)(B|D|Q|W)rm$",
                                                                 "^VPBLENDDrmi$")>;

def ADLPWriteResGroup170 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup170], (instregex "^VPBROADCAST(B|W)rm$")>;
def : InstRW<[ADLPWriteResGroup170, ReadAfterVecXLd], (instregex "^(V?)PALIGNRrmi$")>;

def ADLPWriteResGroup171 : SchedWriteRes<[ADLPPort05]>;
def : InstRW<[ADLPWriteResGroup171], (instregex "^(V?)PALIGNRrri$",
                                                "^VPALIGNRYrri$",
                                                "^VPBROADCAST(B|D|Q|W)rr$")>;

def ADLPWriteResGroup172 : SchedWriteRes<[ADLPPort00_06, ADLPPort05]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup172], (instregex "^PAUSE$")>;

def ADLPWriteResGroup173 : SchedWriteRes<[ADLPPort01, ADLPPort02_03_11]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup173, ReadAfterLd], (instregex "^P(DEP|EXT)(32|64)rm$")>;

def ADLPWriteResGroup174 : SchedWriteRes<[ADLPPort01_05, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup174], (instregex "^(V?)PEXTR(D|Q)mr$")>;

def ADLPWriteResGroup175 : SchedWriteRes<[ADLPPort00_01, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 2, 1];
  let Latency = 9;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup175, ReadAfterVecXLd], (instregex "^(V?)PH(ADD|SUB)SWrm$")>;

def ADLPWriteResGroup176 : SchedWriteRes<[ADLPPort00_01, ADLPPort01_05]> {
  let ResourceCycles = [1, 2];
  let Latency = 2;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup176], (instregex "^(V?)PH(ADD|SUB)SWrr$",
                                                "^VPH(ADD|SUB)SWYrr$")>;

def ADLPWriteResGroup177 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup177], (instregex "^POP(16|32|64)rmm$",
                                                "^PUSH(16|32)rmm$")>;

def ADLPWriteResGroup178 : SchedWriteRes<[ADLPPort02_03]> {
  let Latency = 5;
}
def : InstRW<[ADLPWriteResGroup178], (instregex "^POPA(16|32)$",
                                                "^POPF32$")>;

def ADLPWriteResGroup179 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [6, 2, 1, 1];
  let Latency = 5;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup179], (instregex "^POPF16$")>;

def ADLPWriteResGroup180 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 1, 1];
  let Latency = 5;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup180], (instregex "^POPF64$")>;

def ADLPWriteResGroup181 : SchedWriteRes<[ADLPPort02_03_11]> {
  let Latency = 0;
}
def : InstRW<[ADLPWriteResGroup181], (instregex "^PREFETCH(T0|T1|T2|NTA)$")>;

def ADLPWriteResGroup182 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11, ADLPPort06]> {
  let ResourceCycles = [1, 1, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup182], (instregex "^PTWRITE((64)?)m$")>;

def ADLPWriteResGroup183 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort06]> {
  let ResourceCycles = [1, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup183], (instregex "^PTWRITE64r$")>;

def ADLPWriteResGroup184 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort06]> {
  let ResourceCycles = [2, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup184], (instregex "^PTWRITEr$")>;

def ADLPWriteResGroup185 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup185], (instregex "^PUSH64r$")>;

def ADLPWriteResGroup186 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup186], (instregex "^PUSH64rmm$")>;

def ADLPWriteResGroup187 : SchedWriteRes<[ADLPPort02_03_07, ADLPPort04]>;
def : InstRW<[ADLPWriteResGroup187], (instregex "^PUSHA(16|32)$",
                                                "^PUSHF32$",
                                                "^ST_F(32|64)m$")>;

def ADLPWriteResGroup188 : SchedWriteRes<[ADLPPort00_06, ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup188], (instregex "^PUSHF64$")>;

def ADLPWriteResGroup189 : SchedWriteRes<[ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup189], (instregex "^PUSH(F|G)S64$")>;

def ADLPWriteResGroup190 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [2, 3, 2];
  let Latency = 8;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup190], (instregex "^RC(L|R)(16|32|64)rCL$")>;

def ADLPWriteResGroup191 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let ResourceCycles = [1, 2];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup191, WriteRMW], (instregex "^RC(L|R)8m(1|i)$")>;

def ADLPWriteResGroup192 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [1, 5, 2];
  let Latency = 20;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup192, WriteRMW], (instregex "^RCL8mCL$")>;

def ADLPWriteResGroup193 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [2, 5, 2];
  let Latency = 7;
  let NumMicroOps = 9;
}
def : InstRW<[ADLPWriteResGroup193], (instregex "^RCL8rCL$")>;

def ADLPWriteResGroup194 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [2, 4, 3];
  let Latency = 20;
  let NumMicroOps = 9;
}
def : InstRW<[ADLPWriteResGroup194, WriteRMW], (instregex "^RCR8mCL$")>;

def ADLPWriteResGroup195 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [3, 4, 3];
  let Latency = 9;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup195], (instregex "^RCR8rCL$")>;

def ADLPWriteResGroup196 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_05, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort01_05_10, ADLPPort05]> {
  let ResourceCycles = [1, 6, 1, 10, 20, 8, 5, 1, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 54;
}
def : InstRW<[ADLPWriteResGroup196], (instregex "^RDMSR$")>;

def ADLPWriteResGroup197 : SchedWriteRes<[ADLPPort01]> {
  let Latency = AlderlakePModel.MaxLatency;
}
def : InstRW<[ADLPWriteResGroup197], (instregex "^RDPID64$")>;

def ADLPWriteResGroup198 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup198], (instregex "^RDPKRUr$")>;

def ADLPWriteResGroup199 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort05]> {
  let ResourceCycles = [9, 6, 2, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 18;
}
def : InstRW<[ADLPWriteResGroup199], (instregex "^RDPMC$")>;

def ADLPWriteResGroup200 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [2, 3, 2, 5, 7, 3, 1, 2];
  let Latency = 1386;
  let NumMicroOps = 25;
}
def : InstRW<[ADLPWriteResGroup200], (instregex "^RDRAND16r$")>;

def ADLPWriteResGroup201 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [2, 3, 2, 5, 7, 3, 1, 2];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 25;
}
def : InstRW<[ADLPWriteResGroup201], (instregex "^RDRAND(32|64)r$")>;

def ADLPWriteResGroup202 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [2, 3, 3, 5, 7, 1, 4];
  let Latency = 1381;
  let NumMicroOps = 25;
}
def : InstRW<[ADLPWriteResGroup202], (instregex "^RDSEED16r$")>;

def ADLPWriteResGroup203 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [2, 3, 3, 5, 7, 1, 4];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 25;
}
def : InstRW<[ADLPWriteResGroup203], (instregex "^RDSEED(32|64)r$")>;

def ADLPWriteResGroup204 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort05]> {
  let ResourceCycles = [5, 6, 3, 1];
  let Latency = 18;
  let NumMicroOps = 15;
}
def : InstRW<[ADLPWriteResGroup204], (instregex "^RDTSC$")>;

def ADLPWriteResGroup205 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort05]> {
  let ResourceCycles = [2, 2, 1, 2, 7, 4, 3];
  let Latency = 42;
  let NumMicroOps = 21;
}
def : InstRW<[ADLPWriteResGroup205], (instregex "^RDTSCP$")>;

def ADLPWriteResGroup206 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup206], (instregex "^RET64$")>;

def ADLPWriteResGroup207 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 1];
  let Latency = 6;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup207], (instregex "^RETI(16|32|64)$")>;

def ADLPWriteResGroup208 : SchedWriteRes<[]>;
def : InstRW<[ADLPWriteResGroup208], (instregex "^REX64_PREFIX$")>;

def ADLPWriteResGroup209 : SchedWriteRes<[ADLPPort00_06]> {
  let ResourceCycles = [2];
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup209, WriteRMW], (instregex "^RO(L|R)(16|32|64)m(1|i|CL)$")>;

def ADLPWriteResGroup210 : SchedWriteRes<[ADLPPort00_06]> {
  let ResourceCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup210], (instregex "^RO(L|R)(8|16|32|64)r(1|i)$")>;

def ADLPWriteResGroup211 : SchedWriteRes<[ADLPPort00_06]> {
  let ResourceCycles = [2];
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup211, WriteRMW], (instregex "^RO(L|R)8m(1|i)$",
                                                          "^(ROL|SAR|SHR)8mCL$",
                                                          "^(ROR|SHL)8mCL$")>;

def ADLPWriteResGroup212 : SchedWriteRes<[ADLPPort00_06]> {
  let ResourceCycles = [2];
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup212], (instregex "^SAHF$")>;

def ADLPWriteResGroup213 : SchedWriteRes<[ADLPPort00_06]> {
  let Latency = 13;
}
def : InstRW<[ADLPWriteResGroup213, WriteRMW], (instregex "^S(A|H)R8m(1|i)$",
                                                          "^SHL8m(1|i)$")>;

def ADLPWriteResGroup214 : SchedWriteRes<[ADLPPort00_06, ADLPPort02_03_11]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup214, ReadAfterLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^S(A|H)RX(32|64)rm$",
                                                                                                                              "^SHLX(32|64)rm$")>;

def ADLPWriteResGroup215 : SchedWriteRes<[ADLPPort00_06]> {
  let Latency = 3;
}
def : InstRW<[ADLPWriteResGroup215], (instregex "^S(A|H)RX(32|64)rr$",
                                                "^SHLX(32|64)rr$")>;

def ADLPWriteResGroup216 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 2, 1, 1, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup216], (instregex "^SERIALIZE$")>;

def ADLPWriteResGroup217 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup217], (instregex "^SFENCE$")>;

def ADLPWriteResGroup218 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [1, 2, 2, 2];
  let Latency = 21;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup218], (instregex "^S(G|I)DT64m$")>;

def ADLPWriteResGroup219 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 9;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup219, ReadAfterVecXLd], (instregex "^SHA1MSG1rm$")>;

def ADLPWriteResGroup220 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup220], (instregex "^SHA1MSG1rr$")>;

def ADLPWriteResGroup221 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 2, 1, 2, 1];
  let Latency = 13;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup221, ReadAfterVecXLd], (instregex "^SHA1MSG2rm$")>;

def ADLPWriteResGroup222 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_06, ADLPPort01_05]> {
  let ResourceCycles = [2, 2, 1, 2];
  let Latency = 6;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup222], (instregex "^SHA1MSG2rr$")>;

def ADLPWriteResGroup223 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort01_05, ADLPPort02_03_11]> {
  let Latency = 8;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup223, ReadAfterVecXLd], (instregex "^SHA1NEXTErm$")>;

def ADLPWriteResGroup224 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort01_05]> {
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup224], (instregex "^SHA1NEXTErr$")>;

def ADLPWriteResGroup225 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup225, ReadAfterVecXLd], (instregex "^SHA1RNDS4rmi$",
                                                                 "^SHA256RNDS2rm$")>;

def ADLPWriteResGroup226 : SchedWriteRes<[ADLPPort05]> {
  let Latency = 6;
}
def : InstRW<[ADLPWriteResGroup226], (instregex "^SHA1RNDS4rri$",
                                                "^SHA256RNDS2rr$")>;

def ADLPWriteResGroup227 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_06, ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [3, 2, 1, 1, 1];
  let Latency = 12;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup227, ReadAfterVecXLd], (instregex "^SHA256MSG1rm$")>;

def ADLPWriteResGroup228 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_01_05, ADLPPort00_06, ADLPPort05]> {
  let ResourceCycles = [3, 2, 1, 1];
  let Latency = 5;
  let NumMicroOps = 7;
}
def : InstRW<[ADLPWriteResGroup228], (instregex "^SHA256MSG1rr$")>;

def ADLPWriteResGroup229 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 13;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup229, ReadAfterVecXLd], (instregex "^SHA256MSG2rm$")>;

def ADLPWriteResGroup230 : SchedWriteRes<[ADLPPort05]> {
  let ResourceCycles = [2];
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup230], (instregex "^SHA256MSG2rr$")>;

def ADLPWriteResGroup231 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 13;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup231], (instregex "^SHRD16mri8$")>;

def ADLPWriteResGroup232 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup232], (instregex "^SLDT(32|64)r$")>;

def ADLPWriteResGroup233 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort05]> {
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup233], (instregex "^SMSW16r$")>;

def ADLPWriteResGroup234 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort05]> {
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup234], (instregex "^SMSW(32|64)r$")>;

def ADLPWriteResGroup235 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup235], (instregex "^STD$")>;

def ADLPWriteResGroup236 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [1, 4, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 6;
}
def : InstRW<[ADLPWriteResGroup236], (instregex "^STI$")>;

def ADLPWriteResGroup237 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 1];
  let Latency = 8;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup237], (instregex "^STOSB$")>;

def ADLPWriteResGroup238 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort04_09, ADLPPort07_08]> {
  let ResourceCycles = [2, 1, 1];
  let Latency = 7;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup238], (instregex "^STOS(L|Q|W)$")>;

def ADLPWriteResGroup239 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort01]> {
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup239], (instregex "^STR(32|64)r$")>;

def ADLPWriteResGroup240 : SchedWriteRes<[ADLPPort00]> {
  let Latency = 2;
}
def : InstRW<[ADLPWriteResGroup240], (instregex "^(TST|XAM)_F$",
                                                "^UCOM_FPPr$")>;

def ADLPWriteResGroup241 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [3, 1];
  let Latency = 9;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup241, ReadAfterVecXLd, ReadAfterVecXLd, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault], (instregex "^VBLENDVP(D|S)rm$",
                                                                                                                                                   "^VPBLENDVBrm$")>;

def ADLPWriteResGroup242 : SchedWriteRes<[ADLPPort00_01_05]> {
  let ResourceCycles = [3];
  let Latency = 3;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup242], (instregex "^VBLENDVP(D|S)rr$",
                                                "^VPBLENDVBrr$")>;

def ADLPWriteResGroup243 : SchedWriteRes<[ADLPPort00, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [6, 7, 18];
  let Latency = 81;
  let NumMicroOps = 31;
}
def : InstRW<[ADLPWriteResGroup243], (instregex "^VERRm$")>;

def ADLPWriteResGroup244 : SchedWriteRes<[ADLPPort00, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [6, 7, 17];
  let Latency = 74;
  let NumMicroOps = 30;
}
def : InstRW<[ADLPWriteResGroup244], (instregex "^VERRr$")>;

def ADLPWriteResGroup245 : SchedWriteRes<[ADLPPort00, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [5, 8, 21];
  let Latency = 81;
  let NumMicroOps = 34;
}
def : InstRW<[ADLPWriteResGroup245], (instregex "^VERWm$")>;

def ADLPWriteResGroup246 : SchedWriteRes<[ADLPPort00, ADLPPort01, ADLPPort02_03_11]> {
  let ResourceCycles = [5, 8, 20];
  let Latency = 74;
  let NumMicroOps = 33;
}
def : InstRW<[ADLPWriteResGroup246], (instregex "^VERWr$")>;

def ADLPWriteResGroup247 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 1, 2, 4];
  let Latency = 29;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup247, WriteVecMaskedGatherWriteback], (instregex "^VGATHER(DPD|QPS)Yrm$",
                                                                               "^VGATHERQPDYrm$",
                                                                               "^VPGATHER(D|Q)QYrm$",
                                                                               "^VPGATHERQDYrm$")>;

def ADLPWriteResGroup248 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 1, 1, 2];
  let Latency = 20;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup248, WriteVecMaskedGatherWriteback], (instregex "^VGATHER(DPD|QPS)rm$",
                                                                               "^VGATHERQPDrm$",
                                                                               "^VPGATHER(D|Q)Qrm$",
                                                                               "^VPGATHERQDrm$")>;

def ADLPWriteResGroup249 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 1, 2, 8];
  let Latency = 30;
  let NumMicroOps = 12;
}
def : InstRW<[ADLPWriteResGroup249, WriteVecMaskedGatherWriteback], (instregex "^VGATHERDPSYrm$",
                                                                               "^VPGATHERDDYrm$")>;

def ADLPWriteResGroup250 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 1, 2, 4];
  let Latency = 28;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup250, WriteVecMaskedGatherWriteback], (instregex "^VGATHERDPSrm$",
                                                                               "^VPGATHERDDrm$")>;

def ADLPWriteResGroup251 : SchedWriteRes<[ADLPPort01_05, ADLPPort05]> {
  let ResourceCycles = [1, 2];
  let Latency = 5;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup251], (instregex "^VH(ADD|SUB)P(D|S)rr$")>;

def ADLPWriteResGroup252 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort02_03_11]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup252, ReadAfterVecYLd], (instregex "^VINSERT(F|I)128rm$",
                                                                 "^VP(ADD|SUB)(B|D|Q|W)Yrm$")>;

def ADLPWriteResGroup253 : SchedWriteRes<[ADLPPort00, ADLPPort00_06, ADLPPort02_03_11]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup253], (instregex "^VLDMXCSR$")>;

def ADLPWriteResGroup254 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03, ADLPPort02_03_07, ADLPPort04, ADLPPort05, ADLPPort06]> {
  let ResourceCycles = [8, 1, 1, 1, 1, 1, 2, 3];
  let Latency = 40;
  let NumMicroOps = 18;
}
def : InstRW<[ADLPWriteResGroup254], (instregex "^VMCLEARm$")>;

def ADLPWriteResGroup255 : SchedWriteRes<[ADLPPort00]> {
  let Latency = 5;
}
def : InstRW<[ADLPWriteResGroup255], (instregex "^VMOVMSKP(D|S)Yrr$")>;

def ADLPWriteResGroup256 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 521;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup256], (instregex "^VMOVNTDQmr$")>;

def ADLPWriteResGroup257 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 473;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup257], (instregex "^VMOVNTPDmr$")>;

def ADLPWriteResGroup258 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 494;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup258], (instregex "^VMOVNTPSYmr$")>;

def ADLPWriteResGroup259 : SchedWriteRes<[ADLPPort04_09, ADLPPort07_08]> {
  let Latency = 470;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup259], (instregex "^VMOVNTPSmr$")>;

def ADLPWriteResGroup260 : SchedWriteRes<[ADLPPort02_03_11, ADLPPort05]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup260, ReadAfterVecYLd], (instregex "^VPACK(S|U)S(DW|WB)Yrm$",
                                                                 "^VPCMPGTQYrm$")>;
def : InstRW<[ADLPWriteResGroup260, ReadAfterVecXLd], (instregex "^VPCLMULQDQYrm$")>;

def ADLPWriteResGroup261 : SchedWriteRes<[ADLPPort01_05, ADLPPort02_03_11]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup261, ReadAfterVecYLd], (instregex "^VPBLENDWYrmi$",
                                                                 "^VSHUFP(D|S)Yrmi$")>;

def ADLPWriteResGroup262 : SchedWriteRes<[ADLPPort00_01, ADLPPort02_03_11]> {
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup262], (instregex "^VPDP(BU|WS)SD((SY)?)rm$",
                                                "^VPDP(BU|WS)SD(S|Y)rm$")>;

def ADLPWriteResGroup263 : SchedWriteRes<[ADLPPort00_01, ADLPPort01_05, ADLPPort02_03_11]> {
  let ResourceCycles = [1, 2, 1];
  let Latency = 10;
  let NumMicroOps = 4;
}
def : InstRW<[ADLPWriteResGroup263, ReadAfterVecYLd], (instregex "^VPH(ADD|SUB)SWYrm$")>;

def ADLPWriteResGroup264 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10]> {
  let ResourceCycles = [1, 2, 3, 3, 1];
  let Latency = 16;
  let NumMicroOps = 10;
}
def : InstRW<[ADLPWriteResGroup264], (instregex "^VZEROALL$")>;

def ADLPWriteResGroup265 : SchedWriteRes<[ADLPPort00_01_05_06]> {
  let ResourceCycles = [2];
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup265], (instregex "^WAIT$")>;

def ADLPWriteResGroup266 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [8, 6, 19, 63, 21, 15, 1, 10, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 144;
}
def : InstRW<[ADLPWriteResGroup266], (instregex "^WRMSR$")>;

def ADLPWriteResGroup267 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06, ADLPPort01, ADLPPort05]> {
  let ResourceCycles = [2, 1, 4, 1];
  let Latency = AlderlakePModel.MaxLatency;
  let NumMicroOps = 8;
}
def : InstRW<[ADLPWriteResGroup267], (instregex "^WRPKRUr$")>;

def ADLPWriteResGroup268 : SchedWriteRes<[ADLPPort00_01_05_06_10]> {
  let ResourceCycles = [2];
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup268, WriteRMW], (instregex "^XADD(16|32|64)rm$")>;

def ADLPWriteResGroup269 : SchedWriteRes<[ADLPPort00_01_05_06_10]> {
  let ResourceCycles = [2];
  let Latency = 13;
  let NumMicroOps = 2;
}
def : InstRW<[ADLPWriteResGroup269, WriteRMW], (instregex "^XADD8rm$")>;

def ADLPWriteResGroup270 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let ResourceCycles = [4, 1];
  let Latency = 39;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup270, WriteRMW], (instregex "^XCHG(16|32)rm$")>;

def ADLPWriteResGroup271 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let ResourceCycles = [5, 1];
  let Latency = 39;
  let NumMicroOps = 6;
}
def : InstRW<[ADLPWriteResGroup271, WriteRMW], (instregex "^XCHG64rm$")>;

def ADLPWriteResGroup272 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_06]> {
  let ResourceCycles = [4, 1];
  let Latency = 40;
  let NumMicroOps = 5;
}
def : InstRW<[ADLPWriteResGroup272, WriteRMW], (instregex "^XCHG8rm$")>;

def ADLPWriteResGroup273 : SchedWriteRes<[ADLPPort00, ADLPPort00_01_05_06, ADLPPort00_05, ADLPPort01, ADLPPort05, ADLPPort06]> {
  let ResourceCycles = [2, 4, 2, 1, 2, 4];
  let Latency = 17;
  let NumMicroOps = 15;
}
def : InstRW<[ADLPWriteResGroup273], (instregex "^XCH_F$")>;

def ADLPWriteResGroup274 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01]> {
  let ResourceCycles = [7, 3, 8, 5];
  let Latency = 4;
  let NumMicroOps = 23;
}
def : InstRW<[ADLPWriteResGroup274], (instregex "^XGETBV$")>;

def ADLPWriteResGroup275 : SchedWriteRes<[ADLPPort00_01_05_06_10, ADLPPort02_03_11]> {
  let ResourceCycles = [2, 1];
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[ADLPWriteResGroup275], (instregex "^XLAT$")>;

def ADLPWriteResGroup276 : SchedWriteRes<[ADLPPort00_01_05_06, ADLPPort01, ADLPPort02_03, ADLPPort06]> {
  let ResourceCycles = [21, 1, 1, 8];
  let Latency = 37;
  let NumMicroOps = 31;
}
def : InstRW<[ADLPWriteResGroup276], (instregex "^XRSTOR((S|64|S64)?)$")>;

def ADLPWriteResGroup277 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [14, 25, 44, 21, 21, 4, 1, 9, 1];
  let Latency = 42;
  let NumMicroOps = 140;
}
def : InstRW<[ADLPWriteResGroup277], (instregex "^XSAVE$")>;

def ADLPWriteResGroup278 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [14, 25, 44, 21, 21, 4, 1, 9, 1];
  let Latency = 41;
  let NumMicroOps = 140;
}
def : InstRW<[ADLPWriteResGroup278], (instregex "^XSAVE64$")>;

def ADLPWriteResGroup279 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [1, 19, 36, 52, 23, 4, 2, 12, 2];
  let Latency = 42;
  let NumMicroOps = 151;
}
def : InstRW<[ADLPWriteResGroup279], (instregex "^XSAVEC$")>;

def ADLPWriteResGroup280 : SchedWriteRes<[ADLPPort00, ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [1, 19, 36, 53, 23, 4, 2, 12, 2];
  let Latency = 42;
  let NumMicroOps = 152;
}
def : InstRW<[ADLPWriteResGroup280], (instregex "^XSAVEC64$")>;

def ADLPWriteResGroup281 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [25, 35, 52, 27, 4, 1, 10, 1];
  let Latency = 46;
  let NumMicroOps = 155;
}
def : InstRW<[ADLPWriteResGroup281], (instregex "^XSAVEOPT$")>;

def ADLPWriteResGroup282 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [25, 35, 53, 27, 4, 1, 10, 1];
  let Latency = 46;
  let NumMicroOps = 156;
}
def : InstRW<[ADLPWriteResGroup282], (instregex "^XSAVEOPT64$")>;

def ADLPWriteResGroup283 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [23, 32, 53, 29, 30, 4, 2, 9, 2];
  let Latency = 42;
  let NumMicroOps = 184;
}
def : InstRW<[ADLPWriteResGroup283], (instregex "^XSAVES$")>;

def ADLPWriteResGroup284 : SchedWriteRes<[ADLPPort00_01, ADLPPort00_05, ADLPPort00_06, ADLPPort01, ADLPPort01_05, ADLPPort02_03_11, ADLPPort04_09, ADLPPort05, ADLPPort07_08]> {
  let ResourceCycles = [23, 33, 53, 29, 32, 4, 2, 8, 2];
  let Latency = 42;
  let NumMicroOps = 186;
}
def : InstRW<[ADLPWriteResGroup284], (instregex "^XSAVES64$")>;

def ADLPWriteResGroup285 : SchedWriteRes<[ADLPPort00_01_05, ADLPPort00_01_05_06_10, ADLPPort00_05_06, ADLPPort00_06, ADLPPort01, ADLPPort01_05_10, ADLPPort05]> {
  let ResourceCycles = [4, 23, 2, 14, 8, 1, 2];
  let Latency = 5;
  let NumMicroOps = 54;
}
def : InstRW<[ADLPWriteResGroup285], (instregex "^XSETBV$")>;

}
