//=- AArch64SchedHIP12.td - Hisilicon HIP12 Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Hisilicon HIP12 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def HIP12Model : SchedMachineModel {
  let IssueWidth            =   8; // Micro-ops dispatched at a time.
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16, HasMTE,
                                                     HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

let SchedModel = HIP12Model in {

// Define the (19) issue ports.
def HIP12UnitB   : ProcResource<2>;  // Branch 0/1
def HIP12UnitS0  : ProcResource<1>;  // Integer Single-Cycle pipe 0
def HIP12UnitS1  : ProcResource<1>;  // Integer Single-Cycle pipe 1
def HIP12UnitSM2 : ProcResource<1>;  // Integer Single/multi-Cycle pipe2
def HIP12UnitS3  : ProcResource<1>;  // Integer Single-Cycle pipe 3
def HIP12UnitS4  : ProcResource<1>;  // Integer Single-Cycle pipe 4
def HIP12UnitSM5 : ProcResource<1>;  // Integer Single/multi-Cycle pipe5
def HIP12UnitV0  : ProcResource<1>;  // FP/ASIMD/SVE 0
def HIP12UnitV1  : ProcResource<1>;  // FP/ASIMD/SVE 1
def HIP12UnitV2  : ProcResource<1>;  // FP/ASIMD/SVE 2
def HIP12UnitV3  : ProcResource<1>;  // FP/ASIMD/SVE 3
def HIP12UnitLD  : ProcResource<3>;  // Load 0/1/2
def HIP12UnitST  : ProcResource<2>;  // Store 0/1
def HIP12UnitSTD : ProcResource<2>;  // Store data 0/1

def HIP12UnitALU     : ProcResGroup<[HIP12UnitS0, HIP12UnitS1, HIP12UnitSM2, HIP12UnitS3, HIP12UnitS4, HIP12UnitSM5]>; // Integer Single-Cycle 0/1/3/4 and Integer Single/Multi-Cycle 2/5
def HIP12UnitALU0134 : ProcResGroup<[HIP12UnitS0, HIP12UnitS1, HIP12UnitS3, HIP12UnitS4]>;  // Integer Single-Cycle 0/1/3/4
def HIP12UnitALU14   : ProcResGroup<[HIP12UnitS1, HIP12UnitS4]>;  // Integer Single-Cycle 1/4
def HIP12UnitALU25   : ProcResGroup<[HIP12UnitSM2, HIP12UnitSM5]>;  // Integer Single/Multi-Cycle 2/5
def HIP12UnitALU1425 : ProcResGroup<[HIP12UnitS1, HIP12UnitS4, HIP12UnitSM2, HIP12UnitSM5]>;  // Integer Single-Cycle 1/4 or Integer Single/Multi-Cycle 2/5
def HIP12UnitV       : ProcResGroup<[HIP12UnitV0, HIP12UnitV1, HIP12UnitV2, HIP12UnitV3]>;  // FP/ASIMD/SVE 0/1/2/3
def HIP12UnitV02     : ProcResGroup<[HIP12UnitV0, HIP12UnitV2]>;  // FP/ASIMD/SVE 0/2

// In SVE256 mode, V0&V1 and V2&V3 are combined into two SVE256 ports.
// Actually it should be used as (V0+v1) || (V2+V3)
// But now modelled as
//   SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { NumMicroOps = 1 }
def HIP12UnitV13     : ProcResGroup<[HIP12UnitV1, HIP12UnitV3]>;  // FP/ASIMD/SVE 1/3

// Like [SU]DIV, pointer authentication instructions, they can be issued
// to 2 pipes but share 1 execution unit. So add a dummy unit as a workaround.
def HIP12UnitDummy   : ProcResource<1>;

// No forwarding is provided for these types.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadST,      0>;
def : ReadAdvance<ReadVLD,     0>;

def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }
def : WriteRes<WriteLDHi,    []> { let Latency = 4; }

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the HIP12.
//===----------------------------------------------------------------------===//

// Define generic 0 micro-op types

def HIP12Write_0c : SchedWriteRes<[]> { let Latency = 0; }

// Define generic 1 micro-op types

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1B : SchedWriteRes<[HIP12UnitB]> {}

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1ALU : SchedWriteRes<[HIP12UnitALU]> {}

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1ALU1425 : SchedWriteRes<[HIP12UnitALU1425]> {}

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1ALU14 : SchedWriteRes<[HIP12UnitALU14]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1ALU14 : SchedWriteRes<[HIP12UnitALU14]> {}

let Latency = 2, NumMicroOps = 1 in
def HIP12Write_2c_1ALU25 : SchedWriteRes<[HIP12UnitALU25]> {}

let Latency = 3, NumMicroOps = 1 in
def HIP12Write_3c_1ALU25 : SchedWriteRes<[HIP12UnitALU25]> {}

let Latency = 2, NumMicroOps = 1 in
def HIP12Write_2c_1ALU25_1Dummy : SchedWriteRes<[HIP12UnitALU25,
                                                 HIP12UnitDummy]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1ALU25_1Dummy : SchedWriteRes<[HIP12UnitALU25,
                                                 HIP12UnitDummy]> {}

let Latency = 6, NumMicroOps = 1 in
def HIP12Write_6c_1ALU25_1Dummy : SchedWriteRes<[HIP12UnitALU25,
                                                 HIP12UnitDummy]> {}

let Latency = 8, NumMicroOps = 1 in
def HIP12Write_8c_1ALU25_1Dummy : SchedWriteRes<[HIP12UnitALU25,
                                                 HIP12UnitDummy]> {}

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 2, NumMicroOps = 1 in
def HIP12Write_2c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 3, NumMicroOps = 1 in
def HIP12Write_3c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 5, NumMicroOps = 1 in
def HIP12Write_5c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 1 in
def HIP12Write_6c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 8, NumMicroOps = 1 in
def HIP12Write_8c_1V : SchedWriteRes<[HIP12UnitV]> {}

let Latency = 5, NumMicroOps = 1 in
def HIP12Write_5c_1V_2rc : SchedWriteRes<[HIP12UnitV]> {
  let ReleaseAtCycles = [2];
}

let Latency = 7, NumMicroOps = 1 in
def HIP12Write_7c_1V_2rc : SchedWriteRes<[HIP12UnitV]> {
  let ReleaseAtCycles = [2];
}

let Latency = 9, NumMicroOps = 1 in
def HIP12Write_9c_1V_2rc : SchedWriteRes<[HIP12UnitV]> {
  let ReleaseAtCycles = [2];
}

let Latency = 2, NumMicroOps = 1 in
def HIP12Write_2c_1V02 : SchedWriteRes<[HIP12UnitV02]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1V02 : SchedWriteRes<[HIP12UnitV02]> {}

let Latency = 5, NumMicroOps = 1 in
def HIP12Write_5c_1V02 : SchedWriteRes<[HIP12UnitV02]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1LD : SchedWriteRes<[HIP12UnitLD]> {}

let Latency = 6, NumMicroOps = 1 in
def HIP12Write_6c_1LD : SchedWriteRes<[HIP12UnitLD]> {}

let Latency = 8, NumMicroOps = 1 in
def HIP12Write_8c_1LD : SchedWriteRes<[HIP12UnitLD]> {}

// Define generic 2 micro-op types

let Latency = 1, NumMicroOps = 2 in
def HIP12Write_1c_1B_1ALU14 : SchedWriteRes<[HIP12UnitB, HIP12UnitALU14]> {}

let Latency = 4, NumMicroOps = 2 in
def HIP12Write_4c_1ALU14_1V : SchedWriteRes<[HIP12UnitALU14, HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_1ALU14_1V : SchedWriteRes<[HIP12UnitALU14, HIP12UnitV]> {}

let Latency = 7, NumMicroOps = 2 in
def HIP12Write_7c_1ALU14_1V : SchedWriteRes<[HIP12UnitALU14, HIP12UnitV]> {}

let Latency = 3, NumMicroOps = 2 in
def HIP12Write_3c_1ALU25_1ALU : SchedWriteRes<[HIP12UnitALU25, HIP12UnitALU]> {}

let Latency = 4, NumMicroOps = 2 in
def HIP12Write_4c_1ALU25_1ALU : SchedWriteRes<[HIP12UnitALU25, HIP12UnitALU]> {}

let Latency = 5, NumMicroOps = 2 in
def HIP12Write_5c_1ALU25_1Dummy_1B : SchedWriteRes<[HIP12UnitALU25,
                                                    HIP12UnitDummy,
                                                    HIP12UnitB]> {}

let Latency = 8, NumMicroOps = 2 in
def HIP12Write_8c_1ALU_1LD : SchedWriteRes<[HIP12UnitALU, HIP12UnitLD]> {}

let Latency = 2, NumMicroOps = 2 in
def HIP12Write_2c_2V : SchedWriteRes<[HIP12UnitV, HIP12UnitV]> {}

let Latency = 3, NumMicroOps = 2 in
def HIP12Write_3c_2V : SchedWriteRes<[HIP12UnitV, HIP12UnitV]> {}

let Latency = 4, NumMicroOps = 2 in
def HIP12Write_4c_2V : SchedWriteRes<[HIP12UnitV, HIP12UnitV]> {}

let Latency = 5, NumMicroOps = 2 in
def HIP12Write_5c_2V : SchedWriteRes<[HIP12UnitV, HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_2V : SchedWriteRes<[HIP12UnitV, HIP12UnitV]> {}

let Latency = 2, NumMicroOps = 2 in
def HIP12Write_2c_1V02_1ALU0134 : SchedWriteRes<[HIP12UnitV02,
                                                 HIP12UnitALU0134]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_1V02_1ALU : SchedWriteRes<[HIP12UnitV02, HIP12UnitALU]> {}

let Latency = 7, NumMicroOps = 2 in
def HIP12Write_7c_1V02_1ALU25 : SchedWriteRes<[HIP12UnitV02,
                                               HIP12UnitALU25]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_2LD : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD]> {}

let Latency = 7, NumMicroOps = 2 in
def HIP12Write_7c_2LD : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_1LD_1ALU : SchedWriteRes<[HIP12UnitLD, HIP12UnitALU]> {}

let Latency = 8, NumMicroOps = 2 in
def HIP12Write_8c_1LD_1ALU25_1Dummy : SchedWriteRes<[HIP12UnitLD,
                                                     HIP12UnitALU25,
                                                     HIP12UnitDummy]> {}

let Latency = 8, NumMicroOps = 2 in
def HIP12Write_8c_1LD_1V : SchedWriteRes<[HIP12UnitLD, HIP12UnitV]> {}

let Latency = 1, NumMicroOps = 2 in
def HIP12Write_1c_1ST_1STD : SchedWriteRes<[HIP12UnitST, HIP12UnitSTD]> {}

let Latency = 3, NumMicroOps = 2 in
def HIP12Write_3c_1ST_1STD : SchedWriteRes<[HIP12UnitST, HIP12UnitSTD]> {}

// Define generic 3 micro-op types

let Latency = 4, NumMicroOps = 3 in
def HIP12Write_4c_3V : SchedWriteRes<[HIP12UnitV, HIP12UnitV, HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 3 in
def HIP12Write_6c_3V : SchedWriteRes<[HIP12UnitV, HIP12UnitV, HIP12UnitV]> {}

let Latency = 7, NumMicroOps = 3 in
def HIP12Write_7c_3V : SchedWriteRes<[HIP12UnitV, HIP12UnitV, HIP12UnitV]> {}

let Latency = 9, NumMicroOps = 3 in
def HIP12Write_9c_3V : SchedWriteRes<[HIP12UnitV, HIP12UnitV, HIP12UnitV]> {}

let Latency = 8, NumMicroOps = 3 in
def HIP12Write_8c_1LD_2V : SchedWriteRes<[HIP12UnitLD,
                                          HIP12UnitV, HIP12UnitV]> {}

let Latency = 1, NumMicroOps = 3 in
def HIP12Write_1c_1ST_1STD_1ALU : SchedWriteRes<[HIP12UnitST,
                                                 HIP12UnitSTD,
                                                 HIP12UnitALU]> {}

let Latency = 1, NumMicroOps = 3 in
def HIP12Write_1c_1ST_1STD_1V : SchedWriteRes<[HIP12UnitST,
                                               HIP12UnitSTD,
                                               HIP12UnitV]> {}

let Latency = 3, NumMicroOps = 3 in
def HIP12Write_3c_1ALU_1ST_1STD : SchedWriteRes<[HIP12UnitALU,
                                                 HIP12UnitST,
                                                 HIP12UnitSTD]> {}

// Define generic 4 micro-op types

let Latency = 6, NumMicroOps = 4 in
def HIP12Write_6c_2ALU0134_2V : SchedWriteRes<[HIP12UnitALU0134,
                                               HIP12UnitALU0134,
                                               HIP12UnitV,
                                               HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 4 in
def HIP12Write_6c_2V_2ALU0134 : SchedWriteRes<[HIP12UnitV,
                                               HIP12UnitV,
                                               HIP12UnitALU0134,
                                               HIP12UnitALU0134]> {}

let Latency = 6, NumMicroOps = 4 in
def HIP12Write_6c_4V : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                      HIP12UnitV, HIP12UnitV]> {}

let Latency = 8, NumMicroOps = 4 in
def HIP12Write_8c_4V : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                      HIP12UnitV, HIP12UnitV]> {}

let Latency = 9, NumMicroOps = 4 in
def HIP12Write_9c_4V : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                      HIP12UnitV, HIP12UnitV]> {}

let Latency = 12, NumMicroOps = 4 in
def HIP12Write_12c_4V : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                       HIP12UnitV, HIP12UnitV]> {}

let Latency = 6, NumMicroOps = 4 in
def HIP12Write_6c_2LD_2ALU : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                            HIP12UnitALU, HIP12UnitALU]> {}

let Latency = 8, NumMicroOps = 4 in
def HIP12Write_8c_2LD_2V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitV, HIP12UnitV]> {}

let Latency = 9, NumMicroOps = 4 in
def HIP12Write_9c_4LD : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                       HIP12UnitLD, HIP12UnitLD]> {}

let Latency = 1, NumMicroOps = 4 in
def HIP12Write_1c_2ST_2STD : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                            HIP12UnitSTD, HIP12UnitSTD]> {}

let Latency = 4, NumMicroOps = 4 in
def HIP12Write_4c_2V_1ST_1STD : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                               HIP12UnitST, HIP12UnitSTD]> {}

// Define generic 5 micro-op types

let Latency = 8, NumMicroOps = 5 in
def HIP12Write_8c_2LD_3V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitV, HIP12UnitV,
                                          HIP12UnitV]> {}

// Define generic 6 micro-op types

let Latency = 8, NumMicroOps = 6 in
def HIP12Write_8c_3LD_3V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitLD, HIP12UnitV,
                                          HIP12UnitV, HIP12UnitV]> {}

let Latency = 1, NumMicroOps = 6 in
def HIP12Write_1c_2ST_2STD_2ALU : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                                 HIP12UnitSTD, HIP12UnitSTD,
                                                 HIP12UnitALU, HIP12UnitALU]> {}

let Latency = 1, NumMicroOps = 6 in
def HIP12Write_1c_2ST_2STD_2V : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                               HIP12UnitSTD, HIP12UnitSTD,
                                               HIP12UnitV, HIP12UnitV]> {}

let Latency = 4, NumMicroOps = 6 in
def HIP12Write_4c_2ST_2STD_2V : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                               HIP12UnitSTD, HIP12UnitSTD,
                                               HIP12UnitV, HIP12UnitV]> {}

let Latency = 2, NumMicroOps = 6 in
def HIP12Write_2c_3ST_3STD : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD]> {}

// Define generic 7 micro-op types

let Latency = 8, NumMicroOps = 7 in
def HIP12Write_8c_3LD_4V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitLD,
                                          HIP12UnitV, HIP12UnitV,
                                          HIP12UnitV, HIP12UnitV]> {}

// Define generic 8 micro-op types

let Latency = 9, NumMicroOps = 8 in
def HIP12Write_9c_4LD_4V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitLD, HIP12UnitLD,
                                          HIP12UnitV, HIP12UnitV,
                                          HIP12UnitV, HIP12UnitV]> {}

let Latency = 10, NumMicroOps = 8 in
def HIP12Write_10c_8LD : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                        HIP12UnitLD, HIP12UnitLD,
                                        HIP12UnitLD, HIP12UnitLD,
                                        HIP12UnitLD, HIP12UnitLD]> {}

let Latency = 4, NumMicroOps = 8 in
def HIP12Write_4c_4V_2ST_2STD : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitST, HIP12UnitST,
                                               HIP12UnitSTD, HIP12UnitSTD]> {}

let Latency = 2, NumMicroOps = 8 in
def HIP12Write_2c_4ST_4STD : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitST,
                                            HIP12UnitSTD, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD]> {}

let Latency = 4, NumMicroOps = 8 in
def HIP12Write_4c_4ST_4STD : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitST,
                                            HIP12UnitSTD, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD]> {}

// Define generic 9 micro-op types

let Latency = 10, NumMicroOps = 9 in
def HIP12Write_10c_3LD_6V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                           HIP12UnitLD,
                                           HIP12UnitV, HIP12UnitV,
                                           HIP12UnitV, HIP12UnitV,
                                           HIP12UnitV, HIP12UnitV]> {}

// Define generic 12 micro-op types

let Latency = 11, NumMicroOps = 12 in
def HIP12Write_11c_4LD_8V : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                           HIP12UnitLD, HIP12UnitLD,
                                           HIP12UnitV, HIP12UnitV,
                                           HIP12UnitV, HIP12UnitV,
                                           HIP12UnitV, HIP12UnitV,
                                           HIP12UnitV, HIP12UnitV]> {}

let Latency = 5, NumMicroOps = 12 in
def HIP12Write_5c_6V_3ST_3STD : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitST, HIP12UnitST,
                                               HIP12UnitST,
                                               HIP12UnitSTD, HIP12UnitSTD,
                                               HIP12UnitSTD]> {}

// Define generic 16 micro-op types

let Latency = 5, NumMicroOps = 16 in
def HIP12Write_5c_8V_4ST_4STD : SchedWriteRes<[HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitV, HIP12UnitV,
                                               HIP12UnitST, HIP12UnitST,
                                               HIP12UnitST, HIP12UnitST,
                                               HIP12UnitSTD, HIP12UnitSTD,
                                               HIP12UnitSTD, HIP12UnitSTD]> {}

let Latency = 8, NumMicroOps = 16 in
def HIP12Write_8c_8ST_8STD : SchedWriteRes<[HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitST,
                                            HIP12UnitST, HIP12UnitST,
                                            HIP12UnitSTD, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD,
                                            HIP12UnitSTD, HIP12UnitSTD]> {}


// Define SVE256 1 micro-op types

let Latency = 1, NumMicroOps = 1 in
def HIP12Write_1c_1V02_1V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {}

let Latency = 2, NumMicroOps = 1 in
def HIP12Write_2c_1V02_1V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {}

let Latency = 3, NumMicroOps = 1 in
def HIP12Write_3c_1V02_1V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {}

let Latency = 4, NumMicroOps = 1 in
def HIP12Write_4c_1V02_1V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {}

let Latency = 5, NumMicroOps = 1 in
def HIP12Write_5c_1V02_1V13_2rc : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {
  let ReleaseAtCycles = [2, 2];
}

let Latency = 7, NumMicroOps = 1 in
def HIP12Write_7c_1V02_1V13_2rc : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {
  let ReleaseAtCycles = [2, 2];
}

let Latency = 9, NumMicroOps = 1 in
def HIP12Write_9c_1V02_1V13_2rc : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> {
  let ReleaseAtCycles = [2, 2];
}

let Latency = 20, NumMicroOps = 1 in
def HIP12Write_20c_1V02_1V13_12rc : SchedWriteRes<[HIP12UnitV02,
                                                   HIP12UnitV13]> {
  let ReleaseAtCycles = [12, 12];
}

let Latency = 26, NumMicroOps = 1 in
def HIP12Write_26c_1V02_1V13_10rc : SchedWriteRes<[HIP12UnitV02,
                                                   HIP12UnitV13]> {
  let ReleaseAtCycles = [10, 10];
}

let Latency = 42, NumMicroOps = 1 in
def HIP12Write_42c_1V02_1V13_18rc : SchedWriteRes<[HIP12UnitV02,
                                                   HIP12UnitV13]> {
  let ReleaseAtCycles = [18, 18];
}

// Define SVE256 2 micro-op types

let Latency = 5, NumMicroOps = 2 in
def HIP12Write_5c_1ALU0134_1V02_1V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                      HIP12UnitV02,
                                                      HIP12UnitV13]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_1ALU0134_1V02_1V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                      HIP12UnitV02,
                                                      HIP12UnitV13]> {}

let Latency = 9, NumMicroOps = 2 in
def HIP12Write_9c_1ALU0134_1V02_1V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                      HIP12UnitV02,
                                                      HIP12UnitV13]> {}

let Latency = 10, NumMicroOps = 2 in
def HIP12Write_10c_1ALU0134_1V02_1V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                       HIP12UnitV02,
                                                       HIP12UnitV13]> {}

let Latency = 4, NumMicroOps = 2 in
def HIP12Write_4c_2V02_2V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 5, NumMicroOps = 2 in
def HIP12Write_5c_2V02_2V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 6, NumMicroOps = 2 in
def HIP12Write_6c_2V02_2V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 7, NumMicroOps = 2 in
def HIP12Write_7c_2V02_2V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 9, NumMicroOps = 2 in
def HIP12Write_9c_2V02_2V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 8, NumMicroOps = 2 in
def HIP12Write_8c_1LD_1V02_1V13 : SchedWriteRes<[HIP12UnitLD,
                                                 HIP12UnitV02,
                                                 HIP12UnitV13]> {}

let Latency = 9, NumMicroOps = 2 in
def HIP12Write_9c_1V02_1V13_1ALU25_18rc : SchedWriteRes<[HIP12UnitV02,
                                                         HIP12UnitV13,
                                                         HIP12UnitALU25]> {
  let ReleaseAtCycles = [18, 18, 1];
}

// Define SVE256 3 micro-op types

let Latency = 7, NumMicroOps = 3 in
def HIP12Write_7c_3V02_3V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV02, HIP12UnitV13,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 8, NumMicroOps = 3 in
def HIP12Write_8c_3V02_3V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV02, HIP12UnitV13,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 9, NumMicroOps = 3 in
def HIP12Write_9c_3V02_3V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV02, HIP12UnitV13,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 4, NumMicroOps = 3 in
def HIP12Write_4c_1V02_1V13_1ST_1STD : SchedWriteRes<[HIP12UnitV02,
                                                      HIP12UnitV13,
                                                      HIP12UnitST,
                                                      HIP12UnitSTD]> {}

// Define SVE256 4 micro-op types

let Latency = 8, NumMicroOps = 4 in
def HIP12Write_8c_4V02_4V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV02, HIP12UnitV02,
                                             HIP12UnitV13, HIP12UnitV13,
                                             HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 10, NumMicroOps = 4 in
def HIP12Write_10c_4V02_4V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 11, NumMicroOps = 4 in
def HIP12Write_11c_4V02_4V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 12, NumMicroOps = 4 in
def HIP12Write_12c_4V02_4V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 11, NumMicroOps = 4 in
def HIP12Write_11c_2ALU0134_2V02_2V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitV02,
                                                       HIP12UnitV02,
                                                       HIP12UnitV13,
                                                       HIP12UnitV13]> {}

let Latency = 8, NumMicroOps = 4 in
def HIP12Write_8c_2LD_2V02_2V13 : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                                 HIP12UnitV02, HIP12UnitV02,
                                                 HIP12UnitV13, HIP12UnitV13]> {}

// Define SVE256 5 micro-op types

let Latency = 13, NumMicroOps = 5 in
def HIP12Write_13c_5V02_5V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 14, NumMicroOps = 5 in
def HIP12Write_14c_5V02_5V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 15, NumMicroOps = 5 in
def HIP12Write_15c_5V02_5V13 : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV02,
                                              HIP12UnitV02, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13,
                                              HIP12UnitV13, HIP12UnitV13]> {}

// Define SVE256 6 micro-op types

let Latency = 10, NumMicroOps = 6 in
def HIP12Write_10c_3ALU0134_3V02_3V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitV02, HIP12UnitV02,
                                                       HIP12UnitV02, HIP12UnitV13,
                                                       HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 10, NumMicroOps = 6 in
def HIP12Write_10c_3LD_3V02_3V13 : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitLD,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 4, NumMicroOps = 6 in
def HIP12Write_4c_2V02_2V13_2ST_2STD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitSTD, HIP12UnitSTD]> {}

// Define SVE256 8 micro-op types

let Latency = 11, NumMicroOps = 8 in
def HIP12Write_11c_4ALU0134_4V02_4V13 : SchedWriteRes<[HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitALU0134,
                                                       HIP12UnitV02, HIP12UnitV02,
                                                       HIP12UnitV02, HIP12UnitV02,
                                                       HIP12UnitV13, HIP12UnitV13,
                                                       HIP12UnitV13, HIP12UnitV13]> {}

let Latency = 11, NumMicroOps = 8 in
def HIP12Write_11c_4LD_4V02_4V13 : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13]> {}

// Define SVE256 9 micro-op types

let Latency = 10, NumMicroOps = 9 in
def HIP12Write_10c_3LD_6V02_6V13 : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitLD,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13]> {}

// Define SVE256 12 micro-op types

let Latency = 11, NumMicroOps = 12 in
def HIP12Write_11c_4LD_8V02_8V13 : SchedWriteRes<[HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitLD, HIP12UnitLD,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV02, HIP12UnitV02,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13,
                                                  HIP12UnitV13, HIP12UnitV13]> {}

// Define SVE256 18 micro-op types

let Latency = 7, NumMicroOps = 18 in
def HIP12Write_7c_6V02_6V13_6ST_6STD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitSTD, HIP12UnitSTD,
                                                      HIP12UnitSTD, HIP12UnitSTD,
                                                      HIP12UnitSTD, HIP12UnitSTD]> {}

// Define SVE256 24 micro-op types

let Latency = 7, NumMicroOps = 24 in
def HIP12Write_7c_8V02_8V13_8ST_8STD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV02, HIP12UnitV02,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitV13, HIP12UnitV13,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitST, HIP12UnitST,
                                                      HIP12UnitSTD, HIP12UnitSTD,
                                                      HIP12UnitSTD, HIP12UnitSTD,
                                                      HIP12UnitSTD, HIP12UnitSTD,
                                                      HIP12UnitSTD, HIP12UnitSTD]> {}

//===----------------------------------------------------------------------===//
// Define predicates
//===----------------------------------------------------------------------===//

// Check for shift = 0
def HIP12AnyShiftBy0 : MCSchedPredicate<
                         CheckAll<[
                           CheckShiftBy0,
                           CheckAny<[
                             CheckShiftLSL,
                             CheckShiftLSR,
                             CheckShiftASR,
                             CheckShiftROR]>]>>;

// Identify LDR/STR Q-form scaled (no-extend) FP instructions
def HIP12QForm : MCSchedPredicate<
                   CheckAll<[
                     CheckAny<[CheckQForm]>,
                     CheckImmOperand<4, 1>,
                     CheckNot<CheckAny<[
                                CheckExtUXTW,
                                CheckExtSXTW,
                                CheckExtUXTX,
                                CheckExtSXTX]>>]>>;

//===----------------------------------------------------------------------===//
// Define predicate-controlled types
//===----------------------------------------------------------------------===//

def HIP12Write_Arith : SchedWriteVariant<[
                         SchedVar<HIP12AnyShiftBy0, [HIP12Write_1c_1ALU]>,
                         SchedVar<NoSchedPred,      [HIP12Write_2c_1ALU25]>]>;

def HIP12Write_Ariths : SchedWriteVariant<[
                          SchedVar<HIP12AnyShiftBy0, [HIP12Write_1c_1ALU1425]>,
                          SchedVar<NoSchedPred,      [HIP12Write_2c_1ALU25]>]>;

def HIP12Write_Logical : SchedWriteVariant<[
                           SchedVar<HIP12AnyShiftBy0, [HIP12Write_1c_1ALU]>,
                           SchedVar<NoSchedPred,      [HIP12Write_1c_1ALU14]>]>;

def HIP12Write_Logicals : SchedWriteVariant<[
                            SchedVar<HIP12AnyShiftBy0, [HIP12Write_1c_1ALU1425]>,
                            SchedVar<NoSchedPred,      [HIP12Write_2c_1ALU25]>]>;

def HIP12Write_IMAL : SchedWriteVariant<[
                        SchedVar<IsReg3ZeroPred, [HIP12Write_2c_1ALU25]>,
                        SchedVar<NoSchedPred,    [HIP12Write_3c_1ALU25_1ALU]>]>;

def HIP12Write_LDRro : SchedWriteVariant<[
                         SchedVar<IsExtWorXPred, [HIP12Write_8c_1ALU_1LD]>,
                         SchedVar<HIP12QForm,    [HIP12Write_8c_1LD]>,
                         SchedVar<NoSchedPred,   [HIP12Write_6c_1LD]>]>;

def HIP12Write_STRro : SchedWriteVariant<[
                         SchedVar<IsExtWorXPred, [HIP12Write_3c_1ALU_1ST_1STD]>,
                         SchedVar<HIP12QForm,    [HIP12Write_3c_1ST_1STD]>,
                         SchedVar<NoSchedPred,   [HIP12Write_1c_1ST_1STD]>]>;

//===----------------------------------------------------------------------===//
// Define forwarded types
//===----------------------------------------------------------------------===//

// FP multiply
// FP fused multiply-add
def HIP12Wr_FMA : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_FMA : SchedReadAdvance<2, [WriteFMul, HIP12Wr_FMA]>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
def HIP12Wr_VA : SchedWriteRes<[HIP12UnitV]> { let Latency = 2; }
def HIP12Rd_VA : SchedReadAdvance<1, [HIP12Wr_VA]>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def HIP12Wr_VDOT : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VDOT : SchedReadAdvance<2, [HIP12Wr_VDOT]>;

// ASIMD matrix multiply-accumulate
def HIP12Wr_VMMA : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VMMA : SchedReadAdvance<2, [HIP12Wr_VMMA]>;

// ASIMD multiply accumulate, B/H/S form
def HIP12Wr_VMA : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VMA : SchedReadAdvance<2, [HIP12Wr_VMA]>;

// ASIMD multiply accumulate high
def HIP12Wr_VMAH : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VMAH : SchedReadAdvance<1, [HIP12Wr_VMAH]>;

// ASIMD multiply accumulate long
def HIP12Wr_VMAL : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VMAL : SchedReadAdvance<2, [HIP12Wr_VMAL]>;

// ASIMD multiply accumulate saturating long
def HIP12Wr_VMAQL : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VMAQL : SchedReadAdvance<1, [HIP12Wr_VMAQL]>;

// ASIMD pairwise add and accumulate long
def HIP12Wr_VPA : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VPA : SchedReadAdvance<2, [HIP12Wr_VPA]>;

// ASIMD shift accumulate
def HIP12Wr_VSA : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Rd_VSA : SchedReadAdvance<1, [HIP12Wr_VSA]>;

// ASIMD FP complex multiply-add
def HIP12Wr_VFCMA : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_VFCMA : SchedReadAdvance<2, [HIP12Wr_VFCMA]>;

// ASIMD FP multiply
// ASIMD FP fused multiply-add
def HIP12Wr_VFM  : SchedWriteRes<[HIP12UnitV]> { let Latency = 3; }
def HIP12Wr_VFMA : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_VFMA : SchedReadAdvance<2, [HIP12Wr_VFM, HIP12Wr_VFMA]>;

// ASIMD FP fused multiply-add long
def HIP12Wr_VFMAL : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_VFMAL : SchedReadAdvance<2, [HIP12Wr_VFMAL]>;

// Bfloat16
// ASIMD multiply accumulate long
def HIP12Wr_VBFMAL : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_VBFMAL : SchedReadAdvance<2, [HIP12Wr_VBFMAL]>;

// ASIMD FP reciprocal step
def HIP12Wr_VRECP : SchedWriteRes<[HIP12UnitV]> { let Latency = 4; }
def HIP12Rd_VRECP : SchedReadAdvance<2, [HIP12Wr_VRECP]>;

// Arithmetic, absolute diff accum SVE256
def HIP12Wr_ZA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 2; }
def HIP12Rd_ZA : SchedReadAdvance<1, [HIP12Wr_ZA]>;

// Arithmetic, absolute diff accum long SVE256
def HIP12Wr_ZAL : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZAL : SchedReadAdvance<2, [HIP12Wr_ZAL]>;

// Arithmetic, pairwise add and accum long SVE256
def HIP12Wr_ZPA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZPA : SchedReadAdvance<2, [HIP12Wr_ZPA]>;

// Arithmetic, shift and accumulate SVE256
def HIP12Wr_ZSA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZSA : SchedReadAdvance<1, [HIP12Wr_ZSA]>;

// Complex dot product SVE256
// Dot product SVE256
// Dot product, 8-bit, using signed and unsigned integers SVE256
def HIP12Wr_ZDOT : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZDOT : SchedReadAdvance<2, [HIP12Wr_ZDOT]>;

// Complex multiply-add B, H, S element size SVE256
def HIP12Wr_ZCMABHS : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZCMABHS : SchedReadAdvance<2, [HIP12Wr_ZCMABHS]>;

// Complex multiply-add D element size SVE256
def HIP12Wr_ZCMAD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                   HIP12UnitV13, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZCMAD : SchedReadAdvance<2, [HIP12Wr_ZCMAD]>;

// Matrix multiply-accumulate SVE256
def HIP12Wr_ZMMA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZMMA : SchedReadAdvance<2, [HIP12Wr_ZMMA]>;

// Multiply accumulate, B, H, S element size SVE256
def HIP12Wr_ZMABHS : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZMABHS : SchedReadAdvance<2, [HIP12Wr_ZMABHS]>;

// Multiply accumulate, D element size SVE256
def HIP12Wr_ZMAD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                  HIP12UnitV13, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZMAD : SchedReadAdvance<2, [HIP12Wr_ZMAD]>;

// Multiply accumulate long, B, H, S element size SVE256
def HIP12Wr_ZMALBHS : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Rd_ZMALBHS : SchedReadAdvance<2, [HIP12Wr_ZMALBHS]>;

// Multiply accumulate long, D element size SVE256
def HIP12Wr_ZMALD : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                   HIP12UnitV13, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZMALD : SchedReadAdvance<2, [HIP12Wr_ZMALD]>;

// Multiply accumulate saturating doubling long regular, B, H, S element size SVE256
// Multiply accumulate saturating doubling long regular, D element size SVE256
// Multiply saturating rounding doubling regular/complex accumulate, B, H, S element size SVE256
// Multiply saturating rounding doubling regular/complex accumulate, D element size SVE256
def HIP12Wr_ZMASQLBHS : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Wr_ZMASQLD   : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                       HIP12UnitV13, HIP12UnitV13]> { let Latency = 4; }
def HIP12Wr_ZMASQBHS  : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 3; }
def HIP12Wr_ZMASQD    : SchedWriteRes<[HIP12UnitV02, HIP12UnitV02,
                                       HIP12UnitV13, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZMASQ     : SchedReadAdvance<1, [HIP12Wr_ZMASQLBHS, HIP12Wr_ZMASQLD,
                                             HIP12Wr_ZMASQBHS, HIP12Wr_ZMASQD]>;

// Floating point complex multiply add(vectors) SVE256
// Floating point complex multiply add(indexed) SVE256
def HIP12Wr_ZFCMA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZFCMA : SchedReadAdvance<2, [HIP12Wr_ZFCMA]>;

// Floating point multiply accumulate SVE256
def HIP12Wr_ZFMA : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZFMA : SchedReadAdvance<2, [HIP12Wr_ZFMA]>;

// Floating point multiply add/sub accumulate long SVE256
def HIP12Wr_ZFMAL : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZFMAL : SchedReadAdvance<2, [HIP12Wr_ZFMAL]>;

// BFloat16
// Multiply accumulate long SVE256
def HIP12Wr_ZBFMAL : SchedWriteRes<[HIP12UnitV02, HIP12UnitV13]> { let Latency = 4; }
def HIP12Rd_ZBFMAL : SchedReadAdvance<2, [HIP12Wr_ZBFMAL]>;

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

def : InstRW<[WriteI], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// 3.3 Branch Instructions
//===----------------------------------------------------------------------===//

// Branch, immed
// instrs B, B.cond
def : SchedAlias<WriteBr, HIP12Write_1c_1B>;

// Branch, register
// instrs BR, RET
def : SchedAlias<WriteBrReg, HIP12Write_1c_1B>;

// Branch and link, immed
def : InstRW<[HIP12Write_1c_1B_1ALU14], (instrs BL)>;

// Branch and link, register
def : InstRW<[HIP12Write_1c_1B_1ALU14], (instrs BLR)>;

// Compare and branch
// instrs CBZ, CBNZ, TBZ, TBNZ
// alias to WriteBr

//===----------------------------------------------------------------------===//
// 3.4 Arithmetic and Logical Instructions
//===----------------------------------------------------------------------===//

// ALU, basic
// instrs ADD, ADC, AND, BIC, EON, EOR, ORN, ORR, SUB, SBC
def : SchedAlias<WriteI, HIP12Write_1c_1ALU>;

// ALU, basic, flagset
// instrs ADDS, ADCS, ANDS, BICS, SUBS, SBCS
def : InstRW<[HIP12Write_1c_1ALU1425], (instregex "^(ADD|SUB)S[WX]r[ir]$",
                                                  "^(ADC|SBC)S[WX]r$",
                                                  "^ANDS[WX]ri$",
                                                  "^(AND|BIC)S[WX]rr$")>;

// ALU, extend and shift
// instrs ADD{S},SUB{S}
def : SchedAlias<WriteIEReg, HIP12Write_2c_1ALU25>;

// Arithmetic, shift
// instrs ADD, SUB
def : SchedAlias<WriteISReg, HIP12Write_Arith>;

// Arithmetic flagset, shift
// instrs ADDS, SUBS
def : InstRW<[HIP12Write_Ariths], (instregex "^(ADD|SUB)S[WX]rs$")>;

// Conditional compare
// instrs CCMN, CCMP
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^CCM[NP][WX][ir]")>;

// Conditional select
// instrs CSEL, CSINC, CSINV, CSNEG
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "CSEL[WX]r", "CSINC[WX]r",
                                                "CSINV[WX]r", "CSNEG[WX]r")>;

// Convert floating-point condition flags
def : SchedAlias<WriteSys, HIP12Write_1c_1ALU14>;
def : InstRW<[HIP12Write_1c_1ALU14], (instrs AXFLAG, XAFLAG)>;

// Flag manipulation instructions
def : InstRW<[HIP12Write_1c_1ALU14], (instrs SETF8, SETF16, RMIF, CFINV)>;

// Logical, shift, no flagset
// instrs AND, BIC, EON, EOR, ORN, ORR
def : InstRW<[HIP12Write_Logical], (instregex "^(AND|BIC|EON|EOR|ORN|ORR)[WX]rs$")>;

// Logical, shift, flagset
// instrs ANDS, BICS
def : InstRW<[HIP12Write_Logicals], (instregex "^(AND|BIC)S[WX]rs$")>;

//===----------------------------------------------------------------------===//
// 3.5 Divide and Multiply Instructions
//===----------------------------------------------------------------------===//

// Multiply, W-form
// instrs MUL, MNEG
// Multiply, X-form
// instrs MUL, MNEG
def : SchedAlias<WriteIM32, HIP12Write_2c_1ALU25>;
def : SchedAlias<WriteIM64, HIP12Write_3c_1ALU25>;

// Divide, W-form
// instrs SDIV, UDIV
// Divide, X-form
// instrs SDIV, UDIV
def : SchedAlias<WriteID32, HIP12Write_6c_1ALU25_1Dummy>;
def : SchedAlias<WriteID64, HIP12Write_8c_1ALU25_1Dummy>;

// Multiply accumulate, W-form
// instrs MADD, MSUB
def : InstRW<[HIP12Write_3c_1ALU25_1ALU], (instrs MADDWrrr, MSUBWrrr)>;

// Multiply accumulate, X-form
// instrs MADD, MSUB
def : InstRW<[HIP12Write_4c_1ALU25_1ALU], (instrs MADDXrrr, MSUBXrrr)>;

// Multiply accumulate long
// instrs SMADDL, SMSUBL, UMADDL, UMSUBL
// Multiply long
// instrs SMNEGL, SMULL, UMNEGL, UMULL
def : InstRW<[HIP12Write_IMAL], (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
// instrs SMULH, UMULH
def : InstRW<[HIP12Write_3c_1ALU25], (instrs SMULHrr, UMULHrr)>;

//===----------------------------------------------------------------------===//
// 3.6 Pointer Authentication Instructions
//===----------------------------------------------------------------------===//

// Authenticate data address
// instrs AUTDA, AUTDB, AUTDZA, AUTDZB
// Authenticate instruction address
// instrs AUTIA, AUTIB, AUTIA1716, AUTIB1716, AUTIASP, AUTIBSP, AUTIAZ, AUTIBZ, AUTIZA, AUTIZB
// Compute pointer authentication code for data address
// instrs PACDA, PACDB, PACDZA, PACDZB
// Compute pointer authentication code, using generic key
// instrs PACGA
// Compute pointer authentication code for instruction address
// instrs PACIA, PACIB, PACIA1716, PACIB1716, PACIASP, PACIBSP, PACIAZ, PACIBZ, PACIZA, PACIZB
def : InstRW<[HIP12Write_4c_1ALU25_1Dummy], (instregex "^AUT", "^PAC")>;

// Branch and link, register, with pointer authentication
def : InstRW<[HIP12Write_5c_1ALU25_1Dummy_1B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ)>;

// Branch, register, with pointer authentication
def : InstRW<[HIP12Write_5c_1ALU25_1Dummy_1B], (instrs BRAA, BRAAZ, BRAB, BRABZ)>;

// Branch, return, with pointer authentication
// instrs RETAA, RETAB, ERETAA, ERETAB
def : InstRW<[HIP12Write_5c_1ALU25_1Dummy_1B], (instrs RETAA, RETAB, ERETAA, ERETAB)>;

// Load register, with pointer authentication
// instrs LDRAA, LDRAB
def : InstRW<[HIP12Write_8c_1LD_1ALU25_1Dummy], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[HIP12Write_2c_1ALU25_1Dummy], (instrs XPACD, XPACI, XPACLRI)>;

//===----------------------------------------------------------------------===//
// 3.7 Miscellaneous Data-Processing Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteIS, HIP12Write_1c_1ALU14>;

// Address generation
def : InstRW<[HIP12Write_1c_1ALU14], (instrs ADR, ADRP)>;

// Bitfield extract, one reg
// instrs EXTR
def : SchedAlias<WriteExtr, HIP12Write_1c_1ALU14>;

// Bitfield extract, two regs
// instrs EXTR
def : InstRW<[HIP12Write_1c_1ALU14], (instrs EXTRWrri, EXTRXrri)>;

// Bitfield move, basic
// instrs SBFM, UBFM
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^(S|U)BFM[WX]ri$")>;

// Bitfield move, insert
// instrs BFM
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^BFM[WX]ri$")>;

// Move immed
// instrs MOVN, MOVK, MOVZ
def : SchedAlias<WriteImm, HIP12Write_1c_1ALU>;
def : InstRW<[HIP12Write_1c_1ALU], (instregex "^MOV(N|K|Z)[WX]i$")>;

// Count leading
// instrs CLS, CLZ
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^CL(S|Z)[WX]r$")>;

// Reverse bits/bytes
// instrs RBIT, REV, REV16, REV32
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^(RBIT|REV|REV16|REV32)[WX]r$")>;

// Variable shift
// instrs ASRV, LSLV, LSRV, RORV
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^(ASR|LSL|LSR|ROR)V[WX]r$")>;

//===----------------------------------------------------------------------===//
// 3.8 Load Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteAdr,   HIP12Write_1c_1ALU>;
def : SchedAlias<WriteLD,    HIP12Write_4c_1LD>;
def : SchedAlias<WriteLDIdx, HIP12Write_4c_1LD>;

//===----------------------------------------------------------------------===//
// 3.9 Store Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteST,    HIP12Write_1c_1ST_1STD>;
def : SchedAlias<WriteSTP,   HIP12Write_1c_1ST_1STD>;
def : SchedAlias<WriteSTIdx, HIP12Write_1c_1ST_1STD>;

//===----------------------------------------------------------------------===//
// 3.10 FP Data Processing Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteF, HIP12Write_2c_1V>;

// FP absolute value
// instrs FABS
def : InstRW<[HIP12Write_1c_1V], (instregex "^FABS[HSD]r$")>;

// FP absolute value
// instrs FABD
def : InstRW<[HIP12Write_2c_1V], (instregex "^FABD(16|32|64)$")>;

// FP negate
// instrs FNEG
def : InstRW<[HIP12Write_1c_1V], (instregex "^FNEG[HSD]r$")>;

// FP compare
// instrs FCMP{E}
def : SchedAlias<WriteFCmp, HIP12Write_2c_1V02_1ALU0134>;
def : InstRW<[HIP12Write_2c_1V02_1ALU0134], (instregex "^FCMPE?[HSD]r[ri]$")>;

// FP conditional compare
// instrs FCCMP{E}
def : InstRW<[HIP12Write_4c_1ALU14_1V], (instregex "^FCCMPE?[HSD]rr$")>;

// FP conditional select
// instrs FCSEL
def : InstRW<[HIP12Write_6c_1ALU14_1V], (instregex "^FCSEL[HSD]rrr")>;

def : SchedAlias<WriteFDiv, HIP12Write_4c_1V>;

// FP divide, H-form
// instrs FDIV
def : InstRW<[HIP12Write_4c_1V],  (instrs FDIVHrr)>;

// FP divide, S-form
// instrs FDIV
def : InstRW<[HIP12Write_6c_1V], (instrs FDIVSrr)>;

// FP divide, D-form
// instrs FDIV
def : InstRW<[HIP12Write_8c_1V], (instrs FDIVDrr)>;

// FP square root, H-form
// instrs FSQRT
def : InstRW<[HIP12Write_4c_1V], (instrs FSQRTHr)>;

// FP square root, S-form
// instrs FSQRT
def : InstRW<[HIP12Write_6c_1V], (instrs FSQRTSr)>;

// FP square root, D-form
// instrs FSQRT
def : InstRW<[HIP12Write_8c_1V], (instrs FSQRTDr)>;

// FP fused multiply-add
// instrs FMADD, FMSUB, FNMADD, FNMSUB
def : InstRW<[HIP12Wr_FMA, ReadDefault, ReadDefault, HIP12Rd_FMA],
             (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP max/min
// instrs FMAX, FMAXNM, FMIN, FMINNM
def : InstRW<[HIP12Write_2c_1V], (instregex "^(FMAX|FMIN)(NM)?[HSD]rr$")>;

// FP add
// instrs FADD, FSUB
def : InstRW<[HIP12Write_2c_1V], (instregex "^F(ADD|SUB)[HSD]rr$")>;

// FP multiply
// instrs FMUL, FNMUL
def : WriteRes<WriteFMul, [HIP12UnitV]> { let Latency = 3; }

// FP round to FP integral
// instrs FRINTA, FRINTI, FRINTM, FRINTN, FRINTP, FRINTX, FRINTZ, FRINT32Z, FRINT32X, FRINT64Z, FRINT64X
def : InstRW<[HIP12Write_3c_1V], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                            "^FRINT(32|64)[XZ][SD]r$")>;

//===----------------------------------------------------------------------===//
// 3.11 FP Miscellaneous Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteFCvt, HIP12Write_7c_1ALU14_1V>;

// FP covert, from gen to vec reg
// instrs SCVTF, UCVTF
def : InstRW<[HIP12Write_7c_1ALU14_1V], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP covert, from vec to gen reg
// instrs FCVTAS, FCVTAU, FCVTMS, FCVTMU, FCVTNS, FCVTNU, FCVTPS, FCVTPU, FCVTZS, FCVTZU
def : InstRW<[HIP12Write_5c_1V], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri?$")>;

// FP convert, Javascript from to gen reg
def : InstRW<[HIP12Write_5c_1V02], (instrs FJCVTZS)>;

// FP convert, from vec to vec reg
// instrs FCVT, FCVTXN
def : InstRW<[HIP12Write_3c_1V], (instrs FCVTSHr, FCVTDHr, FCVTHSr, FCVTDSr,
                                         FCVTHDr, FCVTSDr, FCVTXNv1i64)>;

// FP move, immed
// instrs FMOV
def : SchedAlias<WriteFImm, HIP12Write_2c_1V>;

// FP move, register
// instrs FMOV
def : InstRW<[HIP12Write_2c_1V], (instregex "^FMOV[HSD]r$")>;

// FP transfer, from gen to low half of vec reg
// instrs FMOV
def : InstRW<[HIP12Write_4c_1ALU14],
             (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
// instrs FMOV
def : InstRW<[HIP12Write_6c_1ALU14_1V], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
// instrs FMOV
def : SchedAlias<WriteFCopy, HIP12Write_2c_1V>;

//===----------------------------------------------------------------------===//
// 3.12 FP Load Instructions
//===----------------------------------------------------------------------===//

// Load vector reg, literal
// instrs LDR
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDR[SDQ]l$")>;

// Load vector reg, unscaled immed
// instrs LDUR
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDUR[BHSDQ]i$")>;

// Load vector reg, SIMD&FP
// instrs LDAPUR
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDAPUR")>;

// Load vector reg, immed post-index
// instrs LDR
// Load vector reg, immed pre-index
// instrs LDR
def : InstRW<[WriteAdr, HIP12Write_6c_1LD_1ALU],
             (instregex "^LDR[BHSDQ](pre|post)$")>;

// Load vector reg, unsigned immed
// instrs LDR
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDR[BHSDQ]ui$")>;

// Load vector reg, register offset, no-extend, basic
// instrs LDR
// Load vector reg, register offset, no-extend, scale by 2/4/8
// instrs LDR
// Load vector reg, register offset, no-extend, scale by 16
// instrs LDR
// Load vector reg, register offset, extend
// instrs LDR
// Load vector reg, register offset, extend, scale by 2/4/8/16
// instrs LDR
def : InstRW<[HIP12Write_LDRro, ReadAdrBase], (instregex "^LDR[BHSDQ]ro[WX]$")>;

// Load vector pair, immed offset, S/D-form
// instrs LDP, LDNP
def : InstRW<[HIP12Write_6c_1LD, WriteLDHi], (instregex "^LDN?P[SD]i$")>;

// Load vector pair, immed offset, Q-form
// instrs LDP, LDNP
def : InstRW<[HIP12Write_6c_2LD, WriteLDHi], (instrs LDPQi, LDNPQi)>;

// Load vector pair, immed post-index, S/D-form
// instrs LDP
// Load vector pair, immed pre-index, S/D-form
// instrs LDP
def : InstRW<[WriteAdr, HIP12Write_6c_1LD_1ALU, WriteLDHi],
             (instregex "^LDP[SD](pre|post)$")>;

// Load vector pair, immed post-index, Q-form
// instrs LDP
// Load vector pair, immed pre-index, Q-form
// instrs LDP
def : InstRW<[WriteAdr, HIP12Write_6c_2LD_2ALU, WriteLDHi],
             (instrs LDPQpre, LDPQpost)>;

//===----------------------------------------------------------------------===//
// 3.13 FP Store Instructions
//===----------------------------------------------------------------------===//

// Store vector reg, unscaled immed
// instrs STUR
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index
// instrs STR
// Store vector reg, immed pre-index
// instrs STR
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD_1ALU],
             (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector reg, unsigned immed
// instrs STR
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^STR[BHSDQ]ui$")>;

// Store vector reg, reg offset, no-extend, basic
// instrs STR
// Store vector reg offset, no-extend, scaled by 2/4/8
// instrs STR
// Store vector reg offset, no-extend, scaled by 16
// instrs STR
// Store vector reg, reg offset, extend
// instrs STR
// Store vector reg, reg offset, extend, scale by 2/4/8/16
// instrs STR
def : InstRW<[HIP12Write_STRro],
             (instregex "^STR[BHSDQ]ro[WX]$")>;

// Store vector pair, immed offset, S/D-form
// instrs STP, STNP
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^STN?P[SD]i$")>;

// Store vector pair, immed offset, Q-form
// instrs STP, STNP
def : InstRW<[HIP12Write_1c_2ST_2STD], (instrs STPQi, STNPQi)>;

// Store vector pair, immed post-index, S/D-form
// instrs STP
// Store vector pair, immed pre-index, S/D-form
// instrs STP
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD_1ALU],
             (instregex "^STP[SD](pre|post)$")>;

// Store vector pair, immed post-index, Q-form
// instrs STP
// Store vector pair, immed pre-index, Q-form
// instrs STP
def : InstRW<[WriteAdr, HIP12Write_1c_2ST_2STD_2ALU],
             (instrs STPQpre, STPQpost)>;

//===----------------------------------------------------------------------===//
// 3.14 ASIMD Integer Instructions
//===----------------------------------------------------------------------===//

def : SchedAlias<WriteVd, HIP12Write_2c_1V>;
def : SchedAlias<WriteVq, HIP12Write_2c_1V>;

// ASIMD absolute diff
// instrs SABD, UABD
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]ABDv")>;

// ASIMD absolute diff accum
// instrs SABA, UABA
// ASIMD absolute diff accum long
// instrs SABAL{2}, UABAL{2}
def : InstRW<[HIP12Wr_VA, HIP12Rd_VA], (instregex "^[SU]ABAL?v")>;

// ASIMD absolute diff long
// instrs SABDL{2} UABDL{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]ABDLv")>;

// ASIMD arith, basic
// instrs ABS, ADD, NEG, SUB
def : InstRW<[HIP12Write_1c_1V], (instregex "^(ABS|ADD|NEG|SUB)v")>;

// ASIMD arith, basic
// instrs SADDL{2}, SHADD, SHSUB, SSUBL{2}, UADDL{2}, UHADD, UHSUB, USUBL{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]H(ADD|SUB)v",
                                            "^[SU](ADD|SUB)Lv")>;

// ASIMD arith wide
// instrs SADDW{2}, SSUBW{2}, UADDW{2}, USUBW{2}
def : InstRW<[HIP12Write_1c_1V], (instregex "^[SU](ADD|SUB)Wv")>;

// ASIMD arith, complex
// instrs ADDHN{2}, SQABS, SQADD, SQNEG, SQSUB, SUBHN{2}, SUQADD, UQADD, UQSUB, USQADD
def : InstRW<[HIP12Write_2c_1V],
             (instregex "^(ADDHN|SQABS|SQADD|SQNEG|SQSUB|SUBHN|SUQADD|UQADD|UQSUB|USQADD)v")>;

// ASIMD arith, complex
// instrs RADDHN{2}, RSUBHN{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^R(ADD|SUB)HNv")>;

// ASIMD arith, complex
// instrs SRHADD, URHADD
def : InstRW<[HIP12Write_2c_1V], (instregex "[SU]RHADDv")>;

// ASIMD arith, pair-wise
// instrs ADDP
def : InstRW<[HIP12Write_2c_1V], (instregex "^ADDPv")>;

// ASIMD arith, pair-wise
// instrs SADDLP, UADDLP
def : InstRW<[HIP12Write_3c_1V], (instregex "[SU]ADDLPv")>;

// ASIMD arith, reduce
// instrs ADDV
def : InstRW<[HIP12Write_3c_1V], (instregex "^ADDVv")>;

// ASIMD arith, reduce, S form
// instrs SADDLV, UADDLV
def : InstRW<[HIP12Write_5c_2V], (instregex "^[SU]ADDLVv4i32v$")>;

// ASIMD arith, reduce, H form
// instrs SADDLV, UADDLV
def : InstRW<[HIP12Write_7c_3V], (instregex "^[SU]ADDLVv8i16v$",
                                            "^[SU]ADDLVv4i16v$")>;

// ASIMD arith, reduce, B form
// instrs SADDLV, UADDLV
def : InstRW<[HIP12Write_9c_4V], (instregex "^[SU]ADDLVv16i8v$",
                                            "^[SU]ADDLVv8i8v$")>;

// ASIMD compare
// instrs CMGT, CMEQ, CMGE, CMLT, CMLE, CMTST, CMHI, CMHS
// Handled by SchedAlias<WriteV[dq], ...>

// ASIMD dot product
// instrs SDOT, UDOT
// ASIMD dot product using signed and unsigned integers
// instrs SUDOT, USDOT
def : InstRW<[HIP12Wr_VDOT, HIP12Rd_VDOT],
             (instregex "^([SU]|SU|US)DOT(lane)?(v8|v16)i8$")>;

// ASIMD logical
// instrs AND, NOT, ORN, ORR, MOV, BIC, EOR, MVN
def : InstRW<[HIP12Write_1c_1V], (instregex "^(AND|NOT|ORN|ORR|BIC|EOR)v")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[HIP12Wr_VMMA, HIP12Rd_VMMA], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD max/min
// instrs SMAX, SMIN, UMAX, UMIN
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU](MAX|MIN)v")>;

// ASIMD max/min pair-wise
// instrs SMAXP, SMINP, UMAXP, UMINP
def : InstRW<[HIP12Write_3c_1V], (instregex "^[SU](MAX|MIN)Pv")>;

// ASIMD max/min, reduce, S form
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_6c_2V], (instregex "^[SU](MAX|MIN)Vv4i32v$")>;

// ASIMD max/min, reduce, H form
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_9c_3V], (instregex "^[SU](MAX|MIN)Vv4i16v$",
                                            "^[SU](MAX|MIN)Vv8i16v$")>;

// ASIMD max/min, reduce, B form
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_12c_4V], (instregex "^[SU](MAX|MIN)Vv8i8v$",
                                             "^[SU](MAX|MIN)Vv16i8v$")>;

// ASIMD multiply, B/H/S form
// instrs MUL, SQDMULH, SQRDMULH
def : InstRW<[HIP12Write_3c_1V], (instregex "^MULv", "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate, B/H/S form
// instrs MLA, MLS
def : InstRW<[HIP12Wr_VMA, HIP12Rd_VMA], (instregex "^MLAv", "^MLSv")>;

// ASIMD multiply accumulate high
// instrs SQRDMLAH, SQRDMLSH
def : InstRW<[HIP12Wr_VMAH, HIP12Rd_VMAH], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply long
// instrs SMULL{2}, SQDMULL{2}, UMULL{2}
def : InstRW<[HIP12Write_3c_1V], (instregex "^[SU]MULLv", "^SQDMULL[iv]")>;

// ASIMD multiply accumulate long
// instrs SMLAL{2}, SMLSL{2}, UMLAL{2}, UMLSL{2},
def : InstRW<[HIP12Wr_VMAL, HIP12Rd_VMAL], (instregex "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate saturating long
// instrs SQDMLAL{2}, SQDMLSL{2}
def : InstRW<[HIP12Wr_VMAQL, HIP12Rd_VMAQL], (instregex "^SQDML[AS]L[iv]")>;

// ASIMD multiply/multiply long (8x8) polynomial
// instrs PMUL, PMULL{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^PMULL?(v8i8|v16i8)$")>;

// ASIMD pairwise add and accumulate long
// instrs SADALP, UADALP
def : InstRW<[HIP12Wr_VPA, HIP12Rd_VPA], (instregex "^[SU]ADALPv")>;

// ASIMD shift accumulate
// instrs SSRA, SRSRA, USRA, URSRA
def : InstRW<[HIP12Wr_VSA, HIP12Rd_VSA], (instregex "^[SU]SRA[dv]", "^[SU]RSRA[dv]")>;

// ASIMD shift by immed, basic
// instrs SHL, SHLL{2}, SHRN{2}, SSHLL{2}, SSHR, SXTL{2}, USHLL{2}, USHR, UXTL{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^SHL[dv]", "^SHLLv", "^SHRNv",
                                            "^SSHLLv", "^SSHR[dv]", "^USHLLv",
                                            "^USHR[dv]")>;

// ASIMD shift by immed and insert, basic
// instrs SLI, SRI
def : InstRW<[HIP12Write_2c_1V], (instregex "^SLI[dv]", "^SRI[dv]")>;

// ASIMD shift by immed, complex
// instrs RSHRN{2}, SQRSHRN{2}, SQRSHRUN{2}, SRSHR, UQRSHRN{2}, URSHR
def : InstRW<[HIP12Write_3c_1V], (instregex "^RSHRNv", "^UQRSHRN[bhsv]",
                                            "^SQRSHRU?N[bhsv]", "^[SU]RSHR[dv]")>;

// ASIMD shift by immed, complex
// instrs SQSHRN{2}, SQSHRUN{2}, SQSHLU, UQSHRN{2},
// ASIMD shift by immed, complex
// instrs SQSHL, UQSHL
def : InstRW<[HIP12Write_2c_1V],
             (instregex "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHRU?N[bhsv]", "^UQSHRN[bhsv]")>;

// ASIMD shift by register, basic
// instrs SSHL, USHL
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]SHLv")>;

// ASIMD shift by register, complex
// instrs SRSHL, SQRSHL, URSHL, UQRSHL,
def : InstRW<[HIP12Write_3c_1V],
             (instregex "^[SU]RSHLv", "^[SU]QRSHLv")>;

// ASIMD shift by register, complex
// instrs SQSHL, UQSHL
def : InstRW<[HIP12Write_2c_1V],
             (instregex "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)$")>;

//===----------------------------------------------------------------------===//
// 3.15 ASIMD Floating-Point Instructions
//===----------------------------------------------------------------------===//

// ASIMD FP absolute value
// instrs FABS
def : InstRW<[HIP12Write_1c_1V], (instregex "^FABSv")>;

// ASIMD FP negative value
// instrs FNEG
def : InstRW<[HIP12Write_1c_1V], (instregex "^FNEGv")>;

// ASIMD FP absolute difference
// instrs FABD
def : InstRW<[HIP12Write_2c_1V], (instregex "^FABDv")>;

// ASIMD FP arith
// instrs FADD, FSUB
def : InstRW<[HIP12Write_2c_1V], (instregex "^F(ADD|SUB)v")>;

// ASIMD FP add pairwise
// instrs FADDP
def : InstRW<[HIP12Write_3c_1V], (instregex "^FADDPv")>;

// ASIMD FP compare
// instrs FACGE, FACGT, FCMEQ, FCMGE, FCMGT, FCMLE, FCMLT
def : InstRW<[HIP12Write_2c_1V], (instregex "^FACG[ET]v",
                                            "^FCM(EQ|GE|GT|LE|LT)v")>;

// ASIMD FP convert long
// instrs FCVTL{2}
def : InstRW<[HIP12Write_4c_2V], (instregex "^FCVTLv")>;

// ASIMD FP convert narrow
// instrs FCVTXN{2}
def : InstRW<[HIP12Write_4c_2V], (instregex "^FCVTXN(v2|v4)f32")>;

// ASIMD FP convert narrow D-form
// instrs FCVTN{2}
def : InstRW<[HIP12Write_3c_1V], (instregex "^FCVTN(v2i32|v4i16)")>;

// ASIMD FP convert narrow Q-form
// instrs FCVTN{2}
def : InstRW<[HIP12Write_4c_2V], (instregex "^FCVTN(v4i32|v8i16)")>;

// ASIMD FP convert to Integer/Fixed point, D-form
// instrs FCVTNS, FCVTNU, FCVTMS, FCVTMU, FCVTAS, FCVTAU, FCVTPS, FCVTPU, FCVTZS, FCVTZU
// ASIMD FP convert from Integer/Fixed-point to FP, D-form
// instrs SCVTF, UCVTF
def : InstRW<[HIP12Write_3c_1V],
             (instregex "^FCVT[AMNPZ][SU](v2f32|v4f16)$",
                        "^FCVT[AMNPZ][SU](v2i32|v4i16)_shift$",
                        "^FCVT[AMNPZ][SU]v1i(16|32|64)$",
                        "^FCVTZ[SU][hsd]$",
                        "^[SU]CVTF(v2f32|v4f16)$",
                        "^[SU]CVTF(v2i32|v4i16)_shift$",
                        "^[SU]CVTFv1i(16|32|64)$",
                        "^[SU]CVTF[hsd]$")>;

// ASIMD FP convert to Integer/Fixed point, Q-form
// instrs FCVTNS, FCVTNU, FCVTMS, FCVTMU, FCVTAS, FCVTAU, FCVTPS, FCVTPU, FCVTZS, FCVTZU
// ASIMD FP convert from Integer/Fixed-point to FP, Q-form
// instrs SCVTF, UCVTF
def : InstRW<[HIP12Write_4c_2V],
             (instregex "^FCVT[AMNPZ][SU](v2f64|v4f32|v8f16)$",
                        "^FCVT[AMNPZ][SU](v2i64|v4i32|v8i16)_shift$",
                        "^[SU]CVTF(v2f64|v4f32|v8f16)$",
                        "^[SU]CVTF(v2i64|v4i32|v8i16)_shift$")>;

// ASIMD FP divide, D-form, F16
// instrs FDIV
def : InstRW<[HIP12Write_4c_1V], (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
// instrs FDIV
def : InstRW<[HIP12Write_6c_1V], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
// instrs FDIV
def : InstRW<[HIP12Write_5c_1V_2rc], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
// instrs FDIV
def : InstRW<[HIP12Write_7c_1V_2rc], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
// instrs FDIV
def : InstRW<[HIP12Write_9c_1V_2rc], (instrs FDIVv2f64)>;

// ASIMD FP square root, D-form, F16
// instrs FSQRT
def : InstRW<[HIP12Write_4c_1V], (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
// instrs FSQRT
def : InstRW<[HIP12Write_6c_1V], (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
// instrs FSQRT
def : InstRW<[HIP12Write_5c_1V_2rc], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
// instrs FSQRT
def : InstRW<[HIP12Write_7c_1V_2rc], (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
// instrs FSQRT
def : InstRW<[HIP12Write_9c_1V_2rc], (instrs FSQRTv2f64)>;

// ASIMD FP complex multiply-add
def : InstRW<[HIP12Wr_VFCMA, HIP12Rd_VFCMA], (instregex "^FCMLAv")>;

// ASIMD FP complex add
def : InstRW<[HIP12Write_3c_1V], (instregex "^FCADDv")>;

// ASIMD FP max/min
// instrs FMAX, FMAXNM, FMIN, FMINNM,
def : InstRW<[HIP12Write_2c_1V], (instregex "^F(MAX|MIN)(NM)?v")>;

// ASIMD FP max/min, pairwise
// instrs FMAXP, FMAXNMP, FMINP, FMINNMP
def : InstRW<[HIP12Write_3c_1V], (instregex "^F(MAX|MIN)(NM)?Pv")>;

// ASIMD FP max/min reduce, F16
// instrs FMAXV, FMAXNMV, FMINV, FMINNMV
// ASIMD FP max/min reduce, F32, F64
// instrs FMAXV, FMAXNMV, FMINV, FMINNMV
def : InstRW<[HIP12Write_3c_1V], (instregex "^F(MAX|MIN)(NM)?Vv")>;

// ASIMD FP multiply
// instrs FMUL, FMULX
def : InstRW<[HIP12Wr_VFM], (instregex "^FMULv", "^FMULXv")>;

// ASIMD FP fused multiply-add
// instrs FMLA, FMLS
def : InstRW<[HIP12Wr_VFMA, HIP12Rd_VFMA], (instregex "^FMLAv", "^FMLSv")>;

// ASIMD FP fused multiply-add long
// instrs FMLAL{2}, FMLSL{2}
def : InstRW<[HIP12Wr_VFMAL, HIP12Rd_VFMAL], (instregex "^FML[AS]L2?(lane)?v")>;

// ASIMD FP round to FP integral, D-form
// instrs FRINTN, FRINTM, FRINTP, FRINTZ, FRINTA, FRINTX, FRINTI, FRINT32X, FRINT64X, FRINT32Z, FRINT64Z
def : InstRW<[HIP12Write_3c_1V],
             (instregex "^FRINT[AIMNPXZ](v2f32|v4f16)$",
                        "^FRINT(32|64)[XZ](v2f32|v4f16)$")>;

// ASIMD FP round to FP integral, Q-form
// instrs FRINTN, FRINTM, FRINTP, FRINTZ, FRINTA, FRINTX, FRINTI, FRINT32X, FRINT64X, FRINT32Z, FRINT64Z
def : InstRW<[HIP12Write_4c_2V],
             (instregex "^FRINT[AIMNPXZ](v8f16|v4f32|v2f64)$",
                        "^FRINT(32|64)[XZ](v8f16|v4f32|v2f64)$")>;

//===----------------------------------------------------------------------===//
// 3.16 ASIMD Bfloat16 (BF16) Instructions
//===----------------------------------------------------------------------===//

// ASIMD convert, F32 to BF16
// instrs BFCVTN{2}
def : InstRW<[HIP12Write_4c_2V], (instrs BFCVTN, BFCVTN2)>;

// ASIMD dot product
// instrs BFDOT
def : InstRW<[HIP12Write_6c_2V], (instrs BFDOTv4bf16, BFDOTv8bf16)>;

// ASIMD matrix multiply accumulate
// instrs BFMMLA
def : InstRW<[HIP12Write_8c_4V], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
// instrs BFMLALB, BFMLALT
def : InstRW<[HIP12Wr_VBFMAL, HIP12Rd_VBFMAL],
             (instrs BFMLALB, BFMLALBIdx, BFMLALT, BFMLALTIdx)>;

// Scalar convert, F32 to BF16
// instrs BFCVT
def : InstRW<[HIP12Write_3c_1V], (instrs BFCVT)>;

//===----------------------------------------------------------------------===//
// 3.17 ASIMD Miscellaneous Instructions
//===----------------------------------------------------------------------===//

// ASIMD bit reverse
// instrs RBIT
def : InstRW<[HIP12Write_1c_1V], (instregex "^RBITv")>;

// ASIMD bitwise insert
// instrs BIF, BIT, BSL
def : InstRW<[HIP12Write_1c_1V], (instregex "^(BIF|BIT|BSL)v")>;

// ASIMD count
// instrs CLZ, CLS
def : InstRW<[HIP12Write_1c_1V], (instregex "^CL[Z|S]v")>;

// ASIMD count, D/S
// instrs CNT
def : InstRW<[HIP12Write_2c_1V], (instrs CNTv16i8)>;

// ASIMD count, B/H
// instrs CNT
def : InstRW<[HIP12Write_1c_1V], (instrs CNTv8i8)>;

// ASIMD duplicate, gen reg
// instrs DUP
def : InstRW<[HIP12Write_6c_2ALU0134_2V], (instregex "^DUPv.+gpr")>;

// ASIMD duplicate, element
// instrs DUP
def : InstRW<[HIP12Write_2c_1V], (instregex "^DUPv.+lane")>;

// ASIMD extract
// instrs EXT
def : InstRW<[HIP12Write_2c_1V], (instregex "^EXTv")>;

// ASIMD extract narrow
// instrs XTN{2}
def : InstRW<[HIP12Write_2c_1V], (instregex "^XTNv")>;

// ASIMD extract narrow, saturating
// instrs SQXTN{2}, SQXTUN{2}, UQXTN{2},
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD insert, element to element
// instrs INS
def : InstRW<[HIP12Write_2c_1V], (instregex "^INSvi(8|16|32|64)lane$")>;

// ASIMD FP move, immed
// instrs FMOV
def : InstRW<[HIP12Write_2c_1V], (instregex "^FMOVv(2|4|8)f")>;

// ASIMD move, integer immediate
// instrs MOVI
def : InstRW<[HIP12Write_2c_1V], (instregex "^MOVIv")>;

// ASIMD move, integer immediate
// instrs MVNI
def : InstRW<[HIP12Write_1c_1V], (instregex "^MVNIv")>;

// ASIMD reciprocal and square root estimate, D-form
// instrs URECPE, URSQRTE
def : InstRW<[HIP12Write_3c_1V], (instrs URECPEv2i32, URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, Q-form
// instrs URECPE, URSQRTE
def : InstRW<[HIP12Write_4c_2V], (instrs URECPEv4i32, URSQRTEv4i32)>;

// ASIMD FP reciprocal and square root estimate, D-form
// instrs FRECPE, FRSQRTE
def : InstRW<[HIP12Write_3c_1V],
             (instregex "^FRECPE(v4f16|v2f32)$",
                        "^FRSQRTE(v4f16|v2f32)$")>;

// ASIMD FP reciprocal and square root estimate, Q-form
// instrs FRECPE, FRSQRTE
def : InstRW<[HIP12Write_4c_2V],
             (instregex "^FRECPE(v8f16|v4f32|v2f64)$",
                        "^FRSQRTE(v8f16|v4f32|v2f64)$")>;

// ASIMD FP reciprocal exponent
// instrs FRECPX
def : InstRW<[HIP12Write_3c_1V], (instregex "^FRECPXv")>;

// ASIMD FP reciprocal step
// instrs FRECPS, FRSQRTS
def : InstRW<[HIP12Wr_VRECP, HIP12Rd_VRECP, HIP12Rd_VRECP],
             (instregex "^FRECPS(32|64|v)", "^FRSQRTS(32|64|v)")>;

// ASIMD reverse
// instrs REV16, REV32, REV64
def : InstRW<[HIP12Write_2c_1V], (instregex "^REV(16|32|64)v")>;

// ASIMD table lookup 1 or 2 table regs
// instrs TBL
def : InstRW<[HIP12Write_2c_1V], (instrs TBLv8i8One, TBLv16i8One,
                                         TBLv8i8Two, TBLv16i8Two)>;

// ASIMD table lookup 3 table regs
// instrs TBL
def : InstRW<[HIP12Write_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// ASIMD table lookup 4 table regs
// instrs TBL
def : InstRW<[HIP12Write_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension 1 table reg
// instrs TBX
def : InstRW<[HIP12Write_2c_1V], (instrs TBXv8i8One, TBXv16i8One)>;

// ASIMD table lookup extension 2 table reg
// instrs TBX
def : InstRW<[HIP12Write_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup extension 3 table reg
// instrs TBX
def : InstRW<[HIP12Write_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension 4 table reg
// instrs TBX
def : InstRW<[HIP12Write_6c_4V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD move FP to general register
// instrs SMOV, UMOV
def : InstRW<[HIP12Write_2c_1V], (instregex "^[SU]MOVv")>;

// ASIMD transfer, gen reg to element
// instrs INS
def : InstRW<[HIP12Write_6c_2V_2ALU0134], (instregex "^INSvi(8|16|32|64)gpr$")>;

// ASIMD transpose
// instrs TRN1, TRN2
def : InstRW<[HIP12Write_2c_1V], (instregex "^TRN[12]v")>;

// ASIMD uzip/zip
// instrs UZP1, UZP2, ZIP1, ZIP2
def : InstRW<[HIP12Write_2c_1V], (instregex "^UZP[12]v", "^ZIP[12]v")>;

//===----------------------------------------------------------------------===//
// 3.18 ASIMD Load Instructions
//===----------------------------------------------------------------------===//

// SIMD load, 1-element, multiple, 1-reg
// instrs LD1
def : InstRW<[HIP12Write_6c_1LD],           (instregex "^LD1Onev(8b|4h|2s|1d)$",
                                                       "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_6c_1LD], (instregex "^LD1Onev(8b|4h|2s|1d)_POST$",
                                                       "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// SIMD load, 1-element, multiple, 2-reg
// instrs LD1
def : InstRW<[HIP12Write_6c_1LD],           (instregex "^LD1Twov(8b|4h|2s|1d)$",
                                                       "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_6c_1LD], (instregex "^LD1Twov(8b|4h|2s|1d)_POST$",
                                                       "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// SIMD load, 1-element, multiple, 3-reg
// instrs LD1
def : InstRW<[HIP12Write_7c_2LD],           (instregex "^LD1Threev(8b|4h|2s|1d)$",
                                                       "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_7c_2LD], (instregex "^LD1Threev(8b|4h|2s|1d)_POST$",
                                                       "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// SIMD load, 1-element, multiple, 4-reg
// instrs LD1
def : InstRW<[HIP12Write_7c_2LD],           (instregex "^LD1Fourv(8b|4h|2s|1d)$",
                                                       "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_7c_2LD], (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$",
                                                       "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// SIMD load, 1-element, single, 1 lane
// instrs LD1
def : InstRW<[HIP12Write_8c_1LD_1V],           (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_1V], (instregex "LD1i(8|16|32|64)_POST$")>;

// SIMD load, 1-element, single, replicate to all lanes
// instrs LD1R
def : InstRW<[HIP12Write_8c_1LD_1V],           (instregex "LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_1V], (instregex "LD1Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[HIP12Write_8c_1LD_1V],           (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_1V], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

// SIMD load, 2-elements, multiple, Q form
// instrs LD2
def : InstRW<[HIP12Write_8c_2LD_2V],           (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_2LD_2V], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

// SIMD load, 2-elements, multiple, D form
// instrs LD2
def : InstRW<[HIP12Write_8c_1LD_2V],           (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_2V], (instregex "LD2Twov(8b|4h|2s)_POST$")>;

// SIMD load, 2-element, single, 1 lane
// instrs LD2
def : InstRW<[HIP12Write_8c_1LD_2V],           (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_2V], (instregex "LD2i(8|16|32|64)_POST$")>;

// SIMD load, 2-element, single, replicate to all lanes
// instrs LD2R
def : InstRW<[HIP12Write_8c_1LD_2V],           (instregex "LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_2V], (instregex "LD2Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[HIP12Write_8c_1LD_2V],           (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_1LD_2V], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

// SIMD load LD3 (multiple structures), Q-form, B/H/S
// instrs LD3
def : InstRW<[HIP12Write_10c_3LD_6V],           (instregex "LD3Threev(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, HIP12Write_10c_3LD_6V], (instregex "LD3Threev(16b|8h|4s)_POST$")>;

// SIMD load LD3 (multiple structures), Q-form,D
// instrs LD3
def : InstRW<[HIP12Write_8c_3LD_3V],           (instregex "LD3Threev2d$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_3LD_3V], (instregex "LD3Threev2d_POST$")>;

// SIMD load LD3 (multiple structures), D-form
// instrs LD3
def : InstRW<[HIP12Write_8c_2LD_3V],           (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_2LD_3V], (instregex "LD3Threev(8b|4h|2s)_POST$")>;

// SIMD load, 3-element, single, 1 lane
// instrs LD3
def : InstRW<[HIP12Write_8c_2LD_3V],           (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_2LD_3V], (instregex "LD3i(8|16|32|64)_POST$")>;

// SIMD load, 3-element, single, replicate to all lanes
// instrs LD3R
def : InstRW<[HIP12Write_8c_2LD_3V],           (instregex "LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_2LD_3V], (instregex "LD3Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[HIP12Write_8c_2LD_3V],           (instregex "LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_2LD_3V], (instregex "LD3Rv(16b|8h|4s|2d)_POST$")>;

// SIMD load, 4-element, multiple, Q-form, B/H/S
// instrs LD4
def : InstRW<[HIP12Write_11c_4LD_8V],           (instregex "LD4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, HIP12Write_11c_4LD_8V], (instregex "LD4Fourv(16b|8h|4s)_POST$")>;

// SIMD load, 4-element, multiple, Q-form, D
// instrs LD4
def : InstRW<[HIP12Write_9c_4LD_4V],           (instregex "LD4Fourv2d$")>;
def : InstRW<[WriteAdr, HIP12Write_9c_4LD_4V], (instregex "LD4Fourv2d_POST$")>;

// SIMD load, 4-element, multiple, D-form
// instrs LD4
def : InstRW<[HIP12Write_8c_3LD_4V],           (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_3LD_4V], (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// SIMD load, 4-element, single, 1 lane
// instrs LD4
def : InstRW<[HIP12Write_8c_3LD_4V],           (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_3LD_4V], (instregex "LD4i(8|16|32|64)_POST$")>;

// SIMD load, 4-element, single, replicate to all lanes
// instrs LD4R
def : InstRW<[HIP12Write_8c_3LD_4V],           (instregex "LD4Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_3LD_4V], (instregex "LD4Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[HIP12Write_8c_3LD_4V],           (instregex "LD4Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_8c_3LD_4V], (instregex "LD4Rv(16b|8h|4s|2d)_POST$")>;

//===----------------------------------------------------------------------===//
// 3.19 ASIMD Store Instructions
//===----------------------------------------------------------------------===//

// SIMD store, 1-element, multiple, 1 reg, Q-form
// instrs ST1
def : InstRW<[HIP12Write_1c_1ST_1STD],           (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// SIMD store, 1-element, multiple, 2 reg, Q-form
// instrs ST1
def : InstRW<[HIP12Write_1c_2ST_2STD],           (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_2ST_2STD], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// SIMD store, 1-element, multiple, 3 reg, Q-form
// instrs ST1
def : InstRW<[HIP12Write_2c_3ST_3STD],           (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_2c_3ST_3STD], (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// SIMD store, 1-element, multiple, 4 reg, Q-form
// instrs ST1
def : InstRW<[HIP12Write_2c_4ST_4STD],           (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_2c_4ST_4STD], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// SIMD store, 1-element, multiple, 1 reg, D-form
// instrs ST1
def : InstRW<[HIP12Write_1c_1ST_1STD_1V],           (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD_1V], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// SIMD store, 1-element, multiple, 2 reg, D-form
// instrs ST1
def : InstRW<[HIP12Write_1c_1ST_1STD_1V],           (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD_1V], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// SIMD store, 1-element, multiple, 3 reg, D-form
// instrs ST1
def : InstRW<[HIP12Write_1c_2ST_2STD_2V],           (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_2ST_2STD_2V], (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// SIMD store, 1-element, multiple, 4 reg, D-form
// instrs ST1
def : InstRW<[HIP12Write_4c_2ST_2STD_2V],           (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_2ST_2STD_2V], (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// SIMD store, 1-element, single, 1 lane
// instrs ST1
def : InstRW<[HIP12Write_1c_1ST_1STD],           (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD], (instregex "ST1i(8|16|32|64)_POST$")>;

// SIMD store, 2-element, multiple, Q-form
// instrs ST2
def : InstRW<[HIP12Write_1c_2ST_2STD],           (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_2ST_2STD], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// SIMD store, 2-element, multiple, D-form
// instrs ST2
def : InstRW<[HIP12Write_1c_1ST_1STD],           (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD], (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// SIMD store, 2-element, single, 1lane
// instrs ST2
def : InstRW<[HIP12Write_1c_1ST_1STD],           (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_1c_1ST_1STD], (instregex "ST2i(8|16|32|64)_POST$")>;

// SIMD store, 3-element, multiple, Q-form,B,H,S
// instrs ST3
def : InstRW<[HIP12Write_5c_6V_3ST_3STD],           (instregex "ST3Threev(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, HIP12Write_5c_6V_3ST_3STD], (instregex "ST3Threev(16b|8h|4s)_POST$")>;

// SIMD store, 3-element, multiple, Q-form,D
// instrs ST3
def : InstRW<[HIP12Write_5c_6V_3ST_3STD],           (instregex "ST3Threev2d$")>;
def : InstRW<[WriteAdr, HIP12Write_5c_6V_3ST_3STD], (instregex "ST3Threev2d_POST$")>;

// SIMD store, 3-element, multiple, D-form
// instrs ST3
def : InstRW<[HIP12Write_4c_4V_2ST_2STD],           (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_4V_2ST_2STD], (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// SIMD store, 3-element, single, 1 lane
// instrs ST3
def : InstRW<[HIP12Write_4c_4V_2ST_2STD],           (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_4V_2ST_2STD], (instregex "ST3i(8|16|32|64)_POST$")>;

// SIMD store, 4-element, multiple, Q-form
// instrs ST4
def : InstRW<[HIP12Write_5c_8V_4ST_4STD],           (instregex "ST4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, HIP12Write_5c_8V_4ST_4STD], (instregex "ST4Fourv(16b|8h|4s|2d)_POST$")>;

// SIMD store, 4-element, multiple, D-form
// instrs ST4
def : InstRW<[HIP12Write_4c_4V_2ST_2STD],           (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_4V_2ST_2STD], (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// SIMD store, 4-element, single, 1 lane,B/H/S
// instrs ST4
def : InstRW<[HIP12Write_4c_2V_1ST_1STD],           (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_2V_1ST_1STD], (instregex "ST4i(8|16|32)_POST$")>;

// SIMD store, 4-element, single, 1 lane,D
// instrs ST4
def : InstRW<[HIP12Write_4c_4V_2ST_2STD],           (instregex "ST4i64$")>;
def : InstRW<[WriteAdr, HIP12Write_4c_4V_2ST_2STD], (instregex "ST4i64_POST$")>;

//===----------------------------------------------------------------------===//
// 3.20 Cryptography Extensions
//===----------------------------------------------------------------------===//

// crypto AES ops
// instrs AESD, AESE, AESMC, AESIMC
def : InstRW<[HIP12Write_2c_1V], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
// instrs PMULL{2}
def : InstRW<[HIP12Write_2c_1V], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration ops
def : InstRW<[HIP12Write_2c_1V], (instregex "^SHA1H")>;

// Crypto SHA1 hash acceleration ops
def : InstRW<[HIP12Write_4c_1V02], (instregex "^SHA1[CMP]")>;

// Crypto SHA1 schedule acceleration ops
// instrs SHA1SU0, SHA1SU1
def : InstRW<[HIP12Write_2c_1V02], (instregex "^SHA1(SU0|SU1)")>;

// Crypto SHA256 hash acceleration ops
// instrs SHA256H, SHA256H2
def : InstRW<[HIP12Write_4c_1V02], (instregex "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
// instrs SHA256SU0, SHA256SU1
def : InstRW<[HIP12Write_2c_1V02], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
// instrs SHA512H, SHA512H2, SHA512SU0, SHA512SU1
def : InstRW<[HIP12Write_2c_1V02], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[HIP12Write_1c_1V], (instrs BCAX, EOR3, RAX1)>;

// Crypto SHA3 ops
def : InstRW<[HIP12Write_2c_1V], (instrs XAR)>;

// Crypto SM3 ops
// instrs SM3PARTW1, SM3PARTW2, SM3SS1, SM3TT1A, SM3TT1B, SM3TT2A, SM3TT2B
def : InstRW<[HIP12Write_2c_1V02], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                              "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
// instrs SM4E, SM4EKEY
def : InstRW<[HIP12Write_4c_1V], (instrs SM4E, SM4ENCKEY)>;

//===----------------------------------------------------------------------===//
// 3.21 CRC
//===----------------------------------------------------------------------===//

// CRC checksum ops
// instrs CRC32, CRC32C
def : InstRW<[HIP12Write_2c_1ALU25], (instregex "^CRC32")>;

//===----------------------------------------------------------------------===//
// 3.22 SVE Predicate instructions
//===----------------------------------------------------------------------===//

// Loop control, based on predicate
// instrs BRKA, BRKB
def : InstRW<[HIP12Write_2c_1V02], (instrs BRKA_PPmP, BRKA_PPzP,
                                           BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
// instrs BRKAS, BRKBS
def : InstRW<[HIP12Write_2c_1V02], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
// instrs BRKN, BRKPA, BRKPB
def : InstRW<[HIP12Write_2c_2V], (instrs BRKN_PPzP, BRKPA_PPzPP,
                                         BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
// instrs BRKNS, BRKPAS, BRKPBS
def : InstRW<[HIP12Write_2c_2V], (instrs BRKNS_PPzP, BRKPAS_PPzPP,
                                         BRKPBS_PPzPP)>;

// Loop control, based on GPR
// instrs WHILEGE, WHILEGT, WHILEHI, WHILEHS, WHILELE, WHILELO, WHILELS, WHILELT, WHILERW, WHILEWR
def : InstRW<[HIP12Write_2c_1ALU25],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]",
                        "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Loop terminate
// instrs CTERMEQ, CTERMNE
def : InstRW<[HIP12Write_1c_1ALU14], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// Predicate counting scalar
// instrs ADDPL, ADDVL, CNTB, CNTH, CNTW, CNTD, DECB, DECH, DECW, DECD, INCB, INCH, INCW, INCD, RDVL
def : InstRW<[HIP12Write_1c_1ALU],
             (instregex "^(CNT|SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI",
                        "^UQ(DEC|INC)[BHWD]_WPiI")>;
def : InstRW<[HIP12Write_1c_1ALU], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;

// Predicate counting scalar
// instrs SQDECB, SQDECH, SQDECW, SQDECD, SQINCB, SQINCH, SQINCW, SQINCD, UQDECB, UQDECH, UQDECW, UQDECD, UQINCB, UQINCH, UQINCW, UQINCD
def : InstRW<[HIP12Write_2c_1ALU25], (instregex "^(DEC|INC)[BHWD]_XPiI")>;

// Predicate counting scalar, active predicate
// instrs DECP, INCP
def : InstRW<[HIP12Write_6c_1V02_1ALU],
             (instregex "^(DEC|INC)P_XP_[BHSD]")>;

// Predicate counting scalar, active predicate
// instrs CNTP
def : InstRW<[HIP12Write_5c_2V], (instregex "^CNTP_XPP_[BHSD]")>;

// Predicate counting scalar, active predicate
// instrs SQDECP, SQINCP, UQDECP, UQINCP
def : InstRW<[HIP12Write_7c_1V02_1ALU25],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;

// Predicate counting vector, active predicate
// instrs DECP, INCP,
def : InstRW<[HIP12Write_3c_2V], (instregex "^(DEC|INC)P_ZP_[HSD]")>;

// Predicate counting vector, active predicate
// instrs SQDECP, SQINCP, UQDECP, UQINCP
def : InstRW<[HIP12Write_4c_2V],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;

// Predicate logical
// instrs AND, BIC, EOR, MOV, NAND, NOR, NOT, ORN, ORR
def : InstRW<[HIP12Write_2c_2V], (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
// instrs ANDS, BICS, EORS, NANDS, NORS, NOTS, ORNS, ORRS
def : InstRW<[HIP12Write_2c_2V], (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
// instrs REV
def : InstRW<[HIP12Write_2c_2V], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
// instrs SEL
def : InstRW<[HIP12Write_2c_2V], (instrs SEL_PPPP)>;

// Predicate set
// instrs PFALSE, PTRUE
def : InstRW<[HIP12Write_2c_2V], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
// instrs PTRUES
def : InstRW<[HIP12Write_2c_2V], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
// instrs PFIRST, PNEXT
def : InstRW<[HIP12Write_2c_2V], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;

// Predicate test
// instrs PTEST
def : InstRW<[HIP12Write_2c_2V], (instrs PTEST_PP, PTEST_PP_ANY, PTEST_PP_FIRST)>;

// Predicate transpose
// instrs TRN1, TRN2
def : InstRW<[HIP12Write_2c_2V], (instregex "^TRN[12]_PPP_[BHSD]")>;

// Predicate unpack and widen
// instrs PUNPKHI, PUNPKLO
def : InstRW<[HIP12Write_2c_2V], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
// instrs ZIP1, ZIP2, UZP1, UZP2
def : InstRW<[HIP12Write_2c_2V], (instregex "^(ZIP|UZP)[12]_PPP_[BHSD]")>;

//===----------------------------------------------------------------------===//
// 3.23 SVE Integer Instructions
//===----------------------------------------------------------------------===//

// Arithmetic, absolute diff SABD, UABD SVE256
// instrs SABD, UABD
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^[SU]ABD_ZPmZ_[BHSD]",
                                                   "^[SU]ABD_ZPZZ_[BHSD]")>;

// Arithmetic, absolute diff accum SVE256
// instrs SABA, UABA
def : InstRW<[HIP12Wr_ZA, HIP12Rd_ZA], (instregex "^[SU]ABA_ZZZ_[BHSD]")>;

// Arithmetic, absolute diff accum long SVE256
// instrs SABALB, SABALT, UABALB, UABALT
def : InstRW<[HIP12Wr_ZAL, HIP12Rd_ZAL], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]")>;

// Arithmetic, absolute diff long SVE256
// instrs SABDLB, SABDLT, UABDLB, UABDLT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]")>;

// Arithmetic, address generation SVE256
// instrs ADR
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^ADR_[SU]XTW_ZZZ_D_[0123]",
                                                   "^ADR_LSL_ZZZ_[SD]_[0123]")>;

// Arithmetic, basic SVE256
// instrs ABS, ADD, SUB, SUBR, NEG, CNOT
def : InstRW<[HIP12Write_1c_1V02_1V13],
             (instregex "^(ABS|ADD|CNOT|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZI_[BHSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]")>;

// Arithmetic, basic SVE256
// instrs SHADD, SHSUB, SHSUBR, SUBHNB, SUBHNT, UHADD, UHSUB, UHSUBR
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^[SU]H(ADD|SUB|SUBR)_(ZPmZ|ZPZZ)_[BHSD]")>;

// Arithmetic, basic SVE256
// instrs SADDWB, SADDWT, SSUBWB, SSUBWT, UADDWB, USUBWB, USUBWT
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^[SU](ADD|SUB)W[BT]_ZZZ_[HSD]")>;

// Arithmetic, basic SVE256
// instrs SADDLB, SADDLBT, SADDLT, SSUBLB, SSUBLBT, SSUBLT, SSUBLTB, UADDLB, UADDLT, USUBLB, USUBLT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^[SU](ADD|SUB)L[BT]_ZZZ_[HSD]")>;

// Arithmetic, complex SVE256
// instrs SQADD, SQSUB, UQADD, UQSUB
// instrs ADDHNB, ADDHNT, RADDHNB, RADDHNT, RSUBHNB, RSUBHNT, SQABS, SQNEG, SQSUBR, SRHADD, SUQADD, UQSUBR, USQADD, URHADD
def : InstRW<[HIP12Write_2c_1V02_1V13],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^SQ(ABS|ADD|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;

// Arithmetic, large integer SVE256
// instrs ADCLB, ADCLT, SBCLB, SBCLT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]")>;

// Arithmetic, pairwise add SVE256
// instrs ADDP
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^ADDP_ZPmZ_[BHSD]")>;

// Arithmetic, pairwise add and accum long SVE256
// instrs SADALP, UADALP
def : InstRW<[HIP12Wr_ZPA, ReadDefault, HIP12Rd_ZPA], (instregex "^[SU]ADALP_ZPmZ_[HSD]")>;

// Arithmetic, shift SVE256
// instrs ASR, LSR, LSL ASRR, LSLR, LSRR
def : InstRW<[HIP12Write_2c_1V02_1V13],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZ[IZ]_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;

// Arithmetic, shift and accumulate SVE256
// instrs SRSRA, SSRA, URSRA, USRA
def : InstRW<[HIP12Wr_ZSA, HIP12Rd_ZSA], (instregex "^[SU]R?SRA_ZZI_[BHSD]")>;

// Arithmetic, shift by immediate SVE256
// instrs SHRNB, SHRNT, SSHLLB, SSHLLT, USHLLB, USHLLT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^SHRN[BT]_ZZI_[BHS]",
                                                   "^[SU]SHLL[BT]_ZZI_[HSD]")>;

// Arithmetic, shift by immediate and insert SVE256
// instrs SLI, SRI
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^(SLI|SRI)_ZZI_[BHSD]")>;

// Arithmetic, shift complex SVE256
// instrs SQSHLU, SQSHRNB, SQSHRNT, SQSHRUNB, SQSHRUNT, UQSHRNB, UQSHRNT
// Arithmetic, shift complex SVE256
// instrs SQSHL, SQSHLR, UQSHL, UQSHLR
// Arithmetic, shift complex SVE256
// instrs RSHRNB, RSHRNT, SQRSHRNB, SQRSHRNT, SQRSHRUNB, SQRSHRUNT, UQRSHRNB, UQRSHRNT
// Arithmetic, shift complex SVE256
// instrs SQRSHL, SQRSHLR, UQRSHL, UQRSHLR
def : InstRW<[HIP12Write_2c_1V02_1V13],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_ZPmZ_[BHSD]",
                        "^[SU]QR?SHL_ZPZZ_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift right for divide SVE256
// instrs ASRD
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^ASRD_(ZPmI|ZPZI)_[BHSD]")>;

// Arithmetic, shift rounding SVE256
// instrs SRSHL, SRSHLR, URSHL, URSHLR
// Arithmetic, shift rounding SVE256
// instrs SRSHR, URSHR
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^[SU]RSHLR?_ZPmZ_[BHSD]",
                                                   "^[SU]RSHL_ZPZZ_[BHSD]",
                                                   "^[SU]RSHR_(ZPmI|ZPZI)_[BHSD]")>;

// Bit manipulation SVE256
// instrs BDEP, BEXT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^(BDEP|BEXT)_ZZZ_[BHSD]")>;

// Bit manipulation SVE256
// instrs BGRP
def : InstRW<[HIP12Write_6c_2V02_2V13], (instregex "^BGRP_ZZZ_[BHSD]")>;

// Bitwise select SVE256
// instrs BSL, BSL1N, BSL2N, NBSL
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ",
                                                   "^(EON|NAND|NOR)_ZZZ")>;

// Count/reverse bits SVE256
// instrs CLS, CLZ, RBIT
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^(CLS|CLZ|RBIT)_ZPmZ_[BHSD]")>;

// Count/reverse bits, B H form SVE256
// instrs CNT
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^CNT_ZPmZ_[BH]")>;

// Count/reverse bits, S D-form SVE256
// instrs CNT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^CNT_ZPmZ_[SD]")>;

// Broadcast logical bitmask immediate to vector SVE256
// instrs DUPM, MOV
def : InstRW<[HIP12Write_1c_1V02_1V13], (instrs DUPM_ZI)>;

// Compare and set flags
// instrs CMPEQ, CMPGE, CMPGT, CMPHI, CMPHS, CMPLE, CMPLO, CMPLS, CMPLT, CMPNE
def : InstRW<[HIP12Write_2c_1V02_1V13],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;

// Complex add SVE256
// instrs CADD
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^CADD_ZZI_[BHSD]")>;

// Complex add SVE256
// instrs SQCADD
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^SQCADD_ZZI_[BHSD]")>;

// Complex dot product SVE256
// instrs CDOT
def : InstRW<[HIP12Wr_ZDOT, HIP12Rd_ZDOT],
             (instrs CDOT_ZZZ_S, CDOT_ZZZI_S, CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size SVE256
// instrs CMLA
def : InstRW<[HIP12Wr_ZCMABHS, HIP12Rd_ZCMABHS],
             (instregex "^CMLA_ZZZ_[BHS]", "^CMLA_ZZZI_[HS]")>;

// Complex multiply-add D element size SVE256
// instrs CMLA
def : InstRW<[HIP12Wr_ZCMAD, HIP12Rd_ZCMAD], (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form SVE256
// instrs CLASTA, CLASTB
def : InstRW<[HIP12Write_9c_1ALU0134_1V02_1V13], (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms SVE256
// instrs CLASTA, CLASTB
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms SVE256
// instrs SPLICE
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^SPLICE_ZPZZ?_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms SVE256
// instrs COMPACT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^COMPACT_ZPZ_[SD]")>;

// Convert to floating point SVE256
// instrs SCVTF, UCVTF
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                                   "^[SU]CVTF_ZPmZ_StoD",
                                                   "^[SU]CVTF_ZPmZ_Sto[HS]",
                                                   "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar SVE256
// instrs CPY
def : InstRW<[HIP12Write_6c_1ALU0134_1V02_1V13], (instregex "^CPY_ZPmR_[BHSD]")>;

// Copy, imm SVE256
// instrs CPY
// Copy, scalar SIMD&FP SVE256
// instrs CPY
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^CPY_ZPm[IV]_[BHSD]",
                                                   "^CPY_ZPzI_[BHSD]")>;

// Divides, 32-bit SVE256
// instrs SDIV, SDIVR, UDIV, UDIVR
def : InstRW<[HIP12Write_7c_2V02_2V13], (instregex "^[SU]DIVR?_ZPmZ_S",
                                                   "^[SU]DIV_ZPZZ_S")>;

// Divides, 64-bit SVE256
// instrs SDIV, SDIVR, UDIV, UDIVR
def : InstRW<[HIP12Write_9c_2V02_2V13], (instregex "^[SU]DIVR?_ZPmZ_D",
                                                   "^[SU]DIV_ZPZZ_D")>;

// Dot product SVE256
// instrs SDOT, UDOT
def : InstRW<[HIP12Wr_ZDOT, HIP12Rd_ZDOT], (instregex "^[SU]DOT_ZZZI?_BtoS",
                                                      "^[SU]DOT_ZZZI?_HtoD")>;

// Dot product, 8-bit, using signed and unsigned integers SVE256
// instrs SUDOT, USDOT
def : InstRW<[HIP12Wr_ZDOT, HIP12Rd_ZDOT], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Duplicate, indexed SVE256
// instrs DUP, MOV
// Duplicate, immediate SVE256
// instrs DUP, MOV
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^DUP_ZI_[BHSD]",
                                                   "^DUP_ZZI_[BHSDQ]")>;

// Duplicate, scalar SVE256
// instrs DUP, MOV
def : InstRW<[HIP12Write_6c_1ALU0134_1V02_1V13], (instregex "^DUP_ZR_[BHSD]")>;

// Extend, sign or zero SVE256
// instrs SXTB, SXTH, SXTW, UXTB, UXTH, UXTW
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                                   "^[SU]XTH_ZPmZ_[SD]",
                                                   "^[SU]XTW_ZPmZ_[D]")>;

// Extract SVE256
// instrs EXT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instrs EXT_ZZI, EXT_ZZI_CONSTRUCTIVE, EXT_ZZI_B)>;

// Extract narrow saturating SVE256
// instrs SQXTNB, SQXTNT, SQXTUNB, SQXTUNT, UQXTNB, UQXTNT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]",
                                                   "^SQXTUN[BT]_ZZ_[BHS]")>;

// Insert operation, scalar SVE256
// instrs INSR
def : InstRW<[HIP12Write_6c_1ALU0134_1V02_1V13], (instregex "^INSR_ZR_[BHSD]")>;

// Insert operation, SIMD and FP scalar SVE256
// instrs INSR
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^INSR_ZV_[BHSD]")>;

// Extract operation, SIMD and FP scalar SVE256
// instrs LASTA, LASTB
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^LAST[AB]_VPZ_[BHSD]")>;

// Extract operation, scalar SVE256
// instrs LASTA, LASTB
def : InstRW<[HIP12Write_5c_1ALU0134_1V02_1V13], (instregex "^LAST[AB]_RPZ_[BHSD]")>;

// Histogram operations SVE256
// instrs HISTCNT, HISTSEG
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^HISTCNT_ZPzZZ_[SD]",
                                                   "^HISTSEG_ZZZ")>;

// Horizontal operations, B, H, S form, immediate operands only SVE256
// instrs INDEX
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^INDEX_II_[BHS]")>;

// Horizontal operations, D form, immediate operands only SVE256
// instrs INDEX
def : InstRW<[HIP12Write_4c_2V02_2V13], (instrs INDEX_II_D)>;

// Horizontal operations, B, H, S form, scalar start, immediate increment SVE256
// instrs INDEX
def : InstRW<[HIP12Write_10c_1ALU0134_1V02_1V13], (instregex "^INDEX_RI_[BHS]")>;

// Horizontal operations, D form, scalar start, immediate increment SVE256
// instrs INDEX
def : InstRW<[HIP12Write_11c_2ALU0134_2V02_2V13], (instregex "^INDEX_RI_D")>;

// Horizontal operations, B, H, S form, immediate start, scalar increment SVE256
// instrs INDEX
def : InstRW<[HIP12Write_10c_1ALU0134_1V02_1V13], (instregex "^INDEX_IR_[BHS]")>;

// Horizontal operations, D form, immediate start, scalar increment SVE256
// instrs INDEX
def : InstRW<[HIP12Write_11c_2ALU0134_2V02_2V13], (instregex "^INDEX_IR_D")>;

// Horizontal operations, B, H, S form, scalar operands only SVE256
// instrs INDEX
def : InstRW<[HIP12Write_10c_3ALU0134_3V02_3V13], (instregex "^INDEX_RR_[BHS]")>;

// Horizontal operations, D form, scalar, scalar operands only SVE256
// instrs INDEX
def : InstRW<[HIP12Write_11c_4ALU0134_4V02_4V13], (instregex "^INDEX_RR_D")>;

// Logical SVE256
// instrs AND, ORR, EOR, BIC, NOT, EON, MOV, ORN
// Logical SVE256
// instrs EORBT, EORTB
def : InstRW<[HIP12Write_1c_1V02_1V13],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|ORR)_ZZZ",
                        "^EOR(BT|TB)_ZZZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^NOT_ZPmZ_[BHSD]")>;

// Max/min, basic and pairwise SVE256
// instrs SMAX, SMIN, UMAX, UMIN,
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                                   "^[SU](MAX|MIN)_ZPmZ_[BHSD]",
                                                   "^[SU](MAX|MIN)_ZPZZ_[BHSD]")>;

// Max/Min, basic and pairwise SVE256
// instrs SMAXP, SMINP, UMAXP, UMINP
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^[SU](MAX|MIN)P_ZPmZ_[BHSD]")>;

// Matching operations
// instrs MATCH, NMATCH
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Matrix multiply-accumulate SVE256
// instrs SMMLA, UMMLA, USMMLA
def : InstRW<[HIP12Wr_ZMMA, HIP12Rd_ZMMA],
             (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix SVE256
// instrs MOVPRFX
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]",
                                                   "^MOVPRFX_ZZ")>;

// Multiply, B, H, S element size SVE256
// instrs MUL, SMULH, UMULH
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_[BHS]",
                                                   "^MUL_ZPZZ_[BHS]",
                                                   "^[SU]MULH_(ZPmZ|ZZZ)_[BHS]",
                                                   "^[SU]MULH_ZPZZ_[BHS]")>;

// Multiply, D element size SVE256
// instrs MUL, SMULH, UMULH
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_D",
                                                   "^MUL_ZPZZ_D",
                                                   "^[SU]MULH_(ZPmZ|ZZZ)_D",
                                                   "^[SU]MULH_ZPZZ_D")>;

// Multiply long, B, H, S element size SVE256
// instrs SMULLB, SMULLT, UMULLB, UMULLT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^[SU]MULL[BT]_ZZZI_[SD]",
                                                   "^[SU]MULL[BT]_ZZZ_[HSD]")>;

// Multiply accumulate, B, H, S element size SVE256
// instrs MLA, MLS, MAD, MSB
def : InstRW<[HIP12Wr_ZMABHS, HIP12Rd_ZMABHS],
             (instregex "^ML[AS]_ZZZI_[HS]", "^ML[AS]_ZPZZZ_[BHS]")>;
def : InstRW<[HIP12Wr_ZMABHS, ReadDefault, HIP12Rd_ZMABHS],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]")>;

// Multiply accumulate, D element size SVE256
// instrs MLA, MLS, MAD, MSB
def : InstRW<[HIP12Wr_ZMAD, HIP12Rd_ZMAD],
             (instregex "^ML[AS]_ZZZI_D", "^ML[AS]_ZPZZZ_D")>;
def : InstRW<[HIP12Wr_ZMAD, ReadDefault, HIP12Rd_ZMAD],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_D")>;

// Multiply accumulate long, B, H, S element size SVE256
// instrs SMLALB, MLALT, SMLSLB, SMLSLT, UMLALB, UMLALT, UMLSLB, UMLSLT
def : InstRW<[HIP12Wr_ZMALBHS, HIP12Rd_ZMALBHS],
             (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HS]",
                        "^[SU]ML[AS]L[BT]_ZZZI_S")>;

// Multiply accumulate long, D element size SVE256
// instrs SMLALB, MLALT, SMLSLB, SMLSLT, UMLALB, UMLALT, UMLSLB, UMLSLT
def : InstRW<[HIP12Wr_ZMALD, HIP12Rd_ZMALD],
             (instregex "^[SU]ML[AS]L[BT]_ZZZ_D", "^[SU]ML[AS]L[BT]_ZZZI_D")>;

// Multiply accumulate saturating doubling long regular, B, H, S element size SVE256
// instrs SQDMLALB, SQDMLALT, SQDMLALBT, SQDMLSLB, SQDMLSLT, SQDMLSLBT
def : InstRW<[HIP12Wr_ZMASQLBHS, HIP12Rd_ZMASQ],
             (instregex "^SQDML[AS]L(B|T|BT)_ZZZ_[HS]",
                        "^SQDML[AS]L[BT]_ZZZI_[S]")>;

// Multiply accumulate saturating doubling long regular, D element size SVE256
// instrs SQDMLALB, SQDMLALT, SQDMLALBT, SQDMLSLB, SQDMLSLT, SQDMLSLBT
def : InstRW<[HIP12Wr_ZMASQLD, HIP12Rd_ZMASQ],
             (instregex "^SQDML[AS]L(B|T|BT)_ZZZ_D",
                        "^SQDML[AS]L[BT]_ZZZI_D")>;

// Multiply saturating doubling high, B, H, S element size SVE256
// instrs SQDMULH
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^SQDMULH_ZZZ_[BHS]",
                                                   "^SQDMULH_ZZZI_[HS]")>;

// Multiply saturating doubling high, D element size SVE256
// instrs SQDMULH
def : InstRW<[HIP12Write_4c_2V02_2V13], (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long, B, H, S element size SVE256
// instrs SQDMULLB, SQDMULLT
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^SQDMULL[BT]_ZZZ_[HS]",
                                                   "^SQDMULL[BT]_ZZZI_S")>;

// Multiply saturating doubling long, D element size SVE256
// instrs SQDMULLB, SQDMULLT
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^SQDMULL[BT]_ZZZ_D",
                                                   "^SQDMULL[BT]_ZZZI_D")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S element size SVE256
// instrs SQRDMLAH, SQRDMLSH
// Multiply saturating rounding doubling regular/complex accumulate, B, H, S element size SVE256
// instrs SQRDCMLAH
def : InstRW<[HIP12Wr_ZMASQBHS, HIP12Rd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZ_[BHS]",
                                                           "^SQRDCMLAH_ZZZ_[BHS]",
                                                           "^SQRDML[AS]H_ZZZI_[HS]",
                                                           "^SQRDCMLAH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element size SVE256
// instrs SQRDMLAH, SQRDMLSH
// Multiply saturating rounding doubling regular/complex accumulate, D element size SVE256
// instrs SQRDCMLAH
def : InstRW<[HIP12Wr_ZMASQD, HIP12Rd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZI?_D",
                                                         "^SQRDCMLAH_ZZZ_D")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size SVE256
// instrs SQRDMULH
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^SQRDMULH_ZZZ_[BHS]",
                                                   "^SQRDMULH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex, D element size SVE256
// instrs SQRDMULH
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^SQRDMULH_ZZZI?_D")>;

// Multiply/multiply long, (8x8) polynomial SVE256
// instrs PMUL, PMULLB, PMULLT
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^PMUL_ZZZ_B",
                                                   "^PMULL[BT]_ZZZ_[HDQ]")>;

// Predicate counting vector SVE256
// instrs SQINCH, SQINCW, SQINCD, SQDECH, SQDECW, SQDECD, INCH, INCW, INCD, DECH, DECW, DECD, UQINCH, UQINCW, UQINCD, UQDECH, UQDECW, UQDECD
def : InstRW<[HIP12Write_5c_2V02_2V13], (instregex "^([SU]Q)?(DEC|INC)[HWD]_ZPiI")>;

// Reciprocal estimate SVE256
// instrs URECPE, URSQRTE
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, D SVE256
// instrs UADDV
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^UADDV_VPZ_D")>;

// Reduction, arithmetic, S SVE256
// instrs SADDV, UADDV
def : InstRW<[HIP12Write_7c_3V02_3V13], (instregex "^[SU]ADDV_VPZ_S")>;

// Reduction, arithmetic, H SVE256
// instrs SADDV, UADDV
def : InstRW<[HIP12Write_10c_4V02_4V13], (instregex "^[SU]ADDV_VPZ_H")>;

// Reduction, arithmetic, B SVE256
// instrs SADDV, UADDV
def : InstRW<[HIP12Write_13c_5V02_5V13], (instregex "^[SU]ADDV_VPZ_B")>;

// Reduction, arithmetic, D SVE256
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_5c_2V02_2V13], (instregex "^[SU](MAX|MIN)V_VPZ_D")>;

// Reduction, arithmetic, S SVE256
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_8c_3V02_3V13], (instregex "^[SU](MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, H SVE256
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_11c_4V02_4V13], (instregex "^[SU](MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, B SVE256
// instrs SMAXV, SMINV, UMAXV, UMINV
def : InstRW<[HIP12Write_14c_5V02_5V13], (instregex "^[SU](MAX|MIN)V_VPZ_B")>;

// Reduction, logical SVE256
// instrs ANDV, EORV, ORV
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^(AND|EOR|OR)V_VPZ_[BHSD]")>;

// Reverse, vector SVE256
// instrs REV
// Reverse, vector SVE256
// instrs REVB, REVH, REVW
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^REV_ZZ_[BHSD]",
                                                   "^REVB_ZPmZ_[HSD]",
                                                   "^REVH_ZPmZ_[SD]",
                                                   "^REVW_ZPmZ_D")>;

// Select, vector form SVE256
// instrs MOV, SEL
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^SEL_ZPZZ_[BHSD]")>;

// Table lookup SVE256
// instrs TBL
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^TBL_ZZZZ?_[BHSD]")>;

// Table lookup extension SVE256
// instrs TBX
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^TBX_ZZZ_[BHSD]")>;

// Transpose, vector form SVE256
// instrs TRN1, TRN2
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^TRN[12]_ZZZ_[BHSDQ]")>;

// Unpack and extend SVE256
// instrs SUNPKHI, SUNPKLO, UUNPKHI, UUNPKLO
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]")>;

// Zip/unzip SVE256
// instrs UZP1, UZP2, ZIP1, ZIP2
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]")>;

//===----------------------------------------------------------------------===//
// 3.24 SVE Floating-point Instructions
//===----------------------------------------------------------------------===//

// Floating point absolute value SVE256
// instrs FABS
// Floating point negative value SVE256
// instrs FNEG
def : InstRW<[HIP12Write_1c_1V02_1V13],
             (instregex "^FABS_ZPmZ_[HSD]",
                        "^FNEG_ZPmZ_[HSD]")>;

// Floating point absolute difference SVE256
// instrs FABD
// Floating point arithmetic SVE256
// instrs FADD, FSUB, FSUBR
def : InstRW<[HIP12Write_2c_1V02_1V13],
             (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ)_[HSD]",
                        "^F(ADD|SUB)_ZPZ[IZ]_[HSD]",
                        "^FABD_ZP[mZ]Z_[HSD]",
                        "^FSUBR_ZPm[IZ]_[HSD]",
                        "^FSUBR_(ZPZI|ZPZZ)_[HSD]")>;

// Floating point arithmetic pairwise SVE256
// instrs FADDP
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^FADDP_ZPmZZ_[HSD]")>;

// Floating point associative add, F16 SVE256
// instrs FADDA
def : InstRW<[HIP12Write_42c_1V02_1V13_18rc], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32 SVE256
// instrs FADDA
def : InstRW<[HIP12Write_26c_1V02_1V13_10rc], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64 SVE256
// instrs FADDA
def : InstRW<[HIP12Write_20c_1V02_1V13_12rc], (instrs FADDA_VPZ_D)>;

// Floating point compare SVE256
// instrs FACGE, FACGT, FACLE, FACLT, FCMEQ, FCMGE, FCMGT, FCMLE, FCMLT, FCMNE, FCMUO
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^FACG[ET]_PPzZZ_[HSD]",
                                                   "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]",
                                                   "^FCM(LE|LT)_PPzZ0_[HSD]",
                                                   "^FCMUO_PPzZZ_[HSD]")>;

// Floating point complex add
// instrs FCADD
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^FCADD_ZPmZ_[HSD]")>;

// Floating point complex multiply add(vectors) SVE256
// instrs FCMLA
def : InstRW<[HIP12Wr_ZFCMA, ReadDefault, HIP12Rd_ZFCMA], (instregex "^FCMLA_ZPmZZ_[HSD]")>;

// Floating point complex multiply add(indexed) SVE256
// instrs FCMLA
def : InstRW<[HIP12Wr_ZFCMA, HIP12Rd_ZFCMA], (instregex "^FCMLA_ZZZI_[HS]")>;

// Floating point convert, long or narrow SVE256
// instrs FCVT
// Floating point convert, long or narrow SVE256
// instrs FCVTLT, FCVTNT
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                                   "^FCVTLT_ZPmZ_HtoS",
                                                   "^FCVTNT_ZPmZ_StoH",
                                                   "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                                   "^FCVTLT_ZPmZ_StoD",
                                                   "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd SVE256
// instrs FCVTX, FCVTXNT
def : InstRW<[HIP12Write_4c_2V02_2V13], (instrs FCVTX_ZPmZ_DtoS, FCVTXNT_ZPmZ_DtoS)>;

// Floating point base2 log SVE256
// instrs FLOGB
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FLOGB_(ZPmZ|ZPZZ)_[HSD]")>;

// Floating point convert to integer SVE256
// instrs FCVTZS, FCVTZU
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FCVTZ[SU]_ZPmZ_HtoH",
                                                   "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)",
                                                   "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy SVE256
// instrs FCPY, FDUP, FMOV
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^FCPY_ZPmI_[HSD]", "^FDUP_ZI_[HSD]")>;

// Floating point divide, F16 SVE256
// instrs FDIV, FDIVR
def : InstRW<[HIP12Write_5c_1V02_1V13_2rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32 SVE256
// instrs FDIV, FDIVR
def : InstRW<[HIP12Write_7c_1V02_1V13_2rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64 SVE256
// instrs FDIV, FDIVR
def : InstRW<[HIP12Write_9c_1V02_1V13_2rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise SVE256
// instrs FMAXP, FMAXNMP, FMINP, FMINNMP
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max SVE256
// instrs FMAX, FMIN, FMAXNM, FMINNM
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^F(MAX|MIN)(NM)?_ZPm[IZ]_[HSD]",
                                                   "^F(MAX|MIN)(NM)?_ZPZ[IZ]_[HSD]")>;

// Floating point multiply SVE256
// instrs FMUL
// instrs FSCALE, FMULX
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^(FSCALE|FMULX)_ZPmZ_[HSD]",
                                                   "^FMULX_ZPZZ_[HSD]",
                                                   "^FMUL_(ZPm[IZ]|ZZZI?)_[HSD]",
                                                   "^FMUL_ZPZ[IZ]_[HSD]")>;

// Floating point multiply accumulate SVE256
// instrs FMLA, FMLS, FMAD, FMSB, FNMAD, FNMLA, FNMLS, FNMSB
def : InstRW<[HIP12Wr_ZFMA, ReadDefault, HIP12Rd_ZFMA],
             (instregex "^FN?ML[AS]_ZPmZZ_[HSD]",
                        "^FN?(MAD|MSB)_ZPmZZ_[HSD]")>;
def : InstRW<[HIP12Wr_ZFMA, HIP12Rd_ZFMA],
             (instregex "^FML[AS]_ZZZI_[HSD]",
                        "^FN?ML[AS]_ZPZZZ_[HSD]")>;

// Floating point multiply add/sub accumulate long SVE256
// instrs FMLALB, FMLALT, FMLSLB, FMLSLT
def : InstRW<[HIP12Wr_ZFMAL, HIP12Rd_ZFMAL], (instregex "^FML[AS]L[BT]_ZZZI?_SHH")>;

// Floating point reciprocal estimate SVE256
// instrs FRECPE, FRECPX, FRSQRTE
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FR(ECP|SQRT)E_ZZ_[HSD]",
                                                   "^FRECPX_ZPmZ_[HSD]")>;

// Floating point reciprocal step SVE256
// instrs FRECPS, FRSQRTS
def : InstRW<[HIP12Write_4c_1V02_1V13], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]")>;

// Floating point reduction, F16 SVE256
// instrs FADDV
def : InstRW<[HIP12Write_15c_5V02_5V13], (instregex "^FADDV_VPZ_H")>;

// Floating point reduction, F32 SVE256
// instrs FADDV
def : InstRW<[HIP12Write_12c_4V02_4V13], (instregex "^FADDV_VPZ_S")>;

// Floating point reduction, F64 SVE256
// instrs FADDV
def : InstRW<[HIP12Write_9c_3V02_3V13], (instregex "^FADDV_VPZ_D")>;

// Floating point reduction SVE256
// instrs FMAXNMV, FMAXV, FMINNMV, FMINV
def : InstRW<[HIP12Write_6c_2V02_2V13], (instregex "^(FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_[HSD]")>;

// Floating point round to integral SVE256
// instrs FRINTA, FRINTM, FRINTN, FRINTP, FRINTX, FRINTZ
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FRINT[AIMNPXZ]_ZPmZ_[HSD]")>;

// Floating point square root, F16 SVE256
// instrs FSQRT
def : InstRW<[HIP12Write_5c_1V02_1V13_2rc], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32 SVE256
// instrs FSQRT
def : InstRW<[HIP12Write_7c_1V02_1V13_2rc], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64 SVE256
// instrs FSQRT
def : InstRW<[HIP12Write_9c_1V02_1V13_2rc], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation SVE256
// instrs FEXPA
def : InstRW<[HIP12Write_4c_2V02_2V13], (instregex "^FEXPA_ZZ_[HSD]")>;

// Floating point trigonometric multiply add SVE256
// instrs FTMAD
def : InstRW<[HIP12Write_4c_1V02_1V13], (instregex "^FTMAD_ZZI_[HSD]")>;

// Floating point trigonometric, miscellaneous SVE256
// instrs FTSMUL
def : InstRW<[HIP12Write_3c_1V02_1V13], (instregex "^FTSMUL_ZZZ_[HSD]")>;

// Floating point trigonometric, miscellaneous SVE256
// instrs FTSSEL
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^FTSSEL_ZZZ_[HSD]")>;

//===----------------------------------------------------------------------===//
// 3.25 SVE BFlot16 (BF16) Instructions
//===----------------------------------------------------------------------===//

// Convert, F32 to BF16 SVE256
// instrs BFCVT, BFCVTNT
def : InstRW<[HIP12Write_4c_2V02_2V13], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product SVE256
// instrs BFDOT
def : InstRW<[HIP12Write_6c_2V02_2V13], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate SVE256
// instrs BFMMLA
def : InstRW<[HIP12Write_8c_4V02_4V13], (instrs BFMMLA_ZZZ_HtoS)>;

// Multiply accumulate long SVE256
// instrs BFMLALB, BFMLALT, BFMLALB, BFMLALT
def : InstRW<[HIP12Wr_ZBFMAL, HIP12Rd_ZBFMAL], (instregex "^BFMLAL[BT]_ZZZI?")>;

//===----------------------------------------------------------------------===//
// 3.26 SVE Load Instructions
//===----------------------------------------------------------------------===//

// Load vector
// instrs LDR
def : InstRW<[HIP12Write_6c_1LD], (instrs LDR_ZXI)>;

// Load predicate SVE256
// instrs LDR
def : InstRW<[HIP12Write_8c_1LD_1V02_1V13], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
// instrs LD1B, LD1H, LD1W, LD1D, LD1SB, LD1SH, LD1SW
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LD1[BHWD]_IMM$",
                                             "^LD1S?B_[HSD]_IMM$",
                                             "^LD1S?H_[SD]_IMM$",
                                             "^LD1S?W_D_IMM$" )>;

// Contiguous load, scalar + scalar
// instrs LD1B, LD1H, LD1W, LD1D, LD1SB, LD1SH, LD1SW
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LD1[BHWD]$",
                                             "^LD1S?B_[HSD]$",
                                             "^LD1S?H_[SD]$",
                                             "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm SVE256
// instrs LD1RB, LD1RH, LD1RW, LD1RD, LD1RSB, LD1RSH, LD1RSW
def : InstRW<[HIP12Write_8c_1LD_1V02_1V13], (instregex "^LD1R[BHWD]_IMM$",
                                                       "^LD1RS?B_[HSD]_IMM$",
                                                       "^LD1RS?H_[SD]_IMM$",
                                                       "^LD1RW_D_IMM$",
                                                       "^LD1RSW_IMM$")>;

// Contiguous load broadcast, scalar + imm
// instrs LD1RQB, LD1RQH, LD1RQW, LD1RQD
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
// instrs LD1RQB, LD1RQH, LD1RQW, LD1RQD
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LD1RQ_[BHWD]$")>;

// Non-temporal load, scalar + imm
// instrs LDNT1B, LDNT1H, LDNT1W, LDNT1D
// Non-temporal load, scalar + scalar
// instrs LDNT1B, LDNT1H, LDNT1W, LDNT1D
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDNT1[BHWD]_ZR[IR]$")>;

// Non-temporal gather load, vector + scalar 32-bit element size
// instrs LDNT1B, LDNT1H, LDNT1W, LDNT1SB, LDNT1SH
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDNT1[BHW]_ZZR_S$",
                                             "^LDNT1S[BH]_ZZR_S$")>;

// Non-temporal gather load, vector + scalar 64-bit element size
// instrs LDNT1B, LDNT1H, LDNT1W, LDNT1SB, LDNT1SH, LDNT1SW
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[HIP12Write_6c_1LD], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
// instrs LDFF1B, LDFF1H, LDFF1W, LDFF1D, LDFF1SB, LDFF1SH, LDFF1SW, LDFF1SD
def : InstRW<[HIP12Write_6c_1LD], (instregex "^LDFF1[BHWD]$",
                                             "^LDFF1S?B_[HSD]$",
                                             "^LDFF1S?H_[SD]$",
                                             "^LDFF1S?W_D$")>;

// Contiguous non-faulting load, scalar + imm
// instrs LDNF1B, LDNF1D, LDNF1H, LDNF1W, LDNF1SB, LDNF1SH, LDNF1SW
def : InstRW<[HIP12Write_6c_1LD],  (instregex "^LDNF1[BHWD]_IMM$",
                                              "^LDNF1S?B_[HSD]_IMM$",
                                              "^LDNF1S?H_[SD]_IMM$",
                                              "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm SVE256
// instrs LD2B, LD2D, LD2H, LD2W
def : InstRW<[HIP12Write_8c_2LD_2V02_2V13], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar SVE256
// instrs LD2B, LD2D, LD2H, LD2W
def : InstRW<[HIP12Write_8c_2LD_2V02_2V13], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm SVE256
// instrs LD3B, LD3H, LD3W
def : InstRW<[HIP12Write_10c_3LD_6V02_6V13], (instregex "^LD3[BHW]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + imm SVE256
// instrs LD3D
def : InstRW<[HIP12Write_10c_3LD_3V02_3V13], (instregex "^LD3D_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar SVE256
// instrs LD3B, LD3H, LD3W
def : InstRW<[HIP12Write_10c_3LD_6V02_6V13], (instregex "^LD3[BHW]$")>;

// Contiguous Load three structures to three vectors, scalar + scalar SVE256
// instrs LD3D
def : InstRW<[HIP12Write_10c_3LD_3V02_3V13], (instregex "^LD3D$")>;

// Contiguous Load four structures to four vectors, scalar + imm SVE256
// instrs LD4B, LD4H LD4W
def : InstRW<[HIP12Write_11c_4LD_8V02_8V13], (instregex "^LD4[BHW]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + imm SVE256
// instrs LD4D
def : InstRW<[HIP12Write_11c_4LD_4V02_4V13], (instregex "^LD4D_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar SVE256
// instrs LD4B, LD4H, LD4W
def : InstRW<[HIP12Write_11c_4LD_8V02_8V13], (instregex "^LD4[BHW]$")>;

// Contiguous Load four structures to four vectors, scalar + scalar SVE256
// instrs LD4D
def : InstRW<[HIP12Write_11c_4LD_4V02_4V13], (instregex "^LD4D$")>;

// Gather load, vector + imm, 32- bit element size SVE256
// instrs LD1B, LD1H, LD1W, LD1SB, LD1SH, LD1SW, LDFF1B, LDFF1H, LDFF1W, LDFF1SB, LDFF1SH, LDFF1SW
def : InstRW<[HIP12Write_10c_8LD], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                              "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64- bit element size SVE256
// instrs LD1B, LD1H, LD1W, LD1D, LD1SB, LD1SH, LD1SW, LDFF1B, LDFF1D, LDFF1H, LDFF1W, LDFF1SB, LDFF1SD, LDFF1SH, LDFF1SW
def : InstRW<[HIP12Write_9c_4LD],  (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                              "^GLD(FF)?1D_IMM$")>;

// Gather load, 32-bit scaled offset SVE256
// instrs LD1H, LD1SH, LDFF1H, LDFF1SH, LD1W, LDFF1W, LDFF1SW
def : InstRW<[HIP12Write_10c_8LD],
             (instregex "^GLD(FF)?1S?H_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;

// Gather load, 64-bit scaled offset SVE256
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[HIP12Write_10c_8LD],
             (instregex "^GLD(FF)?1S?[HW]_D_([SU]XTW_)?SCALED$",
                        "^GLD(FF)?1D_([SU]XTW_)?SCALED$")>;

// Gather load, 32-bit unpacked unscaled offset SVE256
// instrs LD1B, LD1SB, LDFF1B, LDFF1SB, LD1D, LDFF1D, LD1H, LD1SH, LDFF1H, LDFF1SH, LD1W, LD1SW, LDFF1W, LDFF1SW
def : InstRW<[HIP12Write_9c_4LD],
              (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                         "^GLD(FF)?1W_[SU]XTW$")>;

// Gather load, 64-bit unpacked unscaled offset SVE256
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[HIP12Write_9c_4LD],
             (instregex "^GLD(FF)?1S?[BHW]_D(_[SU]XTW)?$",
                        "^GLD(FF)?1D(_[SU]XTW)?$")>;

//===----------------------------------------------------------------------===//
// 3.27 SVE Store Instructions
//===----------------------------------------------------------------------===//

// Store from predicate reg
// instrs STR
def : InstRW<[HIP12Write_4c_1V02_1V13_1ST_1STD], (instrs STR_PXI)>;

// Store from vector reg
// instrs STR
def : InstRW<[HIP12Write_1c_1ST_1STD], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
// instrs ST1B, ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^ST1[BHWD]_IMM$",
                                                  "^ST1B_[HSD]_IMM$",
                                                  "^ST1H_[SD]_IMM$",
                                                  "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
// instrs ST1B, ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^ST1H(_[SD])?$")>;
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^ST1[BWD]$",
                                                  "^ST1B_[HSD]$",
                                                  "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm SVE256
// instrs ST2B, ST2H, ST2W, ST2D
def : InstRW<[HIP12Write_4c_2V02_2V13_2ST_2STD], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar SVE256
// instrs ST2B, ST2H, ST2W, ST2D
def : InstRW<[HIP12Write_4c_2V02_2V13_2ST_2STD], (instrs ST2H)>;
def : InstRW<[HIP12Write_4c_2V02_2V13_2ST_2STD], (instregex "^ST2[BWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm SVE256
// instrs ST3B, ST3H, ST3W, ST3D
def : InstRW<[HIP12Write_7c_6V02_6V13_6ST_6STD], (instregex "^ST3[BHWD]_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar SVE256
// instrs ST3B, ST3H, ST3W, ST3D
def : InstRW<[HIP12Write_7c_6V02_6V13_6ST_6STD], (instregex "^ST3[BHWD]$")>;

// Contiguous store four structures from four vectors, scalar + imm SVE256
// instrs ST4B, ST4H, ST4W, ST4D
def : InstRW<[HIP12Write_7c_8V02_8V13_8ST_8STD], (instregex "^ST4[BHWD]_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar SVE256
// instrs ST4B, ST4H, ST4W, ST4D
def : InstRW<[HIP12Write_7c_8V02_8V13_8ST_8STD], (instregex "^ST4[BHWD]$")>;

// Non-temporal store, scalar + imm
// instrs STNT1B, STNT1H, STNT1W, STNT1D
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non-temporal store, scalar + scalar
// instrs STNT1B, STNT1H, STNT1W, STNT1D
def : InstRW<[HIP12Write_1c_1ST_1STD], (instrs STNT1H_ZRR)>;
def : InstRW<[HIP12Write_1c_1ST_1STD], (instregex "^STNT1[BWD]_ZRR$")>;

// Scatter non-temporal store, vector + scalar 32-bit element size SVE256
// instrs STNT1B, STNT1H, STNT1W
def : InstRW<[HIP12Write_8c_8ST_8STD], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non-temporal store, vector + scalar 64-bit element size SVE256
// instrs STNT1B, STNT1H, STNT1W, STNT1D
def : InstRW<[HIP12Write_4c_4ST_4STD], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size SVE256
// instrs ST1B, ST1H, ST1W
def : InstRW<[HIP12Write_8c_8ST_8STD], (instregex "^SST1[BH]_S_IMM$",
                                                  "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size SVE256
// instrs ST1B, ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_4c_4ST_4STD],  (instregex "^SST1[BHW]_D_IMM$",
                                                   "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset SVE256
// instrs ST1H, ST1W
def : InstRW<[HIP12Write_8c_8ST_8STD],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset SVE256
// instrs ST1B, ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_8c_8ST_8STD], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                                  "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset SVE256
// instrs ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_8c_8ST_8STD], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                                  "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset SVE256
// instrs ST1B, ST1H, ST1W
def : InstRW<[HIP12Write_8c_8ST_8STD], (instregex "^SST1[BH]_S_[SU]XTW$",
                                                  "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset SVE256
// instrs ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_4c_4ST_4STD], (instregex "^SST1[HW]_D_SCALED$",
                                                  "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset SVE256
// instrs ST1B, ST1H, ST1W, ST1D
def : InstRW<[HIP12Write_4c_4ST_4STD], (instregex "^SST1[BHW]_D$",
                                                  "^SST1D$")>;

//===----------------------------------------------------------------------===//
// 3.28 SVE Miscellaneous Instructions
//===----------------------------------------------------------------------===//

// Read first fault register, unpredicated
// instrs RDFFR
def : InstRW<[HIP12Write_2c_1V02_1V13], (instrs RDFFR_P)>;

// Read first fault register, predicated
// instrs RDFFR
def : InstRW<[HIP12Write_2c_1V02_1V13], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
// instrs RDFFRS
def : InstRW<[HIP12Write_2c_1V02_1V13], (instrs RDFFRS_PPz)>;

// Set first fault register
// instrs SETFFR
def : InstRW<[HIP12Write_0c], (instrs SETFFR)>;

// Write to first fault register
// instrs WRFFR
def : InstRW<[HIP12Write_9c_1V02_1V13_1ALU25_18rc], (instrs WRFFR)>;

//===----------------------------------------------------------------------===//
// Prefetch
//===----------------------------------------------------------------------===//

// NOTE: This is not specified in the SOG.
def : InstRW<[HIP12Write_4c_1LD], (instregex "^PRF[BHWD]")>;

//===----------------------------------------------------------------------===//
// 3.29 SVE Cryptographic Instructions
//===----------------------------------------------------------------------===//

// crypto AES ops SVE256
// instrs AESD, AESE, AESMC, AESIMC
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^AES[DE]_ZZZ_B$",
                                                   "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops SVE256
// instrs BCAX, EOR3, RAX1
def : InstRW<[HIP12Write_1c_1V02_1V13], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                                   "^RAX1_ZZZ_D$")>;

// Crypto SHA3 ops SVE256
// instrs XAR
def : InstRW<[HIP12Write_2c_1V02_1V13], (instregex "^XAR_ZZZI_[BHSD]$")>;

// Crypto SM4 ops SVE256
// instrs SM4E, SM4EKEY
def : InstRW<[HIP12Write_4c_1V02_1V13], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

} // SchedModel = HIP12Model
