//=- AArch64SchedNeoverseN3.td - NeoverseN3 Scheduling Defs --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Arm Neoverse N3 processors.
//
//===----------------------------------------------------------------------===//

def NeoverseN3Model : SchedMachineModel {
    let IssueWidth            =  10; // Micro-ops dispatched at a time.
    let MicroOpBufferSize     = 160; // Entries in micro-op re-order buffer. NOTE: Copied from N2.
    let LoadLatency           =   4; // Optimistic load latency.
    let MispredictPenalty     =  10; // Extra cycles for mispredicted branch. NOTE: Copied from N2.
    let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
    let CompleteModel         =   1;

    list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
        [HasSVE2p1, HasSVEB16B16, HasPAuthLR, HasCPA, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Neoverse N3.
// Instructions are first fetched and then decoded into internal Macro-OPerations
// (MOPs). From there, the MOPs proceed through register renaming and dispatch stages.
// A MOP can be split into two Micro-OPerations (µOPs) further down the pipeline
// after the decode stage. Once dispatched, µOPs wait for their operands and issue
// out-of-order to one of thirteen issue pipelines. Each issue pipeline can accept
// one µOP per cycle.

let SchedModel = NeoverseN3Model in {

// Define the (13) issue ports.
def N3UnitB   : ProcResource<2>;  // Branch 0/1
def N3UnitS   : ProcResource<2>;  // Integer Single-Cycle 0/1
def N3UnitM0  : ProcResource<1>;  // Integer Single/Multi-Cycle 0
def N3UnitM1  : ProcResource<1>;  // Integer Single/Multi-Cycle 1
def N3UnitL01 : ProcResource<2>;  // Load/Store 0/1
def N3UnitL2  : ProcResource<1>;  // Load 2
def N3UnitD   : ProcResource<2>;  // Integer Store data 0/1
def N3UnitV0  : ProcResource<1>;  // FP/ASIMD 0
def N3UnitV1  : ProcResource<1>;  // FP/ASIMD 1

def N3UnitV : ProcResGroup<[N3UnitV0, N3UnitV1]>;
def N3UnitM : ProcResGroup<[N3UnitM0, N3UnitM1]>;
def N3UnitL : ProcResGroup<[N3UnitL01, N3UnitL2]>;
def N3UnitI : ProcResGroup<[N3UnitS, N3UnitM0, N3UnitM1]>;

//===----------------------------------------------------------------------===//

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

// NOTE: Copied from N2.
def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the Neoverse N3.

//===----------------------------------------------------------------------===//
// Define generic 0 micro-op types

def N3Write_0c : SchedWriteRes<[]> {
    let Latency = 0;
    let NumMicroOps = 0;
}

def N3Write_4c : SchedWriteRes<[]> {
    let Latency = 4;
    let NumMicroOps = 0;
}

//===----------------------------------------------------------------------===//
// Define generic 1 micro-op types

def N3Write_1c_1B   : SchedWriteRes<[N3UnitB]>   { let Latency = 1; }
def N3Write_1c_1I   : SchedWriteRes<[N3UnitI]>   { let Latency = 1; }
def N3Write_2c_1M   : SchedWriteRes<[N3UnitM]>   { let Latency = 2; }
def N3Write_2c_1M0  : SchedWriteRes<[N3UnitM0]>  { let Latency = 2; }
def N3Write_3c_1M   : SchedWriteRes<[N3UnitM]>   { let Latency = 3; }
def N3Write_1c_1M   : SchedWriteRes<[N3UnitM]>   { let Latency = 1; }
def N3Write_4c_1M   : SchedWriteRes<[N3UnitM]>   { let Latency = 4; }
def N3Write_1c_1S   : SchedWriteRes<[N3UnitS]>   { let Latency = 1; }
def N3Write_4c_1L   : SchedWriteRes<[N3UnitL]>   { let Latency = 4; }
def N3Write_2c_1V   : SchedWriteRes<[N3UnitV]>   { let Latency = 2; }
def N3Write_5c_1V0  : SchedWriteRes<[N3UnitV0]>  { let Latency = 5; }
def N3Write_7c_1V0  : SchedWriteRes<[N3UnitV0]>  { let Latency = 7; }
def N3Write_12c_1V0 : SchedWriteRes<[N3UnitV0]>  { let Latency = 12; }
def N3Write_3c_1V   : SchedWriteRes<[N3UnitV]>   { let Latency = 3; }
def N3Write_4c_1V   : SchedWriteRes<[N3UnitV]>   { let Latency = 4; }
def N3Write_3c_1V0  : SchedWriteRes<[N3UnitV0]>  { let Latency = 3; }
def N3Write_3c_1M0  : SchedWriteRes<[N3UnitM0]>  { let Latency = 3; }
def N3Write_6c_1L   : SchedWriteRes<[N3UnitL]>   { let Latency = 6; }
def N3Write_4c_1V1  : SchedWriteRes<[N3UnitV1]>  { let Latency = 4; }
def N3Write_3c_1V1  : SchedWriteRes<[N3UnitV1]>  { let Latency = 3; }
def N3Write_4c_1V0  : SchedWriteRes<[N3UnitV0]>  { let Latency = 4; }
def N3Write_2c_1V0  : SchedWriteRes<[N3UnitV0]>  { let Latency = 2; }
def N3Write_2c_1V1  : SchedWriteRes<[N3UnitV1]>  { let Latency = 2; }
def N3Write_5c_1V   : SchedWriteRes<[N3UnitV]>   { let Latency = 5; }
def N3Write_1c_1L01 : SchedWriteRes<[N3UnitL01]> { let Latency = 1; }

def N3Write_12c_1M0_12 : SchedWriteRes<[N3UnitM0]> {
    let Latency = 12;
    let ReleaseAtCycles = [12];
}

def N3Write_20c_1M0_20 : SchedWriteRes<[N3UnitM0]> {
    let Latency = 20;
    let ReleaseAtCycles = [20];
}

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

def N3Write_1c_1B_1S : SchedWriteRes<[N3UnitB, N3UnitS]> {
    let Latency = 1;
    let NumMicroOps = 2;
}

def N3Write_2c_1M_1B : SchedWriteRes<[N3UnitM, N3UnitB]> {
    let Latency = 2;
    let NumMicroOps = 2;
}

def N3Write_5c_1L_1S : SchedWriteRes<[N3UnitL, N3UnitS]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_4c_2L : SchedWriteRes<[N3UnitL, N3UnitL]> {
    let Latency = 4;
    let NumMicroOps = 2;
}

def N3Write_1c_1L01_1D : SchedWriteRes<[N3UnitL01, N3UnitD]> {
    let Latency = 1;
    let NumMicroOps = 2;
}

def N3Write_5c_1L_1I : SchedWriteRes<[N3UnitL, N3UnitI]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_6c_2L : SchedWriteRes<[N3UnitL, N3UnitL]> {
    let Latency = 6;
    let NumMicroOps = 2;
}

def N3Write_2c_1L01_1V : SchedWriteRes<[N3UnitL01, N3UnitV]> {
    let Latency = 2;
    let NumMicroOps = 2;
}

def N3Write_6c_2V1 : SchedWriteRes<[N3UnitV1, N3UnitV1]> {
    let Latency = 6;
    let NumMicroOps = 2;
}

def N3Write_4c_2V0 : SchedWriteRes<[N3UnitV0, N3UnitV0]> {
    let Latency = 4;
    let NumMicroOps = 2;
}

def N3Write_8c_2V0 : SchedWriteRes<[N3UnitV0, N3UnitV0]> {
    let Latency = 8;
    let NumMicroOps = 2;
}

def N3Write_13c_2V0 : SchedWriteRes<[N3UnitV0, N3UnitV0]> {
    let Latency = 13;
    let NumMicroOps = 2;
}

def N3Write_4c_2V : SchedWriteRes<[N3UnitV, N3UnitV]> {
    let Latency = 4;
    let NumMicroOps = 2;
}

def N3Write_2c_2V : SchedWriteRes<[N3UnitV, N3UnitV]> {
    let Latency = 2;
    let NumMicroOps = 2;
}

def N3Write_8c_1L_1V : SchedWriteRes<[N3UnitL, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 2;
}

def N3Write_2c_1V_1L01 : SchedWriteRes<[N3UnitV, N3UnitL01]> {
    let Latency = 2;
    let NumMicroOps = 2;
}

def N3Write_5c_2V0 : SchedWriteRes<[N3UnitV0, N3UnitV0]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_7c_1L_1M : SchedWriteRes<[N3UnitL, N3UnitM]> {
    let Latency = 7;
    let NumMicroOps = 2;
}

def N3Write_8c_1V_1L : SchedWriteRes<[N3UnitV, N3UnitL]> {
    let Latency = 8;
    let NumMicroOps = 2;
}

def N3Write_5c_1M0_1V : SchedWriteRes<[N3UnitM0, N3UnitV]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_5c_1V1_1V : SchedWriteRes<[N3UnitV1, N3UnitV]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_8c_1M0_1V : SchedWriteRes<[N3UnitM0, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 2;
}

def N3Write_5c_1M_1L : SchedWriteRes<[N3UnitM, N3UnitL]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

def N3Write_7c_1V_1V1 : SchedWriteRes<[N3UnitV, N3UnitV1]> {
    let Latency = 7;
    let NumMicroOps = 2;
}

def N3Write_5c_1V_1V1 : SchedWriteRes<[N3UnitV, N3UnitV1]> {
    let Latency = 5;
    let NumMicroOps = 2;
}

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

def N3Write_3c_1L01_1V_1I : SchedWriteRes<[N3UnitL01, N3UnitV, N3UnitI]> {
    let Latency = 3;
    let NumMicroOps = 3;
}

def N3Write_2c_1L01_1V_1I : SchedWriteRes<[N3UnitL01, N3UnitV, N3UnitI]> {
    let Latency = 2;
    let NumMicroOps = 3;
}

def N3Write_6c_3V : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 6;
    let NumMicroOps = 3;
}

def N3Write_4c_3V : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 4;
    let NumMicroOps = 3;
}

def N3Write_6c_3L : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL]> {
    let Latency = 6;
    let NumMicroOps = 3;
}

def N3Write_8c_2L_1V : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 3;
}

def N3Write_7c_1M_1M0_1V : SchedWriteRes<[N3UnitM, N3UnitM0, N3UnitV]> {
    let Latency = 7;
    let NumMicroOps = 3;
}

def N3Write_5c_1M_1L_1I : SchedWriteRes<[N3UnitM, N3UnitL, N3UnitI]> {
    let Latency = 5;
    let NumMicroOps = 3;
}

def N3Write_4c_1I_2L : SchedWriteRes<[N3UnitI, N3UnitL, N3UnitL]> {
    let Latency = 4;
    let NumMicroOps = 3;
}

def N3Write_1c_1L01_1D_1I : SchedWriteRes<[N3UnitL01, N3UnitD, N3UnitI]> {
    let Latency = 1;
    let NumMicroOps = 3;
}

def N3Write_2c_1L01_1I_1V : SchedWriteRes<[N3UnitL01, N3UnitI, N3UnitV]> {
    let Latency = 2;
    let NumMicroOps = 3;
}

//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types

def N3Write_8c_2V_2V1 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV1, N3UnitV1]> {
    let Latency = 8;
    let NumMicroOps = 4;
}

def N3Write_6c_2I_2L : SchedWriteRes<[N3UnitI, N3UnitI, N3UnitL, N3UnitL]> {
    let Latency = 6;
    let NumMicroOps = 4;
}

def N3Write_6c_4V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 6;
    let NumMicroOps = 4;
}

def N3Write_8c_4V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 8;
    let NumMicroOps = 4;
}

def N3Write_10c_4V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 10;
    let NumMicroOps = 4;
}

def N3Write_6c_4V : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 6;
    let NumMicroOps = 4;
}

def N3Write_7c_4L : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL, N3UnitL]> {
    let Latency = 7;
    let NumMicroOps = 4;
}

def N3Write_2c_2L01_2V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitV, N3UnitV]> {
    let Latency = 2;
    let NumMicroOps = 4;
}

def N3Write_4c_2V_2L01 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitL01, N3UnitL01]> {
    let Latency = 4;
    let NumMicroOps = 4;
}

def N3Write_2c_2V_2L01 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitL01, N3UnitL01]> {
    let Latency = 2;
    let NumMicroOps = 4;
}

def N3Write_8c_4V : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 4;
}

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

def N3Write_4c_3V_3L01 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV,
                                        N3UnitL01, N3UnitL01, N3UnitL01]> {
    let Latency = 4;
    let NumMicroOps = 6;
}

def N3Write_2c_3V_3L01 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV,
                                        N3UnitL01, N3UnitL01, N3UnitL01]> {
    let Latency = 2;
    let NumMicroOps = 6;
}

def N3Write_4c_3L01_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                        N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 4;
    let NumMicroOps = 6;
}

def N3Write_3c_3L01_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                        N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 3;
    let NumMicroOps = 6;
}

def N3Write_6c_3L01_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                        N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 6;
    let NumMicroOps = 6;
}

def N3Write_8c_3L_3V : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL,
                                      N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 6;
}

def N3Write_10c_3L_3V : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL,
                                       N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 10;
    let NumMicroOps = 6;
}

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

def N3Write_8c_4L_3V : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL, N3UnitL,
                                      N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 7;
}

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types

def N3Write_12c_8V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0,
                                     N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 12;
    let NumMicroOps = 8;
}

def N3Write_4c_4V_4L01 : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV, N3UnitV,
                                        N3UnitL01, N3UnitL01, N3UnitL01, N3UnitL01]> {
    let Latency = 4;
    let NumMicroOps = 8;
}

def N3Write_8c_8V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0,
                                    N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 8;
    let NumMicroOps = 8;
}

def N3Write_16c_8V : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV, N3UnitV,
                                    N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 16;
    let NumMicroOps = 8;
}

def N3Write_3c_4L01_4V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01, N3UnitL01,
                                        N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 3;
    let NumMicroOps = 8;
}

def N3Write_8c_4L_4V : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL, N3UnitL,
                                      N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 8;
    let NumMicroOps = 8;
}

//===----------------------------------------------------------------------===//
// Define generic 9 micro-op types

def N3Write_10c_6V_3L : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV,
                                       N3UnitV, N3UnitV, N3UnitV,
                                       N3UnitL, N3UnitL, N3UnitL]> {
    let Latency = 10;
    let NumMicroOps = 9;
}

def N3Write_4c_3L01_3I_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                           N3UnitI, N3UnitI, N3UnitI,
                                           N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 4;
    let NumMicroOps = 9;
}

def N3Write_3c_3L01_3I_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                           N3UnitI, N3UnitI, N3UnitI,
                                           N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 3;
    let NumMicroOps = 9;
}

def N3Write_6c_3L01_3I_3V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01,
                                           N3UnitI, N3UnitI, N3UnitI,
                                           N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 6;
    let NumMicroOps = 9;
}

def N3Write_12c_5V_4L : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV, N3UnitV, N3UnitV,
                                       N3UnitL, N3UnitL, N3UnitL, N3UnitL]> {
    let Latency = 12;
    let NumMicroOps = 9;
}

def N3Write_9c_3V_3L_3I : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV,
                                         N3UnitL, N3UnitL, N3UnitL,
                                         N3UnitI, N3UnitI, N3UnitI]> {
    let Latency = 9;
    let NumMicroOps = 9;
}

//===----------------------------------------------------------------------===//
// Define generic 12 micro-op types

def N3Write_3c_4L01_4I_4V : SchedWriteRes<[N3UnitL01, N3UnitL01, N3UnitL01, N3UnitL01,
                                           N3UnitI, N3UnitI, N3UnitI, N3UnitI,
                                           N3UnitV, N3UnitV, N3UnitV, N3UnitV]> {
    let Latency = 3;
    let NumMicroOps = 12;
}

def N3Write_9c_4L_4V_4I : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL, N3UnitL,
                                         N3UnitV, N3UnitV, N3UnitV, N3UnitV,
                                         N3UnitI, N3UnitI, N3UnitI, N3UnitI]> {
    let Latency = 9;
    let NumMicroOps = 12;
}

//===----------------------------------------------------------------------===//
// Define generic 14 micro-op types

def N3Write_13c_4L_5V_5I : SchedWriteRes<[N3UnitL, N3UnitL, N3UnitL, N3UnitL,
                                          N3UnitV, N3UnitV, N3UnitV, N3UnitV, N3UnitV,
                                          N3UnitI, N3UnitI, N3UnitI, N3UnitI, N3UnitI]> {
    let Latency = 13;
    let NumMicroOps = 14;
}

//===----------------------------------------------------------------------===//
// Define generic 15 micro-op types

def N3Write_11c_6V_3L_6I : SchedWriteRes<[N3UnitV, N3UnitV, N3UnitV,
                                          N3UnitV, N3UnitV, N3UnitV,
                                          N3UnitL, N3UnitL, N3UnitL,
                                          N3UnitI, N3UnitI, N3UnitI,
                                          N3UnitI, N3UnitI, N3UnitI]> {
    let Latency = 11;
    let NumMicroOps = 15;
}

//===----------------------------------------------------------------------===//
// Define generic 16 micro-op types

def N3Write_16c_16V0 : SchedWriteRes<[N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0,
                                      N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0,
                                      N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0,
                                      N3UnitV0, N3UnitV0, N3UnitV0, N3UnitV0]> {
    let Latency = 16;
    let NumMicroOps = 16;
}

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// Branch Instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr, N3Write_1c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, N3Write_1c_1B>;

// Branch and link, immed
// Branch and link, register
def : InstRW<[N3Write_1c_1B_1S], (instrs BL, BLR)>;

// Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------

// ALU, basic
// ALU, basic, flagset
// Arithmetic, immediate to logical address tag
// Conditional compare
// Conditional select
def : SchedAlias<WriteI, N3Write_1c_1I>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, N3Write_2c_1M>;

def N3WriteISReg : SchedWriteVariant<[
                     SchedVar<IsCheapLSL,  [N3Write_1c_1I]>,
                     SchedVar<NoSchedPred, [N3Write_2c_1M]>]>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def : SchedAlias<WriteISReg, N3WriteISReg>;

// Convert floating-point condition flags
def : SchedAlias<WriteSys, N3Write_1c_1I>;

// Flag manipulation instructions
def : InstRW<[N3Write_1c_1I], (instrs SETF8, SETF16, RMIF, CFINV)>;

// Insert Random Tags
def : InstRW<[N3Write_2c_1M0], (instrs IRG, IRGstack)>;

// Insert Tag Mask
// Subtract Pointer
// Subtract Pointer, flagset
def : InstRW<[N3Write_1c_1I], (instrs GMI, SUBP, SUBPS)>;

// Logical, shift, no flagset
def : InstRW<[N3Write_1c_1I],
             (instregex "^(AND|BIC|EON|EOR|ORN|ORR)[WX]rs$")>;

// Logical, shift, flagset
def : InstRW<[N3Write_2c_1M], (instregex "^(AND|BIC)S[WX]rs$")>;

// Divide and Multiply Instructions
// -----------------------------------------------------------------------------

// Integer divides are performed using an iterative algorithm and block any
// subsequent divide operations until complete.

// Divide, W-form
def : SchedAlias<WriteID32, N3Write_12c_1M0_12>;

// Divide, X-form
def : SchedAlias<WriteID64, N3Write_20c_1M0_20>;

def N3Wr_IM   : SchedWriteRes<[N3UnitM]>  { let Latency = 2; }
def N3Wr_IMA  : SchedWriteRes<[N3UnitM0]> { let Latency = 2; }
def N3Wr_IMUL : SchedWriteVariant<[
                  SchedVar<IsReg3ZeroPred, [N3Wr_IM]>,
                  SchedVar<NoSchedPred,    [N3Wr_IMA]>]>;
def N3Rd_IMA  : SchedReadAdvance<1, [N3Wr_IMA]>;

def : SchedAlias<WriteIM32, N3Write_2c_1M0>;
def : SchedAlias<WriteIM64, N3Write_2c_1M0>;

// NOTE: Modified from V2

// Multiply
// Multiply accumulate, W-form
// Multiply accumulate, X-form
// Multiply accumulate long
// Multiply long
def : InstRW<[N3Wr_IMUL, ReadIM, ReadIM, N3Rd_IMA],
             (instregex "^M(ADD|SUB)[WX]rrr$",
                        "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[N3Write_3c_1M], (instrs SMULHrr, UMULHrr)>;

// Pointer Authentication Instructions
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
def : InstRW<[N3Write_1c_1M], (instrs AUTDA, AUTDB, AUTDZA, AUTDZB,
                                      AUTIA, AUTIB, AUTIA1716, AUTIB1716,
                                      AUTIASP, AUTIBSP, AUTIAZ, AUTIBZ, AUTIZA,
                                      AUTIZB)>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[N3Write_2c_1M_1B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA,
                                         BRAAZ, BRAB, BRABZ, RETAA, RETAB,
                                         ERETAA, ERETAB)>;

// Compute pointer authentication code for data address
def : InstRW<[N3Write_4c_1M], (instrs PACDA, PACDB, PACDZA, PACDZB)>;

// Compute pointer authentication code, using generic key
def : InstRW<[N3Write_4c_1M], (instrs PACGA)>;

// Compute pointer authentication code for instruction address
def : InstRW<[N3Write_4c_1M], (instrs PACIA, PACIB, PACIA1716, PACIB1716,
                                      PACIASP, PACIBSP, PACIAZ, PACIBZ, PACIZA,
                                      PACIZB)>;

// Load register, with pointer authentication
def : InstRW<[N3Write_5c_1M_1L], (instregex "^LDRA[AB]indexed")>;
def : InstRW<[N3Write_5c_1M_1L_1I], (instregex "^LDRA[AB]writeback")>;

// Strip pointer authentication code
def : InstRW<[N3Write_1c_1M], (instrs XPACD, XPACI, XPACLRI)>;

// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Address generation
def : InstRW<[N3Write_1c_1S], (instrs ADR, ADRP)>;

// Bitfield extract, one, two regs
def : SchedAlias<WriteExtr, N3Write_1c_1I>;

// Bitfield move, basic
// Bitfield move, insert
// Variable shift
def : SchedAlias<WriteIS, N3Write_1c_1I>;

// Count leading
// Reverse bits/bytes
// Covered by WriteI

// Move immed
def : SchedAlias<WriteImm, N3Write_1c_1I>;

// Load instructions
// -----------------------------------------------------------------------------

// Load register, literal
def : InstRW<[N3Write_5c_1L_1S], (instrs LDRWl, LDRXl, LDRSWl, PRFMl)>;

// Load register, unscaled immed
// Load register, immed post-index
// Load register, immed pre-index
// Load register, unsigned immed
// Load register, immed unprivileged
def : SchedAlias<WriteAdr, N3Write_1c_1I>;
def : SchedAlias<WriteLD, N3Write_4c_1L>;

// Load register, register offset, basic
// Load register, register offset, scale by 4/8
// Load register, register offset, scale by 2
// Load register, register offset, extend
// Load register, register offset, extend, scale by 4/8
// Load register, register offset, extend, scale by 2
def : SchedAlias<WriteLDIdx, N3Write_4c_1L>;

def : SchedAlias<WriteLDHi, N3Write_4c>;

// Load pair, signed immed offset, normal, W-form
def : InstRW<[WriteLD, WriteLDHi], (instrs LDPWi, LDNPWi)>;

// Load pair, signed immed offset, normal, X-form
def : InstRW<[N3Write_4c_2L, WriteLDHi], (instrs LDPXi, LDNPXi)>;

// Load pair, signed immed offset, signed words
def : InstRW<[N3Write_4c_1I_2L, WriteLDHi], (instrs LDPSWi)>;

// Load pair, immed post-index or immed pre-index, normal, W-form
def : InstRW<[WriteAdr, WriteLD, WriteLDHi], (instrs LDPWpost, LDPWpre)>;

// Load pair, immed post-index or immed pre-index, normal, X-form
def : InstRW<[WriteAdr, N3Write_4c_2L, WriteLDHi], (instrs LDPXpost, LDPXpre)>;

// Load pair, immed post-index or immed pre-index, signed words
def : InstRW<[WriteAdr, N3Write_4c_1I_2L, WriteLDHi], (instrs LDPSWpost, LDPSWpre)>;

// Store instructions
// -----------------------------------------------------------------------------

// Store register, unscaled immed
// Store register, immed unprivileged
// Store register, unsigned immed
// Store register, immed post-index
// Store register, immed pre-index
def : SchedAlias<WriteST, N3Write_1c_1L01_1D>;

// Store register, register offset, basic
// Store register, register offset, scaled by 4/8
// Store register, register offset, scaled by 2
// Store register, register offset, extend
// Store register, register offset, extend, scale by 4/8
// Store register, register offset, extend, scale by 2
def : SchedAlias<WriteSTIdx, N3Write_1c_1L01_1D>;

// Store pair, immed offset
// Store pair, immed post-index
// Store pair, immed pre-index
def : SchedAlias<WriteSTP, N3Write_1c_1L01_1D>;

// Tag Load instructions
// -----------------------------------------------------------------------------

// Load allocation tag
def : InstRW<[N3Write_5c_1L_1I], (instrs LDG)>;

// Load multiple allocation tags
def : InstRW<[N3Write_4c_1L], (instrs LDGM)>;

// Tag store instructions
// -----------------------------------------------------------------------------

// Store allocation tags to one or two granules, post-index
// Store allocation tags to one or two granules, pre-index
// Store allocation tag to one or two granules, zeroing, post-index
// Store allocation Tag to one or two granules, zeroing, pre-index
def : InstRW<[N3Write_1c_1L01_1D_1I], (instregex "^STZ?2?G(Post|Pre)Index$")>;

// Store allocation tags to one or two granules, signed offset
// Store allocation tag to two granules, zeroing, signed offset
def : InstRW<[N3Write_1c_1L01_1D], (instregex "^STZ?2?Gi$")>;

// Store allocation tag and reg pair to memory, post-Index
// Store allocation tag and reg pair to memory, pre-Index
def : InstRW<[N3Write_1c_1L01_1D_1I], (instrs STGPpost, STGPpre)>;

// Store allocation tag and reg pair to memory, signed offset
// Store multiple allocation tags
// Store multiple allocation tags, zeroing
def : InstRW<[N3Write_1c_1L01_1D], (instrs STGPi, STGM, STZGM)>;

// FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF, N3Write_2c_1V>;

// FP compare
def : SchedAlias<WriteFCmp, N3Write_2c_1V>;

// FP divide and square root operations are now performed using
// a fully pipelined data path.

// FP divide, H-form
// FP square root, H-form
def : InstRW<[N3Write_5c_1V0], (instrs FDIVHrr, FSQRTHr)>;

// FP divide, S-form
// FP square root, S-form
def : SchedAlias<WriteFDiv , N3Write_7c_1V0>;

// FP divide, D-form
// FP square root, D-form
def : InstRW<[N3Write_12c_1V0], (instrs FDIVDrr, FSQRTDr)>;

// FP multiply
def : SchedAlias<WriteFMul, N3Write_3c_1V>;

// FP multiply accumulate
def : InstRW<[N3Write_4c_1V], (instregex "^(FMADD|FMSUB|FNMADD|FNMSUB)[DHS]rrr$")>;

// FP round to integral
def : InstRW<[N3Write_3c_1V0], (instregex "^FRINT([AIMNPXZ]|32X|64X|32Z|64Z)[DHS]r$")>;

// FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from gen to vec reg
def : InstRW<[N3Write_3c_1M0], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
// FP convert, Javascript from vec to gen reg
// FP convert, from vec to vec reg
def : SchedAlias<WriteFCvt, N3Write_3c_1V0>;

// FP move, immed
def : SchedAlias<WriteFImm, N3Write_2c_1V>;

// FP move, register
def : InstRW<[N3Write_2c_1V], (instrs FMOVHr, FMOVSr, FMOVDr)>;

// FP transfer, from gen to low half of vec reg
def : InstRW<[N3Write_3c_1M0], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
def : InstRW<[N3Write_5c_1M0_1V], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, N3Write_3c_1V>;

// FP load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal, S/D/Q forms
// Load vector reg, unscaled immed
// Load vector reg, unsigned immed
def : InstRW<[N3Write_6c_1L], (instregex "^LDR[SDQ]l$",
                                         "^LDUR[BHSDQ]i$",
                                         "^LDR[BHSDQ]ui$")>;
// Load vector reg, immed post-index
// Load vector reg, immed pre-index
def : InstRW<[WriteAdr, N3Write_6c_1L], (instregex "^LDR[BHSDQ](post|pre)$")>;

// Load vector reg, register offset, basic
// Load vector reg, register offset, scale, S/D-form
// Load vector reg, register offset, scale, H/Q-form
// Load vector reg, register offset, extend
// Load vector reg, register offset, extend, scale, S/D-form
// Load vector reg, register offset, extend, scale, H/Q-form
def : InstRW<[N3Write_6c_1L], (instregex "^LDR[BHSDQ]ro[WX]$")>;

// Load vector pair, immed offset, S/D-form
def : InstRW<[N3Write_6c_1L, WriteLDHi], (instregex "^LDN?P[SD]i$")>;

// Load vector pair, immed offset, Q-form
def : InstRW<[N3Write_6c_2L, WriteLDHi], (instrs LDPQi, LDNPQi)>;

// Load vector pair, immed post-index, S/D-form
// Load vector pair, immed pre-index, S/D-form
// Load vector pair, immed post-index, Q-form
// Load vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, N3Write_6c_2I_2L, WriteLDHi], (instregex "^LDP[SDQ](post|pre)$")>;

// FP store instructions
// -----------------------------------------------------------------------------

// Store vector reg, unscaled immed, B/H/S/D-form
// Store vector reg, unscaled immed, Q-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index, B/H/S/D-form
// Store vector reg, immed post-index, Q-form
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V_1I], (instregex "^STR[BHSDQ]post$")>;

// Store vector reg, immed pre-index, B/H/S/D-form
def : InstRW<[WriteAdr, N3Write_3c_1L01_1V_1I], (instregex "^STR[BHSD]pre$")>;

// Store vector reg, immed pre-index, Q-form
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V_1I], (instrs STRQpre)>;

// Store vector reg, unsigned immed, B/H/S/D-form
// Store vector reg, unsigned immed, Q-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STR[BHSDQ]ui$")>;

// Store vector reg, register offset, basic, B/H/S/D-form
// Store vector reg, register offset, scale, H-form
// Store vector reg, register offset, scale, S/D-form
// Store vector reg, register offset, extend, B/H/S/D-form
// Store vector reg, register offset, extend, scale, H-form
// Store vector reg, register offset, extend, scale, S/D-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STR[BHSD]ro[WX]$")>;

def N3WriteSTRQro : SchedWriteVariant<[
  SchedVar<ScaledIdxPred, [N3Write_2c_1L01_1V_1I]>,
  SchedVar<NoSchedPred,   [N3Write_2c_1L01_1V]>]>;

// Store vector reg, register offset, basic, Q-form
// Store vector reg, register offset, scale, Q-form
// Store vector reg, register offset, extend, Q-form
// Store vector reg, register offset, extend, scale, Q-form
def : InstRW<[N3WriteSTRQro], (instregex "^STRQro[WX]$")>;

// Store vector pair, immed offset, S-form
// Store vector pair, immed offset, D-form
// Store vector pair, immed offset, Q-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STN?P[SDQ]i$")>;

// Store vector pair, immed post-index, S-form
// Store vector pair, immed post-index, D-form
// Store vector pair, immed post-index, Q-form
// Store vector pair, immed pre-index, S-form
// Store vector pair, immed pre-index, D-form
// Store vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V_1I], (instregex "^STP[SDQ](post|pre)$")>;

// ASIMD integer instructions
// -----------------------------------------------------------------------------

// ASIMD absolute diff
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD arith, pair-wise
// ASIMD compare
// ASIMD logical
// ASIMD max/min, basic and pair-wise
def : SchedAlias<WriteVd, N3Write_2c_1V>;
def : SchedAlias<WriteVq, N3Write_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
// ASIMD pairwise add and accumulate long
// ASIMD shift accumulate
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]ABAL?v",
                                          "^[SU]ADALPv",
                                          "^[SU]R?SRAv")>;

// ASIMD arith, reduce, 4H/4S
def : InstRW<[N3Write_3c_1V1], (instregex "^[SU]?ADDL?Vv4i(16|32)v$")>;

// ASIMD arith, reduce, 8B/8H
def : InstRW<[N3Write_5c_1V1_1V], (instregex "^[SU]?ADDL?Vv8i(8|16)v$")>;

// ASIMD arith, reduce, 16B
def : InstRW<[N3Write_6c_2V1], (instregex "^[SU]?ADDL?Vv16i8v$")>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def : InstRW<[N3Write_3c_1V], (instregex "^([SU]|SU|US)DOT(lane)?(v8|v16)i8$")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[N3Write_3c_1V], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD max/min, reduce, 4H/4S
def : InstRW<[N3Write_3c_1V1], (instregex "^[SU](MAX|MIN)Vv4i(16|32)v$")>;

// ASIMD max/min, reduce, 8B/8H
def : InstRW<[N3Write_5c_1V1_1V], (instregex "^[SU](MAX|MIN)Vv8i(8|16)v$")>;

// ASIMD max/min, reduce, 16B
def : InstRW<[N3Write_6c_2V1], (instregex "[SU](MAX|MIN)Vv16i8v$")>;

// ASIMD multiply
def : InstRW<[N3Write_4c_1V0], (instregex "^MULv", "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate
def : InstRW<[N3Write_4c_1V0], (instregex "^MLAv", "^MLSv")>;

// ASIMD multiply accumulate high
def : InstRW<[N3Write_4c_1V0], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply accumulate long
def : InstRW<[N3Write_4c_1V0], (instregex "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate saturating long
def : InstRW<[N3Write_4c_1V0], (instregex "^SQDMLALv", "^SQDMLSLv")>;

// ASIMD multiply/multiply long (8x8) polynomial, D-form
// ASIMD multiply/multiply long (8x8) polynomial, Q-form
def : InstRW<[N3Write_2c_1V0], (instregex "^PMULL?(v8i8|v16i8)$")>;

// ASIMD multiply long
def : InstRW<[N3Write_4c_1V0], (instregex "^[SU]MULLv", "^SQDMULLv")>;

// ASIMD shift by immed, basic
def : InstRW<[N3Write_2c_1V1], (instregex "^SHLv", "^SHLLv", "^SHRNv",
                                          "^SSHLLv", "^SSHRv", "^USHLLv",
                                          "^USHRv")>;

// ASIMD shift by immed and insert, basic
def : InstRW<[N3Write_2c_1V1], (instregex "^SLIv", "^SRIv")>;

// ASIMD shift by immed, complex
def : InstRW<[N3Write_4c_1V1],
             (instregex "^RSHRNv", "^SQRSHRNv", "^SQRSHRUNv",
                        "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHRNv", "^SQSHRUNv", "^SRSHRv", "^UQRSHRNv",
                        "^UQSHRNv", "^URSHRv")>;

// ASIMD shift by register, basic
def : InstRW<[N3Write_2c_1V1], (instregex "^[SU]SHLv")>;

// ASIMD shift by register, complex
def : InstRW<[N3Write_4c_1V1],
             (instregex "^[SU]RSHLv", "^[SU]QRSHLv",
                        "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)$")>;

// ASIMD floating-point instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP max/min, normal
// ASIMD FP negate
// Covered by WriteV[dq]

// ASIMD FP complex add
def : InstRW<[N3Write_3c_1V], (instregex "^FCADDv")>;

// ASIMD FP complex multiply add
def : InstRW<[N3Write_4c_1V], (instregex "^FCMLAv")>;

// ASIMD FP convert, long (F16 to F32)
def : InstRW<[N3Write_4c_2V0], (instregex "^FCVTL(v4|v8)i16")>;

// ASIMD FP convert, long (F32 to F64)
def : InstRW<[N3Write_3c_1V0], (instregex "^FCVTL(v2|v4)i32")>;

// ASIMD FP convert, narrow (F32 to F16)
def : InstRW<[N3Write_4c_2V0], (instregex "^FCVTN(v4|v8)i16")>;

// ASIMD FP convert, narrow (F64 to F32)
def : InstRW<[N3Write_3c_1V0], (instregex "^FCVTN(v2|v4)i32",
                                          "^FCVTXN(v2|v4)f32")>;

// ASIMD FP convert, other, D-form F32 and Q-form F64
def : InstRW<[N3Write_3c_1V0], (instregex "^[FSU]CVT[AMNPZ][SU]v2f(32|64)$",
                                          "^[SU]CVTFv2f(32|64)$")>;

// ASIMD FP convert, other, D-form F16 and Q-form F32
def : InstRW<[N3Write_4c_2V0], (instregex "^[FSU]CVT[AMNPZ][SU]v4f(16|32)$",
                                          "^[SU]CVTFv4f(16|32)$")>;

// ASIMD FP convert, other, Q-form F16
def : InstRW<[N3Write_6c_4V0], (instregex "^[FSU]CVT[AMNPZ][SU]v8f16$",
                                          "^[SU]CVTFv8f16$")>;

// ASIMD FP divide and square root operations are now performed using
// a fully pipelined data path.

// ASIMD FP divide, D-form, F16
def : InstRW<[N3Write_8c_4V0], (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
def : InstRW<[N3Write_8c_2V0], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[N3Write_12c_8V0], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[N3Write_10c_4V0], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[N3Write_13c_2V0], (instrs FDIVv2f64)>;

// ASIMD FP arith, max/min, pairwise
def : InstRW<[N3Write_3c_1V], (instregex "^FADDPv", "^FMAXPv", "^FMAXNMPv",
                                         "^FMINPv", "^FMINNMPv")>;

// ASIMD FP max/min, reduce, F32 and D-form F16
def : InstRW<[N3Write_4c_2V], (instregex "^(FMAX|FMIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
def : InstRW<[N3Write_6c_3V], (instregex "^(FMAX|FMIN)(NM)?Vv8i16v$")>;

// ASIMD FP multiply
def : InstRW<[N3Write_3c_1V], (instregex "^FMULv", "^FMULXv")>;

// ASIMD FP multiply accumulate
def : InstRW<[N3Write_4c_1V], (instregex "^FMLAv", "^FMLSv")>;

// ASIMD FP multiply accumulate long
def : InstRW<[N3Write_4c_1V], (instregex "^FMLALv", "^FMLSLv")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[N3Write_3c_1V0],
             (instregex "^FRINT[AIMNPXZ]v2f(32|64)$",
                        "^FRINT(32|64)[XZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[N3Write_4c_2V0],
             (instregex "^FRINT[AIMNPXZ]v4f(16|32)$",
                        "^FRINT(32|64)[XZ]v4f32$")>;

// ASIMD FP round, Q-form F16
def : InstRW<[N3Write_6c_4V0], (instregex "^FRINT[AIMNPXZ]v8f16$")>;

// ASIMD FP square root, D-form, F16
def : InstRW<[N3Write_8c_4V0], (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
def : InstRW<[N3Write_8c_2V0], (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[N3Write_12c_8V0], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
def : InstRW<[N3Write_10c_4V0], (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[N3Write_13c_2V0], (instrs FSQRTv2f64)>;

// ASIMD BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
def : InstRW<[N3Write_4c_2V0], (instrs BFCVTN, BFCVTN2)>;

// ASIMD dot product
def : InstRW<[N3Write_4c_1V], (instrs BFDOTv4bf16, BFDOTv8bf16)>;

// ASIMD matrix multiply accumulate
def : InstRW<[N3Write_5c_1V], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
def : InstRW<[N3Write_4c_1V], (instrs BFMLALB, BFMLALBIdx, BFMLALT, BFMLALTIdx)>;

// Scalar convert, F32 to BF16
def : InstRW<[N3Write_3c_1V0], (instrs BFCVT)>;

// ASIMD miscellaneous instructions
// -----------------------------------------------------------------------------

// ASIMD bit reverse
// ASIMD bitwise insert
// ASIMD count
// ASIMD duplicate, element
// ASIMD extract
// ASIMD extract narrow
// ASIMD insert, element to element
// ASIMD move, FP immed
// ASIMD move, integer immed
// ASIMD reverse
// ASIMD table lookup, 1 or 2 table regs
// ASIMD table lookup extension, 1 table reg
// ASIMD transpose
// ASIMD unzip/zip
// Covered by WriteV[dq]

// ASIMD duplicate, gen reg
def : InstRW<[N3Write_3c_1M0], (instregex "^DUPv.+gpr")>;

// ASIMD extract narrow, saturating
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form U32
def : InstRW<[N3Write_3c_1V0], (instrs URECPEv2i32, URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, Q-form U32
def : InstRW<[N3Write_4c_2V0], (instrs URECPEv4i32, URSQRTEv4i32)>;

// ASIMD reciprocal and square root estimate, D-form F32 and scalar forms
def : InstRW<[N3Write_3c_1V0], (instrs FRECPEv1f16, FRECPEv1i32,
                                       FRECPEv1i64, FRECPEv2f32,
                                       FRSQRTEv1f16, FRSQRTEv1i32,
                                       FRSQRTEv1i64, FRSQRTEv2f32)>;

// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
def : InstRW<[N3Write_4c_2V0], (instrs FRECPEv4f16, FRECPEv4f32,
                                       FRSQRTEv4f16, FRSQRTEv4f32)>;

// ASIMD reciprocal and square root estimate, Q-form F16
def : InstRW<[N3Write_6c_4V0], (instrs FRECPEv8f16, FRSQRTEv8f16)>;

// ASIMD reciprocal exponent
def : InstRW<[N3Write_3c_1V0], (instregex "^FRECPXv")>;

// ASIMD reciprocal step
def : InstRW<[N3Write_4c_1V], (instregex "^FRECPSv", "^FRSQRTSv")>;

// ASIMD table lookup, 3 table regs
def : InstRW<[N3Write_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// ASIMD table lookup, 4 table regs
def : InstRW<[N3Write_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 2 table reg
def : InstRW<[N3Write_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup extension, 3 table reg
def : InstRW<[N3Write_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
def : InstRW<[N3Write_6c_4V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, element to gen reg
def : InstRW<[N3Write_2c_2V], (instregex "^SMOVvi(((8|16)to(32|64))|32to64)$",
                                         "^UMOVvi(8|16|32|64)$")>;

// ASIMD transfer, gen reg to element
def : InstRW<[N3Write_5c_1M0_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;

// ASIMD load instructions
// -----------------------------------------------------------------------------

// ASIMD load, 1 element, multiple, 1 reg, D-form
def : InstRW<[N3Write_6c_1L],
             (instregex "^LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_1L],
             (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[N3Write_6c_1L], (instregex "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_1L],
             (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
def : InstRW<[N3Write_6c_2L], (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_2L],
             (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[N3Write_6c_2L], (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_2L],
             (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
def : InstRW<[N3Write_6c_3L], (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_3L],
             (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[N3Write_6c_3L], (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_3L],
             (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
def : InstRW<[N3Write_7c_4L], (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_7c_4L],
             (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[N3Write_7c_4L], (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_7c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[N3Write_8c_1L_1V], (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_8c_1L_1V], (instregex "LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
def : InstRW<[N3Write_6c_1L], (instregex "LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_1L], (instregex "LD1Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[N3Write_6c_1L], (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_1L], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
def : InstRW<[N3Write_8c_1L_1V], (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N3Write_8c_1L_1V], (instregex "LD2Twov(8b|4h|2s)_POST$")>;

// ASIMD load, 2 element, multiple, Q-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[N3Write_8c_2L_1V], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_8c_2L_1V], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[N3Write_8c_1L_1V], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_8c_1L_1V], (instregex "LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
def : InstRW<[N3Write_6c_2L], (instregex "LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_2L],  (instregex "LD2Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[N3Write_6c_2L], (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_2L], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
def : InstRW<[N3Write_8c_3L_3V], (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N3Write_8c_3L_3V], (instregex "LD3Threev(8b|4h|2s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, B/H/S
def : InstRW<[N3Write_10c_3L_3V], (instregex "LD3Threev(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N3Write_10c_3L_3V], (instregex "LD3Threev(16b|8h|4s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[N3Write_10c_3L_3V], (instregex "LD3Threev(2d)$")>;
def : InstRW<[WriteAdr, N3Write_10c_3L_3V], (instregex "LD3Threev(2d)_POST$")>;

// ASIMD load, 3 element, one lane, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[N3Write_8c_3L_3V], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_8c_3L_3V], (instregex "LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
def : InstRW<[N3Write_6c_3L], (instregex "LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_3L], (instregex "LD3Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[N3Write_6c_3L], (instregex "LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_6c_3L], (instregex "LD3Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
def : InstRW<[N3Write_8c_4L_4V], (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N3Write_8c_4L_4V], (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[N3Write_8c_4L_4V], (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_8c_4L_4V], (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[N3Write_8c_4L_4V], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_8c_4L_4V], (instregex "LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, B/H/S
def : InstRW<[N3Write_8c_4L_3V], (instregex "LD4Rv(8b|4h|2s)$",
                                            "LD4Rv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N3Write_8c_4L_3V], (instregex "LD4Rv(8b|4h|2s)_POST$",
                                                      "LD4Rv(16b|8h|4s)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, D
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[N3Write_8c_4L_4V], (instregex "LD4Rv1d$", "LD4Rv2d$")>;
def : InstRW<[WriteAdr, N3Write_8c_4L_4V], (instregex "LD4Rv1d_POST$", "LD4Rv2d_POST$")>;

// ASIMD store instructions
// -----------------------------------------------------------------------------

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[N3Write_2c_1L01_1V], (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
def : InstRW<[N3Write_2c_2L01_2V], (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2L01_2V], (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[N3Write_2c_2L01_2V], (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2L01_2V], (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[N3Write_2c_2L01_2V], (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2L01_2V], (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[N3Write_2c_2L01_2V], (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2L01_2V], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[N3Write_2c_1L01_1V], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1L01_1V], (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[N3Write_2c_1V_1L01], (instregex "ST2Twov(8b|4h|2s)$",
                                              "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1V_1L01], (instregex "ST2Twov(8b|4h|2s)_POST$",
                                                        "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[N3Write_2c_1V_1L01], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_2c_1V_1L01], (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[N3Write_4c_2V_2L01], (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N3Write_4c_2V_2L01], (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, B/H/S
def : InstRW<[N3Write_4c_3V_3L01], (instregex "ST3Threev(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N3Write_4c_3V_3L01], (instregex "ST3Threev(16b|8h|4s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[N3Write_2c_3V_3L01], (instregex "ST3Threev2d$")>;
def : InstRW<[WriteAdr, N3Write_2c_3V_3L01], (instregex "ST3Threev2d_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[N3Write_2c_2V_2L01], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2V_2L01], (instregex "ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[N3Write_4c_2V_2L01], (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N3Write_4c_2V_2L01], (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[N3Write_4c_4V_4L01], (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N3Write_4c_4V_4L01], (instregex "ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[N3Write_2c_2V_2L01], (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2V_2L01], (instregex "ST4Fourv(2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H/S
// ASIMD store, 4 element, one lane, D
def : InstRW<[N3Write_2c_2V_2L01], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N3Write_2c_2V_2L01], (instregex "ST4i(8|16|32|64)_POST$")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[N3Write_2c_1V], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[N3Write_2c_1V0], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
def : InstRW<[N3Write_2c_1V0], (instregex "^SHA1(H|SU0|SU1)")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[N3Write_4c_1V0], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[N3Write_2c_1V0], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[N3Write_2c_1V0], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[N3Write_2c_1V], (instrs BCAX, EOR3, RAX1, XAR)>;

// Crypto SM3 ops
def : InstRW<[N3Write_2c_1V0], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                          "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[N3Write_4c_1V0], (instrs SM4E, SM4ENCKEY)>;

// CRC
// -----------------------------------------------------------------------------

// CRC checksum ops
def : InstRW<[N3Write_2c_1M0], (instregex "^CRC32")>;

// SVE Predicate instructions
// -----------------------------------------------------------------------------

// Loop control, based on predicate
def : InstRW<[N3Write_2c_1M], (instrs BRKA_PPmP, BRKA_PPzP,
                                      BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[N3Write_2c_1M], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[N3Write_2c_1M], (instrs BRKN_PPzP, BRKPA_PPzPP, BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[N3Write_2c_1M], (instrs BRKNS_PPzP, BRKPAS_PPzPP, BRKPBS_PPzPP)>;

// Loop control, based on GPR
def : InstRW<[N3Write_2c_1M],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]$")>;
def : InstRW<[N3Write_2c_1M], (instregex "^WHILE(RW|WR)_PXX_[BHSD]$")>;

// Loop terminate
def : InstRW<[N3Write_1c_1M], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// Predicate counting scalar
def : InstRW<[N3Write_1c_1I], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;
def : InstRW<[N3Write_1c_1I],
             (instregex "^(CNT|SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI",
                        "^UQ(DEC|INC)[BHWD]_WPiI")>;

// Predicate counting scalar, ALL, {1,2,4}
def : InstRW<[N3Write_1c_1I], (instregex "^(DEC|INC)[BHWD]_XPiI")>;

// Predicate counting scalar, active predicate
def : InstRW<[N3Write_2c_1M],
             (instregex "^CNTP_XPP_[BHSD]",
                        "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;

// Predicate counting vector, active predicate
def : InstRW<[N3Write_7c_1M_1M0_1V],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;

// Predicate logical
def : InstRW<[N3Write_1c_1M],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
def : InstRW<[N3Write_1c_1M],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
def : InstRW<[N3Write_2c_1M], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
def : InstRW<[N3Write_1c_1M], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[N3Write_2c_1M], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
def : InstRW<[N3Write_2c_1M], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
def : InstRW<[N3Write_2c_1M], (instregex "^PFIRST_B$", "^PNEXT_[BHSD]$")>;

// Predicate test
def : InstRW<[N3Write_1c_1M], (instrs PTEST_PP)>;

// Predicate transpose
def : InstRW<[N3Write_2c_1M], (instregex "^TRN[12]_PPP_[BHSDQ]$")>;

// Predicate unpack and widen
def : InstRW<[N3Write_2c_1M], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[N3Write_2c_1M], (instregex "^(ZIP|UZP)[12]_PPP_[BHSDQ]$")>;

// SVE integer instructions
// -----------------------------------------------------------------------------

// Arithmetic, absolute diff
def : InstRW<[N3Write_2c_1V], (instregex "^[SU]ABD_ZPmZ_[BHSD]",
                                         "^[SU]ABD_ZPZZ_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]ABA_ZZZ_[BHSD]$")>;

// Arithmetic, absolute diff accum long
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]$")>;

// Arithmetic, absolute diff long
def : InstRW<[N3Write_2c_1V], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]$")>;

// Arithmetic, basic
def : InstRW<[N3Write_2c_1V],
             (instregex "^(ABS|ADD|CNOT|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZI_[BHSD]",
                        "^ADR_[SU]XTW_ZZZ_D_[0123]",
                        "^ADR_LSL_ZZZ_[SD]_[0123]",
                        "^[SU](ADD|SUB)[LW][BT]_ZZZ_[HSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^[SU]H(ADD|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]")>;

// Arithmetic, complex
def : InstRW<[N3Write_2c_1V],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^SQ(ABS|ADD|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;

// Arithmetic, large integer
def : InstRW<[N3Write_2c_1V], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]$")>;

// Arithmetic, pairwise add
def : InstRW<[N3Write_2c_1V], (instregex "^ADDP_ZPmZ_[BHSD]$")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]ADALP_ZPmZ_[HSD]$")>;

// Arithmetic, shift
def : InstRW<[N3Write_2c_1V1],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZ[IZ]_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[N3Write_4c_1V1],
             (instregex "^(SRSRA|SSRA|URSRA|USRA)_ZZI_[BHSD]$")>;

// Arithmetic, shift by immediate
// Arithmetic, shift by immediate and insert
def : InstRW<[N3Write_2c_1V1],
             (instregex "^(SHRNB|SHRNT|SSHLLB|SSHLLT|USHLLB|USHLLT|SLI|SRI)_ZZI_[BHSD]$")>;

// Arithmetic, shift complex
def : InstRW<[N3Write_4c_1V1],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_ZPmZ_[BHSD]",
                        "^[SU]QR?SHL_ZPZZ_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift right for divide
def : InstRW<[N3Write_4c_1V1], (instregex "^ASRD_(ZPmI|ZPZI)_[BHSD]")>;

// Arithmetic, shift rounding
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]RSHLR?_ZPmZ_[BHSD]",
                                          "^[SU]RSHL_ZPZZ_[BHSD]",
                                          "^[SU]RSHR_(ZPmI|ZPZI)_[BHSD]")>;

// Bit manipulation
def : InstRW<[N3Write_4c_2V0], (instregex "^(BDEP|BEXT|BGRP)_ZZZ_[BHSD]")>;

// Bitwise select
def : InstRW<[N3Write_2c_1V], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ$")>;

// Count/reverse bits
def : InstRW<[N3Write_2c_1V], (instregex "^(CLS|CLZ|CNT|RBIT)_ZPmZ_[BHSD]")>;

// Broadcast logical bitmask immediate to vector
def : InstRW<[N3Write_2c_1V], (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[N3Write_2c_1V],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]$",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]$")>;

// Complex add
def : InstRW<[N3Write_2c_1V], (instregex "^(SQ)?CADD_ZZI_[BHSD]$")>;

// Complex dot product 8-bit element
def : InstRW<[N3Write_3c_1V], (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[N3Write_4c_1V0], (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^CMLA_ZZZ_[BHS]$", "^CMLA_ZZZI_[HS]$")>;

// Complex multiply-add D element size
def : InstRW<[N3Write_5c_2V0], (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form
def : InstRW<[N3Write_8c_1M0_1V], (instregex "^CLAST[AB]_RPZ_[BHSD]$")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[N3Write_2c_1V], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]$",
                                         "^COMPACT_ZPZ_[SD]$",
                                         "^SPLICE_ZPZZ?_[BHSD]$")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[N3Write_3c_1V0], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                          "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 32b to single or half
def : InstRW<[N3Write_4c_2V0], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 16b to half
def : InstRW<[N3Write_6c_4V0], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[N3Write_5c_1M0_1V], (instregex "^CPY_ZPmR_[BHSD]$")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[N3Write_2c_1V], (instregex "^CPY_ZPm[IV]_[BHSD]$",
                                         "^CPY_ZPzI_[BHSD]$")>;

// SVE integer divide operations are now performed using
// a fully pipelined data path.

// Divides, 32 bit
def : InstRW<[N3Write_8c_8V0], (instregex "^[SU]DIVR?_ZPmZ_S",
                                          "^[SU]DIV_ZPZZ_S")>;

// Divides, 64 bit
def : InstRW<[N3Write_16c_16V0], (instregex "^[SU]DIVR?_ZPmZ_D",
                                            "^[SU]DIV_ZPZZ_D")>;

// Dot product, 8 bit
def : InstRW<[N3Write_3c_1V], (instregex "^[SU]DOT_ZZZI?_S$")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[N3Write_3c_1V], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[N3Write_4c_1V0], (instregex "^[SU]DOT_ZZZI?_D$")>;

// Duplicate, immediate and indexed form
def : InstRW<[N3Write_2c_1V], (instregex "^DUP_ZI_[BHSD]$",
                                         "^DUP_ZZI_[BHSDQ]$")>;

// Duplicate, scalar form
def : InstRW<[N3Write_3c_1M0], (instregex "^DUP_ZR_[BHSD]$")>;

// Extend, sign or zero
def : InstRW<[N3Write_2c_1V], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                         "^[SU]XTH_ZPmZ_[SD]",
                                         "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[N3Write_2c_1V], (instrs EXT_ZZI, EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[N3Write_4c_1V1], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]$",
                                          "^SQXTUN[BT]_ZZ_[BHS]$")>;

// Extract/insert operation, SIMD and FP scalar form
def : InstRW<[N3Write_2c_1V], (instregex "^LAST[AB]_VPZ_[BHSD]$",
                                         "^INSR_ZV_[BHSD]$")>;

// Extract/insert operation, scalar
def : InstRW<[N3Write_5c_1V], (instregex "^LAST[AB]_RPZ_[BHSD]$",
                                         "^INSR_ZR_[BHSD]$")>;

// Histogram operations
def : InstRW<[N3Write_2c_1V], (instregex "^HISTCNT_ZPzZZ_[SD]$",
                                         "^HISTSEG_ZZZ$")>;

// Horizontal operations, B, H, S form, immediate operands only
def : InstRW<[N3Write_2c_1V], (instregex "^INDEX_II_[BHS]$")>;

// Horizontal operations, B, H, S form, scalar, immediate operands / scalar operands only / immediate, scalar operands
def : InstRW<[N3Write_5c_1M0_1V], (instregex "^INDEX_(IR|RI|RR)_[BHS]$")>;

// Horizontal operations, D form, immediate operands only
def : InstRW<[N3Write_2c_1V], (instrs INDEX_II_D)>;

// Horizontal operations, D form, scalar, immediate operands / scalar operands only / immediate, scalar operands
def : InstRW<[N3Write_5c_1M0_1V], (instregex "^INDEX_(IR|RI|RR)_D$")>;

// Logical
def : InstRW<[N3Write_2c_1V],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|ORR)_ZZZ",
                        "^EOR(BT|TB)_ZZZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^NOT_ZPmZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[N3Write_2c_1V], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                         "^[SU](MAX|MIN)P?_ZPmZ_[BHSD]",
                                         "^[SU](MAX|MIN)_ZPZZ_[BHSD]")>;

// Matching operations
def : InstRW<[N3Write_2c_1V], (instregex "^N?MATCH_PPzZZ_[BH]$")>;

// Matrix multiply-accumulate
def : InstRW<[N3Write_3c_1V], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[N3Write_2c_1V], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]$",
                                         "^MOVPRFX_ZZ$")>;

// Multiply, B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_[BHS]",
                                          "^MUL_ZPZZ_[BHS]",
                                          "^[SU]MULH_(ZPmZ|ZZZ)_[BHS]",
                                          "^[SU]MULH_ZPZZ_[BHS]")>;

// Multiply, D element size
def : InstRW<[N3Write_5c_2V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_D",
                                          "^MUL_ZPZZ_D",
                                          "^[SU]MULH_(ZPmZ|ZZZ)_D",
                                          "^[SU]MULH_ZPZZ_D")>;

// Multiply long
def : InstRW<[N3Write_4c_1V0], (instregex "^[SU]MULL[BT]_ZZZI_[SD]$",
                                          "^[SU]MULL[BT]_ZZZ_[HSD]$")>;

// Multiply accumulate, B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^ML[AS]_ZZZI_[BHS]$",
                                          "^(ML[AS]|MAD|MSB)_(ZPmZZ|ZPZZZ)_[BHS]")>;

// Multiply accumulate, D element size
def : InstRW<[N3Write_5c_2V0], (instregex "^ML[AS]_ZZZI_D$",
                                          "^(ML[AS]|MAD|MSB)_(ZPmZZ|ZPZZZ)_D")>;

// Multiply accumulate long
def : InstRW<[N3Write_4c_1V0], (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]$",
                                          "^[SU]ML[AS]L[BT]_ZZZI_[SD]$")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[N3Write_4c_1V0], (instregex "^SQDML[AS](LB|LT|LBT)_ZZZ_[HSD]$",
                                          "^SQDML[AS](LB|LT)_ZZZI_[SD]$")>;

// Multiply saturating doubling high, B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^SQDMULH_ZZZ_[BHS]$",
                                          "^SQDMULH_ZZZI_[HS]$")>;

// Multiply saturating doubling high, D element size
def : InstRW<[N3Write_5c_2V0], (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long
def : InstRW<[N3Write_4c_1V0], (instregex "^SQDMULL[BT]_ZZZ_[HSD]$",
                                          "^SQDMULL[BT]_ZZZI_[SD]$")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^SQRDML[AS]H_ZZZ_[BHS]$",
                                          "^SQRDCMLAH_ZZZ_[BHS]$",
                                          "^SQRDML[AS]H_ZZZI_[HS]$",
                                          "^SQRDCMLAH_ZZZI_[HS]$")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element size
def : InstRW<[N3Write_5c_2V0], (instregex "^SQRDML[AS]H_ZZZI?_D$",
                                          "^SQRDCMLAH_ZZZ_D$")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[N3Write_4c_1V0], (instregex "^SQRDMULH_ZZZ_[BHS]$",
                                          "^SQRDMULH_ZZZI_[HS]$")>;

// Multiply saturating rounding doubling regular/complex, D element size
def : InstRW<[N3Write_5c_2V0], (instregex "^SQRDMULH_ZZZI?_D$")>;

// Multiply/multiply long, (8x8) polynomial
def : InstRW<[N3Write_2c_1V0], (instregex "^PMUL_ZZZ_B$",
                                          "^PMULL[BT]_ZZZ_[HDQ]$")>;

// Predicate counting vector
def : InstRW<[N3Write_2c_1V],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)[HWD]_ZPiI$")>;

// Reciprocal estimate
def : InstRW<[N3Write_4c_1V0], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[N3Write_8c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_B")>;

// Reduction, arithmetic, H form
def : InstRW<[N3Write_7c_1V_1V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[N3Write_4c_1V], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[N3Write_4c_1V], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[N3Write_5c_1V_1V1], (instregex "^(AND|EOR|OR)V_VPZ_[BHSD]$")>;

// Reverse, vector
def : InstRW<[N3Write_2c_1V], (instregex "^REV_ZZ_[BHSD]$",
                                         "^REVB_ZPmZ_[HSD]$",
                                         "^REVH_ZPmZ_[SD]$",
                                         "^REVW_ZPmZ_D$")>;

// Select, vector form
// Table lookup
// Table lookup extension
// Transpose, vector form
// Unpack and extend
// Zip/unzip
def : InstRW<[N3Write_2c_1V], (instregex "^SEL_ZPZZ_[BHSD]$",
                                         "^TBL_ZZZZ?_[BHSD]$",
                                         "^TBX_ZZZ_[BHSD]$",
                                         "^TRN[12]_ZZZ_[BHSDQ]$",
                                         "^[SU]UNPK(HI|LO)_ZZ_[HSD]$",
                                         "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]$")>;

// SVE floating-point instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[N3Write_2c_1V], (instregex "^FAB[SD]_ZPmZ_[HSD]",
                                         "^FABD_ZPZZ_[HSD]",
                                         "^FABS_ZPmZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[N3Write_2c_1V], (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ)_[HSD]",
                                         "^F(ADD|SUB)_ZPZ[IZ]_[HSD]",
                                         "^FADDP_ZPmZZ_[HSD]",
                                         "^FNEG_ZPmZ_[HSD]",
                                         "^FSUBR_ZPm[IZ]_[HSD]",
                                         "^FSUBR_(ZPZI|ZPZZ)_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[N3Write_16c_8V], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[N3Write_8c_4V], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[N3Write_4c_2V], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[N3Write_2c_1V], (instregex "^FAC(GE|GT)_PPzZZ_[HSD]$",
                                         "^FCM(EQ|GE|GT|NE|UO)_PPzZZ_[HSD]$",
                                         "^FCM(EQ|GE|GT|LE|LT|NE)_PPzZ0_[HSD]$")>;

// Floating point complex add
def : InstRW<[N3Write_3c_1V], (instregex "^FCADD_ZPmZ_[HSD]$")>;

// Floating point complex multiply add
def : InstRW<[N3Write_4c_1V], (instregex "^FCMLA_ZPmZZ_[HSD]$",
                                         "^FCMLA_ZZZI_[HS]$")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[N3Write_4c_2V0], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                          "^FCVTLT_ZPmZ_HtoS",
                                          "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32 or F64 to F16)
def : InstRW<[N3Write_3c_1V0], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                          "^FCVTLT_ZPmZ_StoD",
                                          "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[N3Write_3c_1V0], (instrs FCVTX_ZPmZ_DtoS, FCVTXNT_ZPmZ_DtoS)>;

// Floating point base2 log, F16
def : InstRW<[N3Write_6c_4V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[N3Write_4c_2V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[N3Write_3c_1V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[N3Write_6c_4V0], (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[N3Write_4c_2V0], (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[N3Write_3c_1V0],
             (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[N3Write_2c_1V], (instregex "^FCPY_ZPmI_[HSD]$",
                                         "^FDUP_ZI_[HSD]$")>;

// SVE FP divide and square root operations are now performed using
// a fully pipelined data path.

// Floating point divide, F16
def : InstRW<[N3Write_12c_8V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[N3Write_10c_4V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[N3Write_13c_2V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point arith, min/max pairwise
def : InstRW<[N3Write_3c_1V], (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[N3Write_2c_1V], (instregex "^F(MAX|MIN)(NM)?_ZPm[IZ]_[HSD]",
                                         "^F(MAX|MIN)(NM)?_ZPZ[IZ]_[HSD]")>;

// Floating point multiply
def : InstRW<[N3Write_3c_1V], (instregex "^(FSCALE|FMULX)_ZPmZ_[HSD]",
                                         "^FMULX_ZPZZ_[HSD]",
                                         "^FMUL_(ZPm[IZ]|ZZZI?)_[HSD]",
                                         "^FMUL_ZPZ[IZ]_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[N3Write_4c_1V], (instregex "^F(N?M(AD|SB)|N?ML[AS])_ZPmZZ_[HSD]$",
                                         "^FN?ML[AS]_ZPZZZ_[HSD]",
                                         "^FML[AS]_ZZZI_[HSD]$")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[N3Write_4c_1V], (instregex "^FML[AS]L[BT]_ZZZI?_SHH$")>;

// Floating point reciprocal estimate, F16
def : InstRW<[N3Write_6c_4V0], (instregex "^FR(ECP|SQRT)E_ZZ_H", "^FRECPX_ZPmZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[N3Write_4c_2V0], (instregex "^FR(ECP|SQRT)E_ZZ_S", "^FRECPX_ZPmZ_S")>;

// Floating point reciprocal estimate, F64
def : InstRW<[N3Write_3c_1V0], (instregex "^FR(ECP|SQRT)E_ZZ_D", "^FRECPX_ZPmZ_D")>;

// Floating point reciprocal step
def : InstRW<[N3Write_4c_1V], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]$")>;

// Floating point reduction, F16
def : InstRW<[N3Write_6c_3V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_H$")>;

// Floating point reduction, F32
def : InstRW<[N3Write_4c_2V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_S$")>;

// Floating point reduction, F64
def : InstRW<[N3Write_2c_1V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_D$")>;

// Floating point round to integral, F16
def : InstRW<[N3Write_6c_4V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[N3Write_4c_2V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[N3Write_3c_1V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[N3Write_12c_8V0], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[N3Write_10c_4V0], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root F64
def : InstRW<[N3Write_13c_2V0], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[N3Write_2c_1V], (instregex "^FEXPA_ZZ_[HSD]$")>;

// Floating point trigonometric multiply add
def : InstRW<[N3Write_4c_1V], (instregex "^FTMAD_ZZI_[HSD]$")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[N3Write_3c_1V], (instregex "^FTS(MUL|SEL)_ZZZ_[HSD]$")>;

// SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[N3Write_4c_2V0], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[N3Write_4c_1V], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[N3Write_5c_1V], (instrs BFMMLA_ZZZ)>;

// Multiply accumulate long
def : InstRW<[N3Write_4c_1V], (instregex "^BFMLAL[BT]_ZZZ(I)?$")>;

// SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[N3Write_6c_1L], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[N3Write_7c_1L_1M], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[N3Write_6c_1L], (instregex "^LD1[BHWD]_IMM$",
                                         "^LD1S?B_[HSD]_IMM$",
                                         "^LD1S?H_[SD]_IMM$",
                                         "^LD1S?W_D_IMM$" )>;

// Contiguous load, scalar + scalar
def : InstRW<[N3Write_6c_1L], (instregex "^LD1[BHWD]$",
                                         "^LD1S?B_[HSD]$",
                                         "^LD1S?H_[SD]$",
                                         "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[N3Write_6c_1L], (instregex "^LD1R[BHWD]_IMM$",
                                         "^LD1RSW_IMM$",
                                         "^LD1RS?B_[HSD]_IMM$",
                                         "^LD1RS?H_[SD]_IMM$",
                                         "^LD1RS?W_D_IMM$",
                                         "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[N3Write_6c_1L], (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
def : InstRW<[N3Write_6c_1L], (instregex "^LDNT1[BHWD]_ZRI$")>;

// Non temporal load, scalar + scalar
def : InstRW<[N3Write_6c_1L], (instregex "^LDNT1[BHWD]_ZRR$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[N3Write_7c_4L], (instregex "^LDNT1[BHW]_ZZR_S$",
                                         "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[N3Write_6c_2L], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[N3Write_6c_2L], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[N3Write_6c_1L], (instregex "^LDFF1[BHWD]$",
                                         "^LDFF1S?B_[HSD]$",
                                         "^LDFF1S?H_[SD]$",
                                         "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[N3Write_6c_1L], (instregex "^LDNF1[BHWD]_IMM$",
                                         "^LDNF1S?B_[HSD]_IMM$",
                                         "^LDNF1S?H_[SD]_IMM$",
                                         "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[N3Write_8c_1V_1L], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[N3Write_8c_1V_1L], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[N3Write_8c_3L_3V], (instregex "^LD3D_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[N3Write_10c_6V_3L], (instregex "^LD3[BHW]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[N3Write_9c_3V_3L_3I], (instregex "^LD3D$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[N3Write_11c_6V_3L_6I], (instregex "^LD3[BHW]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[N3Write_8c_4L_4V], (instregex "^LD4D_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[N3Write_12c_5V_4L], (instregex "^LD4[BHW]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[N3Write_9c_4L_4V_4I], (instregex "^LD4D$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[N3Write_13c_4L_5V_5I], (instregex "^LD4[BHW]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[N3Write_7c_4L], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                         "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[N3Write_6c_2L], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                         "^GLD(FF)?1D_IMM$")>;

// Gather load, 64-bit element size
def : InstRW<[N3Write_6c_2L],
             (instregex "^GLD(FF)?1S?[BHW]_D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1S?[BHW]_D(_SCALED)?$",
                        "^GLD(FF)?1D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1D(_SCALED)?$")>;

// Gather load, 32-bit element size
def : InstRW<[N3Write_7c_4L],
             (instregex "^GLD(FF)?1S?[HW]_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED",
                        "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                        "^GLD(FF)?1W_[SU]XTW$")>;

// SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[N3Write_1c_1L01], (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[N3Write_2c_1L01_1V], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^ST1[BHWD]_IMM$",
                                              "^ST1B_[HSD]_IMM$",
                                              "^ST1H_[SD]_IMM$",
                                              "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[N3Write_2c_1L01_1I_1V], (instregex "^ST1H(_[SD])?$")>;

// Contiguous store, scalar + scalar
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^ST1[BWD]$",
                                              "^ST1B_[HSD]$",
                                              "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^ST2[BHWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[N3Write_4c_3L01_3V], (instregex "^ST3[BHW]_IMM$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[N3Write_3c_3L01_3V], (instregex "^ST3D_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[N3Write_4c_3L01_3I_3V], (instregex "^ST3[BHW]$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[N3Write_3c_3L01_3I_3V], (instregex "^ST3D$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[N3Write_6c_3L01_3V], (instregex "^ST4[BHW]_IMM$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[N3Write_3c_4L01_4V], (instregex "^ST4D_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[N3Write_3c_4L01_4I_4V], (instregex "^ST4D$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[N3Write_6c_3L01_3I_3V], (instregex "^ST4[BHW]$")>;

// Non temporal store, scalar + imm
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STNT1[BHWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[N3Write_2c_2L01_2V], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[N3Write_2c_2L01_2V], (instregex "^SST1[BH]_S_IMM$",
                                              "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^SST1[BHW]_D_IMM$",
                                              "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[N3Write_2c_2L01_2V],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                              "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                              "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[N3Write_2c_2L01_2V], (instregex "^SST1[BH]_S_[SU]XTW$",
                                              "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^SST1[HW]_D_SCALED$",
                                              "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[N3Write_2c_1L01_1V], (instregex "^SST1[BHW]_D$",
                                              "^SST1D$")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[N3Write_2c_1M], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[N3Write_2c_1M], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[N3Write_2c_1M], (instrs RDFFRS_PPz)>;

// Set first fault register
def : InstRW<[N3Write_0c], (instrs SETFFR)>;

// Write to first fault register
def : InstRW<[N3Write_2c_1M0], (instrs WRFFR)>;

// Prefetch
def : InstRW<[N3Write_4c_1L], (instregex "^PRF[BHWD]")>;

// SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[N3Write_2c_1V], (instregex "^AES[DE]_ZZZ_B$",
                                         "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[N3Write_2c_1V], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                         "^RAX1_ZZZ_D$",
                                         "^XAR_ZZZI_[BHSD]$")>;

// Crypto SM4 ops
def : InstRW<[N3Write_4c_1V0], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

}
