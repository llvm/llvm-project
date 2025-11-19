//=- AArch64SchedNeoverseV3AE.td - NeoverseV3AE Scheduling Defs --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Arm Neoverse V3AE processors.
// All information is taken from the V3AE Software Optimisation guide:
//
// https://developer.arm.com/documentation/109703/300/?lang=en
//
//===----------------------------------------------------------------------===//

def NeoverseV3AEModel : SchedMachineModel {
  let IssueWidth            =  10; // Expect best value to be slightly higher than V2
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer. NOTE: Copied from Neoverse-V2
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.  NOTE: Copied from N2.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasCPA, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Neoverse V3AE.
// Instructions are first fetched and then decoded into internal macro-ops
// (MOPs). From there, the MOPs proceed through register renaming and dispatch
// stages. A MOP can be split into two micro-ops further down the pipeline
// after the decode stage. Once dispatched, micro-ops wait for their operands
// and issue out-of-order to one of nineteen issue pipelines. Each issue
// pipeline can accept one micro-op per cycle.

let SchedModel = NeoverseV3AEModel in {

// Define the (19) issue ports.
def V3AEUnitB   : ProcResource<3>;  // Branch 0/1/2
def V3AEUnitS0  : ProcResource<1>;  // Integer single-cycle 0
def V3AEUnitS1  : ProcResource<1>;  // Integer single-cycle 1
def V3AEUnitS2  : ProcResource<1>;  // Integer single-cycle 2
def V3AEUnitS3  : ProcResource<1>;  // Integer single-cycle 3
def V3AEUnitS4  : ProcResource<1>;  // Integer single-cycle 4
def V3AEUnitS5  : ProcResource<1>;  // Integer single-cycle 5
def V3AEUnitM0  : ProcResource<1>;  // Integer single/multicycle 0
def V3AEUnitM1  : ProcResource<1>;  // Integer single/multicycle 1
def V3AEUnitV0  : ProcResource<1>;  // FP/ASIMD 0
def V3AEUnitV1  : ProcResource<1>;  // FP/ASIMD 1
def V3AEUnitLS0 : ProcResource<1>;  // Load/Store 0
def V3AEUnitL12 : ProcResource<2>;  // Load 1/2
def V3AEUnitST1 : ProcResource<1>;  // Store 1
def V3AEUnitD   : ProcResource<2>;  // Store data 0/1
def V3AEUnitFlg : ProcResource<4>;  // Flags

def V3AEUnitS   : ProcResGroup<[V3AEUnitS0, V3AEUnitS1, V3AEUnitS2, V3AEUnitS3, V3AEUnitS4, V3AEUnitS5]>;  // Integer single-cycle 0/1/2/3/4/5
def V3AEUnitI   : ProcResGroup<[V3AEUnitS0, V3AEUnitS1, V3AEUnitS2, V3AEUnitS3, V3AEUnitS4, V3AEUnitS5, V3AEUnitM0, V3AEUnitM1]>;  // Integer single-cycle 0/1/2/3/4/5 and single/multicycle 0/1
def V3AEUnitM   : ProcResGroup<[V3AEUnitM0, V3AEUnitM1]>;  // Integer single/multicycle 0/1
def V3AEUnitLSA : ProcResGroup<[V3AEUnitLS0, V3AEUnitL12, V3AEUnitST1]>; // Supergroup of L+SA
def V3AEUnitL   : ProcResGroup<[V3AEUnitLS0, V3AEUnitL12]>; // Load/Store 0 and Load 1/2
def V3AEUnitSA  : ProcResGroup<[V3AEUnitLS0, V3AEUnitST1]>; // Load/Store 0 and Store 1
def V3AEUnitV   : ProcResGroup<[V3AEUnitV0, V3AEUnitV1]>;  // FP/ASIMD 0/1

// Define commonly used read types.

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

// NOTE: Copied from N2.
def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }
def : WriteRes<WriteLDHi,    []> { let Latency = 4; }

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the Neoverse V3AE.

//===----------------------------------------------------------------------===//

// Define generic 0 micro-op types
def V3AEWrite_0c : SchedWriteRes<[]> { let Latency = 0; }

// Define generic 1 micro-op types

def V3AEWrite_1c_1B    : SchedWriteRes<[V3AEUnitB]>   { let Latency = 1; }
def V3AEWrite_1c_1F_1Flg : SchedWriteRes<[V3AEUnitI, V3AEUnitFlg]>   { let Latency = 1; }
def V3AEWrite_1c_1I    : SchedWriteRes<[V3AEUnitI]>   { let Latency = 1; }
def V3AEWrite_1c_1M    : SchedWriteRes<[V3AEUnitM]>   { let Latency = 1; }
def V3AEWrite_1c_1SA   : SchedWriteRes<[V3AEUnitSA]>  { let Latency = 1; }
def V3AEWrite_2c_1M    : SchedWriteRes<[V3AEUnitM]>   { let Latency = 2; }
def V3AEWrite_2c_1M_1Flg : SchedWriteRes<[V3AEUnitM, V3AEUnitFlg]>   { let Latency = 2; }
def V3AEWrite_3c_1M    : SchedWriteRes<[V3AEUnitM]>   { let Latency = 3; }
def V3AEWrite_2c_1M0   : SchedWriteRes<[V3AEUnitM0]>  { let Latency = 2; }
def V3AEWrite_3c_1M0   : SchedWriteRes<[V3AEUnitM0]>  { let Latency = 3; }
def V3AEWrite_4c_1M0   : SchedWriteRes<[V3AEUnitM0]>  { let Latency = 4; }
def V3AEWrite_12c_1M0  : SchedWriteRes<[V3AEUnitM0]>  { let Latency = 12;
                                                    let ReleaseAtCycles = [12]; }
def V3AEWrite_20c_1M0  : SchedWriteRes<[V3AEUnitM0]>  { let Latency = 20;
                                                    let ReleaseAtCycles = [20]; }
def V3AEWrite_4c_1L    : SchedWriteRes<[V3AEUnitL]>   { let Latency = 4; }
def V3AEWrite_6c_1L    : SchedWriteRes<[V3AEUnitL]>   { let Latency = 6; }
def V3AEWrite_2c_1V    : SchedWriteRes<[V3AEUnitV]>   { let Latency = 2; }
def V3AEWrite_2c_1V0   : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 2; }
def V3AEWrite_3c_1V    : SchedWriteRes<[V3AEUnitV]>   { let Latency = 3; }
def V3AEWrite_4c_1V    : SchedWriteRes<[V3AEUnitV]>   { let Latency = 4; }
def V3AEWrite_5c_1V    : SchedWriteRes<[V3AEUnitV]>   { let Latency = 5; }
def V3AEWrite_6c_1V    : SchedWriteRes<[V3AEUnitV]>   { let Latency = 6; }
def V3AEWrite_12c_1V   : SchedWriteRes<[V3AEUnitV]>   { let Latency = 12; }
def V3AEWrite_3c_1V0   : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 3; }
def V3AEWrite_4c_1V0   : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 4; }
def V3AEWrite_9c_1V0   : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 9; }
def V3AEWrite_10c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 10; }
def V3AEWrite_8c_1V1   : SchedWriteRes<[V3AEUnitV1]> { let Latency = 8; }
def V3AEWrite_12c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 12;
                                                    let ReleaseAtCycles = [11]; }
def V3AEWrite_13c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 13; }
def V3AEWrite_15c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 15; }
def V3AEWrite_13c_1V1  : SchedWriteRes<[V3AEUnitV1]> { let Latency = 13;
                                                   let ReleaseAtCycles = [8]; }
def V3AEWrite_16c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 16; }
def V3AEWrite_20c_1V0  : SchedWriteRes<[V3AEUnitV0]>  { let Latency = 20;
                                                    let ReleaseAtCycles = [20]; }
def V3AEWrite_2c_1V1   : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 2; }
def V3AEWrite_3c_1V1   : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 3; }
def V3AEWrite_4c_1V1   : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 4; }
def V3AEWrite_6c_1V1   : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 6; }
def V3AEWrite_10c_1V1  : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 10; }
def V3AEWrite_6c_1SA   : SchedWriteRes<[V3AEUnitSA]>  { let Latency = 6; }

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

def V3AEWrite_1c_1B_1S : SchedWriteRes<[V3AEUnitB, V3AEUnitS]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1M0_1B : SchedWriteRes<[V3AEUnitM0, V3AEUnitB]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_9c_1M0_1L : SchedWriteRes<[V3AEUnitM0, V3AEUnitL]> {
  let Latency     = 9;
  let NumMicroOps = 2;
}

def V3AEWrite_3c_1I_1M : SchedWriteRes<[V3AEUnitI, V3AEUnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}

def V3AEWrite_1c_2M : SchedWriteRes<[V3AEUnitM, V3AEUnitM]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def V3AEWrite_3c_2M : SchedWriteRes<[V3AEUnitM, V3AEUnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}

def V3AEWrite_4c_2M : SchedWriteRes<[V3AEUnitM, V3AEUnitM]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def V3AEWrite_5c_1L_1I : SchedWriteRes<[V3AEUnitL, V3AEUnitI]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1I_1L : SchedWriteRes<[V3AEUnitI, V3AEUnitL]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_7c_1I_1L : SchedWriteRes<[V3AEUnitI, V3AEUnitL]> {
  let Latency     = 7;
  let NumMicroOps = 2;
}

def V3AEWrite_1c_1SA_1D : SchedWriteRes<[V3AEUnitSA, V3AEUnitD]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def V3AEWrite_5c_1M0_1V : SchedWriteRes<[V3AEUnitM0, V3AEUnitV]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def V3AEWrite_2c_1SA_1V : SchedWriteRes<[V3AEUnitSA, V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def V3AEWrite_2c_2V  : SchedWriteRes<[V3AEUnitV, V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def V3AEWrite_5c_1V1_1V : SchedWriteRes<[V3AEUnitV1, V3AEUnitV]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def V3AEWrite_4c_2V0 : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def V3AEWrite_4c_2V : SchedWriteRes<[V3AEUnitV, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_2V : SchedWriteRes<[V3AEUnitV, V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_2L : SchedWriteRes<[V3AEUnitL, V3AEUnitL]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_8c_1L_1V : SchedWriteRes<[V3AEUnitL, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 2;
}

def V3AEWrite_4c_1SA_1V : SchedWriteRes<[V3AEUnitSA, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def V3AEWrite_3c_1M0_1M  : SchedWriteRes<[V3AEUnitM0, V3AEUnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}

def V3AEWrite_4c_1M0_1M  : SchedWriteRes<[V3AEUnitM0, V3AEUnitM]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def V3AEWrite_1c_1M0_1M  : SchedWriteRes<[V3AEUnitM0, V3AEUnitM]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def V3AEWrite_2c_1M0_1M  : SchedWriteRes<[V3AEUnitM0, V3AEUnitM]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_2V1 : SchedWriteRes<[V3AEUnitV1, V3AEUnitV1]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_5c_2V0 : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def V3AEWrite_5c_1V1_1M0 : SchedWriteRes<[V3AEUnitV1, V3AEUnitM0]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1V1_1M0 : SchedWriteRes<[V3AEUnitV1, V3AEUnitM0]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_7c_1M0_1V0 : SchedWriteRes<[V3AEUnitM0, V3AEUnitV0]> {
  let Latency     = 7;
  let NumMicroOps = 2;
}

def V3AEWrite_2c_1V0_1M : SchedWriteRes<[V3AEUnitV0, V3AEUnitM]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def V3AEWrite_3c_1V0_1M : SchedWriteRes<[V3AEUnitV0, V3AEUnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1V_1V1 : SchedWriteRes<[V3AEUnitV, V3AEUnitV1]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1L_1M : SchedWriteRes<[V3AEUnitL, V3AEUnitM]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_6c_1L_1I : SchedWriteRes<[V3AEUnitL, V3AEUnitI]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def V3AEWrite_8c_1M0_1V : SchedWriteRes<[V3AEUnitM0, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 2;
}

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

def V3AEWrite_1c_1SA_1D_1I : SchedWriteRes<[V3AEUnitSA, V3AEUnitD, V3AEUnitI]> {
  let Latency     = 1;
  let NumMicroOps = 3;
}

def V3AEWrite_2c_1SA_1V_1I : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitI]> {
  let Latency     = 2;
  let NumMicroOps = 3;
}

def V3AEWrite_2c_1SA_2V : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 3;
}

def V3AEWrite_4c_1SA_2V : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 3;
}

def V3AEWrite_9c_1L_2V : SchedWriteRes<[V3AEUnitL, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 3;
}

def V3AEWrite_4c_3V  : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency = 4;
  let NumMicroOps = 3;
}

def V3AEWrite_7c_1M_1M0_1V : SchedWriteRes<[V3AEUnitM, V3AEUnitM0, V3AEUnitV]> {
  let Latency     = 7;
  let NumMicroOps = 3;
}

def V3AEWrite_2c_1SA_1I_1V : SchedWriteRes<[V3AEUnitSA, V3AEUnitI, V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 3;
}

def V3AEWrite_6c_3L : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL]> {
  let Latency     = 6;
  let NumMicroOps = 3;
}

def V3AEWrite_6c_3V : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 3;
}

def V3AEWrite_8c_1L_2V : SchedWriteRes<[V3AEUnitL, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 3;
}

//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types

def V3AEWrite_2c_1SA_2V_1I : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitI]> {
  let Latency     = 2;
  let NumMicroOps = 4;
}

def V3AEWrite_5c_1I_3L : SchedWriteRes<[V3AEUnitI, V3AEUnitL, V3AEUnitL, V3AEUnitL]> {
  let Latency     = 5;
  let NumMicroOps = 4;
}

def V3AEWrite_6c_4V0 : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0, V3AEUnitV0, V3AEUnitV0]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}

def V3AEWrite_8c_4V : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def V3AEWrite_6c_2V_2V1 : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV1,
                                         V3AEUnitV1]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}

def V3AEWrite_6c_4V : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}

def V3AEWrite_8c_2L_2V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def V3AEWrite_9c_2L_2V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 4;
}

def V3AEWrite_2c_2SA_2V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitV,
                                         V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 4;
}

def V3AEWrite_4c_2SA_2V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitV,
                                         V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 4;
}

def V3AEWrite_8c_2M0_2V0 : SchedWriteRes<[V3AEUnitM0, V3AEUnitM0, V3AEUnitV0,
                                          V3AEUnitV0]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def V3AEWrite_8c_2V_2V1 : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV1,
                                         V3AEUnitV1]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def V3AEWrite_4c_2M0_2M : SchedWriteRes<[V3AEUnitM0, V3AEUnitM0, V3AEUnitM,
                                         V3AEUnitM]> {
  let Latency     = 4;
  let NumMicroOps = 4;
}

def V3AEWrite_5c_2M0_2M : SchedWriteRes<[V3AEUnitM0, V3AEUnitM0, V3AEUnitM,
                                         V3AEUnitM]> {
  let Latency     = 5;
  let NumMicroOps = 4;
}

def V3AEWrite_6c_2I_2L : SchedWriteRes<[V3AEUnitI, V3AEUnitI, V3AEUnitL, V3AEUnitL]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}

def V3AEWrite_7c_4L : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL, V3AEUnitL]> {
  let Latency     = 7;
  let NumMicroOps = 4;
}

def V3AEWrite_6c_1SA_3V : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}

//===----------------------------------------------------------------------===//
// Define generic 5 micro-op types

def V3AEWrite_2c_1SA_2V_2I : SchedWriteRes<[V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitI, V3AEUnitI]> {
  let Latency     = 2;
  let NumMicroOps = 5;
}

def V3AEWrite_8c_2L_3V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 5;
}

def V3AEWrite_9c_1L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 5;
}

def V3AEWrite_10c_1L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV]> {
  let Latency     = 10;
  let NumMicroOps = 5;
}

def V3AEWrite_6c_5V : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                     V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 5;
}

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

def V3AEWrite_8c_3L_3V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 6;
}

def V3AEWrite_9c_3L_3V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 6;
}

def V3AEWrite_9c_2L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitV,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 6;
}

def V3AEWrite_9c_2L_2V_2I : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitV,
                                           V3AEUnitV, V3AEUnitI, V3AEUnitI]> {
  let Latency     = 9;
  let NumMicroOps = 6;
}

def V3AEWrite_9c_2V_4V1 : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV1,
                                         V3AEUnitV1, V3AEUnitV1, V3AEUnitV1]> {
  let Latency     = 9;
  let NumMicroOps = 6;
}

def V3AEWrite_2c_3SA_3V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 6;
}

def V3AEWrite_4c_2SA_4V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 6;
}

def V3AEWrite_5c_2SA_4V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 5;
  let NumMicroOps = 6;
}

def V3AEWrite_4c_2SA_2I_2V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitI,
                                            V3AEUnitI, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 6;
}

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

def V3AEWrite_8c_3L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 7;
}

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types

def V3AEWrite_2c_4SA_4V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                        V3AEUnitSA, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV]> {
  let Latency     = 2;
  let NumMicroOps = 8;
}

def V3AEWrite_4c_4SA_4V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV]> {
  let Latency     = 4;
  let NumMicroOps = 8;
}

def V3AEWrite_6c_2SA_6V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 8;
}

def V3AEWrite_8c_4L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV]> {
  let Latency     = 8;
  let NumMicroOps = 8;
}

//===----------------------------------------------------------------------===//
// Define generic 9 micro-op types

def V3AEWrite_6c_3SA_6V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 6;
  let NumMicroOps = 9;
}

def V3AEWrite_10c_1L_8V : SchedWriteRes<[V3AEUnitL, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV]> {
  let Latency     = 10;
  let NumMicroOps = 9;
}

def V3AEWrite_10c_3V_3L_3I : SchedWriteRes<[V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                            V3AEUnitI, V3AEUnitI, V3AEUnitI]> {
  let Latency     = 10;
  let NumMicroOps = 9;
}

//===----------------------------------------------------------------------===//
// Define generic 10 micro-op types

def V3AEWrite_9c_6L_4V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitL, V3AEUnitL, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 10;
}

//===----------------------------------------------------------------------===//
// Define generic 12 micro-op types

def V3AEWrite_5c_4SA_8V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 5;
  let NumMicroOps = 12;
}

def V3AEWrite_9c_4L_8V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                        V3AEUnitL, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                        V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 9;
  let NumMicroOps = 12;
}

def V3AEWrite_10c_4L_8V : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                         V3AEUnitL, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 10;
  let NumMicroOps = 12;
}

//===----------------------------------------------------------------------===//
// Define generic 16 micro-op types

def V3AEWrite_7c_4SA_12V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                          V3AEUnitSA, V3AEUnitV, V3AEUnitV,
                                          V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                          V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                          V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                          V3AEUnitV]> {
  let Latency     = 7;
  let NumMicroOps = 16;
}

def V3AEWrite_10c_4L_8V_4I : SchedWriteRes<[V3AEUnitL, V3AEUnitL, V3AEUnitL,
                                            V3AEUnitL, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                            V3AEUnitI]> {
  let Latency     = 10;
  let NumMicroOps = 16;
}

//===----------------------------------------------------------------------===//
// Define generic 18 micro-op types

def V3AEWrite_7c_9SA_9V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                         V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 7;
  let NumMicroOps = 18;
}

//===----------------------------------------------------------------------===//
// Define generic 27 micro-op types

def V3AEWrite_7c_9SA_9I_9V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                            V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                            V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 7;
  let NumMicroOps = 27;
}

//===----------------------------------------------------------------------===//
// Define generic 36 micro-op types

def V3AEWrite_11c_18SA_18V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitSA, V3AEUnitSA, V3AEUnitSA,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                            V3AEUnitV, V3AEUnitV, V3AEUnitV]> {
  let Latency     = 11;
  let NumMicroOps = 36;
}

//===----------------------------------------------------------------------===//
// Define generic 54 micro-op types

def V3AEWrite_11c_18SA_18I_18V : SchedWriteRes<[V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitSA, V3AEUnitSA,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitI, V3AEUnitI, V3AEUnitI,
                                                V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV, V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV, V3AEUnitV,
                                                V3AEUnitV]> {
  let Latency     = 11;
  let NumMicroOps = 54;
}

//===----------------------------------------------------------------------===//
// Define predicate-controlled types

def V3AEWrite_ArithI : SchedWriteVariant<[
                       SchedVar<IsCheapLSL,  [V3AEWrite_1c_1I]>,
                       SchedVar<NoSchedPred, [V3AEWrite_2c_1M]>]>;

def V3AEWrite_ArithF : SchedWriteVariant<[
                       SchedVar<IsCheapLSL,  [V3AEWrite_1c_1F_1Flg]>,
                       SchedVar<NoSchedPred, [V3AEWrite_2c_1M_1Flg]>]>;

def V3AEWrite_Logical : SchedWriteVariant<[
                        SchedVar<NeoverseNoLSL, [V3AEWrite_1c_1F_1Flg]>,
                        SchedVar<NoSchedPred,   [V3AEWrite_2c_1M_1Flg]>]>;

def V3AEWrite_Extr : SchedWriteVariant<[
                     SchedVar<IsRORImmIdiomPred, [V3AEWrite_1c_1I]>,
                     SchedVar<NoSchedPred,       [V3AEWrite_3c_1I_1M]>]>;

def V3AEWrite_LdrHQ : SchedWriteVariant<[
                      SchedVar<NeoverseHQForm,  [V3AEWrite_7c_1I_1L]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_6c_1L]>]>;

def V3AEWrite_StrHQ : SchedWriteVariant<[
                      SchedVar<NeoverseHQForm,  [V3AEWrite_2c_1SA_1V_1I]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_2c_1SA_1V]>]>;

def V3AEWrite_0or1c_1I : SchedWriteVariant<[
                      SchedVar<NeoverseZeroMove, [V3AEWrite_0c]>,
                      SchedVar<NoSchedPred,      [V3AEWrite_1c_1I]>]>;

def V3AEWrite_0or2c_1V : SchedWriteVariant<[
                      SchedVar<NeoverseZeroMove, [V3AEWrite_0c]>,
                      SchedVar<NoSchedPred,      [V3AEWrite_2c_1V]>]>;

def V3AEWrite_0or3c_1M0 : SchedWriteVariant<[
                      SchedVar<NeoverseZeroMove, [V3AEWrite_0c]>,
                      SchedVar<NoSchedPred,      [V3AEWrite_3c_1M0]>]>;

def V3AEWrite_2or3c_1M : SchedWriteVariant<[
                      SchedVar<NeoversePdIsPg,  [V3AEWrite_3c_1M]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_2c_1M]>]>;

def V3AEWrite_1or2c_1M : SchedWriteVariant<[
                      SchedVar<NeoversePdIsPg,  [V3AEWrite_2c_1M]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_1c_1M]>]>;

def V3AEWrite_3or4c_1M0_1M : SchedWriteVariant<[
                      SchedVar<NeoversePdIsPg,  [V3AEWrite_4c_1M0_1M]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_3c_1M0_1M]>]>;

def V3AEWrite_2or3c_1V0 : SchedWriteVariant<[
                      SchedVar<NeoversePdIsPg,  [V3AEWrite_3c_1V0]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_2c_1V0]>]>;

def V3AEWrite_2or3c_1V0_1M : SchedWriteVariant<[
                      SchedVar<NeoversePdIsPg,  [V3AEWrite_3c_1V0_1M]>,
                      SchedVar<NoSchedPred,     [V3AEWrite_2c_1V0_1M]>]>;

def V3AEWrite_IncDec : SchedWriteVariant<[
                      SchedVar<NeoverseCheapIncDec, [V3AEWrite_1c_1I]>,
                      SchedVar<NoSchedPred,         [V3AEWrite_2c_1M]>]>;

//===----------------------------------------------------------------------===//
// Define forwarded types

// NOTE: SOG, p. 16, n. 2: Accumulator forwarding is not supported for
// consumers of 64 bit multiply high operations?
def V3AEWr_IM   : SchedWriteRes<[V3AEUnitM]>  { let Latency = 2; }

def V3AEWr_FMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_FMA : SchedReadAdvance<2, [WriteFMul, V3AEWr_FMA]>;

def V3AEWr_VA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VA : SchedReadAdvance<3, [V3AEWr_VA]>;

def V3AEWr_VDOT : SchedWriteRes<[V3AEUnitV]> { let Latency = 3; }
def V3AERd_VDOT : SchedReadAdvance<2, [V3AEWr_VDOT]>;

def V3AEWr_VMMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 3; }
def V3AERd_VMMA : SchedReadAdvance<2, [V3AEWr_VMMA]>;

def V3AEWr_VMA : SchedWriteRes<[V3AEUnitV0]> { let Latency = 4; }
def V3AERd_VMA : SchedReadAdvance<3, [V3AEWr_VMA]>;

def V3AEWr_VMAH : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> { let Latency = 4; }
def V3AERd_VMAH : SchedReadAdvance<2, [V3AEWr_VMAH]>;

def V3AEWr_VMAL : SchedWriteRes<[V3AEUnitV0]> { let Latency = 4; }
def V3AERd_VMAL : SchedReadAdvance<3, [V3AEWr_VMAL]>;

def V3AEWr_VPA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VPA : SchedReadAdvance<3, [V3AEWr_VPA]>;

def V3AEWr_VSA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VSA : SchedReadAdvance<3, [V3AEWr_VSA]>;

def V3AEWr_VFCMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VFCMA : SchedReadAdvance<2, [V3AEWr_VFCMA]>;

def V3AEWr_VFM  : SchedWriteRes<[V3AEUnitV]> { let Latency = 3; }
def V3AEWr_VFMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VFMA : SchedReadAdvance<2, [V3AEWr_VFM, V3AEWr_VFMA]>;

def V3AEWr_VFMAL : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_VFMAL : SchedReadAdvance<2, [V3AEWr_VFMAL]>;

def V3AEWr_VBFDOT : SchedWriteRes<[V3AEUnitV]> { let Latency = 5; }
def V3AERd_VBFDOT : SchedReadAdvance<2, [V3AEWr_VBFDOT]>;
def V3AEWr_VBFMMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 6; }
def V3AERd_VBFMMA : SchedReadAdvance<2, [V3AEWr_VBFMMA]>;
def V3AEWr_VBFMAL : SchedWriteRes<[V3AEUnitV]> { let Latency = 5; }
def V3AERd_VBFMAL : SchedReadAdvance<3, [V3AEWr_VBFMAL]>;

def V3AEWr_CRC : SchedWriteRes<[V3AEUnitM0]> { let Latency = 2; }
def V3AERd_CRC : SchedReadAdvance<1, [V3AEWr_CRC]>;

def V3AEWr_ZA  : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_ZA  : SchedReadAdvance<3, [V3AEWr_ZA]>;
def V3AEWr_ZPA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_ZPA : SchedReadAdvance<3, [V3AEWr_ZPA]>;
def V3AEWr_ZSA : SchedWriteRes<[V3AEUnitV1]> { let Latency = 4; }
def V3AERd_ZSA : SchedReadAdvance<3, [V3AEWr_ZSA]>;

def V3AEWr_ZDOTB : SchedWriteRes<[V3AEUnitV]>   { let Latency = 3; }
def V3AERd_ZDOTB : SchedReadAdvance<2, [V3AEWr_ZDOTB]>;
def V3AEWr_ZDOTH : SchedWriteRes<[V3AEUnitV0]> { let Latency = 3; }
def V3AERd_ZDOTH : SchedReadAdvance<2, [V3AEWr_ZDOTH]>;

// NOTE: SOG p. 43: Complex multiply-add B, H, S element size: How to reduce
// throughput to 1 in case of forwarding?
def V3AEWr_ZCMABHS : SchedWriteRes<[V3AEUnitV0]> { let Latency = 4; }
def V3AERd_ZCMABHS : SchedReadAdvance<3, [V3AEWr_ZCMABHS]>;
def V3AEWr_ZCMAD   : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> { let Latency = 5; }
def V3AERd_ZCMAD   : SchedReadAdvance<2, [V3AEWr_ZCMAD]>;

def V3AEWr_ZMMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 3; }
def V3AERd_ZMMA : SchedReadAdvance<2, [V3AEWr_ZMMA]>;

def V3AEWr_ZMABHS : SchedWriteRes<[V3AEUnitV0]> { let Latency = 4; }
def V3AERd_ZMABHS : SchedReadAdvance<3, [V3AEWr_ZMABHS]>;
def V3AEWr_ZMAD  : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> { let Latency = 5; }
def V3AERd_ZMAD  : SchedReadAdvance<2, [V3AEWr_ZMAD]>;

def V3AEWr_ZMAL : SchedWriteRes<[V3AEUnitV0]> { let Latency = 4; }
def V3AERd_ZMAL : SchedReadAdvance<3, [V3AEWr_ZMAL]>;

def V3AEWr_ZMASQL   : SchedWriteRes<[V3AEUnitV0]>            { let Latency = 4; }
def V3AEWr_ZMASQBHS : SchedWriteRes<[V3AEUnitV0]>            { let Latency = 4; }
def V3AEWr_ZMASQD   : SchedWriteRes<[V3AEUnitV0, V3AEUnitV0]> { let Latency = 5; }
def V3AERd_ZMASQ    : SchedReadAdvance<2, [V3AEWr_ZMASQL, V3AEWr_ZMASQBHS,
                                         V3AEWr_ZMASQD]>;

def V3AEWr_ZFCMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 5; }
def V3AERd_ZFCMA : SchedReadAdvance<3, [V3AEWr_ZFCMA]>;

def V3AEWr_ZFMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_ZFMA : SchedReadAdvance<2, [V3AEWr_ZFMA]>;

def V3AEWr_ZFMAL : SchedWriteRes<[V3AEUnitV]> { let Latency = 4; }
def V3AERd_ZFMAL : SchedReadAdvance<2, [V3AEWr_ZFMAL]>;

def V3AEWr_ZBFDOT : SchedWriteRes<[V3AEUnitV]> { let Latency = 5; }
def V3AERd_ZBFDOT : SchedReadAdvance<2, [V3AEWr_ZBFDOT]>;
def V3AEWr_ZBFMMA : SchedWriteRes<[V3AEUnitV]> { let Latency = 6; }
def V3AERd_ZBFMMA : SchedReadAdvance<2, [V3AEWr_ZBFMMA]>;
def V3AEWr_ZBFMAL : SchedWriteRes<[V3AEUnitV]> { let Latency = 5; }
def V3AERd_ZBFMAL : SchedReadAdvance<3, [V3AEWr_ZBFMAL]>;

//===----------------------------------------------------------------------===//
// Define types with long resource cycles (rc)

def V3AEWrite_6c_1V1_5rc    : SchedWriteRes<[V3AEUnitV1]>  { let Latency =  6; let ReleaseAtCycles = [ 5]; }
def V3AEWrite_9c_1V1_2rc    : SchedWriteRes<[V3AEUnitV1]>  { let Latency =  9; let ReleaseAtCycles = [ 2]; }
def V3AEWrite_9c_1V1_4rc    : SchedWriteRes<[V3AEUnitV1]>  { let Latency =  9; let ReleaseAtCycles = [ 4]; }
def V3AEWrite_10c_1V1_9rc   : SchedWriteRes<[V3AEUnitV1]>  { let Latency = 10; let ReleaseAtCycles = [ 9]; }
def V3AEWrite_11c_1V1_4rc  : SchedWriteRes<[V3AEUnitV1]> { let Latency = 11; let ReleaseAtCycles = [ 4]; }
def V3AEWrite_13c_1V1_8rc : SchedWriteRes<[V3AEUnitV1]> { let Latency = 13; let ReleaseAtCycles = [8]; }
def V3AEWrite_14c_1V1_2rc : SchedWriteRes<[V3AEUnitV1]> { let Latency = 14; let ReleaseAtCycles = [2]; }

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// §3.3 Branch instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr,    V3AEWrite_1c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, V3AEWrite_1c_1B>;

// Branch and link, immed
// Branch and link, register
def : InstRW<[V3AEWrite_1c_1B_1S], (instrs BL, BLR)>;

// §3.4 Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------

// ALU, basic
def : SchedAlias<WriteI, V3AEWrite_1c_1I>;

// ALU, basic, flagset
def : InstRW<[V3AEWrite_1c_1F_1Flg],
             (instregex "^(ADD|SUB)S[WX]r[ir]$",
                        "^(ADC|SBC)S[WX]r$",
                        "^ANDS[WX]ri$",
                        "^(AND|BIC)S[WX]rr$")>;
def : InstRW<[V3AEWrite_0or1c_1I], (instregex "^MOVZ[WX]i$")>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, V3AEWrite_2c_1M>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def : SchedAlias<WriteISReg, V3AEWrite_ArithI>;
def : InstRW<[V3AEWrite_ArithF],
             (instregex "^(ADD|SUB)S[WX]rs$")>;

// Arithmetic, immediate to logical address tag
def : InstRW<[V3AEWrite_2c_1M], (instrs ADDG, SUBG)>;

// Conditional compare
def : InstRW<[V3AEWrite_1c_1F_1Flg], (instregex "^CCM[NP][WX][ir]")>;

// Convert floating-point condition flags
// Flag manipulation instructions
def : WriteRes<WriteSys, []> { let Latency = 1; }

// Insert Random Tags
def : InstRW<[V3AEWrite_2c_1M], (instrs IRG, IRGstack)>;

// Insert Tag Mask
// Subtract Pointer
def : InstRW<[V3AEWrite_1c_1I], (instrs GMI, SUBP)>;

// Subtract Pointer, flagset
def : InstRW<[V3AEWrite_1c_1F_1Flg], (instrs SUBPS)>;

// Logical, shift, no flagset
def : InstRW<[V3AEWrite_1c_1I],    (instregex "^(AND|BIC|EON|EOR|ORN)[WX]rs$")>;
def : InstRW<[V3AEWrite_0or1c_1I], (instregex "^ORR[WX]rs$")>;

// Logical, shift, flagset
def : InstRW<[V3AEWrite_Logical], (instregex "^(AND|BIC)S[WX]rs$")>;

// Move and shift instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteImm, V3AEWrite_1c_1I>;

// §3.5 Divide and multiply instructions
// -----------------------------------------------------------------------------

// SDIV, UDIV
def : SchedAlias<WriteID32,  V3AEWrite_12c_1M0>;
def : SchedAlias<WriteID64,  V3AEWrite_20c_1M0>;

def : SchedAlias<WriteIM32, V3AEWrite_2c_1M>;
def : SchedAlias<WriteIM64, V3AEWrite_2c_1M>;

// Multiply
// Multiply accumulate, W-form
// Multiply accumulate, X-form
def : InstRW<[V3AEWr_IM], (instregex "^M(ADD|SUB)[WX]rrr$")>;

// Multiply accumulate long
// Multiply long
def : InstRW<[V3AEWr_IM], (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[V3AEWrite_3c_1M], (instrs SMULHrr, UMULHrr)>;

// §3.6 Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code, using generic key
// Compute pointer authentication code for instruction address
def : InstRW<[V3AEWrite_4c_1M0], (instregex "^AUT", "^PAC")>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[V3AEWrite_6c_1M0_1B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA,
                                            BRAAZ, BRAB, BRABZ, RETAA, RETAB,
                                            ERETAA, ERETAB)>;


// Load register, with pointer authentication
def : InstRW<[V3AEWrite_9c_1M0_1L], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[V3AEWrite_2c_1M0], (instrs XPACD, XPACI, XPACLRI)>;

// §3.7 Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Address generation
def : InstRW<[V3AEWrite_1c_1I], (instrs ADR, ADRP)>;

// Bitfield extract, one reg
// Bitfield extract, two regs
def : SchedAlias<WriteExtr, V3AEWrite_Extr>;
def : InstRW<[V3AEWrite_Extr], (instrs EXTRWrri, EXTRXrri)>;

// Bitfield move, basic
def : SchedAlias<WriteIS, V3AEWrite_1c_1I>;

// Bitfield move, insert
def : InstRW<[V3AEWrite_2c_1M], (instregex "^BFM[WX]ri$")>;

// §3.8 Load instructions
// -----------------------------------------------------------------------------

// NOTE: SOG p. 19: Throughput of LDN?P X-form should be 2, but reported as 3.

def : SchedAlias<WriteLD,    V3AEWrite_4c_1L>;
def : SchedAlias<WriteLDIdx, V3AEWrite_4c_1L>;

// Load register, literal
def : InstRW<[V3AEWrite_5c_1L_1I], (instrs LDRWl, LDRXl, LDRSWl, PRFMl)>;

// Load pair, signed immed offset, signed words
def : InstRW<[V3AEWrite_5c_1I_3L, WriteLDHi], (instrs LDPSWi)>;

// Load pair, immed post-index or immed pre-index, signed words
def : InstRW<[WriteAdr, V3AEWrite_5c_1I_3L, WriteLDHi],
             (instregex "^LDPSW(post|pre)$")>;

// §3.9 Store instructions
// -----------------------------------------------------------------------------

// NOTE: SOG, p. 20: Unsure if STRH uses pipeline I.

def : SchedAlias<WriteST,    V3AEWrite_1c_1SA_1D>;
def : SchedAlias<WriteSTIdx, V3AEWrite_1c_1SA_1D>;
def : SchedAlias<WriteSTP,   V3AEWrite_1c_1SA_1D>;
def : SchedAlias<WriteAdr,   V3AEWrite_1c_1I>;

// §3.10 Tag load instructions
// -----------------------------------------------------------------------------

// Load allocation tag
// Load multiple allocation tags
def : InstRW<[V3AEWrite_4c_1L], (instrs LDG, LDGM)>;

// §3.11 Tag store instructions
// -----------------------------------------------------------------------------

// Store allocation tags to one or two granules, post-index
// Store allocation tags to one or two granules, pre-index
// Store allocation tag to one or two granules, zeroing, post-index
// Store Allocation Tag to one or two granules, zeroing, pre-index
// Store allocation tag and reg pair to memory, post-Index
// Store allocation tag and reg pair to memory, pre-Index
def : InstRW<[V3AEWrite_1c_1SA_1D_1I], (instrs STGPreIndex, STGPostIndex,
                                                ST2GPreIndex, ST2GPostIndex,
                                                STZGPreIndex, STZGPostIndex,
                                                STZ2GPreIndex, STZ2GPostIndex,
                                                STGPpre, STGPpost)>;

// Store allocation tags to one or two granules, signed offset
// Store allocation tag to two granules, zeroing, signed offset
// Store allocation tag and reg pair to memory, signed offset
// Store multiple allocation tags
def : InstRW<[V3AEWrite_1c_1SA_1D], (instrs STGi, ST2Gi, STZGi,
                                             STZ2Gi, STGPi, STGM, STZGM)>;

// §3.12 FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF,     V3AEWrite_2c_1V>;

// FP compare
def : SchedAlias<WriteFCmp,  V3AEWrite_2c_1V0>;

// FP divide, square root
def : SchedAlias<WriteFDiv,  V3AEWrite_6c_1V1>;

// FP divide, H-form
def : InstRW<[V3AEWrite_6c_1V1],  (instrs FDIVHrr)>;
// FP divide, S-form
def : InstRW<[V3AEWrite_8c_1V1], (instrs FDIVSrr)>;
// FP divide, D-form
def : InstRW<[V3AEWrite_13c_1V1], (instrs FDIVDrr)>;

// FP square root, H-form
def : InstRW<[V3AEWrite_6c_1V1],  (instrs FSQRTHr)>;
// FP square root, S-form
def : InstRW<[V3AEWrite_8c_1V1],  (instrs FSQRTSr)>;
// FP square root, D-form
def : InstRW<[V3AEWrite_13c_1V1], (instrs FSQRTDr)>;

// FP multiply
def : WriteRes<WriteFMul, [V3AEUnitV]> { let Latency = 3; }

// FP multiply accumulate
def : InstRW<[V3AEWr_FMA, ReadDefault, ReadDefault, V3AERd_FMA],
             (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP round to integral
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                            "^FRINT(32|64)[XZ][SD]r$")>;

// §3.13 FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from gen to vec reg
def : InstRW<[V3AEWrite_3c_1M0], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
def : InstRW<[V3AEWrite_3c_1V0],
             (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri?$")>;

// FP convert, Javascript from vec to gen reg
def : SchedAlias<WriteFCvt, V3AEWrite_3c_1V0>;

// FP convert, from vec to vec reg
def : InstRW<[V3AEWrite_3c_1V], (instrs FCVTSHr, FCVTDHr, FCVTHSr, FCVTDSr,
                                        FCVTHDr, FCVTSDr, FCVTXNv1i64)>;

// FP move, immed
// FP move, register
def : SchedAlias<WriteFImm, V3AEWrite_2c_1V>;

// FP transfer, from gen to low half of vec reg
def : InstRW<[V3AEWrite_0or3c_1M0],
             (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
def : InstRW<[V3AEWrite_5c_1M0_1V], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, V3AEWrite_2c_2V>;

// §3.14 FP load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal, S/D/Q forms
def : InstRW<[V3AEWrite_7c_1I_1L], (instregex "^LDR[SDQ]l$")>;

// Load vector reg, unscaled immed
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LDUR[BHSDQ]i$")>;

// Load vector reg, immed post-index
// Load vector reg, immed pre-index
def : InstRW<[WriteAdr, V3AEWrite_6c_1I_1L],
             (instregex "^LDR[BHSDQ](pre|post)$")>;

// Load vector reg, unsigned immed
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LDR[BHSDQ]ui$")>;

// Load vector reg, register offset, basic
// Load vector reg, register offset, scale, S/D-form
// Load vector reg, register offset, scale, H/Q-form
// Load vector reg, register offset, extend
// Load vector reg, register offset, extend, scale, S/D-form
// Load vector reg, register offset, extend, scale, H/Q-form
def : InstRW<[V3AEWrite_LdrHQ, ReadAdrBase], (instregex "^LDR[BHSDQ]ro[WX]$")>;

// Load vector pair, immed offset, S/D-form
def : InstRW<[V3AEWrite_6c_1L, WriteLDHi], (instregex "^LDN?P[SD]i$")>;

// Load vector pair, immed offset, Q-form
def : InstRW<[V3AEWrite_6c_2L, WriteLDHi], (instrs LDPQi, LDNPQi)>;

// Load vector pair, immed post-index, S/D-form
// Load vector pair, immed pre-index, S/D-form
def : InstRW<[WriteAdr, V3AEWrite_6c_1I_1L, WriteLDHi],
             (instregex "^LDP[SD](pre|post)$")>;

// Load vector pair, immed post-index, Q-form
// Load vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, V3AEWrite_6c_2I_2L, WriteLDHi], (instrs LDPQpost,
                                                                LDPQpre)>;

// §3.15 FP store instructions
// -----------------------------------------------------------------------------

// Store vector reg, unscaled immed, B/H/S/D-form
// Store vector reg, unscaled immed, Q-form
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index, B/H/S/D-form
// Store vector reg, immed post-index, Q-form
// Store vector reg, immed pre-index, B/H/S/D-form
// Store vector reg, immed pre-index, Q-form
def : InstRW<[WriteAdr, V3AEWrite_2c_1SA_1V_1I],
             (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector reg, unsigned immed, B/H/S/D-form
// Store vector reg, unsigned immed, Q-form
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^STR[BHSDQ]ui$")>;

// Store vector reg, register offset, basic, B/H/S/D-form
// Store vector reg, register offset, basic, Q-form
// Store vector reg, register offset, scale, H-form
// Store vector reg, register offset, scale, S/D-form
// Store vector reg, register offset, scale, Q-form
// Store vector reg, register offset, extend, B/H/S/D-form
// Store vector reg, register offset, extend, Q-form
// Store vector reg, register offset, extend, scale, H-form
// Store vector reg, register offset, extend, scale, S/D-form
// Store vector reg, register offset, extend, scale, Q-form
def : InstRW<[V3AEWrite_StrHQ, ReadAdrBase],
             (instregex "^STR[BHSDQ]ro[WX]$")>;

// Store vector pair, immed offset, S-form
// Store vector pair, immed offset, D-form
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^STN?P[SD]i$")>;

// Store vector pair, immed offset, Q-form
def : InstRW<[V3AEWrite_2c_1SA_2V], (instrs STPQi, STNPQi)>;

// Store vector pair, immed post-index, S-form
// Store vector pair, immed post-index, D-form
// Store vector pair, immed pre-index, S-form
// Store vector pair, immed pre-index, D-form
def : InstRW<[WriteAdr, V3AEWrite_2c_1SA_1V_1I],
             (instregex "^STP[SD](pre|post)$")>;

// Store vector pair, immed post-index, Q-form
def : InstRW<[V3AEWrite_2c_1SA_2V_1I], (instrs STPQpost)>;

// Store vector pair, immed pre-index, Q-form
def : InstRW<[V3AEWrite_2c_1SA_2V_2I], (instrs STPQpre)>;

// §3.16 ASIMD integer instructions
// -----------------------------------------------------------------------------

// ASIMD absolute diff
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD arith, pair-wise
// ASIMD compare
// ASIMD logical
// ASIMD max/min, basic and pair-wise
def : SchedAlias<WriteVd, V3AEWrite_2c_1V>;
def : SchedAlias<WriteVq, V3AEWrite_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
def : InstRW<[V3AEWr_VA, V3AERd_VA], (instregex "^[SU]ABAL?v")>;

// ASIMD arith, reduce, 4H/4S
def : InstRW<[V3AEWrite_3c_1V1], (instregex "^(ADDV|[SU]ADDLV)v4(i16|i32)v$")>;

// ASIMD arith, reduce, 8B/8H
def : InstRW<[V3AEWrite_5c_1V1_1V],
             (instregex "^(ADDV|[SU]ADDLV)v8(i8|i16)v$")>;

// ASIMD arith, reduce, 16B
def : InstRW<[V3AEWrite_6c_2V1], (instregex "^(ADDV|[SU]ADDLV)v16i8v$")>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def : InstRW<[V3AEWr_VDOT, V3AERd_VDOT],
             (instregex "^([SU]|SU|US)DOT(lane)?(v8|v16)i8$")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[V3AEWr_VMMA, V3AERd_VMMA], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD max/min, reduce, 4H/4S
def : InstRW<[V3AEWrite_3c_1V1], (instregex "^[SU](MAX|MIN)Vv4i16v$",
                                            "^[SU](MAX|MIN)Vv4i32v$")>;

// ASIMD max/min, reduce, 8B/8H
def : InstRW<[V3AEWrite_5c_1V1_1V], (instregex "^[SU](MAX|MIN)Vv8i8v$",
                                               "^[SU](MAX|MIN)Vv8i16v$")>;

// ASIMD max/min, reduce, 16B
def : InstRW<[V3AEWrite_6c_2V1], (instregex "[SU](MAX|MIN)Vv16i8v$")>;

// ASIMD multiply
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^MULv", "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate
def : InstRW<[V3AEWr_VMA, V3AERd_VMA], (instregex "^MLAv", "^MLSv")>;

// ASIMD multiply accumulate high
def : InstRW<[V3AEWr_VMAH, V3AERd_VMAH], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply accumulate long
def : InstRW<[V3AEWr_VMAL, V3AERd_VMAL], (instregex "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate saturating long
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SQDML[AS]L[iv]")>;

// ASIMD multiply/multiply long (8x8) polynomial, D-form
// ASIMD multiply/multiply long (8x8) polynomial, Q-form
def : InstRW<[V3AEWrite_3c_1V], (instregex "^PMULL?(v8i8|v16i8)$")>;

// ASIMD multiply long
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^[SU]MULLv", "^SQDMULL[iv]")>;

// ASIMD pairwise add and accumulate long
def : InstRW<[V3AEWr_VPA, V3AERd_VPA], (instregex "^[SU]ADALPv")>;

// ASIMD shift accumulate
def : InstRW<[V3AEWr_VSA, V3AERd_VSA], (instregex "^[SU]SRA[dv]", "^[SU]RSRA[dv]")>;

// ASIMD shift by immed, basic
def : InstRW<[V3AEWrite_2c_1V], (instregex "^SHL[dv]", "^SHLLv", "^SHRNv",
                                           "^SSHLLv", "^SSHR[dv]", "^USHLLv",
                                           "^USHR[dv]")>;

// ASIMD shift by immed and insert, basic
def : InstRW<[V3AEWrite_2c_1V], (instregex "^SLI[dv]", "^SRI[dv]")>;

// ASIMD shift by immed, complex
def : InstRW<[V3AEWrite_4c_1V],
             (instregex "^RSHRNv", "^SQRSHRU?N[bhsv]", "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHRU?N[bhsv]", "^SRSHR[dv]", "^UQRSHRN[bhsv]",
                        "^UQSHRN[bhsv]", "^URSHR[dv]")>;

// ASIMD shift by register, basic
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU]SHLv")>;

// ASIMD shift by register, complex
def : InstRW<[V3AEWrite_4c_1V],
             (instregex "^[SU]RSHLv", "^[SU]QRSHLv",
                        "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)$")>;

// §3.17 ASIMD floating-point instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP complex add
// ASIMD FP max/min, normal
// ASIMD FP max/min, pairwise
// ASIMD FP negate
// Handled by SchedAlias<WriteV[dq], ...>

// ASIMD FP complex multiply add
def : InstRW<[V3AEWr_VFCMA, V3AERd_VFCMA], (instregex "^FCMLAv")>;

// ASIMD FP convert, long (F16 to F32)
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FCVTL(v4|v8)i16")>;

// ASIMD FP convert, long (F32 to F64)
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FCVTL(v2|v4)i32")>;

// ASIMD FP convert, narrow (F32 to F16)
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FCVTN(v4|v8)i16")>;

// ASIMD FP convert, narrow (F64 to F32)
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FCVTN(v2|v4)i32",
                                            "^FCVTXN(v2|v4)f32")>;

// ASIMD FP convert, other, D-form F32 and Q-form F64
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FCVT[AMNPZ][SU]v2f(32|64)$",
                                            "^FCVT[AMNPZ][SU]v2i(32|64)_shift$",
                                            "^FCVT[AMNPZ][SU]v1i64$",
                                            "^FCVTZ[SU]d$",
                                            "^[SU]CVTFv2f(32|64)$",
                                            "^[SU]CVTFv2i(32|64)_shift$",
                                            "^[SU]CVTFv1i64$",
                                            "^[SU]CVTFd$")>;

// ASIMD FP convert, other, D-form F16 and Q-form F32
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FCVT[AMNPZ][SU]v4f(16|32)$",
                                            "^FCVT[AMNPZ][SU]v4i(16|32)_shift$",
                                            "^FCVT[AMNPZ][SU]v1i32$",
                                            "^FCVTZ[SU]s$",
                                            "^[SU]CVTFv4f(16|32)$",
                                            "^[SU]CVTFv4i(16|32)_shift$",
                                            "^[SU]CVTFv1i32$",
                                            "^[SU]CVTFs$")>;

// ASIMD FP convert, other, Q-form F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FCVT[AMNPZ][SU]v8f16$",
                                            "^FCVT[AMNPZ][SU]v8i16_shift$",
                                            "^FCVT[AMNPZ][SU]v1f16$",
                                            "^FCVTZ[SU]h$",
                                            "^[SU]CVTFv8f16$",
                                            "^[SU]CVTFv8i16_shift$",
                                            "^[SU]CVTFv1i16$",
                                            "^[SU]CVTFh$")>;

// ASIMD FP divide, D-form, F16
def : InstRW<[V3AEWrite_9c_1V1_4rc], (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
def : InstRW<[V3AEWrite_9c_1V1_2rc], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[V3AEWrite_13c_1V1_8rc], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[V3AEWrite_11c_1V1_4rc], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[V3AEWrite_14c_1V1_2rc], (instrs FDIVv2f64)>;

// ASIMD FP max/min, reduce, F32 and D-form F16
def : InstRW<[V3AEWrite_4c_2V], (instregex "^(FMAX|FMIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
def : InstRW<[V3AEWrite_6c_3V], (instregex "^(FMAX|FMIN)(NM)?Vv8i16v$")>;

// ASIMD FP multiply
def : InstRW<[V3AEWr_VFM], (instregex "^FMULv", "^FMULXv")>;

// ASIMD FP multiply accumulate
def : InstRW<[V3AEWr_VFMA, V3AERd_VFMA], (instregex "^FMLAv", "^FMLSv")>;

// ASIMD FP multiply accumulate long
def : InstRW<[V3AEWr_VFMAL, V3AERd_VFMAL], (instregex "^FML[AS]L2?(lane)?v")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[V3AEWrite_3c_1V0],
             (instregex "^FRINT[AIMNPXZ]v2f(32|64)$",
                        "^FRINT(32|64)[XZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[V3AEWrite_4c_2V0],
             (instregex "^FRINT[AIMNPXZ]v4f(16|32)$",
                        "^FRINT(32|64)[XZ]v4f32$")>;

// ASIMD FP round, Q-form F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FRINT[AIMNPXZ]v8f16$")>;

// ASIMD FP square root, D-form, F16
def : InstRW<[V3AEWrite_9c_1V1_4rc], (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
def : InstRW<[V3AEWrite_9c_1V1_2rc], (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[V3AEWrite_13c_1V1_8rc], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
def : InstRW<[V3AEWrite_11c_1V1_4rc], (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[V3AEWrite_14c_1V1_2rc], (instrs FSQRTv2f64)>;

// §3.18 ASIMD BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
def : InstRW<[V3AEWrite_4c_2V0], (instrs BFCVTN, BFCVTN2)>;

// ASIMD dot product
def : InstRW<[V3AEWr_VBFDOT, V3AERd_VBFDOT], (instrs BFDOTv4bf16, BFDOTv8bf16)>;

// ASIMD matrix multiply accumulate
def : InstRW<[V3AEWr_VBFMMA, V3AERd_VBFMMA], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
def : InstRW<[V3AEWr_VBFMAL, V3AERd_VBFMAL], (instrs BFMLALB, BFMLALBIdx, BFMLALT,
                                                     BFMLALTIdx)>;

// Scalar convert, F32 to BF16
def : InstRW<[V3AEWrite_3c_1V0], (instrs BFCVT)>;

// §3.19 ASIMD miscellaneous instructions
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
// ASIMD table lookup extension, 1 table reg
// ASIMD transpose
// ASIMD unzip/zip
// Handled by SchedAlias<WriteV[dq], ...>
def : InstRW<[V3AEWrite_0or2c_1V], (instrs MOVID, MOVIv2d_ns)>;

// ASIMD duplicate, gen reg
def : InstRW<[V3AEWrite_3c_1M0], (instregex "^DUPv.+gpr")>;

// ASIMD extract narrow, saturating
def : InstRW<[V3AEWrite_4c_1V], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form U32
def : InstRW<[V3AEWrite_3c_1V0], (instrs URECPEv2i32, URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, Q-form U32
def : InstRW<[V3AEWrite_4c_2V0], (instrs URECPEv4i32, URSQRTEv4i32)>;

// ASIMD reciprocal and square root estimate, D-form F32 and scalar forms
def : InstRW<[V3AEWrite_3c_1V0], (instrs FRECPEv1f16, FRECPEv1i32,
                                         FRECPEv1i64, FRECPEv2f32,
                                         FRSQRTEv1f16, FRSQRTEv1i32,
                                         FRSQRTEv1i64, FRSQRTEv2f32)>;

// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
def : InstRW<[V3AEWrite_4c_2V0], (instrs FRECPEv4f16, FRECPEv4f32,
                                         FRSQRTEv4f16, FRSQRTEv4f32)>;

// ASIMD reciprocal and square root estimate, Q-form F16
def : InstRW<[V3AEWrite_6c_4V0], (instrs FRECPEv8f16, FRSQRTEv8f16)>;

// ASIMD reciprocal exponent
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FRECPXv")>;

// ASIMD reciprocal step
def : InstRW<[V3AEWrite_4c_1V], (instregex "^FRECPS(32|64|v)",
                                         "^FRSQRTS(32|64|v)")>;

// ASIMD table lookup, 1 or 2 table regs
def : InstRW<[V3AEWrite_2c_1V], (instrs TBLv8i8One, TBLv16i8One,
                                      TBLv8i8Two, TBLv16i8Two)>;

// ASIMD table lookup, 3 table regs
def : InstRW<[V3AEWrite_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// ASIMD table lookup, 4 table regs
def : InstRW<[V3AEWrite_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 2 table reg
def : InstRW<[V3AEWrite_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup extension, 3 table reg
def : InstRW<[V3AEWrite_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
def : InstRW<[V3AEWrite_6c_5V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, element to gen reg
def : InstRW<[V3AEWrite_2c_2V], (instregex "^[SU]MOVv")>;

// ASIMD transfer, gen reg to element
def : InstRW<[V3AEWrite_5c_1M0_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;

// §3.20 ASIMD load instructions
// -----------------------------------------------------------------------------

// ASIMD load, 1 element, multiple, 1 reg, D-form
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_1L],
             (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_1L],
             (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
def : InstRW<[V3AEWrite_6c_2L], (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_2L],
             (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[V3AEWrite_6c_2L], (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_2L],
             (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
def : InstRW<[V3AEWrite_6c_3L], (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_3L],
             (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[V3AEWrite_6c_3L], (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_3L],
             (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
def : InstRW<[V3AEWrite_7c_4L], (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_7c_4L],
             (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[V3AEWrite_7c_4L], (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_7c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[V3AEWrite_8c_1L_1V],           (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_1V], (instregex "LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
def : InstRW<[V3AEWrite_8c_1L_1V],           (instregex "LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_1V], (instregex "LD1Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[V3AEWrite_8c_1L_1V],           (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_1V], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_8c_1L_2V],           (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_2V], (instregex "LD2Twov(8b|4h|2s)_POST$")>;

// ASIMD load, 2 element, multiple, Q-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_8c_2L_2V],           (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_2L_2V], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[V3AEWrite_8c_1L_2V],           (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_2V], (instregex "LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
def : InstRW<[V3AEWrite_8c_1L_2V],            (instregex "LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_2V],  (instregex "LD2Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[V3AEWrite_8c_1L_2V],           (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_1L_2V], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_8c_2L_3V],           (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_2L_3V], (instregex "LD3Threev(8b|4h|2s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_8c_3L_3V],           (instregex "LD3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_3L_3V], (instregex "LD3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, one lane, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[V3AEWrite_8c_2L_3V],           (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_2L_3V], (instregex "LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
def : InstRW<[V3AEWrite_8c_2L_3V],           (instregex "LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_2L_3V], (instregex "LD3Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[V3AEWrite_8c_3L_3V],           (instregex "LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_3L_3V], (instregex "LD3Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_8c_3L_4V],           (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_3L_4V], (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_9c_6L_4V],           (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_9c_6L_4V], (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[V3AEWrite_8c_3L_4V],           (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_3L_4V], (instregex "LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
def : InstRW<[V3AEWrite_8c_3L_4V],           (instregex "LD4Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_3L_4V], (instregex "LD4Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[V3AEWrite_8c_4L_4V],           (instregex "LD4Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_8c_4L_4V], (instregex "LD4Rv(16b|8h|4s|2d)_POST$")>;

// §3.21 ASIMD store instructions
// -----------------------------------------------------------------------------

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[V3AEWrite_2c_1SA_1V],           (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_1SA_1V], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[V3AEWrite_2c_1SA_1V],           (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_1SA_1V], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[V3AEWrite_2c_1SA_1V],           (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_1SA_1V], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[V3AEWrite_2c_2SA_2V],           (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_2SA_2V], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
def : InstRW<[V3AEWrite_2c_2SA_2V],           (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_2SA_2V], (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[V3AEWrite_2c_3SA_3V],           (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_3SA_3V], (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[V3AEWrite_2c_2SA_2V],           (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_2SA_2V], (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[V3AEWrite_2c_4SA_4V],           (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_2c_4SA_4V], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[V3AEWrite_4c_1SA_2V],           (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_4c_1SA_2V], (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_4c_1SA_2V],           (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_4c_1SA_2V], (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_4c_2SA_4V],           (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_4c_2SA_4V], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[V3AEWrite_4c_1SA_2V],           (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_4c_1SA_2V], (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_5c_2SA_4V],           (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_5c_2SA_4V], (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_6c_3SA_6V],           (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_3SA_6V], (instregex "ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[V3AEWrite_5c_2SA_4V],           (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_5c_2SA_4V], (instregex "ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[V3AEWrite_6c_2SA_6V],           (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_2SA_6V], (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[V3AEWrite_7c_4SA_12V],           (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, V3AEWrite_7c_4SA_12V], (instregex "ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[V3AEWrite_5c_4SA_8V],           (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, V3AEWrite_5c_4SA_8V], (instregex "ST4Fourv(2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H/S
def : InstRW<[V3AEWrite_6c_1SA_3V],           (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, V3AEWrite_6c_1SA_3V], (instregex "ST4i(8|16|32)_POST$")>;

// ASIMD store, 4 element, one lane, D
def : InstRW<[V3AEWrite_4c_2SA_4V],            (instregex "ST4i(64)$")>;
def : InstRW<[WriteAdr, V3AEWrite_4c_2SA_4V],  (instregex "ST4i(64)_POST$")>;

// §3.22 Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[V3AEWrite_2c_1V], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[V3AEWrite_2c_1V], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
def : InstRW<[V3AEWrite_2c_1V0], (instregex "^SHA1(H|SU0|SU1)")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[V3AEWrite_2c_1V0], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[V3AEWrite_2c_1V0], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[V3AEWrite_2c_1V], (instrs BCAX, EOR3, RAX1, XAR)>;

// Crypto SM3 ops
def : InstRW<[V3AEWrite_2c_1V0], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                            "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[V3AEWrite_4c_1V0], (instrs SM4E, SM4ENCKEY)>;

// §3.23 CRC
// -----------------------------------------------------------------------------

def : InstRW<[V3AEWr_CRC, V3AERd_CRC], (instregex "^CRC32")>;

// §3.24 SVE Predicate instructions
// -----------------------------------------------------------------------------

// Loop control, based on predicate
def : InstRW<[V3AEWrite_2or3c_1M], (instrs BRKA_PPmP, BRKA_PPzP,
                                         BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[V3AEWrite_2or3c_1M], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[V3AEWrite_2or3c_1M], (instrs BRKN_PPzP, BRKPA_PPzPP,
                                         BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[V3AEWrite_2or3c_1M], (instrs BRKNS_PPzP, BRKPAS_PPzPP,
                                         BRKPBS_PPzPP)>;

// Loop control, based on GPR
def : InstRW<[V3AEWrite_3c_2M],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]")>;
def : InstRW<[V3AEWrite_3c_2M], (instregex "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Loop terminate
def : InstRW<[V3AEWrite_1c_2M], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// Predicate counting scalar
def : InstRW<[V3AEWrite_2c_1M], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;
def : InstRW<[V3AEWrite_2c_1M],
             (instregex "^(CNT|SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI",
                        "^UQ(DEC|INC)[BHWD]_WPiI")>;

// Predicate counting scalar, ALL, {1,2,4}
def : InstRW<[V3AEWrite_IncDec], (instregex "^(DEC|INC)[BHWD]_XPiI")>;

// Predicate counting scalar, active predicate
def : InstRW<[V3AEWrite_2c_1M],
             (instregex "^CNTP_XPP_[BHSD]",
                        "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;

// Predicate counting vector, active predicate
def : InstRW<[V3AEWrite_7c_1M_1M0_1V],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;

// Predicate logical
def : InstRW<[V3AEWrite_1or2c_1M],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
def : InstRW<[V3AEWrite_1or2c_1M],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
def : InstRW<[V3AEWrite_2c_1M], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
def : InstRW<[V3AEWrite_1c_1M], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[V3AEWrite_2c_1M], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
def : InstRW<[V3AEWrite_2c_1M], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
def : InstRW<[V3AEWrite_2c_1M], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;

// Predicate test
def : InstRW<[V3AEWrite_1c_1M], (instrs PTEST_PP)>;

// Predicate transpose
def : InstRW<[V3AEWrite_2c_1M], (instregex "^TRN[12]_PPP_[BHSD]")>;

// Predicate unpack and widen
def : InstRW<[V3AEWrite_2c_1M], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[V3AEWrite_2c_1M], (instregex "^(ZIP|UZP)[12]_PPP_[BHSD]")>;

// §3.25 SVE integer instructions
// -----------------------------------------------------------------------------

// Arithmetic, absolute diff
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU]ABD_ZPmZ_[BHSD]",
                                           "^[SU]ABD_ZPZZ_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[V3AEWr_ZA, V3AERd_ZA], (instregex "^[SU]ABA_ZZZ_[BHSD]")>;

// Arithmetic, absolute diff accum long
def : InstRW<[V3AEWr_ZA, V3AERd_ZA], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]")>;

// Arithmetic, absolute diff long
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]")>;

// Arithmetic, basic
def : InstRW<[V3AEWrite_2c_1V],
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
def : InstRW<[V3AEWrite_2c_1V],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^SQ(ABS|ADD|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;

// Arithmetic, large integer
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]")>;

// Arithmetic, pairwise add
def : InstRW<[V3AEWrite_2c_1V], (instregex "^ADDP_ZPmZ_[BHSD]")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[V3AEWr_ZPA, ReadDefault, V3AERd_ZPA],
             (instregex "^[SU]ADALP_ZPmZ_[HSD]")>;

// Arithmetic, shift
def : InstRW<[V3AEWrite_2c_1V1],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZ[IZ]_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[V3AEWr_ZSA, V3AERd_ZSA], (instregex "^[SU]R?SRA_ZZI_[BHSD]")>;

// Arithmetic, shift by immediate
def : InstRW<[V3AEWrite_2c_1V], (instregex "^SHRN[BT]_ZZI_[BHS]",
                                           "^[SU]SHLL[BT]_ZZI_[HSD]")>;

// Arithmetic, shift by immediate and insert
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(SLI|SRI)_ZZI_[BHSD]")>;

// Arithmetic, shift complex
def : InstRW<[V3AEWrite_4c_1V],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_ZPmZ_[BHSD]",
                        "^[SU]QR?SHL_ZPZZ_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift right for divide
def : InstRW<[V3AEWrite_4c_1V], (instregex "^ASRD_(ZPmI|ZPZI)_[BHSD]")>;

// Arithmetic, shift rounding
def : InstRW<[V3AEWrite_4c_1V], (instregex "^[SU]RSHLR?_ZPmZ_[BHSD]",
                                           "^[SU]RSHL_ZPZZ_[BHSD]",
                                           "^[SU]RSHR_(ZPmI|ZPZI)_[BHSD]")>;

// Bit manipulation
def : InstRW<[V3AEWrite_6c_2V1], (instregex "^(BDEP|BEXT|BGRP)_ZZZ_[BHSD]")>;

// Bitwise select
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ")>;

// Count/reverse bits
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(CLS|CLZ|CNT|RBIT)_ZPmZ_[BHSD]")>;

// Broadcast logical bitmask immediate to vector
def : InstRW<[V3AEWrite_2c_1V], (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[V3AEWrite_2or3c_1V0],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;

// Complex add
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(SQ)?CADD_ZZI_[BHSD]")>;

// Complex dot product 8-bit element
def : InstRW<[V3AEWr_ZDOTB, V3AERd_ZDOTB], (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[V3AEWr_ZDOTH, V3AERd_ZDOTH], (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size
def : InstRW<[V3AEWr_ZCMABHS, V3AERd_ZCMABHS], (instregex "^CMLA_ZZZ_[BHS]",
                                                          "^CMLA_ZZZI_[HS]")>;

// Complex multiply-add D element size
def : InstRW<[V3AEWr_ZCMAD, V3AERd_ZCMAD], (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form
def : InstRW<[V3AEWrite_8c_1M0_1V], (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[V3AEWrite_3c_1V1], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]",
                                            "^COMPACT_ZPZ_[SD]",
                                            "^SPLICE_ZPZZ?_[BHSD]")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                            "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 32b to single or half
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 16b to half
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[V3AEWrite_5c_1M0_1V], (instregex "^CPY_ZPmR_[BHSD]")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[V3AEWrite_2c_1V], (instregex "^CPY_ZPm[IV]_[BHSD]",
                                           "^CPY_ZPzI_[BHSD]")>;

// Divides, 32 bit
def : InstRW<[V3AEWrite_12c_1V0], (instregex "^[SU]DIVR?_ZPmZ_S",
                                             "^[SU]DIV_ZPZZ_S")>;

// Divides, 64 bit
def : InstRW<[V3AEWrite_20c_1V0], (instregex "^[SU]DIVR?_ZPmZ_D",
                                             "^[SU]DIV_ZPZZ_D")>;

// Dot product, 8 bit
def : InstRW<[V3AEWr_ZDOTB, V3AERd_ZDOTB], (instregex "^[SU]DOT_ZZZI?_BtoS")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[V3AEWr_ZDOTB, V3AERd_ZDOTB], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[V3AEWr_ZDOTH, V3AERd_ZDOTH], (instregex "^[SU]DOT_ZZZI?_HtoD")>;

// Duplicate, immediate and indexed form
def : InstRW<[V3AEWrite_2c_1V], (instregex "^DUP_ZI_[BHSD]",
                                           "^DUP_ZZI_[BHSDQ]")>;

// Duplicate, scalar form
def : InstRW<[V3AEWrite_3c_1M0], (instregex "^DUP_ZR_[BHSD]")>;

// Extend, sign or zero
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                           "^[SU]XTH_ZPmZ_[SD]",
                                           "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[V3AEWrite_2c_1V], (instrs EXT_ZZI, EXT_ZZI_CONSTRUCTIVE, EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[V3AEWrite_4c_1V], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]",
                                           "^SQXTUN[BT]_ZZ_[BHS]")>;

// Extract operation, SIMD and FP scalar form
def : InstRW<[V3AEWrite_3c_1V1], (instregex "^LAST[AB]_VPZ_[BHSD]")>;

// Extract operation, scalar
def : InstRW<[V3AEWrite_6c_1V1_1M0], (instregex "^LAST[AB]_RPZ_[BHSD]")>;

// Histogram operations
def : InstRW<[V3AEWrite_2c_1V], (instregex "^HISTCNT_ZPzZZ_[SD]",
                                           "^HISTSEG_ZZZ")>;

// Horizontal operations, B, H, S form, immediate operands only
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^INDEX_II_[BHS]")>;

// Horizontal operations, B, H, S form, scalar, immediate operands/ scalar
// operands only / immediate, scalar operands
def : InstRW<[V3AEWrite_7c_1M0_1V0], (instregex "^INDEX_(IR|RI|RR)_[BHS]")>;

// Horizontal operations, D form, immediate operands only
def : InstRW<[V3AEWrite_5c_2V0], (instrs INDEX_II_D)>;

// Horizontal operations, D form, scalar, immediate operands)/ scalar operands
// only / immediate, scalar operands
def : InstRW<[V3AEWrite_8c_2M0_2V0], (instregex "^INDEX_(IR|RI|RR)_D")>;

// insert operation, SIMD and FP scalar form
def : InstRW<[V3AEWrite_2c_1V], (instregex "^INSR_ZV_[BHSD]")>;

// insert operation, scalar
def : InstRW<[V3AEWrite_5c_1V1_1M0], (instregex "^INSR_ZR_[BHSD]")>;

// Logical
def : InstRW<[V3AEWrite_2c_1V],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|ORR)_ZZZ",
                        "^EOR(BT|TB)_ZZZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^NOT_ZPmZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                           "^[SU](MAX|MIN)P?_ZPmZ_[BHSD]",
                                           "^[SU](MAX|MIN)_ZPZZ_[BHSD]")>;

// Matching operations
// FIXME: SOG p. 44, n. 5: If the consuming instruction has a flag source, the
// latency for this instruction is 4 cycles.
def : InstRW<[V3AEWrite_2or3c_1V0_1M], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Matrix multiply-accumulate
def : InstRW<[V3AEWr_ZMMA, V3AERd_ZMMA], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[V3AEWrite_2c_1V], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]",
                                           "^MOVPRFX_ZZ")>;

// Multiply, B, H, S element size
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_[BHS]",
                                            "^MUL_ZPZZ_[BHS]",
                                            "^[SU]MULH_(ZPmZ|ZZZ)_[BHS]",
                                            "^[SU]MULH_ZPZZ_[BHS]")>;

// Multiply, D element size
def : InstRW<[V3AEWrite_5c_2V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_D",
                                           "^MUL_ZPZZ_D",
                                           "^[SU]MULH_(ZPmZ|ZZZ)_D",
                                           "^[SU]MULH_ZPZZ_D")>;

// Multiply long
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^[SU]MULL[BT]_ZZZI_[SD]",
                                            "^[SU]MULL[BT]_ZZZ_[HSD]")>;

// Multiply accumulate, B, H, S element size
def : InstRW<[V3AEWr_ZMABHS, V3AERd_ZMABHS],
             (instregex "^ML[AS]_ZZZI_[HS]", "^ML[AS]_ZPZZZ_[BHS]")>;
def : InstRW<[V3AEWr_ZMABHS, ReadDefault, V3AERd_ZMABHS],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]")>;

// Multiply accumulate, D element size
def : InstRW<[V3AEWr_ZMAD, V3AERd_ZMAD],
             (instregex "^ML[AS]_ZZZI_D", "^ML[AS]_ZPZZZ_D")>;
def : InstRW<[V3AEWr_ZMAD, ReadDefault, V3AERd_ZMAD],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_D")>;

// Multiply accumulate long
def : InstRW<[V3AEWr_ZMAL, V3AERd_ZMAL], (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]",
                                                    "^[SU]ML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[V3AEWr_ZMASQL, V3AERd_ZMASQ],
             (instregex "^SQDML[AS]L(B|T|BT)_ZZZ_[HSD]",
                        "^SQDML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply saturating doubling high, B, H, S element size
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SQDMULH_ZZZ_[BHS]",
                                            "^SQDMULH_ZZZI_[HS]")>;

// Multiply saturating doubling high, D element size
def : InstRW<[V3AEWrite_5c_2V0], (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SQDMULL[BT]_ZZZ_[HSD]",
                                            "^SQDMULL[BT]_ZZZI_[SD]")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S
// element size
def : InstRW<[V3AEWr_ZMASQBHS, V3AERd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZ_[BHS]",
                                                         "^SQRDCMLAH_ZZZ_[BHS]",
                                                         "^SQRDML[AS]H_ZZZI_[HS]",
                                                         "^SQRDCMLAH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element
// size
def : InstRW<[V3AEWr_ZMASQD, V3AERd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZI?_D",
                                                   "^SQRDCMLAH_ZZZ_D")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SQRDMULH_ZZZ_[BHS]",
                                            "^SQRDMULH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex, D element size
def : InstRW<[V3AEWrite_5c_2V0], (instregex "^SQRDMULH_ZZZI?_D")>;

// Multiply/multiply long, (8x8) polynomial
def : InstRW<[V3AEWrite_2c_1V], (instregex "^PMUL_ZZZ_B",
                                           "^PMULL[BT]_ZZZ_[HDQ]")>;

// Predicate counting vector
def : InstRW<[V3AEWrite_2c_1V], (instregex "^([SU]Q)?(DEC|INC)[HWD]_ZPiI")>;

// Reciprocal estimate
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[V3AEWrite_9c_2V_4V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_B")>;

// Reduction, arithmetic, H form
def : InstRW<[V3AEWrite_8c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[V3AEWrite_6c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[V3AEWrite_4c_2V], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[V3AEWrite_6c_1V_1V1], (instregex "^(AND|EOR|OR)V_VPZ_[BHSD]")>;

// Reverse, vector
def : InstRW<[V3AEWrite_2c_1V], (instregex "^REV_ZZ_[BHSD]",
                                           "^REVB_ZPmZ_[HSD]",
                                           "^REVH_ZPmZ_[SD]",
                                           "^REVW_ZPmZ_D")>;

// Select, vector form
def : InstRW<[V3AEWrite_2c_1V], (instregex "^SEL_ZPZZ_[BHSD]")>;

// Table lookup
def : InstRW<[V3AEWrite_2c_1V], (instregex "^TBL_ZZZZ?_[BHSD]")>;

// Table lookup extension
def : InstRW<[V3AEWrite_2c_1V], (instregex "^TBX_ZZZ_[BHSD]")>;

// Transpose, vector form
def : InstRW<[V3AEWrite_2c_1V], (instregex "^TRN[12]_ZZZ_[BHSDQ]")>;

// Unpack and extend
def : InstRW<[V3AEWrite_2c_1V], (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]")>;

// Zip/unzip
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]")>;

// §3.26 SVE floating-point instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[V3AEWrite_2c_1V], (instregex "^FAB[SD]_ZPmZ_[HSD]",
                                           "^FABD_ZPZZ_[HSD]",
                                           "^FABS_ZPmZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[V3AEWrite_2c_1V], (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ)_[HSD]",
                                           "^F(ADD|SUB)_ZPZ[IZ]_[HSD]",
                                           "^FADDP_ZPmZZ_[HSD]",
                                           "^FNEG_ZPmZ_[HSD]",
                                           "^FSUBR_ZPm[IZ]_[HSD]",
                                           "^FSUBR_(ZPZI|ZPZZ)_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[V3AEWrite_10c_1V1_9rc], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[V3AEWrite_6c_1V1_5rc], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[V3AEWrite_4c_1V], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[V3AEWrite_2c_1V0], (instregex "^FACG[ET]_PPzZZ_[HSD]",
                                            "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]",
                                            "^FCM(LE|LT)_PPzZ0_[HSD]",
                                            "^FCMUO_PPzZZ_[HSD]")>;

// Floating point complex add
def : InstRW<[V3AEWrite_3c_1V], (instregex "^FCADD_ZPmZ_[HSD]")>;

// Floating point complex multiply add
def : InstRW<[V3AEWr_ZFCMA, ReadDefault, V3AERd_ZFCMA], (instregex "^FCMLA_ZPmZZ_[HSD]")>;
def : InstRW<[V3AEWr_ZFCMA, V3AERd_ZFCMA],              (instregex "^FCMLA_ZZZI_[HS]")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                            "^FCVTLT_ZPmZ_HtoS",
                                            "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32
// or F64 to F16)
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                            "^FCVTLT_ZPmZ_StoD",
                                            "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[V3AEWrite_3c_1V0], (instrs FCVTX_ZPmZ_DtoS, FCVTXNT_ZPmZ_DtoS)>;

// Floating point base2 log, F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[V3AEWrite_3c_1V0],
             (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[V3AEWrite_2c_1V], (instregex "^FCPY_ZPmI_[HSD]",
                                           "^FDUP_ZI_[HSD]")>;

// Floating point divide, F16
def : InstRW<[V3AEWrite_13c_1V1_8rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[V3AEWrite_11c_1V1_4rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[V3AEWrite_14c_1V1_2rc], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise
def : InstRW<[V3AEWrite_2c_1V], (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[V3AEWrite_2c_1V], (instregex "^F(MAX|MIN)(NM)?_ZPm[IZ]_[HSD]",
                                           "^F(MAX|MIN)(NM)?_ZPZ[IZ]_[HSD]")>;

// Floating point multiply
def : InstRW<[V3AEWrite_3c_1V], (instregex "^(FSCALE|FMULX)_ZPmZ_[HSD]",
                                           "^FMULX_ZPZZ_[HSD]",
                                           "^FMUL_(ZPm[IZ]|ZZZI?)_[HSD]",
                                           "^FMUL_ZPZ[IZ]_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[V3AEWr_ZFMA, ReadDefault, V3AERd_ZFMA],
             (instregex "^FN?ML[AS]_ZPmZZ_[HSD]",
                        "^FN?(MAD|MSB)_ZPmZZ_[HSD]")>;
def : InstRW<[V3AEWr_ZFMA, V3AERd_ZFMA],
             (instregex "^FML[AS]_ZZZI_[HSD]",
                        "^FN?ML[AS]_ZPZZZ_[HSD]")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[V3AEWr_ZFMAL, V3AERd_ZFMAL], (instregex "^FML[AS]L[BT]_ZZZI?_SHH")>;

// Floating point reciprocal estimate, F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FR(ECP|SQRT)E_ZZ_H", "^FRECPX_ZPmZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FR(ECP|SQRT)E_ZZ_S", "^FRECPX_ZPmZ_S")>;

// Floating point reciprocal estimate, F64
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FR(ECP|SQRT)E_ZZ_D", "^FRECPX_ZPmZ_D")>;

// Floating point reciprocal step
def : InstRW<[V3AEWrite_4c_1V], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]")>;

// Floating point reduction, F16
def : InstRW<[V3AEWrite_8c_4V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_H")>;

// Floating point reduction, F32
def : InstRW<[V3AEWrite_6c_3V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_S")>;

// Floating point reduction, F64
def : InstRW<[V3AEWrite_4c_2V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_D")>;

// Floating point round to integral, F16
def : InstRW<[V3AEWrite_6c_4V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[V3AEWrite_4c_2V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[V3AEWrite_3c_1V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[V3AEWrite_13c_1V1_8rc], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[V3AEWrite_11c_1V1_4rc], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64
def : InstRW<[V3AEWrite_14c_1V1_2rc], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[V3AEWrite_3c_1V1], (instregex "^FEXPA_ZZ_[HSD]")>;

// Floating point trigonometric multiply add
def : InstRW<[V3AEWrite_4c_1V], (instregex "^FTMAD_ZZI_[HSD]")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[V3AEWrite_3c_1V], (instregex "^FTS(MUL|SEL)_ZZZ_[HSD]")>;

// §3.27 SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[V3AEWrite_4c_1V], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[V3AEWr_ZBFDOT, V3AERd_ZBFDOT], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[V3AEWr_ZBFMMA, V3AERd_ZBFMMA], (instrs BFMMLA_ZZZ_HtoS)>;

// Multiply accumulate long
def : InstRW<[V3AEWr_ZBFMAL, V3AERd_ZBFMAL], (instregex "^BFMLAL[BT]_ZZZI?")>;

// §3.28 SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[V3AEWrite_6c_1L], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[V3AEWrite_6c_1L_1M], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1[BHWD]_IMM$",
                                           "^LD1S?B_[HSD]_IMM$",
                                           "^LD1S?H_[SD]_IMM$",
                                           "^LD1S?W_D_IMM$" )>;
// Contiguous load, scalar + scalar
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1[BHWD]$",
                                           "^LD1S?B_[HSD]$",
                                           "^LD1S?H_[SD]$",
                                           "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1R[BHWD]_IMM$",
                                           "^LD1RS?B_[HSD]_IMM$",
                                           "^LD1RS?H_[SD]_IMM$",
                                           "^LD1RW_D_IMM$",
                                           "^LD1RSW_IMM$",
                                           "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
// Non temporal load, scalar + scalar
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LDNT1[BHWD]_ZR[IR]$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[V3AEWrite_9c_2L_4V], (instregex "^LDNT1[BHW]_ZZR_S$",
                                              "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[V3AEWrite_9c_2L_2V], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[V3AEWrite_9c_2L_2V], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[V3AEWrite_6c_1L_1I], (instregex "^LDFF1[BHWD]$",
                                              "^LDFF1S?B_[HSD]$",
                                              "^LDFF1S?H_[SD]$",
                                              "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[V3AEWrite_6c_1L], (instregex "^LDNF1[BHWD]_IMM$",
                                           "^LDNF1S?B_[HSD]_IMM$",
                                           "^LDNF1S?H_[SD]_IMM$",
                                           "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[V3AEWrite_8c_2L_2V], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[V3AEWrite_9c_2L_2V_2I], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[V3AEWrite_9c_3L_3V], (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[V3AEWrite_10c_3V_3L_3I], (instregex "^LD3[BHWD]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[V3AEWrite_9c_4L_8V], (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[V3AEWrite_10c_4L_8V_4I], (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[V3AEWrite_9c_1L_4V], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                              "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[V3AEWrite_9c_1L_4V], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                              "^GLD(FF)?1D_IMM$")>;

// Gather load, 32-bit scaled offset
def : InstRW<[V3AEWrite_10c_1L_8V],
             (instregex "^GLD(FF)?1S?H_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;

// Gather load, 64-bit scaled offset
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[V3AEWrite_10c_1L_4V],
             (instregex "^GLD(FF)?1S?[HW]_D_([SU]XTW_)?SCALED$",
                        "^GLD(FF)?1D_([SU]XTW_)?SCALED$")>;

// Gather load, 32-bit unpacked unscaled offset
def : InstRW<[V3AEWrite_9c_1L_4V], (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                                            "^GLD(FF)?1W_[SU]XTW$")>;

// Gather load, 64-bit unpacked unscaled offset
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[V3AEWrite_9c_1L_2V],
             (instregex "^GLD(FF)?1S?[BHW]_D(_[SU]XTW)?$",
                        "^GLD(FF)?1D(_[SU]XTW)?$")>;

// §3.29 SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[V3AEWrite_1c_1SA], (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[V3AEWrite_2c_1SA_1V], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^ST1[BHWD]_IMM$",
                                               "^ST1B_[HSD]_IMM$",
                                               "^ST1H_[SD]_IMM$",
                                               "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[V3AEWrite_2c_1SA_1I_1V], (instregex "^ST1H(_[SD])?$")>;
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^ST1[BWD]$",
                                               "^ST1B_[HSD]$",
                                               "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[V3AEWrite_4c_1SA_1V], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[V3AEWrite_4c_2SA_2I_2V], (instrs ST2H)>;
def : InstRW<[V3AEWrite_4c_2SA_2V], (instregex "^ST2[BWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[V3AEWrite_7c_9SA_9V], (instregex "^ST3[BHWD]_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[V3AEWrite_7c_9SA_9I_9V], (instregex "^ST3[BHWD]$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[V3AEWrite_11c_18SA_18V], (instregex "^ST4[BHWD]_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[V3AEWrite_11c_18SA_18I_18V], (instregex "^ST4[BHWD]$")>;

// Non temporal store, scalar + imm
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[V3AEWrite_2c_1SA_1I_1V], (instrs STNT1H_ZRR)>;
def : InstRW<[V3AEWrite_2c_1SA_1V], (instregex "^STNT1[BWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[V3AEWrite_4c_4SA_4V], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[V3AEWrite_4c_4SA_4V], (instregex "^SST1[BH]_S_IMM$",
                                               "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^SST1[BHW]_D_IMM$",
                                               "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[V3AEWrite_4c_4SA_4V],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                               "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                               "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[V3AEWrite_4c_4SA_4V], (instregex "^SST1[BH]_S_[SU]XTW$",
                                               "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^SST1[HW]_D_SCALED$",
                                               "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[V3AEWrite_2c_2SA_2V], (instregex "^SST1[BHW]_D$",
                                               "^SST1D$")>;

// §3.30 SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[V3AEWrite_2c_1M0], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[V3AEWrite_3or4c_1M0_1M], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[V3AEWrite_3or4c_1M0_1M], (instrs RDFFRS_PPz)>;

// Set first fault register
// Write to first fault register
def : InstRW<[V3AEWrite_2c_1M0], (instrs SETFFR, WRFFR)>;

// Prefetch
// NOTE: This is not specified in the SOG.
def : InstRW<[V3AEWrite_4c_1L], (instregex "^PRF[BHWD]")>;

// §3.31 SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[V3AEWrite_2c_1V], (instregex "^AES[DE]_ZZZ_B$",
                                         "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[V3AEWrite_2c_1V], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                         "^RAX1_ZZZ_D$",
                                         "^XAR_ZZZI_[BHSD]$")>;

// Crypto SM4 ops
def : InstRW<[V3AEWrite_4c_1V0], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

}
