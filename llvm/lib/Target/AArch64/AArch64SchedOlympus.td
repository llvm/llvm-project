//=- AArch64SchedOlympus.td - Olympus Scheduling Defs --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the NVIDIA Olympus processors.
// All information is taken from the Olympus Software Optimisation guide:
//
// https://docs.nvidia.com/olympus-cpu-core-software-optimization-guide-dp12531-001v0-7.pdf
//
//===----------------------------------------------------------------------===//

def OlympusModel : SchedMachineModel {
  // NOTE: Unless otherwise stated, values unspecified in the Olympus SWOG are
  // copied from the Neoverse V2 model.
  let IssueWidth            =  10; // Maximum macro-ops dispatched per cycle.
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasCPA, HasCSSC,
                                                     HasMatMulFP64]);
}

//===----------------------------------------------------------------------===//
// In the Olympus core, instructions are first fetched and decoded into
// internal macro-ops (MOps). Those MOps are then renamed and dispatched to the
// out-of-order portion of the core. A MOp can be split into two micro-ops
// (uOps) further down the pipeline after instruction decode. Once dispatched,
// a uOp waits for its operands to become available and issues out-of-order to
// one of many execution pipelines. Each execution pipeline can accept one uOp
// per cycle.

let SchedModel = OlympusModel in {

// Note: The RCU is not specified in the SWOG, therefore we assume we can commit
// as many MOps as we can dispatch each cycle.
def OlympusRCU : RetireControlUnit<OlympusModel.MicroOpBufferSize, 10>;

// Define the issue ports.
def OlympusUnitB   : ProcResource<4>;  // Branch 0/1/2/3
def OlympusUnitS   : ProcResource<6>;  // Integer single-cycle 0/1/2/3/4/5
def OlympusUnitM0  : ProcResource<1>;  // Integer single/multicycle 0
def OlympusUnitM1  : ProcResource<1>;  // Integer single/multicycle 1
def OlympusUnitL   : ProcResource<4>;  // Load 0/1/2/3
def OlympusUnitSA  : ProcResource<2>;  // Store 0/1
def OlympusUnitD   : ProcResource<2>;  // Integer store data 0/1
def OlympusUnitV0  : ProcResource<1>;  // FP/ASIMD 0
def OlympusUnitV1  : ProcResource<1>;  // FP/ASIMD 1
def OlympusUnitV2  : ProcResource<1>;  // FP/ASIMD 2
def OlympusUnitV3  : ProcResource<1>;  // FP/ASIMD 3
def OlympusUnitV45 : ProcResource<2>;  // FP/ASIMD 4/5
def OlympusUnitF   : ProcResource<6>;  // Flags

def OlympusUnitM     : ProcResGroup<[OlympusUnitM0, OlympusUnitM1]>;                                               // Integer single/multicycle 0/1
def OlympusUnitI     : ProcResGroup<[OlympusUnitS, OlympusUnitM0, OlympusUnitM1]>;                                 // Integer single-cycle 0/1/2/3/4/5 and single/multicycle 0/1
def OlympusUnitV03   : ProcResGroup<[OlympusUnitV0, OlympusUnitV3]>;                                               // FP/ASIMD 0/3
def OlympusUnitV12   : ProcResGroup<[OlympusUnitV1, OlympusUnitV2]>;                                               // FP/ASIMD 1/2
def OlympusUnitV0123 : ProcResGroup<[OlympusUnitV0, OlympusUnitV1, OlympusUnitV2, OlympusUnitV3]>;                 // FP/ASIMD 0/1/2/3 (also used for vector store data)
def OlympusUnitV     : ProcResGroup<[OlympusUnitV0, OlympusUnitV1, OlympusUnitV2, OlympusUnitV3, OlympusUnitV45]>; // FP/ASIMD 0/1/2/3/4/5

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
// Define customized scheduler read/write types specific to the Olympus.

// Define generic 0 micro-op types

let NumMicroOps = 0 in {
  def OlympusWrite_0c : SchedWriteRes<[]> { let Latency = 0; }
  def OlympusWrite_6c : SchedWriteRes<[]> { let Latency = 6; }
} // NumMicroOps = 0

// Define generic 1 micro-op types

def OlympusWrite_1c_1B     : SchedWriteRes<[OlympusUnitB]>     { let Latency = 1; }
def OlympusWrite_1c_1I     : SchedWriteRes<[OlympusUnitI]>     { let Latency = 1; }
def OlympusWrite_1c_1M     : SchedWriteRes<[OlympusUnitM]>     { let Latency = 1; }
def OlympusWrite_1c_1M0    : SchedWriteRes<[OlympusUnitM0]>    { let Latency = 1; }
def OlympusWrite_1c_1L     : SchedWriteRes<[OlympusUnitL]>     { let Latency = 1; }
def OlympusWrite_2c_1M     : SchedWriteRes<[OlympusUnitM]>     { let Latency = 2; }
def OlympusWrite_2c_1M0    : SchedWriteRes<[OlympusUnitM0]>    { let Latency = 2; }
def OlympusWrite_2c_1V     : SchedWriteRes<[OlympusUnitV]>     { let Latency = 2; }
def OlympusWrite_2c_1V0    : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 2; }
def OlympusWrite_2c_1V0123 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 2; }
def OlympusWrite_2c_1V03   : SchedWriteRes<[OlympusUnitV03]>   { let Latency = 2; }
def OlympusWrite_2c_1V1    : SchedWriteRes<[OlympusUnitV1]>    { let Latency = 2; }
def OlympusWrite_3c_1M     : SchedWriteRes<[OlympusUnitM]>     { let Latency = 3; }
def OlympusWrite_3c_1M0    : SchedWriteRes<[OlympusUnitM0]>    { let Latency = 3; }
def OlympusWrite_3c_1V     : SchedWriteRes<[OlympusUnitV]>     { let Latency = 3; }
def OlympusWrite_3c_1V0    : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 3; }
def OlympusWrite_3c_1V0123 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 3; }
def OlympusWrite_3c_1V03   : SchedWriteRes<[OlympusUnitV03]>   { let Latency = 3; }
def OlympusWrite_3c_1V1    : SchedWriteRes<[OlympusUnitV1]>    { let Latency = 3; }
def OlympusWrite_3c_1V12   : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 3; }
def OlympusWrite_4c_1M0    : SchedWriteRes<[OlympusUnitM0]>    { let Latency = 4; }
def OlympusWrite_4c_1L     : SchedWriteRes<[OlympusUnitL]>     { let Latency = 4; }
def OlympusWrite_4c_1V     : SchedWriteRes<[OlympusUnitV]>     { let Latency = 4; }
def OlympusWrite_4c_1V0    : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 4; }
def OlympusWrite_4c_1V0123 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusWrite_4c_1V03   : SchedWriteRes<[OlympusUnitV03]>   { let Latency = 4; }
def OlympusWrite_4c_1V1    : SchedWriteRes<[OlympusUnitV1]>    { let Latency = 4; }
def OlympusWrite_5c_1V     : SchedWriteRes<[OlympusUnitV]>     { let Latency = 5; }
def OlympusWrite_6c_1L     : SchedWriteRes<[OlympusUnitL]>     { let Latency = 6; }
def OlympusWrite_6c_1V     : SchedWriteRes<[OlympusUnitV]>     { let Latency = 6; }
def OlympusWrite_6c_1V1    : SchedWriteRes<[OlympusUnitV1]>    { let Latency = 6; }
def OlympusWrite_6c_1V12   : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 6; }
def OlympusWrite_7c_1V0    : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 7; let ReleaseAtCycles = [7]; }
def OlympusWrite_8c_1V12   : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 8; }
def OlympusWrite_9c_1V0    : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 9; }
def OlympusWrite_10c_1V0   : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 10; }
def OlympusWrite_10c_1V1   : SchedWriteRes<[OlympusUnitV1]>    { let Latency = 10; }
def OlympusWrite_12c_1M    : SchedWriteRes<[OlympusUnitM]>     { let Latency = 12; let ReleaseAtCycles = [12]; }
def OlympusWrite_12c_1V    : SchedWriteRes<[OlympusUnitV]>     { let Latency = 12; }
def OlympusWrite_12c_1V45  : SchedWriteRes<[OlympusUnitV45]>   { let Latency = 12; let ReleaseAtCycles = [12]; }
def OlympusWrite_13c_1V0   : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 13; }
def OlympusWrite_13c_1V12  : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 13; }
def OlympusWrite_15c_1V0   : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 15; }
def OlympusWrite_16c_1V0   : SchedWriteRes<[OlympusUnitV0]>    { let Latency = 16; }
def OlympusWrite_20c_1M    : SchedWriteRes<[OlympusUnitM]>     { let Latency = 20; let ReleaseAtCycles = [20]; }
def OlympusWrite_20c_1V45  : SchedWriteRes<[OlympusUnitV45]>   { let Latency = 20; let ReleaseAtCycles = [20]; }

// These types are multi-pumped.
def OlympusWrite_4c_1V0123_2 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def OlympusWrite_5c_1V0123_2 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 5; let ReleaseAtCycles = [2]; }
def OlympusWrite_6c_1V0123_4 : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 6; let ReleaseAtCycles = [4]; }
def OlympusWrite_7c_1V03_6   : SchedWriteRes<[OlympusUnitV03]>   { let Latency = 7; let ReleaseAtCycles = [6]; }
def OlympusWrite_9c_1V12_2   : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 9; let ReleaseAtCycles = [2]; }
def OlympusWrite_9c_1V12_4   : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 9; let ReleaseAtCycles = [4]; }
def OlympusWrite_11c_1V03_10 : SchedWriteRes<[OlympusUnitV03]>   { let Latency = 11; let ReleaseAtCycles = [10]; }
def OlympusWrite_11c_1V12_4  : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 11; let ReleaseAtCycles = [4]; }
def OlympusWrite_13c_1V12_8  : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 13; let ReleaseAtCycles = [8]; }
def OlympusWrite_14c_1V12_2  : SchedWriteRes<[OlympusUnitV12]>   { let Latency = 14; let ReleaseAtCycles = [2]; }

// Hack to get the flagset throughputs right.
def OlympusWrite_1c_1F : SchedWriteRes<[OlympusUnitI, OlympusUnitF]> { let Latency = 1; }

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

let NumMicroOps = 2 in {

  def OlympusWrite_1c_1I_1B      : SchedWriteRes<[OlympusUnitI, OlympusUnitB]>         { let Latency = 1; }
  def OlympusWrite_1c_1M0_1M     : SchedWriteRes<[OlympusUnitM0, OlympusUnitM]>        { let Latency = 1; }
  def OlympusWrite_1c_1SA_1D     : SchedWriteRes<[OlympusUnitSA, OlympusUnitD]>        { let Latency = 1; }
  def OlympusWrite_1c_2M         : SchedWriteRes<[OlympusUnitM, OlympusUnitM]>         { let Latency = 1; }
  def OlympusWrite_2c_1L_1V      : SchedWriteRes<[OlympusUnitL, OlympusUnitV]>         { let Latency = 2; }
  def OlympusWrite_2c_1M0_1M     : SchedWriteRes<[OlympusUnitM0, OlympusUnitM]>        { let Latency = 2; }
  def OlympusWrite_2c_1M_1V03    : SchedWriteRes<[OlympusUnitV03, OlympusUnitM]>       { let Latency = 2; }
  def OlympusWrite_2c_1V0_1M     : SchedWriteRes<[OlympusUnitV0, OlympusUnitM]>        { let Latency = 2; }
  def OlympusWrite_3c_1I_1M      : SchedWriteRes<[OlympusUnitI, OlympusUnitM]>         { let Latency = 3; }
  def OlympusWrite_3c_1M_1M0     : SchedWriteRes<[OlympusUnitM, OlympusUnitM0]>        { let Latency = 3; }
  def OlympusWrite_3c_1M_1V03    : SchedWriteRes<[OlympusUnitV03, OlympusUnitM]>       { let Latency = 3; }
  def OlympusWrite_3c_1SA_1V0123 : SchedWriteRes<[OlympusUnitSA, OlympusUnitV0123]>    { let Latency = 3; }
  def OlympusWrite_3c_1V0_1M     : SchedWriteRes<[OlympusUnitV0, OlympusUnitM]>        { let Latency = 3; }
  def OlympusWrite_3c_2V03       : SchedWriteRes<[OlympusUnitV03, OlympusUnitV03]>     { let Latency = 3; }
  def OlympusWrite_3c_2M         : SchedWriteRes<[OlympusUnitM, OlympusUnitM]>         { let Latency = 3; }
  def OlympusWrite_4c_1L_1V      : SchedWriteRes<[OlympusUnitL, OlympusUnitV]>         { let Latency = 4; }
  def OlympusWrite_4c_1M_1M0     : SchedWriteRes<[OlympusUnitM, OlympusUnitM0]>        { let Latency = 4; }
  def OlympusWrite_4c_1SA_1D     : SchedWriteRes<[OlympusUnitSA, OlympusUnitD]>        { let Latency = 4; }
  def OlympusWrite_4c_2M         : SchedWriteRes<[OlympusUnitM, OlympusUnitM]>         { let Latency = 4; }
  def OlympusWrite_4c_2V         : SchedWriteRes<[OlympusUnitV, OlympusUnitV]>         { let Latency = 4; }
  def OlympusWrite_4c_2V0        : SchedWriteRes<[OlympusUnitV0, OlympusUnitV0]>       { let Latency = 4; }
  def OlympusWrite_5c_1B_1M0     : SchedWriteRes<[OlympusUnitB, OlympusUnitM0]>        { let Latency = 5; }
  def OlympusWrite_5c_1I_1L      : SchedWriteRes<[OlympusUnitI, OlympusUnitL]>         { let Latency = 5; }
  def OlympusWrite_5c_1L_1F      : SchedWriteRes<[OlympusUnitL, OlympusUnitF]>         { let Latency = 5; }
  def OlympusWrite_5c_1M0_1V     : SchedWriteRes<[OlympusUnitM0, OlympusUnitV]>        { let Latency = 5; }
  def OlympusWrite_5c_1M_1L      : SchedWriteRes<[OlympusUnitM, OlympusUnitL]>         { let Latency = 5; }
  def OlympusWrite_5c_1M_1V      : SchedWriteRes<[OlympusUnitM, OlympusUnitV]>         { let Latency = 5; }
  def OlympusWrite_5c_2V         : SchedWriteRes<[OlympusUnitV, OlympusUnitV]>         { let Latency = 5; }
  def OlympusWrite_5c_2V0        : SchedWriteRes<[OlympusUnitV0, OlympusUnitV0]>       { let Latency = 5; }
  def OlympusWrite_5c_1V_1V0123  : SchedWriteRes<[OlympusUnitV, OlympusUnitV0123]>     { let Latency = 5; }
  def OlympusWrite_6c_1I_1L      : SchedWriteRes<[OlympusUnitI, OlympusUnitL]>         { let Latency = 6; }
  def OlympusWrite_6c_1L_1S      : SchedWriteRes<[OlympusUnitL, OlympusUnitS]>         { let Latency = 6; }
  def OlympusWrite_6c_1V03_1V12  : SchedWriteRes<[OlympusUnitV03, OlympusUnitV12]>     { let Latency = 6; }
  def OlympusWrite_6c_1V1_1M0    : SchedWriteRes<[OlympusUnitV1, OlympusUnitM0]>       { let Latency = 6; }
  def OlympusWrite_6c_1V_1V0123  : SchedWriteRes<[OlympusUnitV, OlympusUnitV0123]>     { let Latency = 6; }
  def OlympusWrite_6c_2L         : SchedWriteRes<[OlympusUnitL, OlympusUnitL]>         { let Latency = 6; }
  def OlympusWrite_6c_2V         : SchedWriteRes<[OlympusUnitV, OlympusUnitV]>         { let Latency = 6; }
  def OlympusWrite_6c_2V12       : SchedWriteRes<[OlympusUnitV12, OlympusUnitV12]>     { let Latency = 6; }
  def OlympusWrite_6c_2V0123     : SchedWriteRes<[OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 6; }
  def OlympusWrite_7c_1F_1L      : SchedWriteRes<[OlympusUnitF, OlympusUnitL]>         { let Latency = 7; }
  def OlympusWrite_7c_1I_1L      : SchedWriteRes<[OlympusUnitI, OlympusUnitL]>         { let Latency = 7; }
  def OlympusWrite_7c_1M_1V0123  : SchedWriteRes<[OlympusUnitM, OlympusUnitV0123]>     { let Latency = 7; }
  def OlympusWrite_8c_1L_1V      : SchedWriteRes<[OlympusUnitL, OlympusUnitV]>         { let Latency = 8; }
  def OlympusWrite_8c_1M0_1L     : SchedWriteRes<[OlympusUnitM0, OlympusUnitL]>        { let Latency = 8; }

  // These types are multi-pumped.
  def OlympusWrite_8c_1M_1V0123_2 : SchedWriteRes<[OlympusUnitM, OlympusUnitV0123]> { let Latency = 8; let ReleaseAtCycles = [1, 2]; }

}  // NumMicroOps = 2

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

let NumMicroOps = 3 in {

  def OlympusWrite_2c_1L_1S_1V      : SchedWriteRes<[OlympusUnitL, OlympusUnitS, OlympusUnitV]>         { let Latency = 2; }
  def OlympusWrite_2c_1SA_1D_1I     : SchedWriteRes<[OlympusUnitSA, OlympusUnitD, OlympusUnitI]>        { let Latency = 2; }
  def OlympusWrite_3c_1I_1SA_1V0123 : SchedWriteRes<[OlympusUnitI, OlympusUnitSA, OlympusUnitV0123]>    { let Latency = 3; }
  def OlympusWrite_4c_3V            : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV]>         { let Latency = 4; }
  def OlympusWrite_5c_1I_1B_1M0     : SchedWriteRes<[OlympusUnitI, OlympusUnitB, OlympusUnitM0]>        { let Latency = 5; }
  def OlympusWrite_5c_1SA_1V_1V0123 : SchedWriteRes<[OlympusUnitSA, OlympusUnitV, OlympusUnitV0123]>    { let Latency = 5; }
  def OlympusWrite_6c_3L            : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL]>         { let Latency = 6; }
  def OlympusWrite_6c_3V            : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV]>         { let Latency = 6; }
  def OlympusWrite_7c_2M_1V         : SchedWriteRes<[OlympusUnitM, OlympusUnitM, OlympusUnitV]>         { let Latency = 7; }
  def OlympusWrite_8c_1L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitV, OlympusUnitV]>         { let Latency = 8; }
  def OlympusWrite_8c_1M_1V03_1V12  : SchedWriteRes<[OlympusUnitM, OlympusUnitV03, OlympusUnitV12]>     { let Latency = 8; }
  def OlympusWrite_8c_2V_1V0123     : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV0123]>     { let Latency = 8; }
  def OlympusWrite_9c_1L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitV, OlympusUnitV]>         { let Latency = 9; }
  def OlympusWrite_9c_2L_1V03       : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV03]>       { let Latency = 9; }
  def OlympusWrite_9c_1V_2V0123     : SchedWriteRes<[OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 9; }

}  // NumMicroOps = 3

//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types

let NumMicroOps = 4 in {

  def OlympusWrite_2c_2L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV]>           { let Latency = 2; }
  def OlympusWrite_3c_2SA_2V0123    : SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 3; }
  def OlympusWrite_4c_2L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV]>           { let Latency = 4; }
  def OlympusWrite_4c_2SA_2V0123    : SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 4; }
  def OlympusWrite_5c_1I_3L         : SchedWriteRes<[OlympusUnitI, OlympusUnitL, OlympusUnitL, OlympusUnitL]>           { let Latency = 5; }
  def OlympusWrite_6c_2I_2L         : SchedWriteRes<[OlympusUnitI, OlympusUnitI, OlympusUnitL, OlympusUnitL]>           { let Latency = 6; }
  def OlympusWrite_6c_4L            : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitL]>           { let Latency = 6; }
  def OlympusWrite_6c_4V            : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>           { let Latency = 6; }
  def OlympusWrite_6c_4V0           : SchedWriteRes<[OlympusUnitV0, OlympusUnitV0, OlympusUnitV0, OlympusUnitV0]>       { let Latency = 6; }
  def OlympusWrite_7c_1SA_2V_1V0123 : SchedWriteRes<[OlympusUnitSA, OlympusUnitV, OlympusUnitV, OlympusUnitV0123]>      { let Latency = 7; }
  def OlympusWrite_8c_2L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV]>           { let Latency = 8; }
  def OlympusWrite_8c_2V_2V1        : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV1, OlympusUnitV1]>         { let Latency = 8; }
  def OlympusWrite_8c_4V            : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>           { let Latency = 8; }
  def OlympusWrite_9c_2L_2V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV]>           { let Latency = 9; }
  def OlympusWrite_9c_2L_2V1        : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV1, OlympusUnitV1]>         { let Latency = 9; }
  def OlympusWrite_9c_2V_2V0123     : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123]>   { let Latency = 9; }
  def OlympusWrite_11c_2V_2V0123    : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123]>   { let Latency = 11; }

}  // NumMicroOps = 4

//===----------------------------------------------------------------------===//
// Define generic 5 micro-op types

let NumMicroOps = 5 in {

  def OlympusWrite_4c_2SA_1V03_2V0123 : SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitV03, OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 4; }
  def OlympusWrite_6c_5V              : SchedWriteRes<[OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>             { let Latency = 6; }
  def OlympusWrite_8c_1L_4V           : SchedWriteRes<[OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>             { let Latency = 8; }
  def OlympusWrite_8c_2L_3V           : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV]>             { let Latency = 8; }
  def OlympusWrite_9c_1I_2L_2V        : SchedWriteRes<[OlympusUnitI, OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV]>             { let Latency = 9; }
  def OlympusWrite_9c_4L_1V03         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitV03]>           { let Latency = 9; }
  def OlympusWrite_10c_1L_4V          : SchedWriteRes<[OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>             { let Latency = 10; }

}  // NumMicroOps = 5

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

let NumMicroOps = 6 in {

  def OlympusWrite_2c_3L_3V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV]>                { let Latency = 2; }
  def OlympusWrite_4c_3SA_3V0123    : SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 4; }
  def OlympusWrite_5c_2SA_2V_2V0123 : SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitV, OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123]>      { let Latency = 5; }
  def OlympusWrite_8c_2L_4V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV]>                { let Latency = 8; }
  def OlympusWrite_8c_3L_3V         : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitV, OlympusUnitV, OlympusUnitV]>                { let Latency = 8; }
  def OlympusWrite_9c_4L_2V03       : SchedWriteRes<[OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitL, OlympusUnitV03, OlympusUnitV03]>            { let Latency = 9; }

}  // NumMicroOps = 6

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

def OlympusWrite_5c_1I_2SA_2V_2V0123 :
      SchedWriteRes<[OlympusUnitI, OlympusUnitSA, OlympusUnitSA, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 5;
  let NumMicroOps = 7;
}

def OlympusWrite_8c_3L_4V : SchedWriteRes<[OlympusUnitL, OlympusUnitL,
                                           OlympusUnitL, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV]> {
  let Latency = 8;
  let NumMicroOps = 7;
}

def OlympusWrite_9c_1I_3L_3V : SchedWriteRes<[OlympusUnitI, OlympusUnitL,
                                              OlympusUnitL, OlympusUnitL,
                                              OlympusUnitV, OlympusUnitV,
                                              OlympusUnitV]> {
  let Latency = 9;
  let NumMicroOps = 7;
}

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types

def OlympusWrite_2c_4L_4V : SchedWriteRes<[OlympusUnitL, OlympusUnitL,
                                           OlympusUnitL, OlympusUnitL,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV]> {
  let Latency = 2;
  let NumMicroOps = 8;
}

def OlympusWrite_4c_4SA_4V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123]> {
  let Latency = 4;
  let NumMicroOps = 8;
}

def OlympusWrite_7c_2SA_4V_2V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV0123,
                     OlympusUnitV0123]> {
  let Latency = 7;
  let NumMicroOps = 8;
}

def OlympusWrite_8c_4L_4V : SchedWriteRes<[OlympusUnitL, OlympusUnitL,
                                           OlympusUnitL, OlympusUnitL,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV]> {
  let Latency = 8;
  let NumMicroOps = 8;
}

//===----------------------------------------------------------------------===//
// Define generic 9 micro-op types

def OlympusWrite_5c_4SA_1V03_4V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV03, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 5;
  let NumMicroOps = 9;
}

def OlympusWrite_6c_3SA_3V_3V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV0123,
                     OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 6;
  let NumMicroOps = 9;
}

def OlympusWrite_10c_1L_8V : SchedWriteRes<[OlympusUnitL, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV]> {
  let Latency = 10;
  let NumMicroOps = 9;
}

//===----------------------------------------------------------------------===//
// Define generic 10 micro-op types

def OlympusWrite_5c_4SA_2V03_4V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV03, OlympusUnitV03, OlympusUnitV0123,
                     OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 5;
  let NumMicroOps = 10;
}

def OlympusWrite_6c_1I_3SA_3V_3V0123 :
      SchedWriteRes<[OlympusUnitI, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 6;
  let NumMicroOps = 10;
}

//===----------------------------------------------------------------------===//
// Define generic 12 micro-op types

def OlympusWrite_6c_4SA_4V_4V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123]> {
  let Latency = 6;
  let NumMicroOps = 12;
}

def OlympusWrite_9c_4L_8V : SchedWriteRes<[OlympusUnitL, OlympusUnitL,
                                           OlympusUnitL, OlympusUnitL,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV,
                                           OlympusUnitV, OlympusUnitV]> {
  let Latency = 9;
  let NumMicroOps = 12;
}

def OlympusWrite_10c_4L_8V : SchedWriteRes<[OlympusUnitL, OlympusUnitL,
                                            OlympusUnitL, OlympusUnitL,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV,
                                            OlympusUnitV, OlympusUnitV]> {
  let Latency = 10;
  let NumMicroOps = 12;
}

//===----------------------------------------------------------------------===//
// Define generic 13 micro-op types

def OlympusWrite_6c_1I_4SA_4V_4V0123 :
      SchedWriteRes<[OlympusUnitI, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitSA, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 6;
  let NumMicroOps = 13;
}

def OlympusWrite_10c_1I_4L_8V : SchedWriteRes<[OlympusUnitI, OlympusUnitL,
                                               OlympusUnitL, OlympusUnitL,
                                               OlympusUnitL, OlympusUnitV,
                                               OlympusUnitV, OlympusUnitV,
                                               OlympusUnitV, OlympusUnitV,
                                               OlympusUnitV, OlympusUnitV,
                                               OlympusUnitV]> {
  let Latency = 10;
  let NumMicroOps = 13;
}

//===----------------------------------------------------------------------===//
// Define generic 16 micro-op types

def OlympusWrite_8c_4SA_8V_4V0123 :
      SchedWriteRes<[OlympusUnitSA, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV0123, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123]> {
  let Latency = 8;
  let NumMicroOps = 16;
}

//===----------------------------------------------------------------------===//
// Define generic 17 micro-op types

def OlympusWrite_8c_1I_4SA_8V_4V0123 :
      SchedWriteRes<[OlympusUnitI, OlympusUnitSA, OlympusUnitSA, OlympusUnitSA,
                     OlympusUnitSA, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV, OlympusUnitV, OlympusUnitV,
                     OlympusUnitV, OlympusUnitV0123, OlympusUnitV0123,
                     OlympusUnitV0123, OlympusUnitV0123]> {
  let Latency = 8;
  let NumMicroOps = 17;
}

//===----------------------------------------------------------------------===//
// Define predicates

// Check if SVE INC/DEC (scalar), ALL, {1, 2, 4, 8}
def OlympusIsCheapIncDec : MCSchedPredicate<
                             CheckAll<[CheckOpcode<[
                                         INCB_XPiI, INCH_XPiI,
                                         INCW_XPiI, INCD_XPiI,
                                         DECB_XPiI, DECH_XPiI,
                                         DECW_XPiI, DECD_XPiI]>,
                                       CheckImmOperand<2, 31>,
                                       CheckAny<[
                                         CheckImmOperand<3, 1>,
                                         CheckImmOperand<3, 2>,
                                         CheckImmOperand<3, 4>,
                                         CheckImmOperand<3, 8>,]>]>>;

// Check EXTR, ROR alias or imms == 0
def OlympusIsCheapExtr : MCSchedPredicate<  // EXTR Rd, Rs, Rs, #Imm
                           CheckAll<[CheckOpcode<[EXTRWrri, EXTRXrri]>,
                                     CheckAny<[CheckSameRegOperand<1, 2>,
                                               CheckImmOperand<3, 0>]>]>>;

// Check if logic shift or arithmetic LSL <= 4.
def OlympusIsCheapShift : MCSchedPredicate<
                            CheckAny<[IsLogicShiftOp,
                                      CheckAll<[IsArithShiftOp,
                                                CheckShiftLSL,
                                                CheckAny<[CheckShiftBy0,
                                                          CheckShiftBy1,
                                                          CheckShiftBy2,
                                                          CheckShiftBy3,
                                                          CheckShiftBy4]>]>]>>;

// Check if extending from W/X.
def OlympusIsCheapExtend : MCSchedPredicate<
                             CheckAll<[IsArithExtOp,
                                       CheckAny<[CheckExtUXTW, CheckExtUXTX,
                                                 CheckExtSXTW, CheckExtSXTX]>]>>;

// Check if STRH has scaled offset (shift amount of 1)
def OlympusIsScaledSTRH : MCSchedPredicate<CheckImmOperand<4, 1>>;

//===----------------------------------------------------------------------===//
// Define predicate-controlled types

def OlympusWrite_ScaledSTRH : SchedWriteVariant<[
                                SchedVar<OlympusIsScaledSTRH, [OlympusWrite_2c_1SA_1D_1I]>,
                                SchedVar<NoSchedPred,         [OlympusWrite_1c_1SA_1D]>]>;

def OlympusWrite_ShiftI : SchedWriteVariant<[
                            SchedVar<OlympusIsCheapShift, [OlympusWrite_1c_1I]>,
                            SchedVar<NoSchedPred,         [OlympusWrite_2c_1M]>]>;

def OlympusWrite_ShiftF : SchedWriteVariant<[
                            SchedVar<OlympusIsCheapShift, [OlympusWrite_1c_1F]>,
                            SchedVar<NoSchedPred,         [OlympusWrite_2c_1M]>]>;

def OlympusWrite_ExtendI : SchedWriteVariant<[
                             SchedVar<OlympusIsCheapExtend, [OlympusWrite_1c_1I]>,
                             SchedVar<NoSchedPred,          [OlympusWrite_2c_1M]>]>;

def OlympusWrite_ExtendF : SchedWriteVariant<[
                             SchedVar<OlympusIsCheapExtend, [OlympusWrite_1c_1F]>,
                             SchedVar<NoSchedPred,          [OlympusWrite_2c_1M]>]>;

def OlympusWrite_Logical : SchedWriteVariant<[
                             SchedVar<NeoverseNoLSL, [OlympusWrite_1c_1F]>,
                             SchedVar<NoSchedPred,   [OlympusWrite_2c_1M]>]>;

def OlympusWrite_Extr : SchedWriteVariant<[
                          SchedVar<OlympusIsCheapExtr, [OlympusWrite_1c_1I]>,
                          SchedVar<NoSchedPred,        [OlympusWrite_3c_1I_1M]>]>;

def OlympusWrite_0or1c_1I : SchedWriteVariant<[
                              SchedVar<NeoverseZeroMove, [OlympusWrite_0c]>,
                              SchedVar<NoSchedPred,      [OlympusWrite_1c_1I]>]>;

def OlympusWrite_0or2c_1V : SchedWriteVariant<[
                              SchedVar<NeoverseZeroMove, [OlympusWrite_0c]>,
                              SchedVar<NoSchedPred,      [OlympusWrite_2c_1V]>]>;

def OlympusWrite_0or3c_1M : SchedWriteVariant<[
                              SchedVar<NeoverseZeroMove, [OlympusWrite_0c]>,
                              SchedVar<NoSchedPred,      [OlympusWrite_3c_1M]>]>;

def OlympusWrite_2c_1V03_or_1M_1V03 : SchedWriteVariant<[
                                        SchedVar<NeoversePdIsPg, [OlympusWrite_2c_1M_1V03]>,
                                        SchedVar<NoSchedPred,    [OlympusWrite_2c_1V03]>]>;

def OlympusWrite_IncDec : SchedWriteVariant<[
                            SchedVar<OlympusIsCheapIncDec, [OlympusWrite_1c_1I]>,
                            SchedVar<NoSchedPred,          [OlympusWrite_2c_1M]>]>;

//===----------------------------------------------------------------------===//
// Define forwarded types

def OlympusWr_FRS : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_FRS : SchedReadAdvance<2, [OlympusWr_FRS]>;

def OlympusWr_VA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VA : SchedReadAdvance<2, [OlympusWr_VA]>;

def OlympusWr_VPA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VPA : SchedReadAdvance<2, [OlympusWr_VPA]>;

def OlympusWr_VSA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VSA : SchedReadAdvance<2, [OlympusWr_VSA]>;

def OlympusWr_VDOT : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
def OlympusRd_VDOT : SchedReadAdvance<1, [OlympusWr_VDOT]>;

def OlympusWr_VMMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
def OlympusRd_VMMA : SchedReadAdvance<1, [OlympusWr_VMMA]>;

def OlympusWr_VFCMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VFCMA : SchedReadAdvance<2, [OlympusWr_VFCMA]>;

def OlympusWr_VFMAL : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VFMAL : SchedReadAdvance<2, [OlympusWr_VFMAL]>;

def OlympusWr_ZBFMAL : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_ZBFMAL : SchedReadAdvance<2, [OlympusWr_ZBFMAL]>;
def OlympusWr_ZBFDOT : SchedWriteRes<[OlympusUnitV]> { let Latency = 5; }
def OlympusRd_ZBFDOT : SchedReadAdvance<2, [OlympusWr_ZBFDOT]>;
def OlympusWr_ZBFMMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 6; }
def OlympusRd_ZBFMMA : SchedReadAdvance<2, [OlympusWr_ZBFMMA]>;

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// Branch instructions
// -----------------------------------------------------------------------------

// Branch, simple
// Compare and branch
def : SchedAlias<WriteBr, OlympusWrite_1c_1B>;
def : SchedAlias<WriteBrReg, OlympusWrite_1c_1B>;

// Branch and link
def : InstRW<[OlympusWrite_1c_1I_1B], (instrs BL, BLR)>;

// Arithmetic and logical instructions
// -----------------------------------------------------------------------------

// ALU/Logical basic
def : SchedAlias<WriteI,   OlympusWrite_1c_1I>;
def : SchedAlias<WriteImm, OlympusWrite_1c_1I>;
def : InstRW<[OlympusWrite_1c_1I], (instrs GMI, SUBP)>;
def : InstRW<[OlympusWrite_0or1c_1I], (instregex "^ORR[WX]rs$")>;

// ALU/Logical basic, flag write
def : InstRW<[OlympusWrite_1c_1F], (instregex "^(ADD|AND|SUB)S[WX]ri$",
                                              "^(ADC|SBC)S[WX]r$",
                                              "^SUBPS$")>;

// ALU, src extend from H/B
def : SchedAlias<WriteIEReg, OlympusWrite_ExtendI>;
def : InstRW<[OlympusWrite_ExtendF], (instregex "^(ADD|SUB)S[WX]rx")>;

// ALU, src LSL shift > 4 or LSR/ASR/ROR shift
def : SchedAlias<WriteISReg, OlympusWrite_ShiftI>;
def : InstRW<[OlympusWrite_ShiftF], (instregex "^(ADD|SUB)S[WX]rs$")>;

// Logical, flag write, src shift
def : InstRW<[OlympusWrite_Logical], (instregex "^(AND|BIC)S[WX]rs$")>;

// Conditional compare
def : InstRW<[OlympusWrite_1c_1F], (instregex "^CCM[NP][WX][ir]$")>;

// Flag manipulation
def : SchedAlias<WriteSys, OlympusWrite_1c_1F>;
def : InstRW<[OlympusWrite_1c_1F], (instrs SETF8, SETF16, RMIF)>;

// Arithmetic to tag
def : InstRW<[OlympusWrite_2c_1M], (instrs ADDG, SUBG)>;

// Insert Random Tag
def : InstRW<[OlympusWrite_2c_1M], (instrs IRG, IRGstack)>;

// Divide and multiply instructions
// -----------------------------------------------------------------------------

// Divide
def : SchedAlias<WriteID32, OlympusWrite_12c_1M>;
def : SchedAlias<WriteID64, OlympusWrite_20c_1M>;

// Multiply accumulate
def : SchedAlias<WriteIM32, OlympusWrite_2c_1M>;
def : SchedAlias<WriteIM64, OlympusWrite_2c_1M>;
def OlympusWr_IM  : SchedWriteRes<[OlympusUnitM]> { let Latency = 2; }
def OlympusWr_IMA : SchedWriteRes<[OlympusUnitM]> { let Latency = 2; }
def OlympusWr_IMX : SchedWriteVariant<[
                      SchedVar<IsReg3ZeroPred, [OlympusWr_IM]>,
                      SchedVar<NoSchedPred,    [OlympusWr_IMA]>]>;
def OlympusRd_IMA : SchedReadAdvance<1, [OlympusWr_IM, OlympusWr_IMA]>;
def : InstRW<[OlympusWr_IMX, ReadIM, ReadIM, OlympusRd_IMA],
             (instregex "^M(ADD|SUB)[WX]rrr$")>;
def : InstRW<[OlympusWr_IMX, ReadIM, ReadIM, OlympusRd_IMA],
             (instregex "^[SU]M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[OlympusWrite_3c_1M], (instrs SMULHrr, UMULHrr)>;

// Pointer authentication instructions
// -----------------------------------------------------------------------------

// Authenticate address
// Compute pointer authentication code
def : InstRW<[OlympusWrite_4c_1M0], (instregex "^AUT", "^PAC")>;

// Strip pointer authentication code
def : InstRW<[OlympusWrite_2c_1M0], (instrs XPACD, XPACI, XPACLRI)>;

// Branch, register with authentication
def : InstRW<[OlympusWrite_5c_1B_1M0], (instrs BRAA, BRAAZ, BRAB, BRABZ, RETAA,
                                               RETAB, ERETAA, ERETAB)>;

// Branch and link, register with authentication
def : InstRW<[OlympusWrite_5c_1I_1B_1M0], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ)>;

// Load, register with authentication
def : InstRW<[OlympusWrite_8c_1M0_1L], (instregex "^LDRA[AB]indexed")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1M0_1L], (instregex "^LDRA[AB]writeback")>;

// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Address generation
def : InstRW<[OlympusWrite_1c_1F], (instrs ADR, ADRP)>;

// Extract, ROR alias or imms==0
// Extract, other
def : SchedAlias<WriteExtr, OlympusWrite_Extr>;

// Bitfield move, basic
// Count leading
// Move immed
// Reverse bits/bypes
// Variable shift
def : SchedAlias<WriteIS, OlympusWrite_1c_1I>;
def : InstRW<[OlympusWrite_0or1c_1I], (instregex "^MOVZ[WX]i$")>;

// Bitfield move, insert
def : InstRW<[OlympusWrite_2c_1M], (instregex "^BFM[WX]ri$")>;

// CRC checksum
def OlympusWr_CRC : SchedWriteRes<[OlympusUnitM]> { let Latency = 2; }
def OlympusRd_CRC : SchedReadAdvance<1, [OlympusWr_CRC]>;
def : InstRW<[OlympusWr_CRC, OlympusRd_CRC], (instregex "^CRC32")>;

// Load instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteLD,    OlympusWrite_4c_1L>;
def : SchedAlias<WriteLDIdx, OlympusWrite_4c_1L>;

// Load register, possible wback
// LDR (immediate)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDR[WX]ui$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L], (instregex "^LDR[WX](pre|post)$")>;
// LDRB (immediate)
def : InstRW<[OlympusWrite_4c_1L], (instrs LDRBBui)>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L], (instregex "^LDRBB(pre|post)$")>;
// LDRH (immediate)
def : InstRW<[OlympusWrite_4c_1L], (instrs LDRHHui)>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L], (instregex "^LDRHH(pre|post)$")>;
// LDRSB, LDRSH (immediate)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDRS[BH][WX]ui$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L], (instregex "^LDRS[BH][WX](pre|post)$")>;
// LDRSW (immediate)
def : InstRW<[OlympusWrite_4c_1L], (instrs LDRSWui)>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L], (instregex "^LDRSW(pre|post)$")>;

// Load register, bare
// LDR (register)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDR[WX]ro[WX]$")>;
// LDRB (register)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDRBBro[WX]$")>;
// LDRH (register)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDRHHro[WX]$")>;
// LDRSB, LDRSH (register)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDRS[BH][WX]ro[WX]$")>;
// LDRSW (register)
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDRSWro[WX]$")>;
// LDUR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDUR[WX]i$")>;
// LDURB
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDURBBi$")>;
// LDURH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDURHHi$")>;
// LDURSB, LDURSH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDURS[BH][WX]i$")>;
// LDURSW
def : InstRW<[OlympusWrite_4c_1L], (instrs LDURSWi)>;
// LDAR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAR[WX]$")>;
// LDARB, LDARH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAR[BH]$")>;
// LDAPR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPR[WX]$")>;
// LDAPRB, LDAPRH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPR[BH]$")>;
// LDAPUR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPURX?i$")>;
// LDAPURB, LDAPURH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPUR[BH]i$")>;
// LDAPURSB, LDAPURSH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPURS[BH][WX]i$")>;
// LDAPURSW
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAPURSWi$")>;
// LDLAR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDLAR[WX]$")>;
// LDLARB, LDLARH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDLAR[BH]$")>;
// LDTR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDTR[WX]i$")>;
// LDTRB, LDTRH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDTR[BH]i$")>;
// LDTRSB, LDTRSH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDTRS[BH][WX]i$")>;
// LDTRSW
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDTRSWi$")>;
// LDXR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDXR[WX]$")>;
// LDXRB, LDXRH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDXR[BH]$")>;
// LDAXR
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAXR[WX]$")>;
// LDAXRB, LDAXRH
def : InstRW<[OlympusWrite_4c_1L], (instregex "^LDAXR[BH]$")>;

// Load pair, possible wback
// LDP
def : InstRW<[OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDP[WX]i$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDP[WX](pre|post)$")>;
// LDPSW
def : InstRW<[OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDPSWi$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDPSW(pre|post)$")>;

// Load pair, bare
// LDNP
def : InstRW<[OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDNP[WX]i$")>;
// LDXP, LDAXP
def : InstRW<[OlympusWrite_4c_1L, WriteLDHi], (instregex "^LDA?XP[WX]$")>;

// Load literal
// LDR (literal)
// LDRSW (literal)
// PRFM (literal)
def : InstRW<[OlympusWrite_5c_1I_1L], (instrs LDRWl, LDRXl, LDRSWl, PRFMl)>;

// Load allocation tag
// LDG
def : InstRW<[OlympusWrite_5c_1I_1L], (instrs LDG, LDGM)>;

// Store instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteST,    OlympusWrite_1c_1SA_1D>;
def : SchedAlias<WriteSTIdx, OlympusWrite_1c_1SA_1D>;
def : SchedAlias<WriteSTP,   OlympusWrite_1c_1SA_1D>;
def : SchedAlias<WriteAdr,   OlympusWrite_1c_1I>;

// Store register, possible wback
// STR (immediate)
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STR[WX]ui$")>;
def : InstRW<[WriteAdr, OlympusWrite_1c_1SA_1D], (instregex "^STR[WX](pre|post)$")>;
// STRB (immediate)
def : InstRW<[OlympusWrite_1c_1SA_1D], (instrs STRBBui)>;
def : InstRW<[WriteAdr, OlympusWrite_1c_1SA_1D], (instregex "^STRBB(pre|post)$")>;
// STRH (immediate)
def : InstRW<[OlympusWrite_1c_1SA_1D], (instrs STRHHui)>;
def : InstRW<[WriteAdr, OlympusWrite_1c_1SA_1D], (instregex "^STRHH(pre|post)$")>;

// Store register, scaled by 2
// STRH (register)
def : InstRW<[OlympusWrite_ScaledSTRH], (instregex "^STRHHro[WX]$")>;

// Store register, bare
// STR (register)
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STR[WX]ro[WX]$")>;
// STRB (register)
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STRBBro[WX]$")>;
// STUR
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STUR[WX]i$")>;
// STURB
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STURBBi$")>;
// STURH
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STURHHi$")>;
// STLR
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLR[WX]$")>;
// STLRB, STLRH
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLR[BH]$")>;
// STLLR
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLLR[WX]$")>;
// STLLRB, STLLRH
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLLR[BH]$")>;
// STLUR
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLUR[WX]i$")>;
// STLURB, STLURH
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STLUR[BH]i$")>;
// STTR
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STTR[WX]i$")>;
// STTRB, STTRH
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STTR[BH]i$")>;

// Store pair, general
// STP
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STP[WX]i$")>;
def : InstRW<[WriteAdr, OlympusWrite_1c_1SA_1D], (instregex "^STP[WX](pre|post)$")>;

// Store pair, non-temporal
// STNP
def : InstRW<[OlympusWrite_1c_1SA_1D], (instregex "^STNP[WX]i$")>;

// Store exclusive
// STXR
def : InstRW<[OlympusWrite_4c_1SA_1D], (instregex "^STXR[WX]$")>;
// STXRB, STXRH
def : InstRW<[OlympusWrite_4c_1SA_1D], (instregex "^STXR[BH]$")>;
// STLXR
def : InstRW<[OlympusWrite_4c_1SA_1D], (instregex "^STLXR[WX]$")>;
// STLXRB, STLXRH
def : InstRW<[OlympusWrite_4c_1SA_1D], (instregex "^STLXR[BH]$")>;
// STXP, STLXP
def : InstRW<[OlympusWrite_4c_1SA_1D], (instregex "^STL?XP[WX]$")>;

// Store allocation tag
// ST2G
// STG
// STZ2G
// STZG
// STGP
def : InstRW<[OlympusWrite_1c_1SA_1D], (instrs STGi, ST2Gi, STZGi,
                                               STZ2Gi, STGPi, STGM, STZGM)>;
def : InstRW<[WriteAdr, OlympusWrite_1c_1SA_1D],  (instregex "^ST2?G(Pre|Post)Index$",
                                                             "^STZ2?G(Pre|Post)Index$",
                                                             "^STGP(pre|post)$")>;

// Scalar/SIMD floating point instructions
// -----------------------------------------------------------------------------

// FP general
def : SchedAlias<WriteF,    OlympusWrite_2c_1V>;
def : SchedAlias<WriteFImm, OlympusWrite_2c_1V>;

// FMOV (scalar, immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMOV[HSD]i$")>;
// FMOV (register)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMOV[HSD]r$")>;
// FMOV (vector, immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMOV(v4f16|v8f16|v2f32|v4f32|v2f64)_ns$")>;
// FCSEL
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCSEL[HSD]rrr$")>;
// FABS (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABS[HSD]r$")>;
// FABD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABD(16|32|64)$")>;
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABD(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FABS (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABS(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FNEG (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FNEG[HSD]r$")>;
// FNEG (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FNEG(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FADD (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADD[HSD]rr$")>;
// FADD (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADD(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FADDP (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADDP(v2i16|v2i32|v2i64)p$")>;
// FADDP (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADDP(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FCADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCADD(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FSUB (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUB[HSD]rr$")>;
// FSUB (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUB(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FAC<cc>
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FACG[ET](16|32|64)$")>;
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FACG[ET](v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FCM<cc> (register)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCM(EQ|GE|GT)(16|32|64)$")>;
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCM(EQ|GE|GT)(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FCM<cc> (zero)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCM(EQ|GE|GT|LE|LT)(v1i16|v1i32|v1i64|v4i16|v8i16|v2i32|v4i32|v2i64)rz$")>;
// FMAX/FMIN(NM) (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^(FMAX|FMIN)(NM)?(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FMAX/FMIN(NM) (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^(FMAX|FMIN)(NM)?[HSD]rr$")>;
// FMAX(NM)P/FMIN(NM)P (scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^(FMAX|FMIN)(NM)?P(v2i16|v2i32|v2i64)p$")>;
// FMAXNMP/FMINNMP (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^(FMAX|FMIN)NMP(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FMAXP/FMINP (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAXP(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FAMAX/FAMIN
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FA(MAX|MIN)(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;

// FP compare
def : SchedAlias<WriteFCmp,  OlympusWrite_2c_1V03>;
// FCCMP,FCCMPE
def : InstRW<[OlympusWrite_2c_1V03], (instregex "^FCCMPE?[HSD]rr$")>;
// FCMP, FCMPE
def : InstRW<[OlympusWrite_2c_1V03], (instregex "^FCMPE?[HSD](rr|ri)$")>;

// FP multiply
def : WriteRes<WriteFMul, [OlympusUnitV]> { let Latency = 3; }
def OlympusWr_FMul : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
// FMUL, FNMUL (scalar)
def : InstRW<[OlympusWr_FMul], (instregex "^FN?MUL[HSD]rr$")>;
// FMUL, FMULX (by element)
def : InstRW<[OlympusWr_FMul], (instregex "^FMULX?(v1i16|v1i32|v1i64|v4i16|v8i16|v2i32|v4i32|v2i64)_indexed$")>;
// FMUL (vector)
def : InstRW<[OlympusWr_FMul], (instregex "^FMUL(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FMULX
def : InstRW<[OlympusWr_FMul], (instregex "^FMULX(16|32|64)$")>;
def : InstRW<[OlympusWr_FMul], (instregex "^FMULX(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;
// FSCALE
def : InstRW<[OlympusWr_FMul], (instregex "^FSCALE(v4f16|v8f16|v2f32|v4f32|v2f64)$")>;

// FP multiply accumulate
def OlympusWr_VFMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_VFMA : SchedReadAdvance<0, [OlympusWr_FMul, OlympusWr_VFMA], [1, 2]>;
// FN?MADD, FN?MSUB
def : InstRW<[OlympusWr_VFMA, ReadDefault, ReadDefault, OlympusRd_VFMA], (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;
// FCMLA (by element)
def : InstRW<[OlympusWr_VFCMA, OlympusRd_VFCMA], (instregex "^FCMLA(v4f16|v8f16|v4f32)_indexed$")>;
// FCMLA
def : InstRW<[OlympusWr_VFCMA, OlympusRd_VFCMA], (instregex "^FCMLA(v4f16|v2f32|v8f16|v4f32|v2f64)$")>;
// FMLA, FMLS (by element)
def : InstRW<[OlympusWr_VFMA, OlympusRd_VFMA], (instregex "^FML[AS]v.+_indexed$")>;
// FMLA, FMLS (vector)
def : InstRW<[OlympusWr_VFMA, OlympusRd_VFMA], (instregex "^FML[AS](v4f16|v2f32|v8f16|v4f32|v2f64)$")>;
// FMLAL, FMLAL2 (by element)
// FMLSL, FMLSL2 (by element)
def : InstRW<[OlympusWr_VFMAL, OlympusRd_VFMAL], (instregex "^FML[AS]L2?lanev")>;
// FMLAL, FMLAL2 (vector)
// FMLSL, FMLSL2 (vector)
def : InstRW<[OlympusWr_VFMAL, OlympusRd_VFMAL], (instregex "^FML[AS]L2?(v4f16|v2f32|v8f16|v4f32)$")>;

// FP reduction, DP/SP/(HP 64b)
// FMAXNMV, FMAXV, FMINMV, FMINV
def : InstRW<[OlympusWrite_4c_2V], (instregex "^(FMAX|FMIN)(NM)?V(v4i16|v4i32)v$")>;

// FP reduction, HP 128b
// FMAXNMV, FMAXV, FMINMV, FMINV
def : InstRW<[OlympusWrite_6c_3V], (instregex "^(FMAX|FMIN)(NM)?Vv8i16v$")>;

// FP divide/sqrt DP, scalar
def : SchedAlias<WriteFDiv,  OlympusWrite_8c_1V12>;
// FDIV, FSQRT (scalar)
def : InstRW<[OlympusWrite_13c_1V12], (instrs FDIVDrr, FSQRTDr)>;

// FP divide/sqrt SP, scalar
// FDIV, FSQRT (scalar)
def : InstRW<[OlympusWrite_8c_1V12], (instrs FDIVSrr, FSQRTSr)>;

// FP divide/sqrt HP, scalar
// FDIV, FSQRT (scalar)
def : InstRW<[OlympusWrite_6c_1V12], (instrs FDIVHrr, FSQRTHr)>;

// FP divide/sqrt, vector DP 128b
// FDIV, FSQRT (vector)
def : InstRW<[OlympusWrite_14c_1V12_2], (instrs FDIVv2f64, FSQRTv2f64)>;

// FP divide/sqrt, vector SP 128b
// FDIV, FSQRT (vector)
def : InstRW<[OlympusWrite_11c_1V12_4], (instrs FDIVv4f32, FSQRTv4f32)>;

// FP divide/sqrt, vector SP 64b
// FDIV, FSQRT (vector)
def : InstRW<[OlympusWrite_9c_1V12_2], (instrs FDIVv2f32, FSQRTv2f32)>;

// FP divide/sqrt, vector HP 128b
// FDIV, FSQRT (vector)
def : InstRW<[OlympusWrite_13c_1V12_8], (instrs FDIVv8f16, FSQRTv8f16)>;

// FP divide/sqrt, vector HP 64b
// FDIV, FSQRT (vector)
def : InstRW<[OlympusWrite_9c_1V12_4], (instrs FDIVv4f16, FSQRTv4f16)>;

// FP round to int, scalar
// FRINT<cc> (scalar)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRINT[AIMNPXZ][HSD]r$")>;
// FRINT32X (scalar)
// FRINT32Z (scalar)
// FRINT64X (scalar)
// FRINT64Z (scalar)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRINT(32|64)[XZ][SD]r$")>;

// FP round to int, vector, DP 128b or SP 64b
// FRINT<cc> (vector)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRINT[AIMNPXZ](v2f64|v2f32)$")>;
// FRINT32X (vector)
// FRINT32Z (vector)
// FRINT64X (vector)
// FRINT64Z (vector)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRINT(32|64)[XZ](v2f32|v2f64)$")>;

// FP round to int, vector, SP 128b or HP 64b
// FRINT<cc> (vector)
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRINT[AIMNPXZ](v4f32|v4f16)$")>;
// FRINT32X (vector)
// FRINT32Z (vector)
// FRINT64X (vector)
// FRINT64Z (vector)
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRINT(32|64)[XZ]v4f32$")>;

// FP round to int, vector, HP 128b
// FRINT<cc> (vector)
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FRINT[AIMNPXZ]v8f16$")>;

// FP convert, scalar
// FCVT
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTSHr, FCVTDHr, FCVTHSr, FCVTDSr, FCVTHDr, FCVTSDr)>;

// FCVTAS (vector)
// FCVTAU (vector)
// FCVTMS (vector)
// FCVTMU (vector)
// FCVTNS (vector)
// FCVTNU (vector)
// FCVTPS (vector)
// FCVTPU (vector)
// FCVTZS (vector, integer)
// FCVTZU (vector, integer)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVT[AMNPZ][SU](v1i32|v1i64)$")>;
// FCVTZS (vector, fixed-point)
// FCVTZU (vector, fixed-point)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTZ[SU][dsh]$")>;
// SCVTF (vector, integer)
// UCVTF (vector, integer)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU]CVTF(v1i16|v1i32|v1i64)$")>;
// SCVTF (vector, fixed-point)
// UCVTF (vector, fixed-point)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU]CVTF[dsh]$")>;
// FCVTXN
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTXNv1i64)>;

// FP convert, vector between F32 and F64
// FCVTL, FCVTL2
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTL(v2i32|v4i32)$")>;
// FCVTN, FCVTN2
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTN(v2i32|v4i32)$")>;

// FP convert, vector F32 to F16
// FCVTN, FCVTN2
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^FCVTN(v4i16|v8i16)$")>;

// FP convert, vector F16 to F32
// FCVTL, FCVTL2
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVTL(v4i16|v8i16)$")>;

// FP convert, Javascript
def : SchedAlias<WriteFCvt, OlympusWrite_4c_1V03>;
// FJCVTZS
def : InstRW<[OlympusWrite_4c_1V03], (instrs FJCVTZS)>;

// FP convert, vector, DP 128b or SP 64b
// FCVT<cc> (vector)
// FCVTZS (vector, integer)
// FCVTZU (vector, integer)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVT[AMNPZ][SU](v2f32|v2f64)$")>;
// FCVTZS (vector, fixed-point)
// FCVTZU (vector, fixed-point)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTZ[SU](v2i32|v2i64)_shift$")>;

// SCVTF (vector, integer)
// UCVTF (vector, integer)
// SCVTF (vector, fixed-point)
// UCVTF (vector, fixed-point)
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU]CVTF(v2f32|v2f64)$",
                                                  "^[SU]CVTF(v2i32|v2i64)_shift$")>;
// FCVTXN, FCVTXN2
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTXNv2f32, FCVTXNv4f32)>;

// FP convert, vector, SP 128b or HP 64b
// FCVT<cc> (vector)
// FCVTZS (vector, integer)
// FCVTZU (vector, integer)
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVT[AMNPZ][SU](v4f32|v4f16)$")>;
// FCVTZS (vector, fixed-point)
// FCVTZU (vector, fixed-point)
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVTZ[SU](v4i16|v4i32)_shift$")>;

// SCVTF (vector, integer)
// UCVTF (vector, integer)
// SCVTF (vector, fixed-point)
// UCVTF (vector, fixed-point)
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^[SU]CVTF(v4f16|v4f32)$",
                                                    "^[SU]CVTF(v4i16|v4i32)_shift$")>;

// FP convert, vector, HP 128b
// FCVT<cc> (vector)
// FCVTZS (vector, integer)
// FCVTZU (vector, integer)
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FCVT[AMNPZ][SU]v8f16$",
                                                    "^FCVT[AMNPZ][SU]v1f16$")>;
// FCVTZS (vector, fixed-point)
// FCVTZU (vector, fixed-point)
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FCVTZ[SU]v8i16_shift$")>;

// SCVTF (vector, integer)
// UCVTF (vector, integer)
// SCVTF (vector, fixed-point)
// UCVTF (vector, fixed-point)
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^[SU]CVTFv8f16$",
                                                    "^[SU]CVTFv8i16_shift$")>;

// FP convert from gen to vec reg
// SCVTF (scalar, integer)
// UCVTF (scalar, integer)
// SCVTF (scalar, fixed-point)
// UCVTF (scalar, fixed-point)
def : InstRW<[OlympusWrite_5c_1M_1V], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
// FCVT<cc> (scalar)
// FCVTZS (scalar, integer)
// FCVTZU (scalar, integer)
def : InstRW<[OlympusWrite_4c_1V03], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]r$")>;
// FCVTZS (scalar, fixed-point)
// FCVTZU (scalar, fixed-point)
def : InstRW<[OlympusWrite_4c_1V03], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri$")>;

// FP reciprocal and square root estimate, scalar
// FRECPX
// FRECPE
// FRSQRTE
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRECPX(v1f16|v1i32|v1i64)$")>;
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRECPE(v1f16|v1i32|v1i64)$")>;
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRSQRTE(v1f16|v1i32|v1i64)$")>;

// FP reciprocal and square root estimate, vector, DP 128b or SP 64b
// FRECPE
// FRSQRTE
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRECPE(v2f64|v2f32)$")>;
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRSQRTE(v2f64|v2f32)$")>;

// FP reciprocal and square root estimate, vector, SP 128b or HP 64b
// FRECPE
// FRSQRTE
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRECPE(v4f32|v4f16)$")>;
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRSQRTE(v4f32|v4f16)$")>;

// FP reciprocal and square root estimate, vector, HP 128b
// FRECPE
// FRSQRTE
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FRECPEv8f16$")>;
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FRSQRTEv8f16$")>;

// FP reciprocal and square root step
// FRECPS
// FRSQRTS
def : InstRW<[OlympusWr_FRS, OlympusRd_FRS, OlympusRd_FRS],
             (instregex "^FRECPS(16|32|64|v)", "^FRSQRTS(16|32|64|v)")>;

// FP mov, from vec to gen reg
def : SchedAlias<WriteFCopy, OlympusWrite_3c_1V03>;
def : InstRW<[OlympusWrite_3c_1V03], (instrs FMOVSWr, FMOVHWr, FMOVDXr, FMOVHXr)>;

// FP mov, from gen to low half of vec reg
def : InstRW<[OlympusWrite_0or3c_1M], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP mov, from gen to high half of vec
def : InstRW<[OlympusWrite_5c_1M_1V], (instrs FMOVXDHighr)>;

// SIMD integer instructions
// -----------------------------------------------------------------------------

// General

def : SchedAlias<WriteVd, OlympusWrite_2c_1V>;
def : SchedAlias<WriteVq, OlympusWrite_2c_1V>;

// ABS
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ABS(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// ADD (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ADD(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// ADDHN, ADDHN2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ADDHN(v8i16_v8i8|v8i16_v16i8|v4i32_v4i16|v2i64_v2i32|v4i32_v8i16|v2i64_v4i32)$")>;
// ADDP (scalar)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ADDPv2i64p$")>;
// ADDP (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ADDP(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// AND (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^AND(v8i8|v16i8)$")>;
// BIC (vector, immediate)
// BIC (vector, register)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^BIC(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// BIF
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^BIF(v8i8|v16i8)$")>;
// BIT
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^BIT(v8i8|v16i8)$")>;
// BSL
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^BSL(v8i8|v16i8)$")>;
// CLS (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CLS(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// CLZ (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CLZ(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// CMEQ, CMGE, CMGT, CMHI, CMHS (register)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CM(EQ|GE|GT|HI|HS)(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// CMEQ, CMGE, CMGT, CMLE, CMLT (zero)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CM(EQ|GE|GT|LE|LT)(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)rz$")>;
// CMTST
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CMTST(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// CNT
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^CNT(v8i8|v16i8)$")>;
// DUP (element)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^DUP(i8|i16|i32|i64)$",
                                                "^DUP(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)lane$")>;
// EOR (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^EOR(v8i8|v16i8)$")>;
// EXT
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^EXT(v8i8|v16i8)$")>;
// INS (element)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^INS(vi8|vi16|vi32|vi64)lane$")>;
// LUTI2,  LUTI4
def : InstRW<[OlympusWrite_3c_1V],  (instregex "^LUT[24]_[BH]")>;
// MOVI
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^MOVI(v2i32|v4i16|v4i32|v8i16)$")>;
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^MOVI(v2s_msl|v4s_msl|v8b_ns|v16b_ns)$")>;
def : InstRW<[OlympusWrite_0or2c_1V], (instrs MOVID, MOVIv2d_ns)>;
// MVNI
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^MVNI(v2i32|v4i16|v4i32|v8i16)$")>;
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^MVNI(v2s_msl|v4s_msl)$")>;
// NEG (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^NEG(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// NOT
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^NOT(v8i8|v16i8)$")>;
// ORN (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ORN(v8i8|v16i8)$")>;
// ORR (vector, immediate)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ORR(v4i32|v8i16|v2i32|v4i16)$")>;
// ORR (vector, register)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^ORR(v8i8|v16i8)$")>;
// RADDHN, RADDHN2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^RADDHN(v8i16_v8i8|v8i16_v16i8|v4i32_v4i16|v2i64_v2i32|v4i32_v8i16|v2i64_v4i32)$")>;
// RBIT (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^RBIT(v8i8|v16i8)$")>;
// REV16 (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^REV16(v8i8|v16i8)$")>;
// REV32 (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^REV32(v8i8|v4i16|v16i8|v8i16)$")>;
// REV64
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^REV64(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// RSUBHN, RSUBHN2, SUBHN, SUBHN2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^R?SUBHN(v8i16_v8i8|v8i16_v16i8|v4i32_v4i16|v2i64_v2i32|v4i32_v8i16|v2i64_v4i32)$")>;
// SHL
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SHLd$", "^SHL(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SHLL, SHLL2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SHLL(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SHRN, SHRN2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SHRN(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// SLI, SRI
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^S(LI|RI)d$", "^S(LI|RI)(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SUB (vector)
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SUB(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// TRN1, TRN2
// UZP1, UZP2
// ZIP1, ZIP2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^(TRN|UZP|ZIP)[12](v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// XTN, XTN2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^XTN(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SABD
// UABD
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]ABD(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SABDL, SABDL2
// UABDL, UABDL2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]ABDL(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SADDL, SADDL2
// UADDL, UADDL2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]ADDL(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SADDLP, UADDLP
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]ADDLP(v8i8_v4i16|v4i16_v2i32|v2i32_v1i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SADDW, SADDW2
// UADDW, UADDW2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]ADDW(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SHADD, SRHADD
// UHADD, URHADD
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]R?HADD(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SHSUB
// UHSUB
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]HSUB(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SMIN, SMAX
// SMINP, SMAXP
// UMIN, UMAX
// UMINP, UMAXP
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU](MIN|MAX)P?(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SQADD
// UQADD
// SQSUB
// UQSUB
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]Q(ADD|SUB)(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// SQABS
// SQNEG
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SQ(ABS|NEG)(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// SSHL
// USHL
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]SHL(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// SSHLL, SSHLL2
// USHLL, USHLL2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]SHLL(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// SSHR
// USHR
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^SSHRd$", "^USHRd$", "^[SU]SHR(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SSUBL, SSUBL2
// SSUBW, SSUBW2
// USUBL, USUBL2
// USUBW, USUBW2
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^[SU]SUB[LW](v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SUQADD
// USQADD
def : InstRW<[OlympusWrite_2c_1V],  (instregex "^(SU|US)QADD(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;

// Shift

// RSHRN, RSHRN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^RSHRN(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// SQRSHL
// UQRSHL
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^[SU]QRSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// SQRSHRN, SQRSHRN2
// SQRSHRUN, SQRSHRUN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^SQRSHRU?N[bhs]", "^SQRSHRU?N(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// UQRSHRN, UQRSHRN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^UQRSHRU?N[bhs]", "^UQRSHRN(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// SQSHL (immediate)
// UQSHL (immediate)
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^[SU]QSHL[bhsd]", "^[SU]QSHL(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SQSHL (register)
// UQSHL (register)
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)$")>;
// SQSHLU
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^SQSHLU[bhsd]", "^SQSHLU(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SQSHRN, SQSHRN2
// SQSHRUN, SQSHRUN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^SQSHRU?N[bhs]", "^SQSHRU?N(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// UQSHRN, UQSHRN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^UQSHRN[bhs]", "^UQSHRN(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)_shift$")>;
// SRSHL
// URSHL
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^[SU]RSHL(v8i8|v4i16|v2i32|v1i64|v16i8|v8i16|v4i32|v2i64)$")>;
// SRSHR
// URSHR
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^[SU]RSHRd$", "^[SU]RSHR(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SQXTN, SQXTN2
// SQXTUN, SQXTUN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^SQXTU?N(v1i8|v1i16|v1i32|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// UQXTN, UQXTN2
def : InstRW<[OlympusWrite_4c_1V],  (instregex "^UQXTN(v1i8|v1i16|v1i32|v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;

// Multiply

// MUL (vector)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// MUL (by element)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL(v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SMULL, SMULL2 (vector)
// UMULL, UMULL2 (vector)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^[SU]MULL(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SMULL, SMULL2 (by element)
// UMULL, UMULL2 (by element)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^[SU]MULL(v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SQDMULH (vector)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULH(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)$")>;
// SQDMULL, SQDMULL2 (vector)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULL(i16|i32)$", "^SQDMULL(v4i16_v4i32|v2i32_v2i64|v8i16_v4i32|v4i32_v2i64)$")>;
// SQRDMULH (vector)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQRDMULH(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)$")>;
// SQDMULH (by element)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULH(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SQDMULL, SQDMULL2 (by element)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULL(v4i16|v2i32|v8i16|v4i32|v1i32|v1i64)_indexed$")>;
// SQRDMULH (by element)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQRDMULH(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)_indexed$")>;

// Multiply accumulate

// MLA (vector)
// MLS (vector)
def OlympusWr_VMA : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusRd_VMA : SchedReadAdvance<2, [OlympusWr_VMA]>;
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^ML[AS](v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SMLAL, SMLAL2 (vector)
// SMLSL, SMLSL2 (vector)
// UMLAL, UMLAL2 (vector)
// UMLSL, UMLSL2 (vector)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^[SU]ML[AS]L(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// MLA (by element)
// MLS (by element)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^ML[AS](v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SMLAL, SMLAL2 (by element)
// SMLSL, SMLSL2 (by element)
// UMLAL, UMLAL2 (by element)
// UMLSL, UMLSL2 (by element)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^[SU]ML[AS]L(v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SQDMLAL, SQDMLAL2 (vector)
// SQDMLSL, SQDMLSL2 (vector)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^SQDML[AS]L(i16|i32|v4i16_v4i32|v2i32_v2i64|v8i16_v4i32|v4i32_v2i64)$")>;
// SQRDMLAH (vector)
// SQRDMLSH (vector)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^SQRDML[AS]H(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)$")>;
// SQDMLAL, SQDMLAL2 (by element)
// SQDMLSL, SQDMLSL2 (by element)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^SQDML[AS]L(v1i32|v1i64|v4i16|v2i32|v8i16|v4i32)_indexed$")>;
// SQRDMLAH (by element)
// SQRDMLSH (by element)
def : InstRW<[OlympusWr_VMA, OlympusRd_VMA], (instregex "^SQRDML[AS]H(v1i16|v1i32|v4i16|v2i32|v8i16|v4i32)_indexed$")>;

// Abs diff/shift accumulate
// SABA
// UABA
def : InstRW<[OlympusWr_VA, OlympusRd_VA],  (instregex "^[SU]ABA(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32)$")>;
// SABAL, SABAL2
// UABAL, UABAL2
def : InstRW<[OlympusWr_VA, OlympusRd_VA],  (instregex "^[SU]ABAL(v8i8_v8i16|v4i16_v4i32|v2i32_v2i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SADALP
// UADALP
def : InstRW<[OlympusWr_VPA, OlympusRd_VPA], (instregex "^[SU]ADALP(v8i8_v4i16|v4i16_v2i32|v2i32_v1i64|v16i8_v8i16|v8i16_v4i32|v4i32_v2i64)$")>;
// SRSRA
// URSRA
def : InstRW<[OlympusWr_VSA, OlympusRd_VSA],  (instregex "^[SU]RSRAd$", "^[SU]RSRA(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;
// SSRA
// USRA
def : InstRW<[OlympusWr_VSA, OlympusRd_VSA],  (instregex "^[SU]SRAd$", "^[SU]SRA(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)_shift$")>;

// Dot product, matrix multiply

// SDOT (vector)
// UDOT (vector)
// USDOT (vector)
def : InstRW<[OlympusWr_VDOT, OlympusRd_VDOT], (instregex "^(S|U|US)DOT(v8i8|v16i8)$")>;
// SDOT (by element)
// SUDOT (by element)
// UDOT (by element)
// USDOT (by element)
def : InstRW<[OlympusWr_VDOT, OlympusRd_VDOT], (instregex "^(S|U|SU|US)DOTlane(v8i8|v16i8)$")>;
// SMMLA (vector)
// UMMLA (vector)
// USMMLA (vector)
def : InstRW<[OlympusWr_VMMA, OlympusRd_VMMA], (instregex "^(U|S|US)MMLA$")>;

// Arith reduce, 4H/4S
// ADDV
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^ADDV(v4i16|v4i32)v$")>;
// SADDLV
// UADDLV
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU]ADDLV(v4i16|v4i32)v$")>;
// SMAXV
// UMAXV
// SMINV
// UMINV
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU](MAX|MIN)V(v4i16|v4i32)v$")>;

// Arith reduce, 8B/8H
// ADDV
def : InstRW<[OlympusWrite_5c_1V_1V0123], (instregex "^ADDV(v8i8|v8i16)v$")>;
// SADDLV
// UADDLV
def : InstRW<[OlympusWrite_5c_1V_1V0123], (instregex "^[SU]ADDLV(v8i8|v8i16)v$")>;
// SMAXV
// UMAXV
// SMINV
// UMINV
def : InstRW<[OlympusWrite_5c_1V_1V0123], (instregex "^[SU](MAX|MIN)V(v8i8|v8i16)v$")>;

// Arith reduce, 16B
// ADDV
def : InstRW<[OlympusWrite_6c_2V0123], (instregex "^ADDVv16i8v$")>;
// SADDLV
// UADDLV
def : InstRW<[OlympusWrite_6c_2V0123], (instregex "^[SU]ADDLVv16i8v$")>;
// SMAXV
// UMAXV
// SMINV
// UMINV
def : InstRW<[OlympusWrite_6c_2V0123], (instregex "[SU](MAX|MIN)Vv16i8v$")>;

// Polynomial multiply
// PMUL
def : InstRW<[OlympusWrite_3c_1V], (instregex "^PMUL(v8i8|v16i8)$")>;

// Polynomial multiply long
// PMULL, PMULL2
def : InstRW<[OlympusWrite_2c_1V], (instregex "^PMULL(v8i8|v16i8)$")>;

// Reciprocal and square root estimate, 2S
// URECPE
// URSQRTE
def : InstRW<[OlympusWrite_3c_1V0123], (instrs URECPEv2i32, URSQRTEv2i32)>;

// Reciprocal and square root estimate, 4S
// URECPE
// URSQRTE
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs URECPEv4i32, URSQRTEv4i32)>;

// Table lookup, single or two register table
def : InstRW<[OlympusWrite_2c_1V], (instrs TBLv8i8One, TBLv16i8One,
                                           TBLv8i8Two, TBLv16i8Two)>;

// Table lookup, three register table
def : InstRW<[OlympusWrite_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// Table lookup, four register table
def : InstRW<[OlympusWrite_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// Table lookup extension, single register table
def : InstRW<[OlympusWrite_2c_1V], (instrs TBXv8i8One, TBXv16i8One)>;

// Table lookup extension, two register table
def : InstRW<[OlympusWrite_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// Table lookup extension, three register table
def : InstRW<[OlympusWrite_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// Table lookup extension, four register table
def : InstRW<[OlympusWrite_6c_5V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// Transfer vec to gen register
// SMOV
def : InstRW<[OlympusWrite_3c_2V03], (instregex "^SMOV(vi8to32|vi8to64|vi16to32|vi16to64|vi32to64)_idx0$")>;
def : InstRW<[OlympusWrite_3c_2V03], (instregex "^SMOV(vi8to32|vi16to32|vi8to64|vi16to64|vi32to64)$")>;
// UMOV
def : InstRW<[OlympusWrite_3c_2V03], (instregex "^UMOV(vi8|vi16|vi32|vi64)_idx0$")>;
def : InstRW<[OlympusWrite_3c_2V03], (instregex "^UMOV(vi8|vi16|vi32|vi64)$")>;

// Transfer gen to vec register
// DUP (general)
def : InstRW<[OlympusWrite_3c_1M], (instregex "^DUP(v8i8|v4i16|v2i32|v16i8|v8i16|v4i32|v2i64)gpr$")>;

// Insert gen to vec register
// INS (general)
def : InstRW<[OlympusWrite_5c_1M_1V], (instregex "^INS(vi8|vi16|vi32|vi64)gpr$")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// AES, SHA3 operations
// AESD
// AESE
// AESIMC
// AESMC
def : InstRW<[OlympusWrite_2c_1V], (instregex "^AES[DE]rr$", "^AESI?MCrr$")>;
// BCAX
// EOR3
// RAX1
// XAR
def : InstRW<[OlympusWrite_2c_1V], (instrs BCAX, EOR3, RAX1, XAR)>;

// Polynomial Multiply Long
def : InstRW<[OlympusWrite_2c_1V], (instrs PMULLv1i64, PMULLv2i64)>;

// SHA general
// SHA1H
// SHA1SU0
// SHA1SU1
def : InstRW<[OlympusWrite_2c_1V0], (instrs SHA1Hrr, SHA1SU0rrr, SHA1SU1rr)>;
// SHA256SU0
// SHA256SU1
def : InstRW<[OlympusWrite_2c_1V0], (instrs SHA256SU0rr, SHA256SU1rrr)>;
// SHA512H2
// SHA512H
// SHA512SU0
// SHA512SU1
def : InstRW<[OlympusWrite_2c_1V0], (instrs SHA512H2, SHA512H, SHA512SU0, SHA512SU1)>;

// SHA hash accelaration
// SHA1C
// SHA1M
// SHA1P
// SHA256H2
// SHA256H
def : InstRW<[OlympusWrite_4c_1V0], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// SM3 operations
def : InstRW<[OlympusWrite_2c_1V0123], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                          "^SM3TT[12][AB]$")>;

// SM4 operations
def : InstRW<[OlympusWrite_4c_1V0], (instrs SM4E, SM4ENCKEY)>;

// SIMD FP8 instructions
// -----------------------------------------------------------------------------

// Convert to/from F16
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^F[12]CVTL2?$",
                                                    "^FCVTN_F16v(8|16)f8$")>;

// Convert from F32
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTN_F32v8f8, FCVTN_F322v16f8)>;

// Dot product, multiply accumulate
def OlympusWr_F8MA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_F8MA : SchedReadAdvance<2, [OlympusWr_F8MA]>;
// FDOT (8-bit floating-point to half-precision, vector)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FDOTv(4|8)f16$")>;
// FDOT (8-bit floating-point to single-precision, vector)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FDOTv(2|4)f32$")>;
// FDOT (8-bit floating-point to half-precision, by element)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FDOTlanev(4|8)f16$")>;
// FDOT (8-bit floating-point to single-precision, by element)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FDOTlanev(2|4)f32$")>;
// FMLALB, FMLALT (vector)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FMLAL[BT]v16i8_v8f16$")>;
// FMLALLBB, FMLALLBT, FMLALLTB, FMLALLTT (vector)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FMLALL[BT][BT]v4f32$")>;
// FMLALB, FMLALT (by element)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FMLAL[BT]lanev8f16$")>;
// FMLALLBB, FMLALLBT, FMLALLTB, FMLALLTT (by element)
def : InstRW<[OlympusWr_F8MA, OlympusRd_F8MA], (instregex "^FMLALL[BT][BT]lanev4f32$")>;

// SIMD BF16 instructions
// -----------------------------------------------------------------------------

// Convert from F8
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^BF[12]CVTL2?$")>;

// Convert from F32, vector
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs BFCVTN, BFCVTN2)>;

// Convert from F32, scalar
def : InstRW<[OlympusWrite_3c_1V0123], (instrs BFCVT)>;

// Multiply accumulate
def OlympusWr_BF16MA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_BF16MA : SchedReadAdvance<2, [OlympusWr_BF16MA]>;
def : InstRW<[OlympusWr_BF16MA, OlympusRd_BF16MA], (instregex "^BFMLAL[BT](Idx)?$")>;

// Dot product
def OlympusWr_BF16DOT : SchedWriteRes<[OlympusUnitV]> { let Latency = 5; }
def OlympusRd_BF16DOT : SchedReadAdvance<2, [OlympusWr_BF16DOT]>;
def : InstRW<[OlympusWr_BF16DOT, OlympusRd_BF16DOT], (instregex "^BFDOTv[48]bf16$",
                                                                "^BF16DOTlanev[48]bf16$")>;

// Matrix multiply accumulate
def OlympusWr_BF16MMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 6; }
def OlympusRd_BF16MMA : SchedReadAdvance<2, [OlympusWr_BF16MMA]>;
def : InstRW<[OlympusWr_BF16MMA, OlympusRd_BF16MMA], (instrs BFMMLA)>;

// FP load instructions
// -----------------------------------------------------------------------------

// Load vector register, immediate
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LDR[BHSDQ]ui$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LDR[BHSDQ](pre|post)$")>;

// Load vector register, unscaled immediate
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LDUR[BHSDQ]i$")>;

// Load vector register, register
def : InstRW<[OlympusWrite_7c_1I_1L, ReadAdrBase], (instregex "^LDR[BHSDQ]ro[WX]$")>;

// Load vector register, literal
def : InstRW<[OlympusWrite_7c_1I_1L], (instregex "^LDR[SDQ]l$")>;

// Load vector pair, 128b
// Load vector pair non-temporal, 128b
def : InstRW<[OlympusWrite_6c_1L, OlympusWrite_6c_1L], (instrs LDPQi, LDNPQi)>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L, OlympusWrite_6c_1L], (instregex "^LDPQ(pre|post)$")>;

// Load vector pair, 32b/64b
// Load vector pair non-temporal, 32b/64b
def : InstRW<[OlympusWrite_6c_1L, OlympusWrite_6c], (instregex "^LDN?P[SD]i$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L, OlympusWrite_6c], (instregex "^LDP[SD](pre|post)$")>;

// FP store instructions
// -----------------------------------------------------------------------------

// Store vector register, immediate
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instregex "^STR[BHSDQ]ui$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123], (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector register, unscaled immediate
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instregex "^STUR[BHSDQ]i$")>;

// Store vector register, register
def : InstRW<[OlympusWrite_3c_1I_1SA_1V0123, ReadAdrBase], (instregex "^STR[BHSDQ]ro[WX]$")>;

// Store vector pair, 128 bit
// Store vector pair non-temporal, 128 bit
def : InstRW<[OlympusWrite_3c_1SA_1V0123, OlympusWrite_3c_1SA_1V0123], (instrs STPQi, STNPQi)>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123, OlympusWrite_3c_1SA_1V0123], (instregex "^STPQ(pre|post)$")>;

// Store vector pair, 32/64 bit
// Store vector pair non-temporal, 32/64 bit
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instregex "^STP[SD]i$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123], (instregex "^STN?P[SD](pre|post)$")>;

// SIMD load instructions
// -----------------------------------------------------------------------------

// Load 1 element, multiple, 1 register
def : InstRW<[OlympusWrite_6c_1L],           (instregex "^LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[OlympusWrite_6c_1L],           (instregex "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// Load 1 element, multiple, 2 registers, D form
def : InstRW<[OlympusWrite_6c_1L],           (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;

// Load 1 element, multiple, 2 registers, Q form
def : InstRW<[OlympusWrite_6c_2L],           (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_2L], (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// Load 1 element, multiple, 3 registers, D form
def : InstRW<[OlympusWrite_6c_2L],           (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_2L], (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;

// Load 1 element, multiple, 3 registers, Q form
def : InstRW<[OlympusWrite_6c_3L],           (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_3L], (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// Load 1 element, multiple, 4 registers, D form
def : InstRW<[OlympusWrite_6c_2L],           (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_2L], (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// Load 1 element, multiple, 4 registers, Q form
def : InstRW<[OlympusWrite_6c_4L],           (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_4L], (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// Load 1 element, one lane
def : InstRW<[OlympusWrite_8c_1L_1V],           (instregex "^LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_1V], (instregex "^LD1i(8|16|32|64)_POST$")>;

// Load 1 element, all lanes
def : InstRW<[OlympusWrite_6c_1L],           (instregex "^LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LD1Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[OlympusWrite_6c_1L],           (instregex "^LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_1L], (instregex "^LD1Rv(16b|8h|4s|2d)_POST$")>;

// Load 2 element, multiple, D form
def : InstRW<[OlympusWrite_8c_1L_2V],           (instregex "^LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_2V], (instregex "^LD2Twov(8b|4h|2s)_POST$")>;

// Load 2 element, multiple, Q form
def : InstRW<[OlympusWrite_8c_2L_2V],           (instregex "^LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_2V], (instregex "^LD2Twov(16b|8h|4s|2d)_POST$")>;

// Load 2 element, one lane
def : InstRW<[OlympusWrite_8c_1L_2V],           (instregex "^LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_2V], (instregex "^LD2i(8|16|32|64)_POST$")>;

// Load 2 element, all lanes
def : InstRW<[OlympusWrite_8c_1L_2V],           (instregex "^LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_2V], (instregex "^LD2Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[OlympusWrite_8c_1L_2V],           (instregex "^LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_2V], (instregex "^LD2Rv(16b|8h|4s|2d)_POST$")>;

// Load 3 element, multiple, D form
def : InstRW<[OlympusWrite_8c_2L_3V],           (instregex "^LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_3V], (instregex "^LD3Threev(8b|4h|2s)_POST$")>;

// Load 3 element, multiple, Q form
def : InstRW<[OlympusWrite_8c_3L_3V],           (instregex "^LD3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_3L_3V], (instregex "^LD3Threev(16b|8h|4s|2d)_POST$")>;

// Load 3 element, one lane
def : InstRW<[OlympusWrite_8c_2L_3V],           (instregex "^LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_3V], (instregex "^LD3i(8|16|32|64)_POST$")>;

// Load 3 element, all lanes
def : InstRW<[OlympusWrite_8c_2L_3V],           (instregex "^LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_3V], (instregex "^LD3Rv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[OlympusWrite_8c_2L_3V],           (instregex "^LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_3V], (instregex "^LD3Rv(16b|8h|4s|2d)_POST$")>;

// Load 4 element, multiple, D form
def : InstRW<[OlympusWrite_8c_2L_4V],           (instregex "^LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_4V], (instregex "^LD4Fourv(8b|4h|2s)_POST$")>;

// Load 4 element, multiple, Q form
def : InstRW<[OlympusWrite_9c_4L_8V],           (instregex "^LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_9c_4L_8V], (instregex "^LD4Fourv(16b|8h|4s|2d)_POST$")>;

// Load 4 element, one lane, B/H/S
def : InstRW<[OlympusWrite_8c_1L_4V],           (instregex "^LD4i(8|16|32)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_4V], (instregex "^LD4i(8|16|32)_POST$")>;

// Load 4 element, one lane, D
def : InstRW<[OlympusWrite_8c_2L_4V],           (instrs LD4i64)>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_4V], (instrs LD4i64_POST)>;

// Load 4 element, all lanes, B/H/S
def : InstRW<[OlympusWrite_8c_1L_4V],           (instregex "^LD4Rv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_4V], (instregex "^LD4Rv(8b|4h|2s)_POST$")>;
def : InstRW<[OlympusWrite_8c_1L_4V],           (instregex "^LD4Rv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_1L_4V], (instregex "^LD4Rv(16b|8h|4s)_POST$")>;

// Load 4 element, all lanes, D
def : InstRW<[OlympusWrite_8c_2L_4V],           (instregex "^LD4Rv[12]d$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_2L_4V], (instregex "^LD4Rv[12]d_POST$")>;

// SIMD store instructions
// -----------------------------------------------------------------------------

// Store 1 element, multiple, 1 register
def : InstRW<[OlympusWrite_3c_1SA_1V0123],           (instregex "^ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123], (instregex "^ST1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[OlympusWrite_3c_1SA_1V0123],           (instregex "^ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123], (instregex "^ST1Onev(16b|8h|4s|2d)_POST$")>;

// Store 1 element, multiple, 2 registers, D form
def : InstRW<[OlympusWrite_3c_1SA_1V0123],           (instregex "^ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_1SA_1V0123], (instregex "^ST1Twov(8b|4h|2s|1d)_POST$")>;

// Store 1 element, multiple, 2 registers, Q form
def : InstRW<[OlympusWrite_3c_2SA_2V0123],           (instregex "^ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_3c_2SA_2V0123], (instregex "^ST1Twov(16b|8h|4s|2d)_POST$")>;

// Store 1 element, multiple, 3 registers, D form
def : InstRW<[OlympusWrite_4c_2SA_2V0123],           (instregex "^ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_2SA_2V0123], (instregex "^ST1Threev(8b|4h|2s|1d)_POST$")>;

// Store 1 element, multiple, 3 registers, Q form
def : InstRW<[OlympusWrite_4c_3SA_3V0123],           (instregex "^ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_3SA_3V0123], (instregex "^ST1Threev(16b|8h|4s|2d)_POST$")>;

// Store 1 element, multiple, 4 registers, D form
def : InstRW<[OlympusWrite_4c_2SA_2V0123],           (instregex "^ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_2SA_2V0123], (instregex "^ST1Fourv(8b|4h|2s|1d)_POST$")>;

// Store 1 element, multiple, 4 registers, Q form
def : InstRW<[OlympusWrite_4c_4SA_4V0123],           (instregex "^ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_4c_4SA_4V0123], (instregex "^ST1Fourv(16b|8h|4s|2d)_POST$")>;

// Store 1 element, one lane
def : InstRW<[OlympusWrite_5c_1SA_1V_1V0123],           (instregex "^ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_1SA_1V_1V0123], (instregex "^ST1i(8|16|32|64)_POST$")>;

// Store 2 element, multiple, D form
def : InstRW<[OlympusWrite_5c_1SA_1V_1V0123],           (instregex "^ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_1SA_1V_1V0123], (instregex "^ST2Twov(8b|4h|2s)_POST$")>;

// Store 2 element, multiple, Q form
def : InstRW<[OlympusWrite_5c_2SA_2V_2V0123],           (instregex "^ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_2SA_2V_2V0123], (instregex "^ST2Twov(16b|8h|4s|2d)_POST$")>;

// Store 2 element, one lane
def : InstRW<[OlympusWrite_5c_1SA_1V_1V0123],           (instregex "^ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_1SA_1V_1V0123], (instregex "^ST2i(8|16|32|64)_POST$")>;

// Store 3 element, multiple, D form
def : InstRW<[OlympusWrite_5c_2SA_2V_2V0123],           (instregex "^ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_2SA_2V_2V0123], (instregex "^ST3Threev(8b|4h|2s)_POST$")>;

// Store 3 element, multiple, Q form
def : InstRW<[OlympusWrite_6c_3SA_3V_3V0123],           (instregex "^ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, OlympusWrite_6c_3SA_3V_3V0123], (instregex "^ST3Threev(16b|8h|4s|2d)_POST$")>;

// Store 3 element, one lane
def : InstRW<[OlympusWrite_5c_2SA_2V_2V0123],           (instregex "^ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, OlympusWrite_5c_2SA_2V_2V0123], (instregex "^ST3i(8|16|32|64)_POST$")>;

// Store 4 element, multiple, D form
def : InstRW<[OlympusWrite_7c_2SA_4V_2V0123],           (instregex "^ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_7c_2SA_4V_2V0123], (instregex "^ST4Fourv(8b|4h|2s)_POST$")>;

// Store 4 element, multiple, Q form, D element
def : InstRW<[OlympusWrite_6c_4SA_4V_4V0123],           (instrs ST4Fourv2d)>;
def : InstRW<[WriteAdr, OlympusWrite_6c_4SA_4V_4V0123], (instrs ST4Fourv2d_POST)>;

// Store 4 element, multiple, Q form, B/H/S element
def : InstRW<[OlympusWrite_8c_4SA_8V_4V0123],           (instregex "^ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, OlympusWrite_8c_4SA_8V_4V0123], (instregex "^ST4Fourv(16b|8h|4s)_POST$")>;

// Store 4 element, one lane, D
def : InstRW<[OlympusWrite_5c_2SA_2V_2V0123],           (instrs ST4i64)>;
def : InstRW<[WriteAdr, OlympusWrite_5c_2SA_2V_2V0123], (instrs ST4i64_POST)>;

// Store 4 element, one lane, B/H/S
def : InstRW<[OlympusWrite_7c_1SA_2V_1V0123],           (instregex "^ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, OlympusWrite_7c_1SA_2V_1V0123], (instregex "^ST4i(8|16|32)_POST$")>;

// SVE Predicate instructions
// -----------------------------------------------------------------------------

// Predicate logical
// Predicate logical, flag setting
def : InstRW<[OlympusWrite_1c_1M], (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)S?_PPzPP")>;
// Loop terminate
def : InstRW<[OlympusWrite_1c_1M], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;
// Predicate test
def : InstRW<[OlympusWrite_1c_1M], (instrs PTEST_PP, PTEST_PP_ANY, PTEST_PP_FIRST)>;
// Predicate select
def : InstRW<[OlympusWrite_1c_1M], (instrs SEL_PPPP)>;

// Control, counting
// Predicate counting scalar
def : InstRW<[OlympusWrite_2c_1M], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;
def : InstRW<[OlympusWrite_2c_1M],
             (instregex "^(CNT|SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI",
                        "^UQ(DEC|INC)[BHWD]_WPiI")>;
// Loop control, based on predicate
def : InstRW<[OlympusWrite_2c_1M], (instrs BRKA_PPmP, BRKA_PPzP,
                                           BRKB_PPmP, BRKB_PPzP)>;
// Loop control, based on predicate and flag setting
def : InstRW<[OlympusWrite_2c_1M], (instrs BRKAS_PPzP, BRKBS_PPzP)>;
// Loop control, propagating
def : InstRW<[OlympusWrite_2c_1M], (instrs BRKN_PPzP, BRKPA_PPzPP,
                                           BRKPB_PPzPP)>;
// Loop control, propagating and flag setting
def : InstRW<[OlympusWrite_2c_1M], (instrs BRKNS_PPzP, BRKPAS_PPzPP,
                                           BRKPBS_PPzPP)>;
// Predicate counting scalar, active predicate
def : InstRW<[OlympusWrite_2c_1M],
             (instregex "^CNTP_XPP_[BHSD]",
                        "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;
// Predicate find first/next
def : InstRW<[OlympusWrite_2c_1M], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;
// Loop control, based on GPR
def : InstRW<[OlympusWrite_2c_1M],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]")>;
def : InstRW<[OlympusWrite_2c_1M], (instregex "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Misc processing
// Predicate set
def : InstRW<[OlympusWrite_2c_1M], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;
// Predicate set/initialize, set flags
def : InstRW<[OlympusWrite_2c_1M], (instregex "^PTRUES_[BHSD]")>;
// Predicate unpack and widen
def : InstRW<[OlympusWrite_2c_1M], (instrs PUNPKHI_PP, PUNPKLO_PP)>;
// Predicate reverse
def : InstRW<[OlympusWrite_2c_1M], (instregex "^REV_PP_[BHSD]")>;
// Predicate transpose
def : InstRW<[OlympusWrite_2c_1M], (instregex "^TRN[12]_PPP_[BHSD]")>;
// Predicate zip/unzip
def : InstRW<[OlympusWrite_2c_1M], (instregex "^(ZIP|UZP)[12]_PPP_[BHSD]")>;

// Predicate counting scalar, ALL #{1,2,4,8}
// Predicate counting scalar, other
def : InstRW<[OlympusWrite_IncDec], (instregex "^(DEC|INC)[BHWD]_XPiI")>;

// Predicate counting vector
def : InstRW<[OlympusWrite_7c_2M_1V],
             (instregex "^([SU]Q)?(DEC|INC)P_ZP_[HSD]")>;

// Read first fault register, unpredicated
def : InstRW<[OlympusWrite_2c_1M0], (instrs RDFFR_P)>;

// Read first fault register, predicated
// Read first fault register and set flags
def : InstRW<[OlympusWrite_3c_1M_1M0], (instrs RDFFR_PPz, RDFFRS_PPz)>;

// Write/Set first fault register
def : InstRW<[OlympusWrite_1c_1M0], (instrs WRFFR, SETFFR)>;

// SVE floating-point instructions
// -----------------------------------------------------------------------------

// FP arithmetic
// FABD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABD_ZP[mZ]Z_[HSD]")>;
// FABS
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FABS_ZPmZ_[HSD]")>;
// FADD (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADD_ZP[mZ]I_[HSD]")>;
// FADD (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADD_ZP[mZ]Z_[HSD]")>;
// FADDP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADDP_ZPmZZ_[HSD]")>;
// FADD (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FADD_ZZZ_[HSD]")>;
// FAMAX
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FAMAX_ZP[mZ]Z_[HSD]")>;
// FMAX (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAX_ZP[mZ]I_[HSD]")>;
// FMAX (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAX_ZP[mZ]Z_[HSD]")>;
// FMAXNM (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAXNM_ZP[mZ]I_[HSD]")>;
// FMAXNM (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAXNM_ZP[mZ]Z_[HSD]")>;
// FMAXNMP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAXNMP_ZPmZZ_[HSD]")>;
// FMAXP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMAXP_ZPmZZ_[HSD]")>;
// FAMIN
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FAMIN_ZP[mZ]Z_[HSD]")>;
// FMIN (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMIN_ZP[mZ]I_[HSD]")>;
// FMIN (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMIN_ZP[mZ]Z_[HSD]")>;
// FMINNM (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMINNM_ZP[mZ]I_[HSD]")>;
// FMINNM (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMINNM_ZP[mZ]Z_[HSD]")>;
// FMINNMP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMINNMP_ZPmZZ_[HSD]")>;
// FMINP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FMINP_ZPmZZ_[HSD]")>;
// FCPY
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FCPY_ZPmI_[HSD]")>;
// FDUP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FDUP_ZI_[HSD]")>;
// FNEG
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FNEG_ZPmZ_[HSD]")>;
// FSUB (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUB_ZP[mZ]I_[HSD]")>;
// FSUB (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUB_ZP[mZ]Z_[HSD]")>;
// FSUBR (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUBR_ZP[mZ]I_[HSD]")>;
// FSUBR (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUBR_ZP[mZ]Z_[HSD]")>;
// FSUB (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^FSUB_ZZZ_[HSD]")>;

// FP associative add, F16
def : InstRW<[OlympusWrite_11c_1V03_10], (instrs FADDA_VPZ_H)>;

// FP associative add, F32
def : InstRW<[OlympusWrite_7c_1V03_6], (instrs FADDA_VPZ_S)>;

// FP associative add, F64
def : InstRW<[OlympusWrite_4c_2V], (instrs FADDA_VPZ_D)>;

// FP compare
// FAC<cc>
def : InstRW<[OlympusWrite_2c_1V03], (instregex "^FACG[ET]_PPzZZ_[HSD]")>;
// FCM<cc> (zero)
def : InstRW<[OlympusWrite_2c_1V03], (instregex "^FCM(EQ|GE|GT|LE|LT|NE)_PPzZ0_[HSD]")>;
// FCM<cc> (vectors)
def : InstRW<[OlympusWrite_2c_1V03], (instregex "^FCM(EQ|GE|GT|NE|UO)_PPzZZ_[HSD]")>;

// FP multiply
def OlympusWr_ZFMul : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
// FMUL (immediate)
def : InstRW<[OlympusWr_ZFMul], (instregex "^FMUL_ZP[mZ]I_[HSD]")>;
// FMUL (vectors, predicated)
def : InstRW<[OlympusWr_ZFMul], (instregex "^FMUL_ZP[mZ]Z_[HSD]")>;
// FMULX
def : InstRW<[OlympusWr_ZFMul], (instregex "^FMULX_ZP[mZ]Z_[HSD]")>;
// FMUL (vectors, unpredicated)
def : InstRW<[OlympusWr_ZFMul], (instregex "^FMUL_ZZZ_[HSD]")>;
// FMUL (indexed)
def : InstRW<[OlympusWr_ZFMul], (instregex "^FMUL_ZZZI_[HSD]")>;

// FP misc
// FCADD
def : InstRW<[OlympusWrite_3c_1V], (instregex "^FCADD_ZPmZ_[HSD]")>;
// FSCALE
def : InstRW<[OlympusWrite_3c_1V], (instregex "^FSCALE_ZPmZ_[HSD]")>;
// FTSMUL
def : InstRW<[OlympusWrite_3c_1V], (instregex "^FTSMUL_ZZZ_[HSD]")>;
// FTSSEL
def : InstRW<[OlympusWrite_3c_1V], (instregex "^FTSSEL_ZZZ_[HSD]")>;

// FP multiply accumulate
def OlympusWr_ZFMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_ZFMA : SchedReadAdvance<0, [OlympusWr_ZFMul, OlympusWr_ZFMA], [1, 2]>;
// FMLA (vectors)
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FMLA_ZP[mZ]ZZ_[HSD]")>;
// FMLS (vectors)
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FMLS_ZP[mZ]ZZ_[HSD]")>;
// FNMLA
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FNMLA_ZP[mZ]ZZ_[HSD]")>;
// FNMLS
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FNMLS_ZP[mZ]ZZ_[HSD]")>;
// FMLA (indexed)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instregex "^FMLA_ZZZI_[HSD]")>;
// FMLALB (vectors, FP16 to FP32)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLALB_ZZZ_SHH)>;
// FMLALB (indexed, FP16 to FP32)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLALB_ZZZI_SHH)>;
// FMLALT (vectors, FP16 to FP32)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLALT_ZZZ_SHH)>;
// FMLALT (indexed, FP16 to FP32)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLALT_ZZZI_SHH)>;
// FMLS (indexed)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instregex "^FMLS_ZZZI_[HSD]")>;
// FMLSLB (vectors)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLSLB_ZZZ_SHH)>;
// FMLSLB (indexed)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLSLB_ZZZI_SHH)>;
// FMLSLT (vectors)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLSLT_ZZZ_SHH)>;
// FMLSLT (indexed)
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instrs FMLSLT_ZZZI_SHH)>;
// FMAD
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FMAD_ZPmZZ_[HSD]")>;
// FNMAD
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FNMAD_ZPmZZ_[HSD]")>;
// FMSB
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FMSB_ZPmZZ_[HSD]")>;
// FNMSB
def : InstRW<[OlympusWr_ZFMA, ReadDefault, OlympusRd_ZFMA], (instregex "^FNMSB_ZPmZZ_[HSD]")>;
// FRECPS
def : InstRW<[OlympusWrite_4c_1V], (instregex "^FRECPS_ZZZ_[HSD]")>;
// FRSQRTS
def : InstRW<[OlympusWrite_4c_1V], (instregex "^FRSQRTS_ZZZ_[HSD]")>;
// FTMAD
def : InstRW<[OlympusWr_ZFMA, OlympusRd_ZFMA], (instregex "^FTMAD_ZZI_[HSD]")>;

// FP complex multiply accumulate
def OlympusWr_ZFCMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 5; }
def OlympusRd_ZFCMA : SchedReadAdvance<3, [OlympusWr_ZFCMA]>;
// FCMLA (vectors)
def : InstRW<[OlympusWr_ZFCMA, ReadDefault, OlympusRd_ZFCMA], (instregex "^FCMLA_ZPmZZ_[HSD]")>;
// FCMLA (indexed)
def : InstRW<[OlympusWr_ZFCMA, OlympusRd_ZFCMA], (instregex "^FCMLA_ZZZI_[HS]")>;

// FP convert, to/from F64
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                                  "^FCVTLT_ZPmZ_StoD",
                                                  "^FCVTNT_ZPmZ_DtoS")>;
// FCVTX
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTX_ZPmZ_DtoS)>;
// FCVTXNT
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTXNT_ZPmZ_DtoS)>;

// FP convert, F32 to/from F16
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                                    "^FCVTLT_ZPmZ_HtoS",
                                                    "^FCVTNT_ZPmZ_StoH")>;

// FP logarithm, estimates, round, int convert, F16
// FCVTZS
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FCVTZS_ZPmZ_HtoH")>;
// FCVTZU
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FCVTZU_ZPmZ_HtoH")>;
// FLOGB
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FLOGB_ZP[mZ]Z_H")>;
// FRECPE
def : InstRW<[OlympusWrite_6c_1V0123_4], (instrs FRECPE_ZZ_H)>;
// FRECPX
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FRECPX_ZPmZ_H")>;
// FRSQRTE
def : InstRW<[OlympusWrite_6c_1V0123_4], (instrs FRSQRTE_ZZ_H)>;
// FRINT<r>
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// FP logarithm, estimates, round, int convert, F32
// FCVTZS
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVTZS_ZPmZ_(HtoS|StoS)")>;
// FCVTZU
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FCVTZU_ZPmZ_(HtoS|StoS)")>;
// FLOGB
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FLOGB_ZP[mZ]Z_S")>;
// FRECPE
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs FRECPE_ZZ_S)>;
// FRECPX
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRECPX_ZPmZ_S")>;
// FRSQRTE
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs FRSQRTE_ZZ_S)>;
// FRINT<r>
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// FP logarithm, estimates, round, int convert, F64
// FCVTZS
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTZS_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;
// FCVTZU
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FCVTZU_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;
// FLOGB
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FLOGB_ZP[mZ]Z_D")>;
// FRECPE
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FRECPE_ZZ_D)>;
// FRECPX
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRECPX_ZPmZ_D")>;
// FRSQRTE
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FRSQRTE_ZZ_D)>;
// FRINT<r>
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// FP exponent
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^FEXPA_ZZ_[HSD]")>;

// FP divide, sqrt, F16
def : InstRW<[OlympusWrite_13c_1V12_8], (instregex "^FDIVR?_ZP[mZ]Z_H", "^FSQRT_ZPmZ_H")>;

// FP divide, sqrt, F32
def : InstRW<[OlympusWrite_11c_1V12_4], (instregex "^FDIVR?_ZP[mZ]Z_S", "^FSQRT_ZPmZ_S")>;

// FP divide, sqrt, F64
def : InstRW<[OlympusWrite_14c_1V12_2], (instregex "^FDIVR?_ZP[mZ]Z_D", "^FSQRT_ZPmZ_D")>;

// FP reduction, F16
def : InstRW<[OlympusWrite_8c_4V], (instregex "^(FADD|FMAX(NM)?|FMIN(NM)?)V_VPZ_H")>;

// FP reduction, F32
def : InstRW<[OlympusWrite_6c_3V], (instregex "^(FADD|FMAX(NM)?|FMIN(NM)?)V_VPZ_S")>;

// FP reduction, F64
def : InstRW<[OlympusWrite_4c_2V], (instregex "^(FADD|FMAX(NM)?|FMIN(NM)?)V_VPZ_D")>;

// SVE FP8 instructions
// -----------------------------------------------------------------------------

// Convert to/from F16
// F1CVT, F2CVT
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^F[12]CVT_ZZ_BtoH")>;
// F1CVTLT, F2CVTLT
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^F[12]CVTLT_ZZ_BtoH")>;
// FCVTN
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs FCVTN_Z2Z_HtoB)>;

// Convert from F32
// FCVTNB
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTNB_Z2Z_StoB)>;
// FCVTNT (unpredicated)
def : InstRW<[OlympusWrite_3c_1V0123], (instrs FCVTNT_Z2Z_StoB)>;

// Dot product, multiply accumulate
def OlympusWr_ZF8MA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_ZF8MA : SchedReadAdvance<2, [OlympusWr_ZF8MA]>;
// FDOT (4-way, vectors)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FDOT_ZZZ_BtoS)>;
// FDOT (4-way, indexed)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FDOT_ZZZI_BtoS)>;
// FDOT (2-way, vectors, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FDOT_ZZZ_BtoH)>;
// FDOT (2-way, indexed, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FDOT_ZZZI_BtoH)>;
// FMLALB (vectors, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALB_ZZZ)>;
// FMLALB (indexed, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALB_ZZZI)>;
// FMLALLBB (vectors)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLBB_ZZZ)>;
// FMLALLBB (indexed)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLBB_ZZZI)>;
// FMLALLBT (vectors)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLBT_ZZZ)>;
// FMLALLBT (indexed)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLBT_ZZZI)>;
// FMLALLTB (vectors)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLTB_ZZZ)>;
// FMLALLTB (indexed)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLTB_ZZZI)>;
// FMLALLTT (vectors)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLTT_ZZZ)>;
// FMLALLTT (indexed)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALLTT_ZZZI)>;
// FMLALT (vectors, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALT_ZZZ)>;
// FMLALT (indexed, FP8 to FP16)
def : InstRW<[OlympusWr_ZF8MA, OlympusRd_ZF8MA], (instrs FMLALT_ZZZI)>;

// SVE BF16 instructions
// -----------------------------------------------------------------------------

// Convert to/from F8
// BF1CVT, BF2CVT
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^BF[12]CVT_ZZ_BtoH")>;
// BF1CVTLT, BF2CVTLT
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^BF[12]CVTLT_ZZ_BtoH")>;
// BFCVTN
def : InstRW<[OlympusWrite_4c_1V0123_2], (instrs BFCVTN_Z2Z_HtoB)>;

// Convert from F32
def : InstRW<[OlympusWrite_4c_1V0123], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Multiply accumulate
def : InstRW<[OlympusWr_ZBFMAL, OlympusRd_ZBFMAL], (instregex "^BFMLAL[BT]_ZZZI?")>;

// Dot product
def : InstRW<[OlympusWr_ZBFDOT, OlympusRd_ZBFDOT], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[OlympusWr_ZBFMMA, OlympusRd_ZBFMMA], (instrs BFMMLA_ZZZ_HtoS)>;

// SVE integer instructions
// -----------------------------------------------------------------------------

// General
// ABS
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ABS_ZPmZ_[BHSD]")>;
// ADCLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADCLB_ZZZ_[SD]")>;
// ADCLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADCLT_ZZZ_[SD]")>;
// ADD (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADD_ZP[mZ]Z_[BHSD]")>;
// ADDP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADDP_ZPmZ_[BHSD]")>;
// ADD (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADD_ZI_[BHSD]")>;
// ADD (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADD_ZZZ_[BHSD]")>;
// CADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CADD_ZZI_[BHSD]")>;
// ADDHNB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADDHNB_ZZZ_[BHS]")>;
// ADDHNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADDHNT_ZZZ_[BHS]")>;
// RADDHNB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^RADDHNB_ZZZ_[BHS]")>;
// RADDHNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^RADDHNT_ZZZ_[BHS]")>;
// ADR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ADR_LSL_ZZZ_[SD]_[0123]",
                                              "^ADR_[SU]XTW_ZZZ_D_[0123]")>;
// AND (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^AND_ZP[mZ]Z_[BHSD]")>;
// AND (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instrs AND_ZI)>;
// AND (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instrs AND_ZZZ)>;
// ASR (immediate, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASR_ZP[mZ]I_[BHSD]")>;
// ASR (wide elements, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASR_WIDE_ZPmZ_[BHS]")>;
// ASR (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASR_ZP[mZ]Z_[BHSD]")>;
// ASRR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASRR_ZPmZ_[BHSD]")>;
// ASR (immediate, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASR_ZZI_[BHSD]")>;
// ASR (wide elements, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ASR_WIDE_ZZZ_[BHS]")>;
// BIC (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^BIC_ZP[mZ]Z_[BHSD]")>;
// BIC (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instrs BIC_ZZZ)>;
// BSL1N
def : InstRW<[OlympusWrite_2c_1V], (instrs BSL1N_ZZZZ)>;
// BSL2N
def : InstRW<[OlympusWrite_2c_1V], (instrs BSL2N_ZZZZ)>;
// BSL
def : InstRW<[OlympusWrite_2c_1V], (instrs BSL_ZZZZ)>;
// CLS
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CLS_ZPmZ_[BHSD]")>;
// CLZ
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CLZ_ZPmZ_[BHSD]")>;
// CNOT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CNOT_ZPmZ_[BHSD]")>;
// CNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CNT_ZPmZ_[BHSD]")>;
// CPY (immediate, zeroing)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CPY_ZPzI_[BHSD]")>;
// CPY (immediate, merging)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CPY_ZPmI_[BHSD]")>;
// CPY (SIMD&FP scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^CPY_ZPmV_[BHSD]")>;
// DECD, DECH, DECW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^DEC[HWD]_ZPiI")>;
// DUP (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^DUP_ZI_[BHSD]")>;
// DUP (indexed)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^DUP_ZZI_[BHSDQ]")>;
// DUPM
def : InstRW<[OlympusWrite_2c_1V], (instrs DUPM_ZI)>;
// EOR (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^EOR_ZP[mZ]Z_[BHSD]")>;
// EOR (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instrs EOR_ZI)>;
// EOR (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instrs EOR_ZZZ)>;
// EORBT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^EORBT_ZZZ_[BHSD]")>;
// EORTB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^EORTB_ZZZ_[BHSD]")>;
// EXT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^EXT_ZZI")>;
// HISTCNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^HISTCNT_ZPzZZ_[SD]")>;
// HISTSEG
def : InstRW<[OlympusWrite_2c_1V], (instrs HISTSEG_ZZZ)>;
// INCD, INCH, INCW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^INC[HWD]_ZPiI")>;
// INSR (SIMD&FP scalar)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^INSR_ZV_[BHSD]")>;
// LSL (immediate, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSL_ZP[mZ]I_[BHSD]")>;
// LSL (wide elements, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSL_WIDE_ZPmZ_[BHS]")>;
// LSL (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSL_ZP[mZ]Z_[BHSD]")>;
// LSLR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSLR_ZPmZ_[BHSD]")>;
// LSR (immediate, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSR_ZP[mZ]I_[BHSD]")>;
// LSR (wide elements, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSR_WIDE_ZPmZ_[BHS]")>;
// LSR (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSR_ZP[mZ]Z_[BHSD]")>;
// LSRR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSRR_ZPmZ_[BHSD]")>;
// LSL (immediate, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSL_ZZI_[BHSD]")>;
// LSL (wide elements, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSL_WIDE_ZZZ_[BHS]")>;
// LSR (immediate, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSR_ZZI_[BHSD]")>;
// LSR (wide elements, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LSR_WIDE_ZZZ_[BHS]")>;
// LUTI2
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LUTI2_ZZZI_[BH]")>;
// LUTI4
def : InstRW<[OlympusWrite_2c_1V], (instregex "^LUTI4_ZZZI_[BH]",
                                              "^LUTI4_Z2ZZI")>;
// MOVPRFX (predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]")>;
// MOVPRFX (unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instrs MOVPRFX_ZZ)>;
// NBSL
def : InstRW<[OlympusWrite_2c_1V], (instrs NBSL_ZZZZ)>;
// NEG
def : InstRW<[OlympusWrite_2c_1V], (instregex "^NEG_ZPmZ_[BHSD]")>;
// NOT (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^NOT_ZPmZ_[BHSD]")>;
// ORR (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ORR_ZP[mZ]Z_[BHSD]")>;
// ORR (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instrs ORR_ZI)>;
// ORR (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instrs ORR_ZZZ)>;
// PMUL
def : InstRW<[OlympusWrite_2c_1V], (instrs PMUL_ZZZ_B)>;
// PMULLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^PMULLB_ZZZ_[HDQ]")>;
// PMULLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^PMULLT_ZZZ_[HDQ]")>;
// RBIT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^RBIT_ZPmZ_[BHSD]")>;
// REV (vector)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^REV_ZZ_[BHSD]")>;
// REVB, REVH, REVW
def : InstRW<[OlympusWrite_2c_1V], (instregex "^REVB_ZPmZ_[HSD]",
                                              "^REVH_ZPmZ_[SD]",
                                              "^REVW_ZPmZ_D")>;
// SBCLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SBCLB_ZZZ_[SD]")>;
// SBCLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SBCLT_ZZZ_[SD]")>;
// SEL (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SEL_ZPZZ_[BHSD]")>;
// SHRNB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SHRNB_ZZI_[BHS]")>;
// SHRNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SHRNT_ZZI_[BHS]")>;
// SLI
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SLI_ZZI_[BHSD]")>;
// SRI
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SRI_ZZI_[BHSD]")>;
// SUB (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUB_ZP[mZ]Z_[BHSD]")>;
// SUBR (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUBR_ZP[mZ]Z_[BHSD]")>;
// SUB (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUB_ZI_[BHSD]")>;
// SUB (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUB_ZZZ_[BHSD]")>;
// SUBR (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUBR_ZI_[BHSD]")>;
// RSUBHNB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^RSUBHNB_ZZZ_[BHS]")>;
// RSUBHNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^RSUBHNT_ZZZ_[BHS]")>;
// SUBHNB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUBHNB_ZZZ_[BHS]")>;
// SUBHNT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUBHNT_ZZZ_[BHS]")>;
// TBL
def : InstRW<[OlympusWrite_2c_1V], (instregex "^TBL_ZZZZ?_[BHSD]")>;
// TBX
def : InstRW<[OlympusWrite_2c_1V], (instregex "^TBX_ZZZ_[BHSD]")>;
// TRN1, TRN2 (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^TRN[12]_ZZZ_[BHSD]")>;
// UZP1, UZP2 (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UZP[12]_ZZZ_[BHSD]")>;
// ZIP1, ZIP2 (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^ZIP[12]_ZZZ_[BHSD]")>;
// SABD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SABD_ZP[mZ]Z_[BHSD]")>;
// UABD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UABD_ZP[mZ]Z_[BHSD]")>;
// SABDLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SABDLB_ZZZ_[HSD]")>;
// SABDLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SABDLT_ZZZ_[HSD]")>;
// UABDLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UABDLB_ZZZ_[HSD]")>;
// UABDLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UABDLT_ZZZ_[HSD]")>;
// SADDLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SADDLB_ZZZ_[HSD]")>;
// SADDLBT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SADDLBT_ZZZ_[HSD]")>;
// SADDLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SADDLT_ZZZ_[HSD]")>;
// SADDWB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SADDWB_ZZZ_[HSD]")>;
// SADDWT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SADDWT_ZZZ_[HSD]")>;
// UADDLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UADDLB_ZZZ_[HSD]")>;
// UADDLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UADDLT_ZZZ_[HSD]")>;
// UADDWB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UADDWB_ZZZ_[HSD]")>;
// UADDWT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UADDWT_ZZZ_[HSD]")>;
// SHADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SHADD_ZPmZ_[BHSD]")>;
// SRHADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SRHADD_ZPmZ_[BHSD]")>;
// UHADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UHADD_ZPmZ_[BHSD]")>;
// URHADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^URHADD_ZPmZ_[BHSD]")>;
// SHSUB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SHSUB_ZP[mZ]Z_[BHSD]")>;
// SHSUBR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SHSUBR_ZPmZ_[BHSD]")>;
// UHSUB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UHSUB_ZP[mZ]Z_[BHSD]")>;
// UHSUBR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UHSUBR_ZPmZ_[BHSD]")>;
// SMAX (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMAX_ZP[mZ]Z_[BHSD]")>;
// SMAXP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMAXP_ZPmZ_[BHSD]")>;
// UMAX (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMAX_ZP[mZ]Z_[BHSD]")>;
// UMAXP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMAXP_ZPmZ_[BHSD]")>;
// SMAX (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMAX_ZI_[BHSD]")>;
// UMAX (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMAX_ZI_[BHSD]")>;
// SMIN (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMIN_ZP[mZ]Z_[BHSD]")>;
// SMINP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMINP_ZPmZ_[BHSD]")>;
// UMIN (vectors)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMIN_ZP[mZ]Z_[BHSD]")>;
// UMINP
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMINP_ZPmZ_[BHSD]")>;
// SMIN (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SMIN_ZI_[BHSD]")>;
// UMIN (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UMIN_ZI_[BHSD]")>;
// SQADD (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQADD_ZPmZ_[BHSD]")>;
// UQADD (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQADD_ZPmZ_[BHSD]")>;
// USQADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USQADD_ZPmZ_[BHSD]")>;
// SQABS
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQABS_ZPmZ_[BHSD]")>;
// SQADD (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQADD_ZI_[BHSD]")>;
// SQADD (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQADD_ZZZ_[BHSD]")>;
// SQCADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQCADD_ZZI_[BHSD]")>;
// UQADD (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQADD_ZI_[BHSD]")>;
// UQADD (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQADD_ZZZ_[BHSD]")>;
// SQDECD (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQDECD_ZPiI)>;
// SQDECH (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQDECH_ZPiI)>;
// SQDECW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQDECW_ZPiI)>;
// UQDECD (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQDECD_ZPiI)>;
// UQDECH (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQDECH_ZPiI)>;
// UQDECW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQDECW_ZPiI)>;
// SQINCD (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQINCD_ZPiI)>;
// SQINCH (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQINCH_ZPiI)>;
// SQINCW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs SQINCW_ZPiI)>;
// UQINCD (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQINCD_ZPiI)>;
// UQINCH (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQINCH_ZPiI)>;
// UQINCW (vector)
def : InstRW<[OlympusWrite_2c_1V], (instrs UQINCW_ZPiI)>;
// SQNEG
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQNEG_ZPmZ_[BHSD]")>;
// SQSUB (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQSUB_ZPmZ_[BHSD]")>;
// SQSUBR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQSUBR_ZPmZ_[BHSD]")>;
// UQSUB (vectors, predicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQSUB_ZPmZ_[BHSD]")>;
// UQSUBR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQSUBR_ZPmZ_[BHSD]")>;
// SQSUB (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQSUB_ZI_[BHSD]")>;
// SQSUB (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SQSUB_ZZZ_[BHSD]")>;
// UQSUB (immediate)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQSUB_ZI_[BHSD]")>;
// UQSUB (vectors, unpredicated)
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UQSUB_ZZZ_[BHSD]")>;
// SSHLLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSHLLB_ZZI_[HSD]")>;
// SSHLLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSHLLT_ZZI_[HSD]")>;
// USHLLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USHLLB_ZZI_[HSD]")>;
// USHLLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USHLLT_ZZI_[HSD]")>;
// SSUBLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBLB_ZZZ_[HSD]")>;
// SSUBLBT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBLBT_ZZZ_[HSD]")>;
// SSUBLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBLT_ZZZ_[HSD]")>;
// SSUBLTB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBLTB_ZZZ_[HSD]")>;
// SSUBWB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBWB_ZZZ_[HSD]")>;
// SSUBWT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SSUBWT_ZZZ_[HSD]")>;
// USUBLB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USUBLB_ZZZ_[HSD]")>;
// USUBLT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USUBLT_ZZZ_[HSD]")>;
// USUBWB
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USUBWB_ZZZ_[HSD]")>;
// USUBWT
def : InstRW<[OlympusWrite_2c_1V], (instregex "^USUBWT_ZZZ_[HSD]")>;
// SUNPKHI, SUNPKLO
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUNPK(HI|LO)_ZZ_[HSD]")>;
// UUNPKHI, UUNPKLO
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UUNPK(HI|LO)_ZZ_[HSD]")>;
// SUQADD
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SUQADD_ZPmZ_[BHSD]")>;
// SXTB, SXTH, SXTW
def : InstRW<[OlympusWrite_2c_1V], (instregex "^SXTB_ZPmZ_[HSD]",
                                              "^SXTH_ZPmZ_[SD]",
                                              "^SXTW_ZPmZ_D")>;
// UXTB, UXTH, UXTW
def : InstRW<[OlympusWrite_2c_1V], (instregex "^UXTB_ZPmZ_[HSD]",
                                              "^UXTH_ZPmZ_[SD]",
                                              "^UXTW_ZPmZ_D")>;

// Shift, complex
// ASRD
def : InstRW<[OlympusWrite_4c_1V], (instregex "^ASRD_ZP[mZ]I_[BHSD]")>;
// RSHRNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^RSHRNB_ZZI_[BHS]")>;
// RSHRNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^RSHRNT_ZZI_[BHS]")>;
// SQRSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHLR_ZPmZ_[BHSD]")>;
// SQRSHL
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHL_ZP[mZ]Z_[BHSD]")>;
// SQSHL (immediate)
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHL_ZP[mZ]I_[BHSD]")>;
// SQSHL (vectors)
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHL_ZP[mZ]Z_[BHSD]")>;
// SQSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHLR_ZPmZ_[BHSD]")>;
// SQSHLU
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHLU_ZP[mZ]I_[BHSD]")>;
// SRSHL
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SRSHL_ZP[mZ]Z_[BHSD]")>;
// SRSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SRSHLR_ZPmZ_[BHSD]")>;
// UQRSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQRSHLR_ZPmZ_[BHSD]")>;
// UQRSHL
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQRSHL_ZP[mZ]Z_[BHSD]")>;
// UQSHL (immediate)
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQSHL_ZP[mZ]I_[BHSD]")>;
// UQSHL (vectors)
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQSHL_ZP[mZ]Z_[BHSD]")>;
// UQSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQSHLR_ZPmZ_[BHSD]")>;
// URSHL
def : InstRW<[OlympusWrite_4c_1V], (instregex "^URSHL_ZP[mZ]Z_[BHSD]")>;
// URSHLR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^URSHLR_ZPmZ_[BHSD]")>;
// SQRSHRNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHRNB_ZZI_[BHS]")>;
// SQRSHRNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHRNT_ZZI_[BHS]")>;
// SQRSHRUNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHRUNB_ZZI_[BHS]")>;
// SQRSHRUNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQRSHRUNT_ZZI_[BHS]")>;
// SQSHRNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHRNB_ZZI_[BHS]")>;
// SQSHRNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHRNT_ZZI_[BHS]")>;
// SQSHRUNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHRUNB_ZZI_[BHS]")>;
// SQSHRUNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQSHRUNT_ZZI_[BHS]")>;
// UQRSHRNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQRSHRNB_ZZI_[BHS]")>;
// UQRSHRNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQRSHRNT_ZZI_[BHS]")>;
// UQSHRNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQSHRNB_ZZI_[BHS]")>;
// UQSHRNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQSHRNT_ZZI_[BHS]")>;
// SRSHR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SRSHR_ZP[mZ]I_[BHSD]")>;
// URSHR
def : InstRW<[OlympusWrite_4c_1V], (instregex "^URSHR_ZP[mZ]I_[BHSD]")>;
// SQXTNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQXTNB_ZZ_[BHS]")>;
// SQXTNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQXTNT_ZZ_[BHS]")>;
// SQXTUNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQXTUNB_ZZ_[BHS]")>;
// SQXTUNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^SQXTUNT_ZZ_[BHS]")>;
// UQXTNB
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQXTNB_ZZ_[BHS]")>;
// UQXTNT
def : InstRW<[OlympusWrite_4c_1V], (instregex "^UQXTNT_ZZ_[BHS]")>;

// Shift and accumulate
def OlympusWr_ZSA : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_ZSA : SchedReadAdvance<2, [OlympusWr_ZSA]>;
// SRSRA
def : InstRW<[OlympusWr_ZSA, OlympusRd_ZSA], (instregex "^SRSRA_ZZI_[BHSD]")>;
// SSRA
def : InstRW<[OlympusWr_ZSA, OlympusRd_ZSA], (instregex "^SSRA_ZZI_[BHSD]")>;
// URSRA
def : InstRW<[OlympusWr_ZSA, OlympusRd_ZSA], (instregex "^URSRA_ZZI_[BHSD]")>;
// USRA
def : InstRW<[OlympusWr_ZSA, OlympusRd_ZSA], (instregex "^USRA_ZZI_[BHSD]")>;

// Multiply, B/H/S
// MUL (vectors, predicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL_ZP[mZ]Z_[BHS]")>;
// SMULH (predicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULH_ZP[mZ]Z_[BHS]")>;
// UMULH (predicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULH_ZP[mZ]Z_[BHS]")>;
// MUL (immediate)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL_ZI_[BHS]")>;
// MUL (vectors, unpredicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL_ZZZ_[BHS]")>;
// SMULH (unpredicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULH_ZZZ_[BHS]")>;
// SQDMULH (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULH_ZZZ_[BHS]")>;
// SQRDMULH (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQRDMULH_ZZZ_[BHS]")>;
// UMULH (unpredicated)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULH_ZZZ_[BHS]")>;
// SQDMULH (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULH_ZZZI_[HS]")>;
// SQRDMULH (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQRDMULH_ZZZI_[HS]")>;
// MUL (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^MUL_ZZZI_[HS]")>;

// Multiply, D
// MUL (vectors, predicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instregex "^MUL_ZP[mZ]Z_D")>;
// SMULH (predicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instregex "^SMULH_ZP[mZ]Z_D")>;
// UMULH (predicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instregex "^UMULH_ZP[mZ]Z_D")>;
// MUL (vectors, unpredicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs MUL_ZZZ_D)>;
// SMULH (unpredicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs SMULH_ZZZ_D)>;
// SQDMULH (vectors)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs SQDMULH_ZZZ_D)>;
// SQRDMULH (vectors)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs SQRDMULH_ZZZ_D)>;
// UMULH (unpredicated)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs UMULH_ZZZ_D)>;
// MUL (immediate)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs MUL_ZI_D)>;
// SQDMULH (indexed)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs SQDMULH_ZZZI_D)>;
// SQRDMULH (indexed)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs SQRDMULH_ZZZI_D)>;
// MUL (indexed)
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs MUL_ZZZI_D)>;

// Multiply long
// SMULLB (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULLB_ZZZ_[HSD]")>;
// SMULLB (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULLB_ZZZI_[SD]")>;
// SMULLT (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULLT_ZZZ_[HSD]")>;
// SMULLT (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SMULLT_ZZZI_[SD]")>;
// SQDMULLB (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULLB_ZZZ_[HSD]")>;
// SQDMULLB (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULLB_ZZZI_[SD]")>;
// SQDMULLT (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULLT_ZZZ_[HSD]")>;
// SQDMULLT (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^SQDMULLT_ZZZI_[SD]")>;
// UMULLB (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULLB_ZZZ_[HSD]")>;
// UMULLB (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULLB_ZZZI_[SD]")>;
// UMULLT (vectors)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULLT_ZZZ_[HSD]")>;
// UMULLT (indexed)
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^UMULLT_ZZZI_[SD]")>;

// Multiply accumulate, B/H/S
def OlympusWr_ZMA_BHS  : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusRd_ZMA_BHS  : SchedReadAdvance<2, [OlympusWr_ZMA_BHS]>;
def OlympusWr_ZMASQ_BHS : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusWr_ZMASQ_D   : SchedWriteRes<[OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 5; }
def OlympusWr_ZMASQL   : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusRd_ZMASQ    : SchedReadAdvance<2, [OlympusWr_ZMASQL, OlympusWr_ZMASQ_BHS, OlympusWr_ZMASQ_D]>;
// CMLA (vectors)
def : InstRW<[OlympusWr_ZMA_BHS, OlympusRd_ZMA_BHS], (instregex "^CMLA_ZZZ_[BHS]")>;
// CMLA (indexed)
def : InstRW<[OlympusWr_ZMA_BHS, OlympusRd_ZMA_BHS], (instregex "^CMLA_ZZZI_[HS]")>;
// MLA (vectors)
def : InstRW<[OlympusWr_ZMA_BHS, ReadDefault, OlympusRd_ZMA_BHS], (instregex "^MLA_ZP[mZ]ZZ_[BHS]")>;
// MLS (vectors)
def : InstRW<[OlympusWr_ZMA_BHS, ReadDefault, OlympusRd_ZMA_BHS], (instregex "^MLS_ZP[mZ]ZZ_[BHS]")>;
// MLA (indexed)
def : InstRW<[OlympusWr_ZMA_BHS, OlympusRd_ZMA_BHS], (instregex "^MLA_ZZZI_[HS]")>;
// MLS (indexed)
def : InstRW<[OlympusWr_ZMA_BHS, OlympusRd_ZMA_BHS], (instregex "^MLS_ZZZI_[HS]")>;
// SQRDCMLAH (vectors)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDCMLAH_ZZZ_[BHS]")>;
// SQRDMLAH (vectors)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDMLAH_ZZZ_[BHS]")>;
// SQRDMLSH (vectors)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDMLSH_ZZZ_[BHS]")>;
// SQRDCMLAH (indexed)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDCMLAH_ZZZI_[HS]")>;
// SQRDMLAH (indexed)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDMLAH_ZZZI_[HS]")>;
// SQRDMLSH (indexed)
def : InstRW<[OlympusWr_ZMASQ_BHS, OlympusRd_ZMASQ], (instregex "^SQRDMLSH_ZZZI_[HS]")>;

// Multiply accumulate, D
def OlympusWr_ZMA_D : SchedWriteRes<[OlympusUnitV0123, OlympusUnitV0123]> { let Latency = 5; }
def OlympusRd_ZMA_D : SchedReadAdvance<2, [OlympusWr_ZMA_D]>;
// CMLA (vectors)
def : InstRW<[OlympusWr_ZMA_D, OlympusRd_ZMA_D], (instrs CMLA_ZZZ_D)>;
// MLA (vectors)
def : InstRW<[OlympusWr_ZMA_D, ReadDefault, OlympusRd_ZMA_D], (instregex "^MLA_ZP[mZ]ZZ_D")>;
// MLS (vectors)
def : InstRW<[OlympusWr_ZMA_D, ReadDefault, OlympusRd_ZMA_D], (instregex "^MLS_ZP[mZ]ZZ_D")>;
// MLA (indexed)
def : InstRW<[OlympusWr_ZMA_D, OlympusRd_ZMA_D], (instrs MLA_ZZZI_D)>;
// MLS (indexed)
def : InstRW<[OlympusWr_ZMA_D, OlympusRd_ZMA_D], (instrs MLS_ZZZI_D)>;
// SQRDCMLAH (vectors)
def : InstRW<[OlympusWr_ZMASQ_D, OlympusRd_ZMASQ], (instrs SQRDCMLAH_ZZZ_D)>;
// SQRDMLAH (vectors)
def : InstRW<[OlympusWr_ZMASQ_D, OlympusRd_ZMASQ], (instrs SQRDMLAH_ZZZ_D)>;
// SQRDMLSH (vectors)
def : InstRW<[OlympusWr_ZMASQ_D, OlympusRd_ZMASQ], (instrs SQRDMLSH_ZZZ_D)>;
// SQRDMLAH (indexed)
def : InstRW<[OlympusWr_ZMASQ_D, OlympusRd_ZMASQ], (instrs SQRDMLAH_ZZZI_D)>;
// SQRDMLSH (indexed)
def : InstRW<[OlympusWr_ZMASQ_D, OlympusRd_ZMASQ], (instrs SQRDMLSH_ZZZI_D)>;

// Multiply accumulate long
def OlympusWr_ZMAL : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 4; }
def OlympusRd_ZMAL : SchedReadAdvance<2, [OlympusWr_ZMAL]>;
// SMLALB (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLALB_ZZZ_[HSD]")>;
// SMLALB (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLALB_ZZZI_[SD]")>;
// SMLALT (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLALT_ZZZ_[HSD]")>;
// SMLALT (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLALT_ZZZI_[SD]")>;
// SMLSLB (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLSLB_ZZZ_[HSD]")>;
// SMLSLB (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLSLB_ZZZI_[SD]")>;
// SMLSLT (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLSLT_ZZZ_[HSD]")>;
// SMLSLT (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^SMLSLT_ZZZI_[SD]")>;
// SQDMLALB (vectors)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLALB_ZZZ_[HSD]")>;
// SQDMLALB (indexed)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLALB_ZZZI_[SD]")>;
// SQDMLALBT
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLALBT_ZZZ_[HSD]")>;
// SQDMLALT (vectors)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLALT_ZZZ_[HSD]")>;
// SQDMLALT (indexed)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLALT_ZZZI_[SD]")>;
// SQDMLSLB (vectors)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLSLB_ZZZ_[HSD]")>;
// SQDMLSLB (indexed)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLSLB_ZZZI_[SD]")>;
// SQDMLSLBT
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLSLBT_ZZZ_[HSD]")>;
// SQDMLSLT (vectors)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLSLT_ZZZ_[HSD]")>;
// SQDMLSLT (indexed)
def : InstRW<[OlympusWr_ZMASQL, OlympusRd_ZMASQ], (instregex "^SQDMLSLT_ZZZI_[SD]")>;
// UMLALB (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLALB_ZZZ_[HSD]")>;
// UMLALB (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLALB_ZZZI_[SD]")>;
// UMLALT (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLALT_ZZZ_[HSD]")>;
// UMLALT (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLALT_ZZZI_[SD]")>;
// UMLSLB (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLSLB_ZZZ_[HSD]")>;
// UMLSLB (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLSLB_ZZZI_[SD]")>;
// UMLSLT (vectors)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLSLT_ZZZ_[HSD]")>;
// UMLSLT (indexed)
def : InstRW<[OlympusWr_ZMAL, OlympusRd_ZMAL], (instregex "^UMLSLT_ZZZI_[SD]")>;

// Multiply add/sub
def OlympusWr_ZMAD  : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 5; }
def OlympusRd_ZMAD  : SchedReadAdvance<2, [OlympusWr_ZMAD]>;
// MAD
def : InstRW<[OlympusWr_ZMAD, ReadDefault, OlympusRd_ZMAD], (instregex "^MAD_ZPmZZ_[BHSD]")>;
// MSB
def : InstRW<[OlympusWr_ZMAD, ReadDefault, OlympusRd_ZMAD], (instregex "^MSB_ZPmZZ_[BHSD]")>;

// Other arithmetic accumulate
def OlympusWr_ZA  : SchedWriteRes<[OlympusUnitV]> { let Latency = 4; }
def OlympusRd_ZA  : SchedReadAdvance<2, [OlympusWr_ZA]>;
// SABA
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^SABA_ZZZ_[BHSD]")>;
// SABALB
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^SABALB_ZZZ_[HSD]")>;
// SABALT
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^SABALT_ZZZ_[HSD]")>;
// UABA
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^UABA_ZZZ_[BHSD]")>;
// UABALB
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^UABALB_ZZZ_[HSD]")>;
// UABALT
def : InstRW<[OlympusWr_ZA, OlympusRd_ZA], (instregex "^UABALT_ZZZ_[HSD]")>;
// SADALP
def : InstRW<[OlympusWr_ZA, ReadDefault, OlympusRd_ZA], (instregex "^SADALP_ZPmZ_[HSD]")>;
// UADALP
def : InstRW<[OlympusWr_ZA, ReadDefault, OlympusRd_ZA], (instregex "^UADALP_ZPmZ_[HSD]")>;

// Dot product, matrix multiply, 8 bit
def OlympusWr_ZDOT_B : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
def OlympusRd_ZDOT_B : SchedReadAdvance<1, [OlympusWr_ZDOT_B]>;
def OlympusWr_ZMMA : SchedWriteRes<[OlympusUnitV]> { let Latency = 3; }
def OlympusRd_ZMMA : SchedReadAdvance<1, [OlympusWr_ZMMA]>;
// CDOT (vectors)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs CDOT_ZZZ_S)>;
// CDOT (indexed)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs CDOT_ZZZI_S)>;
// SDOT (4-way, vectors)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs SDOT_ZZZ_BtoS)>;
// UDOT (4-way, vectors)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs UDOT_ZZZ_BtoS)>;
// SDOT (4-way, indexed)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs SDOT_ZZZI_BtoS)>;
// UDOT (4-way, indexed)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs UDOT_ZZZI_BtoS)>;
// USDOT (vectors)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs USDOT_ZZZ)>;
// SUDOT
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs SUDOT_ZZZI)>;
// USDOT (indexed)
def : InstRW<[OlympusWr_ZDOT_B, OlympusRd_ZDOT_B], (instrs USDOT_ZZZI)>;
// SMMLA
def : InstRW<[OlympusWr_ZMMA, OlympusRd_ZMMA], (instrs SMMLA_ZZZ)>;
// UMMLA
def : InstRW<[OlympusWr_ZMMA, OlympusRd_ZMMA], (instrs UMMLA_ZZZ)>;
// USMMLA
def : InstRW<[OlympusWr_ZMMA, OlympusRd_ZMMA], (instrs USMMLA_ZZZ)>;

// Dot product, 16 bit
def OlympusWr_ZDOT_H : SchedWriteRes<[OlympusUnitV0123]> { let Latency = 3; }
def OlympusRd_ZDOT_H : SchedReadAdvance<1, [OlympusWr_ZDOT_H]>;
// CDOT (vectors)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs CDOT_ZZZ_D)>;
// CDOT (indexed)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs CDOT_ZZZI_D)>;
// SDOT (4-way, vectors)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs SDOT_ZZZ_HtoD)>;
// UDOT (4-way, vectors)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs UDOT_ZZZ_HtoD)>;
// SDOT (4-way, indexed)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs SDOT_ZZZI_HtoD)>;
// UDOT (4-way, indexed)
def : InstRW<[OlympusWr_ZDOT_H, OlympusRd_ZDOT_H], (instrs UDOT_ZZZI_HtoD)>;

// Bit manipulation
// BDEP
def : InstRW<[OlympusWrite_6c_2V12], (instregex "^BDEP_ZZZ_[BHSD]")>;
// BEXT
def : InstRW<[OlympusWrite_6c_2V12], (instregex "^BEXT_ZZZ_[BHSD]")>;
// BGRP
def : InstRW<[OlympusWrite_6c_2V12], (instregex "^BGRP_ZZZ_[BHSD]")>;

// Compare and set flags
// CMP<cc> (immediate)
def : InstRW<[OlympusWrite_2c_1V03_or_1M_1V03], (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZI_[BHSD]")>;
// CMP<cc> (wide elements)
def : InstRW<[OlympusWrite_2c_1V03_or_1M_1V03], (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;
// CMP<cc> (vectors)
def : InstRW<[OlympusWrite_2c_1V03_or_1M_1V03], (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZZ_[BHSD]")>;

// Extract, FP scalar and vector
// CLASTA (SIMD&FP scalar)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^CLASTA_VPZ_[BHSD]")>;
// CLASTA (vectors)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^CLASTA_ZPZ_[BHSD]")>;
// CLASTB (SIMD&FP scalar)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^CLASTB_VPZ_[BHSD]")>;
// CLASTB (vectors)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^CLASTB_ZPZ_[BHSD]")>;
// LASTA (SIMD&FP scalar)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^LASTA_VPZ_[BHSD]")>;
// LASTB (SIMD&FP scalar)
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^LASTB_VPZ_[BHSD]")>;
// COMPACT
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^COMPACT_ZPZ_[SD]")>;
// SPLICE
def : InstRW<[OlympusWrite_3c_1V12], (instregex "^SPLICE_ZPZZ?_[BHSD]")>;

// Extract, gen scalar conditional
def : InstRW<[OlympusWrite_8c_1M_1V03_1V12], (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Extract, gen scalar unconditional
def : InstRW<[OlympusWrite_6c_1V03_1V12], (instregex "^LAST[AB]_RPZ_[BHSD]")>;

// Convert int to FP, 64b or to F64
def : InstRW<[OlympusWrite_3c_1V0123], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                                  "^[SU]CVTF_ZPmZ_StoD")>;

// Convert int to FP, 32b to F16/F32
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert int to FP, 16b
def : InstRW<[OlympusWrite_6c_1V0123_4], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy/Insert, from gen
// CPY (scalar)
def : InstRW<[OlympusWrite_5c_1M_1V], (instregex "^CPY_ZPmR_[BHSD]")>;
// INSR (scalar)
def : InstRW<[OlympusWrite_5c_1M_1V], (instregex "^INSR_ZR_[BHSD]")>;

// Duplicate, from gen
def : InstRW<[OlympusWrite_3c_1M], (instregex "^DUP_ZR_[BHSD]")>;

// Divide
// Divides, 32 bit
def : InstRW<[OlympusWrite_12c_1V45], (instregex "^[SU]DIV_ZP[mZ]Z_S", "^[SU]DIVR_ZPmZ_S")>;
// Divides, 64 bit
def : InstRW<[OlympusWrite_20c_1V45], (instregex "^[SU]DIV_ZP[mZ]Z_D", "^[SU]DIVR_ZPmZ_D")>;

// Index, immediates, B/H/S
def : InstRW<[OlympusWrite_4c_1V0123], (instregex "^INDEX_II_[BHS]")>;

// Index, immediates, D
def : InstRW<[OlympusWrite_5c_1V0123_2], (instrs INDEX_II_D)>;

// Index, scalar, B/H/S
def : InstRW<[OlympusWrite_7c_1M_1V0123], (instregex "^INDEX_(IR|RI|RR)_[BHS]")>;

// Index, scalar, D
def : InstRW<[OlympusWrite_8c_1M_1V0123_2], (instregex "^INDEX_(IR|RI|RR)_D")>;

// Matching operations
def : InstRW<[OlympusWrite_2c_1V03_or_1M_1V03], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Reciprocal estimate
def : InstRW<[OlympusWrite_4c_1V0123_2], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, logical
def : InstRW<[OlympusWrite_6c_1V_1V0123], (instregex "^(AND|EOR|OR)V_VPZ_[BHSD]")>;

// Reduction, add, B form
def : InstRW<[OlympusWrite_11c_2V_2V0123], (instregex "^[SU]ADDV_VPZ_B")>;

// Reduction, add, H form
def : InstRW<[OlympusWrite_9c_1V_2V0123], (instregex "^[SU]ADDV_VPZ_H")>;

// Reduction, add, S form
def : InstRW<[OlympusWrite_8c_2V_1V0123], (instregex "^[SU]ADDV_VPZ_S")>;

// Reduction, min/max, B form
def : InstRW<[OlympusWrite_9c_1V_2V0123], (instregex "^[SU](MAX|MIN)V_VPZ_B")>;

// Reduction, min/max, H form
def : InstRW<[OlympusWrite_8c_2V_1V0123], (instregex "^[SU](MAX|MIN)V_VPZ_H")>;

// Reduction, min/max, S form
def : InstRW<[OlympusWrite_6c_1V_1V0123], (instregex "^[SU](MAX|MIN)V_VPZ_S")>;

// Reduction, D form
def : InstRW<[OlympusWrite_5c_2V], (instregex "^[SU](MAX|MIN)V_VPZ_D",
                                              "^UADDV_VPZ_D")>;

// SVE cryptography instructions
// -----------------------------------------------------------------------------

// AES, SHA3 operations
// AESD
// AESE
// AESIMC
// AESMC
def : InstRW<[OlympusWrite_2c_1V], (instregex "^AES[DE]_ZZZ_B$",
                                              "^AESI?MC_ZZ_B$")>;
// BCAX
def : InstRW<[OlympusWrite_2c_1V], (instrs BCAX_ZZZZ)>;
// EOR3
def : InstRW<[OlympusWrite_2c_1V], (instrs EOR3_ZZZZ)>;
// RAX1
def : InstRW<[OlympusWrite_2c_1V], (instrs RAX1_ZZZ_D)>;
// XAR
def : InstRW<[OlympusWrite_2c_1V], (instregex "^XAR_ZZZI_[BHSD]")>;

// SM4 operations
def : InstRW<[OlympusWrite_4c_1V0], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

// SVE load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[OlympusWrite_6c_1L], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[OlympusWrite_5c_1M_1L], (instrs LDR_PXI)>;

// Contiguous load single structure
// LD1* (scalar plus immediate, single register)
// LD1S* (scalar plus immediate)
// LD1* (scalar plus scalar, single register)
// LD1S* (scalar plus scalar)
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LD1[BHWD](_IMM)?$",
                                              "^LD1S?B_[HSD](_IMM)?$",
                                              "^LD1S?H_[SD](_IMM)?$",
                                              "^LD1S?W_D(_IMM)?$")>;
// LD1R*
// LD1RS*
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LD1R[BHWD]_IMM$",
                                              "^LD1RS?B_[HSD]_IMM$",
                                              "^LD1RS?H_[SD]_IMM$",
                                              "^LD1RW_D_IMM$",
                                              "^LD1RSW_IMM$")>;
// LD1RQ* (scalar plus immediate)
// LD1RQ* (scalar plus scalar)
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LD1RQ_[BHWD](_IMM)?$")>;
// LDNF1*
// LDNF1S*
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LDNF1[BHWD]_IMM$",
                                              "^LDNF1S?B_[HSD]_IMM$",
                                              "^LDNF1S?H_[SD]_IMM$",
                                              "^LDNF1S?W_D_IMM$")>;
// LDNT1* (scalar plus immediate, single register)
// LDNT1* (scalar plus scalar, single register)
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LDNT1[BHWD]_ZR[IR]$")>;
// LDFF1* (scalar plus scalar)
// LDFF1S* (scalar plus scalar)
def : InstRW<[OlympusWrite_6c_1L], (instregex "^LDFF1[BHWD]$",
                                              "^LDFF1S?B_[HSD]$",
                                              "^LDFF1S?H_[SD]$",
                                              "^LDFF1S?W_D$")>;

// Contiguous load two structures, scalar + imm
def : InstRW<[OlympusWrite_8c_2L_2V], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous load two structures, scalar + scalar
def : InstRW<[OlympusWrite_9c_1I_2L_2V], (instregex "^LD2[BHWD]$")>;

// Contiguous load three structures, scalar + imm
def : InstRW<[OlympusWrite_8c_3L_3V], (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous load three structures, scalar + scalar
def : InstRW<[OlympusWrite_9c_1I_3L_3V], (instregex "^LD3[BHWD]$")>;

// Contiguous load four structures, scalar + imm
def : InstRW<[OlympusWrite_9c_4L_8V], (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous load four structures, scalar + scalar
def : InstRW<[OlympusWrite_10c_1I_4L_8V], (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element
def : InstRW<[OlympusWrite_9c_4L_1V03], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                                   "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element
def : InstRW<[OlympusWrite_9c_2L_1V03], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                                   "^GLD(FF)?1D_IMM$")>;

// Gather load, scalar + vector, H form 32-bit scaled offset
def : InstRW<[OlympusWrite_9c_4L_2V03], (instregex "^GLD(FF)?1S?H_S_[SU]XTW_SCALED$")>;

// Gather load, scalar + vector, other 32-bit element
def : InstRW<[OlympusWrite_9c_4L_1V03], (instregex "^GLD(FF)?1W_[SU]XTW(_SCALED)?$",
                                                   "^GLD(FF)?1S?[BH]_S_[SU]XTW$")>;

// Gather load, scalar + vector, 64-bit element
def : InstRW<[OlympusWrite_9c_2L_1V03], (instregex "^GLD(FF)?1S?B_D(_[SU]XTW)?$",
                                                   "^GLD(FF)?1S?[HW]_D(_[SU]XTW)?(_SCALED)?$",
                                                   "^GLD(FF)?1D(_[SU]XTW)?(_SCALED)?$")>;

// Gather non-temporal load, 32-bit element
def : InstRW<[OlympusWrite_9c_4L_1V03], (instregex "^LDNT1[BHW]_ZZR_S$",
                                                   "^LDNT1S[BH]_ZZR_S$")>;

// Gather non-temporal load, 64-bit element
def : InstRW<[OlympusWrite_9c_2L_1V03], (instregex "^LDNT1S?[BHW]_ZZR_D$",
                                                   "^LDNT1D_ZZR_D$")>;

// SVE store instructions
// -----------------------------------------------------------------------------

// Store from predicate
def : InstRW<[OlympusWrite_1c_1SA_1D], (instrs STR_PXI)>;

// Store from vector
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instrs STR_ZXI)>;

// Contiguous store single structure
// ST1* (scalar plus immediate, single register)
// ST1* (scalar plus scalar, single register)
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instregex "^ST1[BHWD](_IMM)?$",
                                                      "^ST1B_[HSD](_IMM)?$",
                                                      "^ST1H_[SD](_IMM)?$",
                                                      "^ST1W_D(_IMM)?$")>;
// STNT1* (scalar plus immediate, single register)
// STNT1* (scalar plus scalar, single register)
def : InstRW<[OlympusWrite_3c_1SA_1V0123], (instregex "^STNT1[BHWD]_ZR[IR]$")>;

// Contiguous store two structures, scalar + imm
def : InstRW<[OlympusWrite_5c_2SA_2V_2V0123], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures, scalar + scalar
def : InstRW<[OlympusWrite_5c_1I_2SA_2V_2V0123], (instregex "^ST2[BHWD]$")>;

// Contiguous store three structures, scalar + imm
def : InstRW<[OlympusWrite_6c_3SA_3V_3V0123], (instregex "^ST3[BHWD]_IMM$")>;

// Contiguous store three structures, scalar + scalar
def : InstRW<[OlympusWrite_6c_1I_3SA_3V_3V0123], (instregex "^ST3[BHWD]$")>;

// Contiguous store four structures, scalar + imm, B/H/W
def : InstRW<[OlympusWrite_8c_4SA_8V_4V0123], (instregex "^ST4[BHW]_IMM$")>;

// Contiguous store four structures, scalar + imm, D
def : InstRW<[OlympusWrite_6c_4SA_4V_4V0123], (instrs ST4D_IMM)>;

// Contiguous store four structures, scalar + scalar, B/H/W
def : InstRW<[OlympusWrite_8c_1I_4SA_8V_4V0123], (instregex "^ST4[BHW]$")>;

// Contiguous store four structures, scalar + scalar, D
def : InstRW<[OlympusWrite_6c_1I_4SA_4V_4V0123], (instrs ST4D)>;

// Scatter store, vector + imm, 32-bit element
def : InstRW<[OlympusWrite_5c_4SA_1V03_4V0123], (instregex "^SST1[BH]_S_IMM$",
                                                           "^SST1W_IMM$")>;

// Scatter store, vector + imm, 64-bit element
def : InstRW<[OlympusWrite_4c_2SA_1V03_2V0123], (instregex "^SST1[BHW]_D_IMM$",
                                                           "^SST1D_IMM$")>;

// Scatter store, scalar + vector, H form 32-bit scaled offset
def : InstRW<[OlympusWrite_5c_4SA_2V03_4V0123], (instregex "^SST1H_S_[SU]XTW_SCALED$")>;

// Scatter store, scalar + vector, 32-bit element
def : InstRW<[OlympusWrite_5c_4SA_1V03_4V0123], (instregex "^SST1[BH]_S_[SU]XTW$",
                                                           "^SST1W_[SU]XTW(_SCALED)?$")>;

// Scatter store, scalar + vector, 64-bit element
def : InstRW<[OlympusWrite_4c_2SA_1V03_2V0123], (instregex "^SST1[BHW]_D(_[SU]XTW)?$",
                                                           "^SST1[HW]_D(_[SU]XTW)?_SCALED$",
                                                           "^SST1D(_[SU]XTW)?(_SCALED)?$")>;

// Scatter non-temporal store, 32-bit element
def : InstRW<[OlympusWrite_5c_4SA_1V03_4V0123], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non-temporal store, 64-bit element
def : InstRW<[OlympusWrite_4c_2SA_1V03_2V0123], (instregex "^STNT1[BHWD]_ZZR_D")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Prefetch
// NOTE: Not specified in the SWOG.
def : InstRW<[OlympusWrite_6c_1L], (instregex "^PRF[BHWD]")>;

}
