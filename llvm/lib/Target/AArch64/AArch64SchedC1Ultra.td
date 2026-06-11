//=- AArch64SchedC1Ultra.td - C1 Ultra Scheduling Defs --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the C1 Ultra processors.
// Information is taken from the C1 Ultra Software Optimization Guide
//
// Base scheduling model documentation
// https://developer.arm.com/documentation/111079/3-0
//
// SME2 scheduling model documentation
// https://developer.arm.com/documentation/111048/0103
//
//===----------------------------------------------------------------------===//

def C1UltraModel : SchedMachineModel {
  let IssueWidth            =  10;
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer. NOTE: Copied from Neoverse-V3
  let LoadLatency           =   4; // Optimistic load latency. NOTE: Copied from Neoverse-V3
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch. NOTE: Copied from N3.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasCPA, HasMTE, HasCSSC,
                                                     HasMOPS, HasMOPS_GO]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on C1 Ultra.
// Instructions are first fetched and then decoded into internal macro-ops
// (MOPs). From there, the MOPs proceed through register renaming and dispatch
// stages. A MOP can be split into two micro-ops further down the pipeline
// after the decode stage.

let SchedModel = C1UltraModel in {

// Define the issue ports.
def C1UUnitB      : ProcResource<3>;  // Branch 0/1/2
def C1UUnitS0     : ProcResource<1>;  // Integer single-cycle 0
def C1UUnitS1     : ProcResource<1>;  // Integer single-cycle 1
def C1UUnitS2     : ProcResource<1>;  // Integer single-cycle 2
def C1UUnitS3     : ProcResource<1>;  // Integer single-cycle 3
def C1UUnitS4     : ProcResource<1>;  // Integer single-cycle 4
def C1UUnitS5     : ProcResource<1>;  // Integer single-cycle 5
def C1UUnitM0     : ProcResource<1>;  // Integer single/multicycle 0
def C1UUnitM1     : ProcResource<1>;  // Integer single/multicycle 1
def C1UUnitV0     : ProcResource<1>;  // FP/ASIMD 0
def C1UUnitV1     : ProcResource<1>;  // FP/ASIMD 1
def C1UUnitV2     : ProcResource<1>;  // FP/ASIMD 2
def C1UUnitV3     : ProcResource<1>;  // FP/ASIMD 3
def C1UUnitV4     : ProcResource<1>;  // FP/ASIMD 4
def C1UUnitV5     : ProcResource<1>;  // FP/ASIMD 5
def C1UUnitLS0    : ProcResource<1>;  // Load/Store 0
def C1UUnitLS1    : ProcResource<1>;  // Load/Store 1
def C1UUnitL23    : ProcResource<2>;  // Load 2/3
def C1UUnitD      : ProcResource<2>;  // Store data 0/1
def C1UUnitFlg    : ProcResource<4>;  // Flags

// Define the CME coprocessor pipelines from the C1-SME2 guide.
def C1UUnitCMELD       : ProcResource<1>;  // LD
def C1UUnitCMEST       : ProcResource<1>;  // ST
def C1UUnitCMEPF       : ProcResource<1>;  // PF
def C1UUnitCMEVX0ALU   : ProcResource<1>;  // VX0ALU
def C1UUnitCMEVX1ALU   : ProcResource<1>;  // VX1ALU
def C1UUnitCMEVX0FPDOT : ProcResource<1>;  // VX0FPDOT
def C1UUnitCMEVX1FPDOT : ProcResource<1>;  // VX1FPDOT
def C1UUnitCMEVX0MUL   : ProcResource<1>;  // VX0MUL
def C1UUnitCMEVX1MUL   : ProcResource<1>;  // VX1MUL
def C1UUnitCMEPERMF    : ProcResource<1>;  // PERMF
def C1UUnitCMEPERMS    : ProcResource<1>;  // PERMS
def C1UUnitCMEMC       : ProcResource<1>;  // MC
def C1UUnitCMEMML0     : ProcResource<2>;  // MML0
def C1UUnitCMEMML1     : ProcResource<1>;  // MML1
def C1UUnitCMERNMRET   : ProcResource<3>;  // RNMRET

def C1UUnitS      : ProcResGroup<[C1UUnitS0, C1UUnitS1, C1UUnitS2, C1UUnitS3,
                                  C1UUnitS4, C1UUnitS5]>;
def C1UUnitI      : ProcResGroup<[C1UUnitS0, C1UUnitS1, C1UUnitS2, C1UUnitS3,
                                  C1UUnitS4, C1UUnitS5, C1UUnitM0, C1UUnitM1]>;
def C1UUnitI4     : ProcResGroup<[C1UUnitS0, C1UUnitS2, C1UUnitS4, C1UUnitM0]>;
def C1UUnitM      : ProcResGroup<[C1UUnitM0, C1UUnitM1]>;
def C1UUnitL      : ProcResGroup<[C1UUnitLS0, C1UUnitLS1, C1UUnitL23]>;
def C1UUnitSA     : ProcResGroup<[C1UUnitLS0, C1UUnitLS1]>;
def C1UUnitV      : ProcResGroup<[C1UUnitV0, C1UUnitV1, C1UUnitV2,
                                  C1UUnitV3, C1UUnitV4, C1UUnitV5]>;
def C1UUnitV01    : ProcResGroup<[C1UUnitV0, C1UUnitV1]>;
def C1UUnitV0134  : ProcResGroup<[C1UUnitV0, C1UUnitV1, C1UUnitV3, C1UUnitV4]>;
def C1UUnitCMEVXALU : ProcResGroup<[C1UUnitCMEVX0ALU, C1UUnitCMEVX1ALU]>;
def C1UUnitCMEVXFPDOT : ProcResGroup<[C1UUnitCMEVX0FPDOT,
                                      C1UUnitCMEVX1FPDOT]>;
def C1UUnitCMEVXMUL : ProcResGroup<[C1UUnitCMEVX0MUL, C1UUnitCMEVX1MUL]>;
def C1UUnitCMEPERM  : ProcResGroup<[C1UUnitCMEPERMS, C1UUnitCMEPERMF]>;
def C1UUnitCME3PERMF : ProcResGroup<[C1UUnitPERMF, C1UUnitPERMF, C1UUnitPERMF]>;

// Define commonly used read types.
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
// Define customized scheduler read/write types specific to C1 Ultra.
//===----------------------------------------------------------------------===//

// The approach for modelling SME instructions should not be taken as accurate.
// This is due to limitations of tablegen and the actual behaviour we want to
// to model. The CME is a co-processor. Accurate modelling would allow for us
// to model core-to-processor communication latencies. This is out of the scope
// of what we can achieve at the present. As a consequence we model instructions
// that run on the CME as by only modelling their execution latency once issued
// rather than modelling the more complex relationship between the C1-Ultra and
// the CME co-processor and how that will affect execution latencies. Developers
// must therefore take latencies for SME instructions as inaccurate.
//
// We have several classes of SME instructions that we need to model.
//
//    1. SVE instructions added by SME and available when not in streaming SVE mode
//    2. SVE instructions added by SME but not sent to CME when in streaming SVE mode
//    3. FP/SVE/ASIMD instructions sent to CME when in streaming SVE mode
//    4. SME instructions sent to CME and only available in streaming SVE mode
//
// To model instructions of type (1) and (2) is fairly easy in that we know
// that these instructions aren't "dependent" on streaming SVE mode to model their
// scheduling information. To model these we use a predicate on these instructions
// which checks if the target supports SME (SMESchedPred).
//
// To model instructions of type (3) we recognise that llvm-mca doesn't include
// information of whether we are in streaming SVE mode. However, for the sake of
// completeness we model this class of instructions by adding
// an llvm-mca attribute (mca-streaming-sched) which is interpreted as the core
// being in streaming SVE mode. This allows us to at least have some data on how
// these instructions may be scheduled when in streaming SVE mode using llvm-mca
// if the core supports SME and this attribute is enabled. Otherwise, we default
// to the plain non-streaming SVE mode scheduling information.
//
// Similarly, to model instructions of type (4) we just extend the scheduling model
// to add definitions for this class of instructions which are "gated" via the
// SMESchedPred predicate and mca-streaming-sched attribute. The purpose of
// having these definitions is that information on how these instructions are
// scheduled on the CME coprocessor is useful for analysis.

// SME2 related read write types
// Define generic 0 and 1 micro-op types
def C1UWrite_0c : SchedWriteRes<[]> { let Latency = 0; }
def C1UWrite_1c : SchedWriteRes<[]> { let Latency = 1; }

class C1UCMEWrite<int LatencyCycles, list<ProcResourceKind> Resources>
    : SchedWriteRes<Resources> {
  let Latency = LatencyCycles;
}

class C1UCMEWriteRC<int LatencyCycles, list<ProcResourceKind> Resources,
                    list<int> ReleaseCycles>
    : SchedWriteRes<Resources> {
  let Latency = LatencyCycles;
  let ReleaseAtCycles = ReleaseCycles;
}

class C1UStreamingSchedWrite<SchedWrite StreamingWrite, SchedWrite BaseWrite>
    : SchedWriteVariant<[
        SchedVar<SMEMCStreamingSchedPred, [StreamingWrite]>,
        SchedVar<NoSchedPred,            [BaseWrite]>
      ]>;


// Instructions added by SME and only available in streaming SVE mode
class C1UStreamingOnlySchedWrite<SchedWrite StreamingWrite> : C1UStreamingSchedWrite<StreamingWrite, C1UWrite_1c>;

// Instructions added by SME, available in non-streaming SVE mode or sent to CME
// when in streaming SVE mode.
class C1USMEStreamingSchedWrite<SchedWrite StreamingWrite,
                                SchedWrite SMEWrite>
    : SchedWriteVariant<[
        SchedVar<SMEMCStreamingSchedPred, [StreamingWrite]>,
        SchedVar<SMESchedPred,            [SMEWrite]>,
        SchedVar<NoSchedPred,             [C1UWrite_0c]>
      ]>;

// Detailed CME scheduling derived from the SME2 software optimization guide.
def C1UWrite_1c_CMEPERMS : C1UCMEWrite<1, [C1UUnitCMEPERM]>;
def C1UWrite_1c_CMEPERMS_2rc : C1UCMEWriteRC<1, [C1UUnitCMEPERM], [2]>;
def C1UWrite_1c_CMERNMRET : C1UCMEWrite<1, [C1UUnitCMERNMRET]>;
def C1UWrite_1c_CMEMML0_4rc : C1UCMEWriteRC<1, [C1UUnitCMEMML0], [4]>;
def C1UWrite_2c_1L_CMELD : C1UCMEWrite<2, [C1UUnitL, C1UUnitCMELD]>;
def C1UWrite_2c_1I_1L_CMELD : C1UCMEWrite<2, [C1UUnitI, C1UUnitL,
                                              C1UUnitCMELD]>;
def C1UWrite_2c_1V01_1L_CMELD : C1UCMEWrite<2, [C1UUnitV01, C1UUnitL,
                                                C1UUnitCMELD]>;
def C1UWrite_2c_1I_1V01_1L_CMELD
    : C1UCMEWrite<2, [C1UUnitI, C1UUnitV01, C1UUnitL, C1UUnitCMELD]>;
def C1UWrite_2c_1SA_CMEST : C1UCMEWrite<2, [C1UUnitSA, C1UUnitCMEST]>;
def C1UWrite_2c_1SA_1V01_CMEST : C1UCMEWrite<2, [C1UUnitSA, C1UUnitV01,
                                                 C1UUnitCMEST]>;
def C1UWrite_2c_1SA_1V01_1I_CMEST
    : C1UCMEWrite<2, [C1UUnitSA, C1UUnitV01, C1UUnitI, C1UUnitCMEST]>;
def C1UWrite_2c_2SA_2V01_CMEST : C1UCMEWrite<2, [C1UUnitSA, C1UUnitSA,
                                                 C1UUnitV01, C1UUnitV01,
                                                 C1UUnitCMEST]>;
def C1UWrite_2c_3SA_3V01_CMEST : C1UCMEWrite<2, [C1UUnitSA, C1UUnitSA,
                                                 C1UUnitSA, C1UUnitV01,
                                                 C1UUnitV01, C1UUnitV01,
                                                 C1UUnitCMEST]>;
def C1UWrite_2c_4SA_4V01_CMEST : C1UCMEWrite<2, [C1UUnitSA, C1UUnitSA,
                                                 C1UUnitSA, C1UUnitSA,
                                                 C1UUnitV01, C1UUnitV01,
                                                 C1UUnitV01, C1UUnitV01,
                                                 C1UUnitCMEST]>;
def C1UWrite_4c_CMEPERMS : C1UCMEWrite<4, [C1UUnitCMEPERM]>;
def C1UWrite_4c_CMEPERMS_2rc : C1UCMEWriteRC<4, [C1UUnitCMEPERM], [2]>;
def C1UWrite_5c_CMEPERMS_2rc : C1UCMEWriteRC<5, [C1UUnitCMEPERM], [2]>;
def C1UWrite_4c_CMEPERMF : C1UCMEWrite<4, [C1UUnitCME3PERMF]>;
def C1UWrite_4c_CMEPERMF_1rc : C1UCMEWriteRC<4, [C1UUnitCME3PERMF], [3]>;
def C1UWrite_4c_CMEMML0_4rc : C1UCMEWriteRC<4, [C1UUnitCMEMML0], [4]>;
def C1UWrite_4c_CMEMML0_6rc : C1UCMEWriteRC<4, [C1UUnitCMEMML0], [6]>;
def C1UWrite_4c_CMEMML0_8rc : C1UCMEWriteRC<4, [C1UUnitCMEMML0], [8]>;
def C1UWrite_4c_CMEVXALU : C1UCMEWrite<4, [C1UUnitCMEVXALU]>;
def C1UWrite_4c_CMEVXALU_2rc : C1UCMEWriteRC<4, [C1UUnitCMEVXALU], [2]>;
def C1UWrite_4c_CMEVXALU_4rc : C1UCMEWriteRC<4, [C1UUnitCMEVXALU], [4]>;
def C1UWrite_4c_CMEVX0ALU : C1UCMEWrite<4, [C1UUnitCMEVX0ALU]>;
def C1UWrite_4c_CMEVX0ALU_PERMF_1rc
    : C1UCMEWriteRC<4, [C1UUnitCMEVX0ALU, C1UUnitCME3PERMF], [1, 3]>;
def C1UWrite_4c_CMEVX0ALU_PERMF_2rc
    : C1UCMEWriteRC<4, [C1UUnitCMEVX0ALU, C1UUnitCME3PERMF], [2, 6]>;
def C1UWrite_4c_CMEVXFPDOT : C1UCMEWrite<4, [C1UUnitCMEVXFPDOT]>;
def C1UWrite_4c_CMEVXMUL : C1UCMEWrite<4, [C1UUnitCMEVXMUL]>;
def C1UWrite_4c_CMEMML0 : C1UCMEWrite<4, [C1UUnitCMEMML0]>;
def C1UWrite_8c_CMEVXMUL_1rc : C1UCMEWriteRC<8, [C1UUnitCMEVXMUL], [2]>;
def C1UWrite_8c_CMEVXALU_1rc : C1UCMEWriteRC<8, [C1UUnitCMEVXALU], [2]>;
def C1UWrite_12c_1M0_CMEPERMF_VX0ALU_2rc
    : C1UCMEWriteRC<12, [C1UUnitM0, C1UUnitCME3PERMF, C1UUnitCMEVX0ALU],
                    [1, 6, 2]>;
def C1UWrite_4c_CMEMML0_1rc : C1UCMEWriteRC<4, [C1UUnitCMEMML0], [4]>;
def C1UWrite_4c_CMEMML0_2rc : C1UCMEWriteRC<4, [C1UUnitCMEMML0], [2]>;
def C1UWrite_5c_CMEVXALU_1rc : C1UCMEWriteRC<5, [C1UUnitCMEVXALU], [2]>;
def C1UWrite_5c_CMEVXALU_4rc : C1UCMEWriteRC<5, [C1UUnitCMEVXALU], [4]>;
def C1UWrite_5c_CMEPERMF_6rc : C1UCMEWriteRC<5, [C1UUnitCME3PERMF], [6]>;
def C1UWrite_5c_CMEMML0_2rc : C1UCMEWriteRC<5, [C1UUnitCMEMML0], [2]>;
def C1UWrite_4c_1M0_CMEVXALU : C1UCMEWrite<4, [C1UUnitM0, C1UUnitCMEVXALU]>;
def C1UWrite_4c_1M_CMEPERMS_1rc : C1UCMEWriteRC<4, [C1UUnitM, C1UUnitCMEPERM],
                                               [1, 2]>;
def C1UWrite_4c_2M_CMEPERMS_1rc : C1UCMEWriteRC<4, [C1UUnitM, C1UUnitM,
                                                   C1UUnitCMEPERM],
                                               [2, 2, 2]>;
def C1UWrite_4c_1M_CMEVX0ALU : C1UCMEWrite<4, [C1UUnitM, C1UUnitCMEVX0ALU]>;
def C1UWrite_8c_CMEPERMF_PERMS_1rc : C1UCMEWriteRC<8, [C1UUnitCME3PERMF,
                                                      C1UUnitCMEPERM],
                                                  [3, 2]>;
def C1UWrite_8c_CMEVXALU_PERMS_1rc : C1UCMEWriteRC<8, [C1UUnitCMEVXALU,
                                                      C1UUnitCMEPERM],
                                                  [2, 2]>;
def C1UWrite_12c_CMEVXALU_PERMF_2d3rc
    : C1UCMEWriteRC<12, [C1UUnitCMEVXALU, C1UUnitCME3PERMF], [3, 4]>;
def C1UWrite_16c_CMEVXALU_PERMF_2d3rc
    : C1UCMEWriteRC<16, [C1UUnitCMEVXALU, C1UUnitCME3PERMF], [3, 4]>;
def C1UWrite_16c_CMEVXALU_PERMS_2d3rc
    : C1UCMEWriteRC<16, [C1UUnitCMEVXALU, C1UUnitCMEPERM], [3, 3]>;
def C1UWrite_4c_CMEVXALU_PERMF_1d1rc
    : C1UCMEWriteRC<4, [C1UUnitCMEVXALU, C1UUnitCME3PERMF], [2, 3]>;
def C1UWrite_8c_CMEVXALU_PERMF_1rc
    : C1UCMEWriteRC<8, [C1UUnitCMEVXALU, C1UUnitCME3PERMF], [2, 3]>;
def C1UWrite_20c_CMEVXALU_PERMS_1d2rc
    : C1UCMEWriteRC<20, [C1UUnitCMEVXALU, C1UUnitCMEPERM], [4, 4]>;
def C1UWrite_24c_CMEVXALU_PERMS_2d5rc
    : C1UCMEWriteRC<24, [C1UUnitCMEVXALU, C1UUnitCMEPERM], [5, 5]>;
def C1UWrite_12c_CMEVX0ALU_PERMS_VXALU_2d5rc
    : C1UCMEWriteRC<12, [C1UUnitCMEVX0ALU, C1UUnitCMEPERM,
                         C1UUnitCMEVXALU],
                    [2, 5, 2]>;
def C1UWrite_2c_CMEMML0_2rc : C1UCMEWriteRC<2, [C1UUnitCMEMML0], [2]>;
def C1UWrite_2c_CMEMML0_4rc : C1UCMEWriteRC<2, [C1UUnitCMEMML0], [4]>;
def C1UWrite_2c_CMEPF_2rc : C1UCMEWriteRC<2, [C1UUnitCMEPF], [2]>;
def C1UWrite_6c_CMEVXALU_4rc : C1UCMEWriteRC<6, [C1UUnitCMEVXALU], [4]>;
def C1UWrite_7c_1L_CMELD : C1UCMEWrite<7, [C1UUnitL, C1UUnitCMELD]>;
def C1UWrite_7c_CMEPERMF_12rc : C1UCMEWriteRC<7, [C1UUnitCME3PERMF], [12]>;
def C1UWrite_7c_CMEVXALU_PERMF_1rc
    : C1UCMEWriteRC<7, [C1UUnitCMEVXALU, C1UUnitCME3PERMF], [2, 3]>;
def C1UWrite_7c_CMEVXALU_4rc : C1UCMEWriteRC<7, [C1UUnitCMEVXALU], [4]>;
def C1UWrite_8c_CMEPERMF_VX0ALU_1rc
    : C1UCMEWriteRC<8, [C1UUnitCME3PERMF, C1UUnitCMEVX0ALU], [3, 1]>;
def C1UWrite_9c_CMEMML1 : C1UCMEWrite<9, [C1UUnitCMEMML1]>;
def C1UWrite_9c_CMEMML1_2rc : C1UCMEWriteRC<9, [C1UUnitCMEMML1], [2]>;
def C1UWrite_9c_CMEMML1_5rc : C1UCMEWriteRC<9, [C1UUnitCMEMML1], [5]>;
def C1UWrite_11c_CMEVXALU_2rc : C1UCMEWriteRC<11, [C1UUnitCMEVXALU], [2]>;
def C1UWrite_11c_1SA_CMEST : C1UCMEWrite<11, [C1UUnitSA, C1UUnitCMEST]>;
def C1UWrite_13c_CMEMML1_PERMF_5rc
    : C1UCMEWriteRC<13, [C1UUnitCMEMML1, C1UUnitCME3PERMF], [5, 3]>;
def C1UWrite_10c_CMEMC_10rc : C1UCMEWriteRC<10, [C1UUnitCMEMC], [10]>;
def C1UWrite_11c_CMEMC_11rc : C1UCMEWriteRC<11, [C1UUnitCMEMC], [11]>;
def C1UWrite_12c_CMEMC_12rc : C1UCMEWriteRC<12, [C1UUnitCMEMC], [12]>;
def C1UWrite_17c_CMEMC_17rc : C1UCMEWriteRC<17, [C1UUnitCMEMC], [17]>;
def C1UWrite_25c_CMEMC_25rc : C1UCMEWriteRC<25, [C1UUnitCMEMC], [25]>;

// Define a small set of generic 1 micro-op types as placeholders.
def C1UWrite_2c_1B      : SchedWriteRes<[C1UUnitB]>  { let Latency = 2; }
def C1UWrite_2c_1B_1I   : SchedWriteRes<[C1UUnitB, C1UUnitI]>  { let Latency = 2; }
def C1UWrite_1c_1I      : SchedWriteRes<[C1UUnitI]>  { let Latency = 1; }
def C1UWrite_1c_8I          : SchedWriteRes<[
    C1UUnitI, C1UUnitI, C1UUnitI, C1UUnitI,
    C1UUnitI, C1UUnitI, C1UUnitI, C1UUnitI
]>  { let Latency = 1; }
def C1UWrite_2c_1I      : SchedWriteRes<[C1UUnitI]>  { let Latency = 2; }
def C1UWrite_3c_1I      : SchedWriteRes<[C1UUnitI]>  { let Latency = 3; }
def C1UWrite_2c_1I4     : SchedWriteRes<[C1UUnitI4]> { let Latency = 2; }
def C1UWrite_1c_1M      : SchedWriteRes<[C1UUnitM]>  { let Latency = 1; }
def C1UWrite_1c_2M      : SchedWriteRes<[C1UUnitM, C1UUnitM]>  { let Latency = 1; }
def C1UWrite_2c_1M      : SchedWriteRes<[C1UUnitM]>  { let Latency = 2; }
def C1UWrite_2c_2M      : SchedWriteRes<[C1UUnitM, C1UUnitM]>  { let Latency = 2; }
def C1UWrite_3c_1M      : SchedWriteRes<[C1UUnitM]>  { let Latency = 3; }
def C1UWrite_2c_1M0     : SchedWriteRes<[C1UUnitM0]> { let Latency = 2; }
def C1UWrite_3c_1M0     : SchedWriteRes<[C1UUnitM0]> { let Latency = 3; }
def C1UWrite_4c_1M0     : SchedWriteRes<[C1UUnitM0]> { let Latency = 4; }
def C1UWrite_12c_1M0    : SchedWriteRes<[C1UUnitM0]>  { let Latency = 12;
                                                        let ReleaseAtCycles = [12]; }
def C1UWrite_20c_1M0    : SchedWriteRes<[C1UUnitM0]>  { let Latency = 20;
                                                        let ReleaseAtCycles = [20]; }
def C1UWrite_2c_1V      : SchedWriteRes<[C1UUnitV]>  { let Latency = 2; }
def C1UWrite_2c_3V      : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV]>  { let Latency = 2; }
def C1UWrite_3c_1V      : SchedWriteRes<[C1UUnitV]>  { let Latency = 3; }
def C1UWrite_4c_1V      : SchedWriteRes<[C1UUnitV]>  { let Latency = 4; }
def C1UWrite_4c_2V      : SchedWriteRes<[C1UUnitV, C1UUnitV]>  { let Latency = 4; }
def C1UWrite_4c_3V      : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV]>  { let Latency = 4; }
def C1UWrite_6c_3V      : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV]>  { let Latency = 6; }
def C1UWrite_6c_5V      : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]>  { let Latency = 6; }
def C1UWrite_8c_4V      : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]>  { let Latency = 8; }
def C1UWrite_2c_1V0     : SchedWriteRes<[C1UUnitV0]> { let Latency = 2; }
def C1UWrite_4c_1V0     : SchedWriteRes<[C1UUnitV0]> { let Latency = 4; }
def C1UWrite_6c_6V0     : SchedWriteRes<[C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0, C1UUnitV0, C1UUnitV0, ]> { let Latency = 6; }
def C1UWrite_10c_10V0   : SchedWriteRes<[C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0]> { let Latency = 10; }
def C1UWrite_12c_7V0    : SchedWriteRes<[C1UUnitV0, C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0, C1UUnitV0, C1UUnitV0]> { let Latency = 12; }
def C1UWrite_20c_7V0    : SchedWriteRes<[C1UUnitV0, C1UUnitV0, C1UUnitV0, C1UUnitV0,
                                         C1UUnitV0, C1UUnitV0, C1UUnitV0]> { let Latency = 20; }
def C1UWrite_3c_1V1     : SchedWriteRes<[C1UUnitV1]> { let Latency = 3; }
def C1UWrite_5c_1V1     : SchedWriteRes<[C1UUnitV1]> { let Latency = 5; }
def C1UWrite_8c_1V1     : SchedWriteRes<[C1UUnitV1]> { let Latency = 8; }
def C1UWrite_12c_1V1    : SchedWriteRes<[C1UUnitV1]> { let Latency = 12; }
def C1UWrite_2c_1V01    : SchedWriteRes<[C1UUnitV01]>  { let Latency = 2; }
def C1UWrite_3c_1V01    : SchedWriteRes<[C1UUnitV01]>  { let Latency = 3; }
def C1UWrite_2c_1V0134  : SchedWriteRes<[C1UUnitV0134]>  { let Latency = 2; }
def C1UWrite_3c_1V0134  : SchedWriteRes<[C1UUnitV0134]>  { let Latency = 3; }
def C1UWrite_4c_1V0134  : SchedWriteRes<[C1UUnitV0134]>  { let Latency = 4; }
def C1UWrite_4c_2V0134  : SchedWriteRes<[C1UUnitV0134, C1UUnitV0134]>  { let Latency = 4; }
def C1UWrite_6c_1V0134  : SchedWriteRes<[C1UUnitV0134]>  { let Latency = 6; }
def C1UWrite_6c_4V0134  : SchedWriteRes<[C1UUnitV0134, C1UUnitV0134,
                                         C1UUnitV0134, C1UUnitV0134]> {
  let Latency = 6;
}
def C1UWrite_4c_1L      : SchedWriteRes<[C1UUnitL]>  { let Latency = 4; }
def C1UWrite_4c_1L_1I   : SchedWriteRes<[C1UUnitL, C1UUnitI]>  { let Latency = 4; }
def C1UWrite_6c_1L      : SchedWriteRes<[C1UUnitL]>  { let Latency = 6; }
def C1UWrite_1c_1SA     : SchedWriteRes<[C1UUnitSA]> { let Latency = 1; }

def C1UWrite_1c_1I_1Flg : SchedWriteRes<[C1UUnitI, C1UUnitFlg]> { let Latency = 1; }
def C1UWrite_2c_1I_1Flg : SchedWriteRes<[C1UUnitI, C1UUnitFlg]> { let Latency = 2; }

def C1UWrite_1c_1SA_1D : SchedWriteRes<[C1UUnitSA, C1UUnitD]> { let Latency = 1; }

def C1UWrite_1c_1SA_1D_1I : SchedWriteRes<[C1UUnitSA, C1UUnitD, C1UUnitI]> { let Latency = 1; }

def C1UWrite_2c_1SA_1V01 : SchedWriteRes<[C1UUnitSA, C1UUnitV01]> { let Latency = 2; }
def C1UWrite_2c_2SA_2V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitV01, C1UUnitV01]> { let Latency = 2; }
def C1UWrite_2c_3SA_3V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                         C1UUnitV01, C1UUnitV01, C1UUnitV01]> { let Latency = 2; }
def C1UWrite_2c_4SA_4V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01]> { let Latency = 2; }
def C1UWrite_2c_3SA_6V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01]> { let Latency = 2; }
def C1UWrite_2c_1SA_1V01_1I : SchedWriteRes<[C1UUnitSA, C1UUnitV01, C1UUnitI]> { let Latency = 2; }

def C1UWrite_4c_2L    : SchedWriteRes<[C1UUnitL, C1UUnitL]> { let Latency = 4; }
def C1UWrite_4c_2L_1I : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitI]> { let Latency = 4; }

def C1UWrite_5c_1L_1I : SchedWriteRes<[C1UUnitL, C1UUnitI]> {
  let Latency = 5;
}

def C1UWrite_5c_2L_1I : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitI]> { let Latency = 5; }

def C1UWrite_6c_1L_1I : SchedWriteRes<[C1UUnitL, C1UUnitI]> {
  let Latency = 6;
}

def C1UWrite_6c_2L_1I : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitI]> { let Latency = 6; }

def C1UWrite_7c_1L_1I : SchedWriteRes<[C1UUnitL, C1UUnitI]> {
  let Latency = 7;
}

def C1UWrite_5c_1M0_1V : SchedWriteRes<[C1UUnitM0, C1UUnitV]> {
  let Latency = 5;
}
def C1UWrite_3c_1M0_1V : SchedWriteRes<[C1UUnitM0, C1UUnitV]> {
  let Latency = 3;
}

def C1UWrite_4c_1V0134_1V : SchedWriteRes<[C1UUnitV0134, C1UUnitV]> { let Latency = 4; }

def C1UWrite_8c_1V1_4rc : SchedWriteRes<[C1UUnitV1]> {
  let Latency = 8;
  let ReleaseAtCycles = [4];
}

def C1UWrite_9c_1V1_2rc : SchedWriteRes<[C1UUnitV1]> {
  let Latency = 9;
  let ReleaseAtCycles = [2];
}

def C1UWrite_6c_1M0_1B : SchedWriteRes<[C1UUnitM0, C1UUnitB]> {
  let Latency = 6;
}

def C1UWrite_6c_1M0_1B_1I : SchedWriteRes<[C1UUnitM0, C1UUnitB, C1UUnitI]> {
  let Latency = 6;
}

def C1UWrite_9c_1M0_1L : SchedWriteRes<[C1UUnitM0, C1UUnitL]> {
  let Latency = 9;
}

def C1UWrite_6c_2L : SchedWriteRes<[C1UUnitL, C1UUnitL]> {
  let Latency = 6;
}

def C1UWrite_6c_3L : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL]> {
  let Latency = 6;
}

def C1UWrite_6c_4L : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL, C1UUnitL]> {
  let Latency = 6;
}

def C1UWrite_7c_4L : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL, C1UUnitL]> {
  let Latency = 7;
}

def C1UWrite_6c_1L_1V : SchedWriteRes<[C1UUnitL, C1UUnitV]> {
  let Latency = 6;
}

def C1UWrite_8c_1L_2V : SchedWriteRes<[C1UUnitL, C1UUnitV, C1UUnitV]> {
  let Latency = 8;
}

def C1UWrite_8c_2L_2V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitV, C1UUnitV]> {
  let Latency = 8;
}
def C1UWrite_9c_2L_2V_1I : SchedWriteRes<[C1UUnitL, C1UUnitL,
                                         C1UUnitV, C1UUnitV,
                                         C1UUnitI]> {
  let Latency = 9;
}

def C1UWrite_8c_2L_3V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitV, C1UUnitV,
                                      C1UUnitV]> {
  let Latency = 8;
}
def C1UWrite_9c_2L_3V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitV, C1UUnitV,
                                      C1UUnitV]> {
  let Latency = 9;
}

def C1UWrite_8c_3L_3V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL,
                                      C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 8;
}

def C1UWrite_9c_3L_3V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL,
                                      C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 9;
}

def C1UWrite_9c_3L_3V_1I : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL,
                                         C1UUnitV, C1UUnitV, C1UUnitV,
                                         C1UUnitI]> {
  let Latency = 9;
}
def C1UWrite_10c_4L_8V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL, C1UUnitL,
                                       C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV,
                                       C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 10;
}
def C1UWrite_11c_4L_8V_1I : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL, C1UUnitL,
                                          C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV,
                                          C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV,
                                          C1UUnitI]> {
  let Latency = 11;
}

def C1UWrite_9c_4L_8V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitL, C1UUnitL,
                                       C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV,
                                       C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 9;
}

def C1UWrite_8c_2L_4V : SchedWriteRes<[C1UUnitL, C1UUnitL, C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 8;
}

def C1UWrite_4c_1SA_1V01_1V : SchedWriteRes<[C1UUnitSA, C1UUnitV01, C1UUnitV]> { let Latency = 4; }
def C1UWrite_4c_2SA_2V01_2V : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV, C1UUnitV]> { let Latency = 4; }
def C1UWrite_5c_5SA_5V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01]> {
  let Latency = 5;
}
def C1UWrite_5c_4SA_4V01_4V : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                             C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]> { let Latency = 5; }

def C1UWrite_6c_1SA_1V01_1V : SchedWriteRes<[C1UUnitSA, C1UUnitV01, C1UUnitV]> {
  let Latency = 6;
}

def C1UWrite_6c_6SA_6V01_6V : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                             C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                             C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV, C1UUnitV, C1UUnitV,
                                             C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 6;
}

def C1UWrite_6c_8SA_8V01_8V : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                             C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                             C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                             C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV,
                                             C1UUnitV, C1UUnitV, C1UUnitV, C1UUnitV]> {
  let Latency = 6;
}

def C1UWrite_7c_5SA_5V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01]> {
  let Latency = 7;
}

def C1UWrite_7c_12SA_12V01_12V : SchedWriteRes<[
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV,
    C1UUnitSA, C1UUnitV01, C1UUnitV
]> {
  let Latency = 7;
}

def C1UWrite_7c_1M_1M0_1V : SchedWriteRes<[C1UUnitM, C1UUnitM0, C1UUnitV]> { let Latency = 7; }

def C1UWrite_6c_2V1 : SchedWriteRes<[C1UUnitV1, C1UUnitV1]> { let Latency = 6; }

def C1UWrite_8c_1M0_1V1_1V01 : SchedWriteRes<[C1UUnitM0, C1UUnitV1, C1UUnitV01]> { let Latency = 8; }

def C1UWrite_5c_1V1_1V01 : SchedWriteRes<[C1UUnitV1, C1UUnitV01]> { let Latency = 5; }

def C1UWrite_7c_1M0_1V0134 : SchedWriteRes<[C1UUnitM0, C1UUnitV0134]> { let Latency = 7; }

def C1UWrite_5c_1V_1M0 : SchedWriteRes<[C1UUnitV, C1UUnitM0]> { let Latency = 5; }

def C1UWrite_8c_1V_3V0134 : SchedWriteRes<[C1UUnitV, C1UUnitV0134, C1UUnitV0134, C1UUnitV0134]> { let Latency = 8; }

def C1UWrite_6c_1V_2V0134 : SchedWriteRes<[C1UUnitV, C1UUnitV0134, C1UUnitV0134]> { let Latency = 6; }

def C1UWrite_6c_2V_1V0134 : SchedWriteRes<[C1UUnitV, C1UUnitV, C1UUnitV0134]> { let Latency = 6; }

def C1UWrite_4c_1V_1V0134 : SchedWriteRes<[C1UUnitV, C1UUnitV0134]> { let Latency = 4; }

def C1UWrite_12c_1V0_8rc : SchedWriteRes<[C1UUnitV0]> {
  let Latency = 12;
  let ReleaseAtCycles = [8];
}

def C1UWrite_11c_1V0_4rc : SchedWriteRes<[C1UUnitV0]> {
  let Latency = 11;
  let ReleaseAtCycles = [4];
}

def C1UWrite_13c_1V0_2rc : SchedWriteRes<[C1UUnitV0]> {
  let Latency = 13;
  let ReleaseAtCycles = [2];
}

def C1UWrite_12c_1V1_8rc : SchedWriteRes<[C1UUnitV1]> {
  let Latency = 12;
  let ReleaseAtCycles = [8];
}

def C1UWrite_11c_1V1_4rc : SchedWriteRes<[C1UUnitV1]> {
  let Latency = 11;
  let ReleaseAtCycles = [4];
}

def C1UWrite_13c_1V1_2rc : SchedWriteRes<[C1UUnitV1]> {
  let Latency = 13;
  let ReleaseAtCycles = [2];
}

def C1UWrite_6c_1L_1M : SchedWriteRes<[C1UUnitL, C1UUnitM]> {
  let Latency = 6;
}

def C1UWrite_9c_1V01_1L : SchedWriteRes<[C1UUnitV01, C1UUnitL]> {
  let Latency = 9;
}

def C1UWrite_4c_2SA_2V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA,
                                         C1UUnitV01, C1UUnitV01]> {
  let Latency     = 4;
}
def C1UWrite_4c_2SA_2V01_1I : SchedWriteRes<[C1UUnitSA, C1UUnitSA,
                                            C1UUnitV01, C1UUnitV01,
                                            C1UUnitI]> {
  let Latency     = 4;
}
def C1UWrite_5c_3SA_3V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                         C1UUnitV01, C1UUnitV01, C1UUnitV01]> {
  let Latency     = 5;
}
def C1UWrite_5c_3SA_3V01_1I : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                            C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                            C1UUnitI]> {
  let Latency     = 5;
}
def C1UWrite_7c_4SA_4V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                         C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01]> {
  let Latency     = 7;
}
def C1UWrite_7c_4SA_4V01_1I : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                            C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                            C1UUnitI]> {
  let Latency     = 7;
}

def C1UWrite_4c_1SA_1V01 : SchedWriteRes<[C1UUnitSA, C1UUnitV01]> { let Latency = 4; }
def C1UWrite_4c_3SA_3V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01]> { let Latency = 4; }
def C1UWrite_4c_3SA_6V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01]> { let Latency = 4; }

def C1UWrite_5c_3SA_6V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01]> { let Latency = 5; }

def C1UWrite_6c_3SA_3V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01, C1UUnitV01]> { let Latency = 6; }

def C1UWrite_7c_3SA_6V01 : SchedWriteRes<[C1UUnitSA, C1UUnitSA, C1UUnitSA,
                                          C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01,
                                          C1UUnitV01, C1UUnitV01]> { let Latency = 7;}

// Forwarded types
def C1UWr_FMA     : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_FMA     : SchedReadAdvance<2, [WriteFMul, C1UWr_FMA]>;

def C1UWr_VA      : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_VA      : SchedReadAdvance<3, [C1UWr_VA]>;

def C1UWr_VDOT    : SchedWriteRes<[C1UUnitV]> { let Latency = 3; }
def C1URd_VDOT    : SchedReadAdvance<2, [C1UWr_VDOT]>;

def C1UWr_VMMA    : SchedWriteRes<[C1UUnitV]> { let Latency = 3; }
def C1URd_VMMA    : SchedReadAdvance<2, [C1UWr_VMMA]>;

def C1UWr_MA      : SchedWriteRes<[C1UUnitI4]> { let Latency = 3; }
def C1URd_MA      : SchedReadAdvance<2, [C1UWr_MA]>;
def C1UWrite_MAOrMul : SchedWriteVariant<[
                        SchedVar<IsReg3ZeroPred, [C1UWrite_2c_1I4]>,
                        SchedVar<NoSchedPred,    [C1UWr_MA]>]>;

def C1UWr_VMA     : SchedWriteRes<[C1UUnitV0134]> { let Latency = 4; }
def C1URd_VMA     : SchedReadAdvance<3, [C1UWr_VMA]>;

def C1UWr_VMAH    : SchedWriteRes<[C1UUnitV0134]> { let Latency = 4; }
def C1URd_VMAH    : SchedReadAdvance<2, [C1UWr_VMAH]>;

def C1UWr_VPA     : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_VPA     : SchedReadAdvance<3, [C1UWr_VPA]>;

def C1UWr_VSA     : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_VSA     : SchedReadAdvance<3, [C1UWr_VSA]>;

def C1UWr_VFCMA   : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_VFCMA   : SchedReadAdvance<3, [C1UWr_VFCMA]>;

def C1UWr_VFMA    : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_VFMA    : SchedReadAdvance<2, [C1UWr_VFMA]>;

def C1UWr_VBFDOT  : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_VBFDOT  : SchedReadAdvance<2, [C1UWr_VBFDOT]>;

def C1UWr_VBFMMA  : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_VBFMMA  : SchedReadAdvance<1, [C1UWr_VBFMMA]>;

def C1UWr_VBFMAL  : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_VBFMAL  : SchedReadAdvance<3, [C1UWr_VBFMAL]>;

def C1UWr_ZA      : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_ZA      : SchedReadAdvance<3, [C1UWr_ZA]>;
def C1UWr_ZPA     : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_ZPA     : SchedReadAdvance<3, [C1UWr_ZPA]>;
def C1UWr_ZSA     : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_ZSA     : SchedReadAdvance<3, [C1UWr_ZSA]>;

def C1UWr_ZDOTB   : SchedWriteRes<[C1UUnitV]>   { let Latency = 3; }
def C1URd_ZDOTB   : SchedReadAdvance<2, [C1UWr_ZDOTB]>;
def C1UWr_ZDOTH   : SchedWriteRes<[C1UUnitV0134]> { let Latency = 3; }
def C1URd_ZDOTH   : SchedReadAdvance<2, [C1UWr_ZDOTH]>;

def C1UWr_ZCMABHS : SchedWriteRes<[C1UUnitV0134]> { let Latency = 4; }
def C1URd_ZCMABHS : SchedReadAdvance<3, [C1UWr_ZCMABHS]>;

def C1UWr_ZMMA    : SchedWriteRes<[C1UUnitV]> { let Latency = 3; }
def C1URd_ZMMA    : SchedReadAdvance<2, [C1UWr_ZMMA]>;

def C1UWr_ZMA     : SchedWriteRes<[C1UUnitV0134]> { let Latency = 4; }
def C1URd_ZMA     : SchedReadAdvance<3, [C1UWr_ZMA]>;

def C1UWr_ZMASQL  : SchedWriteRes<[C1UUnitV0134]> { let Latency = 4; }
def C1URd_ZMASQL  : SchedReadAdvance<2, [C1UWr_ZMASQL]>;

def C1UWr_ZFCMA   : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_ZFCMA   : SchedReadAdvance<3, [C1UWr_ZFCMA]>;

def C1UWr_ZFMA    : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_ZFMA    : SchedReadAdvance<2, [C1UWr_ZFMA]>;

def C1UWr_ZFMAL   : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_ZFMAL   : SchedReadAdvance<3, [C1UWr_ZFMAL]>;


def C1UWr_ZBFDOT  : SchedWriteRes<[C1UUnitV]> { let Latency = 4; }
def C1URd_ZBFDOT  : SchedReadAdvance<2, [C1UWr_ZBFDOT]>;
def C1UWr_ZBFMMA  : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_ZBFMMA  : SchedReadAdvance<2, [C1UWr_ZBFMMA]>;
def C1UWr_ZBFMAL  : SchedWriteRes<[C1UUnitV]> { let Latency = 5; }
def C1URd_ZBFMAL  : SchedReadAdvance<3, [C1UWr_ZBFMAL]>;

def C1UWr_Pred    : SchedWriteRes<[C1UUnitM, C1UUnitM]> { let Latency     = 2; }
def C1URd_Pred    : SchedReadAdvance<1, [C1UWr_Pred]>;

// Predicate controlled types
def C1UWrite_ArithI : SchedWriteVariant<[
                       SchedVar<IsCheapLSL,  [C1UWrite_1c_1I]>,
                       SchedVar<NoSchedPred, [C1UWrite_2c_1I]>]>;

def C1UWrite_Extr : SchedWriteVariant<[
                     SchedVar<IsRORImmIdiomPred, [C1UWrite_1c_1I]>,
                     SchedVar<NoSchedPred,       [C1UWrite_3c_1I]>]>;

def C1UWrite_1or2c_1I : SchedWriteVariant<[
                      SchedVar<IsXOrWDest, [C1UWrite_1c_1I]>,
                      SchedVar<NoSchedPred, [C1UWrite_2c_1I]>]>;

def C1UWrite_1or2c_1I_1Flg : SchedWriteVariant<[
                      SchedVar<IsXOrWDest, [C1UWrite_1c_1I_1Flg]>,
                      SchedVar<NoSchedPred, [C1UWrite_2c_1I_1Flg]>]>;


def C1UWrite_2or4c_1SA_1V01 : SchedWriteVariant<[
                                SchedVar<SameZRegDstSrcPred, [C1UWrite_4c_1SA_1V01]>,
                                SchedVar<NoSchedPred, [C1UWrite_2c_1SA_1V01]>]>;

def C1UWrite_2or4c_3SA_6V01 : SchedWriteVariant<[
                                SchedVar<SameZRegDstSrcPred, [C1UWrite_4c_3SA_6V01]>,
                                SchedVar<NoSchedPred, [C1UWrite_2c_3SA_6V01]>]>;

def C1UWrite_4or6c_3SA_3V01 : SchedWriteVariant<[
                                SchedVar<SameZRegDstSrcPred, [C1UWrite_6c_3SA_3V01]>,
                                SchedVar<NoSchedPred, [C1UWrite_4c_3SA_3V01]>]>;

def C1UWrite_5or7c_3SA_6V01 : SchedWriteVariant<[
                                SchedVar<SameZRegDstSrcPred, [C1UWrite_7c_3SA_6V01]>,
                                SchedVar<NoSchedPred, [C1UWrite_5c_3SA_6V01]>]>;


def C1UWrite_5or7c_5SA_5V01 : SchedWriteVariant<[
                                SchedVar<SameZRegDstSrcPred, [C1UWrite_7c_5SA_5V01]>,
                                SchedVar<NoSchedPred, [C1UWrite_5c_5SA_5V01]>]>;

def C1UWr_FMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_FMA>;
def C1UWr_VA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_VA>;
def C1UWr_VDOT_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_VDOT>;
def C1UWr_VMMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEMML0_1rc,
                                            C1UWr_VMMA>;
def C1UWr_VMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_VMA>;
def C1UWr_VMAH_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_VMAH>;
def C1UWr_VPA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_VPA>;
def C1UWr_VSA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_VSA>;
def C1UWr_VFCMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                             C1UWr_VFCMA>;
def C1UWr_VFMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_VFMA>;
def C1UWr_VBFDOT_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXFPDOT,
                                              C1UWr_VBFDOT>;
def C1UWr_VBFMMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEMML0_1rc,
                                              C1UWr_VBFMMA>;
def C1UWr_VBFMAL_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                              C1UWr_VBFMAL>;
def C1UWr_ZA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_ZA>;
def C1UWr_ZPA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_ZPA>;
def C1UWr_ZSA_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_1rc, C1UWr_ZSA>;
def C1UWr_ZDOTB_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_ZDOTB>;
def C1UWr_ZDOTH_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXMUL_1rc,
                                             C1UWr_ZDOTH>;
def C1UWr_ZCMABHS_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                               C1UWr_ZCMABHS>;
def C1UWr_ZMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_ZMA>;
def C1UWr_ZMASQL_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                              C1UWr_ZMASQL>;
def C1UWr_ZFCMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                             C1UWr_ZFCMA>;
def C1UWr_ZFMA_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWr_ZFMA>;
def C1UWr_ZFMAL_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                             C1UWr_ZFMAL>;
def C1UWr_ZBFDOT_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXFPDOT,
                                              C1UWr_ZBFDOT>;
def C1UWr_ZBFMAL_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL,
                                              C1UWr_ZBFMAL>;

def C1UWrite_8c_1V_3V0134_CMEIntReduce
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_8c_1V_3V0134>;
def C1UWrite_6c_1V_2V0134_CMEIntReduce
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_6c_1V_2V0134>;
def C1UWrite_6c_2V_1V0134_CMEIntReduce
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_6c_2V_1V0134>;
def C1UWrite_4c_1V_1V0134_CMEIntReduce
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_4c_1V_1V0134>;
def C1UWrite_4c_2V_CMEIntReduce
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_4c_2V>;
def C1UWrite_8c_4V_CMEFPReduceH
    : C1UStreamingSchedWrite<C1UWrite_24c_CMEVXALU_PERMS_2d5rc,
                             C1UWrite_8c_4V>;
def C1UWrite_6c_3V_CMEFPReduceS
    : C1UStreamingSchedWrite<C1UWrite_20c_CMEVXALU_PERMS_1d2rc,
                             C1UWrite_6c_3V>;
def C1UWrite_8c_4V_CMEFPReduceMaxMin
    : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_8c_4V>;
def C1UWrite_6c_3V_CMEFPReduceMaxMin
    : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_6c_3V>;
def C1UWrite_4c_2V_CMEFPReduceD
    : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMS_2d3rc,
                             C1UWrite_4c_2V>;
def C1UWrite_4c_2V_CMEFPReduceMaxMinD
    : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_4c_2V>;

def C1UWrite_2c_1V_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                C1UWrite_2c_1V>;
def C1UWrite_2c_1V0_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVX0ALU,
                                                 C1UWrite_2c_1V0>;
def C1UWrite_2c_1V01_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                  C1UWrite_2c_1V01>;
def C1UWrite_2c_1V0134_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                    C1UWrite_2c_1V0134>;
def C1UWrite_3c_1V0134_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                    C1UWrite_3c_1V0134>;
def C1UWrite_3c_1V1_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEPERMF_1rc,
                                                 C1UWrite_3c_1V1>;
def C1UWrite_4c_1V_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                C1UWrite_4c_1V>;
def C1UWrite_4c_1V0134_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                    C1UWrite_4c_1V0134>;
def C1UWrite_4c_1V0134_1V_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_4c_1V0134_1V>;
def C1UWrite_4c_1V_1V0134_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_4c_1V_1V0134>;
def C1UWrite_4c_2V_CME : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMF_2d3rc,
                                                C1UWrite_4c_2V>;
def C1UWrite_4c_2V0134_CME : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                    C1UWrite_4c_2V0134>;
def C1UWrite_6c_1V_2V0134_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_6c_1V_2V0134>;
def C1UWrite_6c_2V_1V0134_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_6c_2V_1V0134>;
def C1UWrite_6c_3V_CME : C1UStreamingSchedWrite<C1UWrite_16c_CMEVXALU_PERMF_2d3rc,
                                                C1UWrite_6c_3V>;
def C1UWrite_8c_1V1_4rc_CME
    : C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc, C1UWrite_8c_1V1_4rc>;
def C1UWrite_8c_1V_3V0134_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVXALU_PERMF_2d3rc,
                             C1UWrite_8c_1V_3V0134>;
def C1UWrite_9c_1V1_2rc_CME
    : C1UStreamingSchedWrite<C1UWrite_11c_CMEMC_11rc, C1UWrite_9c_1V1_2rc>;
def C1UWrite_11c_1V0_4rc_CME
    : C1UStreamingSchedWrite<C1UWrite_11c_CMEMC_11rc, C1UWrite_11c_1V0_4rc>;
def C1UWrite_11c_1V1_4rc_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEMC_12rc, C1UWrite_11c_1V1_4rc>;
def C1UWrite_12c_1V0_8rc_CME
    : C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc, C1UWrite_12c_1V0_8rc>;
def C1UWrite_12c_1V1_8rc_CME
    : C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc, C1UWrite_12c_1V1_8rc>;
def C1UWrite_12c_7V0_CME
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEMC_12rc, C1UWrite_12c_7V0>;
def C1UWrite_13c_1V0_2rc_CME
    : C1UStreamingSchedWrite<C1UWrite_17c_CMEMC_17rc, C1UWrite_13c_1V0_2rc>;
def C1UWrite_13c_1V1_2rc_CME
    : C1UStreamingSchedWrite<C1UWrite_25c_CMEMC_25rc, C1UWrite_13c_1V1_2rc>;
def C1UWrite_20c_7V0_CME
    : C1UStreamingSchedWrite<C1UWrite_20c_7V0, C1UWrite_20c_7V0>;

def C1UWrite_3c_1M0_CME : C1UStreamingSchedWrite<C1UWrite_4c_1M0_CMEVXALU,
                                                 C1UWrite_3c_1M0>;
def C1UWrite_5c_1M0_1V_CME : C1UStreamingSchedWrite<C1UWrite_4c_1M0_CMEVXALU,
                                                    C1UWrite_5c_1M0_1V>;
def C1UWrite_5c_1V1_1V01_CME : C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_PERMF_1rc,
                                                      C1UWrite_5c_1V1_1V01>;
def C1UWrite_5c_1V_1M0_CME : C1UStreamingSchedWrite<C1UWrite_4c_1M_CMEPERMS_1rc,
                                                    C1UWrite_5c_1V_1M0>;
def C1UWrite_7c_1M0_1V0134_CME
    : C1UStreamingSchedWrite<C1UCMEWriteRC<12, [C1UUnitM0, C1UUnitCMEVX0ALU,
                                                C1UUnitCMEPERM,
                                                C1UUnitCMEVXALU],
                                            [12, 12, 12, 12]>,
                             C1UWrite_7c_1M0_1V0134>;
def C1UWrite_8c_1M0_1V1_1V01_CME
    : C1UStreamingSchedWrite<C1UCMEWriteRC<8, [C1UUnitCMEVXALU,
                                               C1UUnitCME3PERMF],
                                           [8, 8]>,
                             C1UWrite_8c_1M0_1V1_1V01>;

def C1UWrite_6c_1L_CME : C1UStreamingSchedWrite<C1UWrite_2c_1L_CMELD,
                                                C1UWrite_6c_1L>;
def C1UWrite_6c_1L_1I_CME : C1UStreamingSchedWrite<C1UWrite_2c_1I_1L_CMELD,
                                                   C1UWrite_6c_1L_1I>;
def C1UWrite_6c_1L_1M_CME : C1UStreamingSchedWrite<C1UWrite_2c_1L_CMELD,
                                                   C1UWrite_6c_1L_1M>;
def C1UWrite_6c_2L_1I_CME : C1UStreamingSchedWrite<C1UWrite_2c_1I_1L_CMELD,
                                                   C1UWrite_6c_2L_1I>;
def C1UWrite_7c_1L_1I_CME : C1UStreamingSchedWrite<C1UWrite_2c_1I_1L_CMELD,
                                                   C1UWrite_7c_1L_1I>;
def C1UWrite_8c_2L_2V_CME : C1UStreamingSchedWrite<C1UWrite_2c_1V01_1L_CMELD,
                                                   C1UWrite_8c_2L_2V>;
def C1UWrite_8c_3L_3V_CME : C1UStreamingSchedWrite<C1UWrite_2c_1V01_1L_CMELD,
                                                   C1UWrite_8c_3L_3V>;
def C1UWrite_9c_2L_2V_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1I_1V01_1L_CMELD,
                             C1UWrite_9c_2L_2V_1I>;
def C1UWrite_9c_3L_3V_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1I_1V01_1L_CMELD,
                             C1UWrite_9c_3L_3V_1I>;
def C1UWrite_10c_4L_8V_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1V01_1L_CMELD, C1UWrite_10c_4L_8V>;
def C1UWrite_11c_4L_8V_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1I_1V01_1L_CMELD,
                             C1UWrite_11c_4L_8V_1I>;

def C1UWrite_LdrQ_CME : SchedWriteVariant<[
                          SchedVar<FPIsQForm, [C1UWrite_7c_1L_1I_CME]>,
                          SchedVar<NoSchedPred, [C1UWrite_6c_1L_CME]>]>;

def C1UWrite_1c_1SA_CME : C1UStreamingSchedWrite<C1UWrite_2c_1SA_CMEST,
                                                 C1UWrite_1c_1SA>;
def C1UWrite_2c_1SA_1V01_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1SA_1V01_CMEST,
                             C1UWrite_2c_1SA_1V01>;
def C1UWrite_2c_1SA_1V01_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_1SA_1V01_1I_CMEST,
                             C1UWrite_2c_1SA_1V01_1I>;
def C1UWrite_2c_2SA_2V01_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_2SA_2V01_CMEST,
                             C1UWrite_2c_2SA_2V01>;
def C1UWrite_4c_2SA_2V01_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_2SA_2V01_CMEST,
                             C1UWrite_4c_2SA_2V01>;
def C1UWrite_4c_2SA_2V01_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_2SA_2V01_CMEST,
                             C1UWrite_4c_2SA_2V01_1I>;
def C1UWrite_5c_3SA_3V01_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_3SA_3V01_CMEST,
                             C1UWrite_5c_3SA_3V01>;
def C1UWrite_5c_3SA_3V01_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_3SA_3V01_CMEST,
                             C1UWrite_5c_3SA_3V01_1I>;
def C1UWrite_7c_4SA_4V01_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_4SA_4V01_CMEST,
                             C1UWrite_7c_4SA_4V01>;
def C1UWrite_7c_4SA_4V01_1I_CME
    : C1UStreamingSchedWrite<C1UWrite_2c_4SA_4V01_CMEST,
                             C1UWrite_7c_4SA_4V01_1I>;

def C1UWrite_StrQ_CME : SchedWriteVariant<[
                          SchedVar<FPIsQForm, [C1UWrite_2c_1SA_1V01_1I_CME]>,
                          SchedVar<NoSchedPred, [C1UWrite_2c_1SA_1V01_CME]>]>;

// SME predicates

def C1UWrite_1c_1I_Streaming : SchedWriteVariant<[
                      SchedVar<SMESchedPred, [C1UWrite_1c_1I]>,
                      SchedVar<NoSchedPred,          [C1UWrite_0c]>]>;

def C1UWrite_1c_1M_Streaming : SchedWriteVariant<[
                      SchedVar<SMESchedPred, [C1UWrite_1c_1M]>,
                      SchedVar<NoSchedPred,          [C1UWrite_0c]>]>;

def C1UWrite_2c_1M_Streaming : SchedWriteVariant<[
                      SchedVar<SMESchedPred, [C1UWrite_2c_1M]>,
                      SchedVar<NoSchedPred,          [C1UWrite_0c]>]>;

def C1UWrite_3c_1M_Streaming : SchedWriteVariant<[
                      SchedVar<SMESchedPred, [C1UWrite_3c_1M]>,
                      SchedVar<NoSchedPred,          [C1UWrite_0c]>]>;

def C1UWrite_2c_1V_StreamingCME : C1USMEStreamingSchedWrite<C1UWrite_4c_CMEVXALU,
                                                            C1UWrite_2c_1V>;
def C1UWrite_3c_1V01_StreamingCME
    : C1USMEStreamingSchedWrite<C1UWrite_4c_CMEVXALU, C1UWrite_3c_1V01>;
def C1UWrite_2c_1M_StreamingCME
    : C1USMEStreamingSchedWrite<C1UWrite_4c_1M_CMEVX0ALU, C1UWrite_2c_1M>;
def C1UWrite_2c_2M_StreamingCME
    : C1USMEStreamingSchedWrite<C1UWrite_4c_2M_CMEPERMS_1rc, C1UWrite_2c_2M>;
def C1UWrite_2c_1V_StreamingCMEPERMS
    : C1USMEStreamingSchedWrite<C1UWrite_4c_CMEPERMS, C1UWrite_2c_1V>;
def C1UWrite_2c_1V_StreamingCMEVXALU_PERMS
    : C1USMEStreamingSchedWrite<C1UWrite_8c_CMEVXALU_PERMS_1rc, C1UWrite_2c_1V>;
def C1UWrite_2c_1V_StreamingCMEVXALU_5c
    : C1USMEStreamingSchedWrite<C1UWrite_5c_CMEVXALU_1rc, C1UWrite_2c_1V>;
def C1UWrite_4c_1V_StreamingCMEMML0
    : C1USMEStreamingSchedWrite<C1UWrite_4c_CMEMML0_2rc, C1UWrite_4c_1V>;
def C1UWrite_2c_1M_StreamingCMEVX0ALU
    : C1USMEStreamingSchedWrite<C1UWrite_4c_1M_CMEVX0ALU, C1UWrite_2c_1M>;
def C1UWrite_2c_2M_StreamingCMEPERMS
    : C1USMEStreamingSchedWrite<C1UWrite_4c_2M_CMEPERMS_1rc, C1UWrite_2c_2M>;
def C1UWrite_2c_1V_CMEVXALU
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU, C1UWrite_2c_1V>;
def C1UWrite_2c_1V_CMEPERMS
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEPERMS, C1UWrite_2c_1V>;
def C1UWrite_2c_1V_CMEPERMF_1rc
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEPERMF_1rc, C1UWrite_2c_1V>;
def C1UWrite_2c_1V_CMEPERMS_1c
    : C1UStreamingSchedWrite<C1UWrite_1c_CMEPERMS, C1UWrite_2c_1V>;
def C1UWrite_2c_1V0_CMEVX0ALU
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVX0ALU, C1UWrite_2c_1V0>;
def C1UWrite_3c_1V01_CMEVXALU
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU, C1UWrite_3c_1V01>;
def C1UWrite_3c_1V_CMEVXALU
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU, C1UWrite_3c_1V>;
def C1UWrite_3c_1V_CMEVXMUL
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWrite_3c_1V>;
def C1UWrite_4c_1V_CMEVXALU
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU, C1UWrite_4c_1V>;
def C1UWrite_4c_1V_CMEVXMUL
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWrite_4c_1V>;
def C1UWrite_4c_1V0134_CMEVXMUL
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWrite_4c_1V0134>;
def C1UWrite_3c_1V0134_CMEVXMUL
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWrite_3c_1V0134>;
def C1UWrite_6c_1V0134_CMEVXMUL
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXMUL, C1UWrite_6c_1V0134>;
def C1UWrite_3c_1M0_1V_CMEPERMF_PERMS
    : C1UStreamingSchedWrite<C1UWrite_8c_CMEPERMF_PERMS_1rc,
                             C1UWrite_3c_1M0_1V>;
def C1UWrite_3c_1M0_CMEPERMF
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEPERMF, C1UWrite_3c_1M0>;
def C1UWrite_7c_1M0_1V0134_CMEINDEX_IRRI
    : C1UStreamingSchedWrite<C1UWrite_4c_CMEVXALU_PERMF_1d1rc,
                             C1UWrite_7c_1M0_1V0134>;
def C1UWrite_7c_1M0_1V0134_CMEINDEX_RR
    : C1UStreamingSchedWrite<C1UWrite_12c_CMEVX0ALU_PERMS_VXALU_2d5rc,
                             C1UWrite_7c_1M0_1V0134>;
def C1UWrite_8c_1M0_1V1_1V01_CMECLAST
    : C1UStreamingSchedWrite<C1UWrite_12c_1M0_CMEPERMF_VX0ALU_2rc,
                             C1UWrite_8c_1M0_1V1_1V01>;


// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;
def : WriteRes<WriteSys, []> { let Latency = 1; }


//===----------------------------------------------------------------------===//
// Instruction scheduling classes.
//===----------------------------------------------------------------------===//

// Branch instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr,    C1UWrite_2c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, C1UWrite_2c_1B>;

// Branch and link, immed
// Branch and link, register
def : InstRW<[C1UWrite_2c_1B_1I], (instrs BL, BLR)>;


// Arithmetic and logical operations
// -----------------------------------------------------------------------------

// ALU, basic
def : SchedAlias<WriteI, C1UWrite_1c_1I>;

// ALU, basic, flagset
def : InstRW<[C1UWrite_1c_1I_1Flg],
             (instregex "^(ADC|SBC)S[WX]r$",
                        "^ANDS[WX]ri$",
                        "^(AND|BIC)S[WX]rr$")>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, C1UWrite_1or2c_1I>;

// ALU, extend and shift, flagset
def : InstRW<[C1UWrite_1or2c_1I_1Flg],
             (instregex "^(ADD|SUB)S[WX]r[irx]$",
                        "^(ADD|SUB)SXrx64$")>;
def : InstRW<[C1UWrite_2c_1I_1Flg],
             (instregex "^(ADD|SUB)S[WX]rs$")>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def : SchedAlias<WriteISReg, C1UWrite_ArithI>;

// Arithmetic, immediate to logical address tag
def : InstRW<[C1UWrite_2c_1I], (instrs ADDG, SUBG)>;

// Conditional compare
def : InstRW<[C1UWrite_1c_1I_1Flg], (instregex "^CCM[NP][WX][ir]")>;

// Conditional select
def : InstRW<[C1UWrite_1c_1I],
              (instregex "(CSEL|CSINC|CSINV|CSNEG)(X|W)[ir]")>;

// Convert floating-point condition flags
// Flag manipulation instructions
def : InstRW<[C1UWrite_1c_8I],
              (instrs AXFLAG, XAFLAG, SETF8, SETF16, RMIF, CFINV)>;

// Insert Random Tags
def : InstRW<[C1UWrite_2c_1M], (instrs IRG, IRGstack)>;

// Insert Tag Mask
// Subtract Pointer
def : InstRW<[C1UWrite_1c_1I], (instrs GMI, SUBP)>;

// Subtract Pointer, flagset
def : InstRW<[C1UWrite_1c_1I_1Flg], (instrs SUBPS)>;

// Logical, shift, no flagset
def : InstRW<[C1UWrite_1c_1I],
             (instregex "^(AND|BIC|EON|EOR|ORN)[WX]rs",
                        "^ORR[WX]rs$")>;

// Logical, shift, flagset
def : InstRW<[C1UWrite_1c_1I_1Flg], (instregex "^(AND|BIC)S[WX]rs$")>;

// Move and shift instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteImm, C1UWrite_1c_1I>;

// Divide and multiply instructions
// -----------------------------------------------------------------------------

// SDIV, UDIV
def : SchedAlias<WriteID32,  C1UWrite_12c_1M0>;
def : SchedAlias<WriteID64,  C1UWrite_20c_1M0>;

def : SchedAlias<WriteIM32, C1UWrite_2c_1I4>;
def : SchedAlias<WriteIM64, C1UWrite_2c_1I4>;

// Multiply accumulate, W-form
// Multiply accumulate, X-form
def : InstRW<[C1UWrite_MAOrMul, ReadIM, ReadIM, C1URd_MA],
             (instregex "^M(ADD|SUB)[WX]rrr$")>;

// Multiply accumulate long
def : InstRW<[C1UWrite_MAOrMul, ReadIM, ReadIM, C1URd_MA],
             (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[C1UWrite_2c_1I4], (instrs SMULHrr, UMULHrr)>;

// Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code, using generic key
// Compute pointer authentication code for instruction address
def : InstRW<[C1UWrite_4c_1M0], (instregex "^AUT", "^PAC")>;

// Branch and link, register, with pointer authentication
def : InstRW<[C1UWrite_6c_1M0_1B_1I], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ)>;

// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[C1UWrite_6c_1M0_1B], (instrs BRAA, BRAAZ, BRAB, BRABZ,
                                   RETAA, RETAB, ERETAA, ERETAB)>;


// Load register, with pointer authentication
def : InstRW<[C1UWrite_9c_1M0_1L], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[C1UWrite_2c_1M0], (instrs XPACD, XPACI, XPACLRI)>;

// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Address generation
def : InstRW<[C1UWrite_1c_1I], (instrs ADR, ADRP)>;

// Bitfield extract, one reg
// Bitfield extract, two regs
def : SchedAlias<WriteExtr, C1UWrite_Extr>;
def : InstRW<[C1UWrite_Extr], (instrs EXTRWrri, EXTRXrri)>;

// Bitfield move, basic
def : SchedAlias<WriteIS, C1UWrite_1c_1I>;

// Bitfield move, insert
def : InstRW<[C1UWrite_2c_1I], (instregex "^BFM[WX]ri$")>;

// Count leading
// Move immed
// Reverse bits/bytes
// Variable shift
def : InstRW<[C1UWrite_1c_1I],
             (instregex "^(CLS|CLZ)(W|X)r",
                        "^(MOVN|MOVK|MOVZ)(W|X)i",
                        "^(RBIT|REV(16|32)?)(W|X)r",
                        "^(ASRV|LSLV|LSRV|RORV)(W|X)r")>;
// Load instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteLD,    C1UWrite_4c_1L>;
def : SchedAlias<WriteLDIdx, C1UWrite_4c_1L_1I>;

// Load register, literal
def : InstRW<[C1UWrite_5c_1L_1I], (instrs LDRWl, LDRXl, LDRSWl, PRFMl)>;

// Load pair, signed immed offset, normal X form
// Load pair, signed immed post-index or immed pre-index, normal X form
def : InstRW<[C1UWrite_4c_2L], (instrs LDPXi, LDNPXi)>;
def : InstRW<[C1UWrite_4c_2L_1I], (instrs LDPXpre, LDPXpost)>;

// Load pair, signed immed offset, signed words
def : InstRW<[C1UWrite_5c_2L_1I, WriteLDHi], (instrs LDPSWi)>;

// Load pair, immed post-index or immed pre-index, signed words
def : InstRW<[WriteAdr, C1UWrite_5c_2L_1I, WriteLDHi],
             (instregex "^LDPSW(post|pre)$")>;

// Store instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteST,    C1UWrite_1c_1SA_1D>;
def : SchedAlias<WriteSTIdx, C1UWrite_1c_1SA_1D_1I>;
def : SchedAlias<WriteSTP,   C1UWrite_1c_1SA_1D>;
def : SchedAlias<WriteAdr,   C1UWrite_1c_1I>;

// Tag load instructions
// -----------------------------------------------------------------------------

// Load allocation tag
// Load multiple allocation tags
def : InstRW<[C1UWrite_4c_1L], (instrs LDG, LDGM)>;

// Tag store instructions
// -----------------------------------------------------------------------------

// Store allocation tags to one or two granules, post-index
// Store allocation tags to one or two granules, pre-index
// Store allocation tag to one or two granules, zeroing, post-index
// Store Allocation Tag to one or two granules, zeroing, pre-index
// Store allocation tag and reg pair to memory, post-Index
// Store allocation tag and reg pair to memory, pre-Index
def : InstRW<[C1UWrite_1c_1SA_1D_1I], (instrs STGPreIndex, STGPostIndex,
                                                ST2GPreIndex, ST2GPostIndex,
                                                STZGPreIndex, STZGPostIndex,
                                                STZ2GPreIndex, STZ2GPostIndex,
                                                STGPpre, STGPpost)>;

// Store allocation tags to one or two granules, signed offset
// Store allocation tag to two granules, zeroing, signed offset
// Store allocation tag and reg pair to memory, signed offset
// Store multiple allocation tags
def : InstRW<[C1UWrite_1c_1SA_1D], (instrs STGi, ST2Gi, STZGi,
                                             STZ2Gi, STGPi, STGM, STZGM)>;

// FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF,     C1UWrite_2c_1V_CMEVXALU>;

// FP compare
def : SchedAlias<WriteFCmp,
                 C1UStreamingSchedWrite<C1UWrite_4c_CMEVX0ALU,
                                        C1UWrite_2c_1V01>>;

// FP divide
def : SchedAlias<WriteFDiv,  C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc,
                                                    C1UWrite_5c_1V1>>;

// FP divide, H-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc, C1UWrite_5c_1V1>],
             (instrs FDIVHrr)>;
// FP divide, S-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_11c_CMEMC_11rc, C1UWrite_8c_1V1>],
             (instrs FDIVSrr)>;
// FP divide, D-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_17c_CMEMC_17rc, C1UWrite_12c_1V1>],
             (instrs FDIVDrr)>;

// FP square root, H-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_10c_CMEMC_10rc, C1UWrite_5c_1V1>],
             (instrs FSQRTHr)>;
// FP square root, S-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_12c_CMEMC_12rc, C1UWrite_8c_1V1>],
             (instrs FSQRTSr)>;
// FP square root, D-form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_25c_CMEMC_25rc, C1UWrite_12c_1V1>],
             (instrs FSQRTDr)>;

// FP multiply
def : WriteRes<WriteFMul, [C1UUnitV]> { let Latency = 3; }

// FP multiply accumulate
def : InstRW<[C1UWr_FMA_CME, ReadDefault, ReadDefault, C1URd_FMA],
             (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP round to integral
def : InstRW<[C1UWrite_2c_1V0134_CME], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                             "^FRINT(32|64)[XZ][SD]r$")>;


// FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from gen to vec reg
def : InstRW<[C1UWrite_3c_1M0_CME], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
def : InstRW<[C1UWrite_3c_1V01_CMEVXALU],
             (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri?$")>;

// FP convert, Javascript from vec to gen reg
def : SchedAlias<WriteFCvt, C1UWrite_3c_1V01_CMEVXALU>;

// FP convert, from vec to vec reg
def : InstRW<[C1UWrite_3c_1V0134_CME], (instrs FCVTSHr, FCVTDHr, FCVTHSr, FCVTDSr,
                                          FCVTHDr, FCVTSDr, FCVTXNv1i64)>;

// FP move, immed
// FP move, register
def : SchedAlias<WriteFImm, C1UWrite_2c_1V_CMEVXALU>;

// FP transfer, from gen to low half of vec reg
def : InstRW<[C1UWrite_3c_1M0_CME],
             (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
def : InstRW<[C1UWrite_5c_1M0_1V_CME], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, C1UWrite_2c_1V01_CME>;

// FP load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal, S/D/Q forms
def : InstRW<[C1UWrite_7c_1L_1I_CME], (instregex "^LDR[SDQ]l$")>;

// Load vector reg, unscaled immed
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LDUR[BHSDQ]i$")>;

// Load vector reg, immed post-index
// Load vector reg, immed pre-index
def : InstRW<[WriteAdr, C1UWrite_6c_1L_1I_CME],
             (instregex "^LDR[BHSDQ](pre|post)$")>;

// Load vector reg, unsigned immed
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LDR[BHSDQ]ui$")>;

// Load vector reg, register offset, basic
// Load vector reg, register offset, scale, S/D-form
// Load vector reg, register offset, scale, H/Q-form
// Load vector reg, register offset, extend
// Load vector reg, register offset, extend, scale, S/D-form
// Load vector reg, register offset, extend, scale, H/Q-form
def : InstRW<[C1UWrite_LdrQ_CME, ReadAdrBase], (instregex "^LDR[BHSDQ]ro[WX]$")>;

// Load vector pair, immed offset, S/D-form
// Load vector pair, immed offset, Q-form
def : InstRW<[C1UWrite_6c_1L_CME, WriteLDHi], (instregex "^LDN?P[SDQ]i$")>;

// Load vector pair, immed post-index, S/D-form
// Load vector pair, immed pre-index, S/D-form
def : InstRW<[WriteAdr, C1UWrite_6c_1L_1I_CME, WriteLDHi],
             (instregex "^LDP[SD](pre|post)$")>;

// Load vector pair, immed post-index, Q-form
// Load vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, C1UWrite_6c_2L_1I_CME, WriteLDHi],
             (instregex "^LDPQ(pre|post)$")>;

// FP store instructions
// -----------------------------------------------------------------------------

// Store vector reg, unscaled immed, B/H/S/D-form
// Store vector reg, unscaled immed, Q-form
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index, B/H/S/D-form
// Store vector reg, immed post-index, Q-form
// Store vector reg, immed pre-index, B/H/S/D-form
// Store vector reg, immed pre-index, Q-form
def : InstRW<[WriteAdr, C1UWrite_2c_1SA_1V01_1I_CME],
             (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector reg, unsigned immed, B/H/S/D-form
// Store vector reg, unsigned immed, Q-form
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^STR[BHSDQ]ui$")>;

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
def : InstRW<[C1UWrite_StrQ_CME, ReadAdrBase],
             (instregex "^STR[BHSDQ]ro[WX]$")>;

// Store vector pair, immed offset, S-form
// Store vector pair, immed offset, D-form
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^STN?P[SD]i$")>;

// Store vector pair, immed offset, Q-form
def : InstRW<[C1UWrite_2c_2SA_2V01_CME], (instregex "^STN?PQi$")>;

// Store vector pair, immed post-index, S-form
// Store vector pair, immed post-index, D-form
// Store vector pair, immed pre-index, S-form
// Store vector pair, immed pre-index, D-form
def : InstRW<[WriteAdr, C1UWrite_2c_1SA_1V01_1I_CME],
             (instregex "^STP[SD](pre|post)$")>;

// Store vector pair, immed post-index, Q-form
// Store vector pair, immed pre-index, Q-form
def : InstRW<[C1UWrite_2c_2SA_2V01_CME], (instregex "^STPQ(post|pre)")>;

// ASIMD integer instructions
// -----------------------------------------------------------------------------

// ASIMD absolute diff
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD arith, pair-wise
// ASIMD compare
// ASIMD logical
def : SchedAlias<WriteVd, C1UWrite_2c_1V>;
def : SchedAlias<WriteVq, C1UWrite_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
def : InstRW<[C1UWr_VA_CME, C1URd_VA, C1URd_VA], (instregex "^[SU]ABAL?v")>;

// ASIMD arith, reduce, 4H/4S
def : InstRW<[C1UWrite_2c_1V0134_CME], (instregex "^(ADDV|[SU]ADDLV)v4(i16|i32)v$")>;

// ASIMD arith, reduce, 8B/8H
def : InstRW<[C1UWrite_4c_1V0134_CME],
             (instregex "^(ADDV|[SU]ADDLV)v8(i8|i16)v$")>;

// ASIMD arith, reduce, 16B
def : InstRW<[C1UWrite_4c_2V0134_CME], (instregex "^(ADDV|[SU]ADDLV)v16i8v$")>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def : InstRW<[C1UWr_VDOT_CME, C1URd_VDOT],
             (instregex "^([SU]|SU|US)DOT(lane)?(v8|v16)i8$")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[C1UWr_VMMA_CME, C1URd_VMMA], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD max/min, basic and pair-wise
def : InstRW<[C1UWrite_2c_1V_CME], (instregex "^[SU](MAX|MIN)P?(v16|v8)i8")>;

// ASIMD max/min, reduce, 4H/4S
def : InstRW<[C1UWrite_2c_1V0134_CME], (instregex "^[SU](MAX|MIN)Vv4i16v$",
                                           "^[SU](MAX|MIN)Vv4i32v$")>;

// ASIMD max/min, reduce, 8B/8H
def : InstRW<[C1UWrite_4c_1V0134_1V_CME], (instregex "^[SU](MAX|MIN)Vv8i8v$",
                                                "^[SU](MAX|MIN)Vv8i16v$")>;

// ASIMD max/min, reduce, 16B
def : InstRW<[C1UWrite_4c_2V0134_CME], (instregex "[SU](MAX|MIN)Vv16i8v")>;
// ASIMD multiply
def : InstRW<[C1UWrite_4c_1V0134_CME], (instregex "^MULv", "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate
// ASIMD multiply accumulate long
def : InstRW<[C1UWr_VMA_CME, C1URd_VMA, C1URd_VMA], (instregex "^MLAv", "^MLSv",
                                                "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate high
def : InstRW<[C1UWr_VMAH_CME, C1URd_VMAH, C1URd_VMAH], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply accumulate saturating long
def : InstRW<[C1UWrite_4c_1V0134_CME], (instregex "^SQDML[AS]L[iv]")>;

// ASIMD multiply/multiply long (8x8) polynomial, D-form
// ASIMD multiply/multiply long (8x8) polynomial, Q-form
def : InstRW<[C1UWrite_2c_1V0134_CME], (instregex "^PMULL?(v8i8|v16i8)$")>;

// ASIMD multiply long
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^[SU]MULLv", "^SQDMULL[iv]")>;

// ASIMD pairwise add and accumulate long
def : InstRW<[C1UWr_VPA_CME, C1URd_VPA, C1URd_VPA], (instregex "^[SU]ADALPv")>;

// ASIMD shift accumulate
def : InstRW<[C1UWr_VSA_CME, C1URd_VSA], (instregex "^[SU]SRA[dv]", "^[SU]RSRA[dv]")>;

// ASIMD shift by immed, basic
def : InstRW<[C1UWrite_2c_1V_CME], (instregex "^SHL[dv]", "^SHLLv", "^SHRNv",
                                         "^SSHLLv", "^SSHR[dv]", "^USHLLv",
                                         "^USHR[dv]")>;

// ASIMD shift by immed and insert, basic
def : InstRW<[C1UWrite_2c_1V_CME], (instregex "^SLI[dv]", "^SRI[dv]")>;

// ASIMD shift by immed, complex
// ASIMD shift by register, complex
def : InstRW<[C1UWrite_4c_1V_CME],
             (instregex "^RSHRNv", "^SQRSHRU?N[bhsv]", "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHRU?N[bhsv]", "^SRSHR[dv]", "^UQRSHRN[bhsv]",
                        "^UQSHRN[bhsv]", "^URSHR[dv]",
                        "^[SU]RSHLv", "^[SU]QRSHLv",
                        "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)$")>;

// ASIMD shift by register, basic
def : InstRW<[C1UWrite_2c_1V_CME], (instregex "^[SU]SHLv")>;

// ASIMD floating-point instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP max/min, normal
// ASIMD FP negate
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^FABSv", "^FADDv", "^FSUBv", "^FNEGv",
                        "^FMAXv", "^FMAXNMv", "^FMINv", "^FMINNMv")>;

// ASIMD FP complex add
// ASIMD FP max/min, pairwise
def : InstRW<[C1UWrite_3c_1V_CMEVXMUL], (instregex "^FCADDv")>;
def : InstRW<[C1UWrite_3c_1V_CMEVXALU],
             (instregex "^(FMAXP|FMAXNMP|FMINP|FMINNMP)v")>;

// ASIMD FP complex multiply add
def : InstRW<[C1UWr_VFCMA_CME, C1URd_VFCMA, C1URd_VFCMA], (instregex "^FCMLAv")>;

// ASIMD FP convert, long (F16 to F32)
// ASIMD FP convert, long (F32 to F64)
// ASIMD FP convert, narrow (F32 to F16)
// ASIMD FP convert, narrow (F64 to F32)
// ASIMD FP convert, other, D-form F32 and Q-form F64
// ASIMD FP convert, other, D-form F16 and Q-form F32
// ASIMD FP convert, other, Q-form F16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FCVTL(v4|v8)(i16|i32)",
                                             "^FCVTLv2i32",
                                              "^FCVTN(v4|v8)i16",
                                              "^FCVTN(v2|v4)i32",
                                              "^FCVTXN(v2|v4)f32",
                                              "^FCVT[AMNPZ][SU]v2f(32|64)$",
                                              "^FCVT[AMNPZ][SU]v2i(32|64)_shift$",
                                              "^FCVT[AMNPZ][SU]v1i64$",
                                              "^FCVTZ[SU]d$",
                                              "^[SU]CVTFv2f(32|64)$",
                                              "^[SU]CVTFv2i(32|64)_shift$",
                                              "^[SU]CVTFv1i64$",
                                              "^[SU]CVTFd$",
                                              "^FCVT[AMNPZ][SU]v4f(16|32)$",
                                              "^FCVT[AMNPZ][SU]v4i(16|32)_shift$",
                                              "^FCVT[AMNPZ][SU]v1i32$",
                                              "^FCVTZ[SU]s$",
                                              "^[SU]CVTFv4f(16|32)$",
                                              "^[SU]CVTFv4i(16|32)_shift$",
                                              "^[SU]CVTFv1i32$",
                                              "^[SU]CVTFs$",
                                              "^FCVT[AMNPZ][SU]v8f16$",
                                              "^FCVT[AMNPZ][SU]v8i16_shift$",
                                              "^FCVT[AMNPZ][SU]v1f16$",
                                              "^FCVTZ[SU]h$",
                                              "^[SU]CVTFv8f16$",
                                              "^[SU]CVTFv8i16_shift$",
                                              "^[SU]CVTFv1i16$",
                                              "^[SU]CVTFh$")>;

// ASIMD FP divide, D-form, F16
def : InstRW<[C1UWrite_8c_1V1_4rc_CME], (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
def : InstRW<[C1UWrite_9c_1V1_2rc_CME], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[C1UWrite_12c_1V1_8rc_CME], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[C1UWrite_11c_1V1_4rc_CME], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[C1UWrite_13c_1V1_2rc_CME], (instrs FDIVv2f64)>;

// ASIMD FP max/min, reduce, F32 and D-form F16
def : InstRW<[C1UWrite_4c_2V_CME], (instregex "^(FMAX|FMIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
def : InstRW<[C1UWrite_6c_3V_CME], (instregex "^(FMAX|FMIN)(NM)?Vv8i16v$")>;

// ASIMD FP multiply
def : InstRW<[C1UWrite_3c_1V_CMEVXMUL], (instregex "^FMULv", "^FMULXv")>;
def : InstRW<[C1UWrite_3c_1V_CMEVXMUL], (instrs FMULX16, FMULX32, FMULX64)>;

// ASIMD FP multiply accumulate
// ASIMD FP multiply accumulate long
def : InstRW<[C1UWr_VFMA_CME, C1URd_VFMA, C1URd_VFMA], (instregex "^FMLAv", "^FMLSv",
                                                  "^FML[AS]L2?(lane)?v")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[C1UWrite_3c_1V0134_CME],
             (instregex "^FRINT[AIMNPXZ]v2f(32|64)$",
                        "^FRINT(32|64)[XZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[C1UWrite_3c_1V0134_CME],
             (instregex "^FRINT[AIMNPXZ]v4f(16|32)$",
                        "^FRINT(32|64)[XZ]v4f32$")>;

// ASIMD FP round, Q-form F16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FRINT[AIMNPXZ]v8f16$")>;

// ASIMD FP square root, D-form, F16
def : InstRW<[C1UWrite_8c_1V1_4rc_CME], (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
def : InstRW<[C1UWrite_9c_1V1_2rc_CME], (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[C1UWrite_12c_1V1_8rc_CME], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
def : InstRW<[C1UWrite_11c_1V1_4rc_CME], (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[C1UWrite_13c_1V1_2rc_CME], (instrs FSQRTv2f64)>;

// ASIMD BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
// Scalar convert, F32 to BF16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instrs BFCVTN, BFCVTN2, BFCVT)>;

// ASIMD dot product
def : InstRW<[C1UWr_VBFDOT_CME, C1URd_VBFDOT, C1URd_VBFDOT],
             (instregex "^BFDOTv[48]bf16$",
                        "^BF16DOTlanev[48]bf16$")>;

// ASIMD matrix multiply accumulate
def : InstRW<[C1UWr_VBFMMA_CME, C1URd_VBFMMA, C1URd_VBFMMA], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
def : InstRW<[C1UWr_VBFMAL_CME, C1URd_VBFMAL, C1URd_VBFMAL], (instrs BFMLALB, BFMLALBIdx, BFMLALT,
                                                 BFMLALTIdx)>;

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
// ASIMD table lookup extension, 1 table reg
// ASIMD transpose
// ASIMD unzip/zip
// Handled by SchedAlias<WriteV[dq], ...>

// ASIMD duplicate, gen reg
def : InstRW<[C1UWrite_3c_1M0], (instregex "^DUPv.+gpr")>;

// ASIMD extract narrow, saturating
def : InstRW<[C1UWrite_4c_1V], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form U32
def : InstRW<[C1UWrite_4c_2V0134], (instrs URECPEv2i32, URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, Q-form U32
def : InstRW<[C1UWrite_4c_2V0134], (instrs URECPEv4i32, URSQRTEv4i32)>;

// ASIMD reciprocal and square root estimate, D-form F32 and scalar forms
def : InstRW<[C1UWrite_3c_1V0134], (instrs FRECPEv1f16, FRECPEv1i32,
                                        FRECPEv1i64, FRECPEv2f32, FRECPEv2f64,
                                        FRSQRTEv1f16, FRSQRTEv1i32,
                                        FRSQRTEv1i64, FRSQRTEv2f32, FRSQRTEv2f64)>;

// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
def : InstRW<[C1UWrite_4c_2V0134], (instrs FRECPEv4f16, FRECPEv4f32,
                                        FRSQRTEv4f16, FRSQRTEv4f32)>;

// ASIMD reciprocal and square root estimate, Q-form F16
def : InstRW<[C1UWrite_6c_4V0134], (instrs FRECPEv8f16, FRSQRTEv8f16)>;

// ASIMD reciprocal exponent
def : InstRW<[C1UWrite_3c_1V0134], (instregex "^FRECPXv")>;

// ASIMD reciprocal step
def : InstRW<[C1UWrite_4c_1V], (instregex "^FRECPS(16|32|64|v)",
                                         "^FRSQRTS(16|32|64|v)")>;

// ASIMD table lookup, 1 or 2 table regs
def : InstRW<[C1UWrite_2c_1V], (instrs TBLv8i8One, TBLv16i8One,
                                      TBLv8i8Two, TBLv16i8Two)>;

// ASIMD table lookup, 3 table regs
def : InstRW<[C1UWrite_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// ASIMD table lookup, 4 table regs
def : InstRW<[C1UWrite_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 2 table reg
def : InstRW<[C1UWrite_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup extension, 3 table reg
def : InstRW<[C1UWrite_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
def : InstRW<[C1UWrite_6c_5V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, element to gen reg
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_PERMS_1rc,
                                     C1UWrite_2c_3V>],
             (instregex "^SMOVv")>;
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_PERMS_1rc,
                                     C1UWrite_2c_3V>],
             (instregex "^UMOVvi(8|16)(_idx0)?$")>;
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_8c_CMEVXALU_PERMS_1rc,
                                     C1UWrite_2c_1V>],
             (instregex "^UMOVvi(32|64)$")>;

// ASIMD transfer, gen reg to element
def : InstRW<[C1UWrite_2c_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;

// ASIMD load instructions
// -----------------------------------------------------------------------------

// ASIMD load, 1 element, multiple, 1 reg, D-form
def : InstRW<[C1UWrite_6c_1L], (instregex "^LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1L],
             (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[C1UWrite_6c_1L], (instregex "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1L],
             (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
def : InstRW<[C1UWrite_6c_2L], (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_2L],
             (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[C1UWrite_6c_2L], (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_2L],
             (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
def : InstRW<[C1UWrite_6c_3L], (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_3L],
             (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[C1UWrite_6c_3L], (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_3L],
             (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
def : InstRW<[C1UWrite_6c_4L], (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_7c_4L],
             (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[C1UWrite_6c_4L], (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_7c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[C1UWrite_6c_1L_1V],           (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1L_1V], (instregex "LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
def : InstRW<[C1UWrite_6c_1L_1V],           (instregex "LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1L_1V], (instregex "LD1Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[C1UWrite_6c_1L_1V],           (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1L_1V], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_8c_1L_2V],           (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_1L_2V], (instregex "LD2Twov(8b|4h|2s)_POST$")>;

// ASIMD load, 2 element, multiple, Q-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[C1UWrite_8c_2L_2V],           (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_2V], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[C1UWrite_8c_1L_2V],           (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_1L_2V], (instregex "LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
def : InstRW<[C1UWrite_8c_1L_2V],            (instregex "LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_1L_2V],  (instregex "LD2Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[C1UWrite_8c_1L_2V],           (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_1L_2V], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_9c_2L_3V],           (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_9c_2L_3V], (instregex "LD3Threev(8b|4h|2s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[C1UWrite_9c_3L_3V],           (instregex "LD3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_9c_3L_3V], (instregex "LD3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, one lane, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[C1UWrite_9c_2L_3V],           (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_9c_2L_3V], (instregex "LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
def : InstRW<[C1UWrite_8c_2L_3V],           (instregex "LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_3V], (instregex "LD3Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[C1UWrite_8c_2L_3V],           (instregex "LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_3V], (instregex "LD3Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_8c_2L_4V],           (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_4V], (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[C1UWrite_9c_4L_8V],           (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_9c_4L_8V], (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[C1UWrite_8c_2L_4V],           (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_4V], (instregex "LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
def : InstRW<[C1UWrite_8c_2L_4V],           (instregex "LD4Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_4V], (instregex "LD4Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[C1UWrite_8c_2L_4V],           (instregex "LD4Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_8c_2L_4V], (instregex "LD4Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD store instructions
// -----------------------------------------------------------------------------

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[C1UWrite_2c_1SA_1V01],           (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_1SA_1V01], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[C1UWrite_2c_1SA_1V01],           (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_1SA_1V01], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[C1UWrite_2c_1SA_1V01],           (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_1SA_1V01], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[C1UWrite_2c_2SA_2V01],           (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_2SA_2V01], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
def : InstRW<[C1UWrite_2c_2SA_2V01],           (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_2SA_2V01], (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[C1UWrite_2c_3SA_3V01],           (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_3SA_3V01], (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[C1UWrite_2c_2SA_2V01],           (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_2SA_2V01], (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[C1UWrite_2c_4SA_4V01],           (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_2c_4SA_4V01], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[C1UWrite_4c_1SA_1V01_1V],           (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_1SA_1V01_1V], (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_4c_2SA_2V01_2V],           (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_2SA_2V01_2V], (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[C1UWrite_4c_2SA_2V01_2V],           (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_2SA_2V01_2V], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[C1UWrite_4c_1SA_1V01_1V],           (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_1SA_1V01_1V], (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_4c_2SA_2V01_2V],           (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_2SA_2V01_2V], (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[C1UWrite_5c_4SA_4V01_4V],           (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_5c_4SA_4V01_4V], (instregex "ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[C1UWrite_4c_2SA_2V01_2V],           (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1UWrite_4c_2SA_2V01_2V], (instregex "ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[C1UWrite_6c_6SA_6V01_6V],           (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_6SA_6V01_6V], (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[C1UWrite_7c_12SA_12V01_12V],           (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, C1UWrite_7c_12SA_12V01_12V], (instregex "ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[C1UWrite_6c_8SA_8V01_8V],           (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_8SA_8V01_8V], (instregex "ST4Fourv(2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H/S
def : InstRW<[C1UWrite_6c_1SA_1V01_1V],           (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1SA_1V01_1V], (instregex "ST4i(8|16|32)_POST$")>;

// ASIMD store, 4 element, one lane, D
def : InstRW<[C1UWrite_6c_1SA_1V01_1V],            (instregex "ST4i(64)$")>;
def : InstRW<[WriteAdr, C1UWrite_6c_1SA_1V01_1V],  (instregex "ST4i(64)_POST$")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[C1UWrite_2c_1V0134], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[C1UWrite_2c_1V0134], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 hash acceleration ops
// Crypto SHA1 schedule acceleration ops
def : InstRW<[C1UWrite_2c_1V0], (instregex "^SHA1(H|SU0|SU1|C|M|P)")>;

// Crypto SHA256 hash acceleration ops
def : InstRW<[C1UWrite_4c_1V0], (instregex "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[C1UWrite_2c_1V0], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[C1UWrite_2c_1V0], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instrs BCAX, EOR3, RAX1, XAR)>;

// Crypto SM3 ops
def : InstRW<[C1UWrite_2c_1V0], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                          "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[C1UWrite_4c_1V0], (instrs SM4E, SM4ENCKEY)>;

// CRC
// -----------------------------------------------------------------------------

def : InstRW<[C1UWrite_2c_1M], (instregex "^CRC32")>;

// SVE Predicate instructions
// -----------------------------------------------------------------------------

// Loop control, based on predicate
def : InstRW<[C1UWrite_1c_1M], (instrs BRKA_PPmP, BRKA_PPzP,
                                         BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[C1UWrite_1c_1M], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[C1UWr_Pred, C1URd_Pred, C1URd_Pred, C1URd_Pred], (instrs BRKN_PPzP, BRKPA_PPzPP,
                                         BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[C1UWr_Pred, C1URd_Pred, C1URd_Pred, C1URd_Pred], (instrs BRKNS_PPzP, BRKPAS_PPzPP,
                                         BRKPBS_PPzPP)>;

// Loop control, based on GPR
def : InstRW<[C1UWrite_1c_1M],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]")>;
def : InstRW<[C1UWrite_2c_1M], (instregex "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Loop terminate
def : InstRW<[C1UWrite_1c_2M], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// // Predicate counting scalar
def : InstRW<[C1UWrite_1c_1I], (instregex "^(CNT|DEC|INC)[BHWD]_XPiI")>;
def : InstRW<[C1UWrite_1c_1I], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;
def : InstRW<[C1UWrite_2c_1M],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI",
                        "^UQ(DEC|INC)[BHWD]_WPiI")>;


// Predicate counting scalar, active predicate
def : InstRW<[C1UWrite_2c_1M],
             (instregex "^CNTP_XPP_[BHSD]",
                        "^(DEC|SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;

// Predicate counting vector, active predicate
def : InstRW<[C1UWrite_7c_1M_1M0_1V],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;
def : InstRW<[C1UWrite_7c_1M_1M0_1V], (instregex "^INCP_XP_[BHSD]")>;

// Predicate logical
def : InstRW<[C1UWr_Pred, C1URd_Pred, C1URd_Pred, C1URd_Pred],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
def : InstRW<[C1UWr_Pred, C1URd_Pred, C1URd_Pred, C1URd_Pred],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
def : InstRW<[C1UWrite_2c_1M], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
def : InstRW<[C1UWr_Pred, C1URd_Pred, C1URd_Pred, C1URd_Pred], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[C1UWrite_2c_1M], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
def : InstRW<[C1UWrite_2c_1M], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
def : InstRW<[C1UWrite_2c_1M], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;

// Predicate test
def : InstRW<[C1UWrite_1c_1M], (instrs PTEST_PP, PTEST_PP_ANY, PTEST_PP_FIRST)>;

// Predicate transpose
def : InstRW<[C1UWrite_2c_1M], (instregex "^TRN[12]_PPP_[BHSD]")>;

// Predicate unpack and widen
def : InstRW<[C1UWrite_2c_1M], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[C1UWrite_2c_1M], (instregex "^(ZIP|UZP)[12]_PPP_[BHSD]")>;


// SVE integer instructions
// -----------------------------------------------------------------------------

// Arithmetic, absolute diff
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^[SU]ABD_ZPmZ_[BHSD]",
                                              "^[SU]ABD_ZPZZ_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[C1UWr_ZA_CME, C1URd_ZA, C1URd_ZA], (instregex "^[SU]ABA_ZZZ_[BHSD]")>;

// Arithmetic, absolute diff accum long
def : InstRW<[C1UWr_ZA_CME, C1URd_ZA, C1URd_ZA], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]")>;

// Arithmetic, absolute diff long
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]")>;

// Arithmetic, basic
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^(ABS|ADD|CNOT|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZI_[BHSD]",
                        "^ADR_[SU]XTW_ZZZ_D_[0123]",
                        "^ADR_LSL_ZZZ_[SD]_[0123]",
                        "^[SU](ADD|SUB)[LW][BT]_ZZZ_[HSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^[SU]H(ADD|SUB|SUBR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]")>;

// Arithmetic, complex
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^SQ(ABS|ADD|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;

// Arithmetic, large integer
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]")>;

// Arithmetic, pairwise add
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^ADDP_ZPmZ_[BHSD]")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[C1UWr_ZPA_CME, ReadDefault, C1URd_ZPA, C1URd_ZPA],
             (instregex "^[SU]ADALP_ZPmZ_[HSD]")>;

// Arithmetic, shift
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZ[IZ]_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[C1UWr_ZSA_CME, C1URd_ZSA], (instregex "^[SU]R?SRA_ZZI_[BHSD]")>;

// Arithmetic, shift by immediate
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^SHRN[BT]_ZZI_[BHS]",
                                              "^[SU]SHLL[BT]_ZZI_[HSD]")>;

// Arithmetic, shift by immediate and insert
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^(SLI|SRI)_ZZI_[BHSD]")>;

// Arithmetic, shift complex
def : InstRW<[C1UWrite_4c_1V_CME],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_ZPmZ_[BHSD]",
                        "^[SU]QR?SHL_ZPZZ_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift right for divide
def : InstRW<[C1UWrite_4c_1V_CME], (instregex "^ASRD_(ZPmI|ZPZI)_[BHSD]")>;

// Arithmetic, shift rounding
def : InstRW<[C1UWrite_4c_1V_CME], (instregex "^[SU]RSHLR?_ZPmZ_[BHSD]",
                                           "^[SU]RSHL_ZPZZ_[BHSD]",
                                           "^[SU]RSHR_(ZPmI|ZPZI)_[BHSD]")>;

// Bit manipulation
def : InstRW<[C1UWrite_6c_2V1], (instregex "^(BDEP|BEXT|BGRP)_ZZZ_[BHSD]")>;

// Bitwise select
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ")>;

// Count/reverse bits
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^(CLS|CLZ|CNT|RBIT)_ZPmZ_[BHSD]")>;

// Broadcast logical bitmask immediate to vector
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[C1UWrite_2c_1V0_CME],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;

// Complex add
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^(SQ)?CADD_ZZI_[BHSD]")>;

// Complex dot product 8-bit element
def : InstRW<[C1UWr_ZDOTB_CME, C1URd_ZDOTB, C1URd_ZDOTB], (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[C1UWr_ZDOTH_CME, C1URd_ZDOTH, C1URd_ZDOTH], (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S, D element size
def : InstRW<[C1UWr_ZCMABHS_CME, C1URd_ZCMABHS, C1URd_ZCMABHS], (instregex "^CMLA_ZZZ_[BHSD]",
                                                      "^CMLA_ZZZI_[HS]")>;

// Conditional extract operations, scalar form
def : InstRW<[C1UWrite_8c_1M0_1V1_1V01_CMECLAST],
             (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[C1UWrite_3c_1V1_CME], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]",
                                          "^COMPACT_ZPZ_[SD]",
                                          "^SPLICE_ZPZZ?_[BHSD]")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                             "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 32b to single or half
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 16b to half
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[C1UWrite_3c_1M0_1V_CMEPERMF_PERMS], (instregex "^CPY_ZPmR_[BHSD]")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[C1UWrite_2c_1V_CMEPERMS_1c], (instregex "^CPY_ZPm[IV]_[BHSD]",
                                                 "^CPY_ZPzI_[BHSD]")>;

// Divides, 32 bit
def : InstRW<[C1UWrite_12c_7V0_CME], (instregex "^[SU]DIVR?_ZPmZ_S",
                                           "^[SU]DIV_ZPZZ_S")>;

// Divides, 64 bit
def : InstRW<[C1UWrite_20c_7V0_CME], (instregex "^[SU]DIVR?_ZPmZ_D",
                                           "^[SU]DIV_ZPZZ_D")>;

// Dot product, 8 bit
def : InstRW<[C1UWr_ZDOTB_CME, C1URd_ZDOTB, C1URd_ZDOTB], (instregex "^[SU]DOT_ZZZI?_BtoS")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[C1UWr_ZDOTB_CME, C1URd_ZDOTB, C1URd_ZDOTB], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[C1UWr_ZDOTH_CME, C1URd_ZDOTH, C1URd_ZDOTB], (instregex "^[SU]DOT_ZZZI?_HtoD")>;

// Duplicate, immediate and indexed form
def : InstRW<[C1UWrite_2c_1V_StreamingCMEPERMS], (instregex "^DUP_ZI_[BHSD]",
                                                       "^DUP_ZZI_[BHSDQ]")>;

// Duplicate, scalar form
def : InstRW<[C1UWrite_3c_1M0_CMEPERMF], (instregex "^DUP_ZR_[BHSD]")>;

// Extend, sign or zero
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                              "^[SU]XTH_ZPmZ_[SD]",
                                              "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instrs EXT_ZZI,
                                                EXT_ZZI_CONSTRUCTIVE,
                                                EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[C1UWrite_4c_1V_CMEVXALU], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]",
                                              "^SQXTUN[BT]_ZZ_[BHS]")>;

// Extract operation, SIMD and FP scalar form
def : InstRW<[C1UWrite_3c_1V1_CME], (instregex "^LAST[AB]_VPZ_[BHSD]")>;

// Extract operation, scalar
def : InstRW<[C1UWrite_5c_1V1_1V01_CME], (instregex "^LAST[AB]_RPZ_[BHSD]")>;

// Histogram operations
def : InstRW<[C1UWrite_2c_1V], (instregex "^HISTCNT_ZPzZZ_[SD]",
                                     "^HISTSEG_ZZZ")>;

// Horizontal operations,immediate operands only
def : InstRW<[C1UWrite_4c_1V0134], (instregex "^INDEX_(II)_[BHSD]")>;

// Horizontal operations, scalar, immediate operands / immediate, scalar operands
def : InstRW<[C1UWrite_7c_1M0_1V0134_CMEINDEX_IRRI],
             (instregex "^INDEX_(IR|RI)_[BHSD]")>;

// Horizontal operations, scalar operands only
def : InstRW<[C1UWrite_7c_1M0_1V0134_CMEINDEX_RR], (instregex "^INDEX_RR_[BHSD]")>;

// insert operation, SIMD and FP scalar form
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instregex "^INSR_ZV_[BHSD]")>;

// insert operation, scalar
def : InstRW<[C1UWrite_5c_1V_1M0_CME], (instregex "^INSR_ZR_[BHSD]")>;

// Logical
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|ORR|EON|NAND|NOR)_ZZZ",
                        "^EOR(BT|TB)_ZZZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^NOT_ZPmZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                              "^[SU](MAX|MIN)P?_ZPmZ_[BHSD]",
                                              "^[SU](MAX|MIN)_ZPZZ_[BHSD]")>;

// Matching operations
def : InstRW<[C1UWrite_2c_1V0], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Matrix multiply-accumulate
def : InstRW<[C1UWr_ZMMA, C1URd_ZMMA, C1URd_ZMMA], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]",
                                              "^MOVPRFX_ZZ")>;

// Multiply
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_[BHSD]",
                        "^MUL_ZPZZ_[BHSD]",
                        "^[SU]MULH_(ZPmZ|ZZZ)_[BHSD]",
                        "^[SU]MULH_ZPZZ_[BHSD]")>;

// Multiply long
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^[SU]MULL[BT]_ZZZI_[SD]",
                        "^[SU]MULL[BT]_ZZZ_[HSD]")>;


// Multiply accumulate, B, H, S element size
def : InstRW<[C1UWr_ZMA_CME, ReadDefault, C1URd_ZMA, C1URd_ZMA],
             (instregex "^ML[AS]_ZZZI_[HS]", "^ML[AS]_ZPZZZ_[BHS]")>;
def : InstRW<[C1UWr_ZMA_CME, ReadDefault, C1URd_ZMA, C1URd_ZMA],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]")>;

// Multiply accumulate, D element size
def : InstRW<[C1UWr_ZMA_CME, ReadDefault, C1URd_ZMA, C1URd_ZMA],
             (instregex "^ML[AS]_ZZZI_D", "^ML[AS]_ZPZZZ_D")>;
def : InstRW<[C1UWr_ZMA_CME, ReadDefault, C1URd_ZMA, C1URd_ZMA],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_D")>;



// Multiply accumulate long
def : InstRW<[C1UWr_ZMA_CME, C1URd_ZMA, C1URd_ZMA], (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]",
                                                "^[SU]ML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[C1UWr_ZMASQL_CME, C1URd_ZMASQL, C1URd_ZMASQL],
             (instregex "^SQDML[AS]L(B|T|BT)_ZZZ_[HSD]",
                        "^SQDML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply saturating doubling high
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^SQDMULH_ZZZ_[BHSD]",
                        "^SQDMULH_ZZZI_[HSD]")>;

// Multiply saturating doubling long
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^SQDMULL[BT]_ZZZ_[HSD]",
                        "^SQDMULL[BT]_ZZZI_[SD]")>;

// Multiply saturating rounding doubling regular/complex accumulate
def : InstRW<[C1UWr_ZMASQL_CME, C1URd_ZMASQL, C1URd_ZMASQL], (instregex "^SQRDML[AS]H_ZZZ_[BHSD]",
                                                     "^SQRDCMLAH_ZZZ_[BHSD]",
                                                     "^SQRDML[AS]H_ZZZI_[HSD]",
                                                     "^SQRDCMLAH_ZZZI_[HSD]")>;


// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^SQRDMULH_ZZZ_[BHS]",
                        "^SQRDMULH_ZZZI_[HS]",
                        "^SQRDMULH_ZZZI?_D")>;


// Multiply/multiply long, (8x8) polynomial
def : InstRW<[C1UWrite_2c_1V0134], (instregex "^PMUL_ZZZ_B",
                                        "^PMULL[BT]_ZZZ_[HDQ]")>;

// Predicate counting vector
def : InstRW<[C1UWrite_2c_1V_CME], (instregex "^([SU]Q)?(DEC|INC)[HWD]_ZPiI")>;

// Reciprocal estimate
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[C1UWrite_8c_1V_3V0134_CMEIntReduce],
             (instregex "^[SU]ADDV_VPZ_B")>;
def : InstRW<[C1UWrite_6c_1V_2V0134_CMEIntReduce],
             (instregex "^[SU](MAX|MIN)V_VPZ_B")>;

// // Reduction, arithmetic, H form
def : InstRW<[C1UWrite_6c_1V_2V0134_CMEIntReduce],
             (instregex "^[SU]ADDV_VPZ_H")>;
def : InstRW<[C1UWrite_6c_2V_1V0134_CMEIntReduce],
             (instregex "^[SU](MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[C1UWrite_6c_2V_1V0134_CMEIntReduce],
             (instregex "^[SU]ADDV_VPZ_S")>;
def : InstRW<[C1UWrite_4c_1V_1V0134_CMEIntReduce],
             (instregex "^[SU](MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[C1UWrite_4c_2V_CMEIntReduce],
             (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[C1UWrite_4c_1V_1V0134_CMEIntReduce],
             (instregex "^(AND|EOR|OR)V_VPZ_[BHSD]")>;

// Reverse, vector
def : InstRW<[C1UWrite_2c_1V_StreamingCMEPERMS], (instregex "^REV_ZZ_[BHSD]",
                                                       "^REVB_ZPmZ_[HSD]",
                                                       "^REVH_ZPmZ_[SD]",
                                                       "^REVW_ZPmZ_D")>;

// Select, vector form
def : InstRW<[C1UStreamingSchedWrite<C1UWrite_1c_CMEPERMS_2rc,
                                     C1UWrite_2c_1V>],
             (instregex "^SEL_ZPZZ_[BHSD]")>;

// Table lookup
def : InstRW<[C1UWrite_2c_1V_CMEPERMF_1rc], (instregex "^TBL_ZZZZ?_[BHSD]")>;

// Table lookup extension
def : InstRW<[C1UWrite_2c_1V_CMEPERMF_1rc], (instregex "^TBX_ZZZ_[BHSD]")>;

// Transpose, vector form
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instregex "^TRN[12]_ZZZ_[BHSDQ]")>;

// Unpack and extend
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]")>;

// Zip/unzip
def : InstRW<[C1UWrite_2c_1V_CMEPERMS], (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]")>;

// SVE floating-point instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^FAB[SD]_ZPmZ_[HSD]",
                                              "^FABD_ZPZZ_[HSD]",
                                              "^FABS_ZPmZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[C1UWrite_2c_1V_CMEVXALU], (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ)_[HSD]",
                                              "^F(ADD|SUB)_ZPZ[IZ]_[HSD]",
                                              "^FADDP_ZPmZZ_[HSD]",
                                              "^FNEG_ZPmZ_[HSD]",
                                              "^FSUBR_ZPm[IZ]_[HSD]",
                                              "^FSUBR_(ZPZI|ZPZZ)_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[C1UWrite_10c_10V0], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[C1UWrite_6c_6V0], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[C1UWrite_4c_2V], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[C1UWrite_2c_1V0_CME], (instregex "^FACG[ET]_PPzZZ_[HSD]",
                                          "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]",
                                          "^FCM(LE|LT)_PPzZ0_[HSD]",
                                          "^FCMUO_PPzZZ_[HSD]")>;

// Floating point complex add
def : InstRW<[C1UWrite_3c_1V_CMEVXMUL], (instregex "^FCADD_ZPmZ_[HSD]")>;

// Floating point complex multiply add
def : InstRW<[C1UWr_ZFCMA_CME, ReadDefault, C1URd_ZFCMA, C1URd_ZFCMA], (instregex "^FCMLA_ZPmZZ_[HSD]")>;
def : InstRW<[C1UWr_ZFCMA_CME, C1URd_ZFCMA, C1URd_ZFCMA],              (instregex "^FCMLA_ZZZI_[HS]")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[C1UWrite_4c_1V0134_CME], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                           "^FCVTLT_ZPmZ_HtoS",
                                           "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32
// or F64 to F16)
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                           "^FCVTLT_ZPmZ_StoD",
                                           "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[C1UWrite_3c_1V0134_CME], (instrs FCVTX_ZPmZ_DtoS, FCVTXNT_ZPmZ_DtoS)>;

// Floating point base2 log, F16
def : InstRW<[C1UWrite_3c_1V0134_CMEVXMUL], (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[C1UWrite_3c_1V0134_CMEVXMUL], (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[C1UWrite_3c_1V0134_CMEVXMUL], (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[C1UWrite_3c_1V0134_CME],
             (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[C1UWrite_2c_1V_CMEPERMS_1c], (instregex "^FCPY_ZPmI_[HSD]",
                                                 "^FDUP_ZI_[HSD]")>;

// Floating point divide, F16
def : InstRW<[C1UWrite_12c_1V0_8rc_CME], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[C1UWrite_11c_1V0_4rc_CME], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[C1UWrite_13c_1V0_2rc_CME], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[C1UWrite_2c_1V_CMEVXALU],
             (instregex "^F(MAX|MIN)(NM)?_ZPm[IZ]_[HSD]",
                        "^F(MAX|MIN)(NM)?_ZPZ[IZ]_[HSD]")>;

// Floating point multiply
def : InstRW<[C1UWrite_3c_1V_CMEVXMUL],
             (instregex "^(FSCALE|FMULX)_ZPmZ_[HSD]",
                        "^FMULX_ZPZZ_[HSD]",
                        "^FMUL_(ZPm[IZ]|ZZZI?)_[HSD]",
                        "^FMUL_ZPZ[IZ]_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[C1UWr_ZFMA_CME, ReadDefault, C1URd_ZFMA, C1URd_ZFMA],
             (instregex "^FN?ML[AS]_ZPmZZ_[HSD]",
                        "^FN?(MAD|MSB)_ZPmZZ_[HSD]",
                        "^FN?ML[AS]_ZPZZZ_[HSD]")>;

def : InstRW<[C1UWr_ZFMA_CME, C1URd_ZFMA, C1URd_ZFMA],
             (instregex "^FML[AS]_ZZZI_[HSD]")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[C1UWr_ZFMAL_CME, C1URd_ZFMAL, C1URd_ZFMAL], (instregex "^FML[AS]L[BT]_ZZZI?_SHH")>;

// Floating point reciprocal estimate, F16
def : InstRW<[C1UWrite_6c_1V0134_CMEVXMUL],
             (instregex "^FR(ECP|SQRT)E_ZZ_H", "^FRECPX_ZPmZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[C1UWrite_4c_1V0134_CMEVXMUL],
             (instregex "^FR(ECP|SQRT)E_ZZ_S", "^FRECPX_ZPmZ_S")>;

// Floating point reciprocal estimate, F64
def : InstRW<[C1UWrite_3c_1V0134_CMEVXMUL],
             (instregex "^FR(ECP|SQRT)E_ZZ_D", "^FRECPX_ZPmZ_D")>;

// Floating point reciprocal step
def : InstRW<[C1UWrite_4c_1V_CMEVXMUL], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]")>;

// Floating point reduction, F16
def : InstRW<[C1UWrite_8c_4V_CMEFPReduceH], (instrs FADDV_VPZ_H)>;
def : InstRW<[C1UWrite_8c_4V_CMEFPReduceMaxMin],
             (instregex "^(FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_H")>;

// Floating point reduction, F32
def : InstRW<[C1UWrite_6c_3V_CMEFPReduceS], (instrs FADDV_VPZ_S)>;
def : InstRW<[C1UWrite_6c_3V_CMEFPReduceMaxMin],
             (instregex "^(FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_S")>;

// Floating point reduction, F64
def : InstRW<[C1UWrite_4c_2V_CMEFPReduceD], (instrs FADDV_VPZ_D)>;
def : InstRW<[C1UWrite_4c_2V_CMEFPReduceMaxMinD],
             (instregex "^(FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_D")>;

// Floating point round to integral, F16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[C1UWrite_3c_1V0134_CME], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[C1UWrite_12c_1V1_8rc_CME], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[C1UWrite_11c_1V1_4rc_CME], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64
def : InstRW<[C1UWrite_13c_1V1_2rc_CME], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[C1UWrite_3c_1V1], (instregex "^FEXPA_ZZ_[HSD]")>;

// Floating point trigonometric multiply add
def : InstRW<[C1UWrite_4c_1V], (instregex "^FTMAD_ZZI_[HSD]")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[C1UWrite_3c_1V], (instregex "^FTS(MUL|SEL)_ZZZ_[HSD]")>;


// SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[C1UWrite_3c_1V0134_CME], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[C1UWr_ZBFDOT_CME, C1URd_ZBFDOT, C1URd_ZBFDOT], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[C1UWr_ZBFMMA, C1URd_ZBFMMA, C1URd_ZBFMMA], (instrs BFMMLA_ZZZ_HtoS)>;

// Multiply accumulate long
def : InstRW<[C1UWr_ZBFMAL_CME, C1URd_ZBFMAL, C1URd_ZBFMAL], (instregex "^BFMLAL[BT]_ZZZI?")>;

// SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[C1UWrite_6c_1L_CME], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[C1UWrite_6c_1L_1M_CME], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LD1[BHWD]_IMM$",
                                         "^LD1S?B_[HSD]_IMM$",
                                         "^LD1S?H_[SD]_IMM$",
                                         "^LD1S?W_D_IMM$" )>;
// Contiguous load, scalar + scalar
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LD1[BHWD]$",
                                         "^LD1S?B_[HSD]$",
                                         "^LD1S?H_[SD]$",
                                         "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LD1R[BHWD]_IMM$",
                                         "^LD1RS?B_[HSD]_IMM$",
                                         "^LD1RS?H_[SD]_IMM$",
                                         "^LD1RW_D_IMM$",
                                         "^LD1RSW_IMM$",
                                         "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
// Non temporal load, scalar + scalar
def : InstRW<[C1UWrite_6c_1L_CME], (instregex "^LDNT1[BHWD]_ZR[IR]$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^LDNT1[BHW]_ZZR_S$",
                                         "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[C1UWrite_6c_1L], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^LDFF1[BHWD]$",
                                          "^LDFF1S?B_[HSD]$",
                                          "^LDFF1S?H_[SD]$",
                                          "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[C1UWrite_6c_1L], (instregex "^LDNF1[BHWD]_IMM$",
                                      "^LDNF1S?B_[HSD]_IMM$",
                                      "^LDNF1S?H_[SD]_IMM$",
                                      "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[C1UWrite_8c_2L_2V_CME], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[C1UWrite_9c_2L_2V_1I_CME], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[C1UWrite_8c_3L_3V_CME], (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[C1UWrite_9c_3L_3V_1I_CME], (instregex "^LD3[BHWD]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[C1UWrite_10c_4L_8V_CME], (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[C1UWrite_11c_4L_8V_1I_CME], (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                         "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                         "^GLD(FF)?1D_IMM$")>;

// Gather load, 32-bit scaled offset
def : InstRW<[C1UWrite_9c_1V01_1L],
             (instregex "^GLD(FF)?1S?H_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;
// Gather load, 64-bit scaled offset
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[C1UWrite_9c_1V01_1L],
             (instregex "^GLD(FF)?1S?[HW]_D_([SU]XTW_)?SCALED$",
                        "^GLD(FF)?1D_([SU]XTW_)?SCALED$")>;


// Gather load, 32-bit unpacked unscaled offset
def : InstRW<[C1UWrite_9c_1V01_1L], (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                                         "^GLD(FF)?1W_[SU]XTW$")>;

// Gather load, 64-bit unpacked unscaled offset
// NOTE: These instructions are not specified in the SOG.
def : InstRW<[C1UWrite_9c_1V01_1L],
             (instregex "^GLD(FF)?1S?[BHW]_D(_[SU]XTW)?$",
                        "^GLD(FF)?1D(_[SU]XTW)?$")>;
// SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[C1UWrite_1c_1SA_CME], (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^ST1[BHWD]_IMM$",
                                                "^ST1B_[HSD]_IMM$",
                                                "^ST1H_[SD]_IMM$",
                                                "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^ST1H(_[SD])?$",
                                                  "^ST1[BWD]$",
                                                  "^ST1B_[HSD]$",
                                                  "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[C1UWrite_4c_2SA_2V01_CME], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[C1UWrite_4c_2SA_2V01_1I_CME], (instregex "^ST2[BHWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[C1UWrite_5c_3SA_3V01_CME], (instregex "^ST3[BHWD]_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[C1UWrite_5c_3SA_3V01_1I_CME], (instregex "^ST3[BHWD]$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[C1UWrite_7c_4SA_4V01_CME], (instregex "^ST4[BHWD]_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[C1UWrite_7c_4SA_4V01_1I_CME], (instregex "^ST4[BHWD]$")>;

// Non temporal store, scalar + imm
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[C1UWrite_2c_1SA_1V01_CME], (instregex "^STNT1[BHWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[C1UWrite_5or7c_3SA_6V01], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[C1UWrite_4or6c_3SA_3V01], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[C1UWrite_5or7c_5SA_5V01], (instregex "^SST1[BHW]_S_IMM$",
                                               "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[C1UWrite_4or6c_3SA_3V01], (instregex "^SST1[BHW]_D_IMM$",
                                                "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[C1UWrite_2or4c_3SA_6V01],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[C1UWrite_2or4c_1SA_1V01], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                             "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[C1UWrite_2or4c_1SA_1V01], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                             "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[C1UWrite_2or4c_1SA_1V01], (instregex "^SST1[BH]_S_[SU]XTW$",
                                             "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[C1UWrite_2or4c_1SA_1V01], (instregex "^SST1[HW]_D_SCALED$",
                                             "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[C1UWrite_2or4c_1SA_1V01], (instregex "^SST1[BHW]_D$",
                                             "^SST1D$")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[C1UWrite_2c_1M0], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[C1UWrite_2c_1M0], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[C1UWrite_3c_1M0], (instrs RDFFRS_PPz)>;

// Set first fault register
// Write to first fault register
def : InstRW<[C1UWrite_2c_1M0], (instrs SETFFR, WRFFR)>;

// Prefetch
// NOTE: This is not specified in the SOG.
def : InstRW<[C1UWrite_4c_1L], (instregex "^PRF[BHWD]")>;

// SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[C1UWrite_2c_1V0134], (instregex "^AES[DE]_ZZZ_B$",
                                     "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[C1UWrite_2c_1V], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                 "^RAX1_ZZZ_D$",
                                 "^XAR_ZZZI_[BHSD]$")>;

// Crypto SM4 ops
def : InstRW<[C1UWrite_4c_1V0], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

// SVE instructions added by SME and available when not in Streaming SVE mode
// -----------------------------------------------------------------------------
// BFloat16 floating-point multiply-subtract long from single-precision vector and indexed form
def : InstRW<[C1UWrite_4c_1V_StreamingCMEMML0], (instregex "^BFMLSLB_",
                                                           "^BFMLSLT_")>;

// Floating-point clamp to minimum/maximu m number
def : InstRW<[C1UWrite_2c_1V_StreamingCMEVXALU_5c], (instregex "^FCLAMP_ZZZ")>;

// Half-precision floating-point indexed or vector forms dot product
def : InstRW<[C1UWrite_4c_1V_StreamingCMEMML0], (instregex "^FDOT_ZZZI?_S$")>;

// Predicate selection
def : InstRW<[C1UWrite_2c_2M_StreamingCMEPERMS], (instregex "^PSEL_PPPRI")>;

// Reverse doubleword elements
def : InstRW<[C1UWrite_2c_1V_StreamingCMEPERMS], (instregex "^REVD_ZPmZ")>;

// Signed/unsigned clamp
def : InstRW<[C1UWrite_2c_1V_StreamingCME], (instregex "^(S|U)CLAMP_ZZZ")>;

// Half-precision signed/unsigned dot product vector and indexed forms
def : InstRW<[C1UWrite_3c_1V01_StreamingCME], (instregex "^(S|U)DOT_ZZZI?_HtoS")>;

// Signed/unsigned saturating narrow conversion
def : InstRW<[C1UWrite_2c_1V_StreamingCMEVXALU_PERMS],
             (instregex "^(SQCVTN|SQCVTUN|UQCVTN)_Z2Z_StoH")>;

// Signed/unsigned saturating rounding shift-right narrow
def : InstRW<[C1UWrite_3c_1V01_StreamingCME], (instregex "^(SQRSHRN|SQRSHRUN|UQRSHRN)_Z2ZI_StoH")>;

// Signed/unsigned pair-producing while compare
def : InstRW<[C1UWrite_2c_1M_StreamingCMEVX0ALU],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_2PXX")>;

// SVE instructions added by SME but not sent to CME when in Streaming SVE mode
// -----------------------------------------------------------------------------

// Add/Read multiple of streaming SVE Predicate/vector register size to scalar register
def : InstRW<[C1UWrite_1c_1I_Streaming], (instrs ADDSPL_XXI, ADDSVL_XXI, RDSVLI_XI)>;

// Predicate counting scalar to count from predicate-as-counter
def : InstRW<[C1UWrite_2c_1M_Streaming], (instregex "^CNTP_XCI_[BHSD]$")>;

// Set predicates from predicate-as-counter
def : InstRW<[C1UWrite_2c_1M_Streaming], (instregex "^PEXT_PCI_[BHSD]$")>;

// Set pair of predicates from predicate-as-counter
def : InstRW<[C1UWrite_3c_1M_Streaming], (instregex "^PEXT_2PCI_[BHSD]$")>;

// Predicate as counter set
def : InstRW<[C1UWrite_1c_1M_Streaming], (instregex "^PTRUE_C_[BHSD]$")>;

// Loop control, based on GPR generating predicate as counter
def : InstRW<[C1UWrite_1c_1M_Streaming], (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_CXX_[BHSD]$")>;

// SVE instructions added by SME and only available in Streaming SVE mode
// -----------------------------------------------------------------------------

// Horizontal and vertical ZA reductions.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_4rc>],
             (instregex "^ADDHA_MPPZ_[SD]$", "^ADDVA_MPPZ_[SD]$")>;

// Outer-product ZA updates.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_4rc>],
             (instregex "^BMOPA_MPPZZ_S$", "^BMOPS_MPPZZ_S$",
                        "^(SMOPA|UMOPA|SMOPS|UMOPS)_MPPZZ_(S|D|HtoS)$",
                        "^(SUMOPA|USMOPA|SUMOPS|USMOPS)_MPPZZ_(S|D)$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_4rc>],
             (instregex "^(FMOPA|FMOPS)_MPPZZ_(S|D|H)$",
                        "^FMOP(AL|SL)_MPPZZ$",
                        "^(BFMOPA|BFMOPS)_MPPZZ(_H)?$")>;

// ZA accumulators updated through multi-vector add/subtract operations.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_2rc>],
             (instregex "^(ADD|SUB)_VG2_M2(Z|ZZ|Z2Z)_[SD]$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_4rc>],
             (instregex "^(ADD|SUB)_VG4_M4(Z|ZZ|Z4Z)_[SD]$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_2rc>],
             (instregex "^(FADD|FSUB|BFADD|BFSUB)_VG2_M2Z_[HSD]$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_4rc>],
             (instregex "^(FADD|FSUB|BFADD|BFSUB)_VG4_M4Z_[HSD]$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEVXALU_1rc>],
             (instregex "^ADD_VG2_2ZZ_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_6c_CMEVXALU_4rc>],
             (instregex "^ADD_VG4_4ZZ_")>;

// Streaming-only clamp, convert and shift families.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEVXALU_1rc>],
             (instregex "^FCLAMP_VG2_2Z2Z_", "^BFCLAMP_VG2_2ZZZ",
                        "^(S|U)CLAMP_VG2_2Z2Z_", "^(FMAX|FMIN)_VG2_2(ZZ|Z2Z)_",
                        "^FA(MAX|MIN)_2Z2Z_", "^SQDMULH_VG2_2ZZ_",
                        "^(S|U)RSHL_VG2_2ZZ_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_7c_CMEVXALU_4rc>],
             (instregex "^FCLAMP_VG4_4Z4Z_", "^BFCLAMP_VG4_4ZZZ",
                        "^(S|U)CLAMP_VG4_4Z4Z_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_2rc>],
             (instregex "^(S|U)(MAX|MIN)_VG2_2(ZZ|Z2Z)_",
                        "^(FMAXNM|FMINNM)_VG2_2(ZZ|Z2Z)_",
                        "^SQDMULH_VG2_2Z2Z_", "^(S|U)RSHL_VG2_2Z2Z_",
                        "^(SQRSHR|UQRSHR)_VG2_Z2ZI_",
                        "^(FRINT(A|M|N|P))_2Z2Z_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_4rc>],
             (instregex "^(S|U)(MAX|MIN)_VG4_4Z4Z_",
                        "^(S|U)RSHL_VG4_4Z4Z_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEVXALU_4rc>],
             (instregex "^(FMAX|FMIN|FMAXNM|FMINNM)_VG4_4(ZZ|Z4Z)_",
                        "^(BFMAX|BFMIN|BFMAXNM|BFMINNM)_VG4_4(Z2Z|ZZ)_",
                        "^SQDMULH_VG4_4(ZZ|Z4Z)_", "^(FRINT(A|M|N|P))_4Z4Z_",
                        "^FSCALE_4(ZZ|Z4Z)_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_2rc>],
             (instregex "^(BFMAX|BFMIN|BFMAXNM|BFMINNM)_VG2_2(ZZ|Z2Z)_",
                        "^FSCALE_2(ZZ|Z2Z)_", "^FCVTZS_2Z2Z_StoS$",
                        "^FCVTZU_2Z2Z_StoS$", "^F[12]CVT_(2)?ZZ_BtoH$",
                        "^F[12]CVTL(T_ZZ|_2ZZ)_BtoH$",
                        "^FCVT_Z2Z_StoH$", "^FCVT_2ZZ_H_S$",
                        "^FCVTL_2ZZ_H_S$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_6c_CMEVXALU_4rc>],
             (instregex "^(S|U)RSHL_VG4_4ZZ_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVX0ALU_PERMF_1rc>],
             (instregex "^(S|U)CVTF_2Z2Z_StoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVX0ALU_PERMF_2rc>],
             (instregex "^(S|U)CVTF_4Z4Z_StoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_4rc>],
             (instregex "^FCVTZS_4Z4Z_StoS$", "^FCVTZU_4Z4Z_StoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEVXALU_PERMF_1d1rc>],
             (instregex "^FCVTN_Z2Z_StoH$", "^FCVTN_Z4Z_StoB$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_7c_CMEVXALU_PERMF_1rc>],
             (instregex "^BFCVTN_Z2Z_StoH$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_11c_CMEVXALU_2rc>],
             (instregex "^BFCVT_Z2Z_(StoH|HtoB)$", "^BF[12]CVT_(2)?ZZ_BtoH$",
                        "^BF[12]CVTL(T_ZZ|_2ZZ)_BtoH$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_8c_CMEVXALU_PERMS_1rc>],
             (instregex "^(SQCVT|UQCVT|SQCVTU|SQCVTN|SQCVTUN|UQCVTN)_Z4Z_",
                        "^SQRSHRU_VG2_Z2ZI_",
                        "^(SQRSHR|UQRSHR|SQRSHRU|SQRSHRN|UQRSHRN|SQRSHRUN)_VG4_Z4ZI_")>;

// SME integer and floating-point dot and multiply-accumulate families.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0>],
             (instregex "^FMLALL_MZZI_BtoS$", "^FMLALL_MZZ_BtoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_2rc>],
             (instregex "^(SUDOT|USDOT)_VG2_M2(ZZI|ZZ|Z2Z)_BToS$",
                        "^(SMLAL|UMLAL|SMLSL|UMLSL)_(MZZI|MZZ|VG2_M2)",
                        "^(SMLALL|UMLALL|USMLALL|SMLSLL|UMLSLL)_(MZZI|MZZ|VG2_M2ZZI|VG2_M2ZZ|VG2_M2Z2Z)_(BtoS|HtoD)$",
                        "^(SUMLALL|USMLALL)_(VG2_M2ZZI|VG2_M2ZZ|VG2_M2Z2Z)_BtoS$",
                        "^(SVDOT|UVDOT)_VG2_M2ZZI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_4rc>],
             (instregex "^(SUDOT|USDOT)_VG4_M4(ZZI|ZZ|Z4Z)_BToS$",
                        "^(SMLAL|UMLAL|SMLSL|UMLSL)_VG4_M4",
                        "^(SMLALL|UMLALL|USMLALL|SMLSLL|UMLSLL)_(VG4_M4ZZI|VG4_M4ZZ|VG4_M4Z4Z)_(BtoS|HtoD)$",
                        "^(SUMLALL|USMLALL)_(VG4_M4ZZI|VG4_M4ZZ|VG4_M4Z4Z)_BtoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_2rc>],
             (instregex "^(BFDOT|FMLA|FMLS|BFMLA|BFMLS)_VG2_M2(ZZI|ZZ|Z2Z)(_HtoS|_[HSD])?$",
                        "^(FMLAL|FMLSL|BFMLAL|BFMLSL)_VG2_M2(ZZI|ZZ|Z2Z)(_BtoH|_HtoS)?$",
                        "^FMLALL_VG2_M2(ZZI|ZZ|Z2Z)_BtoS$",
                        "^(FVDOT|BFVDOT)_VG2_M2ZZI_HtoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_4rc>],
             (instregex "^(BFDOT|FMLA|FMLS|BFMLA|BFMLS)_VG4_M4(ZZI|ZZ|Z4Z)(_HtoS|_[HSD])?$",
                        "^(FMLAL|FMLSL|BFMLAL|BFMLSL)_VG4_M4(ZZI|ZZ|Z4Z)(_BtoH|_HtoS)?$",
                        "^FMLALL_VG4_M4(ZZI|ZZ|Z4Z)_BtoS$",
                        "^FVDOT[BT]_VG4_M2ZZI_BtoS$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_6rc>],
             (instregex "^(SUVDOT|USVDOT)_VG4_M4ZZI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEMML0_8rc>],
             (instregex "^(SVDOT|UVDOT)_VG4_M4ZZI_")>;

// Streaming-only misc and ZA move instructions.
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEPERMS>],
             (instregex "^(SUNPK|UUNPK)_VG2_2ZZ_", "^(ZIP|UZP)_VG2_2ZZZ")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEPERMS_2rc>],
             (instregex "^(SUNPK|UUNPK)_VG4_4Z2Z_",
                        "^(ZIP|UZP)_VG4_4Z4Z")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEPERMS>],
             (instregex "^SEL_VG2_2ZC2Z2Z_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEPERMS_2rc>],
             (instregex "^SEL_VG4_4ZC4Z4Z_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEMML0_2rc>],
             (instregex "^MOVA_MXI2Z_", "^MOVA_VG2_MXI2Z$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEMML0_4rc>],
             (instregex "^MOVA_MXI4Z_", "^MOVA_VG4_MXI4Z$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_9c_CMEMML1>],
             (instregex "^MOVA_2ZMXI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_9c_CMEMML1_2rc>],
             (instregex "^MOVA_4ZMXI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_9c_CMEMML1_5rc>],
             (instregex "^MOVA_VG2_2ZMXI$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_13c_CMEMML1_PERMF_5rc>],
             (instregex "^MOVA_VG4_4ZMXI$")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEPERMF_1rc>],
             (instrs MOVT_TIX)>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_8c_CMEPERMF_VX0ALU_1rc>],
             (instrs MOVT_XTI)>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_4c_CMEPERMF_1rc>],
             (instregex "^LUTI[24]_ZTZI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_5c_CMEPERMF_6rc>],
             (instregex "^LUTI[24](_S)?_2ZTZI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_7c_CMEPERMF_12rc>],
             (instregex "^LUTI[24](_S)?_4ZTZI_")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_1c_CMEMML0_4rc>],
             (instregex "^ZERO_M")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_1c_CMERNMRET>],
             (instrs ZERO_T)>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_7c_1L_CMELD>],
             (instregex "^LDR_ZA")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_11c_1SA_CMEST>],
             (instregex "^STR_ZA")>;
def : InstRW<[C1UStreamingOnlySchedWrite<C1UWrite_2c_CMEPF_2rc>],
             (instrs RPRFM)>;
}
