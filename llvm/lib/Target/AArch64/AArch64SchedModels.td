//=- AArch64SchedModels.td - AArch64 Scheduling Models -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unsupported features to disable for scheduling models
//===----------------------------------------------------------------------===//

class AArch64Unsupported { list<Predicate> F; }

let F = [HasSVE2p1, HasSVE2p1_or_SME, HasSVE2p1_or_SME2, HasSVE2p1_or_StreamingSME2, HasSVE2p1_or_SME2p1] in
def SVE2p1Unsupported : AArch64Unsupported;

def SVE2Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSVE2, HasSVE2_or_SME, HasNonStreamingSVE2_or_SME2, HasSSVE_FP8FMA, HasSMEF8F16,
                       HasSSVE_FP8DOT2, HasSSVE_FP8DOT4,
                       HasSMEF8F32, HasSVEAES, HasSVESHA3, HasSVESM4, HasSVEBitPerm,
                       HasSVEB16B16, HasSVEBFSCALE],
                       SVE2p1Unsupported.F);
}

def SVEUnsupported : AArch64Unsupported {
  let F = !listconcat([HasSVE, HasSVE_or_SME],
                      SVE2Unsupported.F);
}

def SME2p3Unsupported : AArch64Unsupported {
  let F = [HasSVE2p3_or_SME2p3, HasSVE_B16MM];
}

def SME2p2Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSME2p2, HasSVE2p2_or_SME2p2,
           HasNonStreamingSVE_or_SME2p2,
           HasNonStreamingSVE2p2_or_SME2p2],
           SME2p3Unsupported.F);
}

def SME2p1Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSME2p1, HasSVE2p1_or_SME2p1,
                       HasSME_MOP4, HasSME_TMOP, HasNonStreamingSVE_or_SSVE_FEXPA,
                       HasNonStreamingSVE_or_SSVE_BitPerm],
                       SME2p2Unsupported.F);
}

def SME2Unsupported : AArch64Unsupported {
  let F = !listconcat([HasSME2, HasNonStreamingSVE2_or_SME2, HasSVE2p1_or_SME2, HasSSVE_FP8FMA,
                      HasSSVE_FP8DOT2, HasSSVE_FP8DOT4,
                      HasSMEF8F16, HasSMEF8F32, HasSMEF16F16_or_SMEF8F16, HasSMEB16B16,
                      HasNonStreamingSVE_or_SSVE_AES, HasSVE2p1_or_StreamingSME2, HasSME2andIsNonStreamingSafe],
                      SME2p1Unsupported.F);
}

def SMEUnsupported : AArch64Unsupported {
  let F = !listconcat([HasSME, HasSMEI16I64, HasSMEF16F16, HasSMEF64F64, HasSMEFA64,
                       HasSMEandIsNonStreamingSafe],
                      SME2Unsupported.F);
}

def MTEUnsupported : AArch64Unsupported {
  let F = [HasMTE];
}

let F = [HasPAuth, HasPAuthLR] in
def PAUnsupported : AArch64Unsupported;

// Cortex-A320 machine model for scheduling and other instruction cost heuristics.
def CortexA320Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // Cortex-A320 is an in-order processor
  let IssueWidth = 1;         // Cortex-A320 is a single-issue processor
  let LoadLatency = 5;
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // Covers instructions applicable to Cortex-A320.

  let FullInstRWOverlapCheck = 0;
}

// Cortex-A53 machine model for scheduling and other instruction cost heuristics.
def CortexA53Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since A53 is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;       // Optimistic load latency assuming bypass.
                             // This is overridden by OperandCycles if the
                             // Itineraries are queried instead.
  let MispredictPenalty = 9; // Based on "Cortex-A53 Software Optimisation
                             // Specification - Instruction Timings"
                             // v 1.0 Spreadsheet
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

// Cortex-A55 machine model for scheduling and other instruction cost heuristics.
def CortexA55Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // The Cortex-A55 is an in-order processor
  let IssueWidth = 2;         // It dual-issues under most circumstances
  let LoadLatency = 4;        // Cycles for loads to access the cache. The
                              // optimisation guide shows that most loads have
                              // a latency of 3, but some have a latency of 4
                              // or 5. Setting it 4 looked to be good trade-off.
  let MispredictPenalty = 8;  // A branch direction mispredict.
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // Covers instructions applicable to Cortex-A55.

  list<Predicate> UnsupportedFeatures = [HasSVE, HasMTE];

  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

// Cortex-A510 machine model for scheduling and other instruction cost heuristics.
def CortexA510Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // The Cortex-A510 is an in-order processor
  let IssueWidth = 3;         // It dual-issues under most circumstances
  let LoadLatency = 3;        // Cycles for loads to access the cache.
                              // Most loads have a latency of 2, but some have higher latencies.
                              // 3 seems to be a good tradeoff
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // Covers instructions applicable to Cortex-A510.

  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

def CortexA57Model : SchedMachineModel {
  let IssueWidth        =   3; // 3-way decode and dispatch
  let MicroOpBufferSize = 128; // 128 micro-op re-order buffer
  let LoadLatency       =   4; // Optimistic load latency
  let MispredictPenalty =  14; // Fetch + Decode/Rename/Dispatch + Branch

  // Enable partial & runtime unrolling. The magic number is chosen based on
  // experiments and benchmarking data.
  let LoopMicroOpBufferSize = 16;
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

// C1-Nano machine model for scheduling and other instruction cost heuristics.
def C1NanoModel : SchedMachineModel {
  let MicroOpBufferSize = 0;  // The C1-Nano is an in-order processor
  let IssueWidth = 3;         // It dual-issues under most circumstances
  let LoadLatency = 3;        // Cycles for loads to access the cache.
                              // 2 is best case, 4 is normal case.
                              // 3 seems to be a good tradeoff
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // The C1-Nano model is incomplete due
                              // SME instruction details that are not
                              // available in the SWOG.

  list<Predicate> UnsupportedFeatures = !listconcat(SVE2Unsupported.F,
                                                    SME2Unsupported.F,
                                                    [HasSVEB16B16, HasSVEBFSCALE,
                                                     HasCPA, HasMTE, HasCSSC]);
}

def C1UltraModel : SchedMachineModel {
  let IssueWidth            =  10;
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer. NOTE: Copied from Neoverse-V3
  let LoadLatency           =   4; // Optimistic load latency. NOTE: Copied from Neoverse-V3
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch. NOTE: Copied from N3.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVE2p1Unsupported.F, SME2p1Unsupported.F,
                                                    [HasSVEB16B16, HasCPA, HasMTE,
                                                     HasCSSC, HasMOPS, HasMOPS_GO,
                                                     HasSVEBFSCALE, HasFAMINMAX]);
}

def C1PremiumModel : SchedMachineModel {
  let IssueWidth            =  10;
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16;
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SME2p1Unsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasSVEBFSCALE,
                                                     HasCPA, HasCSSC,
                                                     HasMOPS, HasMOPS_GO]);
}

def CycloneModel : SchedMachineModel {
  let IssueWidth = 6; // 6 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 192; // Based on the reorder buffer.
  let LoadLatency = 4; // Optimistic load latency.
  let MispredictPenalty = 16; // 14-19 cycles are typical.
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def FalkorModel : SchedMachineModel {
  let IssueWidth = 8;          // 8 uops are dispatched per cycle.
  let MicroOpBufferSize = 128; // Out-of-order with temporary unified issue buffer.
  let LoopMicroOpBufferSize = 16;
  let LoadLatency = 3;         // Optimistic load latency.
  let MispredictPenalty = 11;  // Minimum branch misprediction penalty.
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

def KryoModel : SchedMachineModel {
  let IssueWidth        =   5; // 5-wide issue for expanded uops
  let MicroOpBufferSize = 128; // Out-of-order with temporary unified issue buffer
  let LoadLatency       =   4; // Optimistic load latency
  let MispredictPenalty =  14; // Fetch + Decode/Rename/Dispatch + Branch

  // Enable partial & runtime unrolling. The magic number is chosen based on
  // experiments and benchmarking data.
  let LoopMicroOpBufferSize = 16;
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

def ExynosM3Model : SchedMachineModel {
  let IssueWidth            =   6; // Up to 6 uops per cycle.
  let MicroOpBufferSize     = 228; // ROB size.
  let LoopMicroOpBufferSize =  40; // Based on the instruction queue size.
  let LoadLatency           =   4; // Optimistic load cases.
  let MispredictPenalty     =  16; // Minimum branch misprediction penalty.
  let CompleteModel         =   1; // Use the default model otherwise.

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def ExynosM4Model : SchedMachineModel {
  let IssueWidth            =   6; // Up to 6 uops per cycle.
  let MicroOpBufferSize     = 228; // ROB size.
  let LoopMicroOpBufferSize =  48; // Based on the instruction queue size.
  let LoadLatency           =   4; // Optimistic load cases.
  let MispredictPenalty     =  16; // Minimum branch misprediction penalty.
  let CompleteModel         =   1; // Use the default model otherwise.

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def ExynosM5Model : SchedMachineModel {
  let IssueWidth            =   6; // Up to 6 uops per cycle.
  let MicroOpBufferSize     = 228; // ROB size.
  let LoopMicroOpBufferSize =  60; // Based on the instruction queue size.
  let LoadLatency           =   4; // Optimistic load cases.
  let MispredictPenalty     =  15; // Minimum branch misprediction penalty.
  let CompleteModel         =   1; // Use the default model otherwise.

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

// Cavium ThunderX T8X scheduling machine model.
def ThunderXT8XModel : SchedMachineModel {
  let IssueWidth = 2;         // 2 micro-ops dispatched per cycle.
  let MicroOpBufferSize = 0;  // ThunderX T88/T81/T83 are in-order.
  let LoadLatency = 3;        // Optimistic load latency.
  let MispredictPenalty = 8;  // Branch mispredict penalty.
  let PostRAScheduler = 1;    // Use PostRA scheduler.
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

def ThunderX2T99Model : SchedMachineModel {
  let IssueWidth            =   4; // 4 micro-ops dispatched at a time.
  let MicroOpBufferSize     = 180; // 180 entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  12; // Extra cycles for mispredicted branch.
  // Determined via a mix of micro-arch details and experimentation.
  let LoopMicroOpBufferSize = 128;
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

def A64FXModel : SchedMachineModel {
  let IssueWidth            =   6; // 6 micro-ops dispatched at a time.
  let MicroOpBufferSize     = 180; // 180 entries in micro-op re-order buffer.
  let LoadLatency           =   5; // Optimistic load latency.
  let MispredictPenalty     =  12; // Extra cycles for mispredicted branch.
  // Determined via a mix of micro-arch details and experimentation.
  let LoopMicroOpBufferSize = 128;
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F, SVEUnsupported.F,
                                                    [HasMTE, HasMatMulInt8, HasBF16,
                                                    HasPAuth, HasPAuthLR, HasCPA,
                                                    HasCSSC, HasMOPS, HasMOPS_GO]);
  let FullInstRWOverlapCheck = 0;
}

def ThunderX3T110Model : SchedMachineModel {
  let IssueWidth            =   4; // 4 micro-ops dispatched at a time.
  let MicroOpBufferSize     =  70; // 70 entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  12; // Extra cycles for mispredicted branch.
  // Determined via a mix of micro-arch details and experimentation.
  let LoopMicroOpBufferSize = 128; // FIXME: might be much bigger in TX3.
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

// Huawei TSV110 scheduling machine model.
def TSV110Model : SchedMachineModel {
  let IssueWidth            =   4; // 4 micro-ops dispatched  per cycle.
  let MicroOpBufferSize     = 128; // 128 micro-op re-order buffer
  let LoopMicroOpBufferSize =  16;
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  14; // Fetch + Decode/Rename/Dispatch + Branch
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def HIP12Model : SchedMachineModel {
  let IssueWidth            =   8; // Micro-ops dispatched at a time.
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVEB16B16, HasSVEBFSCALE, HasMTE,
                                                     HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def Ampere1Model : SchedMachineModel {
  let IssueWidth            =   4;  // 4-way decode and dispatch
  let MicroOpBufferSize     = 192;  // re-order buffer size
  let LoadLatency           =   4;  // Optimistic load latency
  let MispredictPenalty     =  10;  // Branch mispredict penalty
  let LoopMicroOpBufferSize =  32;  // Instruction queue size
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    SMEUnsupported.F,
                                                    PAUnsupported.F,
                                                    [HasMTE]);
}

def Ampere1BModel : SchedMachineModel {
  let IssueWidth            =  12;  // Maximum micro-ops dispatch rate.
  let MicroOpBufferSize     = 208;  // micro-op re-order buffer size
  let LoadLatency           =   3;  // Optimistic load latency
  let MispredictPenalty     =  10;  // Branch mispredict penalty
  let LoopMicroOpBufferSize =  32;  // Instruction queue size
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    SMEUnsupported.F,
                                                    PAUnsupported.F,
                                                    [HasMOPS, HasMOPS_GO]);
}

def NeoverseN1Model : SchedMachineModel {
  let IssueWidth            =   3; // This value comes from the decode bandwidth
                                   // and empirical measurements showed that this
                                   // value is better.
  let MicroOpBufferSize     = 128; // NOTE: Copied from Cortex-A76.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  11; // Cycles cost of branch mispredicted.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    SVEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseN2Model : SchedMachineModel {
  let IssueWidth            =   5; // Micro-ops dispatched at a time.
  let MicroOpBufferSize     = 160; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
    [HasSVE2p1, HasSVEB16B16, HasSVEBFSCALE, HasPAuthLR, HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseN3Model : SchedMachineModel {
    let IssueWidth            =   5; // Micro-ops dispatched at a time.
    let MicroOpBufferSize     = 160; // Entries in micro-op re-order buffer. NOTE: Copied from N2.
    let LoadLatency           =   4; // Optimistic load latency.
    let MispredictPenalty     =  10; // Extra cycles for mispredicted branch. NOTE: Copied from N2.
    let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
    let CompleteModel         =   1;

    list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                      SVE2p1Unsupported.F,
        [HasSVE2p1, HasSVEB16B16, HasSVEBFSCALE, HasPAuthLR, HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseV1Model : SchedMachineModel {
  let IssueWidth            =   8; // This value comes from the decode bandwidth
                                   // and empirical measurements showed that this
                                   // value is better.
  let MicroOpBufferSize     = 256; // Micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  11; // Cycles cost of branch mispredicted.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVE2Unsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCPA,
                                                    HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseV2Model : SchedMachineModel {
  let IssueWidth            =   6; // This value comes from the decode bandwidth
                                   // and empirical measurements showed that a
                                   // lower value is better.
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.  NOTE: Copied from N2.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasSVEBFSCALE,
                                                     HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseV3Model : SchedMachineModel {
  let IssueWidth            =  10; // Expect best value to be slightly higher than V2
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer. NOTE: Copied from Neoverse-V2
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.  NOTE: Copied from N2.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasSVEBFSCALE,
                                                     HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

def NeoverseV3AEModel : SchedMachineModel {
  let IssueWidth            =  10; // Expect best value to be slightly higher than V2
  let MicroOpBufferSize     = 320; // Entries in micro-op re-order buffer. NOTE: Copied from Neoverse-V2
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.  NOTE: Copied from N2.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasSVEBFSCALE,
                                                     HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

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
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasSVEBFSCALE,
                                                     HasCPA, HasCSSC,
                                                     HasMatMulFP64, HasMOPS, HasMOPS_GO]);
}

def OryonModel : SchedMachineModel {
  let IssueWidth            =  14;
  let MicroOpBufferSize     = 376;
  let LoadLatency           =   4;
  let MispredictPenalty     =  13; // 13 cycles for mispredicted branch.
  let LoopMicroOpBufferSize =   16; // Oryon-1 does not have loop micro op buffer, we enable this pseudo value to allow for aggressive unrolling based on runtime TC.
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    SMEUnsupported.F,
                                                    MTEUnsupported.F,
                                                    PAUnsupported.F,
                                                    [HasPAuth, HasCSSC, HasMOPS, HasMOPS_GO]);
}
