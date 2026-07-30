//==- AArch64SchedA65.td - Cortex-A65 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM Cortex A65 processors.
//
// Cortex-A65/A65AE and Neoverse-E1 previously reused CortexA53Model directly.
// This model is a verbatim copy of it, split out so that Cortex-A53-specific
// tuning (which targets an in-order core with a 64-bit ASIMD datapath) is not
// applied to these out-of-order cores, whose Advanced SIMD datapath is 128 bits
// wide. It is a placeholder: a model derived from the Cortex-A65 Software
// Optimization Guide is future work.
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Cortex-A65 machine model for scheduling and other instruction cost heuristics.
def CortexA65Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since A65 is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;       // Optimistic load latency assuming bypass.
                             // This is overridden by OperandCycles if the
                             // Itineraries are queried instead.
  let MispredictPenalty = 9; // Based on "Cortex-A65 Software Optimisation
                             // Specification - Instruction Timings"
                             // v 1.0 Spreadsheet
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}


//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since
// Cortex-A65 is in-order.

def A65UnitALU    : ProcResource<2> { let BufferSize = 0; } // Int ALU
def A65UnitMAC    : ProcResource<1> { let BufferSize = 0; } // Int MAC
def A65UnitDiv    : ProcResource<1> { let BufferSize = 0; } // Int Division
def A65UnitLdSt   : ProcResource<1> { let BufferSize = 0; } // Load/Store
def A65UnitB      : ProcResource<1> { let BufferSize = 0; } // Branch
def A65UnitFPALU  : ProcResource<1> { let BufferSize = 0; } // FP ALU
def A65UnitFPMDS  : ProcResource<1> { let BufferSize = 0; } // FP Mult/Div/Sqrt


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types which both map the ProcResources and
// set the latency.

let SchedModel = CortexA65Model in {

// ALU - Despite having a full latency of 4, most of the ALU instructions can
//       forward a cycle earlier and then two cycles earlier in the case of a
//       shift-only instruction. These latencies will be incorrect when the
//       result cannot be forwarded, but modeling isn't rocket surgery.
def : WriteRes<WriteImm, [A65UnitALU]> { let Latency = 3; }
def : WriteRes<WriteI, [A65UnitALU]> { let Latency = 3; }
def : WriteRes<WriteISReg, [A65UnitALU]> { let Latency = 3; }
def : WriteRes<WriteIEReg, [A65UnitALU]> { let Latency = 3; }
def : WriteRes<WriteIS, [A65UnitALU]> { let Latency = 2; }
def : WriteRes<WriteExtr, [A65UnitALU]> { let Latency = 3; }

// MAC
def : WriteRes<WriteIM32, [A65UnitMAC]> { let Latency = 4; }
def : WriteRes<WriteIM64, [A65UnitMAC]> { let Latency = 4; }

// Div
def : WriteRes<WriteID32, [A65UnitDiv]> { let Latency = 4; }
def : WriteRes<WriteID64, [A65UnitDiv]> { let Latency = 4; }

// Load
def : WriteRes<WriteLD, [A65UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDIdx, [A65UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDHi, [A65UnitLdSt]> { let Latency = 4; }

// Vector Load - Vector loads take 1-5 cycles to issue. For the WriteVecLd
//               below, choosing the median of 3 which makes the latency 6.
//               May model this more carefully in the future. The remaining
//               A65WriteVLD# types represent the 1-5 cycle issues explicitly.
def : WriteRes<WriteVLD, [A65UnitLdSt]> { let Latency = 6;
                                          let ReleaseAtCycles = [3]; }
def A65WriteVLD1 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 4; }
def A65WriteVLD2 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def A65WriteVLD3 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }
def A65WriteVLD4 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 7;
                                                  let ReleaseAtCycles = [4]; }
def A65WriteVLD5 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 8;
                                                  let ReleaseAtCycles = [5]; }

// Pre/Post Indexing - Performed as part of address generation which is already
//                     accounted for in the WriteST* latencies below
def : WriteRes<WriteAdr, []> { let Latency = 0; }

// Store
def : WriteRes<WriteST, [A65UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTP, [A65UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTIdx, [A65UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTX, [A65UnitLdSt]> { let Latency = 4; }

// Vector Store - Similar to vector loads, can take 1-3 cycles to issue.
def : WriteRes<WriteVST, [A65UnitLdSt]> { let Latency = 5;
                                          let ReleaseAtCycles = [2];}
def A65WriteVST1 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 4; }
def A65WriteVST2 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def A65WriteVST3 : SchedWriteRes<[A65UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Branch
def : WriteRes<WriteBr, [A65UnitB]>;
def : WriteRes<WriteBrReg, [A65UnitB]>;
def : WriteRes<WriteSys, [A65UnitB]>;
def : WriteRes<WriteBarrier, [A65UnitB]>;
def : WriteRes<WriteHint, [A65UnitB]>;

// FP ALU
def : WriteRes<WriteF, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCmp, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCvt, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCopy, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFImm, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteVd, [A65UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteVq, [A65UnitFPALU]> { let Latency = 6; }

// FP Mul, Div, Sqrt
def : WriteRes<WriteFMul, [A65UnitFPMDS]> { let Latency = 6; }
def : WriteRes<WriteFDiv, [A65UnitFPMDS]> { let Latency = 33;
                                            let ReleaseAtCycles = [29]; }
def A65WriteFMAC : SchedWriteRes<[A65UnitFPMDS]> { let Latency = 10; }
def A65WriteFDivSP : SchedWriteRes<[A65UnitFPMDS]> { let Latency = 18;
                                                     let ReleaseAtCycles = [14]; }
def A65WriteFDivDP : SchedWriteRes<[A65UnitFPMDS]> { let Latency = 33;
                                                     let ReleaseAtCycles = [29]; }
def A65WriteFSqrtSP : SchedWriteRes<[A65UnitFPMDS]> { let Latency = 17;
                                                      let ReleaseAtCycles = [13]; }
def A65WriteFSqrtDP : SchedWriteRes<[A65UnitFPMDS]> { let Latency = 32;
                                                      let ReleaseAtCycles = [28]; }

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types.

// No forwarding for these reads.
def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadST, 0>;
def : ReadAdvance<ReadVLD, 0>;

// ALU - Most operands in the ALU pipes are not needed for two cycles. Shiftable
//       operands are needed one cycle later if and only if they are to be
//       shifted. Otherwise, they too are needed two cycles later. This same
//       ReadAdvance applies to Extended registers as well, even though there is
//       a separate SchedPredicate for them.
def : ReadAdvance<ReadI, 2, [WriteImm,WriteI,
                             WriteISReg, WriteIEReg,WriteIS,
                             WriteID32,WriteID64,
                             WriteIM32,WriteIM64]>;
def A65ReadShifted : SchedReadAdvance<1, [WriteImm,WriteI,
                                          WriteISReg, WriteIEReg,WriteIS,
                                          WriteID32,WriteID64,
                                          WriteIM32,WriteIM64]>;
def A65ReadNotShifted : SchedReadAdvance<2, [WriteImm,WriteI,
                                             WriteISReg, WriteIEReg,WriteIS,
                                             WriteID32,WriteID64,
                                             WriteIM32,WriteIM64]>;
def A65ReadISReg : SchedReadVariant<[
	SchedVar<RegShiftedPred, [A65ReadShifted]>,
	SchedVar<NoSchedPred, [A65ReadNotShifted]>]>;
def : SchedAlias<ReadISReg, A65ReadISReg>;

def A65ReadIEReg : SchedReadVariant<[
	SchedVar<RegExtendedPred, [A65ReadShifted]>,
	SchedVar<NoSchedPred, [A65ReadNotShifted]>]>;
def : SchedAlias<ReadIEReg, A65ReadIEReg>;

// MAC - Operands are generally needed one cycle later in the MAC pipe.
//       Accumulator operands are needed two cycles later.
def : ReadAdvance<ReadIM, 1, [WriteImm,WriteI,
                              WriteISReg, WriteIEReg,WriteIS,
                              WriteID32,WriteID64,
                              WriteIM32,WriteIM64]>;
def : ReadAdvance<ReadIMA, 2, [WriteImm,WriteI,
                               WriteISReg, WriteIEReg,WriteIS,
                               WriteID32,WriteID64,
                               WriteIM32,WriteIM64]>;

// Div
def : ReadAdvance<ReadID, 1, [WriteImm,WriteI,
                              WriteISReg, WriteIEReg,WriteIS,
                              WriteID32,WriteID64,
                              WriteIM32,WriteIM64]>;

//===----------------------------------------------------------------------===//
// Subtarget-specific InstRWs.

//---
// Miscellaneous
//---
def : InstRW<[WriteI], (instrs COPY)>;

//---
// Vector Loads
//---
def : InstRW<[A65WriteVLD1], (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[A65WriteVLD1], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD1], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD2], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD3], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD4], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A65WriteVLD1], (instregex "LD1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD1], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD1], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD3], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD4], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[A65WriteVLD1], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[A65WriteVLD1], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD2], (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[A65WriteVLD4], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A65WriteVLD1], (instregex "LD2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD1], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD4], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A65WriteVLD2], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[A65WriteVLD2], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD4], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A65WriteVLD3], (instregex "LD3Threev2d$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD4], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD3], (instregex "LD3Threev2d_POST$")>;

def : InstRW<[A65WriteVLD2], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[A65WriteVLD2], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVLD5], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A65WriteVLD4], (instregex "LD4Fourv(2d)$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD2], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD5], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVLD4], (instregex "LD4Fourv(2d)_POST$")>;

//---
// Vector Stores
//---
def : InstRW<[A65WriteVST1], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[A65WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A65WriteVST2], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A65WriteVST1], (instregex "ST1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[A65WriteVST1], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[A65WriteVST1], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[A65WriteVST2], (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A65WriteVST1], (instregex "ST2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST1], (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A65WriteVST2], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[A65WriteVST3], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A65WriteVST2], (instregex "ST3Threev(2d)$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST3], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST3Threev(2d)_POST$")>;

def : InstRW<[A65WriteVST2], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[A65WriteVST3], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A65WriteVST2], (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST3], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A65WriteVST2], (instregex "ST4Fourv(2d)_POST$")>;

//---
// Floating Point MAC, DIV, SQRT
//---
def : InstRW<[A65WriteFMAC], (instregex "^FN?M(ADD|SUB).*")>;
def : InstRW<[A65WriteFMAC], (instregex "^FML(A|S).*")>;
def : InstRW<[A65WriteFDivSP], (instrs FDIVSrr)>;
def : InstRW<[A65WriteFDivDP], (instrs FDIVDrr)>;
def : InstRW<[A65WriteFDivSP], (instregex "^FDIVv.*32$")>;
def : InstRW<[A65WriteFDivDP], (instregex "^FDIVv.*64$")>;
def : InstRW<[A65WriteFSqrtSP], (instregex "^.*SQRT.*32$")>;
def : InstRW<[A65WriteFSqrtDP], (instregex "^.*SQRT.*64$")>;

}
