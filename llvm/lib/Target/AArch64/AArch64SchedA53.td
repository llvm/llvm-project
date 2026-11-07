//==- AArch64SchedA53.td - Cortex-A53 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM Cortex A53 processors.
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

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
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO,
                                                     HasFPv2]);
}


//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since
// Cortex-A53 is in-order.

def A53UnitALU    : ProcResource<2> { let BufferSize = 0; } // Int ALU
def A53UnitMAC    : ProcResource<1> { let BufferSize = 0; } // Int MAC
def A53UnitDiv    : ProcResource<1> { let BufferSize = 0; } // Int Division
def A53UnitLdSt   : ProcResource<1> { let BufferSize = 0; } // Load/Store
def A53UnitB      : ProcResource<1> { let BufferSize = 0; } // Branch
// The FP/ASIMD datapath moves 128 bits per cycle, which it can spend either on
// one 128-bit (Q-form) operation or on two 64-bit ones. Model this as two
// lanes shared by the arithmetic and multiply operations: 64-bit and scalar
// operations take a single lane, Q-form operations take both, so a Q-form FMUL
// competes with a scalar FADD for them.
// TODO: a Q-form should hold both lanes in the same cycle. Taking the
// two-unit resource twice gives the right throughput for Q-forms on their
// own, but llvm-mca lets a Q-form overlap a scalar operation, so a Q-form +
// scalar pair is modelled at 1.5 cycles where hardware takes 2.
def A53UnitFPALU  : ProcResource<2> { let BufferSize = 0; } // FP ALU/Mult
// Divide and square root run in their own unit, without holding back the
// arithmetic and multiply operations that follow them.
def A53UnitFPDS   : ProcResource<2> { let BufferSize = 0; } // FP Div/Sqrt
// Some scalar and 64-bit operations only issue as one particular instruction
// of a dual-issue pair: FP compares only as the older one (BeginGroup), and the
// by-element multiplies and the other operations marked EndGroup only as the
// younger one. So two of the same kind never issue together, but a compare and
// a by-element multiply can.
// The younger-only operations also share A53UnitFPRestricted, which the
// accumulating operations hold for two cycles (see A53WriteAccD).
def A53UnitFPRestricted : ProcResource<1> { let BufferSize = 0; }


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types which both map the ProcResources and
// set the latency.

let SchedModel = CortexA53Model in {

// ALU - Despite having a full latency of 4, most of the ALU instructions can
//       forward a cycle earlier and then two cycles earlier in the case of a
//       shift-only instruction. These latencies will be incorrect when the
//       result cannot be forwarded, but modeling isn't rocket surgery.
def : WriteRes<WriteImm, [A53UnitALU]> { let Latency = 3; }
def : WriteRes<WriteI, [A53UnitALU]> { let Latency = 3; }
def : WriteRes<WriteISReg, [A53UnitALU]> { let Latency = 3; }
def : WriteRes<WriteIEReg, [A53UnitALU]> { let Latency = 3; }
def : WriteRes<WriteIS, [A53UnitALU]> { let Latency = 2; }
def : WriteRes<WriteExtr, [A53UnitALU]> { let Latency = 3; }

// MAC
def : WriteRes<WriteIM32, [A53UnitMAC]> { let Latency = 4; }
def : WriteRes<WriteIM64, [A53UnitMAC]> { let Latency = 4; }

// Div
def : WriteRes<WriteID32, [A53UnitDiv]> { let Latency = 4; }
def : WriteRes<WriteID64, [A53UnitDiv]> { let Latency = 4; }

// Load
def : WriteRes<WriteLD, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDIdx, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDHi, [A53UnitLdSt]> { let Latency = 4; }

// Vector Load - Vector loads take 1-5 cycles to issue. For the WriteVecLd
//               below, choosing the median of 3 which makes the latency 6.
//               May model this more carefully in the future. The remaining
//               A53WriteVLD# types represent the 1-5 cycle issues explicitly.
def : WriteRes<WriteVLD, [A53UnitLdSt]> { let Latency = 6;
                                          let ReleaseAtCycles = [3]; }
def A53WriteVLD1 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 4; }
def A53WriteVLD2 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def A53WriteVLD3 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }
def A53WriteVLD4 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 7;
                                                  let ReleaseAtCycles = [4]; }
def A53WriteVLD5 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 8;
                                                  let ReleaseAtCycles = [5]; }

// Pre/Post Indexing - Performed as part of address generation which is already
//                     accounted for in the WriteST* latencies below
def : WriteRes<WriteAdr, []> { let Latency = 0; }

// Store
def : WriteRes<WriteST, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTP, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTIdx, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTX, [A53UnitLdSt]> { let Latency = 4; }

// Vector Store - Similar to vector loads, can take 1-3 cycles to issue.
def : WriteRes<WriteVST, [A53UnitLdSt]> { let Latency = 5;
                                          let ReleaseAtCycles = [2];}
def A53WriteVST1 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 4; }
def A53WriteVST2 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def A53WriteVST3 : SchedWriteRes<[A53UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Branch
def : WriteRes<WriteBr, [A53UnitB]>;
def : WriteRes<WriteBrReg, [A53UnitB]>;
def : WriteRes<WriteSys, [A53UnitB]>;
def : WriteRes<WriteBarrier, [A53UnitB]>;
def : WriteRes<WriteHint, [A53UnitB]>;

// The FP/ASIMD pipeline does not write its results back in program order: a
// later, shorter operation is not held back by an earlier, longer one (an
// independent ADD issues right behind an FMLA on hardware). Every FP/ASIMD
// write is therefore RetireOOO, so that llvm-mca's in-order write-back check
// does not delay the instructions that follow a long-latency operation.

// FP ALU. Scalar floating-point arithmetic and conversion have a 4-cycle
// latency on Cortex-A53.
def : WriteRes<WriteF, [A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// A compare pairs with an independent FP/ASIMD or integer operation, but only
// as the older instruction of the pair, so not with another compare.
// A dependent NZCV consumer only pays a single cycle in practice, but the
// forwarding path that makes that possible cannot be expressed here, so this
// keeps the 3-cycle result latency the rest of the family uses.
def : WriteRes<WriteFCmp, [A53UnitFPALU]> {
  let Latency = 3;
  let RetireOOO = 1;
  let BeginGroup = 1;
}
def : WriteRes<WriteFCvt, [A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// Register transfers between a GPR and an FP/SIMD register (FMOV). The
// generic class only covers what the A53WriteF2R/A53WriteR2F overrides below
// do not; see those for the measured behaviour.
def : WriteRes<WriteFCopy, [A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
def : WriteRes<WriteFImm, [A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
def : WriteRes<WriteVd, [A53UnitFPALU]> {
  let Latency = 6;
  let RetireOOO = 1;
}
// A 128-bit (Q-form) operation fills the whole 128-bit datapath, so it takes
// both lanes.
def : WriteRes<WriteVq, [A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 6;
  let RetireOOO = 1;
}

// FP Mul, Div, Sqrt. Divide and square root are not pipelined and block their
// whole unit, so they hold it for their full occupancy.
def : WriteRes<WriteFMul, [A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// The scalar and 64-bit by-element multiplies only issue as the younger
// instruction of a pair, so at most one per cycle, and their Q-forms take both
// lanes like any other Q-form operation.
def A53WriteFMulIdx : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 4;
  let RetireOOO = 1;
  let EndGroup = 1;
}
def A53WriteFMulIdxQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// The other 4-cycle floating-point operations that only issue as the younger
// instruction of a pair.
def A53WriteVecYng_4 : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 4;
  let RetireOOO = 1;
  let EndGroup = 1;
}
def : WriteRes<WriteFDiv, [A53UnitFPDS, A53UnitFPDS]> {
  let Latency = 33;
  let ReleaseAtCycles = [29, 29];
  let RetireOOO = 1;
}
// FP multiply-accumulate has an 8-cycle result latency, but the MAC pipe reads
// the addend late and has a dedicated forwarding path that feeds one MAC's
// result into the next MAC's addend, so a dependent MAC chain recurs every 4
// cycles. Only the addend forwards; the multiplicands do not.
def A53WriteFMAC : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 8;
  let RetireOOO = 1;
}
def A53WriteFMACIdx : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 8;
  let RetireOOO = 1;
  let EndGroup = 1;
}
// Q-form multiply-accumulate fills the datapath and takes both lanes.
def A53WriteFMACQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 8;
  let RetireOOO = 1;
}
def A53ReadFPM   : SchedReadAdvance<0>;
def A53ReadFPMA4 : SchedReadAdvance<4, [A53WriteFMAC, A53WriteFMACIdx,
                                         A53WriteFMACQ]>;
// Divide and square root, measured on Cortex-A53: single precision 13 and 12
// cycles, double precision 22, with the unit busy for all but the last 3. The
// ASIMD forms take as long as the scalar ones.
def A53WriteFDivSP : SchedWriteRes<[A53UnitFPDS, A53UnitFPDS]> {
  let Latency = 13;
  let ReleaseAtCycles = [10, 10];
  let RetireOOO = 1;
}
def A53WriteFDivDP : SchedWriteRes<[A53UnitFPDS, A53UnitFPDS]> {
  let Latency = 22;
  let ReleaseAtCycles = [19, 19];
  let RetireOOO = 1;
}
def A53WriteFSqrtSP : SchedWriteRes<[A53UnitFPDS, A53UnitFPDS]> {
  let Latency = 12;
  let ReleaseAtCycles = [9, 9];
  let RetireOOO = 1;
}
def A53WriteFSqrtDP : SchedWriteRes<[A53UnitFPDS, A53UnitFPDS]> {
  let Latency = 22;
  let ReleaseAtCycles = [19, 19];
  let RetireOOO = 1;
}
// Reciprocal / reciprocal-square-root Newton-Raphson step (FRECPS/FRSQRTS) and
// estimate (FRECPE/FRSQRTE). Measured on Cortex-A53: the step is 8 cycles and
// the estimate 4, not the flat 6 of WriteVd. These run on the FP/ASIMD ALU pipe,
// so the Q-form variants take both lanes.
def A53WriteFRecpStep  : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 8;
  let RetireOOO = 1;
}
def A53WriteFRecpStepQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 8;
  let RetireOOO = 1;
}
def A53WriteFRecpEst   : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
def A53WriteFRecpEstQ  : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// Integer Advanced SIMD multiply-accumulate. The result latency is 4, but the
// MAC unit forwards one accumulate result straight into the next MAC's
// accumulator with no interlock, so a dependent accumulator chain recurs every
// cycle. Only the accumulator forwards; the multiplicands do not. The
// saturating doubling forms (SQDML[AS]L) do not use this path.
def A53WriteIMACD : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
def A53WriteIMACIdx : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 4;
  let RetireOOO = 1;
  let EndGroup = 1;
}
def A53WriteIMACQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
def A53ReadIMACM  : SchedReadAdvance<0>;
def A53ReadIMACA3 : SchedReadAdvance<3, [A53WriteIMACD, A53WriteIMACIdx,
                                         A53WriteIMACQ]>;

// Bitwise Advanced SIMD logic: 2 cycles to an early reader, 1 to a late one
// such as another logic operation (see A53ReadLate).
def A53WriteLogicD : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
def A53WriteLogicQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
// Scalar floating-point select, and the float-to-float register move. The
// move reads its source late: FMOV -> FMOV is 1 cycle, FMOV -> FADD 2. The
// select only issues as the younger instruction of a pair.
def A53WriteFSel : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 2;
  let RetireOOO = 1;
  let EndGroup = 1;
}
def A53WriteFMov : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
// Advanced SIMD arithmetic that runs at the same 4-cycle latency as its scalar
// counterpart: floating-point arithmetic and conversion, and the integer vector
// multiply. Q-form takes both datapath lanes.
def A53WriteVecD_4 : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
def A53WriteVecQ_4 : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 4;
  let RetireOOO = 1;
}
// The integer operations that only issue as the younger instruction of a pair:
// the by-element multiplies, the across-vector reductions and a few more,
// listed with the InstRWs.
def A53WriteVecYng_2 : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 2;
  let RetireOOO = 1;
  let EndGroup = 1;
}
def A53WriteVecYng_3 : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 3;
  let RetireOOO = 1;
  let EndGroup = 1;
}
// The accumulating operations (absolute-difference and pairwise accumulate, the
// rounding shift-accumulates) and the rounding add/subtract-narrow hold
// A53UnitFPRestricted for two cycles, so they issue one every two cycles. The
// 64-bit forms leave the other lane free for an unrestricted operation.
def A53WriteAccD : SchedWriteRes<[A53UnitFPALU, A53UnitFPRestricted]> {
  let Latency = 4;
  let ReleaseAtCycles = [1, 2];
  let RetireOOO = 1;
  let EndGroup = 1;
}
def A53WriteAccQ : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 2];
  let RetireOOO = 1;
}
// Integer Advanced SIMD ALU, not the flat 6 of WriteVd. Q-form takes both
// datapath lanes. Which ops use which latency is listed with the InstRWs.
def A53WriteIVecD_2 : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
def A53WriteIVecQ_2 : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 2;
  let RetireOOO = 1;
}
def A53WriteIVecD_3 : SchedWriteRes<[A53UnitFPALU]> {
  let Latency = 3;
  let RetireOOO = 1;
}
def A53WriteIVecQ_3 : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 3;
  let RetireOOO = 1;
}

// Operand reads. The FP/ASIMD pipeline reads its source operands at one of two
// stages. FP arithmetic, multiplies, compares, shifts, permutes and most of the
// wider integer operations read them early. Plain integer add/subtract (and
// their halving and saturating forms), bitwise logic, CMTST, CLZ, PMUL and the
// register moves (FMOV, and the FP-to-GPR transfer) read them one cycle later,
// so every result reaches them one cycle sooner. The write latencies in this
// model are the latencies to an early reader; a late reader takes a cycle off
// with A53ReadLate. So a chain of a late reader runs one cycle faster than its
// write latency: ADD -> ADD is 2 cycles, ADD -> FADD is 3.
def A53ReadLate  : SchedReadAdvance<1>;
def A53ReadEarly : SchedReadAdvance<0>;
// The accumulator of the 4-cycle accumulating ops (absolute-difference and
// pairwise accumulate, rounding shift-accumulate) is added in at the end: a
// chain through the accumulator alone recurs every 2 cycles.
def A53ReadAcc2  : SchedReadAdvance<2>;

// Cryptography. The crypto unit is 128 bits wide: every AES/SHA/PMULL op takes
// both FP/ASIMD datapath lanes for one cycle, so they issue one per cycle and
// never alongside another FP/ASIMD op. They read their operands late (see
// A53ReadLate), except the two non-tied sources of SHA1C/M/P, SHA256H/H2 and
// SHA256SU1, which are read early. Latencies to an early reader: AES, SHA1H,
// SHA1SU0/1 and the 64-bit PMULL 3; the 8-bit PMULL and SHA256SU0 4;
// SHA1C/M/P, SHA256H/H2 and SHA256SU1 6.
// AESMC/AESIMC fuse with the AESE/AESD that feeds them (FeatureFuseAES keeps
// the pair together): a chain of fused pairs recurs every 2 cycles, no more
// than AESE alone, so AESMC/AESIMC read an AESE/AESD result with no delay.
// (Any other producer reaches an AESMC/AESIMC one cycle sooner on hardware,
// which a single ReadAdvance cannot also express.)
// On hardware a fused pair also issues in a single cycle; here AESMC keeps its
// own datapath cycle, so independent fused pairs are modelled at one pair per
// two cycles rather than one per cycle.
def A53WriteAES : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 3;
  let RetireOOO = 1;
}
def A53WriteVecQ_6 : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 6;
  let RetireOOO = 1;
}
def A53WriteAESMC : SchedWriteRes<[A53UnitFPALU, A53UnitFPALU]> {
  let Latency = 3;
  let RetireOOO = 1;
}
def A53ReadAESMC : SchedReadAdvance<3, [A53WriteAES]>;

// Register transfers between a GPR and an FP/SIMD register (FMOV). Their
// latency depends on the consumer, not on the direction:
//  - FP-to-GPR: the result reaches an integer ALU consumer in 2 cycles, a MAC
//    or shifted operand in 3, i.e. one cycle later than an ALU result would.
//    It is therefore modelled like the ALU writes above (a nominal latency,
//    with the integer ReadAdvances below listing it), one cycle longer.
//  - GPR-to-FP: the result reaches an FP consumer in 2 cycles.
//  - Both transfers read their source late: any FP/ASIMD result reaches an
//    FP-to-GPR transfer one cycle early (A53ReadLate), an integer ALU result
//    can feed a GPR-to-FP transfer in the same cycle, and a transfer feeding
//    the opposite transfer takes a single cycle, so a GPR/FP round trip costs
//    2 cycles.
// Like the ALU latencies, the FP-to-GPR latency is nominal: consumers see the
// real value through ReadAdvance. Neither transfer is held back by, nor holds
// back, the in-order write-back of its neighbours, so that the late-read
// forwarding above is not lost to the nominal latencies.
let RetireOOO = 1 in {
def A53WriteF2R : SchedWriteRes<[A53UnitFPALU]> { let Latency = 4; }
def A53WriteR2F : SchedWriteRes<[A53UnitFPALU]> { let Latency = 2; }
}
def A53ReadR2F  : SchedReadAdvance<3, [WriteImm, WriteI,
                                       WriteISReg, WriteIEReg, WriteIS,
                                       WriteID32, WriteID64,
                                       WriteIM32, WriteIM64,
                                       WriteLD, A53WriteF2R]>;

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
                             WriteIM32,WriteIM64,A53WriteF2R]>;
def A53ReadShifted : SchedReadAdvance<1, [WriteImm,WriteI,
                                          WriteISReg, WriteIEReg,WriteIS,
                                          WriteID32,WriteID64,
                                          WriteIM32,WriteIM64,A53WriteF2R]>;
def A53ReadNotShifted : SchedReadAdvance<2, [WriteImm,WriteI,
                                             WriteISReg, WriteIEReg,WriteIS,
                                             WriteID32,WriteID64,
                                             WriteIM32,WriteIM64,A53WriteF2R]>;
def A53ReadISReg : SchedReadVariant<[
	SchedVar<RegShiftedPred, [A53ReadShifted]>,
	SchedVar<NoSchedPred, [A53ReadNotShifted]>]>;
def : SchedAlias<ReadISReg, A53ReadISReg>;

def A53ReadIEReg : SchedReadVariant<[
	SchedVar<RegExtendedPred, [A53ReadShifted]>,
	SchedVar<NoSchedPred, [A53ReadNotShifted]>]>;
def : SchedAlias<ReadIEReg, A53ReadIEReg>;

// MAC - Operands are generally needed one cycle later in the MAC pipe.
//       Accumulator operands are needed two cycles later.
def : ReadAdvance<ReadIM, 1, [WriteImm,WriteI,
                              WriteISReg, WriteIEReg,WriteIS,
                              WriteID32,WriteID64,
                              WriteIM32,WriteIM64,A53WriteF2R]>;
def : ReadAdvance<ReadIMA, 2, [WriteImm,WriteI,
                               WriteISReg, WriteIEReg,WriteIS,
                               WriteID32,WriteID64,
                               WriteIM32,WriteIM64,A53WriteF2R]>;

// Div
def : ReadAdvance<ReadID, 1, [WriteImm,WriteI,
                              WriteISReg, WriteIEReg,WriteIS,
                              WriteID32,WriteID64,
                              WriteIM32,WriteIM64,A53WriteF2R]>;

//===----------------------------------------------------------------------===//
// Subtarget-specific InstRWs.

//---
// Miscellaneous
//---
def : InstRW<[WriteI], (instrs COPY)>;

//---
// Vector Loads
//---
def : InstRW<[A53WriteVLD1], (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[A53WriteVLD1], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD1], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD2], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD3], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD4], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A53WriteVLD1], (instregex "LD1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD1], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD1], (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD3], (instregex "LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD4], (instregex "LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[A53WriteVLD1], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[A53WriteVLD1], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD2], (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[A53WriteVLD4], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A53WriteVLD1], (instregex "LD2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD1], (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD4], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A53WriteVLD2], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[A53WriteVLD2], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD4], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A53WriteVLD3], (instregex "LD3Threev2d$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD4], (instregex "LD3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD3], (instregex "LD3Threev2d_POST$")>;

def : InstRW<[A53WriteVLD2], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[A53WriteVLD2], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVLD5], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A53WriteVLD4], (instregex "LD4Fourv(2d)$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD2], (instregex "LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD5], (instregex "LD4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVLD4], (instregex "LD4Fourv(2d)_POST$")>;

//---
// Vector Stores
//---
def : InstRW<[A53WriteVST1], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[A53WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A53WriteVST2], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A53WriteVST1], (instregex "ST1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[A53WriteVST1], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[A53WriteVST1], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[A53WriteVST2], (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A53WriteVST1], (instregex "ST2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST1], (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A53WriteVST2], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[A53WriteVST3], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A53WriteVST2], (instregex "ST3Threev(2d)$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST3], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST3Threev(2d)_POST$")>;

def : InstRW<[A53WriteVST2], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[A53WriteVST3], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)$")>;
def : InstRW<[A53WriteVST2], (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST3], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s)_POST$")>;
def : InstRW<[WriteAdr, A53WriteVST2], (instregex "ST4Fourv(2d)_POST$")>;

//---
// Floating Point MAC, DIV, SQRT
//---
// The scalar forms take the addend as their third source operand, so only the
// third read forwards; the ASIMD forms accumulate into the tied destination,
// which is their first source operand.
def : InstRW<[A53WriteFMAC, A53ReadFPM, A53ReadFPM, A53ReadFPMA4],
      (instregex "^FN?M(ADD|SUB).*")>;
def : InstRW<[A53WriteFMAC, A53ReadFPMA4], (instregex "^FML(A|S)L.*")>;
def : InstRW<[A53WriteFMAC, A53ReadFPMA4],
      (instregex "^FML(A|S)(v2f32|v4f16)$")>;
def : InstRW<[A53WriteFMACIdx, A53ReadFPMA4],
      (instregex "^FML(A|S)v1i(16|32|64)_indexed$",
                 "^FML(A|S)(v2i32|v4i16)_indexed$")>;
def : InstRW<[A53WriteFMulIdx],
      (instregex "^FMULX?v1i(16|32|64)_indexed$",
                 "^FMULX?(v2i32|v4i16)_indexed$")>;
def : InstRW<[A53WriteFMulIdxQ],
      (instregex "^FMULX?(v4i32|v2i64|v8i16)_indexed$")>;
def : InstRW<[A53WriteFMACQ, A53ReadFPMA4],
      (instregex "^FML(A|S)(v4f32|v2f64|v8f16)$",
                 "^FML(A|S)(v4i32|v2i64|v8i16)_indexed$")>;
def : InstRW<[A53WriteFDivSP], (instrs FDIVSrr)>;
def : InstRW<[A53WriteFDivDP], (instrs FDIVDrr)>;
def : InstRW<[A53WriteFDivSP], (instregex "^FDIVv.*32$")>;
def : InstRW<[A53WriteFDivDP], (instregex "^FDIVv.*64$")>;
// Match only the genuine square root (FSQRT*), not the reciprocal-square-root
// step/estimate (FRSQRTS/FRSQRTE), which are handled below. Previously the
// ".*SQRT.*" globs charged those the full square-root latency.
def : InstRW<[A53WriteFSqrtSP], (instrs FSQRTSr)>;
def : InstRW<[A53WriteFSqrtDP], (instrs FSQRTDr)>;
def : InstRW<[A53WriteFSqrtSP], (instregex "^FSQRTv.*32$")>;
def : InstRW<[A53WriteFSqrtDP], (instregex "^FSQRTv.*64$")>;

// Reciprocal / rsqrt step and estimate (see the write classes above).
def : InstRW<[A53WriteFRecpStep],
      (instregex "^FR(ECP|SQRT)S(32|64|v2f32)$")>;
def : InstRW<[A53WriteFRecpStepQ],
      (instregex "^FR(ECP|SQRT)S(v2f64|v4f32)$")>;
def : InstRW<[A53WriteFRecpEst],
      (instregex "^FR(ECP|SQRT)E(v1i32|v1i64|v2f32)$")>;
def : InstRW<[A53WriteFRecpEstQ],
      (instregex "^FR(ECP|SQRT)E(v2f64|v4f32)$")>;
// The reciprocal-exponent and the integer reciprocal estimates share the
// 4-cycle estimate latency.
def : InstRW<[A53WriteFRecpEst],
      (instregex "^FRECPXv1i(32|64)$", "^UR(ECP|SQRT)Ev2i32$")>;
def : InstRW<[A53WriteFRecpEstQ], (instregex "^UR(ECP|SQRT)Ev4i32$")>;

// GPR <-> FP/SIMD register transfers (see the write classes above).
def : InstRW<[A53WriteF2R, A53ReadLate],
      (instrs FMOVSWr, FMOVDXr, FMOVDXHighr, FMOVHWr, FMOVHXr)>;
def : InstRW<[A53WriteR2F, A53ReadR2F],
      (instrs FMOVWSr, FMOVXDr, FMOVXDHighr, FMOVWHr, FMOVXHr)>;

// Floating-point ASIMD arithmetic, and the integer vector multiply, run at the
// same 4-cycle latency as the scalar forms (see the write classes above).
def : InstRW<[A53WriteVecD_4],
      (instregex "^F(ADD|SUB|ABD|MUL|MULX|MAX|MIN|MAXNM|MINNM)P?(v2f32|v4f16)$",
                 "^F(ABS|NEG)(v2f32|v4f16)$",
                 "^FRINT[AIMNPXZ](v2f32|v4f16)$",
                 "^([SU]CVTF|FCVT[AMNPZ][SU])(v1i16|v1i32|v1i64)$",
                 "^([SU]CVTF|FCVT[AMNPZ][SU])(v2f32|v4f16)$",
                 "^([SU]CVTF|FCVT[AMNPZ][SU])(v2i32|v4i16)_shift$",
                 "^FABD(16|32|64)$", "^FMULX(16|32|64)$",
                 "^(FCVTZ[SU]|[SU]CVTF)[hsd]$",
                 "^MULv(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteVecYng_4], (instregex "^MULv(2i32|4i16)_indexed$")>;
def : InstRW<[A53WriteVecYng_4],
      (instregex "^F(MAX|MIN|MAXNM|MINNM)Vv(4i16|4i32|8i16)v$", "^FCVTXNv1i64$")>;
def : InstRW<[A53WriteVecQ_4],
      (instregex "^F(ADD|SUB|ABD|MUL|MULX|MAX|MIN|MAXNM|MINNM)P?(v4f32|v2f64|v8f16)$",
                 "^F(ABS|NEG)(v4f32|v2f64|v8f16)$",
                 "^FRINT[AIMNPXZ](v4f32|v2f64|v8f16)$",
                 "^([SU]CVTF|FCVT[AMNPZ][SU])(v2f64|v4f32|v8f16)$",
                 "^([SU]CVTF|FCVT[AMNPZ][SU])(v2i64|v4i32|v8i16)_shift$",
                 "^FCVT(L|N)2?v(2i32|4i16|4i32|8i16)$",
                 "^FCVTXN2?v(2f32|4f32)$",
                 "^MULv(4i32|8i16|16i8)$", "^MULv(4i32|8i16)_indexed$")>;

// Bitwise ASIMD logic. It reads late, including the tied destination of the
// immediate ORR/BIC forms.
def : InstRW<[A53WriteLogicD, A53ReadLate, A53ReadLate],
      (instregex "^(AND|BIC|EOR|ORN|ORR|NOT)v8i8$", "^(ORR|BIC)v(2i32|4i16)$")>;
def : InstRW<[A53WriteLogicQ, A53ReadLate, A53ReadLate],
      (instregex "^(AND|BIC|EOR|ORN|ORR|NOT)v16i8$", "^(ORR|BIC)v(4i32|8i16)$")>;

// ASIMD compare, permute and insert/select: two cycles. The integer and
// floating-point compares, EXT/REV/DUP/INS/TBL and the bit-select family all
// measure 2 on the source-to-result path.
def : InstRW<[A53WriteIVecD_2],
      (instregex "^CM(EQ|GE|GT|HI|HS|LE|LT)v(1i64|2i32|4i16|8i8)(rz)?$",
                 "^(FCM(EQ|GE|GT)|FACG[ET])(v2f32|v4f16|v1i32|v1i64)(rz)?$",
                 "^FCM(EQ|GE|GT|LE|LT)(v2i32|v4i16)rz$",
                 "^FCM(LE|LT)v1i(16|32|64)rz$", "^FCM(EQ|GE|GT)v1i16rz$",
                 "^EXTv8i8$",
                 "^REV(16|32|64)v(2i32|4i16|8i8)$",
                 "^(FCM(EQ|GE|GT)|FACG[ET])(16|32|64)$",
                 "^DUPv.*gpr$", "^DUPv(2i32|4i16|8i8)lane$", "^DUPi(8|16|32|64)$",
                 "^INSv.*(8|16|32|64)(gpr|lane)$",
                 "^(MOVI|MVNI)(D|v8b_ns|v2i32|v2s_msl|v4i16)$",
                 "^FMOVv(2f32|4f16)_ns$",
                 "^[SU](MAX|MIN)P?v(2i32|4i16|8i8)$",
                 "^TB[LX]v8i8.*$",
                 "^(ZIP|TRN|UZP)[12]v(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteIVecQ_2],
      (instregex "^CM(EQ|GE|GT|HI|HS|LE|LT)v(2i64|4i32|8i16|16i8)(rz)?$",
                 "^(FCM(EQ|GE|GT)|FACG[ET])(v4f32|v2f64|v8f16)(rz)?$",
                 "^FCM(EQ|GE|GT|LE|LT)(v4i32|v2i64|v8i16)rz$",
                 "^EXTv16i8$",
                 "^(MOVI|MVNI)(v2d_ns|v16b_ns|v4i32|v4s_msl|v8i16)$",
                 "^FMOVv(4f32|2f64|8f16)_ns$",
                 "^[SU](MAX|MIN)P?v(4i32|8i16|16i8)$",
                 "^REV(16|32|64)v(4i32|8i16|16i8)$",
                 "^TB[LX]v16i8.*$",
                 "^(ZIP|TRN|UZP)[12]v(2i64|4i32|8i16|16i8)$")>;
// DUP of an element into a whole Q register only takes one lane.
def : InstRW<[A53WriteVecYng_2],
      (instregex "^DUPv(16i8|8i16|4i32|2i64)lane$")>;
// Absolute-difference accumulate, pairwise accumulate and the rounding
// add/subtract-narrow take four cycles from their source operands. The
// accumulator (the tied destination, i.e. the first read) is read late, see
// A53ReadAcc2.
def : InstRW<[A53WriteAccD, A53ReadAcc2],
      (instregex "^[SU]ABAv(2i32|4i16|8i8)$",
                 "^[SU]ADALPv(2i32_v1i64|4i16_v2i32|8i8_v4i16)$")>;
def : InstRW<[A53WriteAccQ, A53ReadAcc2],
      (instregex "^[SU]ABAv(16i8|4i32|8i16)$", "^[SU]ABALv",
                 "^[SU]ADALPv(16i8_v8i16|4i32_v2i64|8i16_v4i32)$")>;
def : InstRW<[A53WriteAccQ], (instregex "^R(ADD|SUB)HNv")>;
def : InstRW<[A53WriteFSel], (instregex "^FCSEL[HSD]rrr$")>;
def : InstRW<[A53WriteFMov, A53ReadLate], (instregex "^FMOV[HSD]r$")>;
// Bit select reads the tied destination early and the two data operands late.
def : InstRW<[A53WriteIVecD_2, A53ReadEarly, A53ReadLate, A53ReadLate],
      (instregex "^(BIF|BIT|BSL)v8i8$")>;
def : InstRW<[A53WriteIVecQ_2, A53ReadEarly, A53ReadLate, A53ReadLate],
      (instregex "^(BIF|BIT|BSL)v16i8$")>;

// CLS takes a cycle more than the plain compares. CMTST reads its operands
// late: 4 cycles to an early reader, 3 to a late one such as another CMTST.
def : InstRW<[A53WriteIVecD_3], (instregex "^CLSv(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteIVecQ_3], (instregex "^CLSv(4i32|8i16|16i8)$")>;
def : InstRW<[A53WriteVecD_4, A53ReadLate, A53ReadLate],
      (instregex "^CMTSTv(1i64|2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteVecQ_4, A53ReadLate, A53ReadLate],
      (instregex "^CMTSTv(2i64|4i32|8i16|16i8)$")>;

// ASIMD widening multiply and narrowing convert. The widening multiplies and
// multiply-accumulates, and the saturating narrows, are four cycles; the plain
// extract-narrow is two and the add/subtract-high-narrow three. The widening
// forms take both lanes. The narrows only take one, even when they read 128 bits
// or write the upper half of a Q register. The scalar SQDMULL/SQDML[AS]L,
// by-element or not, only issue as the younger instruction of a pair.
def : InstRW<[A53WriteVecQ_4],
      (instregex "^[SU]MULLv.*$",
                 "^SQDMULLv[248]i.*$",
                 "^SQDML[AS]Lv[248]i.*$")>;
def : InstRW<[A53WriteVecD_4], (instregex "^([SU]QXTN|SQXTUN)v.*$")>;
def : InstRW<[A53WriteVecYng_4],
      (instregex "^SQD(MULL|MLAL|MLSL)(i16|i32|v1i32_indexed|v1i64_indexed)$")>;
def : InstRW<[A53WriteIVecD_2], (instregex "^XTNv.*$")>;
// The widening add/subtract and absolute difference. The wide forms read their
// wide source late: a chain through it costs 2 cycles.
def : InstRW<[A53WriteIVecQ_3], (instregex "^[SU](ABDL|ADDL|SUBL)v.*$")>;
def : InstRW<[A53WriteIVecQ_3, A53ReadLate, A53ReadEarly],
      (instregex "^[SU](ADD|SUB)Wv.*$")>;
// Integer multiply-accumulate: the accumulator is the tied destination, so it
// is the first read and the only one that forwards.
def : InstRW<[A53WriteIMACD, A53ReadIMACA3],
      (instregex "^ML[AS]v(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteIMACIdx, A53ReadIMACA3],
      (instregex "^ML[AS]v(2i32|4i16)_indexed$")>;
def : InstRW<[A53WriteIMACQ, A53ReadIMACA3],
      (instregex "^ML[AS]v(4i32|8i16|16i8)$", "^ML[AS]v(4i32|8i16)_indexed$",
                 "^[SU]ML[AS]Lv.*$")>;
def : InstRW<[A53WriteIVecQ_3], (instregex "^(ADD|SUB)HNv.*$")>;

// ASIMD shifts. Plain shifts -- by immediate or by register, including the
// insert (SLI/SRI), narrowing (SHRN) and widening (SSHLL/USHLL) forms -- are
// two cycles. The rounding variants and the shift-accumulates are three, and
// the saturating shifts, the saturating narrows and the rounding
// shift-accumulates are four. The widening forms take both lanes, the narrowing
// forms only one. The insert and shift-accumulate forms
// read their tied destination late: one cycle late for SLI/SRI/[SU]SRA (a
// chain through it costs 1 and 2 cycles), two for [SU]RSRA (2 cycles).
def : InstRW<[A53WriteIVecD_2],
      (instregex "^SHL(d|v8i8_shift|v4i16_shift|v2i32_shift)$",
                 "^[SU]SHR(d|v8i8_shift|v4i16_shift|v2i32_shift)$",
                 "^[SU]SHLv(1i64|8i8|4i16|2i32)$")>;
def : InstRW<[A53WriteIVecQ_2],
      (instregex "^SHLv(16i8|8i16|4i32|2i64)_shift$",
                 "^[SU]SHRv(16i8|8i16|4i32|2i64)_shift$",
                 "^[SU]SHLv(16i8|8i16|4i32|2i64)$",
                 "^[SU]SHLLv.*_shift$")>;
def : InstRW<[A53WriteIVecD_2], (instregex "^SHRNv.*_shift$")>;
def : InstRW<[A53WriteIVecD_3],
      (instregex "^[SU]RSHR(d|v8i8_shift|v4i16_shift|v2i32_shift)$",
                 "^[SU]RSHLv(1i64|8i8|4i16|2i32)$")>;
def : InstRW<[A53WriteIVecQ_3],
      (instregex "^[SU]RSHRv(16i8|8i16|4i32|2i64)_shift$",
                 "^[SU]RSHLv(16i8|8i16|4i32|2i64)$")>;
def : InstRW<[A53WriteIVecD_3], (instregex "^RSHRNv.*_shift$")>;
def : InstRW<[A53WriteVecD_4],
      (instregex "^[SU]QSHLU?(b|h|s|d)$",
                 "^[SU]QR?SHL(b|h|s|d|v1i8|v1i16|v1i32|v1i64)$",
                 "^[SU]QSHLU?(v8i8|v4i16|v2i32)_shift$",
                 "^[SU]QR?SHLv(8i8|4i16|2i32)$")>;
def : InstRW<[A53WriteVecQ_4],
      (instregex "^[SU]QSHLU?v(16i8|8i16|4i32|2i64)_shift$",
                 "^[SU]QR?SHLv(16i8|8i16|4i32|2i64)$")>;
def : InstRW<[A53WriteVecD_4],
      (instregex "^([SU]QR?SHRN|SQR?SHRUN)[bhsd]?v?.*_shift$",
                 "^([SU]QR?SHRN|SQR?SHRUN)[bhs]$")>;
def : InstRW<[A53WriteIVecD_2, A53ReadLate, A53ReadEarly],
      (instregex "^(SLI|SRI)(d|v8i8_shift|v4i16_shift|v2i32_shift)$")>;
def : InstRW<[A53WriteIVecQ_2, A53ReadLate, A53ReadEarly],
      (instregex "^(SLI|SRI)v(16i8|8i16|4i32|2i64)_shift$")>;
def : InstRW<[A53WriteIVecD_3, A53ReadLate, A53ReadEarly],
      (instregex "^[SU]SRA(d|v8i8_shift|v4i16_shift|v2i32_shift)$")>;
def : InstRW<[A53WriteIVecQ_3, A53ReadLate, A53ReadEarly],
      (instregex "^[SU]SRAv(16i8|8i16|4i32|2i64)_shift$")>;
def : InstRW<[A53WriteAccD, A53ReadAcc2, A53ReadEarly],
      (instregex "^[SU]RSRA(d|v8i8_shift|v4i16_shift|v2i32_shift)$")>;
def : InstRW<[A53WriteAccQ, A53ReadAcc2, A53ReadEarly],
      (instregex "^[SU]RSRAv(16i8|8i16|4i32|2i64)_shift$")>;

// Integer ASIMD arithmetic (see the write classes above). Accumulate, widening
// and reduction-narrowing ops are intentionally left on the default here: their
// A53 measurements were less clear-cut than the sibling A55 model, so they are
// deferred rather than guessed. Plain add/subtract/negate, the halving forms
// and the saturating add/subtract/negate read their operands late: they are
// 3 (saturating: 4) cycles to an early reader and one cycle less to a late one.
def : InstRW<[A53WriteIVecD_3, A53ReadLate, A53ReadLate],
      (instregex "^(ADD|SUB|NEG)v(1i64|2i32|4i16|8i8)$",
                 "^[SU]R?HADDv(2i32|4i16|8i8)$", "^[SU]HSUBv(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteIVecQ_3, A53ReadLate, A53ReadLate],
      (instregex "^(ADD|SUB|NEG)v(2i64|4i32|8i16|16i8)$",
                 "^[SU]R?HADDv(8i16|4i32|16i8)$", "^[SU]HSUBv(8i16|4i32|16i8)$")>;
def : InstRW<[A53WriteIVecD_3],
      (instregex "^ABSv(1i64|2i32|4i16|8i8)$",
                 "^ADDPv(2i32|4i16|8i8)$",
                 "^[SU]ABDv(2i32|4i16|8i8)$",
                 "^[SU]ADDLPv(2i32_v1i64|4i16_v2i32|8i8_v4i16)$")>;
def : InstRW<[A53WriteVecYng_3],
      (instregex "^(ADDV|SADDLV|UADDLV)v", "^ADDPv2i64p$")>;
def : InstRW<[A53WriteVecD_4, A53ReadLate, A53ReadLate],
      (instregex "^([SU]QADD|[SU]QSUB|SQNEG)v(1i8|1i16|1i32|1i64|2i32|4i16|8i8)$",
                 "^SUQADDv(1i8|1i16|1i32|1i64|2i32|4i16|8i8)$",
                 "^USQADDv(1i16|1i32|1i64|2i32|4i16)$")>;
def : InstRW<[A53WriteVecYng_4, A53ReadLate, A53ReadLate],
      (instregex "^USQADDv(1i8|8i8)$")>;
def : InstRW<[A53WriteIVecQ_3],
      (instregex "^ABSv(2i64|4i32|8i16|16i8)$",
                 "^ADDPv(16i8|2i64|4i32|8i16)$",
                 "^[SU]ABDv(16i8|4i32|8i16)$",
                 "^[SU]ADDLPv(16i8_v8i16|4i32_v2i64|8i16_v4i32)$")>;
def : InstRW<[A53WriteVecQ_4, A53ReadLate, A53ReadLate],
      (instregex "^([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(16i8|2i64|4i32|8i16)$")>;

// The operations that were still inheriting the flat 6-cycle WriteVd/WriteVq.
// CNT/RBIT, the widening shift and the vector-to-GPR moves are two cycles; CLZ
// (3) and the polynomial multiply (4) read their operands late. The saturating
// multiply-high, the saturating absolute, the integer min/max reductions and
// the scalar floating-point pairwise reductions are four.
def : InstRW<[A53WriteIVecD_2],
      (instregex "^(CNT|RBIT)v8i8$",
                 "^[SU]MOVvi.*$")>;
def : InstRW<[A53WriteIVecQ_2],
      (instregex "^(CNT|RBIT)v16i8$",
                 "^SHLLv.*$")>;
def : InstRW<[A53WriteIVecD_3, A53ReadLate],
      (instregex "^CLZv(2i32|4i16|8i8)$")>;
def : InstRW<[A53WriteIVecQ_3, A53ReadLate],
      (instregex "^CLZv(4i32|8i16|16i8)$")>;
def : InstRW<[A53WriteVecD_4, A53ReadLate, A53ReadLate], (instrs PMULv8i8)>;
def : InstRW<[A53WriteVecQ_4, A53ReadLate, A53ReadLate], (instrs PMULv16i8)>;
def : InstRW<[A53WriteVecD_4],
      (instregex "^SQR?DMULH(v1i16|v1i32)$",
                 "^SQR?DMULHv(2i32|4i16)$",
                 "^SQABSv(1i8|1i16|1i32|1i64|2i32|4i16|8i8)$",
                 "^FADDPv2i(16|32|64)p$")>;
def : InstRW<[A53WriteVecYng_4],
      (instregex "^SQR?DMULHv(2i32|4i16|1i16|1i32)_indexed$",
                 "^[SU](MAX|MIN)Vv(4i16v|8i8v|16i8v|4i32v|8i16v)$",
                 "^F(MAX|MIN|MAXNM|MINNM)Pv2i(16|32|64)p$")>;
def : InstRW<[A53WriteVecQ_4],
      (instregex "^SQR?DMULHv(4i32|8i16)(_indexed)?$",
                 "^SQABSv(16i8|2i64|4i32|8i16)$")>;

// Cryptography (see the write classes above).
def : InstRW<[A53WriteAES, A53ReadLate, A53ReadLate], (instrs AESErr, AESDrr)>;
def : InstRW<[A53WriteAESMC, A53ReadAESMC],
      (instrs AESMCrr, AESIMCrr, AESMCrrTied, AESIMCrrTied)>;
def : InstRW<[A53WriteIVecQ_3, A53ReadLate, A53ReadLate, A53ReadLate],
      (instrs SHA1Hrr, SHA1SU0rrr, SHA1SU1rr, PMULLv1i64, PMULLv2i64)>;
def : InstRW<[A53WriteVecQ_4, A53ReadLate, A53ReadLate, A53ReadLate],
      (instrs SHA256SU0rr, PMULLv8i8, PMULLv16i8)>;
def : InstRW<[A53WriteVecQ_6, A53ReadLate, A53ReadEarly, A53ReadEarly],
      (instrs SHA1Crrr, SHA1Mrrr, SHA1Prrr, SHA256Hrrr, SHA256H2rrr,
              SHA256SU1rrr)>;

}
