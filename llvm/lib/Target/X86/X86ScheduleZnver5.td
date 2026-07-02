//=- X86ScheduleZnver5.td - X86 Znver5 Scheduling ------------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Znver5 to support instruction
// scheduling and other instruction cost heuristics.
// Based on:
//  * Software Optimization Guide for the AMD Zen5 Microarchitecture
//    https://docs.amd.com/v/u/en-US/58455_1.00
//===----------------------------------------------------------------------===//

def Znver5Model : SchedMachineModel {
  // AMD SOG, 2.9.7 Dispatch
  // The processor may dispatch up to 8 macro-ops per cycle
  // into the execution engine.
  let IssueWidth = 8;
  // AMD SOG, 2.10.3 Retire Control Unit
  // The retire control unit (RCU) tracks the completion status of all
  // outstanding operations (integer, load/store, and floating-point) and is
  // the final arbiter for exception processing and recovery.
  // The unit can receive up to 8 macro ops dispatched per cycle and track up
  // to 448 macro ops in-flight in non-SMT mode or 224 per thread in SMT mode.
  let MicroOpBufferSize = 448;
  // AMD SOG, 2.9.1 Op Cache
  // The op cache is organized as an associative cache with 64 sets and 16 ways.
  // At each set-way intersection is an entry containing up to 6 instructions or
  // fused instructions. This differs from Zen3 and Zen4 opcache entries, which
  // stored macro-ops, and can improve effective Op Cache storage density. The
  // maximum capacity of the op cache is 6 K instructions or fused instructions.
  // FIXME: not sure how to model this difference from Znver3/4

  // AMD SOG, 2.6.2 L1 Data Cache
  // The L1 data cache has a 4- or 5- cycle integer load-to-use latency.
  // The AGU and LS pipelines are optimized for simple address generation modes.
  // <...> and can achieve 4-cycle load-to-use integer load latency.
  let LoadLatency = 4;
  // The AGU and LS pipelines are optimized for simple address generation modes.
  // <...> and can achieve <...> 7-cycle load-to-use FP load latency.
  int VecLoadLatency = 7;
  // Latency of a simple store operation.
  int StoreLatency = 1;

  // Mean and median value for all instructions with latencies >6
  // Source: Zen5 Instruction Latencies spreadsheet (included with SOG)
  let HighLatency = 11;

  // The branch misprediction penalty is in the range from 12 to 18 cycles,
  // <...>. The common case penalty is 15 cycles.
  // AMD SOG, 2.8 Optimizing Branching
  // The common case penalty is 15 cycles.
  // Assuming a dispatch of 8 ops/cy and a mispredict cost of 15 from
  // the op-cache, we limit the loop buffer to 8*15 = 120 to avoid loop
  // unrolling leading to excessive filling of the op-cache from frontend.
  
  let LoopMicroOpBufferSize = 120;

  let PostRAScheduler = 1; // Enable Post RegAlloc Scheduler pass.

  let CompleteModel = 1;
}

let SchedModel = Znver5Model in {


//===----------------------------------------------------------------------===//
// RCU
//===----------------------------------------------------------------------===//

// AMD SOG, 2.10.3 Retire Control Unit
// The unit can receive up to 8 macro ops dispatched per cycle and track up to
// 448 macro ops in-flight in non-SMT mode or 224 per thread in SMT mode. <...>
// The retire unit handles in-order commit of up to eight macro ops per cycle.
def Zn5RCU : RetireControlUnit<Znver5Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Integer Execution Unit
//

// AMD SOG, 2.4 Superscalar Organization
// The processor uses four decoupled independent integer scheduler queues,
// each one servicing one ALU pipeline and one or two other pipelines

//
// Execution pipes
//===----------------------------------------------------------------------===//

// AMD SOG, 2.10.2 Execution Units

// The processor contains 6 general purpose integer execution pipes.
// Each pipe has an ALU capable of general-purpose integer operations.
def Zn5ALU0 : ProcResource<1>;
def Zn5ALU1 : ProcResource<1>;
def Zn5ALU2 : ProcResource<1>;
def Zn5ALU3 : ProcResource<1>;
def Zn5ALU4 : ProcResource<1>;
def Zn5ALU5 : ProcResource<1>;

// There is also a separate branch execution unit.
def Zn5BRU1 : ProcResource<1>;

// There are three Address Generation Units (AGUs) for all load and store
// address generation. There are also 3 store data movement units
// associated with the same schedulers as the AGUs.
def Zn5AGU0 : ProcResource<1>;
def Zn5AGU1 : ProcResource<1>;
def Zn5AGU2 : ProcResource<1>;
def Zn5AGU3 : ProcResource<1>;

//
// Execution Units
//===----------------------------------------------------------------------===//

// AMD SOG, 2.10.2 Execution Units

// ALU3 additionally has divide <...> execution capability.
defvar Zn5Divider = Zn5ALU3;

// ALU3,4,5 additionally have <...> branch execution capability.
defvar Zn5BRU0 = Zn5ALU3;
defvar Zn5BRU1 = Zn5ALU4;
defvar Zn5BRU2 = Zn5ALU5;

// Integer Multiplication issued on ALU1.
//defvar Zn5Multiplier = Zn5ALU1;
defvar Zn5MUL0 = Zn5ALU0;
defvar Zn5MUL1 = Zn5ALU1;
defvar Zn5MUL2 = Zn5ALU2;

// Execution pipeline grouping
//===----------------------------------------------------------------------===//

// AMD SOG, 2.10.2 Execution Units

// General ALU operations
// Simple bit twiddling: bit test, shift/rotate, bit extraction
// Zn5ALU0,1,2 can also handle CRC in addition to multiply
def Zn5ALU012 : ProcResGroup<[Zn5ALU0, Zn5ALU1, Zn5ALU2]>;

// Zn5ALU3,4,5 handle complex bit twiddling: PDEP/PEXT
def Zn5ALU345 : ProcResGroup<[Zn5ALU3, Zn5ALU4, Zn5ALU5]>;

def Zn5ALU0_5 : ProcResGroup<[Zn5ALU0, Zn5ALU1, Zn5ALU2, Zn5ALU3, Zn5ALU4, Zn5ALU5]>;

// General AGU operations
def Zn5AGU0123 : ProcResGroup<[Zn5AGU0, Zn5AGU1, Zn5AGU2, Zn5AGU3]>;

// Multipliers
def Zn5Multiplier : ProcResGroup<[Zn5MUL0, Zn5MUL1, Zn5MUL2]>;

// Control flow: jumps, calls
def Zn5BRU012 : ProcResGroup<[Zn5BRU0, Zn5BRU1, Zn5BRU2]>;

// Everything that isn't control flow, but still needs to access CC register,
// namely: conditional moves, SETcc.
def Zn5ALU03 : ProcResGroup<[Zn5ALU0, Zn5ALU3]>;

//
// Scheduling
//===----------------------------------------------------------------------===//

// The integer physical register file (PRF) consists of 240 registers.
def Zn5IntegerPRF : RegisterFile<240, [GR64, CCR], [1, 1], [1, 0],
                              8,  // Max moves that can be eliminated per cycle.
                              0>; // Restrict move elimination to zero regs.

// The integer scheduler has a 136 entry macro op capacity.

// AMD SOG, 2.10.2 Execution Units
// The schedulers can receive up to eight macro-ops per cycle.
// The ALU scheduler can issue six micro-ops per cycle and the AGU scheduler can
// issue four micro-ops per cycle.

def Zn5Int : ProcResGroup<[Zn5ALU0, Zn5ALU1, Zn5ALU2,         // scheduler 1
                           Zn5ALU3, Zn5ALU4, Zn5ALU5,         // scheduler 2
                           Zn5AGU0, Zn5AGU1, Zn5AGU2, Zn5AGU3 // scheduler 3
                          ]> {
  // There are two banks of the buffers with 136 macro op capacity.
  // FIXME: where is this value from?
  let BufferSize = 136;
}

//===----------------------------------------------------------------------===//
// Floating-Point Unit
//

// AMD SOG, 2.4 Superscalar Organization
// The processor uses <...> two decoupled independent floating point schedulers
// each servicing two FP pipelines; and one independent floating point scheduler
// servicing store data or FP-to-integer pipelines.

//
// Execution pipes
//===----------------------------------------------------------------------===//

// <...>, and six FPU pipes.
// Agner, 22.10 Floating point execution pipes
// There are six floating point/vector execution pipes,
def Zn5FP0  : ProcResource<1>;
def Zn5FP1  : ProcResource<1>;
def Zn5FP2  : ProcResource<1>;
def Zn5FP3  : ProcResource<1>;
def Zn5FP45 : ProcResource<2>;

//
// Execution Units
//===----------------------------------------------------------------------===//

// (v)FMUL*, (v)FMA*, Floating Point Compares, Blendv(DQ)
defvar Zn5FPFMul0 = Zn5FP0;
defvar Zn5FPFMul1 = Zn5FP1;

// (v)FADD*
// Complex VADD operations are not available in all pipes. (VADDPD etc)
defvar Zn5FPFAdd0 = Zn5FP2;
defvar Zn5FPFAdd1 = Zn5FP3;

// All convert operations except pack/unpack
defvar Zn5FPFCvt0 = Zn5FP2;
defvar Zn5FPFCvt1 = Zn5FP3;

// All Divide and Square Root except Reciprocal Approximation
// FDIV unit can support 2 simultaneous operations in flight
// even though it occupies a single pipe.
// FIXME: BufferSize=2 ?
defvar Zn5FPFDiv = Zn5FP1;

// Moves and Logical operations on Floating Point Data Types
defvar Zn5FPFMisc0 = Zn5FP0;
defvar Zn5FPFMisc1 = Zn5FP1;
defvar Zn5FPFMisc2 = Zn5FP2;
defvar Zn5FPFMisc3 = Zn5FP3;

// Integer Adds, Subtracts, and Compares
// Some complex VADD operations are not available in all pipes.
defvar Zn5FPVAdd0 = Zn5FP0;
defvar Zn5FPVAdd1 = Zn5FP1;
defvar Zn5FPVAdd2 = Zn5FP2;
defvar Zn5FPVAdd3 = Zn5FP3;

// Integer Multiplies, SAD, Blendvb
defvar Zn5FPVMul0 = Zn5FP0;
defvar Zn5FPVMul1 = Zn5FP1;
defvar Zn5FPVMul2 = Zn5FP3;

// Data Shuffles, Packs, Unpacks, Permute
// Some complex shuffle operations are only available in pipe1.
defvar Zn5FPVShuf = Zn5FP1;
defvar Zn5FPVShufAux = Zn5FP2;

// Bit Shift Left/Right operations
defvar Zn5FPVShift0 = Zn5FP1;
defvar Zn5FPVShift1 = Zn5FP2;

// Moves and Logical operations on Packed Integer Data Types
defvar Zn5FPVMisc0 = Zn5FP0;
defvar Zn5FPVMisc1 = Zn5FP1;
defvar Zn5FPVMisc2 = Zn5FP2;
defvar Zn5FPVMisc3 = Zn5FP3;

// *AES*
defvar Zn5FPAES0 = Zn5FP0;
defvar Zn5FPAES1 = Zn5FP1;

// *CLM*
defvar Zn5FPCLM0 = Zn5FP0;
defvar Zn5FPCLM1 = Zn5FP1;

// Execution pipeline grouping
//===----------------------------------------------------------------------===//

// AMD SOG, 2.11.1 Floating Point Execution Resources

// FLD use pipes 1 and 2.
def Zn5FPU12 : ProcResGroup<[Zn5FP1, Zn5FP2]>;

// (v)FMUL*, (v)FMA*, Floating Point Compares, Blendv(DQ)
def Zn5FPFMul01 : ProcResGroup<[Zn5FPFMul0, Zn5FPFMul1]>;

// (v)FADD*
// Some complex VADD operations are not available in all pipes.
def Zn5FPFAdd01 : ProcResGroup<[Zn5FPFAdd0, Zn5FPFAdd1]>;

// All convert operations except pack/unpack
def Zn5FPFCvt01 : ProcResGroup<[Zn5FPFCvt0, Zn5FPFCvt1]>;

// All Divide and Square Root except Reciprocal Approximation
// def Zn5FPFDiv : ProcResGroup<[Zn5FPFDiv]>;

// Moves and Logical operations on Floating Point Data Types
def Zn5FPFMisc0123 : ProcResGroup<[Zn5FPFMisc0, Zn5FPFMisc1, Zn5FPFMisc2, Zn5FPFMisc3]>;

// FIXUP and RANGE use FP01 pipelines
def Zn5FPFMisc01 : ProcResGroup<[Zn5FPFMisc0, Zn5FPFMisc1]>;
def Zn5FPFMisc12 : ProcResGroup<[Zn5FPFMisc1, Zn5FPFMisc2]>;
// SCALE instructions use FP23 pipelines
def Zn5FPFMisc23 : ProcResGroup<[Zn5FPFMisc2, Zn5FPFMisc3]>;
def Zn5FPFMisc123 : ProcResGroup<[Zn5FPFMisc1,Zn5FPFMisc2, Zn5FPFMisc3]>;

// Loads, Stores and Move to General Register (EX) Operations
// Stores and floating point to general purpose register transfer
// have 2 dedicated pipelines (pipe 5 and 6).
defvar Zn5FPLd01 = Zn5FP45;

// AMD SOG, 2.11 Floating-Point Unit

// FP stores are supported on two pipelines,
// but throughput is limited to one per cycle.
let Super = Zn5FP45 in
def Zn5FPSt : ProcResource<1>;

// Integer Adds, and Subtracts
def Zn5FPVAdd0123 : ProcResGroup<[Zn5FPVAdd0, Zn5FPVAdd1, Zn5FPVAdd2, Zn5FPVAdd3]>;
// Compares use first four pipes
def Zn5FPU0123 : ProcResGroup<[Zn5FP0, Zn5FP1, Zn5FP2, Zn5FP3]>;

def Zn5FPVAdd01: ProcResGroup<[Zn5FPVAdd0, Zn5FPVAdd1]>;
def Zn5FPVAdd12: ProcResGroup<[Zn5FPVAdd1, Zn5FPVAdd2]>;
def Zn5FPVAdd03: ProcResGroup<[Zn5FPVAdd0, Zn5FPVAdd3]>;

// AVX512 Opmask pipelines
def Zn5FPOpMask01: ProcResGroup<[Zn5FP0, Zn5FP3]>;
def Zn5FPOpMask4: ProcResGroup<[Zn5FP45]>;

// Integer Multiplies, SAD, Blendvb
def Zn5FPVMul012 : ProcResGroup<[Zn5FPVMul0, Zn5FPVMul1, Zn5FPVMul2]>;

// VNNIs execute in the first two pipes.
def Zn5FPVMul01 : ProcResGroup<[Zn5FPVMul0, Zn5FPVMul1]>;

// Data Shuffles, Packs, Unpacks, Permute
// Some complex shuffle operations are only available in pipe1.
def Zn5FPVShuf01 : ProcResGroup<[Zn5FPVShuf, Zn5FPVShufAux]>;

// Bit Shift Left/Right operations
def Zn5FPVShift01 : ProcResGroup<[Zn5FPVShift0, Zn5FPVShift1]>;

// Moves and Logical operations on Packed Integer Data Types
def Zn5FPVMisc0123 : ProcResGroup<[Zn5FPVMisc0, Zn5FPVMisc1, Zn5FPVMisc2, Zn5FPVMisc3]>;

// *AES*
def Zn5FPAES01 : ProcResGroup<[Zn5FPAES0, Zn5FPAES1]>;

// *CLM*
def Zn5FPCLM01 : ProcResGroup<[Zn5FPCLM0, Zn5FPCLM1]>;


//
// Scheduling
//===----------------------------------------------------------------------===//

// FIXME: not sure if this still holds for Znver5
// Agner, 21.8 Register renaming and out-of-order schedulers
// The floating point register file has 384 vector registers
// of 512b each in zen5.
def Zn5FpPRF : RegisterFile<384, [VR64, VR128, VR256, VR512], [1, 1, 1, 1], [0, 1, 1],
                            6,  // Max moves that can be eliminated per cycle.
                            0>; // Restrict move elimination to zero regs.

// AMD SOG, 2.11 Floating-Point Unit
// The floating-point scheduler has a 3*38 entry macro op capacity.
// <...> the scheduler can issue 1 micro op per cycle for each pipe.
// FIXME: those are two separate schedulers, not a single big one.
// FIXME: not sure if these values hold for Znver5
def Zn5FP : ProcResGroup<[Zn5FP0, Zn5FP2,          /*Zn5FP4,*/ // scheduler 0
                          Zn5FP1, Zn5FP3, Zn5FP45 /*Zn5FP5*/  // scheduler 1
                         ]> {
  let BufferSize = !mul(3, 38);
}

// AMD SOG, 2.11 Floating-Point Unit
// Macro ops can be dispatched to the 96 entry Non-Scheduling Queue (NSQ)
// even if floating-point scheduler is full.
// FIXME: how to model this properly?


//===----------------------------------------------------------------------===//
// Load-Store Unit
//

// The LS unit contains four largely independent pipe-lines
// enabling the execution of four memory operations per cycle.
def Zn5LSU : ProcResource<4>;

// All four memory operations can be loads.
let Super = Zn5LSU in
def Zn5Load : ProcResource<4> {
  // The LS unit can process up to 128 out-of-order loads.
  let BufferSize = 128;
}

def Zn5LoadQueue : LoadQueue<Zn5Load>;

// A maximum of two of the memory operations can be stores.
let Super = Zn5LSU in
def Zn5Store : ProcResource<2> {
  // The LS unit utilizes a 104-entry store queue (STQ).
  let BufferSize = 104;
}

def Zn5StoreQueue : StoreQueue<Zn5Store>;

//===----------------------------------------------------------------------===//
// Basic helper classes.
//===----------------------------------------------------------------------===//

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.

multiclass __Zn5WriteRes<SchedWrite SchedRW, list<ProcResourceKind> ExePorts,
                         int Lat = 1, list<int> Res = [], int UOps = 1> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }
}

multiclass __Zn5WriteResPair<X86FoldableSchedWrite SchedRW,
                             list<ProcResourceKind> ExePorts, int Lat,
                             list<int> Res, int UOps, int LoadLat, int LoadUOps,
                             ProcResourceKind AGU, int LoadRes> {
  defm : __Zn5WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;

  defm : __Zn5WriteRes<SchedRW.Folded,
                       !listconcat([AGU, Zn5Load], ExePorts),
                       !add(Lat, LoadLat),
                       !if(!and(!empty(Res), !eq(LoadRes, 1)),
                         [],
                         !listconcat([1, LoadRes],
                           !if(!empty(Res),
                             !listsplat(1, !size(ExePorts)),
                             Res))),
                       !add(UOps, LoadUOps)>;
}

// For classes without folded loads.
multiclass Zn5WriteResInt<SchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts, int Lat = 1,
                          list<int> Res = [], int UOps = 1> {
  defm : __Zn5WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

multiclass Zn5WriteResXMM<SchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts, int Lat = 1,
                          list<int> Res = [], int UOps = 1> {
  defm : __Zn5WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

multiclass Zn5WriteResYMM<SchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts, int Lat = 1,
                          list<int> Res = [], int UOps = 1> {
  defm : __Zn5WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

multiclass Zn5WriteResZMM<SchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts, int Lat = 1,
                          list<int> Res = [], int UOps = 1> {
  defm : __Zn5WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

// For classes with folded loads.
multiclass Zn5WriteResIntPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __Zn5WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           Znver5Model.LoadLatency,
                           LoadUOps, Zn5AGU0123, LoadRes>;
}

multiclass Zn5WriteResXMMPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __Zn5WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           Znver5Model.VecLoadLatency,
                           LoadUOps, Zn5FPLd01, LoadRes>;
}

multiclass Zn5WriteResYMMPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __Zn5WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           Znver5Model.VecLoadLatency,
                           LoadUOps, Zn5FPLd01, LoadRes>;
}

multiclass Zn5WriteResZMMPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 2,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __Zn5WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           Znver5Model.VecLoadLatency,
                           LoadUOps, Zn5FPLd01, LoadRes>;
}

//===----------------------------------------------------------------------===//
// Here be dragons.
//===----------------------------------------------------------------------===//

def : ReadAdvance<ReadAfterLd, Znver5Model.LoadLatency>;

def : ReadAdvance<ReadAfterVecLd, Znver5Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecXLd, Znver5Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecYLd, Znver5Model.VecLoadLatency>;

// There is 1 cycle of added latency for a result to cross
// from F to I or I to F domain.
def : ReadAdvance<ReadInt2Fpu, -1>;

// Instructions with both a load and a store folded are modeled as a folded
// load + WriteRMW.
defm : Zn5WriteResInt<WriteRMW, [Zn5AGU0123, Zn5Store], Znver5Model.StoreLatency, [1, 1], 0>;

// Loads, stores, and moves, not folded with other operations.
defm : Zn5WriteResInt<WriteLoad, [Zn5AGU0123, Zn5Load], !add(Znver5Model.LoadLatency, 1), [1, 1], 1>;

// Model the effect of clobbering the read-write mask operand of the GATHER operation.
// Does not cost anything by itself, only has latency, matching that of the WriteLoad,
defm : Zn5WriteResInt<WriteVecMaskedGatherWriteback, [], !add(Znver5Model.LoadLatency, 1), [], 0>;

def Zn5WriteMOVSlow : SchedWriteRes<[Zn5AGU0123, Zn5Load]> {
  let Latency = !add(Znver5Model.LoadLatency, 1);
  let ReleaseAtCycles = [3, 1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMOVSlow], (instrs MOV8rm, MOV8rm_NOREX, MOV16rm, MOVSX16rm16, MOVSX16rm32, MOVZX16rm16, MOVSX16rm8, MOVZX16rm8)>;

defm : Zn5WriteResInt<WriteStore, [Zn5AGU0123, Zn5Store], Znver5Model.StoreLatency, [1, 2], 1>;
defm : Zn5WriteResInt<WriteStoreNT, [Zn5AGU0123, Zn5Store], Znver5Model.StoreLatency, [1, 2], 1>;
defm : Zn5WriteResInt<WriteMove, [Zn5ALU0_5], 1, [4], 1>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

def Zn5WriteMOVBE16rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU0_5]> {
  let Latency = Znver5Model.LoadLatency;
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMOVBE16rm], (instrs MOVBE16rm)>;

def Zn5WriteMOVBEmr : SchedWriteRes<[Zn5ALU0_5, Zn5AGU0123, Zn5Store]> {
  let Latency = Znver5Model.StoreLatency;
  let ReleaseAtCycles = [4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteMOVBEmr], (instrs MOVBE16mr, MOVBE32mr, MOVBE64mr)>;

// Arithmetic.
defm : Zn5WriteResIntPair<WriteALU, [Zn5ALU0_5], 1, [1], 1>; // Simple integer ALU op.

def Zn5WriteALUSlow : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteALUSlow], (instrs ADD8i8, ADD16i16, ADD32i32, ADD64i32,
                                        AND8i8, AND16i16, AND32i32, AND64i32,
                                         OR8i8,  OR16i16,  OR32i32,  OR64i32,
                                        SUB8i8, SUB16i16, SUB32i32, SUB64i32,
                                        XOR8i8, XOR16i16, XOR32i32, XOR64i32)>;

def Zn5WriteMoveExtend : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMoveExtend], (instrs MOVSX16rr16, MOVSX16rr32, MOVZX16rr16, MOVSX16rr8, MOVZX16rr8)>;

def Zn5WriteMaterialize32bitImm: SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMaterialize32bitImm], (instrs MOV32ri, MOV32ri_alt, MOV64ri32)>;

def Zn5WritePDEP_PEXT : SchedWriteRes<[Zn5ALU345]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePDEP_PEXT], (instrs PDEP32rr, PDEP64rr,
                                          PEXT32rr, PEXT64rr)>;

defm : Zn5WriteResIntPair<WriteADC, [Zn5ALU0_5], 1, [4], 1>; // Integer ALU + flags op.

def Zn5WriteADC8mr_SBB8mr : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU0_5, Zn5Store]> {
  let Latency = 1;
  let ReleaseAtCycles = [1, 1, 7, 1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteADC8mr_SBB8mr], (instrs ADC8mr, SBB8mr)>;

// This is for simple LEAs with one or two input operands.
// The throughput of simple operand LEAs are max of ALUs.
// They get executed as single cycle macro-op on all ALUs.
defm : Zn5WriteResInt<WriteLEA, [Zn5ALU0_5], 1, [1], 1>;

// This write is used for slow LEA instructions.
// The SlowLEAs have lesser throughput and higher latency.
// They get executed on all ALUs as two macro-ops.
def Zn5Write3OpsLEA : SchedWriteRes<[Zn5AGU0123,Zn5ALU0_5]> {
  let Latency = 2;
  let ReleaseAtCycles = [1,1];
  let NumMicroOps = 2;
}

// On Znver5, a slow LEA is either a 3Ops LEA (base, index, offset),
// or an LEA with a `Scale` value different than 1.
def Zn5SlowLEAPredicate : MCSchedPredicate<
  CheckAny<[
    // A 3-operand LEA (base, index, offset).
    IsThreeOperandsLEAFn,
    // An LEA with a "Scale" different than 1.
    CheckAll<[
      CheckIsImmOperand<2>,
      CheckNot<CheckImmOperand<2, 1>>
    ]>
  ]>
>;

def Zn5WriteLEA : SchedWriteVariant<[
    SchedVar<Zn5SlowLEAPredicate, [Zn5Write3OpsLEA]>,
    SchedVar<NoSchedPred,         [WriteLEA]>
]>;

def : InstRW<[Zn5WriteLEA], (instrs LEA32r, LEA64r, LEA64_32r)>;

def Zn5SlowLEA16r : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 2;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 2;
}

def : InstRW<[Zn5SlowLEA16r], (instrs LEA16r)>;

// Integer multiplication
defm : Zn5WriteResIntPair<WriteIMul8, [Zn5Multiplier], 3, [3], 1>; // Integer 8-bit multiplication.
defm : Zn5WriteResIntPair<WriteIMul16, [Zn5Multiplier], 3, [3], 3, /*LoadUOps=*/1>; // Integer 16-bit multiplication.
defm : Zn5WriteResIntPair<WriteIMul16Imm, [Zn5Multiplier], 4, [4], 2>; // Integer 16-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul16Reg, [Zn5Multiplier], 3, [1], 1>; // Integer 16-bit multiplication by register.
defm : Zn5WriteResIntPair<WriteIMul32, [Zn5Multiplier], 3, [3], 2>;    // Integer 32-bit multiplication.
defm : Zn5WriteResIntPair<WriteMULX32, [Zn5Multiplier], 3, [1], 2>;    // Integer 32-bit Unsigned Multiply Without Affecting Flags.
defm : Zn5WriteResIntPair<WriteIMul32Imm, [Zn5Multiplier], 3, [1], 1>; // Integer 32-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul32Reg, [Zn5Multiplier], 3, [1], 1>; // Integer 32-bit multiplication by register.
defm : Zn5WriteResIntPair<WriteIMul64, [Zn5Multiplier], 3, [3], 2>;    // Integer 64-bit multiplication.
defm : Zn5WriteResIntPair<WriteMULX64, [Zn5Multiplier], 3, [1], 2>;    // Integer 32-bit Unsigned Multiply Without Affecting Flags.
defm : Zn5WriteResIntPair<WriteIMul64Imm, [Zn5Multiplier], 3, [1], 1>; // Integer 64-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul64Reg, [Zn5Multiplier], 3, [1], 1>; // Integer 64-bit multiplication by register.
defm : Zn5WriteResInt<WriteIMulHLd, [], !add(4, Znver5Model.LoadLatency), [], 0>;  // Integer multiplication, high part.
defm : Zn5WriteResInt<WriteIMulH, [], 4, [], 0>;  // Integer multiplication, high part.

defm : Zn5WriteResInt<WriteBSWAP32, [Zn5ALU0_5], 1, [1], 1>; // Byte Order (Endianness) 32-bit Swap.
defm : Zn5WriteResInt<WriteBSWAP64, [Zn5ALU0_5], 1, [1], 1>; // Byte Order (Endianness) 64-bit Swap.

// Latency numbers are not with exegesis measurements
// FIXME: Can we model CMPXCHG as microcoded instructions?!
defm : Zn5WriteResIntPair<WriteCMPXCHG, [Zn5ALU0_5], 3, [12], 5>; // Compare and set, compare and swap.

def Zn5WriteCMPXCHG8rr : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 3;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 3;
}
def : InstRW<[Zn5WriteCMPXCHG8rr], (instrs CMPXCHG8rr)>;

defm : Zn5WriteResInt<WriteCMPXCHGRMW, [Zn5ALU0_5], 3, [12], 6>;     // Compare and set, compare and swap.

def Zn5WriteCMPXCHG8rm_LCMPXCHG8 : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU0_5]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteCMPXCHG8rr.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = !add(Zn5WriteCMPXCHG8rr.NumMicroOps, 2);
}
def : InstRW<[Zn5WriteCMPXCHG8rm_LCMPXCHG8], (instrs CMPXCHG8rm, LCMPXCHG8)>;

def Zn5WriteCMPXCHG8B : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 9;
  let ReleaseAtCycles = [20];
  let NumMicroOps = 15;
}
def : InstRW<[Zn5WriteCMPXCHG8B], (instrs CMPXCHG8B)>;

def Zn5WriteCMPXCHG16B_LCMPXCHG16B : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 19;
  let ReleaseAtCycles = [57];
  let NumMicroOps = 26;
}
def : InstRW<[Zn5WriteCMPXCHG16B_LCMPXCHG16B], (instrs CMPXCHG16B, LCMPXCHG16B)>;

def Zn5WriteWriteXCHGUnrenameable : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteWriteXCHGUnrenameable], (instrs XCHG8rr, XCHG16rr, XCHG16ar)>;

// Add 3 cycle latency for the use with memory operand
def Zn5WriteXCHG8rm_XCHG16rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU0_5]> {
  let Latency = !add(Znver5Model.LoadLatency, 3);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 5;
}
def : InstRW<[Zn5WriteXCHG8rm_XCHG16rm], (instrs XCHG8rm, XCHG16rm)>;

def Zn5WriteXCHG32rm_XCHG64rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU0_5]> {
  let Latency = !add(Znver5Model.LoadLatency, 3);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteXCHG32rm_XCHG64rm], (instrs XCHG32rm, XCHG64rm)>;

// Integer division.
// Agner Fog Znver5 instruction table
defm : Zn5WriteResIntPair<WriteDiv8, [Zn5Divider], 10, [10], 2>;
defm : Zn5WriteResIntPair<WriteDiv16, [Zn5Divider], 11, [11], 2>;
defm : Zn5WriteResIntPair<WriteDiv32, [Zn5Divider], 13, [13], 2>;
defm : Zn5WriteResIntPair<WriteDiv64, [Zn5Divider], 16, [16], 2>;
defm : Zn5WriteResIntPair<WriteIDiv8, [Zn5Divider], 10, [10], 2>;
defm : Zn5WriteResIntPair<WriteIDiv16, [Zn5Divider], 11, [11], 2>;
defm : Zn5WriteResIntPair<WriteIDiv32, [Zn5Divider], 13, [13], 2>;
defm : Zn5WriteResIntPair<WriteIDiv64, [Zn5Divider], 16, [16], 2>;

// FIXME: For Znver4, the values given in the Instruction Latencies spreadsheet
// supplied with the SOG differ rather dramatically in terms of throughput with
// measured values from uops.info. With Znver5, I am using Agner Fog's instruction
// tables, which also differ from the values provided by AMD, but seem much more likely.

defm : Zn5WriteResIntPair<WriteBSF, [Zn5ALU1], 1, [1], 1, /*LoadUOps=*/1>; // Bit scan forward.
defm : Zn5WriteResIntPair<WriteBSR, [Zn5ALU1], 1, [1], 1, /*LoadUOps=*/1>; // Bit scan reverse.

defm : Zn5WriteResIntPair<WritePOPCNT, [Zn5ALU0_5], 1, [1], 1>; // Bit population count.

def Zn5WritePOPCNT16rr : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePOPCNT16rr], (instrs POPCNT16rr)>;

defm : Zn5WriteResIntPair<WriteLZCNT, [Zn5ALU0_5], 1, [1], 1>; // Leading zero count.

// Agner Fog Znver5 instruction table
def Zn5WriteLZCNT16rr : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteLZCNT16rr], (instrs LZCNT16rr)>;

defm : Zn5WriteResIntPair<WriteTZCNT, [Zn5ALU012], 1, [1], 1>; // Trailing zero count.

def Zn5WriteTZCNT16rr : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteTZCNT16rr], (instrs TZCNT16rr, TZCNT32rr, TZCNT64rr)>;

// FIXME: Seeing varying throughput for l,le,b variants
// cmovbe, cmovl, cmovle and cmovnbe, cmovnl, cmovnle
// Agner Fog Znver5 instruction table
defm : Zn5WriteResIntPair<WriteCMOV, [Zn5ALU0_5], 1, [1], 1>; // Conditional move.
defm : Zn5WriteResInt<WriteFCMOV, [Zn5ALU0_5], 7, [3], 3>;
defm : Zn5WriteResInt<WriteSETCC, [Zn5ALU0_5], 1, [1], 1>; // Set register based on condition code.
defm : Zn5WriteResInt<WriteSETCCStore, [Zn5ALU0_5, Zn5AGU0123, Zn5Store], 2, [2, 1, 1], 2>; // FIXME: Microcoded!
defm : Zn5WriteResInt<WriteLAHFSAHF, [Zn5ALU3], 1, [1], 1>; // Load/Store flags in AH.

//FIXME: Bit test memory variants are multiple ops!
defm : Zn5WriteResInt<WriteBitTest, [Zn5ALU012], 1, [1], 1>; // Bit Test
defm : Zn5WriteResInt<WriteBitTestImmLd, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 1), [1, 1, 1], 2>;
defm : Zn5WriteResInt<WriteBitTestRegLd, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 1), [1, 1, 1], 7>;

defm : Zn5WriteResInt<WriteBitTestSet, [Zn5ALU012], 2, [2], 2>; // Bit Test + Set
defm : Zn5WriteResInt<WriteBitTestSetImmLd, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 1], 4>;
defm : Zn5WriteResInt<WriteBitTestSetRegLd, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 1], 9>;

// Integer shifts and rotates.
defm : Zn5WriteResIntPair<WriteShift, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteShiftCL, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteRotate, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;

def Zn5WriteRotateR1 : SchedWriteRes<[Zn5ALU012]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteRotateR1], (instrs RCL8r1, RCL16r1, RCL32r1, RCL64r1,
                                         RCR8r1, RCR16r1, RCR32r1, RCR64r1)>;

def Zn5WriteRotateM1 : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteRotateR1.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteRotateR1.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteRotateM1], (instrs RCL8m1, RCL16m1, RCL32m1, RCL64m1,
                                         RCR8m1, RCR16m1, RCR32m1, RCR64m1)>;

def Zn5WriteRotateRightRI : SchedWriteRes<[Zn5ALU012]> {
  let Latency = 1;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[Zn5WriteRotateRightRI], (instrs RCR8ri, RCR16ri, RCR32ri, RCR64ri)>;

def Zn5WriteRotateRightMI : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteRotateRightRI.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = !add(Zn5WriteRotateRightRI.NumMicroOps, 3);
}
def : InstRW<[Zn5WriteRotateRightMI], (instrs RCR8mi, RCR16mi, RCR32mi, RCR64mi)>;

def Zn5WriteRotateLeftRI : SchedWriteRes<[Zn5ALU012]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteRotateLeftRI], (instrs RCL8ri, RCL16ri, RCL32ri, RCL64ri)>;

def Zn5WriteRotateLeftMI : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteRotateLeftRI.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = !add(Zn5WriteRotateLeftRI.NumMicroOps, 2);
}
def : InstRW<[Zn5WriteRotateLeftMI], (instrs RCL8mi, RCL16mi, RCL32mi, RCL64mi)>;

defm : Zn5WriteResIntPair<WriteRotateCL, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;

def Zn5WriteRotateRightRCL : SchedWriteRes<[Zn5ALU012]> {
  let Latency = 3;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[Zn5WriteRotateRightRCL], (instrs RCR8rCL, RCR16rCL, RCR32rCL, RCR64rCL)>;

def Zn5WriteRotateRightMCL : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteRotateRightRCL.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = !add(Zn5WriteRotateRightRCL.NumMicroOps, 2);
}
def : InstRW<[Zn5WriteRotateRightMCL], (instrs RCR8mCL, RCR16mCL, RCR32mCL, RCR64mCL)>;

def Zn5WriteRotateLeftRCL : SchedWriteRes<[Zn5ALU012]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 9;
}
def : InstRW<[Zn5WriteRotateLeftRCL], (instrs RCL8rCL, RCL16rCL, RCL32rCL, RCL64rCL)>;

def Zn5WriteRotateLeftMCL : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteRotateLeftRCL.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = !add(Zn5WriteRotateLeftRCL.NumMicroOps, 2);
}
def : InstRW<[Zn5WriteRotateLeftMCL], (instrs RCL8mCL, RCL16mCL, RCL32mCL, RCL64mCL)>;

// Double shift instructions.
// AMD SOG 2.10.2 Execution Units
// Shift instructions SHLD and SHRD have a three-cycle latency.
defm : Zn5WriteResInt<WriteSHDrri, [Zn5ALU012], 2, [3], 4>;
defm : Zn5WriteResInt<WriteSHDrrcl, [Zn5ALU012], 2, [3], 5>;
defm : Zn5WriteResInt<WriteSHDmri, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 4], 6>;
defm : Zn5WriteResInt<WriteSHDmrcl, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 4], 6>;

// BMI1 BEXTR/BLS, BMI2 BZHI
defm : Zn5WriteResIntPair<WriteBEXTR, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteBLS, [Zn5ALU0_5], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteBZHI, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
defm : Zn5WriteResInt<WriteZero, [Zn5ALU0_5], 0, [0], 1>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : Zn5WriteResIntPair<WriteJump, [Zn5BRU012], 1, [1], 1>; // FIXME: not from llvm-exegesis

//------  Floating point ------
// Floating point. This covers both scalar and vector operations.
defm : Zn5WriteResInt<WriteFLD0, [Zn5FPLd01, Zn5Load, Zn5FPU12], !add(Znver5Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : Zn5WriteResInt<WriteFLD1, [Zn5FPLd01, Zn5Load, Zn5FP1], !add(Znver5Model.LoadLatency, 7), [1, 1, 1], 1>;
defm : Zn5WriteResInt<WriteFLDC, [Zn5FPLd01, Zn5Load, Zn5FP1], !add(Znver5Model.LoadLatency, 7), [1, 1, 1], 1>;
defm : Zn5WriteResXMM<WriteFLoad, [Zn5FPLd01, Zn5Load, Zn5FPFMisc0123], !add(Znver5Model.VecLoadLatency, 1), [1, 1, 1], 1>;
defm : Zn5WriteResXMM<WriteFLoadX, [Zn5FPLd01, Zn5Load, Zn5FPFMisc0123], !add(Znver5Model.VecLoadLatency, 1), [1, 1, 1], 1>;
defm : Zn5WriteResYMM<WriteFLoadY, [Zn5FPLd01, Zn5Load, Zn5FPFMisc0123], !add(Znver5Model.VecLoadLatency, 1), [1, 1, 1], 1>;
defm : Zn5WriteResXMM<WriteFMaskedLoad, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteFMaskedLoadY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFStore, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

def Zn5WriteWriteFStoreMMX : SchedWriteRes<[Zn5FPSt, Zn5Store]> {
  let Latency = 1; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteWriteFStoreMMX], (instrs MOVHPDmr,  MOVHPSmr,
                                               VMOVHPDmr, VMOVHPSmr)>;

defm : Zn5WriteResXMM<WriteFStoreX, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteFStoreY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFStoreNT, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFStoreNTX, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteFStoreNTY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

defm : Zn5WriteResXMM<WriteFMaskedStore32, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [6, 1], 18>;
defm : Zn5WriteResXMM<WriteFMaskedStore64, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [4, 1], 10>;
defm : Zn5WriteResYMM<WriteFMaskedStore32Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [12, 1], 42>;
defm : Zn5WriteResYMM<WriteFMaskedStore64Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [6, 1], 18>;

defm : Zn5WriteResXMMPair<WriteFAdd, [Zn5FPFAdd01], 2, [1], 1>;  // Floating point add/sub.

def Zn5WriteX87Arith : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, 1); // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1, 1, 24];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteX87Arith], (instrs ADD_FI16m, ADD_FI32m,
                                         SUB_FI16m, SUB_FI32m,
                                         SUBR_FI16m, SUBR_FI32m,
                                         MUL_FI16m, MUL_FI32m)>;

def Zn5WriteX87Div : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, 1); // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1, 1, 62];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteX87Div], (instrs DIV_FI16m, DIV_FI32m,
                                       DIVR_FI16m, DIVR_FI32m)>;

defm : Zn5WriteResXMMPair<WriteFAddX, [Zn5FPFAdd01], 2, [1], 1>; // Floating point add/sub (XMM).
defm : Zn5WriteResYMMPair<WriteFAddY, [Zn5FPFAdd01], 2, [1], 1>; // Floating point add/sub (YMM).
defm : Zn5WriteResZMMPair<WriteFAddZ, [Zn5FPFAdd01], 2, [2], 1>; // Floating point add/sub (ZMM).
defm : Zn5WriteResXMMPair<WriteFAdd64, [Zn5FPFAdd01], 2, [1], 1>;  // Floating point double add/sub.
defm : Zn5WriteResXMMPair<WriteFAdd64X, [Zn5FPFAdd01], 2, [1], 1>; // Floating point double add/sub (XMM).
defm : Zn5WriteResYMMPair<WriteFAdd64Y, [Zn5FPFAdd01], 2, [1], 1>; // Floating point double add/sub (YMM).
defm : Zn5WriteResZMMPair<WriteFAdd64Z, [Zn5FPFAdd01], 2, [2], 1>; // Floating point double add/sub (ZMM).
defm : Zn5WriteResXMMPair<WriteFCmp, [Zn5FPFMul01], 2, [2], 1>;  // Floating point compare.
defm : Zn5WriteResXMMPair<WriteFCmpX, [Zn5FPFMul01], 2, [1], 1>; // Floating point compare (XMM).
defm : Zn5WriteResYMMPair<WriteFCmpY, [Zn5FPFMul01], 2, [1], 1>; // Floating point compare (YMM).
defm : Zn5WriteResZMMPair<WriteFCmpZ, [Zn5FPFMul01], 2, [2], 1>; // Floating point compare (ZMM).
defm : Zn5WriteResXMMPair<WriteFCmp64, [Zn5FPFMul01], 2, [1], 1>;  // Floating point double compare.
defm : Zn5WriteResXMMPair<WriteFCmp64X, [Zn5FPFMul01], 2, [1], 1>; // Floating point double compare (XMM).
defm : Zn5WriteResYMMPair<WriteFCmp64Y, [Zn5FPFMul01], 2, [1], 1>; // Floating point double compare (YMM).
defm : Zn5WriteResZMMPair<WriteFCmp64Z, [Zn5FPFMul01], 2, [2], 1>; // Floating point double compare (ZMM).
defm : Zn5WriteResXMMPair<WriteFCom, [Zn5FPFMul01], 3, [2], 1>; // Floating point compare to flags (X87).
defm : Zn5WriteResXMMPair<WriteFComX, [Zn5FPFMul01], 4, [2], 2>; // Floating point compare to flags (SSE).
defm : Zn5WriteResXMMPair<WriteFMul, [Zn5FPFMul01], 3, [1], 1>;  // Floating point multiplication.
defm : Zn5WriteResXMMPair<WriteFMulX, [Zn5FPFMul01], 3, [1], 1>; // Floating point multiplication (XMM).
defm : Zn5WriteResYMMPair<WriteFMulY, [Zn5FPFMul01], 3, [1], 1>; // Floating point multiplication (YMM).
defm : Zn5WriteResZMMPair<WriteFMulZ, [Zn5FPFMul01], 3, [2], 1>; // Floating point multiplication (ZMM).
defm : Zn5WriteResXMMPair<WriteFMul64, [Zn5FPFMul01], 3, [1], 1>;  // Floating point double multiplication.
defm : Zn5WriteResXMMPair<WriteFMul64X, [Zn5FPFMul01], 3, [1], 1>; // Floating point double multiplication (XMM).
defm : Zn5WriteResYMMPair<WriteFMul64Y, [Zn5FPFMul01], 3, [1], 1>; // Floating point double multiplication (YMM).
defm : Zn5WriteResZMMPair<WriteFMul64Z, [Zn5FPFMul01], 3, [2], 1>; // Floating point double multiplication (ZMM).
defm : Zn5WriteResXMMPair<WriteFDiv, [Zn5FPFDiv], 10, [3], 1>;  // Floating point division.
defm : Zn5WriteResXMMPair<WriteFDivX, [Zn5FPFDiv], 10, [3], 1>; // Floating point division (XMM).
defm : Zn5WriteResYMMPair<WriteFDivY, [Zn5FPFDiv], 10, [3], 1>; // Floating point division (YMM).
defm : Zn5WriteResZMMPair<WriteFDivZ, [Zn5FPFDiv], 10, [6], 1>; // Floating point division (ZMM).
defm : Zn5WriteResXMMPair<WriteFDiv64, [Zn5FPFDiv], 13, [5], 1>;  // Floating point double division.
defm : Zn5WriteResXMMPair<WriteFDiv64X, [Zn5FPFDiv], 13, [5], 1>; // Floating point double division (XMM).
defm : Zn5WriteResYMMPair<WriteFDiv64Y, [Zn5FPFDiv], 13, [5], 1>; // Floating point double division (YMM).
defm : Zn5WriteResZMMPair<WriteFDiv64Z, [Zn5FPFDiv], 13, [10], 1>; // Floating point double division (ZMM).
defm : Zn5WriteResXMMPair<WriteFSqrt, [Zn5FPFDiv], 14, [5], 1>;   // Floating point square root.
defm : Zn5WriteResXMMPair<WriteFSqrtX, [Zn5FPFDiv], 14, [5], 1>;  // Floating point square root (XMM).
defm : Zn5WriteResYMMPair<WriteFSqrtY, [Zn5FPFDiv], 14, [5], 1>;  // Floating point square root (YMM).
defm : Zn5WriteResZMMPair<WriteFSqrtZ, [Zn5FPFDiv], 14, [10], 1>;  // Floating point square root (ZMM).
defm : Zn5WriteResXMMPair<WriteFSqrt64, [Zn5FPFDiv], 20, [9], 1>;  // Floating point double square root.
defm : Zn5WriteResXMMPair<WriteFSqrt64X, [Zn5FPFDiv], 20, [9], 1>; // Floating point double square root (XMM).
defm : Zn5WriteResYMMPair<WriteFSqrt64Y, [Zn5FPFDiv], 20, [9], 1>; // Floating point double square root (YMM).
defm : Zn5WriteResZMMPair<WriteFSqrt64Z, [Zn5FPFDiv], 20, [17], 1>; // Floating point double square root (ZMM).
defm : Zn5WriteResXMMPair<WriteFSqrt80, [Zn5FPFDiv], 22, [23], 1>; // FIXME: latency not from llvm-exegesis  // Floating point long double square root.
defm : Zn5WriteResXMMPair<WriteFRcp, [Zn5FPFMul01], 4, [1], 1>;  // Floating point reciprocal estimate.
defm : Zn5WriteResXMMPair<WriteFRcpX, [Zn5FPFMul01], 4, [1], 1>; // Floating point reciprocal estimate (XMM).
defm : Zn5WriteResYMMPair<WriteFRcpY, [Zn5FPFMul01], 5, [1], 1>; // Floating point reciprocal estimate (YMM).
defm : Zn5WriteResZMMPair<WriteFRcpZ, [Zn5FPFMul01], 5, [2], 1>; // Floating point reciprocal estimate (ZMM).
defm : Zn5WriteResXMMPair<WriteFRsqrt, [Zn5FPFDiv], 4, [1], 1>;  // Floating point reciprocal square root estimate.
defm : Zn5WriteResXMMPair<WriteFRsqrtX, [Zn5FPFDiv], 4, [1], 1>; // Floating point reciprocal square root estimate (XMM).
defm : Zn5WriteResYMMPair<WriteFRsqrtY, [Zn5FPFDiv], 4, [1], 1>; // Floating point reciprocal square root estimate (YMM).
defm : Zn5WriteResZMMPair<WriteFRsqrtZ, [Zn5FPFDiv], 5, [2], 1>; // Floating point reciprocal square root estimate (ZMM).
defm : Zn5WriteResXMMPair<WriteFMA, [Zn5FPFMul01], 4, [2], 1>;  // Fused Multiply Add.
defm : Zn5WriteResXMMPair<WriteFMAX, [Zn5FPFMul01], 4, [1], 1>; // Fused Multiply Add (XMM).
defm : Zn5WriteResYMMPair<WriteFMAY, [Zn5FPFMul01], 4, [1], 1>; // Fused Multiply Add (YMM).
defm : Zn5WriteResZMMPair<WriteFMAZ, [Zn5FPFMul01], 4, [2], 1>; // Fused Multiply Add (ZMM).
defm : Zn5WriteResXMMPair<WriteDPPD, [Zn5FPFMul01], 7, [6], 3, /*LoadUOps=*/2>; // Floating point double dot product.
defm : Zn5WriteResXMMPair<WriteDPPS, [Zn5FPFMul01], 13, [5], 2, /*LoadUOps=*/2>; // Floating point single dot product.
defm : Zn5WriteResYMMPair<WriteDPPSY, [Zn5FPFMul01], 13, [5], 2, /*LoadUOps=*/1>; // Floating point single dot product (YMM).
defm : Zn5WriteResXMMPair<WriteFSign, [Zn5FPFMul01], 1, [2], 1>; // FIXME: latency not from llvm-exegesis  // Floating point fabs/fchs.
defm : Zn5WriteResXMMPair<WriteFRnd, [Zn5FPFCvt01], 3, [1], 1>; // Floating point rounding.
defm : Zn5WriteResYMMPair<WriteFRndY, [Zn5FPFCvt01], 3, [1], 1>; // Floating point rounding (YMM).
defm : Zn5WriteResZMMPair<WriteFRndZ, [Zn5FPFCvt01], 3, [2], 1>; // Floating point rounding (ZMM).

defm : Zn5WriteResXMMPair<WriteFLogic, [Zn5FPVMisc0123], 2, [1], 1>; // Floating point and/or/xor logicals.
defm : Zn5WriteResYMMPair<WriteFLogicY, [Zn5FPVMisc0123], 2, [1], 1>; // Floating point and/or/xor logicals (YMM).
defm : Zn5WriteResZMMPair<WriteFLogicZ, [Zn5FPVMisc0123], 2, [2], 1>; // Floating point and/or/xor logicals (ZMM).
defm : Zn5WriteResXMMPair<WriteFTest, [Zn5FPFMisc12], 1, [2], 2>; // FIXME: latency not from llvm-exegesis // Floating point TEST instructions.
defm : Zn5WriteResYMMPair<WriteFTestY, [Zn5FPFMisc12], 1, [2], 2>; // FIXME: latency not from llvm-exegesis // Floating point TEST instructions (YMM).
defm : Zn5WriteResXMMPair<WriteFShuffle, [Zn5FPVShuf01], 2, [2], 1>; // Floating point vector shuffles.
defm : Zn5WriteResYMMPair<WriteFShuffleY, [Zn5FPVShuf01], 2, [2], 1>; // Floating point vector shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteFShuffleZ, [Zn5FPVShuf01], 2, [2], 1>; // Floating point vector shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteFVarShuffle, [Zn5FPVShuf01], 3, [1], 1>; // Floating point vector variable shuffles.
defm : Zn5WriteResYMMPair<WriteFVarShuffleY, [Zn5FPVShuf01], 3, [1], 1>; // Floating point vector variable shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteFVarShuffleZ, [Zn5FPVShuf01], 3, [2], 1>; // Floating point vector variable shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteFBlend, [Zn5FPFMul01], 2, [1], 1>; // Floating point vector blends.
defm : Zn5WriteResYMMPair<WriteFBlendY, [Zn5FPFMul01], 2, [1], 1>; // Floating point vector blends (YMM).
defm : Zn5WriteResZMMPair<WriteFBlendZ, [Zn5FPFMul01], 2, [2], 1>; // Floating point vector blends (ZMM).
defm : Zn5WriteResXMMPair<WriteFVarBlend, [Zn5FPFMul01], 2, [1], 1>; // Fp vector variable blends.
defm : Zn5WriteResYMMPair<WriteFVarBlendY, [Zn5FPFMul01], 2, [1], 1>; // Fp vector variable blends (YMM).
defm : Zn5WriteResZMMPair<WriteFVarBlendZ, [Zn5FPFMul01], 2, [2], 1>; // Fp vector variable blends (ZMM).

// Horizontal Add/Sub (float and integer)
defm : Zn5WriteResXMMPair<WriteFHAdd, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPFAdd01], 5, [1, 1, 2], 4>;
defm : Zn5WriteResYMMPair<WriteFHAddY, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPFAdd01], 5, [1, 1, 2], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResZMMPair<WriteFHAddZ, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPFAdd01], 5, [1, 1, 2], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResXMMPair<WritePHAdd, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPVAdd0123], 4, [1, 1, 1], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResXMMPair<WritePHAddX, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPVAdd0123], 4, [1, 1, 1], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResYMMPair<WritePHAddY, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPVAdd0123], 4, [1, 1, 1], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResZMMPair<WritePHAddZ, [Zn5FPVAdd0123,Zn5FPVAdd0123,Zn5FPVAdd0123], 4, [1, 1, 1], 3, /*LoadUOps=*/1>;

// Vector integer operations.
defm : Zn5WriteResXMM<WriteVecLoad, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecLoadX, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecLoadY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecLoadNT, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecLoadNTY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecMaskedLoad, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecMaskedLoadY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecStore, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecStoreX, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

// VPDPs
def Zn5WriteVPDPrr : SchedWriteRes<[Zn5FPFAdd01]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteVPDPrr], (instrs VPDPBUSDSrr, VPDPBUSDrr, VPDPWSSDSrr, VPDPWSSDrr
                               )>;

def Zn5WriteFAdd64Xrrk : SchedWriteRes<[Zn5FPFAdd01]> {
  let Latency = 3;
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteFAdd64Xrrk], (instrs VADDPDZ128rrk, VSUBPDZ128rrk, VADDSDZrrbk_Int,
                                    VADDSDZrrk_Int, VSUBSDZrrbk_Int, VSUBSDZrrk_Int,
                                    VADDPSZ128rrk, VSUBPSZ128rrk
                                   )>;

def Zn5WriteFAdd64SSrrk : SchedWriteRes<[Zn5FPFMisc23]> {
  let Latency = 3;
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteFAdd64SSrrk], (instregex "V(ADD|SUB)(PS|SS|PD)(Z|Z256)rrb?k(_Int)?$")>;

def Zn5WriteShufExtractS : SchedWriteRes<[Zn5FPFMisc0]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteShufExtractS], (instrs VEXTRACTF128rri, VEXTRACTI128rri,
                                                VEXTRACTI32X4Z256rri, VEXTRACTI32X4Z256rrik,
                                                VEXTRACTI32X4Z256rrikz, VEXTRACTI64X2Z256rri,
                                                VEXTRACTI64X2Z256rrik, VEXTRACTI64X2Z256rrikz,
                                                VEXTRACTF32X4Z256rri, VEXTRACTF32X4Z256rri,
                                                VEXTRACTF32X4Z256rrik, VEXTRACTF32X4Z256rrikz,
                                                VEXTRACTF64X2Z256rri, VEXTRACTF64X2Z256rrik,
                                                VEXTRACTF64X2Z256rrikz, VSHUFF32X4Z256rri,
                                                VSHUFF32X4Z256rrik, VSHUFF32X4Z256rrikz,
                                                VSHUFF64X2Z256rri, VSHUFF64X2Z256rrik,
                                                VSHUFF64X2Z256rrikz, VSHUFI32X4Z256rri,
                                                VSHUFI32X4Z256rrik, VSHUFI32X4Z256rrikz,
                                                VSHUFI64X2Z256rri, VSHUFI64X2Z256rrik,
                                                VSHUFI64X2Z256rrikz)>;

def Zn5WriteVEXTRACTI128mr : SchedWriteRes<[Zn5FPFMisc0, Zn5FPSt, Zn5Store]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteShufExtractS.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = !add(Zn5WriteShufExtractS.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteVEXTRACTI128mr], (instrs VEXTRACTI128mri, VEXTRACTF128mri)>;

def Zn5WriteVINSERTF128rmr : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPFMisc0]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteShufExtractS.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = !add(Zn5WriteShufExtractS.NumMicroOps, 0);
}
def : InstRW<[Zn5WriteVINSERTF128rmr], (instrs VINSERTF128rmi)>;

def Zn5WriteShufExtract256: SchedWriteRes<[Zn5FPFMisc0]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteShufExtract256], (instrs
        VEXTRACTI32X8Zrri, VEXTRACTI32X8Zrrik,
        VEXTRACTI32X8Zrrikz, VEXTRACTI64X4Zrri,
        VEXTRACTI64X4Zrrik, VEXTRACTI64X4Zrrikz,
        VPERMQZ256ri, VPERMQZ256rik,
        VPERMQZ256rikz, VPERMQZri,
        VPERMQZrik, VPERMQZrikz,
        VEXTRACTF32X8Zrri, VEXTRACTF32X8Zrrik,
        VEXTRACTF32X8Zrrikz, VEXTRACTF64X4Zrri,
        VEXTRACTF64X4Zrrik, VEXTRACTF64X4Zrrikz,
        VPERMPDZ256ri, VPERMPDZ256rik,
        VPERMPDZ256rikz, VPERMPDZri,
        VPERMPDZrik, VPERMPDZrikz
    )>;

def Zn5WriteShufExtractrri: SchedWriteRes<[Zn5FPFMisc0]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteShufExtractrri], (instrs
        VEXTRACTI32X4Zrri, VEXTRACTI32X4Zrrik,
        VEXTRACTI32X4Zrrikz, VEXTRACTI64X2Zrri,
        VEXTRACTI64X2Zrrik, VEXTRACTI64X2Zrrikz,
        VEXTRACTF32X4Zrri, VEXTRACTF32X4Zrrik,
        VEXTRACTF32X4Zrrikz, VEXTRACTF64X2Zrri,
        VEXTRACTF64X2Zrrik, VEXTRACTF64X2Zrrikz,
        VSHUFF32X4Zrri, VSHUFF32X4Zrrik, VSHUFF32X4Zrrikz,
        VSHUFF64X2Zrri, VSHUFF64X2Zrrik, VSHUFF64X2Zrrikz,
        VSHUFI32X4Zrri, VSHUFI32X4Zrrik, VSHUFI32X4Zrrikz,
        VSHUFI64X2Zrri, VSHUFI64X2Zrrik, VSHUFI64X2Zrrikz
    )>;

defm : Zn5WriteResYMM<WriteVecStoreY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecStoreNT, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecStoreNTY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

// Mask Moves are no more microcoded and are single cycle moves with appropriate load/store latencies
defm : Zn5WriteResXMM<WriteVecMaskedStore32, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecMaskedStore64, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecMaskedStore32Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecMaskedStore64Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

defm : Zn5WriteResXMM<WriteVecMoveToGpr, [Zn5FPLd01], 1, [2], 1>;
defm : Zn5WriteResXMM<WriteVecMoveFromGpr, [Zn5FPLd01], 1, [2], 1>;

// Slow Moves
def Zn5MovSlow : SchedWriteRes<[Zn5FPLd01, Zn5FPFMisc0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5MovSlow], (instrs MMX_MOVDQ2Qrr)>;

// The movz which are not zero cycle moves
def Zn5WriteVecMovesZ : SchedWriteRes<[Zn5FPFMisc0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecMovesZ], (instregex
        "VMOV(A|U)(PS|PD)(Z|Z128|Z256)(rrk|rrk_REV|rrkz)",
        "VMOVDQ(A|U)(32|64)(Z|Z128|Z256)(rrk|rrk_REV|rrkz)",
        "VMOVDQU(8|16)(Z|Z128|Z256)(rrk|rrk_REV|rrkz)"
  )>;

// 3 cycle latency
// def Zn5WriteMOVMMX : SchedWriteRes<[Zn5FPLd01, Zn5FPFMisc0123]> {
//   let Latency = 3;
//   let ReleaseAtCycles = [1, 2];
//   let NumMicroOps = 2;
// }
// def : InstRW<[Zn5WriteMOVMMX], (instrs MMX_MOVQ2DQrr)>;

def Zn5WriteMOVMMXSlow : SchedWriteRes<[Zn5FPLd01, Zn5FPFMisc0123]> {
  let Latency = 1;
  let ReleaseAtCycles = [1, 4];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteMOVMMXSlow], (instrs MMX_MOVD64rr, MMX_MOVD64to64rr)>;

// MMX ALUs
def Zn5WriteALUMMX : SchedWriteRes<[Zn5FPFMisc0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteALUMMX], (instrs MMX_PSUBBrr)>;

// MMX CVTs
def Zn5MMX_CVTPD2PIrr : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MMX_CVTPD2PIrr], (instrs MMX_CVTPD2PIrr,
                                          MMX_CVTPI2PDrr,
                                          MMX_CVTPS2PIrr,
                                          MMX_CVTTPS2PIrr)>;

def Zn5WriteCVTDQZrrk : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteCVTDQZrrk], (instregex
        "VCVT(U)?DQ2PD(Z|Z256)rrk$",
        "VCVTNE(2)?PS2BF16Z256rrk$",
        "VCVTPD2(U)?DQZ256rrk$",
        "VCVTP(H|D)2PSZ256rrk$",
        "VCVTPH2PSZ(rrbk|rrk)$",
        "VCVTPS2P(H|D)Z256rrk$",
        "VCVTPS2PDZ(rrk|rrbk)$",
        "VCVT(U)?QQ2PDZrr(b|k|kz|bk|bkz)?$"
    )>;

def Zn5WriteCVTDQZrrbk : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteCVTDQZrrbk], (instregex
        "VCVT(T)?PD2(U)?DQZrrbk$",
        "VCVT(T)?PD2(U)?DQZ256rrbk$",
        "VCVTNE(2)?PS2BF16Zrrk$",
        "VCVT(T)?PD2(U)?DQZrrk$",
        "VCVTP(S|D)2P(H|S)Z(rrk|rrbk)$",
        "VCVT(QQ|UQQ)2PSZ(rrk|rrbk)$"
    )>;

def Zn5WriteCVTDQZ : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteCVTDQZ], (instregex
        "VCVT(U)?DQ2PDZ256(rr|rrkz)$",
        "VCVT(PS|DQ)2PDYrr",
        "VCVTPH2PS(Y|Z256)(rr|rrkz)$",
        "VCVTPS2PDZ256(rr|rrkz)$",
        "VCVT(T)?PS2(QQ|UQQ)Z256(rr|rrkz)$"
    )>;

def Zn5WriteCVTDQZrrkz : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteCVTDQZrrkz], (instregex
        "VCVT(T)?PD2(U)?DQZ256(rr|rrkz)$",
        "VCVT(T)?PD2DQYrr",
        "VCVTPH2PSZ(rr|rrb|rrbkz|rrkz)$",
        "VCVTNE2PS2BF16Z256rrkz$",
        "VCVT(T)?PS2(QQ|UQQ)Z(rr|rrb|rrkz|rrbkz)$",
        "VCVT(QQ|UQQ)2PSZ256(rr|rrkz)$"
    )>;

def Zn5WriteCVTZrrkz : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 7;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteCVTZrrkz], (instregex
        "VCVTNE(2)?PS2BF16Zrrkz$",
        "VCVT(T)?PD2(U)?DQZrr$",
        "VCVT(T)?PD2(U)?DQZ(rrb|rrbkz|rrkz)$",
        "VCVTPS2PHZ(rr|rrb|rrkz|rrbkz)$",
        "VCVT(QQ|UQQ)2PSZrr(b|bkz|kz)?$"
    )>;

def Zn5WriteShuffleSlow : SchedWriteRes<[Zn5FPVShuf01]> {  // MMX
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteShuffleSlow], (instrs MMX_PACKSSDWrr, MMX_PACKSSWBrr
                                    )>;

// KSHIFTS
def Zn5WriteKSHIFT : SchedWriteRes<[Zn5FPOpMask01]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteKSHIFT], (instrs KSHIFTLBki, KSHIFTLDki, KSHIFTLQki,
                                       KSHIFTLWki, KSHIFTRBki, KSHIFTRDki,
                                       KSHIFTRQki, KSHIFTRWki)>;

// Vector integer ALU op, no logicals.
defm : Zn5WriteResXMMPair<WriteVecALU, [Zn5FPVAdd0123], 1, [1], 1>;

def Zn5WriteEXTRQ_INSERTQ : SchedWriteRes<[Zn5FPVShuf01, Zn5FPLd01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteEXTRQ_INSERTQ], (instrs EXTRQ, INSERTQ)>;

def Zn5WriteEXTRQI_INSERTQI : SchedWriteRes<[Zn5FPVShuf01, Zn5FPLd01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteEXTRQI_INSERTQI], (instrs EXTRQI, INSERTQI)>;

// Vector integer ALU op, no logicals (XMM).
defm : Zn5WriteResXMMPair<WriteVecALUX, [Zn5FPVAdd0123], 2, [1], 1>;

def Zn5WriteVecALUXSlow : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUXSlow], (instrs GF2P8MULBrr, VGF2P8MULBrr)>;

def Zn5WriteVecALUZSlow : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZSlow], (instrs
        VPMULTISHIFTQBZrr, VPMULTISHIFTQBZrrk, VPMULTISHIFTQBZrrkz,
        VPSHUFBITQMBZ128rrk, VPMULTISHIFTQBZ256rrk, VPMULTISHIFTQBZ256rrkz,
        VPMULTISHIFTQBZ256rr, VPMULTISHIFTQBZ128rrk, VPMULTISHIFTQBZ128rrkz,
        VPMULTISHIFTQBZ128rr
    )>;

def Zn5WriteVecALUZSlowVPSHUFBITQMB : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZSlowVPSHUFBITQMB], (instrs VPSHUFBITQMBZ256rrk)>;

def Zn5WriteVecALUZSlowCMPs : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZSlowCMPs], (instregex
        "VPCMP(U)?(B|D|Q|W)Zrrik$", "VPCMP(EQ|GT)(B|D|Q|W)Zrrk",
        "VCMP(PS|PD)Z(rrik|rribk)$", "VFPCLASS(PS|PD)Zrik$"
    )>;

def Zn5WriteVecALUZCMPrri : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZCMPrri], (instregex
        "VPCMP(U)?(B|D|Q|W)Z128rri$",
        "VCMPPS(Z128|Z256)rri$", "VFPCLASSPSZ128ri$",
        "VFPCLASSPDZ256ri$",     "VCMPPDZrrib$"
   )>;

def Zn5WriteVecALUZSlowCMPs_128 : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZSlowCMPs_128], (instregex
        "VPCMP(U)?(B|D|Q|W)Z128rrik$", "VPCMP(EQ|GT)(B|D|Q|W)Z128rrk",
        "VCMP(PD|PS)Z128rrik$", "VFPCLASS(PD|PS)Z128rik$",
        "VCMPS(S|D)Zrri(b)?k_Int", "VFPCLASSS(S|D)Zrik", "VCMPSDZrri_Int"
    )>;

def Zn5WriteVecALUZSlowCMPs_256 : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUZSlowCMPs_256], (instregex
        "VPCMP(U)?(B|D|Q|W)Z256rrik$", "VPCMP(EQ|GT)(B|D|Q|W)Z256rrk",
        "VCMP(PD|PS)Z256rrik$", "VFPCLASS(PD|PS)Z256rik$"
    )>;

def Zn5WriteVecOpMask : SchedWriteRes<[Zn5FPOpMask01]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecOpMask], (instrs
        KADDBkk, KADDDkk, KADDQkk, KADDWkk,
        KANDBkk, KANDDkk, KANDQkk, KANDWkk,
        KANDNBkk, KANDNDkk, KANDNQkk, KANDNWkk,
        KMOVBkk, KMOVDkk, KMOVQkk, KMOVWkk,
        KMOVBrk, KMOVDrk, KMOVQrk, KMOVWrk,
        KNOTBkk, KNOTDkk, KNOTQkk, KNOTWkk,
        KORBkk, KORDkk, KORQkk, KORWkk,
        KORTESTBkk, KORTESTDkk, KORTESTQkk, KORTESTWkk,
        KTESTBkk, KTESTDkk, KTESTQkk, KTESTWkk,
        KUNPCKBWkk, KUNPCKDQkk, KUNPCKWDkk,
        KXNORBkk, KXNORDkk, KXNORQkk, KXNORWkk,
        KXORBkk, KXORQkk, KXORWkk
    )>;

def Zn5WriteVecOpMaskMemMov : SchedWriteRes<[Zn5FPOpMask4]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecOpMaskMemMov], (instrs KMOVBmk, KMOVDmk, KMOVQmk, KMOVWmk)>;

def Zn5WriteVecOpMaskKRMov : SchedWriteRes<[Zn5FPOpMask4]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecOpMaskKRMov], (instrs KMOVBkr, KMOVDkr, KMOVQkr, KMOVWkr)>;

def Zn5WriteVecALU2Slow : SchedWriteRes<[Zn5FPVAdd12]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALU2Slow], (instrs VALIGNDZ256rrik, VALIGNDZ256rri,
                                            VALIGNDZ256rrikz)
                                            >;

def Zn5WriteVecALU2Z : SchedWriteRes<[Zn5FPVAdd12]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALU2Z], (instrs VALIGNDZrri, VALIGNQZrri, VALIGNDZrrik,
                                         VALIGNDZrrikz, VALIGNQZrrik, VALIGNQZrrikz)
                                            >;

def Zn5WriteVecALUAlignQ : SchedWriteRes<[Zn5FPVAdd12]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUAlignQ], (instrs VALIGNQZ256rri, VALIGNQZ256rrik,
                                             VALIGNQZ256rrikz)
                                            >;
// Vector integer ALU op, no logicals (YMM).
defm : Zn5WriteResYMMPair<WriteVecALUY, [Zn5FPVAdd0123], 2, [1], 1>;

// SLow variants
def Zn5WriteVecALUYSlow : SchedWriteRes<[Zn5FPVAdd03]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUYSlow], (instrs VPABSBYrr, VPABSDYrr, VPABSWYrr,
                                            VPADDSBYrr, VPADDSWYrr, VPADDUSBYrr, VPADDUSWYrr,
                                            VPSUBSBYrr, VPSUBSWYrr, VPSUBUSBYrr, VPSUBUSWYrr,
                                            VPAVGBYrr, VPAVGWYrr,
                                            VPCMPEQQYrr,
                                            VPSIGNBYrr, VPSIGNDYrr, VPSIGNWYrr)>;


// MMXs
def Zn5WriteVecALUMMX : SchedWriteRes<[Zn5FPVAdd03]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUMMX], (instrs MMX_PABSBrr, MMX_PABSDrr, MMX_PABSWrr,
                                            MMX_PSIGNBrr, MMX_PSIGNDrr, MMX_PSIGNWrr,
                                            MMX_PADDBrr, MMX_PADDDrr, MMX_PADDWrr, MMX_PADDQrr,
                                            MMX_PADDSBrr, MMX_PADDSWrr, MMX_PADDUSBrr, MMX_PADDUSWrr,
                                            MMX_PAVGBrr, MMX_PAVGWrr,
                                            MMX_PANDNrr, MMX_PXORrr, MMX_PANDrr, MMX_PORrr,
                                            MMX_PCMPEQBrr, MMX_PCMPEQDrr,
                                            MMX_PMAXSWrr, MMX_PMAXUBrr, MMX_PMINSWrr, MMX_PMINUBrr,
                                            MMX_PCMPGTBrr, MMX_PCMPGTWrr,
                                            MMX_PSUBWrr, MMX_PSUBSBrr,
                                            MMX_PSUBSWrr, MMX_PSUBUSBrr, MMX_PSUBUSWrr
                                            )>;

defm : Zn5WriteResZMMPair<WriteVecALUZ, [Zn5FPVAdd0123], 2, [2], 1>; // Vector integer ALU op, no logicals (ZMM).

defm : Zn5WriteResXMMPair<WriteVecLogic, [Zn5FPVMisc0123], 1, [1], 1>;  // Vector integer and/or/xor logicals.
defm : Zn5WriteResXMMPair<WriteVecLogicX, [Zn5FPVMisc0123], 2, [1], 1>; // Vector integer and/or/xor logicals (XMM).
defm : Zn5WriteResYMMPair<WriteVecLogicY, [Zn5FPVMisc0123], 2, [1], 1>; // Vector integer and/or/xor logicals (YMM).
defm : Zn5WriteResZMMPair<WriteVecLogicZ, [Zn5FPVMisc0123], 2, [2], 1>; // Vector integer and/or/xor logicals (ZMM).
defm : Zn5WriteResXMMPair<WriteVecTest, [Zn5FPVAdd12, Zn5FPSt], 1, [1, 1], 2>;  // FIXME: latency not from llvm-exegesis // Vector integer TEST instructions.
defm : Zn5WriteResYMMPair<WriteVecTestY, [Zn5FPVAdd12, Zn5FPSt], 1, [1, 1], 2>; // FIXME: latency not from llvm-exegesis  // Vector integer TEST instructions (YMM).
defm : Zn5WriteResZMMPair<WriteVecTestZ, [Zn5FPVAdd12, Zn5FPSt], 1, [2, 2], 2>; // FIXME: latency not from llvm-exegesis  // Vector integer TEST instructions (ZMM).
defm : Zn5WriteResXMMPair<WriteVecShift, [Zn5FPVShift01], 2, [1], 1>;  // Vector integer shifts (default).
defm : Zn5WriteResXMMPair<WriteVecShiftX, [Zn5FPVShift01], 2, [1], 1>; // Vector integer shifts (XMM).
defm : Zn5WriteResYMMPair<WriteVecShiftY, [Zn5FPVShift01], 2, [1], 1>; // Vector integer shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVecShiftZ, [Zn5FPVShift01], 2, [2], 1>; // Vector integer shifts (ZMM).
defm : Zn5WriteResXMMPair<WriteVecShiftImm, [Zn5FPVShift01], 2, [1], 1>;  // Vector integer immediate shifts (default).
defm : Zn5WriteResXMMPair<WriteVecShiftImmX, [Zn5FPVShift01], 2, [1], 1>; // Vector integer immediate shifts (XMM).
defm : Zn5WriteResYMMPair<WriteVecShiftImmY, [Zn5FPVShift01], 2, [1], 1>; // Vector integer immediate shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVecShiftImmZ, [Zn5FPVShift01], 2, [2], 1>; // Vector integer immediate shifts (ZMM).
defm : Zn5WriteResXMMPair<WriteVecIMul, [Zn5FPVMul012], 2, [1], 1>;  // Vector integer multiply (default).
defm : Zn5WriteResXMMPair<WriteVecIMulX, [Zn5FPVMul012], 2, [1], 1>; // Vector integer multiply (XMM).
defm : Zn5WriteResYMMPair<WriteVecIMulY, [Zn5FPVMul012], 2, [1], 1>; // Vector integer multiply (YMM).
defm : Zn5WriteResZMMPair<WriteVecIMulZ, [Zn5FPVMul012], 2, [2], 1>; // Vector integer multiply (ZMM).
defm : Zn5WriteResXMMPair<WritePMULLD, [Zn5FPVMul012], 3, [1], 1>; // Vector PMULLD.
defm : Zn5WriteResYMMPair<WritePMULLDY, [Zn5FPVMul012], 3, [1], 1>; // Vector PMULLD (YMM).
defm : Zn5WriteResZMMPair<WritePMULLDZ, [Zn5FPVMul012], 3, [2], 1>; // Vector PMULLD (ZMM).
defm : Zn5WriteResXMMPair<WriteShuffle, [Zn5FPVShuf01], 2, [2], 1>;  // Vector shuffles.
defm : Zn5WriteResXMMPair<WriteShuffleX, [Zn5FPVShuf01], 2, [2], 1>; // Vector shuffles (XMM).
defm : Zn5WriteResYMMPair<WriteShuffleY, [Zn5FPVShuf01], 2, [2], 1>; // Vector shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteShuffleZ, [Zn5FPVShuf01], 2, [2], 1>; // Vector shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteVarShuffle, [Zn5FPVShuf01], 1, [1], 1>;  // Vector variable shuffles.
defm : Zn5WriteResXMMPair<WriteVarShuffleX, [Zn5FPVShuf01], 2, [1], 1>; // Vector variable shuffles (XMM).
defm : Zn5WriteResYMMPair<WriteVarShuffleY, [Zn5FPVShuf01], 2, [1], 1>; // Vector variable shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteVarShuffleZ, [Zn5FPVShuf01], 2, [2], 1>; // Vector variable shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteBlend, [Zn5FPVMisc0123], 1, [1], 1>; // Vector blends.
defm : Zn5WriteResYMMPair<WriteBlendY, [Zn5FPVMisc0123], 1, [1], 1>; // Vector blends (YMM).
defm : Zn5WriteResZMMPair<WriteBlendZ, [Zn5FPVMisc0123], 1, [2], 1>; // Vector blends (ZMM).
defm : Zn5WriteResXMMPair<WriteVarBlend, [Zn5FPVMul012], 1, [1], 1>; // Vector variable blends.
defm : Zn5WriteResYMMPair<WriteVarBlendY, [Zn5FPVMul012], 1, [1], 1>; // Vector variable blends (YMM).
defm : Zn5WriteResZMMPair<WriteVarBlendZ, [Zn5FPVMul012], 1, [2], 1>; // Vector variable blends (ZMM).
defm : Zn5WriteResXMMPair<WritePSADBW, [Zn5FPVAdd0123], 3, [2], 1>;  // Vector PSADBW.
defm : Zn5WriteResXMMPair<WritePSADBWX, [Zn5FPVAdd0123], 3, [2], 1>; // Vector PSADBW (XMM).
defm : Zn5WriteResYMMPair<WritePSADBWY, [Zn5FPVAdd0123], 3, [2], 1>; // Vector PSADBW (YMM).
defm : Zn5WriteResZMMPair<WritePSADBWZ, [Zn5FPVAdd0123], 3, [3], 1>; // Vector PSADBW (ZMM).
defm : Zn5WriteResXMMPair<WriteMPSAD, [Zn5FPVAdd0123], 5, [8], 4, /*LoadUOps=*/2>; // Vector MPSAD.
defm : Zn5WriteResYMMPair<WriteMPSADY, [Zn5FPVAdd0123], 5, [8], 3, /*LoadUOps=*/1>; // Vector MPSAD (YMM).
defm : Zn5WriteResZMMPair<WriteMPSADZ, [Zn5FPVAdd0123], 5, [16], 3, /*LoadUOps=*/1>; // Vector MPSAD (ZMM).
defm : Zn5WriteResXMMPair<WritePHMINPOS, [Zn5FPVAdd01], 3, [1], 1>;  // Vector PHMINPOS.

// Vector insert/extract operations.
defm : Zn5WriteResXMMPair<WriteVecInsert, [Zn5FPLd01], 2, [2], 2, /*LoadUOps=*/-1>; // Insert gpr to vector element.
defm : Zn5WriteResXMM<WriteVecExtract, [Zn5FPLd01], 1, [2], 2>; // Extract vector element to gpr.
defm : Zn5WriteResXMM<WriteVecExtractSt, [Zn5FPSt, Zn5Store], !add(1, Znver5Model.StoreLatency), [1, 1], 2>; // Extract vector element and store.

// MOVMSK operations.
defm : Zn5WriteResXMM<WriteFMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResXMM<WriteVecMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResYMM<WriteVecMOVMSKY, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResXMM<WriteMMXMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;

// Conversion between integer and float.
defm : Zn5WriteResXMMPair<WriteCvtSD2I, [Zn5FPFCvt01], 1, [1], 1>;  // Double -> Integer.
defm : Zn5WriteResXMMPair<WriteCvtPD2I, [Zn5FPFCvt01], 3, [3], 1>; // Double -> Integer (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPD2IY, [Zn5FPFCvt01], 3, [3], 2>; // Double -> Integer (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPD2IZ, [Zn5FPFCvt01], 3, [3], 2>; // Double -> Integer (ZMM).
defm : Zn5WriteResXMMPair<WriteCvtSS2I, [Zn5FPFCvt01], 1, [1], 2>;  // Float -> Integer.

defm : Zn5WriteResXMMPair<WriteCvtPS2I, [Zn5FPFCvt01], 3, [1], 1>; // Float -> Integer (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPS2IY, [Zn5FPFCvt01], 3, [1], 1>; // Float -> Integer (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPS2IZ, [Zn5FPFCvt01], 3, [2], 2>; // Float -> Integer (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtI2SD, [Zn5FPFCvt01], 3, [1], 2, /*LoadUOps=*/-1>;  // Integer -> Double.
defm : Zn5WriteResXMMPair<WriteCvtI2PD, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Double (XMM).
defm : Zn5WriteResYMMPair<WriteCvtI2PDY, [Zn5FPFCvt01], 3, [2], 2, /*LoadUOps=*/-1>; // Integer -> Double (YMM).
defm : Zn5WriteResZMMPair<WriteCvtI2PDZ, [Zn5FPFCvt01], 6, [4], 4, /*LoadUOps=*/-1>; // Integer -> Double (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtI2SS, [Zn5FPFCvt01], 3, [1], 2, /*LoadUOps=*/-1>;  // Integer -> Float.
defm : Zn5WriteResXMMPair<WriteCvtI2PS, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Float (XMM).
defm : Zn5WriteResYMMPair<WriteCvtI2PSY, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Float (YMM).
defm : Zn5WriteResZMMPair<WriteCvtI2PSZ, [Zn5FPFCvt01], 3, [2], 2>; // Integer -> Float (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtSS2SD, [Zn5FPFCvt01], 3, [1], 1>;  // Float -> Double size conversion.
defm : Zn5WriteResXMMPair<WriteCvtPS2PD, [Zn5FPFCvt01], 3, [1], 1>; // Float -> Double size conversion (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPS2PDY, [Zn5FPFCvt01], 4, [2], 2, /*LoadUOps=*/-1>; // Float -> Double size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPS2PDZ, [Zn5FPFCvt01], 6, [4], 4, /*LoadUOps=*/-1>; // Float -> Double size conversion (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtSD2SS, [Zn5FPFCvt01], 3, [1], 1>;  // Double -> Float size conversion.
defm : Zn5WriteResXMMPair<WriteCvtPD2PS, [Zn5FPFCvt01], 3, [1], 1>; // Double -> Float size conversion (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPD2PSY, [Zn5FPFCvt01], 6, [2], 2>; // Double -> Float size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPD2PSZ, [Zn5FPFCvt01], 7, [4], 4>; // Double -> Float size conversion (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtPH2PS, [Zn5FPFCvt01], 3, [1], 1>; // Half -> Float size conversion.
defm : Zn5WriteResYMMPair<WriteCvtPH2PSY, [Zn5FPFCvt01], 4, [2], 2, /*LoadUOps=*/-1>; // Half -> Float size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPH2PSZ, [Zn5FPFCvt01], 4, [4], 4, /*LoadUOps=*/-1>; // Half -> Float size conversion (ZMM).

defm : Zn5WriteResXMM<WriteCvtPS2PH, [Zn5FPFCvt01], 3, [2], 1>; // Float -> Half size conversion.
defm : Zn5WriteResYMM<WriteCvtPS2PHY, [Zn5FPFCvt01], 6, [2], 2>; // Float -> Half size conversion (YMM).
defm : Zn5WriteResZMM<WriteCvtPS2PHZ, [Zn5FPFCvt01], 6, [2], 2>; // Float -> Half size conversion (ZMM).

defm : Zn5WriteResXMM<WriteCvtPS2PHSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(3, Znver5Model.StoreLatency), [1, 1, 1], 2>; // Float -> Half + store size conversion.
defm : Zn5WriteResYMM<WriteCvtPS2PHYSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(6, Znver5Model.StoreLatency), [2, 1, 1], 3>; // Float -> Half + store size conversion (YMM).
defm : Zn5WriteResYMM<WriteCvtPS2PHZSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(6, Znver5Model.StoreLatency), [2, 1, 1], 3>; // Float -> Half + store size conversion (ZMM).

def Zn5WriteCvtI2PSMMX : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}

// VNNI Instructions
def Zn5WriteVNNI : SchedWriteRes<[Zn5FPVMul01]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVNNI], (instrs VPDPBUSDSYrr, VPDPBUSDYrr, VPDPWSSDSYrr, VPDPWSSDYrr)>;

// AVX512DQ MUL
def Zn5WriteAVX512DQMUL : SchedWriteRes<[Zn5FPVMul01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteAVX512DQMUL], (instregex "VPMUL(L|D)Q(Y|Z)(128?|256?)?(rr|rrk|rrkz)")>;

// VDPBSAD Instructions
def Zn5WritePSADZ : SchedWriteRes<[Zn5FPVAdd0123,Zn5FPVAdd01]> {
  let Latency = 5;
  let ReleaseAtCycles = [1,1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WritePSADZ], (instregex "VDBPSADBW(Z|Z128|Z256)(rri|rrikz)$")>;

// GFNI Instructions
def Zn5WriteGFNI : SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteGFNI], (instregex "VGF2P8(MULB|AFFINEINVQB|AFFINEQB)(Y|Z|Z128|Z256)(rr|rrk|rri|rrik|rrkz)",
                                        "(V)?GF2P8(AFFINEINV|AFFINE)QBrri")>;

def Zn5WriteExtrb : SchedWriteRes<[Zn5FPFMisc12,Zn5FPLd01]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteGFNI], (instregex "VPEXTRBZrri")>;

// MADD/MUL Instructions
def Zn5WriteMADDMUL : SchedWriteRes<[Zn5FPVMul01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMADDMUL], (instregex "(MMX_|V)?PMADD(UBSW|WD)(Y?|Z?|Z128?|Z256?)(rr|rrk|rrkz)",
                                           "(MMX_|V)?PMUL(HRSW|HUW|HW|LW|UDQ)(Y?|Z?|Z128?|Z256?)(rr|rrk|rrkz)")>;

// VPBLENDM Instructions
def Zn5WritePBLENDM : SchedWriteRes<[Zn5FPVMisc0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePBLENDM], (instregex "VPBLENDM(B|D|Q|W)(Y?|Z?|Z128?|Z256?)(rr|rrk|rrkz)",
                                           "VPBLEND(D|W|VB)(Y)?(rri|rrr)")>;

// CRC32 instruction.
defm : Zn5WriteResIntPair<WriteCRC32, [Zn5ALU1], 3, [1], 1>;

def Zn5WriteSHA1MSG1rr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSHA1MSG1rr], (instrs SHA1MSG1rr)>;

def Zn5WriteSHA1MSG1rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteSHA1MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteSHA1MSG1rr.NumMicroOps, 0);
}
def : InstRW<[Zn5WriteSHA1MSG1rm], (instrs SHA1MSG1rm)>;

def Zn5WriteSHA1MSG2rr_SHA1NEXTErr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSHA1MSG2rr_SHA1NEXTErr], (instrs SHA1MSG2rr, SHA1NEXTErr)>;

def Zn5Writerm_SHA1MSG2rm_SHA1NEXTErm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteSHA1MSG2rr_SHA1NEXTErr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteSHA1MSG2rr_SHA1NEXTErr.NumMicroOps, 0);
}
def : InstRW<[Zn5Writerm_SHA1MSG2rm_SHA1NEXTErm], (instrs SHA1MSG2rm, SHA1NEXTErm)>;

def Zn5WriteSHA256MSG1rr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSHA256MSG1rr], (instrs SHA256MSG1rr)>;

def Zn5Writerm_SHA256MSG1rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteSHA256MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 3];
  let NumMicroOps = !add(Zn5WriteSHA256MSG1rr.NumMicroOps, 0);
}
def : InstRW<[Zn5Writerm_SHA256MSG1rm], (instrs SHA256MSG1rm)>;

def Zn5WriteSHA256MSG2rr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 6;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 4;
}
def : InstRW<[Zn5WriteSHA256MSG2rr], (instrs SHA256MSG2rr)>;

def Zn5WriteSHA256MSG2rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPU0123]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteSHA256MSG2rr.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = !add(Zn5WriteSHA256MSG2rr.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteSHA256MSG2rm], (instrs SHA256MSG2rm)>;

def Zn5WriteSHA1RNDS4rri : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 6;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSHA1RNDS4rri], (instrs SHA1RNDS4rri)>;

def Zn5WriteSHA256RNDS2rr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSHA256RNDS2rr], (instrs SHA256RNDS2rr)>;

// Strings instructions.
// Packed Compare Implicit Length Strings, Return Mask
defm : Zn5WriteResXMMPair<WritePCmpIStrM, [Zn5FPVAdd0123], 7, [8], 3, /*LoadUOps=*/1>;
// Packed Compare Explicit Length Strings, Return Mask
defm : Zn5WriteResXMMPair<WritePCmpEStrM, [Zn5FPVAdd0123], 7, [12], 7, /*LoadUOps=*/5>;
// Packed Compare Implicit Length Strings, Return Index
defm : Zn5WriteResXMMPair<WritePCmpIStrI, [Zn5FPVAdd0123], 1, [8], 4>;
// Packed Compare Explicit Length Strings, Return Index
defm : Zn5WriteResXMMPair<WritePCmpEStrI, [Zn5FPVAdd0123], 2, [8], 8, /*LoadUOps=*/4>;

// AES instructions.
defm : Zn5WriteResXMMPair<WriteAESDecEnc, [Zn5FPAES01], 4, [1], 1>; // Decryption, encryption.
defm : Zn5WriteResXMMPair<WriteAESIMC, [Zn5FPAES01], 4, [1], 1>; // InvMixColumn.
defm : Zn5WriteResXMMPair<WriteAESKeyGen, [Zn5FPAES01], 4, [1], 1>; // Key Generation.

// Carry-less multiplication instructions.
defm : Zn5WriteResXMMPair<WriteCLMul, [Zn5FPCLM01], 5, [5], 2>;

// EMMS/FEMMS
defm : Zn5WriteResInt<WriteEMMS, [Zn5ALU0_5], 2, [1], 1>; // FIXME: latency not from llvm-exegesis

// Load/store MXCSR
defm : Zn5WriteResInt<WriteLDMXCSR, [Zn5AGU0123, Zn5Load, Zn5ALU0_5], !add(Znver5Model.LoadLatency, 1), [1, 1, 6], 1>; // FIXME: latency not from llvm-exegesis
defm : Zn5WriteResInt<WriteSTMXCSR, [Zn5ALU0_5, Zn5AGU0123, Zn5Store], !add(1, Znver5Model.StoreLatency), [60, 1, 1], 2>; // FIXME: latency not from llvm-exegesis

// Catch-all for expensive system instructions.
defm : Zn5WriteResInt<WriteSystem, [Zn5ALU0_5], 100, [100], 100>;

def Zn5WriteVZEROUPPER : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 0; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVZEROUPPER], (instrs VZEROUPPER)>;

def Zn5WriteVZEROALL : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 10; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [24];
  let NumMicroOps = 18;
}
def : InstRW<[Zn5WriteVZEROALL], (instrs VZEROALL)>;

// AVX2.
defm : Zn5WriteResYMMPair<WriteFShuffle256, [Zn5FPVShuf], 2, [1], 1, /*LoadUOps=*/2>; // Fp 256-bit width vector shuffles.
defm : Zn5WriteResYMMPair<WriteFVarShuffle256, [Zn5FPVShuf], 7, [1], 2, /*LoadUOps=*/1>; // Fp 256-bit width variable shuffles.
defm : Zn5WriteResYMMPair<WriteShuffle256, [Zn5FPVShuf], 2, [1], 1>; // 256-bit width vector shuffles.

def Zn5WriteVPERM2I128rr_VPERM2F128rr : SchedWriteRes<[Zn5FPVShuf]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVPERM2I128rr_VPERM2F128rr], (instrs VPERM2I128rri, VPERM2F128rri)>;

def Zn5WriteVPERM2F128rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPVShuf]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVPERM2I128rr_VPERM2F128rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = !add(Zn5WriteVPERM2I128rr_VPERM2F128rr.NumMicroOps, 0);
}
def : InstRW<[Zn5WriteVPERM2F128rm], (instrs VPERM2F128rmi)>;

def Zn5WriteVPERMDYrr : SchedWriteRes<[Zn5FPVShuf]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteVPERMDYrr], (instrs VPERMDYrr)>;

def Zn5WriteVPERMYm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPVShuf]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVPERMDYrr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteVPERMDYrr.NumMicroOps, 0);
}
def : InstRW<[Zn5WriteVPERMYm], (instrs VPERMQYmi, VPERMDYrm)>;

defm : Zn5WriteResYMMPair<WriteVPMOV256, [Zn5FPVShuf01], 4, [4], 2, /*LoadUOps=*/-1>; // 256-bit width packed vector width-changing move.
defm : Zn5WriteResYMMPair<WriteVarShuffle256, [Zn5FPVShuf01], 1, [1], 2>; // 256-bit width vector variable shuffles.
defm : Zn5WriteResXMMPair<WriteVarVecShift, [Zn5FPVShift01], 2, [1], 1>; // Variable vector shifts.
defm : Zn5WriteResYMMPair<WriteVarVecShiftY, [Zn5FPVShift01], 2, [1], 1>; // Variable vector shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVarVecShiftZ, [Zn5FPVShift01], 2, [2], 2>; // Variable vector shifts (ZMM).

// Old microcoded instructions that nobody use.
defm : Zn5WriteResInt<WriteMicrocoded, [Zn5ALU0_5], 100, [100], 100>;

// Fence instructions.
defm : Zn5WriteResInt<WriteFence, [Zn5ALU0_5], 1, [100], 1>;

def Zn5WriteLFENCE : SchedWriteRes<[Zn5LSU]> {
  let Latency = 1;
  let ReleaseAtCycles = [30];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteLFENCE], (instrs LFENCE)>;

def Zn5WriteSFENCE : SchedWriteRes<[Zn5LSU]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSFENCE], (instrs SFENCE)>;

// Nop, not very useful expect it provides a model for nops!
defm : Zn5WriteResInt<WriteNop, [Zn5ALU0_5], 0, [1], 1>; // FIXME: latency not from llvm-exegesis

def Zn5WriteXADD : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteXADD], (instrs XADD16rr, XADD32rr,
                                               XADD64rr, XADD8rr)>;

///////////////////////////////////////////////////////////////////////////////
// Zero Cycle Move
///////////////////////////////////////////////////////////////////////////////

def Zn5WriteZeroLatency : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteZeroLatency], (instrs MOV32rr, MOV32rr_REV,
                                            MOV64rr, MOV64rr_REV,
                                            MOVSX32rr32, KXORDkk, MMX_MOVQ2FR64rr, MMX_MOVQ2DQrr,
                                            MMX_PCMPGTDrr, MMX_PCMPEQWrr, MMX_PSUBDrr,
                                            MMX_PSUBQrr, MOV32rr, MOV32rr_REV, VPCMPEQBrr, MMX_MOVFR642Qrr)>;

def Zn5WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                               XCHG64rr, XCHG64ar)>;

defm : Zn5WriteResInt<WriteXCHG, [Zn5ALU0_5], 0, [8], 2>;        // Compare+Exchange - TODO RMW support.

defm : Zn5WriteResXMM<WriteFMoveX, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteFMoveY, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteFMoveZ, [], 0, [], 1>;

defm : Zn5WriteResXMM<WriteVecMove, [Zn5FPFMisc0123], 2, [1], 1>; // MMX
defm : Zn5WriteResXMM<WriteVecMoveX, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteVecMoveY, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteVecMoveZ, [], 0, [], 1>;

def : IsOptimizableRegisterMove<[
  InstructionEquivalenceClass<[
    // GPR variants.
    MOV32rr, MOV32rr_REV,
    MOV64rr, MOV64rr_REV,
    MOVSX32rr32,
    XCHG32rr, XCHG32ar,
    XCHG64rr, XCHG64ar,

    // MMX variants.
    // MMX moves are *NOT* eliminated.

    // SSE variants.
    MOVAPSrr, MOVAPSrr_REV,
    MOVUPSrr, MOVUPSrr_REV,
    MOVAPDrr, MOVAPDrr_REV,
    MOVUPDrr, MOVUPDrr_REV,
    MOVDQArr, MOVDQArr_REV,
    MOVDQUrr, MOVDQUrr_REV,

    // AVX variants.
    VMOVAPSrr, VMOVAPSrr_REV,
    VMOVUPSrr, VMOVUPSrr_REV,
    VMOVAPDrr, VMOVAPDrr_REV,
    VMOVUPDrr, VMOVUPDrr_REV,
    VMOVDQArr, VMOVDQArr_REV,
    VMOVDQUrr, VMOVDQUrr_REV,

    // AVX YMM variants.
    VMOVAPSYrr, VMOVAPSYrr_REV,
    VMOVUPSYrr, VMOVUPSYrr_REV,
    VMOVAPDYrr, VMOVAPDYrr_REV,
    VMOVUPDYrr, VMOVUPDYrr_REV,
    VMOVDQAYrr, VMOVDQAYrr_REV,
    VMOVDQUYrr, VMOVDQUYrr_REV,
  ], TruePred >
]>;

// FIXUP and RANGE Instructions
def Zn5WriteVFIXUPIMMPDZrr_VRANGESDrr : SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVFIXUPIMMPDZrr_VRANGESDrr], (instregex
	"VFIXUPIMM(S|P)(S|D)(Z|Z128|Z256?)rrik", "VFIXUPIMM(S|P)(S|D)(Z?|Z128?|Z256?)rrikz",
  "VFIXUPIMM(S|P)(S|D)(Z128|Z256?)rri",  "VRANGE(S|P)(S|D)(Z?|Z128?|Z256?)rri(b?)",
	"VRANGE(S|P)(S|D)(Z|Z128|Z256?)rri(b?)k","VRANGE(S|P)(S|D)(Z?|Z128?|Z256?)rri(b?)kz",
  "VFIXUPIMM(S|P)(S|D)Z(rrib|rribk|rribkz)"
	)>;

// SCALE & REDUCE instructions
def Zn5WriteSCALEREDUCErr: SchedWriteRes<[Zn5FPFMisc23]> {
  let Latency = 6;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSCALEREDUCErr], (instrs
  VSCALEFSSZrrb_Int, VSCALEFSDZrr, VSCALEFSDZrrbkz_Int, VSCALEFPSZ128rr,
  VSCALEFPSZ128rrkz, VSCALEFPSZ256rr, VSCALEFPSZ256rrkz, VSCALEFSDZrr, VSCALEFPDZ256rrkz,
  VSCALEFSSZrr, VSCALEFSSZrrbkz_Int, VSCALEFSSZrrkz,
  VSCALEFPSZrr, VSCALEFPSZrrbkz, VSCALEFPSZrrkz
	)>;

def : InstRW<[Zn5WriteSCALEREDUCErr], (instregex
  "VREDUCE(SS|SD)Z(rri|rrib|rribkz|rrikz)$",
  "VREDUCE(PS|PD)Z(rri|rrib|rribkz|rrikz)$",
  "VREDUCE(PS|PD)(Z128|Z256)(rri|rrikz)$"
  )>;

def Zn5WriteSCALErrSlow: SchedWriteRes<[Zn5FPFMisc23]> {
  let Latency = 3;
  let ReleaseAtCycles = [3];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSCALErrSlow], (instregex
        "VREDUCE(S|P)(S|D)(Z|Z128)(rrik|rribk)$",
        "VSCALEFPDZ128(rr|rrk|rrkz)",
        "VSCALEFPDZ256(rr|rrk)$",
        "VSCALEFSDZrrb_Int",
        "VSCALEFPSZ128rrk$", "VSCALEFPSZ256rrk$",
        "VSCALEFPSZ(rrb|rrbk|rrk)$",
        "VSCALEFPDZ(rr|rrb|rrbk|rrbkz|rrk|rrkz)(_Int)?",
        "VSCALEFSDZ(rrk|rrbk_Int|rrkz)$",
        "VSCALEFSSZ(rrk|rrbk_Int)$"
	)>;

//BF16PS Instructions
def Zn5WriteBF16: SchedWriteRes<[Zn5FPFMisc23]> {
  let Latency = 6;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteBF16], (instregex
        "(V?)DPBF16PS(Z?|Z128?|Z256?)(r|rk|rkz)"
	)>;

// BUSD and VPMADD Instructions
def Zn5WriteBUSDr_VPMADDr: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteBUSDr_VPMADDr], (instregex
	"VPDP(BU|WS)(S|P)(S|D|DS)(Z|Z128|Z256)(r|rk|rkz)",
        "VPMADD52(H|L)UQ(Z|Z128|Z256)(r|rk|rkz)"
	)>;

// SHIFT instructions
def Zn5WriteSHIFTrr: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSHIFTrr], (instregex
        "VP(LZCNT|SHLD|SHRD?)(D|Q|W|VD|VQ|VW?)(Z?|Z128?|Z256?)(rr|rk|rrk|rrkz|rri|rrik|rrikz)",
        "(V?)P(SLL|SRL|SRA)(D|Q|W|DQ)(Y?|Z?|Z128?|Z256?)(rr|rrk|rrkz)",
        "(V?)P(SLL|SRL|SRA)DQYri",
        "(V?)P(SLL|SRL)DQ(Z?|Z256?)ri",
        "(V?)P(ROL|ROR)(D|Q|VD|VQ)(Z?|Z128?|Z256?)(rr|rrk|rrkz)",
        "(V?)P(ROL|ROR)(D|Q|VD|VQ)(Z256?)(ri|rik|rikz)",
        "(V?)P(ROL|ROR)(D|Q)(Z?|Z128?)(ri|rik|rikz)"
	)>;

def Zn5WriteSHIFTri: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteSHIFTri], (instregex
        "VP(SLL|SRL|SRA)(D|Q|W)(Z|Z128|Z256?)(ri|rik|rikz)"
	)>;

// ALIGN Instructions
def Zn5WriteALIGN: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteALIGN], (instregex
        "(V?)PALIGNR(Z?|Z128?|Z256?)(rri|rrik|rrikz)"
	)>;

//PACK Instructions
def Zn5WritePACK: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePACK], (instregex
        "(V?)PACK(SS|US)(DW|WB)(Z?|Z128?|Z256?)(rr|rrk|rrkz)"
	)>;

// MAX and MIN Instructions
def Zn5WriteFCmp64: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteFCmp64], (instregex
        "(V?)CMP(S|P)(S|D)(rr|rri|rr_Int)",
        "(V?|VP?)(MAX|MIN|MINC|MAXC)(S|P|U)(S|D|Q)(Z?|Z128?|Z256?)(rr|rri|rrk|rrkz)(_Int?)",
        "VP(MAX|MIN)(SQ|UQ)(Z|Z128|Z256)(rr|rrk|rrkz)",
        "(V?)(MAX|MAXC|MIN|MINC)PD(Z|Z128|Z256?)(rr|rrk|rrkz)"
	)>;

// MOV Instructions
def Zn5MOVS: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVS], (instregex
        "(V?)PMOV(SX|ZX)(BD|BQ|BW|WD|WQ|DQ)(Z128?|Z256?)(rr|rrk|rrkz)",
        "(V?)PMOV(SX|QD|UZ|ZX)(BD|BQ|BW?)(Y|Z128?)(rr|rrk|rrkz)",
        "(V?)PMOV(SX|US|ZX)(DQ|WD|QW|WQ?)(Y|Z128?)(rr|rrk|rrkz)",
        "(V?)VMOVDDUP(Z|Z128|Z256)(rr|rrk|rrkz)",
        "VPMOV(DB|DW|QB|QD|QW|SDB|SDW|SQB|SQD|SQW|SWB|USDB|USDW|USQB|USQD|USWB|WB)(Z128?)(rr|rrk|rrkz)"
	)>;

def Zn5MOVSrr: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 5;
  let ReleaseAtCycles = [5];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVSrr], (instregex
        "(V?)PMOV(DB|QB|QW|SDB|SQB|SQW|USDB|USQB|USQW)(Z?)(rr|rrk|rrkz)"
	)>;

//VPTEST Instructions
def Zn5VPTESTZ128: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 3;
  let ReleaseAtCycles = [3];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VPTESTZ128], (instregex
        "(V?)PTEST(N?)(MB|MD|MQ|MW)(Z128?)(rrk)"
	)>;

def Zn5VPTESTZ256: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VPTESTZ256], (instregex
        "(V?)PTEST(N?)(MB|MD|MQ|MW)(Z256?)(rr|rrk)",
        "VFMSUB231SSZrkz_Int"
	)>;

def Zn5VPTESTZ: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 6;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VPTESTZ], (instregex
        "(V?)PTEST(N?)(MB|MD|MQ|MW)(Z?)(rrk)"
	)>;

// Floating point reciprocal estimate (XMM).
def Zn5FRcpXZ: SchedWriteRes<[Zn5FPFMul01]> {
  let Latency = 5;
  let ReleaseAtCycles = [5];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5FRcpXZ], (instregex
        "VRCP14PDZ128r", "VRCP14PDZ128rk", "VRCP14PDZ128rkz",
        "VRCP14PSZ128r", "VRCP14PSZ128rk", "VRCP14PSZ128rkz",
        "VRCP14SDZrr", "VRCP14SDZrrk", "VRCP14SDZrrkz",
        "VRCP14SSZrr", "VRCP14SSZrrk", "VRCP14SSZrrkz"
  )>;

def Zn5FRcpY: SchedWriteRes<[Zn5FPFMul01]> {
  let Latency = 4;
  let ReleaseAtCycles = [5];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5FRcpY], (instrs VRCPPSYr)>;

// CONFLICT Instructions
def Zn5CONFLICTZ128: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 3;
  let ReleaseAtCycles = [3];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5CONFLICTZ128], (instregex
        "VPCONFLICT(D|Q)Z128(rr|rrk|rrkz)",
        "VPCONFLICT(D|Q)(Z|Z256)rrk$"
	)>;

def Zn5CONFLICTrr: SchedWriteRes<[Zn5FPFMisc01,Zn5FPFMisc12,Zn5FPFMisc23]> {
  let Latency = 7;
  let ReleaseAtCycles = [2,2,2];
  let NumMicroOps = 4;
}
def : InstRW<[Zn5CONFLICTrr], (instregex
        "VPCONFLICT(D|Q)(Z|Z256)(rr|rrkz)$"
	)>;

// RSQRT Instructions
def Zn5VRSQRT14PDZ256: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VRSQRT14PDZ256], (instregex
        "VRSQRT14(PD|PS)(Z?|Z128?|Z256?)(r|rr|rk|rrk|rkz|rrkz)",
        "VRSQRT14(SD|SS)Z(rr|rrk|rrkz)"
	)>;


// PERM Instructions
def Zn5PERMILP: SchedWriteRes<[Zn5FPFMisc123]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMILP], (instregex
        "VPERMILP(S|D)(Y|Z|Z128|Z256)?(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2_128: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 3;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMIT2_128], (instregex
	"VPERM(I2|T2)(PS|PD|W)Z128(rr|rrk|rrkz)",
	"VPERM(I2|T2)(B|D|Q)Z128(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2_128rr:SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMIT2_128rr], (instregex
	"V(P?)COMPRESS(B|W|D|Q|PD|PS|SD|SQ)Z128(rr|rrk|rrkz)",
	"VPERM(B|D|Q|W)(Z128?)(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2_128rr2Ops:SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5PERMIT2_128rr2Ops], (instregex
  "VPEXPAND(B|W|D|Q)Z128(rr|rrk|rrkz)",
  "VEXPAND(PD|PS)Z128(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2_256: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMIT2_256], (instregex
	"VPERM(I2|T2)(PS|PD|W)Z256(rr|rrk|rrkz)",
	"VPERMP(S|D)Z256(rr|rrk|rrkz)",
	"V(P?)COMPRESS(B|W|D|Q|PD|PS|SD|SQ)Z256(rr|rrk|rrkz)",
	"VPERM(B|D|Q|W)Z256(rr|rrk|rrkz)",
	"VPERM(I2|Q|T2)(B|D|Q)Z256(rr|rrk|rrkz)",
	"VPERM(PS|PD|Q)Y(ri|rr)"
	)>;

def Zn5PERMIT2_256_2Ops: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5PERMIT2_256_2Ops], (instregex
	"VPEXPAND(B|W|D|Q)Z256(rr|rrk|rrkz)",
  "VEXPAND(PD|PS)Z256(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2Z: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMIT2Z], (instregex
	"VPERM(I2|T2)(PS|PD|W)Z(rr|rrk|rrkz)",
	"VPERM(B|D|W)Z(rr|rrk|rrkz)",
	"VPERM(I2|Q|T2)(B|D|Q)Z(rr|rrk|rrkz)",
	"V(P?)COMPRESS(B|W|D|Q|PD|PS|SD|SQ)Z(rr|rrk|rrkz)",
	"VPERMP(S|D)Z(rr|rrk|rrkz)",
  "VPERMQZ(rr|rrk|rrkz)"
	)>;

def Zn5PERMIT2Z2Ops: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5PERMIT2Z2Ops], (instregex
	"VPEXPAND(B|W|D|Q)Z(rr|rrk|rrkz)",
  "VEXPAND(PD|PS)Z(rr|rrk|rrkz)"
	)>;

// ALU SLOW Misc Instructions
def Zn5VecALUZSlow: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VecALUZSlow], (instrs
	VPABSBZ128rr,      VPABSBZ128rrk,  VPABSBZ128rrkz,   VPABSDZ128rr,
	VPABSDZ128rrk,     VPABSDZ128rrkz, VPABSQZ128rr,     VPABSQZ128rrk,
	VPABSQZ128rrkz,    VPABSWZ128rr,   VPABSWZ128rrk,    VPABSWZ128rrkz,
	VPADDSBZ128rr,     VPADDSBZ128rrk, VPADDSBZ128rrkz,  VPADDSWZ128rr,
	VPADDSWZ128rrk,    VPADDSWZ128rrkz,VPADDUSBZ128rr,   VPADDUSBZ128rrk,
	VPADDUSBZ128rrkz,  VPADDUSWZ128rr, VPADDUSWZ128rrk,  VPADDUSWZ128rrkz,
	VPAVGBZ128rr,      VPAVGBZ128rrk,  VPAVGBZ128rrkz,   VPAVGWZ128rr,
	VPAVGWZ128rrk,     VPAVGWZ128rrkz, VPOPCNTBZ128rr,   VPOPCNTBZ128rrk,
	VPOPCNTBZ128rrkz,  VPOPCNTDZ128rr, VPOPCNTDZ128rrk,  VPOPCNTDZ128rrkz,
	VPOPCNTQZ128rr,    VPOPCNTQZ128rrk,VPOPCNTQZ128rrkz, VPOPCNTWZ128rr,
	VPOPCNTWZ128rrk,   VPOPCNTWZ128rrkz,VPSUBSBZ128rr,   VPSUBSBZ128rrk,
	VPSUBSBZ128rrkz,   VPSUBSWZ128rr,   VPSUBSWZ128rrk,  VPSUBSWZ128rrkz,
	VPSUBUSBZ128rr,    VPSUBUSBZ128rrk, VPSUBUSBZ128rrkz,VPSUBUSWZ128rr,
	VPSUBUSWZ128rrk,   VPSUBUSWZ128rrkz
	)>;

// Zero latency ALU insns
def Zn5WriteALUrr : SchedWriteRes<[Zn5ALU0_5]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteALUrr], (instregex "STC", "CDQ", "CQO", "CLC")>;

///////////////////////////////////////////////////////////////////////////////
// Dependency breaking instructions.
///////////////////////////////////////////////////////////////////////////////

def Zn5WriteZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteALU]>
]>;
def : InstRW<[Zn5WriteZeroIdiom], (instrs XOR32rr, XOR32rr_REV,
                                          XOR64rr, XOR64rr_REV,
                                          SUB32rr, SUB32rr_REV,
                                          SUB64rr, SUB64rr_REV)>;

def Zn5WriteZeroIdiomEFLAGS : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<CheckSameRegOperand<0, 1>>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                                 [WriteALU]>
]>;
def : InstRW<[Zn5WriteZeroIdiomEFLAGS], (instrs CMP8rr,  CMP8rr_REV,
                                                CMP16rr, CMP16rr_REV,
                                                CMP32rr, CMP32rr_REV,
                                                CMP64rr, CMP64rr_REV)>;

def Zn5WriteFZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogic]>
]>;
// NOTE: XORPSrr, XORPDrr are not zero-cycle!
def : InstRW<[Zn5WriteFZeroIdiom], (instrs VXORPSrr, VXORPDrr,
                                           VANDNPSrr, VANDNPDrr)>;

def Zn5WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[Zn5WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                            VANDNPSYrr, VANDNPDYrr)>;

def Zn5WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;
// NOTE: PXORrr,PANDNrr are not zero-cycle!
def : InstRW<[Zn5WriteVZeroIdiomLogicX], (instrs VPXORrr, VPANDNrr)>;

def Zn5WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomLogicY], (instrs VPXORYrr, VPANDNYrr)>;

def Zn5WriteVZeroIdiomLogicZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicZ]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomLogicZ], (instrs VXORPDZ256rr)>;

def Zn5WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;
// NOTE: PSUBBrr, PSUBWrr, PSUBDrr, PSUBQrr,
//       PCMPGTBrr, PCMPGTWrr, PCMPGTDrr, PCMPGTQrr are not zero-cycle!
def : InstRW<[Zn5WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr)>;

def Zn5WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomALUY],
             (instrs VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
                     VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr)>;

def : IsZeroIdiomFunction<[
  // GPR Zero-idioms.
  DepBreakingClass<[ XOR32rr, XOR32rr_REV,
                     XOR64rr, XOR64rr_REV,
                     SUB32rr, SUB32rr_REV,
                     SUB64rr, SUB64rr_REV ], ZeroIdiomPredicate>,

  // SSE XMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    XORPSrr, XORPDrr,
    ANDNPSrr, ANDNPDrr,

    // int variants.
    PXORrr,
    PANDNrr,
    PSUBBrr, PSUBWrr, PSUBDrr, PSUBQrr,
    PSUBSBrr, PSUBSWrr,
    PSUBUSBrr, PSUBUSWrr,
    PCMPGTBrr, PCMPGTWrr, PCMPGTDrr, PCMPGTQrr
  ], ZeroIdiomPredicate>,

  // AVX XMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSrr, VXORPDrr,
    VANDNPSrr, VANDNPDrr,

    // int variants.
    VPXORrr,
    VPANDNrr,
    VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
    VPSUBSBrr, VPSUBSWrr,
    VPSUBUSBrr, VPSUBUSWrr,
    VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr,
  ], ZeroIdiomPredicate>,

  // AVX YMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSYrr, VXORPDYrr,
    VANDNPSYrr, VANDNPDYrr,

    // int variants.
    VPXORYrr,
    VPANDNYrr,
    VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
    VPSUBSBYrr, VPSUBSWYrr,
    VPSUBUSBYrr, VPSUBUSWYrr,
    VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr
  ], ZeroIdiomPredicate>,
]>;

def : IsDepBreakingFunction<[
  // GPR
  DepBreakingClass<[ SBB32rr, SBB32rr_REV,
                     SBB64rr, SBB64rr_REV ], ZeroIdiomPredicate>,
  DepBreakingClass<[ CMP8rr,  CMP8rr_REV,
                     CMP16rr, CMP16rr_REV,
                     CMP32rr, CMP32rr_REV,
                     CMP64rr, CMP64rr_REV ], CheckSameRegOperand<0, 1> >,
  // SSE
  DepBreakingClass<[
    PCMPEQBrr, PCMPEQWrr, PCMPEQDrr, PCMPEQQrr
  ], ZeroIdiomPredicate>,

  // AVX XMM
  DepBreakingClass<[
    VPCMPEQBrr, VPCMPEQWrr, VPCMPEQDrr, VPCMPEQQrr
  ], ZeroIdiomPredicate>,

  // AVX YMM
  DepBreakingClass<[
    VPCMPEQBYrr, VPCMPEQWYrr, VPCMPEQDYrr, VPCMPEQQYrr
  ], ZeroIdiomPredicate>,
]>;

} // SchedModel
