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
//  * AMD Software Optimization Guide for AMD Zen5 Microarchitecture
//    https://www.amd.com/system/files/TechDocs/58455.zip
//===----------------------------------------------------------------------===//

def Znver5Model : SchedMachineModel {
  // AMD SOG Zen5, 2.9.6 Dispatch
  // The processor may dispatch up to 8 macro-ops per cycle 
  // into the execution engine.
  let IssueWidth = 8;
  // AMD SOG Zen5, 2.10.3
  // The retire control unit (RCU) tracks the completion status of all
  // outstanding operations (integer, load/store, and floating-point) and is
  // the final arbiter for exception processing and recovery.
  // The unit can receive up to 8 macro ops dispatched per cycle and track up
  // to 448 macro ops in-flight in non-SMT mode or 160 per thread in SMT mode.
  let MicroOpBufferSize = 448;
  // AMD SOG Zen5, 2.9.1 Op Cache
  // The op cache is organized as an associative cache with 64 sets and 16 ways.
  // At each set-way intersection is an entry containing up to  6 instructions 
  // or fused instructions. This differs from Zen3 and Zen4 opcache entries, 
  // which stored macro-ops, and can improve effective Op Cache storage
  // density. The maximum capacity of the Op Cache is 6 K instructions or 
  // fused instructions. 
  // FIXME: The estimation below is from znver3 and znver4
  // Assuming a maximum dispatch of 9 ops/cy and a mispredict cost of 12cy from
  // the op-cache, we limit the loop buffer to 9*12 = 108 to avoid loop
  // unrolling leading to excessive filling of the op-cache from frontend.
  let LoopMicroOpBufferSize = 108;
  // AMD SOG Zen5, 2.6.2 L1 Data Cache
  // The L1 data cache has a 4- or 5- cycle integer load-to-use latency.
  // AMD SOG Zen5, 2.12 L1 Data Cache
  // The AGU and LS pipelines are optimized for simple address generation modes.
  // <...> and can achieve 4-cycle load-to-use integer load latency.
  let LoadLatency = 4;
  // AMD SOG Zen5, 2.12 L1 Data Cache
  // The AGU and LS pipelines are optimized for simple address generation modes.
  // <...> and can achieve <...> 7-cycle load-to-use FP load latency.
  int VecLoadLatency = 7;
  // Latency of a simple store operation.
  int StoreLatency = 1;
  // FIXME:
  let HighLatency = 25; // FIXME: any better choice?
  // AMD SOG Zen5, 2.8 Optimizing Branching
  // The branch misprediction penalty is in the range from 12 to 18 cycles,
  // <...>. The common case penalty is 15 cycles.
  let MispredictPenalty = 15;

  let PostRAScheduler = 1; // Enable Post RegAlloc Scheduler pass.

  let CompleteModel = 1;
}

let SchedModel = Znver5Model in {


//===----------------------------------------------------------------------===//
// RCU
//===----------------------------------------------------------------------===//

// AMD SOG Zen5, 2.10.3 Retire Control Unit
// The unit can receive up to 8 macro ops dispatched per cycle and track up to
// 448 macro ops in-flight in non-SMT mode or 224 per thread in SMT mode. <...>
// The retire unit handles in-order commit of up to eight macro ops per cycle.
def Zn5RCU : RetireControlUnit<Znver5Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Integer Execution Unit
//

// AMD SOG Zen5, 2.4 Superscalar Organization
// The processor uses four decoupled independent integer scheduler queues,
// each one servicing one ALU pipeline and one or two other pipelines

//
// Execution pipes
//===----------------------------------------------------------------------===//

// AMD SOG Zen5, 2.10.2 Execution Units
// The processor contains 6 general purpose integer execution pipes. Each pipe 
// has an ALU capable of general-purpose integer operations. ALU0/ALU1/ALU2 
// additionally have multiply/CRC capability; ALU3/ALU4/ALU5 have 
// PDEP/PEXT/shift/branch execution capability
def Zn5ALU0 : ProcResource<1>;
def Zn5ALU1 : ProcResource<1>;
def Zn5ALU2 : ProcResource<1>;
def Zn5ALU3 : ProcResource<1>;
def Zn5ALU4 : ProcResource<1>;
def Zn5ALU5 : ProcResource<1>;


// AMD SOG Zen5, 2.10.2 Execution Units
// There are four Address Generation Units (AGUs) for all load and store address
// generation. All ALU pipelines can issue store data movement operations, with 
// a bandwidth of two integer stores per cycle to the store queue.
def Zn5AGU0 : ProcResource<1>;
def Zn5AGU1 : ProcResource<1>;
def Zn5AGU2 : ProcResource<1>;
def Zn5AGU3 : ProcResource<1>;

//
// Execution Units
//===----------------------------------------------------------------------===//

// AMD SOG Zen5, 2.10.2 Execution Units
// ALU3 additionally has divide <...> execution capability.
defvar Zn5Divider = Zn5ALU3;

// AMD SOG Zen5, 2.10.2 Execution Units
// ALU3/ALU4/ALU5 have PDEP/PEXT/shift/branch execution capability.
defvar Zn5BRU0 = Zn5ALU3;
defvar Zn5BRU1 = Zn5ALU4;
defvar Zn5BRU2 = Zn5ALU5;

// AMD SOG Zen5, 2.10.2 Execution Units
// ALU0/ALU1/ALU2 additionally have multiply/CRC capability
defvar Zn5Multiplier0 = Zn5ALU0;
defvar Zn5Multiplier1 = Zn5ALU1;
defvar Zn5Multiplier2 = Zn5ALU2;

// Execution pipeline grouping
//===----------------------------------------------------------------------===//

// General ALU operations
def Zn5ALU012345 : ProcResGroup<[Zn5ALU0, Zn5ALU1, Zn5ALU2, Zn5ALU3, Zn5ALU4, Zn5ALU5]>;

// General AGU operations
def Zn5AGU0123 : ProcResGroup<[Zn5AGU0, Zn5AGU1, Zn5AGU2, Zn5AGU3]>;

// Multiplications
def Zn5Multiplier012 : ProcResGroup<[Zn5Multiplier0, Zn5Multiplier1, Zn5Multiplier2]>;

// Control flow: jumps, calls
def Zn5BRU012 : ProcResGroup<[Zn5BRU0, Zn5BRU1, Zn5BRU2]>;

// Everything that isn't control flow, but still needs to access CC register,
// namely: conditional moves, SETcc.
def Zn5ALU345 : ProcResGroup<[Zn5ALU3, Zn5ALU4, Zn5ALU5]>;

// Zn5ALU3, Zn4ALU4, Zn5ALU5 handles complex bit twiddling: CRC/PDEP/PEXT

// Simple bit twiddling: bit test, shift/rotate, bit extraction
def Zn5ALU012 : ProcResGroup<[Zn5ALU0, Zn5ALU1, Zn5ALU2]>;


//
// Scheduling
//===----------------------------------------------------------------------===//

// AMD SOG Zen5, 2.10.3 Retire Control Unit
// The integer physical register file (PRF) consists of 240 registers.
def Zn5IntegerPRF : RegisterFile<240, [GR64, CCR], [1, 1], [1, 0],
                              6,  // Max moves that can be eliminated per cycle.
                              0>; // Restrict move elimination to zero regs.

// AMD SOG Zen5, 2.10.1 Schedulers
// The schedulers can receive up to eight macro-ops per cycle, and track operand
// availability and dependency execution requirements of the contained 
// micro-ops.
def Zn5Int : ProcResGroup<[Zn5ALU0, Zn5AGU0,          // scheduler 0
                           Zn5ALU1, Zn5AGU1,          // scheduler 1
                           Zn5ALU2, Zn5AGU2,          // scheduler 2
                           Zn5ALU3, Zn5AGU3, Zn5BRU0, // scheduler 3
                           Zn5ALU4,          Zn5BRU1, // scheduler 4
                           Zn5ALU5,          Zn5BRU2, // scheduler 5
                          ]> {
  // FIXME: This is from zen3:
  // anandtech, The integer scheduler has a 4*24 entry macro op capacity.
  let BufferSize = !mul(4, 24);
}


//===----------------------------------------------------------------------===//
// Floating-Point Unit
//

// AMD SOG Zen5, 2.4 Superscalar Organization
// The processor uses <...> two decoupled independent floating point schedulers
// each servicing two FP pipelines and one store or FP-to-integer pipeline.

//
// Execution pipes
//===----------------------------------------------------------------------===//

// AMD SOG Zen5, 2.10.1 Schedulers
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
// AMD SOG Zen5, 2.11.1 Floating Point Execution Resources

// (v)FMUL*, (v)FMA*, Floating Point Compares, Blendv(DQ)
defvar Zn5FPFMul0 = Zn5FP0;
defvar Zn5FPFMul1 = Zn5FP1;

// (v)FADD*
defvar Zn5FPFAdd0 = Zn5FP2;
defvar Zn5FPFAdd1 = Zn5FP3;

// All convert operations except pack/unpack
defvar Zn5FPFCvt0 = Zn5FP2;
defvar Zn5FPFCvt1 = Zn5FP3;

// All Divide and Square Root except Reciprocal Approximation
// AMD SOG Zen5, 2.11.1 Floating Point Execution Resources
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

// AMD SOG Zen5, 2.11 Floating-Point Unit
// Stores and floating point to general purpose register transfer
// have 2 dedicated pipelines (pipe 5 and 6).
def Zn5FPU0123 : ProcResGroup<[Zn5FP0, Zn5FP1, Zn5FP2, Zn5FP3]>;

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
// AMD SOG Zen5, 2.11 Floating-Point Unit
// Stores and floating point to general purpose register transfer
// have 2 dedicated pipelines (pipe 5 and 6).
defvar Zn5FPLd01 = Zn5FP45;

// AMD SOG Zen5, 2.11 Floating-Point Unit
// Note that FP stores are supported on two pipelines,
// but throughput is limited to one per cycle.
let Super = Zn5FP45 in
def Zn5FPSt : ProcResource<1>;

// Integer Adds, Subtracts, and Compares
// Some complex VADD operations are not available in all pipes.
def Zn5FPVAdd0123 : ProcResGroup<[Zn5FPVAdd0, Zn5FPVAdd1, Zn5FPVAdd2, Zn5FPVAdd3]>;

def Zn5FPVAdd01: ProcResGroup<[Zn5FPVAdd0, Zn5FPVAdd1]>;
def Zn5FPVAdd12: ProcResGroup<[Zn5FPVAdd1, Zn5FPVAdd2]>;

// AVX512 Opmask pipelines
def Zn5FPOpMask01: ProcResGroup<[Zn5FP2, Zn5FP3]>;
def Zn5FPOpMask4: ProcResGroup<[Zn5FP45]>;

// Integer Multiplies, SAD
def Zn5FPVMul012 : ProcResGroup<[Zn5FPVMul0, Zn5FPVMul1, Zn5FPVMul2]>;

// Data Shuffles, Packs, Unpacks, Permute, Blendvb
// Some complex shuffle operations are only available in pipe1.
def Zn5FPVShuf01 : ProcResGroup<[Zn5FPVShuf, Zn5FPVShufAux]>;

// Note: comparing with zen4, blendvb is moved from Zn5FPVMul012 to Zn5FPVShuf01

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

// Agner, 21.8 Register renaming and out-of-order schedulers
// The floating point register file has 192 vector registers
// of 512b each in zen4.
def Zn5FpPRF : RegisterFile<192, [VR64, VR128, VR256, VR512], [1, 1, 1, 1], [0, 1, 1],
                            6,  // Max moves that can be eliminated per cycle.
                            0>; // Restrict move elimination to zero regs.

// AMD SOG Zen5, 2.11 Floating-Point Unit
// The floating-point scheduler has a 2*32 entry macro op capacity.
// AMD SOG Zen5, 2.11 Floating-Point Unit
// <...> the scheduler can issue 1 micro op per cycle for each pipe.
// FIXME: those are two separate schedulers, not a single big one.
def Zn5FP : ProcResGroup<[Zn5FP0, Zn5FP2,          /*Zn5FP4,*/ // scheduler 0
                          Zn5FP1, Zn5FP3, Zn5FP45 /*Zn5FP5*/  // scheduler 1
                         ]> {
  let BufferSize = !mul(2, 32);
}

// AMD SOG Zen5, 2.11 Floating-Point Unit
// Macro ops can be dispatched to the 64 entry Non Scheduling Queue (NSQ)
// even if floating-point scheduler is full.
// FIXME: how to model this properly?


//===----------------------------------------------------------------------===//
// Load-Store Unit
//

// AMD SOG Zen5, 2.12 Load-Store Unit
// LS unit contains four largely independent pipelines enabling the execution 
// of four memory operations per cycle.
def Zn5LSU : ProcResource<4>;

// AMD SOG Zen5, 2.12 Load-Store Unit
// All three memory operations can be loads.
let Super = Zn5LSU in
def Zn5Load : ProcResource<3> {
  // AMD SOG Zen5, 2.12 Load-Store Unit
  // LS can track up to 64 uncompleted loads and has no specific limit on the 
  // number of completed loads.
  let BufferSize = 64;
}

def Zn5LoadQueue : LoadQueue<Zn5Load>;

// AMD SOG Zen5, 2.12 Load-Store Unit
// A maximum of two of the memory operations can be stores.
let Super = Zn5LSU in
def Zn5Store : ProcResource<2> {
  // AMD SOG Zen5, 2.12 Load-Store Unit
  // he LS unit utilizes a 104-entry store queue (STQ) 
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

// AMD SOG Zen5, 2.11 Floating-Point Unit
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
defm : Zn5WriteResInt<WriteMove, [Zn5ALU012345], 1, [4], 1>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

def Zn5WriteMOVBE16rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012345]> {
  let Latency = Znver5Model.LoadLatency;
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMOVBE16rm], (instrs MOVBE16rm)>;

def Zn5WriteMOVBEmr : SchedWriteRes<[Zn5ALU012345, Zn5AGU0123, Zn5Store]> {
  let Latency = Znver5Model.StoreLatency;
  let ReleaseAtCycles = [4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteMOVBEmr], (instrs MOVBE16mr, MOVBE32mr, MOVBE64mr)>;

// Arithmetic.
defm : Zn5WriteResIntPair<WriteALU, [Zn5ALU012345], 1, [1], 1>; // Simple integer ALU op.

def Zn5WriteALUSlow : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteALUSlow], (instrs ADD8i8, ADD16i16, ADD32i32, ADD64i32,
                                        AND8i8, AND16i16, AND32i32, AND64i32,
                                         OR8i8,  OR16i16,  OR32i32,  OR64i32,
                                        SUB8i8, SUB16i16, SUB32i32, SUB64i32,
                                        XOR8i8, XOR16i16, XOR32i32, XOR64i32)>;

def Zn5WriteMoveExtend : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMoveExtend], (instrs MOVSX16rr16, MOVSX16rr32, MOVZX16rr16, MOVSX16rr8, MOVZX16rr8)>;

def Zn5WriteMaterialize32bitImm: SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteMaterialize32bitImm], (instrs MOV32ri, MOV32ri_alt, MOV64ri32)>;

def Zn5WritePDEP_PEXT : SchedWriteRes<[Zn5ALU1]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePDEP_PEXT], (instrs PDEP32rr, PDEP64rr,
                                          PEXT32rr, PEXT64rr)>;

defm : Zn5WriteResIntPair<WriteADC, [Zn5ALU012345], 1, [4], 1>; // Integer ALU + flags op.

def Zn5WriteADC8mr_SBB8mr : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012345, Zn5Store]> {
  let Latency = 1;
  let ReleaseAtCycles = [1, 1, 7, 1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteADC8mr_SBB8mr], (instrs ADC8mr, SBB8mr)>;

// This is for simple LEAs with one or two input operands.
defm : Zn5WriteResInt<WriteLEA, [Zn5AGU0123], 1, [1], 1>;     // LEA instructions can't fold loads.

// This write is used for slow LEA instructions.
def Zn5Write3OpsLEA : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
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

def Zn5SlowLEA16r : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 2; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [4];
  let NumMicroOps = 2;
}

def : InstRW<[Zn5SlowLEA16r], (instrs LEA16r)>;

// Integer multiplication
defm : Zn5WriteResIntPair<WriteIMul8, [Zn5Multiplier012], 3, [3], 1>; // Integer 8-bit multiplication.
defm : Zn5WriteResIntPair<WriteIMul16, [Zn5Multiplier012], 3, [3], 3, /*LoadUOps=*/1>; // Integer 16-bit multiplication.
defm : Zn5WriteResIntPair<WriteIMul16Imm, [Zn5Multiplier012], 4, [4], 2>; // Integer 16-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul16Reg, [Zn5Multiplier012], 3, [1], 1>; // Integer 16-bit multiplication by register.
defm : Zn5WriteResIntPair<WriteIMul32, [Zn5Multiplier012], 3, [3], 2>;    // Integer 32-bit multiplication.
defm : Zn5WriteResIntPair<WriteMULX32, [Zn5Multiplier012], 3, [1], 2>;    // Integer 32-bit Unsigned Multiply Without Affecting Flags.
defm : Zn5WriteResIntPair<WriteIMul32Imm, [Zn5Multiplier012], 3, [1], 1>; // Integer 32-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul32Reg, [Zn5Multiplier012], 3, [1], 1>; // Integer 32-bit multiplication by register.
defm : Zn5WriteResIntPair<WriteIMul64, [Zn5Multiplier012], 3, [3], 2>;    // Integer 64-bit multiplication.
defm : Zn5WriteResIntPair<WriteMULX64, [Zn5Multiplier012], 3, [1], 2>;    // Integer 32-bit Unsigned Multiply Without Affecting Flags.
defm : Zn5WriteResIntPair<WriteIMul64Imm, [Zn5Multiplier012], 3, [1], 1>; // Integer 64-bit multiplication by immediate.
defm : Zn5WriteResIntPair<WriteIMul64Reg, [Zn5Multiplier012], 3, [1], 1>; // Integer 64-bit multiplication by register.
defm : Zn5WriteResInt<WriteIMulHLd, [], !add(4, Znver5Model.LoadLatency), [], 0>;  // Integer multiplication, high part.
defm : Zn5WriteResInt<WriteIMulH, [], 4, [], 0>;  // Integer multiplication, high part.

defm : Zn5WriteResInt<WriteBSWAP32, [Zn5ALU012345], 1, [1], 1>; // Byte Order (Endianness) 32-bit Swap.
defm : Zn5WriteResInt<WriteBSWAP64, [Zn5ALU012345], 1, [1], 1>; // Byte Order (Endianness) 64-bit Swap.

defm : Zn5WriteResIntPair<WriteCMPXCHG, [Zn5ALU012345], 3, [12], 5>; // Compare and set, compare and swap.

def Zn5WriteCMPXCHG8rr : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 3;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 3;
}
def : InstRW<[Zn5WriteCMPXCHG8rr], (instrs CMPXCHG8rr)>;

defm : Zn5WriteResInt<WriteCMPXCHGRMW, [Zn5ALU012345], 3, [12], 6>;     // Compare and set, compare and swap.

def Zn5WriteCMPXCHG8rm_LCMPXCHG8 : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012345]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteCMPXCHG8rr.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = !add(Zn5WriteCMPXCHG8rr.NumMicroOps, 2);
}
def : InstRW<[Zn5WriteCMPXCHG8rm_LCMPXCHG8], (instrs CMPXCHG8rm, LCMPXCHG8)>;

def Zn5WriteCMPXCHG8B : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 3; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [24];
  let NumMicroOps = 19;
}
def : InstRW<[Zn5WriteCMPXCHG8B], (instrs CMPXCHG8B)>;

def Zn5WriteCMPXCHG16B_LCMPXCHG16B : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 4; // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [59];
  let NumMicroOps = 28;
}
def : InstRW<[Zn5WriteCMPXCHG16B_LCMPXCHG16B], (instrs CMPXCHG16B, LCMPXCHG16B)>;

def Zn5WriteWriteXCHGUnrenameable : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteWriteXCHGUnrenameable], (instrs XCHG8rr, XCHG16rr, XCHG16ar)>;

def Zn5WriteXCHG8rm_XCHG16rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012345]> {
  let Latency = !add(Znver5Model.LoadLatency, 3); // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 5;
}
def : InstRW<[Zn5WriteXCHG8rm_XCHG16rm], (instrs XCHG8rm, XCHG16rm)>;

def Zn5WriteXCHG32rm_XCHG64rm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5ALU012345]> {
  let Latency = !add(Znver5Model.LoadLatency, 2); // FIXME: not from llvm-exegesis
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteXCHG32rm_XCHG64rm], (instrs XCHG32rm, XCHG64rm)>;

// Integer division.
// FIXME: uops for 8-bit division measures as 2. for others it's a guess.
// FIXME: latency for 8-bit division measures as 10. for others it's a guess.
defm : Zn5WriteResIntPair<WriteDiv8, [Zn5Divider], 10, [10], 2>;
defm : Zn5WriteResIntPair<WriteDiv16, [Zn5Divider], 11, [11], 2>;
defm : Zn5WriteResIntPair<WriteDiv32, [Zn5Divider], 13, [13], 2>;
defm : Zn5WriteResIntPair<WriteDiv64, [Zn5Divider], 17, [17], 2>;
defm : Zn5WriteResIntPair<WriteIDiv8, [Zn5Divider], 10, [10], 2>;
defm : Zn5WriteResIntPair<WriteIDiv16, [Zn5Divider], 11, [11], 2>;
defm : Zn5WriteResIntPair<WriteIDiv32, [Zn5Divider], 13, [13], 2>;
defm : Zn5WriteResIntPair<WriteIDiv64, [Zn5Divider], 17, [17], 2>;

defm : Zn5WriteResIntPair<WriteBSF, [Zn5ALU1], 1, [1], 6, /*LoadUOps=*/1>; // Bit scan forward.
defm : Zn5WriteResIntPair<WriteBSR, [Zn5ALU1], 1, [1], 6, /*LoadUOps=*/1>; // Bit scan reverse.

defm : Zn5WriteResIntPair<WritePOPCNT, [Zn5ALU012345], 1, [1], 1>; // Bit population count.

def Zn5WritePOPCNT16rr : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WritePOPCNT16rr], (instrs POPCNT16rr)>;

defm : Zn5WriteResIntPair<WriteLZCNT, [Zn5ALU012345], 1, [1], 1>; // Leading zero count.

def Zn5WriteLZCNT16rr : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 1;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteLZCNT16rr], (instrs LZCNT16rr)>;

defm : Zn5WriteResIntPair<WriteTZCNT, [Zn5ALU012], 2, [1], 2>; // Trailing zero count.

def Zn5WriteTZCNT16rr : SchedWriteRes<[Zn5ALU012345]> {
  let Latency = 2;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteTZCNT16rr], (instrs TZCNT16rr)>;

defm : Zn5WriteResIntPair<WriteCMOV, [Zn5ALU345], 1, [1], 1>; // Conditional move.
defm : Zn5WriteResInt<WriteFCMOV, [Zn5ALU012345], 7, [28], 7>; // FIXME: not from llvm-exegesis // X87 conditional move.
defm : Zn5WriteResInt<WriteSETCC, [Zn5ALU345], 1, [2], 1>; // Set register based on condition code.
defm : Zn5WriteResInt<WriteSETCCStore, [Zn5ALU345, Zn5AGU0123, Zn5Store], 2, [2, 1, 1], 2>; // FIXME: latency not from llvm-exegesis
defm : Zn5WriteResInt<WriteLAHFSAHF, [Zn5ALU3], 1, [1], 1>; // Load/Store flags in AH.

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
  let ReleaseAtCycles = [2];
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
  let Latency = 3;
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
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 9;
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
defm : Zn5WriteResInt<WriteSHDrri, [Zn5ALU012], 2, [3], 4>;
defm : Zn5WriteResInt<WriteSHDrrcl, [Zn5ALU012], 2, [3], 5>;
defm : Zn5WriteResInt<WriteSHDmri, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 4], 6>;
defm : Zn5WriteResInt<WriteSHDmrcl, [Zn5AGU0123, Zn5Load, Zn5ALU012], !add(Znver5Model.LoadLatency, 2), [1, 1, 4], 6>;

// BMI1 BEXTR/BLS, BMI2 BZHI
defm : Zn5WriteResIntPair<WriteBEXTR, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteBLS, [Zn5ALU012345], 1, [1], 1, /*LoadUOps=*/1>;
defm : Zn5WriteResIntPair<WriteBZHI, [Zn5ALU012], 1, [1], 1, /*LoadUOps=*/1>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
defm : Zn5WriteResInt<WriteZero, [Zn5ALU012345], 0, [0], 1>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : Zn5WriteResIntPair<WriteJump, [Zn5BRU012], 1, [1], 1>; // FIXME: not from llvm-exegesis

// Floating point. This covers both scalar and vector operations.
defm : Zn5WriteResInt<WriteFLD0, [Zn5FPLd01, Zn5Load, Zn5FP1], !add(Znver5Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : Zn5WriteResInt<WriteFLD1, [Zn5FPLd01, Zn5Load, Zn5FP1], !add(Znver5Model.LoadLatency, 7), [1, 1, 1], 1>;
defm : Zn5WriteResInt<WriteFLDC, [Zn5FPLd01, Zn5Load, Zn5FP1], !add(Znver5Model.LoadLatency, 7), [1, 1, 1], 1>;
defm : Zn5WriteResXMM<WriteFLoad, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFLoadX, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteFLoadY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFMaskedLoad, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResYMM<WriteFMaskedLoadY, [Zn5FPLd01, Zn5Load], !add(Znver5Model.VecLoadLatency, 1), [1, 1], 1>;
defm : Zn5WriteResXMM<WriteFStore, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;

def Zn5WriteWriteFStoreMMX : SchedWriteRes<[Zn5FPSt, Zn5Store]> {
  let Latency = 2; // FIXME: not from llvm-exegesis
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

defm : Zn5WriteResXMMPair<WriteFAdd, [Zn5FPFAdd01], 3, [1], 1>;  // Floating point add/sub.

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

defm : Zn5WriteResXMMPair<WriteFAddX, [Zn5FPFAdd01], 3, [1], 1>; // Floating point add/sub (XMM).
defm : Zn5WriteResYMMPair<WriteFAddY, [Zn5FPFAdd01], 3, [1], 1>; // Floating point add/sub (YMM).
defm : Zn5WriteResZMMPair<WriteFAddZ, [Zn5FPFAdd01], 3, [2], 1>; // Floating point add/sub (ZMM).
defm : Zn5WriteResXMMPair<WriteFAdd64, [Zn5FPFAdd01], 3, [1], 1>;  // Floating point double add/sub.
defm : Zn5WriteResXMMPair<WriteFAdd64X, [Zn5FPFAdd01], 3, [1], 1>; // Floating point double add/sub (XMM).
defm : Zn5WriteResYMMPair<WriteFAdd64Y, [Zn5FPFAdd01], 3, [1], 1>; // Floating point double add/sub (YMM).
defm : Zn5WriteResZMMPair<WriteFAdd64Z, [Zn5FPFAdd01], 3, [2], 1>; // Floating point double add/sub (ZMM).
defm : Zn5WriteResXMMPair<WriteFCmp, [Zn5FPFMul01], 2, [2], 1>;  // Floating point compare.
defm : Zn5WriteResXMMPair<WriteFCmpX, [Zn5FPFMul01], 2, [1], 1>; // Floating point compare (XMM).
defm : Zn5WriteResYMMPair<WriteFCmpY, [Zn5FPFMul01], 2, [1], 1>; // Floating point compare (YMM).
defm : Zn5WriteResZMMPair<WriteFCmpZ, [Zn5FPFMul01], 2, [2], 1>; // Floating point compare (ZMM).
defm : Zn5WriteResXMMPair<WriteFCmp64, [Zn5FPFMul01], 1, [1], 1>;  // Floating point double compare.
defm : Zn5WriteResXMMPair<WriteFCmp64X, [Zn5FPFMul01], 2, [1], 1>; // Floating point double compare (XMM).
defm : Zn5WriteResYMMPair<WriteFCmp64Y, [Zn5FPFMul01], 2, [1], 1>; // Floating point double compare (YMM).
defm : Zn5WriteResZMMPair<WriteFCmp64Z, [Zn5FPFMul01], 2, [2], 1>; // Floating point double compare (ZMM).
defm : Zn5WriteResXMMPair<WriteFCom, [Zn5FPFMul01], 3, [2], 1>; // FIXME: latency not from llvm-exegesis  // Floating point compare to flags (X87).
defm : Zn5WriteResXMMPair<WriteFComX, [Zn5FPFMul01], 4, [2], 2>;  // FIXME: latency not from llvm-exegesis // Floating point compare to flags (SSE).
defm : Zn5WriteResXMMPair<WriteFMul, [Zn5FPFMul01], 3, [1], 1>;  // Floating point multiplication.
defm : Zn5WriteResXMMPair<WriteFMulX, [Zn5FPFMul01], 3, [1], 1>; // Floating point multiplication (XMM).
defm : Zn5WriteResYMMPair<WriteFMulY, [Zn5FPFMul01], 3, [1], 1>; // Floating point multiplication (YMM).
defm : Zn5WriteResZMMPair<WriteFMulZ, [Zn5FPFMul01], 3, [2], 1>; // Floating point multiplication (ZMM).
defm : Zn5WriteResXMMPair<WriteFMul64, [Zn5FPFMul01], 3, [1], 1>;  // Floating point double multiplication.
defm : Zn5WriteResXMMPair<WriteFMul64X, [Zn5FPFMul01], 3, [1], 1>; // Floating point double multiplication (XMM).
defm : Zn5WriteResYMMPair<WriteFMul64Y, [Zn5FPFMul01], 3, [1], 1>; // Floating point double multiplication (YMM).
defm : Zn5WriteResZMMPair<WriteFMul64Z, [Zn5FPFMul01], 3, [2], 1>; // Floating point double multiplication (ZMM).
defm : Zn5WriteResXMMPair<WriteFDiv, [Zn5FPFDiv], 11, [3], 1>;  // Floating point division.
defm : Zn5WriteResXMMPair<WriteFDivX, [Zn5FPFDiv], 11, [3], 1>; // Floating point division (XMM).
defm : Zn5WriteResYMMPair<WriteFDivY, [Zn5FPFDiv], 11, [3], 1>; // Floating point division (YMM).
defm : Zn5WriteResZMMPair<WriteFDivZ, [Zn5FPFDiv], 11, [6], 1>; // Floating point division (ZMM).
defm : Zn5WriteResXMMPair<WriteFDiv64, [Zn5FPFDiv], 13, [5], 1>;  // Floating point double division.
defm : Zn5WriteResXMMPair<WriteFDiv64X, [Zn5FPFDiv], 13, [5], 1>; // Floating point double division (XMM).
defm : Zn5WriteResYMMPair<WriteFDiv64Y, [Zn5FPFDiv], 13, [5], 1>; // Floating point double division (YMM).
defm : Zn5WriteResZMMPair<WriteFDiv64Z, [Zn5FPFDiv], 13, [10], 1>; // Floating point double division (ZMM).
defm : Zn5WriteResXMMPair<WriteFSqrt, [Zn5FPFDiv], 15, [5], 1>;   // Floating point square root.
defm : Zn5WriteResXMMPair<WriteFSqrtX, [Zn5FPFDiv], 15, [5], 1>;  // Floating point square root (XMM).
defm : Zn5WriteResYMMPair<WriteFSqrtY, [Zn5FPFDiv], 15, [5], 1>;  // Floating point square root (YMM).
defm : Zn5WriteResZMMPair<WriteFSqrtZ, [Zn5FPFDiv], 15, [10], 1>;  // Floating point square root (ZMM).
defm : Zn5WriteResXMMPair<WriteFSqrt64, [Zn5FPFDiv], 21, [9], 1>;  // Floating point double square root.
defm : Zn5WriteResXMMPair<WriteFSqrt64X, [Zn5FPFDiv], 21, [9], 1>; // Floating point double square root (XMM).
defm : Zn5WriteResYMMPair<WriteFSqrt64Y, [Zn5FPFDiv], 21, [9], 1>; // Floating point double square root (YMM).
defm : Zn5WriteResZMMPair<WriteFSqrt64Z, [Zn5FPFDiv], 21, [18], 1>; // Floating point double square root (ZMM).
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
defm : Zn5WriteResXMMPair<WriteDPPS, [Zn5FPFMul01], 11, [8], 8, /*LoadUOps=*/2>; // Floating point single dot product.
defm : Zn5WriteResYMMPair<WriteDPPSY, [Zn5FPFMul01], 11, [8], 7, /*LoadUOps=*/1>; // Floating point single dot product (YMM).
defm : Zn5WriteResXMMPair<WriteFSign, [Zn5FPFMul01], 1, [2], 1>; // FIXME: latency not from llvm-exegesis  // Floating point fabs/fchs.
defm : Zn5WriteResXMMPair<WriteFRnd, [Zn5FPFCvt01], 3, [1], 1>; // Floating point rounding.
defm : Zn5WriteResYMMPair<WriteFRndY, [Zn5FPFCvt01], 3, [1], 1>; // Floating point rounding (YMM).
defm : Zn5WriteResZMMPair<WriteFRndZ, [Zn5FPFCvt01], 3, [2], 1>; // Floating point rounding (ZMM).

defm : Zn5WriteResXMMPair<WriteFLogic, [Zn5FPVMisc0123], 1, [1], 1>; // Floating point and/or/xor logicals.
defm : Zn5WriteResYMMPair<WriteFLogicY, [Zn5FPVMisc0123], 1, [1], 1>; // Floating point and/or/xor logicals (YMM).
defm : Zn5WriteResZMMPair<WriteFLogicZ, [Zn5FPVMisc0123], 1, [2], 1>; // Floating point and/or/xor logicals (ZMM).
defm : Zn5WriteResXMMPair<WriteFTest, [Zn5FPFMisc12], 1, [2], 2>; // FIXME: latency not from llvm-exegesis // Floating point TEST instructions.
defm : Zn5WriteResYMMPair<WriteFTestY, [Zn5FPFMisc12], 1, [2], 2>; // FIXME: latency not from llvm-exegesis // Floating point TEST instructions (YMM).
defm : Zn5WriteResZMMPair<WriteFTestZ, [Zn5FPFMisc12], 1, [4], 1>; // FIXME: latency not from llvm-exegesis // Floating point TEST instructions (ZMM).
defm : Zn5WriteResXMMPair<WriteFShuffle, [Zn5FPVShuf01], 1, [1], 1>; // Floating point vector shuffles.
defm : Zn5WriteResYMMPair<WriteFShuffleY, [Zn5FPVShuf01], 1, [1], 1>; // Floating point vector shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteFShuffleZ, [Zn5FPVShuf01], 1, [2], 1>; // Floating point vector shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteFVarShuffle, [Zn5FPVShuf01], 3, [1], 1>; // Floating point vector variable shuffles.
defm : Zn5WriteResYMMPair<WriteFVarShuffleY, [Zn5FPVShuf01], 3, [1], 1>; // Floating point vector variable shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteFVarShuffleZ, [Zn5FPVShuf01], 3, [2], 1>; // Floating point vector variable shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteFBlend, [Zn5FPFMul01], 1, [1], 1>; // Floating point vector blends.
defm : Zn5WriteResYMMPair<WriteFBlendY, [Zn5FPFMul01], 1, [1], 1>; // Floating point vector blends (YMM).
defm : Zn5WriteResZMMPair<WriteFBlendZ, [Zn5FPFMul01], 1, [2], 1>; // Floating point vector blends (ZMM).
defm : Zn5WriteResXMMPair<WriteFVarBlend, [Zn5FPFMul01], 1, [1], 1>; // Fp vector variable blends.
defm : Zn5WriteResYMMPair<WriteFVarBlendY, [Zn5FPFMul01], 1, [1], 1>; // Fp vector variable blends (YMM).
defm : Zn5WriteResZMMPair<WriteFVarBlendZ, [Zn5FPFMul01], 1, [2], 1>; // Fp vector variable blends (ZMM).

// Horizontal Add/Sub (float and integer)
defm : Zn5WriteResXMMPair<WriteFHAdd, [Zn5FPFAdd0], 4, [2], 3>;
defm : Zn5WriteResYMMPair<WriteFHAddY, [Zn5FPFAdd0], 4, [2], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResZMMPair<WriteFHAddZ, [Zn5FPFAdd0], 6, [4], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResXMMPair<WritePHAdd, [Zn5FPVAdd0], 2, [2], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResXMMPair<WritePHAddX, [Zn5FPVAdd0], 2, [2], 3>;
defm : Zn5WriteResYMMPair<WritePHAddY, [Zn5FPVAdd0], 3, [3], 3, /*LoadUOps=*/1>;
defm : Zn5WriteResZMMPair<WritePHAddZ, [Zn5FPVAdd0], 2, [4], 3, /*LoadUOps=*/1>;

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

def Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr : SchedWriteRes<[Zn5FPFMisc0]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr], (instrs VEXTRACTF128rri, VEXTRACTI128rri)>;

def Zn5WriteVEXTRACTI128mr : SchedWriteRes<[Zn5FPFMisc0, Zn5FPSt, Zn5Store]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = !add(Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteVEXTRACTI128mr], (instrs VEXTRACTI128mri, VEXTRACTF128mri)>;

def Zn5WriteVINSERTF128rmr : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPFMisc0]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = !add(Zn5WriteVEXTRACTF128rr_VEXTRACTI128rr.NumMicroOps, 0);
}
def : InstRW<[Zn5WriteVINSERTF128rmr], (instrs VINSERTF128rmi)>;

defm : Zn5WriteResYMM<WriteVecStoreY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecStoreNT, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResYMM<WriteVecStoreNTY, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [1, 1], 1>;
defm : Zn5WriteResXMM<WriteVecMaskedStore32, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [6, 1], 18>;
defm : Zn5WriteResXMM<WriteVecMaskedStore64, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [4, 1], 10>;
defm : Zn5WriteResYMM<WriteVecMaskedStore32Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [12, 1], 42>;
defm : Zn5WriteResYMM<WriteVecMaskedStore64Y, [Zn5FPSt, Zn5Store], Znver5Model.StoreLatency, [6, 1], 18>;

defm : Zn5WriteResXMM<WriteVecMoveToGpr, [Zn5FPLd01], 1, [2], 1>;
defm : Zn5WriteResXMM<WriteVecMoveFromGpr, [Zn5FPLd01], 1, [2], 1>;

def Zn5WriteMOVMMX : SchedWriteRes<[Zn5FPLd01, Zn5FPFMisc0123]> {
  let Latency = 1;
  let ReleaseAtCycles = [1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteMOVMMX], (instrs MMX_MOVQ2FR64rr, MMX_MOVQ2DQrr)>;

def Zn5WriteMOVMMXSlow : SchedWriteRes<[Zn5FPLd01, Zn5FPFMisc0123]> {
  let Latency = 1;
  let ReleaseAtCycles = [1, 4];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteMOVMMXSlow], (instrs MMX_MOVD64rr, MMX_MOVD64to64rr)>;

defm : Zn5WriteResXMMPair<WriteVecALU, [Zn5FPVAdd0123], 1, [1], 1>;  // Vector integer ALU op, no logicals.

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

defm : Zn5WriteResXMMPair<WriteVecALUX, [Zn5FPVAdd0123], 1, [1], 1>; // Vector integer ALU op, no logicals (XMM).

def Zn5WriteVecALUXSlow : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUXSlow], (instrs PABSBrr, PABSDrr, PABSWrr,
                                            PADDSBrr, PADDSWrr, PADDUSBrr, PADDUSWrr,
                                            PAVGBrr, PAVGWrr,
                                            PSIGNBrr, PSIGNDrr, PSIGNWrr,
                                            VPABSBrr, VPABSDrr, VPABSWrr,
                                            VPADDSBrr, VPADDSWrr, VPADDUSBrr, VPADDUSWrr,
                                            VPAVGBrr, VPAVGWrr,
                                            VPCMPEQQrr,
                                            VPSIGNBrr, VPSIGNDrr, VPSIGNWrr,
                                            PSUBSBrr, PSUBSWrr, PSUBUSBrr, PSUBUSWrr, VPSUBSBrr, VPSUBSWrr, VPSUBUSBrr, VPSUBUSWrr)>;

def Zn5WriteVecOpMask : SchedWriteRes<[Zn5FPOpMask01]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecOpMask], (instrs   KADDBkk, KADDDkk, KADDQkk, KADDWkk,
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
                                            KXORBkk, KXORDkk, KXORQkk, KXORWkk)>;

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
  // TODO: All align instructions are expected to be of 4 cycle latency
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALU2Slow], (instrs VALIGNDZrri, VALIGNDZ128rri, VALIGNDZ256rri,
                                            VALIGNQZrri, VALIGNQZ128rri, VALIGNQZ256rri)
                                            >;
defm : Zn5WriteResYMMPair<WriteVecALUY, [Zn5FPVAdd0123], 1, [1], 1>; // Vector integer ALU op, no logicals (YMM).

def Zn5WriteVecALUYSlow : SchedWriteRes<[Zn5FPVAdd01]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5WriteVecALUYSlow], (instrs VPABSBYrr, VPABSDYrr, VPABSWYrr,
                                            VPADDSBYrr, VPADDSWYrr, VPADDUSBYrr, VPADDUSWYrr,
                                            VPSUBSBYrr, VPSUBSWYrr, VPSUBUSBYrr, VPSUBUSWYrr,
                                            VPAVGBYrr, VPAVGWYrr,
                                            VPCMPEQQYrr,
                                            VPSIGNBYrr, VPSIGNDYrr, VPSIGNWYrr)>;

defm : Zn5WriteResZMMPair<WriteVecALUZ, [Zn5FPVAdd0123], 1, [2], 1>; // Vector integer ALU op, no logicals (ZMM).

defm : Zn5WriteResXMMPair<WriteVecLogic, [Zn5FPVMisc0123], 1, [1], 1>;  // Vector integer and/or/xor logicals.
defm : Zn5WriteResXMMPair<WriteVecLogicX, [Zn5FPVMisc0123], 1, [1], 1>; // Vector integer and/or/xor logicals (XMM).
defm : Zn5WriteResYMMPair<WriteVecLogicY, [Zn5FPVMisc0123], 1, [1], 1>; // Vector integer and/or/xor logicals (YMM).
defm : Zn5WriteResZMMPair<WriteVecLogicZ, [Zn5FPVMisc0123], 1, [2], 1>; // Vector integer and/or/xor logicals (ZMM).
defm : Zn5WriteResXMMPair<WriteVecTest, [Zn5FPVAdd12, Zn5FPSt], 1, [1, 1], 2>;  // FIXME: latency not from llvm-exegesis // Vector integer TEST instructions.
defm : Zn5WriteResYMMPair<WriteVecTestY, [Zn5FPVAdd12, Zn5FPSt], 1, [1, 1], 2>; // FIXME: latency not from llvm-exegesis  // Vector integer TEST instructions (YMM).
defm : Zn5WriteResZMMPair<WriteVecTestZ, [Zn5FPVAdd12, Zn5FPSt], 1, [2, 2], 2>; // FIXME: latency not from llvm-exegesis  // Vector integer TEST instructions (ZMM).
defm : Zn5WriteResXMMPair<WriteVecShift, [Zn5FPVShift01], 1, [1], 1>;  // Vector integer shifts (default).
defm : Zn5WriteResXMMPair<WriteVecShiftX, [Zn5FPVShift01], 2, [2], 1>; // Vector integer shifts (XMM).
defm : Zn5WriteResYMMPair<WriteVecShiftY, [Zn5FPVShift01], 1, [1], 1>; // Vector integer shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVecShiftZ, [Zn5FPVShift01], 1, [2], 1>; // Vector integer shifts (ZMM).
defm : Zn5WriteResXMMPair<WriteVecShiftImm, [Zn5FPVShift01], 1, [1], 1>;  // Vector integer immediate shifts (default).
defm : Zn5WriteResXMMPair<WriteVecShiftImmX, [Zn5FPVShift01], 1, [1], 1>; // Vector integer immediate shifts (XMM).
defm : Zn5WriteResYMMPair<WriteVecShiftImmY, [Zn5FPVShift01], 1, [1], 1>; // Vector integer immediate shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVecShiftImmZ, [Zn5FPVShift01], 1, [2], 1>; // Vector integer immediate shifts (ZMM).
defm : Zn5WriteResXMMPair<WriteVecIMul, [Zn5FPVMul012], 3, [1], 1>;  // Vector integer multiply (default).
defm : Zn5WriteResXMMPair<WriteVecIMulX, [Zn5FPVMul012], 3, [1], 1>; // Vector integer multiply (XMM).
defm : Zn5WriteResYMMPair<WriteVecIMulY, [Zn5FPVMul012], 3, [1], 1>; // Vector integer multiply (YMM).
defm : Zn5WriteResZMMPair<WriteVecIMulZ, [Zn5FPVMul012], 3, [2], 1>; // Vector integer multiply (ZMM).
defm : Zn5WriteResXMMPair<WritePMULLD, [Zn5FPVMul012], 3, [1], 1>; // Vector PMULLD.
defm : Zn5WriteResYMMPair<WritePMULLDY, [Zn5FPVMul012], 3, [1], 1>; // Vector PMULLD (YMM).
defm : Zn5WriteResZMMPair<WritePMULLDZ, [Zn5FPVMul012], 3, [2], 1>; // Vector PMULLD (ZMM).
defm : Zn5WriteResXMMPair<WriteShuffle, [Zn5FPVShuf01], 1, [1], 1>;  // Vector shuffles.
defm : Zn5WriteResXMMPair<WriteShuffleX, [Zn5FPVShuf01], 1, [1], 1>; // Vector shuffles (XMM).
defm : Zn5WriteResYMMPair<WriteShuffleY, [Zn5FPVShuf01], 1, [1], 1>; // Vector shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteShuffleZ, [Zn5FPVShuf01], 1, [2], 1>; // Vector shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteVarShuffle, [Zn5FPVShuf01], 1, [1], 1>;  // Vector variable shuffles.
defm : Zn5WriteResXMMPair<WriteVarShuffleX, [Zn5FPVShuf01], 1, [1], 1>; // Vector variable shuffles (XMM).
defm : Zn5WriteResYMMPair<WriteVarShuffleY, [Zn5FPVShuf01], 1, [1], 1>; // Vector variable shuffles (YMM).
defm : Zn5WriteResZMMPair<WriteVarShuffleZ, [Zn5FPVShuf01], 1, [2], 1>; // Vector variable shuffles (ZMM).
defm : Zn5WriteResXMMPair<WriteBlend, [Zn5FPVMisc0123], 1, [1], 1>; // Vector blends.
defm : Zn5WriteResYMMPair<WriteBlendY, [Zn5FPVMisc0123], 1, [1], 1>; // Vector blends (YMM).
defm : Zn5WriteResZMMPair<WriteBlendZ, [Zn5FPVMisc0123], 1, [2], 1>; // Vector blends (ZMM).
defm : Zn5WriteResXMMPair<WriteVarBlend, [Zn5FPVShuf01], 1, [1], 1>; // Vector variable blends.
defm : Zn5WriteResYMMPair<WriteVarBlendY, [Zn5FPVShuf01], 1, [1], 1>; // Vector variable blends (YMM).
defm : Zn5WriteResZMMPair<WriteVarBlendZ, [Zn5FPVShuf01], 1, [2], 1>; // Vector variable blends (ZMM).
defm : Zn5WriteResXMMPair<WritePSADBW, [Zn5FPVAdd0123], 3, [2], 1>;  // Vector PSADBW.
defm : Zn5WriteResXMMPair<WritePSADBWX, [Zn5FPVAdd0123], 3, [2], 1>; // Vector PSADBW (XMM).
defm : Zn5WriteResYMMPair<WritePSADBWY, [Zn5FPVAdd0123], 3, [2], 1>; // Vector PSADBW (YMM).
defm : Zn5WriteResZMMPair<WritePSADBWZ, [Zn5FPVAdd0123], 4, [4], 1>; // Vector PSADBW (ZMM).
defm : Zn5WriteResXMMPair<WriteMPSAD, [Zn5FPVAdd0123], 4, [8], 4, /*LoadUOps=*/2>; // Vector MPSAD.
defm : Zn5WriteResYMMPair<WriteMPSADY, [Zn5FPVAdd0123], 4, [8], 3, /*LoadUOps=*/1>; // Vector MPSAD (YMM).
defm : Zn5WriteResZMMPair<WriteMPSADZ, [Zn5FPVAdd0123], 4, [16], 3, /*LoadUOps=*/1>; // Vector MPSAD (ZMM).
defm : Zn5WriteResXMMPair<WritePHMINPOS, [Zn5FPVAdd01], 3, [1], 1>;  // Vector PHMINPOS.

// Vector insert/extract operations.
defm : Zn5WriteResXMMPair<WriteVecInsert, [Zn5FPLd01], 1, [2], 2, /*LoadUOps=*/-1>; // Insert gpr to vector element.
defm : Zn5WriteResXMM<WriteVecExtract, [Zn5FPLd01], 1, [2], 2>; // Extract vector element to gpr.
defm : Zn5WriteResXMM<WriteVecExtractSt, [Zn5FPSt, Zn5Store], !add(1, Znver5Model.StoreLatency), [1, 1], 2>; // Extract vector element and store.

// MOVMSK operations.
defm : Zn5WriteResXMM<WriteFMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResXMM<WriteVecMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResYMM<WriteVecMOVMSKY, [Zn5FPVMisc2], 1, [1], 1>;
defm : Zn5WriteResXMM<WriteMMXMOVMSK, [Zn5FPVMisc2], 1, [1], 1>;

// Conversion between integer and float.
defm : Zn5WriteResXMMPair<WriteCvtSD2I, [Zn5FPFCvt01], 1, [1], 1>;  // Double -> Integer.
defm : Zn5WriteResXMMPair<WriteCvtPD2I, [Zn5FPFCvt01], 3, [2], 1>; // Double -> Integer (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPD2IY, [Zn5FPFCvt01], 3, [2], 2>; // Double -> Integer (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPD2IZ, [Zn5FPFCvt01], 3, [4], 2>; // Double -> Integer (ZMM).

def Zn5WriteCvtPD2IMMX : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
defm : Zn5WriteResXMMPair<WriteCvtSS2I, [Zn5FPFCvt01], 5, [5], 2>;  // Float -> Integer.

defm : Zn5WriteResXMMPair<WriteCvtPS2I, [Zn5FPFCvt01], 3, [1], 1>; // Float -> Integer (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPS2IY, [Zn5FPFCvt01], 4, [1], 1>; // Float -> Integer (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPS2IZ, [Zn5FPFCvt01], 4, [2], 2>; // Float -> Integer (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtI2SD, [Zn5FPFCvt01], 4, [2], 2, /*LoadUOps=*/-1>;  // Integer -> Double.
defm : Zn5WriteResXMMPair<WriteCvtI2PD, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Double (XMM).
defm : Zn5WriteResYMMPair<WriteCvtI2PDY, [Zn5FPFCvt01], 3, [2], 2, /*LoadUOps=*/-1>; // Integer -> Double (YMM).
defm : Zn5WriteResZMMPair<WriteCvtI2PDZ, [Zn5FPFCvt01], 4, [4], 4, /*LoadUOps=*/-1>; // Integer -> Double (ZMM).

def Zn5WriteCvtI2PDMMX : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 2;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 2;
}

defm : Zn5WriteResXMMPair<WriteCvtI2SS, [Zn5FPFCvt01], 3, [2], 2, /*LoadUOps=*/-1>;  // Integer -> Float.
defm : Zn5WriteResXMMPair<WriteCvtI2PS, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Float (XMM).
defm : Zn5WriteResYMMPair<WriteCvtI2PSY, [Zn5FPFCvt01], 3, [1], 1>; // Integer -> Float (YMM).
defm : Zn5WriteResZMMPair<WriteCvtI2PSZ, [Zn5FPFCvt01], 3, [2], 2>; // Integer -> Float (ZMM).

def Zn5WriteCvtI2PSMMX : SchedWriteRes<[Zn5FPFCvt01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}

defm : Zn5WriteResXMMPair<WriteCvtSS2SD, [Zn5FPFCvt01], 3, [1], 1>;  // Float -> Double size conversion.
defm : Zn5WriteResXMMPair<WriteCvtPS2PD, [Zn5FPFCvt01], 3, [1], 1>; // Float -> Double size conversion (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPS2PDY, [Zn5FPFCvt01], 4, [2], 2, /*LoadUOps=*/-1>; // Float -> Double size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPS2PDZ, [Zn5FPFCvt01], 6, [4], 4, /*LoadUOps=*/-1>; // Float -> Double size conversion (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtSD2SS, [Zn5FPFCvt01], 3, [1], 1>;  // Double -> Float size conversion.
defm : Zn5WriteResXMMPair<WriteCvtPD2PS, [Zn5FPFCvt01], 3, [1], 1>; // Double -> Float size conversion (XMM).
defm : Zn5WriteResYMMPair<WriteCvtPD2PSY, [Zn5FPFCvt01], 6, [2], 2>; // Double -> Float size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPD2PSZ, [Zn5FPFCvt01], 6, [4], 4>; // Double -> Float size conversion (ZMM).

defm : Zn5WriteResXMMPair<WriteCvtPH2PS, [Zn5FPFCvt01], 3, [1], 1>; // Half -> Float size conversion.
defm : Zn5WriteResYMMPair<WriteCvtPH2PSY, [Zn5FPFCvt01], 4, [2], 2, /*LoadUOps=*/-1>; // Half -> Float size conversion (YMM).
defm : Zn5WriteResZMMPair<WriteCvtPH2PSZ, [Zn5FPFCvt01], 4, [4], 4, /*LoadUOps=*/-1>; // Half -> Float size conversion (ZMM).

defm : Zn5WriteResXMM<WriteCvtPS2PH, [Zn5FPFCvt01], 3, [2], 1>; // Float -> Half size conversion.
defm : Zn5WriteResYMM<WriteCvtPS2PHY, [Zn5FPFCvt01], 6, [2], 2>; // Float -> Half size conversion (YMM).
defm : Zn5WriteResZMM<WriteCvtPS2PHZ, [Zn5FPFCvt01], 6, [2], 2>; // Float -> Half size conversion (ZMM).

defm : Zn5WriteResXMM<WriteCvtPS2PHSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(3, Znver5Model.StoreLatency), [1, 1, 1], 2>; // Float -> Half + store size conversion.
defm : Zn5WriteResYMM<WriteCvtPS2PHYSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(6, Znver5Model.StoreLatency), [2, 1, 1], 3>; // Float -> Half + store size conversion (YMM).
defm : Zn5WriteResYMM<WriteCvtPS2PHZSt, [Zn5FPFCvt01, Zn5FPSt, Zn5Store], !add(6, Znver5Model.StoreLatency), [2, 1, 1], 3>; // Float -> Half + store size conversion (ZMM).

// CRC32 instruction.
defm : Zn5WriteResIntPair<WriteCRC32, [Zn5ALU1], 3, [1], 1>;

def Zn5WriteSHA1MSG1rr : SchedWriteRes<[Zn5FPU0123]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
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
  let Latency = 1;
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
  let Latency = 2;
  let ReleaseAtCycles = [3];
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
  let Latency = 3;
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
defm : Zn5WriteResXMMPair<WritePCmpIStrM, [Zn5FPVAdd0123], 6, [8], 3, /*LoadUOps=*/1>;
// Packed Compare Explicit Length Strings, Return Mask
defm : Zn5WriteResXMMPair<WritePCmpEStrM, [Zn5FPVAdd0123], 6, [12], 7, /*LoadUOps=*/5>;
// Packed Compare Implicit Length Strings, Return Index
defm : Zn5WriteResXMMPair<WritePCmpIStrI, [Zn5FPVAdd0123], 2, [8], 4>;
// Packed Compare Explicit Length Strings, Return Index
defm : Zn5WriteResXMMPair<WritePCmpEStrI, [Zn5FPVAdd0123], 6, [12], 8, /*LoadUOps=*/4>;

// AES instructions.
defm : Zn5WriteResXMMPair<WriteAESDecEnc, [Zn5FPAES01], 4, [1], 1>; // Decryption, encryption.
defm : Zn5WriteResXMMPair<WriteAESIMC, [Zn5FPAES01], 4, [1], 1>; // InvMixColumn.
defm : Zn5WriteResXMMPair<WriteAESKeyGen, [Zn5FPAES01], 4, [1], 1>; // Key Generation.

// Carry-less multiplication instructions.
defm : Zn5WriteResXMMPair<WriteCLMul, [Zn5FPCLM01], 4, [4], 4>;

// EMMS/FEMMS
defm : Zn5WriteResInt<WriteEMMS, [Zn5ALU012345], 2, [1], 1>; // FIXME: latency not from llvm-exegesis

// Load/store MXCSR
defm : Zn5WriteResInt<WriteLDMXCSR, [Zn5AGU0123, Zn5Load, Zn5ALU012345], !add(Znver5Model.LoadLatency, 1), [1, 1, 6], 1>; // FIXME: latency not from llvm-exegesis
defm : Zn5WriteResInt<WriteSTMXCSR, [Zn5ALU012345, Zn5AGU0123, Zn5Store], !add(1, Znver5Model.StoreLatency), [60, 1, 1], 2>; // FIXME: latency not from llvm-exegesis

// Catch-all for expensive system instructions.
defm : Zn5WriteResInt<WriteSystem, [Zn5ALU012345], 100, [100], 100>;

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
defm : Zn5WriteResYMMPair<WriteShuffle256, [Zn5FPVShuf], 1, [1], 1>; // 256-bit width vector shuffles.

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

def Zn5WriteVPERMPSYrr : SchedWriteRes<[Zn5FPVShuf]> {
  let Latency = 7;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteVPERMPSYrr], (instrs VPERMPSYrr)>;

def Zn5WriteVPERMPSYrm : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPVShuf]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVPERMPSYrr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteVPERMPSYrr.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteVPERMPSYrm], (instrs VPERMPSYrm)>;

def Zn5WriteVPERMYri : SchedWriteRes<[Zn5FPVShuf]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteVPERMYri], (instrs VPERMPDYri, VPERMQYri)>;

def Zn5WriteVPERMPDYmi : SchedWriteRes<[Zn5AGU0123, Zn5Load, Zn5FPVShuf]> {
  let Latency = !add(Znver5Model.LoadLatency, Zn5WriteVPERMYri.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = !add(Zn5WriteVPERMYri.NumMicroOps, 1);
}
def : InstRW<[Zn5WriteVPERMPDYmi], (instrs VPERMPDYmi)>;

def Zn5WriteVPERMDYrr : SchedWriteRes<[Zn5FPVShuf]> {
  let Latency = 5;
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

defm : Zn5WriteResYMMPair<WriteVPMOV256, [Zn5FPVShuf01], 4, [3], 2, /*LoadUOps=*/-1>; // 256-bit width packed vector width-changing move.
defm : Zn5WriteResYMMPair<WriteVarShuffle256, [Zn5FPVShuf01], 1, [1], 2>; // 256-bit width vector variable shuffles.
defm : Zn5WriteResXMMPair<WriteVarVecShift, [Zn5FPVShift01], 1, [1], 1>; // Variable vector shifts.
defm : Zn5WriteResYMMPair<WriteVarVecShiftY, [Zn5FPVShift01], 1, [1], 1>; // Variable vector shifts (YMM).
defm : Zn5WriteResZMMPair<WriteVarVecShiftZ, [Zn5FPVShift01], 1, [2], 2>; // Variable vector shifts (ZMM).

// Old microcoded instructions that nobody use.
defm : Zn5WriteResInt<WriteMicrocoded, [Zn5ALU012345], 100, [100], 100>;

// Fence instructions.
defm : Zn5WriteResInt<WriteFence, [Zn5ALU012345], 1, [100], 1>;

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
defm : Zn5WriteResInt<WriteNop, [Zn5ALU012345], 0, [1], 1>; // FIXME: latency not from llvm-exegesis


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
                                               MOVSX32rr32)>;

def Zn5WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                               XCHG64rr, XCHG64ar)>;

defm : Zn5WriteResInt<WriteXCHG, [Zn5ALU012345], 0, [8], 2>;        // Compare+Exchange - TODO RMW support.

defm : Zn5WriteResXMM<WriteFMoveX, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteFMoveY, [], 0, [], 1>;
defm : Zn5WriteResYMM<WriteFMoveZ, [], 0, [], 1>;

defm : Zn5WriteResXMM<WriteVecMove, [Zn5FPFMisc0123], 1, [1], 1>; // MMX
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
	"VRANGE(S|P)(S|D)(Z|Z128|Z256?)rri(b?)k","VRANGE(S|P)(S|D)(Z?|Z128?|Z256?)rri(b?)kz"
	)>;

// SCALE & REDUCE instructions
def Zn5WriteSCALErr: SchedWriteRes<[Zn5FPFMisc23]> {
  let Latency = 6;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 2;
}
def : InstRW<[Zn5WriteSCALErr], (instregex
        "V(SCALEF|REDUCE)(S|P)(S|D)(Z?|Z128?|Z256?)(rr|rrb|rrkz|rrik|rrikz|rri)(_Int?)",
        "(V?)REDUCE(PD|PS|SD|SS)(Z?|Z128?)(rri|rrikz|rrib)"
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
        "(V?)P(SHUFB)(Y|Z|Z128|Z256?)(rr|rrk|rrkz)",
        "(V?)P(ROL|ROR)(D|Q|VD|VQ)(Z?|Z128?|Z256?)(rr|rrk|rrkz)",
        "(V?)P(ROL|ROR)(D|Q|VD|VQ)(Z256?)(ri|rik|rikz)",
        "(V?)P(ROL|ROR)(D|Q)(Z?|Z128?)(ri|rik|rikz)",
	"VPSHUFBITQMBZ128rr", "VFMSUB231SSZrkz_Int"
	)>;

def Zn5WriteSHIFTri: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 1;
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
def Zn5MOVDUPZ: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVDUPZ], (instregex
        "(V?)VMOVDDUP(Z|Z128|Z256)(rr|rrk|rrkz)"
	)>;

def Zn5MOVS: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVS], (instregex
        "(V?)PMOV(SX|ZX)(BD|BQ|BW|WD|WQ|DQ)(Y?|Z128?|Z256?)(rr|rrk|rrkz)",
        "(V?)PMOV(S?|US?)(DB|DW|QB|QD|QW|WB)(Z128|Z256)(rr|rrk|rrkz)"
	)>;

def Zn5MOVSZ: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVSZ], (instregex
        "(V?)PMOV(SX|ZX)(BD|BQ|BW|WD|WQ|DQ)Z(rr|rrk|rrkz)"
	)>;

def Zn5MOVSrr: SchedWriteRes<[Zn5FPFMisc12]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5MOVSrr], (instregex
        "(V?)PMOV(S?|US?)(DB|DW|QB|QD|QW|WB)Z(rr|rrk|rrkz)"
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
        "(V?)PTEST(N?)(MB|MD|MQ|MW)(Z256?)(rr|rrk)"
	)>;

def Zn5VPTESTZ: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 5;
  let ReleaseAtCycles = [5];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VPTESTZ], (instregex
        "(V?)PTEST(N?)(MB|MD|MQ|MW)(Z?)(rrk)"
	)>;

// CONFLICT Instructions
def Zn5CONFLICTZ128: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5CONFLICTZ128], (instregex
        "VPCONFLICT(D|Q)(Z128)(rr|rrk|rrkz)"
	)>;

def Zn5CONFLICTrr: SchedWriteRes<[Zn5FPFMisc01,Zn5FPFMisc12,Zn5FPFMisc23]> {
  let Latency = 6;
  let ReleaseAtCycles = [2,2,2];
  let NumMicroOps = 4;
}
def : InstRW<[Zn5CONFLICTrr], (instregex
        "VPCONFLICT(D|Q)(Z|Z256)(rr|rrkz)"
	)>;

// RSQRT Instructions
def Zn5VRSQRT14PDZ256: SchedWriteRes<[Zn5FPFMisc01]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5VRSQRT14PDZ256], (instregex
        "VRSQRT14(PD|PS)(Z?|Z128?|Z256?)(r|rr|rk|rrk|rkz|rrkz)"
	)>;


// PERM Instructions
def Zn5PERMILP: SchedWriteRes<[Zn5FPFMisc123]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[Zn5PERMILP], (instregex
        "VPERMILP(S|D)(Y|Z|Z128|Z256)(rr|rrk|rrkz)"
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
	"VPEXPAND(B|W)Z256(rr|rrk|rrkz)"
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
	"VPEXPAND(B|W)Z(rr|rrk|rrkz)",
	"VPERMP(S|D)Z(rr|rrk|rrkz)"
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
                                           VXORPSZ128rr,
                                           VXORPDZ128rr,
                                           VANDNPSrr, VANDNPDrr,
                                           VANDNPSZ128rr,
                                           VANDNPDZ128rr)>;

def Zn5WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[Zn5WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                            VXORPSZ256rr,
                                            VXORPDZ256rr,
                                            VANDNPSYrr, VANDNPDYrr,
                                            VANDNPSZ256rr,
                                            VANDNPDZ256rr)>;

def Zn5WriteFZeroIdiomZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicZ]>
]>;
def : InstRW<[Zn5WriteFZeroIdiomZ], (instrs VXORPSZrr, VXORPDZrr,
                                            VANDNPSZrr, VANDNPDZrr)>;

def Zn5WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;
// NOTE: PXORrr,PANDNrr are not zero-cycle!
def : InstRW<[Zn5WriteVZeroIdiomLogicX], (instrs VPXORrr,
                                                 VPXORDZ128rr,
                                                 VPXORQZ128rr,
                                                 VPANDNrr,
                                                 VPANDNDZ128rr,
                                                 VPANDNQZ128rr)>;

def Zn5WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomLogicY], (instrs VPXORYrr,
                                                 VPXORDZ256rr,
                                                 VPXORQZ256rr,
                                                 VPANDNYrr,
                                                 VPANDNDZ256rr,
                                                 VPANDNQZ256rr)>;

def Zn5WriteVZeroIdiomLogicZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicZ]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomLogicZ], (instrs VPXORDZrr, VPXORQZrr,
                                                 VPANDNDZrr, VPANDNQZrr)>;

def Zn5WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;
// NOTE: PSUBBrr, PSUBWrr, PSUBDrr, PSUBQrr,
//       PCMPGTBrr, PCMPGTWrr, PCMPGTDrr, PCMPGTQrr are not zero-cycle!
def : InstRW<[Zn5WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPSUBBZ128rr, VPSUBWZ128rr, VPSUBDZ128rr, VPSUBQZ128rr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr,
                     VPCMPGTBZ128rr, VPCMPGTWZ128rr,
                     VPCMPGTDZ128rr, VPCMPGTQZ128rr)>;

def Zn5WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomALUY],
             (instrs VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
                     VPSUBBZ256rr, VPSUBWZ256rr, VPSUBDZ256rr, VPSUBQZ256rr,
                     VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr,
                     VPCMPGTBZ256rr, VPCMPGTWZ256rr,
                     VPCMPGTDZ256rr, VPCMPGTQZ256rr)>;

def Zn5WriteVZeroIdiomALUZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [Zn5WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUZ]>
]>;
def : InstRW<[Zn5WriteVZeroIdiomALUY],
             (instrs VPSUBBZrr, VPSUBWZrr, VPSUBDZrr, VPSUBQZrr,
                     VPCMPGTBZrr, VPCMPGTWZrr, VPCMPGTDZrr, VPCMPGTQZrr)>;

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

  // AVX ZMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSZrr, VXORPDZrr,
    VXORPSZ128rr, VXORPDZ128rr, VXORPSZ256rr, VXORPDZ256rr,
    VANDNPSZrr, VANDNPDZrr,
    VANDNPSZ128rr, VANDNPDZ128rr, VANDNPSZ256rr, VANDNPDZ256rr,

    // int variants.
    VPCMPGTBZrr, VPCMPGTWZrr, VPCMPGTDZrr, VPCMPGTQZrr,
    VPCMPGTBZ128rr, VPCMPGTWZ128rr, VPCMPGTDZ128rr, VPCMPGTQZ128rr,
    VPCMPGTBZ256rr, VPCMPGTWZ256rr, VPCMPGTDZ256rr, VPCMPGTQZ256rr,
    VPANDNDZrr, VPANDNQZrr,
    VPANDNDZ128rr, VPANDNQZ128rr, VPANDNDZ256rr, VPANDNQZ256rr,
    VPXORDZrr, VPXORQZrr,
    VPXORDZ128rr, VPXORQZ128rr, VPXORDZ256rr, VPXORQZ256rr,
    VPSUBBZrr, VPSUBWZrr, VPSUBDZrr, VPSUBQZrr,
    VPSUBBZ128rr, VPSUBWZ128rr, VPSUBDZ128rr, VPSUBQZ128rr,
    VPSUBBZ256rr, VPSUBWZ256rr, VPSUBDZ256rr, VPSUBQZ256rr,
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

