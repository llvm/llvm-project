//==- RISCVSchedBullet.td - Bullet Scheduling Definitions ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Bullet machine model for scheduling and other instruction cost heuristics.
def BulletModel : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since Bullet is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtV, HasStdExtZvamo, HasStdExtZvlsseg];
}

// The Bullet microarchitecure has two pipelines: A and B.
// Pipe A can handle memory, integer alu and vector operations.
// Pipe B can handle integer alu, control flow, integer multiply and divide,
// and floating point computation.
let SchedModel = BulletModel in {
let BufferSize = 0 in {
def BulletPipeA       : ProcResource<1>;
def BulletPipeB       : ProcResource<1>;
}

let BufferSize = 1 in {
def BulletIDiv        : ProcResource<1> { let Super = BulletPipeB; } // Int Division
def BulletFDiv        : ProcResource<1> { let Super = BulletPipeB; } // FP Division/Sqrt
}

def BulletPipeAB : ProcResGroup<[BulletPipeA, BulletPipeB]>;

// Branching
def : WriteRes<WriteJmp, [BulletPipeB]>;
def : WriteRes<WriteJal, [BulletPipeB]>;
def : WriteRes<WriteJalr, [BulletPipeB]>;
def : WriteRes<WriteJmpReg, [BulletPipeB]>;

// Integer arithmetic and logic
let Latency = 3 in {
def : WriteRes<WriteIALU, [BulletPipeAB]>;
def : WriteRes<WriteIALU32, [BulletPipeAB]>;
def : WriteRes<WriteShift, [BulletPipeAB]>;
def : WriteRes<WriteShift32, [BulletPipeAB]>;
}

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [BulletPipeB]>;
def : WriteRes<WriteIMul32, [BulletPipeB]>;
}

// Integer division
def : WriteRes<WriteIDiv, [BulletPipeB, BulletIDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}
def : WriteRes<WriteIDiv32,  [BulletPipeB, BulletIDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}

// Memory
def : WriteRes<WriteSTB, [BulletPipeA]>;
def : WriteRes<WriteSTH, [BulletPipeA]>;
def : WriteRes<WriteSTW, [BulletPipeA]>;
def : WriteRes<WriteSTD, [BulletPipeA]>;
def : WriteRes<WriteFST32, [BulletPipeA]>;
def : WriteRes<WriteFST64, [BulletPipeA]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [BulletPipeA]>;
def : WriteRes<WriteLDH, [BulletPipeA]>;
def : WriteRes<WriteLDW, [BulletPipeA]>;
def : WriteRes<WriteLDWU, [BulletPipeA]>;
def : WriteRes<WriteLDD, [BulletPipeA]>;
}

let Latency = 2 in {
def : WriteRes<WriteFLD32, [BulletPipeA]>;
def : WriteRes<WriteFLD64, [BulletPipeA]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [BulletPipeA]>;
def : WriteRes<WriteAtomicSTD, [BulletPipeA]>;

let Latency = 3 in {
def : WriteRes<WriteAtomicW, [BulletPipeA]>;
def : WriteRes<WriteAtomicD, [BulletPipeA]>;
def : WriteRes<WriteAtomicLDW, [BulletPipeA]>;
def : WriteRes<WriteAtomicLDD, [BulletPipeA]>;
}

// Single precision.
let Latency = 5 in {
def : WriteRes<WriteFALU32, [BulletPipeB]>;
def : WriteRes<WriteFMul32, [BulletPipeB]>;
def : WriteRes<WriteFMulAdd32, [BulletPipeB]>;
def : WriteRes<WriteFMulSub32, [BulletPipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ32, [BulletPipeB]>;
def : WriteRes<WriteFMinMax32, [BulletPipeB]>;
}

def : WriteRes<WriteFDiv32, [BulletPipeB, BulletFDiv]> { let Latency = 27;
                                                         let ResourceCycles = [1, 26]; }
def : WriteRes<WriteFSqrt32, [BulletPipeB, BulletFDiv]> { let Latency = 27;
                                                          let ResourceCycles = [1, 26]; }

// Double precision
let Latency = 7 in {
def : WriteRes<WriteFALU64, [BulletPipeB]>;
def : WriteRes<WriteFMul64, [BulletPipeB]>;
def : WriteRes<WriteFMulAdd64, [BulletPipeB]>;
def : WriteRes<WriteFMulSub64, [BulletPipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ64, [BulletPipeB]>;
def : WriteRes<WriteFMinMax64, [BulletPipeB]>;
}

def : WriteRes<WriteFDiv64, [BulletPipeB, BulletFDiv]> { let Latency = 56;
                                                         let ResourceCycles = [1, 55]; }
def : WriteRes<WriteFSqrt64, [BulletPipeB, BulletFDiv]> { let Latency = 56;
                                                          let ResourceCycles = [1, 55]; }

// Conversions
let Latency = 3 in {
def : WriteRes<WriteFCvtI32ToF32, [BulletPipeB]>;
def : WriteRes<WriteFCvtI32ToF64, [BulletPipeB]>;
def : WriteRes<WriteFCvtI64ToF32, [BulletPipeB]>;
def : WriteRes<WriteFCvtI64ToF64, [BulletPipeB]>;
def : WriteRes<WriteFCvtF32ToI32, [BulletPipeB]>;
def : WriteRes<WriteFCvtF32ToI64, [BulletPipeB]>;
def : WriteRes<WriteFCvtF32ToF64, [BulletPipeB]>;
def : WriteRes<WriteFCvtF64ToI32, [BulletPipeB]>;
def : WriteRes<WriteFCvtF64ToI64, [BulletPipeB]>;
def : WriteRes<WriteFCvtF64ToF32, [BulletPipeB]>;

def : WriteRes<WriteFClass32, [BulletPipeB]>;
def : WriteRes<WriteFClass64, [BulletPipeB]>;
def : WriteRes<WriteFCmp32, [BulletPipeB]>;
def : WriteRes<WriteFCmp64, [BulletPipeB]>;
def : WriteRes<WriteFMovI32ToF32, [BulletPipeB]>;
def : WriteRes<WriteFMovF32ToI32, [BulletPipeB]>;
def : WriteRes<WriteFMovI64ToF64, [BulletPipeB]>;
def : WriteRes<WriteFMovF64ToI64, [BulletPipeB]>;
}

// Others
def : WriteRes<WriteCSR, [BulletPipeB]>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;


//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShift, 0>;
def : ReadAdvance<ReadShift32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFALU32, 0>;
def : ReadAdvance<ReadFALU64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMulAdd32, 0>;
def : ReadAdvance<ReadFMulSub32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMulAdd64, 0>;
def : ReadAdvance<ReadFMulSub64, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;
}
