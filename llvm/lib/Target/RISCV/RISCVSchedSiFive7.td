//==- RISCVSchedSiFive7.td - SiFive7 Scheduling Definitions --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SiFive7 machine model for scheduling and other instruction cost heuristics.
def SiFive7Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since SiFive7 is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVInstructions];
}

// The SiFive7 microarchitecture has two pipelines: A and B.
// Pipe A can handle memory, integer alu and vector operations.
// Pipe B can handle integer alu, control flow, integer multiply and divide,
// and floating point computation.
let SchedModel = SiFive7Model in {
let BufferSize = 0 in {
def SiFive7PipeA       : ProcResource<1>;
def SiFive7PipeB       : ProcResource<1>;
}

let BufferSize = 1 in {
def SiFive7IDiv        : ProcResource<1> { let Super = SiFive7PipeB; } // Int Division
def SiFive7FDiv        : ProcResource<1> { let Super = SiFive7PipeB; } // FP Division/Sqrt
}

def SiFive7PipeAB : ProcResGroup<[SiFive7PipeA, SiFive7PipeB]>;

// Branching
def : WriteRes<WriteJmp, [SiFive7PipeB]>;
def : WriteRes<WriteJal, [SiFive7PipeB]>;
def : WriteRes<WriteJalr, [SiFive7PipeB]>;
def : WriteRes<WriteJmpReg, [SiFive7PipeB]>;

// Integer arithmetic and logic
let Latency = 3 in {
def : WriteRes<WriteIALU, [SiFive7PipeAB]>;
def : WriteRes<WriteIALU32, [SiFive7PipeAB]>;
def : WriteRes<WriteShiftImm, [SiFive7PipeAB]>;
def : WriteRes<WriteShiftImm32, [SiFive7PipeAB]>;
def : WriteRes<WriteShiftReg, [SiFive7PipeAB]>;
def : WriteRes<WriteShiftReg32, [SiFive7PipeAB]>;
}

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [SiFive7PipeB]>;
def : WriteRes<WriteIMul32, [SiFive7PipeB]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFive7PipeB, SiFive7IDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}
def : WriteRes<WriteIDiv32,  [SiFive7PipeB, SiFive7IDiv]> {
  let Latency = 16;
  let ResourceCycles = [1, 15];
}

// Memory
def : WriteRes<WriteSTB, [SiFive7PipeA]>;
def : WriteRes<WriteSTH, [SiFive7PipeA]>;
def : WriteRes<WriteSTW, [SiFive7PipeA]>;
def : WriteRes<WriteSTD, [SiFive7PipeA]>;
def : WriteRes<WriteFST32, [SiFive7PipeA]>;
def : WriteRes<WriteFST64, [SiFive7PipeA]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [SiFive7PipeA]>;
def : WriteRes<WriteLDH, [SiFive7PipeA]>;
def : WriteRes<WriteLDW, [SiFive7PipeA]>;
def : WriteRes<WriteLDWU, [SiFive7PipeA]>;
def : WriteRes<WriteLDD, [SiFive7PipeA]>;
}

let Latency = 2 in {
def : WriteRes<WriteFLD32, [SiFive7PipeA]>;
def : WriteRes<WriteFLD64, [SiFive7PipeA]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFive7PipeA]>;
def : WriteRes<WriteAtomicSTD, [SiFive7PipeA]>;

let Latency = 3 in {
def : WriteRes<WriteAtomicW, [SiFive7PipeA]>;
def : WriteRes<WriteAtomicD, [SiFive7PipeA]>;
def : WriteRes<WriteAtomicLDW, [SiFive7PipeA]>;
def : WriteRes<WriteAtomicLDD, [SiFive7PipeA]>;
}

// Single precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd32, [SiFive7PipeB]>;
def : WriteRes<WriteFMul32, [SiFive7PipeB]>;
def : WriteRes<WriteFMA32, [SiFive7PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ32, [SiFive7PipeB]>;
def : WriteRes<WriteFMinMax32, [SiFive7PipeB]>;
}

def : WriteRes<WriteFDiv32, [SiFive7PipeB, SiFive7FDiv]> { let Latency = 27;
                                                         let ResourceCycles = [1, 26]; }
def : WriteRes<WriteFSqrt32, [SiFive7PipeB, SiFive7FDiv]> { let Latency = 27;
                                                          let ResourceCycles = [1, 26]; }

// Double precision
let Latency = 7 in {
def : WriteRes<WriteFAdd64, [SiFive7PipeB]>;
def : WriteRes<WriteFMul64, [SiFive7PipeB]>;
def : WriteRes<WriteFMA64, [SiFive7PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ64, [SiFive7PipeB]>;
def : WriteRes<WriteFMinMax64, [SiFive7PipeB]>;
}

def : WriteRes<WriteFDiv64, [SiFive7PipeB, SiFive7FDiv]> { let Latency = 56;
                                                         let ResourceCycles = [1, 55]; }
def : WriteRes<WriteFSqrt64, [SiFive7PipeB, SiFive7FDiv]> { let Latency = 56;
                                                          let ResourceCycles = [1, 55]; }

// Conversions
let Latency = 3 in {
def : WriteRes<WriteFCvtI32ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive7PipeB]>;

def : WriteRes<WriteFClass32, [SiFive7PipeB]>;
def : WriteRes<WriteFClass64, [SiFive7PipeB]>;
def : WriteRes<WriteFCmp32, [SiFive7PipeB]>;
def : WriteRes<WriteFCmp64, [SiFive7PipeB]>;
def : WriteRes<WriteFMovI32ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFMovF32ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFMovI64ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFMovF64ToI64, [SiFive7PipeB]>;
}

// Others
def : WriteRes<WriteCSR, [SiFive7PipeB]>;
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
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
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
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA64, 0>;
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

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbe;
defm : UnsupportedSchedZbf;
defm : UnsupportedSchedZbm;
defm : UnsupportedSchedZbp;
defm : UnsupportedSchedZbr;
defm : UnsupportedSchedZfh;
}
