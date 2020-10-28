//==- RISCVSchedRocket64.td - Rocket Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Rocket machine model for scheduling and other instruction cost heuristics.
def Rocket64Model : SchedMachineModel {
  let MicroOpBufferSize = 0; // Explicitly set to zero since Rocket is in-order.
  let IssueWidth = 1;        // 1 micro-ops are dispatched per cycle.
  let LoadLatency = 3;
  let MispredictPenalty = 3;
  let UnsupportedFeatures = [HasStdExtV, HasStdExtZvlsseg, HasStdExtZvamo];
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since
// Rocket is in-order.

let BufferSize = 0 in {
def Rocket64UnitALU        : ProcResource<1>; // Int ALU
def Rocket64UnitIMul       : ProcResource<1>; // Int Multiply
def Rocket64UnitMem        : ProcResource<1>; // Load/Store
def Rocket64UnitB          : ProcResource<1>; // Branch

def Rocket64UnitFPALU      : ProcResource<1>; // FP ALU
}

let BufferSize = 1 in {
def Rocket64UnitIDiv       : ProcResource<1>; // Int Division
def Rocket64UnitFPDivSqrt  : ProcResource<1>; // FP Divide/Sqrt
}

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types which both map the ProcResources and
// set the latency.

let SchedModel = Rocket64Model in {

def : WriteRes<WriteJmp, [Rocket64UnitB]>;
def : WriteRes<WriteJal, [Rocket64UnitB]>;
def : WriteRes<WriteJalr, [Rocket64UnitB]>;
def : WriteRes<WriteJmpReg, [Rocket64UnitB]>;

def : WriteRes<WriteIALU32, [Rocket64UnitALU]>;
def : WriteRes<WriteIALU, [Rocket64UnitALU]>;
def : WriteRes<WriteShift32, [Rocket64UnitALU]>;
def : WriteRes<WriteShift, [Rocket64UnitALU]>;

let Latency = 4 in {
def : WriteRes<WriteIMul, [Rocket64UnitIMul]>;
def : WriteRes<WriteIMul32, [Rocket64UnitIMul]>;
}

// Integer divide varies based on operand magnitude and sign; worse case latency is 34.
def : WriteRes<WriteIDiv32, [Rocket64UnitIDiv]> {
  let Latency = 34;
  let ResourceCycles = [34];
}
def : WriteRes<WriteIDiv, [Rocket64UnitIDiv]> {
  let Latency = 33;
  let ResourceCycles = [33];
}

// Memory
def : WriteRes<WriteSTB, [Rocket64UnitMem]>;
def : WriteRes<WriteSTH, [Rocket64UnitMem]>;
def : WriteRes<WriteSTW, [Rocket64UnitMem]>;
def : WriteRes<WriteSTD, [Rocket64UnitMem]>;
def : WriteRes<WriteFST32, [Rocket64UnitMem]>;
def : WriteRes<WriteFST64, [Rocket64UnitMem]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [Rocket64UnitMem]>;
def : WriteRes<WriteLDH, [Rocket64UnitMem]>;
def : WriteRes<WriteCSR, [Rocket64UnitALU]>;
}

let Latency = 2 in {
def : WriteRes<WriteLDW, [Rocket64UnitMem]>;
def : WriteRes<WriteLDWU, [Rocket64UnitMem]>;
def : WriteRes<WriteLDD, [Rocket64UnitMem]>;
def : WriteRes<WriteFLD32, [Rocket64UnitMem]>;
def : WriteRes<WriteFLD64, [Rocket64UnitMem]>;

def : WriteRes<WriteAtomicW, [Rocket64UnitMem]>;
def : WriteRes<WriteAtomicD, [Rocket64UnitMem]>;

def : WriteRes<WriteAtomicLDW, [Rocket64UnitMem]>;
def : WriteRes<WriteAtomicLDD, [Rocket64UnitMem]>;
}

def : WriteRes<WriteAtomicSTW, [Rocket64UnitMem]>;
def : WriteRes<WriteAtomicSTD, [Rocket64UnitMem]>;

// Most FP single precision operations are 4 cycles
let Latency = 4 in {
def : WriteRes<WriteFALU32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFSGNJ32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMinMax32, [Rocket64UnitFPALU]>;
}

let Latency = 6 in {
// Most FP double precision operations are 6 cycles
def : WriteRes<WriteFALU64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFSGNJ64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMinMax64, [Rocket64UnitFPALU]>;
}

// Conversion instructions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF32, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtI32ToF64, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF32, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF64, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI32, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI64, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI32, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI64, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF32ToF64, [Rocket32UnitFPALU]>;
def : WriteRes<WriteFCvtF64ToF32, [Rocket32UnitFPALU]>;

def : WriteRes<WriteFClass32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFClass64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFCmp32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFCmp64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMovF32ToI32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMovI32ToF32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMovF64ToI64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMovI64ToF64, [Rocket64UnitFPALU]>;
}

let Latency = 5 in {
def : WriteRes<WriteFMul32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMulAdd32, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMulSub32, [Rocket64UnitFPALU]>;
}

let Latency = 7 in {
def : WriteRes<WriteFMul64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMulAdd64, [Rocket64UnitFPALU]>;
def : WriteRes<WriteFMulSub64, [Rocket64UnitFPALU]>;
}

// FP Divide unit on Rocket is not pipelined, so set resource cycles to latency
let Latency = 20, ResourceCycles = [20] in {
def : WriteRes<WriteFDiv32, [Rocket64UnitFPDivSqrt]>;
def : WriteRes<WriteFDiv64, [Rocket64UnitFPDivSqrt]>;
}

// FP Sqrt unit on Rocket is not pipelined, so set resource cycles to latency
def : WriteRes<WriteFSqrt32, [Rocket64UnitFPDivSqrt]> { let Latency = 20;
                                                        let ResourceCycles = [20]; }
def : WriteRes<WriteFSqrt64, [Rocket64UnitFPDivSqrt]> { let Latency = 25;
                                                        let ResourceCycles = [25]; }

def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types with cycles.
// Dummy definitions for RocketCore.
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
