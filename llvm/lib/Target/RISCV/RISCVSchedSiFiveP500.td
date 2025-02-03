//==- RISCVSchedSiFiveP500.td - SiFiveP500 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

def SiFiveP500Model : SchedMachineModel {
  let IssueWidth = 3;         // 3 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 96; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let CompleteModel = false;
}

// The SiFiveP500 microarchitecure has 7 pipelines:
// Three pipelines for integer operations.
// Two pipelines for FPU operations.
// One pipeline for Load operations.
// One pipeline for Store operations.
let SchedModel = SiFiveP500Model in {

def SiFiveP500IEXQ0       : ProcResource<1>;
def SiFiveP500IEXQ1       : ProcResource<1>;
def SiFiveP500IEXQ2       : ProcResource<1>;
def SiFiveP500FEXQ0       : ProcResource<1>;
def SiFiveP500FEXQ1       : ProcResource<1>;
def SiFiveP500Load        : ProcResource<1>;
def SiFiveP500Store       : ProcResource<1>;

def SiFiveP500IntArith    : ProcResGroup<[SiFiveP500IEXQ0, SiFiveP500IEXQ1, SiFiveP500IEXQ2]>;
defvar SiFiveP500Branch   = SiFiveP500IEXQ0;
defvar SiFiveP500SYS      = SiFiveP500IEXQ1;
defvar SiFiveP500CMOV     = SiFiveP500IEXQ1;
defvar SiFiveP500MulI2F   = SiFiveP500IEXQ2;
def SiFiveP500Div         : ProcResource<1>;

def SiFiveP500FloatArith  : ProcResGroup<[SiFiveP500FEXQ0, SiFiveP500FEXQ1]>;
defvar SiFiveP500F2I      = SiFiveP500FEXQ0;
def SiFiveP500FloatDiv    : ProcResource<1>;

let Latency = 1 in {
// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFiveP500IntArith]>;
def : WriteRes<WriteIALU32, [SiFiveP500IntArith]>;
def : WriteRes<WriteShiftImm, [SiFiveP500IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFiveP500IntArith]>;
def : WriteRes<WriteShiftReg, [SiFiveP500IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFiveP500IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFiveP500Branch]>;
def : WriteRes<WriteJal, [SiFiveP500Branch]>;
def : WriteRes<WriteJalr, [SiFiveP500Branch]>;
}

// CMOV
def P500WriteCMOV : SchedWriteRes<[SiFiveP500Branch, SiFiveP500CMOV]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[P500WriteCMOV], (instrs PseudoCCMOVGPRNoX0)>;

let Latency = 3 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFiveP500MulI2F]>;
def : WriteRes<WriteIMul32, [SiFiveP500MulI2F]>;
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFiveP500MulI2F]>;
def : WriteRes<WriteCPOP32, [SiFiveP500MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFiveP500MulI2F, SiFiveP500Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIDiv32, [SiFiveP500MulI2F, SiFiveP500Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Integer remainder
def : WriteRes<WriteIRem, [SiFiveP500MulI2F, SiFiveP500Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIRem32, [SiFiveP500MulI2F, SiFiveP500Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

let Latency = 1 in {
// Bitmanip
def : WriteRes<WriteRotateImm, [SiFiveP500IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFiveP500IntArith]>;
def : WriteRes<WriteRotateReg, [SiFiveP500IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFiveP500IntArith]>;

def : WriteRes<WriteCLZ, [SiFiveP500IntArith]>;
def : WriteRes<WriteCLZ32, [SiFiveP500IntArith]>;
def : WriteRes<WriteCTZ, [SiFiveP500IntArith]>;
def : WriteRes<WriteCTZ32, [SiFiveP500IntArith]>;

def : WriteRes<WriteORCB, [SiFiveP500IntArith]>;
def : WriteRes<WriteIMinMax, [SiFiveP500IntArith]>;

def : WriteRes<WriteREV8, [SiFiveP500IntArith]>;

def : WriteRes<WriteSHXADD, [SiFiveP500IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFiveP500IntArith]>;
}

// Memory
let Latency = 1 in {
def : WriteRes<WriteSTB, [SiFiveP500Store]>;
def : WriteRes<WriteSTH, [SiFiveP500Store]>;
def : WriteRes<WriteSTW, [SiFiveP500Store]>;
def : WriteRes<WriteSTD, [SiFiveP500Store]>;
def : WriteRes<WriteFST16, [SiFiveP500Store]>;
def : WriteRes<WriteFST32, [SiFiveP500Store]>;
def : WriteRes<WriteFST64, [SiFiveP500Store]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFiveP500Load]>;
def : WriteRes<WriteLDH, [SiFiveP500Load]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDW, [SiFiveP500Load]>;
def : WriteRes<WriteLDD, [SiFiveP500Load]>;
}

let Latency = 6 in {
def : WriteRes<WriteFLD16, [SiFiveP500Load]>;
def : WriteRes<WriteFLD32, [SiFiveP500Load]>;
def : WriteRes<WriteFLD64, [SiFiveP500Load]>;
}

// Atomic memory
let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [SiFiveP500Store]>;
def : WriteRes<WriteAtomicSTD, [SiFiveP500Store]>;
def : WriteRes<WriteAtomicW, [SiFiveP500Load]>;
def : WriteRes<WriteAtomicD, [SiFiveP500Load]>;
def : WriteRes<WriteAtomicLDW, [SiFiveP500Load]>;
def : WriteRes<WriteAtomicLDD, [SiFiveP500Load]>;
}

// Floating point
let Latency = 4 in {
def : WriteRes<WriteFAdd16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP500FloatArith]>;

def : WriteRes<WriteFMul16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP500FloatArith]>;

def : WriteRes<WriteFMA16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMA32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMA64, [SiFiveP500FloatArith]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFSGNJ32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFSGNJ64, [SiFiveP500FloatArith]>;

def : WriteRes<WriteFMinMax16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFiveP500FloatArith]>;
}

// Half precision.
def : WriteRes<WriteFDiv16, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFSqrt16, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Single precision.
def : WriteRes<WriteFDiv32, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFSqrt32, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Double precision
def : WriteRes<WriteFDiv64, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}
def : WriteRes<WriteFSqrt64, [SiFiveP500FEXQ1, SiFiveP500FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFiveP500F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFiveP500FloatArith]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFiveP500FloatArith]>;

def : WriteRes<WriteFClass16, [SiFiveP500F2I]>;
def : WriteRes<WriteFClass32, [SiFiveP500F2I]>;
def : WriteRes<WriteFClass64, [SiFiveP500F2I]>;
def : WriteRes<WriteFCmp16, [SiFiveP500F2I]>;
def : WriteRes<WriteFCmp32, [SiFiveP500F2I]>;
def : WriteRes<WriteFCmp64, [SiFiveP500F2I]>;
def : WriteRes<WriteFMovI16ToF16, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFMovF16ToI16, [SiFiveP500F2I]>;
def : WriteRes<WriteFMovI32ToF32, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFMovF32ToI32, [SiFiveP500F2I]>;
def : WriteRes<WriteFMovI64ToF64, [SiFiveP500MulI2F]>;
def : WriteRes<WriteFMovF64ToI64, [SiFiveP500F2I]>;
}

// Others
def : WriteRes<WriteCSR, [SiFiveP500SYS]>;
def : WriteRes<WriteNop, []>;

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

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
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
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
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// Bitmanip
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsfvcp;
}
