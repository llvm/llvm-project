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
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructions];
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

//Short forward branch
def : WriteRes<WriteSFB, [SiFive7PipeA, SiFive7PipeB]> {
  let Latency = 3;
  let NumMicroOps = 2;
}

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

// Bitmanip
let Latency = 3 in {
// Rotates are in the late-B ALU.
def : WriteRes<WriteRotateImm, [SiFive7PipeB]>;
def : WriteRes<WriteRotateImm32, [SiFive7PipeB]>;
def : WriteRes<WriteRotateReg, [SiFive7PipeB]>;
def : WriteRes<WriteRotateReg32, [SiFive7PipeB]>;

// clz[w]/ctz[w] are in the late-B ALU.
def : WriteRes<WriteCLZ, [SiFive7PipeB]>;
def : WriteRes<WriteCLZ32, [SiFive7PipeB]>;
def : WriteRes<WriteCTZ, [SiFive7PipeB]>;
def : WriteRes<WriteCTZ32, [SiFive7PipeB]>;

// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFive7PipeB]>;
def : WriteRes<WriteCPOP32, [SiFive7PipeB]>;

// orc.b is in the late-B ALU.
def : WriteRes<WriteORCB, [SiFive7PipeB]>;

// rev8 is in the late-A and late-B ALUs.
def : WriteRes<WriteREV8, [SiFive7PipeAB]>;

// shNadd[.uw] is on the early-B and late-B ALUs.
def : WriteRes<WriteSHXADD, [SiFive7PipeB]>;
def : WriteRes<WriteSHXADD32, [SiFive7PipeB]>;
}

// Memory
def : WriteRes<WriteSTB, [SiFive7PipeA]>;
def : WriteRes<WriteSTH, [SiFive7PipeA]>;
def : WriteRes<WriteSTW, [SiFive7PipeA]>;
def : WriteRes<WriteSTD, [SiFive7PipeA]>;
def : WriteRes<WriteFST16, [SiFive7PipeA]>;
def : WriteRes<WriteFST32, [SiFive7PipeA]>;
def : WriteRes<WriteFST64, [SiFive7PipeA]>;

let Latency = 3 in {
def : WriteRes<WriteLDB, [SiFive7PipeA]>;
def : WriteRes<WriteLDH, [SiFive7PipeA]>;
def : WriteRes<WriteLDW, [SiFive7PipeA]>;
def : WriteRes<WriteLDD, [SiFive7PipeA]>;
}

let Latency = 2 in {
def : WriteRes<WriteFLD16, [SiFive7PipeA]>;
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

// Half precision.
let Latency = 5 in {
def : WriteRes<WriteFAdd16, [SiFive7PipeB]>;
def : WriteRes<WriteFMul16, [SiFive7PipeB]>;
def : WriteRes<WriteFMA16, [SiFive7PipeB]>;
}
let Latency = 3 in {
def : WriteRes<WriteFSGNJ16, [SiFive7PipeB]>;
def : WriteRes<WriteFMinMax16, [SiFive7PipeB]>;
}

let Latency = 14, ResourceCycles = [1, 13] in {
def :  WriteRes<WriteFDiv16, [SiFive7PipeB, SiFive7FDiv]>;
def :  WriteRes<WriteFSqrt16, [SiFive7PipeB, SiFive7FDiv]>;
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
def : WriteRes<WriteFCvtI32ToF16, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFive7PipeB]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFive7PipeB]>;

def : WriteRes<WriteFClass16, [SiFive7PipeB]>;
def : WriteRes<WriteFClass32, [SiFive7PipeB]>;
def : WriteRes<WriteFClass64, [SiFive7PipeB]>;
def : WriteRes<WriteFCmp16, [SiFive7PipeB]>;
def : WriteRes<WriteFCmp32, [SiFive7PipeB]>;
def : WriteRes<WriteFCmp64, [SiFive7PipeB]>;
def : WriteRes<WriteFMovI16ToF16, [SiFive7PipeB]>;
def : WriteRes<WriteFMovF16ToI16, [SiFive7PipeB]>;
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
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA64, 0>;
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

def : ReadAdvance<ReadSFB, 0>;

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 2>;
def : ReadAdvance<ReadVSETVL, 2>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
// LMUL Aware
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm : LMULReadAdvance<"ReadVIALUV", 0>;
defm : LMULReadAdvance<"ReadVIALUX", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm : LMULReadAdvance<"ReadVExtV", 0>;
defm : LMULReadAdvance<"ReadVICALUV", 0>;
defm : LMULReadAdvance<"ReadVICALUX", 0>;
defm : LMULReadAdvance<"ReadVShiftV", 0>;
defm : LMULReadAdvance<"ReadVShiftX", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm : LMULReadAdvance<"ReadVICmpV", 0>;
defm : LMULReadAdvance<"ReadVICmpX", 0>;
defm : LMULReadAdvance<"ReadVIMulV", 0>;
defm : LMULReadAdvance<"ReadVIMulX", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm : LMULReadAdvance<"ReadVIMergeV", 0>;
defm : LMULReadAdvance<"ReadVIMergeX", 0>;
defm : LMULReadAdvance<"ReadVIMovV", 0>;
defm : LMULReadAdvance<"ReadVIMovX", 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 14. Vector Floating-Point Instructions
defm "" : LMULReadAdvance<"ReadVFALUV", 0>;
defm "" : LMULReadAdvance<"ReadVFALUF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMIotV", 0>;

// 17. Vector Permutation Instructions
defm "" : LMULReadAdvance<"ReadVIMovVX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXX", 0>;
defm "" : LMULReadAdvance<"ReadVFMovVF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFV", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFX", 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// LMUL Aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

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
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
}
