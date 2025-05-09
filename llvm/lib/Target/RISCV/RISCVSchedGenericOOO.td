//===-- RISCVSchedGenericOOO.td - Generic OOO Processor ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// We assume that:
// * 6-issue out-of-order CPU with 192 ROB entries.
// * Units:
//   * IXU (Integer ALU Unit): 4 units, only one can execute mul/div.
//   * FXU (Floating-point Unit): 2 units.
//   * LSU (Load/Store Unit): 2 units.
// * Latency:
//   * Integer instructions: 1 cycle.
//   * Multiplication instructions: 4 cycles.
//   * Division instructions: 13-21 cycles.
//   * Floating-point instructions: 2-6 cycles.
//   * Floating-point fdiv/fsqrt instructions: 9-21 cycles.
//   * Load/Store:
//     * IXU: 4 cycles.
//     * FXU: 4 cycles.
// * Integer/floating-point/vector div/rem/sqrt/... are non-pipelined.
//
// TODO: Add vector scheduling.
//===----------------------------------------------------------------------===//

def GenericOOOModel : SchedMachineModel {
  int IssueWidth = 6;
  int MicroOpBufferSize = 192;
  int LoadLatency = 4;
  int MispredictPenalty = 8;
  let CompleteModel = 0;
}

let SchedModel = GenericOOOModel in {
//===----------------------------------------------------------------------===//
// Resource groups
//===----------------------------------------------------------------------===//
def GenericOOOBranch : ProcResource<1>;
def GenericOOOMulDiv : ProcResource<1>;
def GenericOOOInt : ProcResource<2>;
def GenericOOOALU
    : ProcResGroup<[GenericOOOBranch, GenericOOOMulDiv, GenericOOOInt]>;
def GenericOOOLSU : ProcResource<2>;
def GenericOOOFMulDiv : ProcResource<1>;
def GenericOOOFloat : ProcResource<1>;
def GenericOOOFPU : ProcResGroup<[GenericOOOFMulDiv, GenericOOOFloat]>;

//===----------------------------------------------------------------------===//
// Branches
//===----------------------------------------------------------------------===//
def : WriteRes<WriteJmp, [GenericOOOBranch]>;
def : WriteRes<WriteJalr, [GenericOOOBranch]>;
def : WriteRes<WriteJal, [GenericOOOBranch]>;

//===----------------------------------------------------------------------===//
// Integer arithmetic and logic
//===----------------------------------------------------------------------===//
def : WriteRes<WriteIALU, [GenericOOOALU]>;
def : WriteRes<WriteIALU32, [GenericOOOALU]>;
def : WriteRes<WriteShiftImm, [GenericOOOALU]>;
def : WriteRes<WriteShiftImm32, [GenericOOOALU]>;
def : WriteRes<WriteShiftReg, [GenericOOOALU]>;
def : WriteRes<WriteShiftReg32, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Integer multiplication
//===----------------------------------------------------------------------===//
let Latency = 4 in {
  def : WriteRes<WriteIMul, [GenericOOOMulDiv]>;
  def : WriteRes<WriteIMul32, [GenericOOOMulDiv]>;
}

//===----------------------------------------------------------------------===//
// Integer division
//===----------------------------------------------------------------------===//
def : WriteRes<WriteIDiv32, [GenericOOOMulDiv]> {
  let Latency = 13;
  let ReleaseAtCycles = [13];
}
def : WriteRes<WriteIDiv, [GenericOOOMulDiv]> {
  let Latency = 21;
  let ReleaseAtCycles = [21];
}
def : WriteRes<WriteIRem32, [GenericOOOMulDiv]> {
  let Latency = 13;
  let ReleaseAtCycles = [13];
}
def : WriteRes<WriteIRem, [GenericOOOMulDiv]> {
  let Latency = 21;
  let ReleaseAtCycles = [21];
}

//===----------------------------------------------------------------------===//
// Integer memory
//===----------------------------------------------------------------------===//
// Load
let Latency = 4 in {
  def : WriteRes<WriteLDB, [GenericOOOLSU]>;
  def : WriteRes<WriteLDH, [GenericOOOLSU]>;
  def : WriteRes<WriteLDW, [GenericOOOLSU]>;
  def : WriteRes<WriteLDD, [GenericOOOLSU]>;
}

// Store
def : WriteRes<WriteSTB, [GenericOOOLSU]>;
def : WriteRes<WriteSTH, [GenericOOOLSU]>;
def : WriteRes<WriteSTW, [GenericOOOLSU]>;
def : WriteRes<WriteSTD, [GenericOOOLSU]>;

//===----------------------------------------------------------------------===//
// Atomic
//===----------------------------------------------------------------------===//
let Latency = 4 in {
  def : WriteRes<WriteAtomicLDW, [GenericOOOLSU]>;
  def : WriteRes<WriteAtomicLDD, [GenericOOOLSU]>;
}

let Latency = 5 in {
  def : WriteRes<WriteAtomicW, [GenericOOOLSU]>;
  def : WriteRes<WriteAtomicD, [GenericOOOLSU]>;
}

def : WriteRes<WriteAtomicSTW, [GenericOOOLSU]>;
def : WriteRes<WriteAtomicSTD, [GenericOOOLSU]>;

//===----------------------------------------------------------------------===//
// Floating-point
//===----------------------------------------------------------------------===//
// Floating-point load
let Latency = 4 in {
  def : WriteRes<WriteFLD32, [GenericOOOLSU]>;
  def : WriteRes<WriteFLD64, [GenericOOOLSU]>;
}

// Floating-point store
def : WriteRes<WriteFST32, [GenericOOOLSU]>;
def : WriteRes<WriteFST64, [GenericOOOLSU]>;

// Arithmetic and logic
let Latency = 2 in {
  def : WriteRes<WriteFAdd32, [GenericOOOFPU]>;
  def : WriteRes<WriteFAdd64, [GenericOOOFPU]>;
}

def : WriteRes<WriteFSGNJ32, [GenericOOOFPU]>;
def : WriteRes<WriteFSGNJ64, [GenericOOOFPU]>;
def : WriteRes<WriteFMinMax32, [GenericOOOFPU]>;
def : WriteRes<WriteFMinMax64, [GenericOOOFPU]>;

// Compare
let Latency = 2 in {
  def : WriteRes<WriteFCmp32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCmp64, [GenericOOOFPU]>;
}

// Multiplication
let Latency = 4 in {
  def : WriteRes<WriteFMul32, [GenericOOOFMulDiv]>;
  def : WriteRes<WriteFMul64, [GenericOOOFMulDiv]>;
}

// FMA
let Latency = 6 in {
  def : WriteRes<WriteFMA32, [GenericOOOFMulDiv]>;
  def : WriteRes<WriteFMA64, [GenericOOOFMulDiv]>;
}

// Division
let Latency = 13, ReleaseAtCycles = [13] in {
  def : WriteRes<WriteFDiv32, [GenericOOOFMulDiv]>;
  def : WriteRes<WriteFSqrt32, [GenericOOOFMulDiv]>;
}

let Latency = 17, ReleaseAtCycles = [17] in {
  def : WriteRes<WriteFDiv64, [GenericOOOFMulDiv]>;
  def : WriteRes<WriteFSqrt64, [GenericOOOFMulDiv]>;
}

// Conversions
let Latency = 2 in {
  def : WriteRes<WriteFCvtI32ToF32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtI32ToF64, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtI64ToF32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtI64ToF64, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFCvtF32ToI32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF32ToI64, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFCvtF64ToI32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF64ToI64, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFCvtF64ToF32, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF32ToF64, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFMovI32ToF32, [GenericOOOFPU]>;
  def : WriteRes<WriteFMovI64ToF64, [GenericOOOFPU]>;
  def : WriteRes<WriteFMovF32ToI32, [GenericOOOFPU]>;
  def : WriteRes<WriteFMovF64ToI64, [GenericOOOFPU]>;
}

// Classify
def : WriteRes<WriteFClass32, [GenericOOOFPU]>;
def : WriteRes<WriteFClass64, [GenericOOOFPU]>;

//===----------------------------------------------------------------------===//
// Zicsr extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteCSR, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zabha extension
//===----------------------------------------------------------------------===//
let Latency = 5 in {
  def : WriteRes<WriteAtomicB, [GenericOOOLSU]>;
  def : WriteRes<WriteAtomicH, [GenericOOOLSU]>;
}

//===----------------------------------------------------------------------===//
// Zba extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteSHXADD, [GenericOOOALU]>;
def : WriteRes<WriteSHXADD32, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zbb extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteCLZ, [GenericOOOALU]>;
def : WriteRes<WriteCTZ, [GenericOOOALU]>;
def : WriteRes<WriteCPOP, [GenericOOOALU]>;
def : WriteRes<WriteCLZ32, [GenericOOOALU]>;
def : WriteRes<WriteCTZ32, [GenericOOOALU]>;
def : WriteRes<WriteCPOP32, [GenericOOOALU]>;
def : WriteRes<WriteRotateReg, [GenericOOOALU]>;
def : WriteRes<WriteRotateImm, [GenericOOOALU]>;
def : WriteRes<WriteRotateReg32, [GenericOOOALU]>;
def : WriteRes<WriteRotateImm32, [GenericOOOALU]>;
def : WriteRes<WriteREV8, [GenericOOOALU]>;
def : WriteRes<WriteORCB, [GenericOOOALU]>;
def : WriteRes<WriteIMinMax, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zbc extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteCLMUL, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zbs extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteSingleBit, [GenericOOOALU]>;
def : WriteRes<WriteSingleBitImm, [GenericOOOALU]>;
def : WriteRes<WriteBEXT, [GenericOOOALU]>;
def : WriteRes<WriteBEXTI, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zbkb extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteBREV8, [GenericOOOALU]>;
def : WriteRes<WritePACK, [GenericOOOALU]>;
def : WriteRes<WritePACK32, [GenericOOOALU]>;
def : WriteRes<WriteZIP, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zbkx extension
//===----------------------------------------------------------------------===//
def : WriteRes<WriteXPERM, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Zfa extension
//===----------------------------------------------------------------------===//
let Latency = 2 in {
  def : WriteRes<WriteFRoundF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFRoundF32, [GenericOOOFPU]>;
  def : WriteRes<WriteFRoundF64, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFLI16, [GenericOOOFPU]>;
  def : WriteRes<WriteFLI32, [GenericOOOFPU]>;
  def : WriteRes<WriteFLI64, [GenericOOOFPU]>;
}

//===----------------------------------------------------------------------===//
// Zfh extension
//===----------------------------------------------------------------------===//
// Zfhmin
// Load/Store
let Latency = 4 in
def : WriteRes<WriteFLD16, [GenericOOOLSU]>;
def : WriteRes<WriteFST16, [GenericOOOLSU]>;

// Conversions
let Latency = 2 in {
  def : WriteRes<WriteFCvtF16ToF64, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF64ToF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF32ToF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF16ToF32, [GenericOOOFPU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFMovI16ToF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFMovF16ToI16, [GenericOOOFPU]>;
}

// Other than Zfhmin
let Latency = 2 in {
  def : WriteRes<WriteFCvtI64ToF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtI32ToF16, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF16ToI64, [GenericOOOFPU]>;
  def : WriteRes<WriteFCvtF16ToI32, [GenericOOOFPU]>;
}

// Arithmetic and logic
let Latency = 2 in
def : WriteRes<WriteFAdd16, [GenericOOOFPU]>;

def : WriteRes<WriteFSGNJ16, [GenericOOOFPU]>;
def : WriteRes<WriteFMinMax16, [GenericOOOFPU]>;

// Compare
let Latency = 2 in
def : WriteRes<WriteFCmp16, [GenericOOOFPU]>;

// Multiplication
let Latency = 4 in
def : WriteRes<WriteFMul16, [GenericOOOFMulDiv]>;

// FMA
let Latency = 6 in
def : WriteRes<WriteFMA16, [GenericOOOFMulDiv]>;

// Division
let Latency = 9, ReleaseAtCycles = [9] in {
  def : WriteRes<WriteFDiv16, [GenericOOOFMulDiv]>;
  def : WriteRes<WriteFSqrt16, [GenericOOOFMulDiv]>;
}

// Classify
def : WriteRes<WriteFClass16, [GenericOOOFPU]>;

//===----------------------------------------------------------------------===//
// Misc
//===----------------------------------------------------------------------===//
let Latency = 0 in
def : WriteRes<WriteNop, [GenericOOOALU]>;

//===----------------------------------------------------------------------===//
// Bypass and advance
//===----------------------------------------------------------------------===//
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
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
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

// Zabha
def : ReadAdvance<ReadAtomicBA, 0>;
def : ReadAdvance<ReadAtomicBD, 0>;
def : ReadAdvance<ReadAtomicHA, 0>;
def : ReadAdvance<ReadAtomicHD, 0>;

// Zba extension
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;

// Zbb extension
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
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;

// Zbc extension
def : ReadAdvance<ReadCLMUL, 0>;

// Zbs extension
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;

// Zbkb
def : ReadAdvance<ReadBREV8, 0>;
def : ReadAdvance<ReadPACK, 0>;
def : ReadAdvance<ReadPACK32, 0>;
def : ReadAdvance<ReadZIP, 0>;

// Zbkx
def : ReadAdvance<ReadXPERM, 0>;

// Zfa extension
def : ReadAdvance<ReadFRoundF32, 0>;
def : ReadAdvance<ReadFRoundF64, 0>;
def : ReadAdvance<ReadFRoundF16, 0>;

// Zfh extension
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;

def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
//===----------------------------------------------------------------------===//
defm : UnsupportedSchedV;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsfvcp;
}
