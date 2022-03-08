//===-- RISCVSchedMIPSP8700.td - MIPS RISC-V Processor -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V processor by MIPS.
//===----------------------------------------------------------------------===//

def MIPSP8700Model : SchedMachineModel {
  int IssueWidth = 4;
  int MicroOpBufferSize = 96; // as per the specification
  int LoadLatency = 4;
  int MispredictPenalty = 8; // TODO: Estimated
  let CompleteModel = 0;
}

let SchedModel = MIPSP8700Model in {

// Handle ALQ Pipelines.
def p8700ALQ : ProcResource<1> { let BufferSize = 16; }
def p8700IssueALU : ProcResource<1> { let Super = p8700ALQ; }


// Handle AGQ Pipelines.
def p8700AGQ : ProcResource<3> { let BufferSize = 16; }
def p8700IssueAL2 : ProcResource<1> { let Super = p8700AGQ; }
def p8700IssueCTISTD : ProcResource<1> { let Super = p8700AGQ; }
def p8700IssueLDST : ProcResource<1> { let Super = p8700AGQ; }
def p8700GpDiv : ProcResource<1>;
def p8700GpMul : ProcResource<1>;
def p8700WriteEitherALU : ProcResGroup<[p8700IssueALU, p8700IssueAL2]>;

let Latency = 1 in {
def : WriteRes<WriteIALU, [p8700WriteEitherALU]>;
def : WriteRes<WriteIALU32, [p8700WriteEitherALU]>;
def : WriteRes<WriteShiftImm, [p8700WriteEitherALU]>;
def : WriteRes<WriteShiftImm32, [p8700WriteEitherALU]>;
def : WriteRes<WriteShiftReg, [p8700WriteEitherALU]>;
def : WriteRes<WriteShiftReg32, [p8700WriteEitherALU]>;

// Handle zba.
def : WriteRes<WriteSHXADD, [p8700WriteEitherALU]>;
def : WriteRes<WriteSHXADD32, [p8700WriteEitherALU]>;

// Handle zbb.
def : WriteRes<WriteRotateReg, [p8700WriteEitherALU]>;
def : WriteRes<WriteRotateImm, [p8700WriteEitherALU]>;
def : WriteRes<WriteCLZ, [p8700WriteEitherALU]>;
def : WriteRes<WriteCTZ, [p8700WriteEitherALU]>;
def : WriteRes<WriteCPOP, [p8700WriteEitherALU]>;
def : WriteRes<WriteRotateReg32, [p8700WriteEitherALU]>;
def : WriteRes<WriteRotateImm32, [p8700WriteEitherALU]>;
def : WriteRes<WriteCLZ32, [p8700WriteEitherALU]>;
def : WriteRes<WriteCTZ32, [p8700WriteEitherALU]>;
def : WriteRes<WriteCPOP32, [p8700WriteEitherALU]>;
def : WriteRes<WriteREV8, [p8700WriteEitherALU]>;
def : WriteRes<WriteORCB, [p8700WriteEitherALU]>;
def : WriteRes<WriteIMinMax, []>;
}

let Latency = 0 in {
def : WriteRes<WriteNop, [p8700WriteEitherALU]>;
}

let Latency = 4 in {
def : WriteRes<WriteLDB, [p8700IssueLDST]>;
def : WriteRes<WriteLDH, [p8700IssueLDST]>;
def : WriteRes<WriteLDW, [p8700IssueLDST]>;
def : WriteRes<WriteLDD, [p8700IssueLDST]>;

def : WriteRes<WriteAtomicW, [p8700IssueLDST]>;
def : WriteRes<WriteAtomicD, [p8700IssueLDST]>;
def : WriteRes<WriteAtomicLDW, [p8700IssueLDST]>;
def : WriteRes<WriteAtomicLDD, [p8700IssueLDST]>;
}

let Latency = 8 in {
def : WriteRes<WriteFLD32, [p8700IssueLDST]>;
def : WriteRes<WriteFLD64, [p8700IssueLDST]>;
}

let Latency = 3 in {
def : WriteRes<WriteSTB, [p8700IssueLDST]>;
def : WriteRes<WriteSTH, [p8700IssueLDST]>;
def : WriteRes<WriteSTW, [p8700IssueLDST]>;
def : WriteRes<WriteSTD, [p8700IssueLDST]>;

def : WriteRes<WriteAtomicSTW, [p8700IssueLDST]>;
def : WriteRes<WriteAtomicSTD, [p8700IssueLDST]>;
}

let Latency = 1 in {
def : WriteRes<WriteFST32, [p8700IssueLDST]>;
def : WriteRes<WriteFST64, [p8700IssueLDST]>;
}

let Latency = 7 in {
def : WriteRes<WriteFMovI32ToF32, [p8700IssueLDST]>;
def : WriteRes<WriteFMovF32ToI32, [p8700IssueLDST]>;
def : WriteRes<WriteFMovI64ToF64, [p8700IssueLDST]>;
def : WriteRes<WriteFMovF64ToI64, [p8700IssueLDST]>;
}

let Latency = 4 in {
def : WriteRes<WriteIMul, [p8700GpMul]>;
def : WriteRes<WriteIMul32, [p8700GpMul]>;
}

let Latency = 8, ReleaseAtCycles = [5] in {
def : WriteRes<WriteIDiv, [p8700GpDiv]>;
def : WriteRes<WriteIDiv32,  [p8700GpDiv]>;
}

def : WriteRes<WriteIRem, []>;
def : WriteRes<WriteIRem32, []>;

// Handle CTISTD Pipeline.
let Latency = 1 in {
def : WriteRes<WriteJmp, [p8700IssueCTISTD]>;
def : WriteRes<WriteJmpReg, [p8700IssueCTISTD]>;
}

let Latency = 2 in {
def : WriteRes<WriteJal, [p8700IssueCTISTD]>;
def : WriteRes<WriteJalr, [p8700IssueCTISTD]>;
}

// Handle FPU Pipelines.
def p8700FPQ : ProcResource<3> { let BufferSize = 16; }
def p8700IssueFPUS : ProcResource<1> { let Super = p8700FPQ; }
def p8700IssueFPUL : ProcResource<1> { let Super = p8700FPQ; }
def p8700IssueFPULoad : ProcResource<1> { let Super = p8700FPQ; }
def p8700FpuApu       : ProcResource<1>;
def p8700FpuLong      : ProcResource<1>;

let Latency = 4, ReleaseAtCycles = [1, 1] in {
def : WriteRes<WriteFCvtI32ToF32, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtI32ToF64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtI64ToF32, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtI64ToF64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF32ToI32, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF32ToI64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF32ToF64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF64ToI32, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF64ToI64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFCvtF64ToF32, [p8700IssueFPUL, p8700FpuApu]>;

def : WriteRes<WriteFAdd32, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFAdd64, [p8700IssueFPUL, p8700FpuApu]>;
}

let Latency = 2, ReleaseAtCycles = [1, 1] in {
def : WriteRes<WriteFSGNJ32, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFMinMax32, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFSGNJ64, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFMinMax64, [p8700IssueFPUS, p8700FpuApu]>;

def : WriteRes<WriteFCmp32, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFCmp64, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFClass32, [p8700IssueFPUS, p8700FpuApu]>;
def : WriteRes<WriteFClass64, [p8700IssueFPUS, p8700FpuApu]>;
}

let Latency = 8, ReleaseAtCycles = [1, 1] in {
def : WriteRes<WriteFMA32, [p8700FpuLong, p8700FpuApu]>;
def : WriteRes<WriteFMA64, [p8700FpuLong, p8700FpuApu]>;
}

let Latency = 5, ReleaseAtCycles = [1, 1] in {
def : WriteRes<WriteFMul32, [p8700FpuLong, p8700FpuApu]>;
def : WriteRes<WriteFMul64, [p8700FpuLong, p8700FpuApu]>;
}

let Latency = 17, ReleaseAtCycles = [1, 17] in {
def : WriteRes<WriteFDiv32, [p8700FpuLong, p8700FpuApu]>;
def : WriteRes<WriteFSqrt32, [p8700FpuLong, p8700FpuApu]>;

def : WriteRes<WriteFDiv64, [p8700IssueFPUL, p8700FpuApu]>;
def : WriteRes<WriteFSqrt64, [p8700IssueFPUL, p8700FpuApu]>;
}

def : WriteRes<WriteCSR, [p8700ALQ]>;

// Bypass and advance.
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;

// Unsupported extensions.
defm : UnsupportedSchedV;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedZvkned;
}
