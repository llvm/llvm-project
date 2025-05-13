//=- RISCVSchedSpacemitX60.td - Spacemit X60 Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Scheduler model for the SpacemiT-X60 processor based on documentation of the
// C908 and experiments on real hardware (bpi-f3).
//
//===----------------------------------------------------------------------===//

def SpacemitX60Model : SchedMachineModel {
  let IssueWidth        = 2; // dual-issue
  let MicroOpBufferSize = 0; // in-order
  let LoadLatency       = 5; // worse case: >= 3
  let MispredictPenalty = 9; // nine-stage

  let CompleteModel = 0;

  let UnsupportedFeatures = [HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr];
}

let SchedModel = SpacemitX60Model in {

//===----------------------------------------------------------------------===//
// Define processor resources for Spacemit-X60

// Information gathered from the C908 user manual:
let BufferSize = 0 in {
  // The LSU supports dual issue for scalar store/load instructions
  def SMX60_LS : ProcResource<2>;

  // An IEU can decode and issue two instructions at the same time
  def SMX60_IEUA : ProcResource<1>;
  def SMX60_IEUB : ProcResource<1>;
  def SMX60_IEU : ProcResGroup<[SMX60_IEUA, SMX60_IEUB]>;

  // Although the X60 does appear to support multiple issue for at least some
  // floating point instructions, this model assumes single issue as
  // increasing it reduces the gains we saw in performance
  def SMX60_FP : ProcResource<1>;
}

//===----------------------------------------------------------------------===//

// Branching
def : WriteRes<WriteJmp, [SMX60_IEUA]>;
def : WriteRes<WriteJal, [SMX60_IEUA]>;
def : WriteRes<WriteJalr, [SMX60_IEUA]>;

// Integer arithmetic and logic
// Latency of ALU instructions is 1, but add.uw is 2
def : WriteRes<WriteIALU32, [SMX60_IEU]>;
def : WriteRes<WriteIALU, [SMX60_IEU]>;
def : WriteRes<WriteShiftImm32, [SMX60_IEU]>;
def : WriteRes<WriteShiftImm, [SMX60_IEU]>;
def : WriteRes<WriteShiftReg32, [SMX60_IEU]>;
def : WriteRes<WriteShiftReg, [SMX60_IEU]>;

// Integer multiplication
def : WriteRes<WriteIMul32, [SMX60_IEU]>  { let Latency = 3; }

// The latency of mul is 5, while in mulh, mulhsu, mulhu is 6
// Worst case latency is used
def : WriteRes<WriteIMul, [SMX60_IEU]>  { let Latency = 6; }

// Integer division/remainder
// TODO: Latency set based on C908 datasheet and hasn't been
// confirmed experimentally.
let Latency = 12, ReleaseAtCycles = [12] in {
  def : WriteRes<WriteIDiv32, [SMX60_IEUA]>;
  def : WriteRes<WriteIRem32, [SMX60_IEUA]>;
}
let Latency = 20, ReleaseAtCycles = [20] in {
  def : WriteRes<WriteIDiv, [SMX60_IEUA]>;
  def : WriteRes<WriteIRem, [SMX60_IEUA]>;
}

// Bitmanip
def : WriteRes<WriteRotateImm, [SMX60_IEU]>;
def : WriteRes<WriteRotateImm32, [SMX60_IEU]>;
def : WriteRes<WriteRotateReg, [SMX60_IEU]>;
def : WriteRes<WriteRotateReg32, [SMX60_IEU]>;

def : WriteRes<WriteCLZ, [SMX60_IEU]>;
def : WriteRes<WriteCLZ32, [SMX60_IEU]>;
def : WriteRes<WriteCTZ, [SMX60_IEU]>;
def : WriteRes<WriteCTZ32, [SMX60_IEU]>;

let Latency = 2 in {
  def : WriteRes<WriteCPOP, [SMX60_IEU]>;
  def : WriteRes<WriteCPOP32, [SMX60_IEU]>;
}

def : WriteRes<WriteORCB, [SMX60_IEU]>;
def : WriteRes<WriteIMinMax, [SMX60_IEU]>;
def : WriteRes<WriteREV8, [SMX60_IEU]>;

let Latency = 2 in {
  def : WriteRes<WriteSHXADD, [SMX60_IEU]>;
  def : WriteRes<WriteSHXADD32, [SMX60_IEU]>;
  def : WriteRes<WriteCLMUL, [SMX60_IEU]>;
}

// Single-bit instructions
def : WriteRes<WriteSingleBit, [SMX60_IEU]>;
def : WriteRes<WriteSingleBitImm, [SMX60_IEU]>;
def : WriteRes<WriteBEXT, [SMX60_IEU]>;
def : WriteRes<WriteBEXTI, [SMX60_IEU]>;

// Memory/Atomic memory
let Latency = 3 in {
  def : WriteRes<WriteSTB, [SMX60_LS]>;
  def : WriteRes<WriteSTH, [SMX60_LS]>;
  def : WriteRes<WriteSTW, [SMX60_LS]>;
  def : WriteRes<WriteSTD, [SMX60_LS]>;
  def : WriteRes<WriteFST16, [SMX60_LS]>;
  def : WriteRes<WriteFST32, [SMX60_LS]>;
  def : WriteRes<WriteFST64, [SMX60_LS]>;
  def : WriteRes<WriteAtomicSTW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicSTD, [SMX60_LS]>;
}

let Latency = 5 in {
  def : WriteRes<WriteLDB, [SMX60_LS]>;
  def : WriteRes<WriteLDH, [SMX60_LS]>;
  def : WriteRes<WriteLDW, [SMX60_LS]>;
  def : WriteRes<WriteLDD, [SMX60_LS]>;
  def : WriteRes<WriteFLD16, [SMX60_LS]>;
  def : WriteRes<WriteFLD32, [SMX60_LS]>;
  def : WriteRes<WriteFLD64, [SMX60_LS]>;
}

// Atomics
let Latency = 5 in {
  def : WriteRes<WriteAtomicLDW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicLDD, [SMX60_LS]>;
  def : WriteRes<WriteAtomicW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicD, [SMX60_LS]>;
}

// Floating point units Half precision
let Latency = 4 in {
  def : WriteRes<WriteFAdd16, [SMX60_FP]>;
  def : WriteRes<WriteFMul16, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ16, [SMX60_FP]>;
  def : WriteRes<WriteFMinMax16, [SMX60_FP]>;
}
def : WriteRes<WriteFMA16, [SMX60_FP]> { let Latency = 5; }

let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv16, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt16, [SMX60_FP]>;
}

// Single precision
let Latency = 4 in {
  def : WriteRes<WriteFAdd32, [SMX60_FP]>;
  def : WriteRes<WriteFMul32, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ32, [SMX60_FP]>;
  def : WriteRes<WriteFMinMax32, [SMX60_FP]>;
}
def : WriteRes<WriteFMA32, [SMX60_FP]> { let Latency = 5; }

let Latency = 15, ReleaseAtCycles = [15] in {
  def :  WriteRes<WriteFDiv32, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt32, [SMX60_FP]>;
}

// Double precision
let Latency = 5 in {
  def : WriteRes<WriteFAdd64, [SMX60_FP]>;
  def : WriteRes<WriteFMul64, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ64, [SMX60_FP]>;
}
def : WriteRes<WriteFMinMax64, [SMX60_FP]> { let Latency = 4; }
def : WriteRes<WriteFMA64, [SMX60_FP]> { let Latency = 6; }

let Latency = 22, ReleaseAtCycles = [22] in {
  def :  WriteRes<WriteFDiv64, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt64, [SMX60_FP]>;
}

// Conversions
let Latency = 6 in {
  def : WriteRes<WriteFCvtF16ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF32ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF32ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF64ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF64ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF16ToI64, [SMX60_IEU]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFCvtI32ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI32ToF32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI32ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF16ToF32, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF16ToF64, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF32ToF16, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF32ToF64, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF64ToF16, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF64ToF32, [SMX60_FP]>;
}

let Latency = 6 in {
  def : WriteRes<WriteFClass16, [SMX60_FP]>;
  def : WriteRes<WriteFClass32, [SMX60_FP]>;
  def : WriteRes<WriteFClass64, [SMX60_FP]>;

  def : WriteRes<WriteFCmp16, [SMX60_FP]>;
  def : WriteRes<WriteFCmp32, [SMX60_FP]>;
  def : WriteRes<WriteFCmp64, [SMX60_FP]>;

  def : WriteRes<WriteFMovF32ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFMovF16ToI16, [SMX60_IEU]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFMovI16ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFMovF64ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFMovI64ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFMovI32ToF32, [SMX60_IEU]>;
}

// Others
def : WriteRes<WriteCSR, [SMX60_IEU]>;
def : WriteRes<WriteNop, [SMX60_IEU]>;

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
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
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
def : ReadAdvance<ReadCLMUL, 0>;
// Single-bit instructions
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
}
