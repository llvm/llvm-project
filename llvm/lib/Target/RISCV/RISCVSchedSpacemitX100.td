//- RISCVSchedSpacemitX100.td - Spacemit X100 Scheduling Defs -*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Scheduler model for the SpacemiT-X100 processor.
//
//===----------------------------------------------------------------------===//

def SpacemitX100Model : SchedMachineModel {
  let IssueWidth = 4;          // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 192; // Max micro-ops that can be buffered.
                               // 64 entry ROB. Max 3 micro-ops share one entry.
  let LoadLatency = 3;         // Cycles for loads to access the cache.
  let MispredictPenalty = 9;   // Extra cycles for a mispredicted branch.

  let CompleteModel = 0;

  let UnsupportedFeatures = [HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVInstructions];
}

let SchedModel = SpacemitX100Model in {

//===----------------------------------------------------------------------===//
// Define processor resources for Spacemit-X100
let BufferSize = 6 in {
  // IQ0, IQ1, BQ: 12 entry queue, can accept 2 ops and issue 1 op per cycle
  // To model the accept bandwidth, split into 2 sub-queues of 6 entry each
  def SMTX100_IQ0 : ProcResource<2>; //Integer Queue 0
  def SMTX100_IQ1 : ProcResource<2>; //Integer Queue 1
  def SMTX100_BQ  : ProcResource<2>; //Branch Queue
}

let BufferSize = 4 in {
  // LSQ: 16 entry queue, can accept 4 ops and issue 2 op per cycle
  // To model the accept bandwidth, split into 4 sub-queues of 4 entry each
  def SMTX100_LSQ : ProcResource<4>; //Load Store Queue
  // FQ0, FQ1: 8 entry queue, can accept 2 ops and issue 1 op per cycle
  // To model the accept bandwidth, split into 2 sub-queues of 4 entry each
  def SMTX100_FQ0 : ProcResource<2>; //Float Queue 0
  def SMTX100_FQ1 : ProcResource<2>; //Float Queue 1
}

def SMTX100_IQ : ProcResGroup<[SMTX100_IQ0, SMTX100_IQ1]>;
def SMTX100_FQ : ProcResGroup<[SMTX100_FQ0, SMTX100_FQ1]>;

//===----------------------------------------------------------------------===//

// Branching
let Latency = 2 in {
  def : WriteRes<WriteJmp,  [SMTX100_BQ]>;
  def : WriteRes<WriteJal,  [SMTX100_BQ]>;
  def : WriteRes<WriteJalr, [SMTX100_BQ]>;
}

// Integer arithmetic and logic
def : WriteRes<WriteIALU32,     [SMTX100_IQ]>;
def : WriteRes<WriteIALU,       [SMTX100_IQ]>;
def : WriteRes<WriteShiftImm32, [SMTX100_IQ]>;
def : WriteRes<WriteShiftImm,   [SMTX100_IQ]>;
def : WriteRes<WriteShiftReg32, [SMTX100_IQ]>;
def : WriteRes<WriteShiftReg,   [SMTX100_IQ]>;

// Integer multiplication
def : WriteRes<WriteIMul32, [SMTX100_IQ1]> { let Latency = 2; }
def : WriteRes<WriteIMul,   [SMTX100_IQ1]> { let Latency = 3; }

// Integer division/remainder
// Latency is 4-14, Worst case latency is used
let Latency = 14, ReleaseAtCycles = [14] in {
  def : WriteRes<WriteIDiv32, [SMTX100_IQ0]>;
  def : WriteRes<WriteIRem32, [SMTX100_IQ0]>;
}
// Latency is 4-22, Worst case latency is used
let Latency = 22, ReleaseAtCycles = [22] in {
  def : WriteRes<WriteIDiv, [SMTX100_IQ0]>;
  def : WriteRes<WriteIRem, [SMTX100_IQ0]>;
}

// Bitmanip
def : WriteRes<WriteRotateImm,   [SMTX100_IQ]>;
def : WriteRes<WriteRotateImm32, [SMTX100_IQ]>;
def : WriteRes<WriteRotateReg,   [SMTX100_IQ]>;
def : WriteRes<WriteRotateReg32, [SMTX100_IQ]>;

def : WriteRes<WriteCLZ,   [SMTX100_IQ]>;
def : WriteRes<WriteCLZ32, [SMTX100_IQ]>;
def : WriteRes<WriteCTZ,   [SMTX100_IQ]>;
def : WriteRes<WriteCTZ32, [SMTX100_IQ]>;

let Latency = 2 in {
  def : WriteRes<WriteCPOP,   [SMTX100_IQ]>;
  def : WriteRes<WriteCPOP32, [SMTX100_IQ]>;
}

def : WriteRes<WriteORCB,    [SMTX100_IQ]>;
def : WriteRes<WriteIMinMax, [SMTX100_IQ]>;
def : WriteRes<WriteREV8,    [SMTX100_IQ]>;

def : WriteRes<WriteSHXADD,   [SMTX100_IQ]>;
def : WriteRes<WriteSHXADD32, [SMTX100_IQ]>;

def : WriteRes<WriteCLMUL,  [SMTX100_IQ]> { let Latency = 2; }

// Single-bit instructions
def : WriteRes<WriteSingleBit,    [SMTX100_IQ]>;
def : WriteRes<WriteSingleBitImm, [SMTX100_IQ]>;
def : WriteRes<WriteBEXT,  [SMTX100_IQ]>;
def : WriteRes<WriteBEXTI, [SMTX100_IQ]>;

// Memory/Atomic memory
def : WriteRes<WriteSTB, [SMTX100_LSQ]>;
def : WriteRes<WriteSTH, [SMTX100_LSQ]>;
def : WriteRes<WriteSTW, [SMTX100_LSQ]>;
def : WriteRes<WriteSTD, [SMTX100_LSQ]>;
def : WriteRes<WriteFST16, [SMTX100_LSQ]>;
def : WriteRes<WriteFST32, [SMTX100_LSQ]>;
def : WriteRes<WriteFST64, [SMTX100_LSQ]>;

let Latency = 3 in {
  def : WriteRes<WriteLDB, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDH, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDW, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDD, [SMTX100_LSQ]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFLD16, [SMTX100_LSQ]>;
  def : WriteRes<WriteFLD32, [SMTX100_LSQ]>;
  def : WriteRes<WriteFLD64, [SMTX100_LSQ]>;
}

// Atomics
// Latency is at least 7, not sure worst case latency now
let Latency = 7 in {
  def : WriteRes<WriteAtomicSTW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicSTD, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicLDW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicLDD, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicD, [SMTX100_LSQ]>;
}

// Floating point units Half precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMul16, [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ16,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax16, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMA16, [SMTX100_FQ]> { let Latency = 5; }


// Latency is 4-12, Worst case latency is used
let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv16,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt16, [SMTX100_FQ0]>;
}

// Single precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd32,    [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ32,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax32, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMul32, [SMTX100_FQ]> { let Latency = 4; }
def : WriteRes<WriteFMA32,  [SMTX100_FQ]> { let Latency = 5; }

// Latency is 4-12, Worst case latency is used
let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv32,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt32, [SMTX100_FQ0]>;
}

// Double precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd64,    [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ64,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax64, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMul64, [SMTX100_FQ]> { let Latency = 4; }
def : WriteRes<WriteFMA64,  [SMTX100_FQ]> { let Latency = 4; }

// Latency is 4-20, Worst case latency is used
let Latency = 20, ReleaseAtCycles = [20] in {
  def :  WriteRes<WriteFDiv64,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt64, [SMTX100_FQ0]>;
}

// Zfa
let Latency = 3 in {
  def : WriteRes<WriteFRoundF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFRoundF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFRoundF64, [SMTX100_FQ1]>;

  def : WriteRes<WriteFLI16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFLI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFLI64, [SMTX100_FQ1]>;
}

// Conversions
let Latency = 3 in {
  def : WriteRes<WriteFCvtF16ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToF32, [SMTX100_FQ1]>;

  def : WriteRes<WriteFClass16, [SMTX100_FQ]>;
  def : WriteRes<WriteFClass32, [SMTX100_FQ]>;
  def : WriteRes<WriteFClass64, [SMTX100_FQ]>;

  def : WriteRes<WriteFCmp16, [SMTX100_FQ]>;
  def : WriteRes<WriteFCmp32, [SMTX100_FQ]>;
  def : WriteRes<WriteFCmp64, [SMTX100_FQ]>;

  def : WriteRes<WriteFMovF16ToI16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI16ToF16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovF32ToI32, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI32ToF32, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovF64ToI64, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI64ToF64, [SMTX100_FQ]>;
}

// Others
def : WriteRes<WriteCSR,   [SMTX100_IQ0]>;
def : WriteRes<WriteNop,   [SMTX100_IQ]>;

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
// Zfa
def : ReadAdvance<ReadFRoundF32, 0>;
def : ReadAdvance<ReadFRoundF64, 0>;
def : ReadAdvance<ReadFRoundF16, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsf;
}

