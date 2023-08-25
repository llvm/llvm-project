//==- RISCVSchedSyntacoreSCR1.td - Syntacore SCR1 Scheduling Definitions --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// SCR1: https://github.com/syntacore/scr1

// This model covers SYNTACORE_SCR1_CFG_RV32IMC_MAX configuration (syntacore-scr1-max).
// SYNTACORE_SCR1_CFG_RV32IC_BASE (syntacore-scr1-base) configuration has essentially
// same scheduling characteristics.

// SCR1 is single-issue in-order processor
def SyntacoreSCR1Model : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 1;
  let LoadLatency = 2;
  let MispredictPenalty = 3;
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVInstructions];
}

let SchedModel = SyntacoreSCR1Model in {

let BufferSize = 0 in {
def SCR1_ALU : ProcResource<1>;
def SCR1_LSU : ProcResource<1>;
def SCR1_MUL : ProcResource<1>;
def SCR1_DIV : ProcResource<1>;
def SCR1_CFU : ProcResource<1>;
}

// Branching
def : WriteRes<WriteJmp, [SCR1_CFU]>;
def : WriteRes<WriteJal, [SCR1_CFU]>;
def : WriteRes<WriteJalr, [SCR1_CFU]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [SCR1_ALU]>;
def : WriteRes<WriteIALU, [SCR1_ALU]>;
def : WriteRes<WriteShiftImm32, [SCR1_ALU]>;
def : WriteRes<WriteShiftImm, [SCR1_ALU]>;
def : WriteRes<WriteShiftReg32, [SCR1_ALU]>;
def : WriteRes<WriteShiftReg, [SCR1_ALU]>;

// Integer multiplication: single-cycle multiplier in SCR1_CFG_RV32IMC_MAX
def : WriteRes<WriteIMul, [SCR1_MUL]>;
def : WriteRes<WriteIMul32, [SCR1_MUL]>;

// Integer division: latency 33, inverse throughput 33
let Latency = 33, ResourceCycles = [33] in {
def : WriteRes<WriteIDiv32, [SCR1_DIV]>;
def : WriteRes<WriteIDiv, [SCR1_DIV]>;
}

// Load/store instructions on SCR1 have latency 2 and inverse throughput 2
// (SCR1_CFG_RV32IMC_MAX includes TCM)
let Latency = 2, ResourceCycles=[2] in {
// Memory
def : WriteRes<WriteSTB, [SCR1_LSU]>;
def : WriteRes<WriteSTH, [SCR1_LSU]>;
def : WriteRes<WriteSTW, [SCR1_LSU]>;
def : WriteRes<WriteSTD, [SCR1_LSU]>;
def : WriteRes<WriteLDB, [SCR1_LSU]>;
def : WriteRes<WriteLDH, [SCR1_LSU]>;
def : WriteRes<WriteLDW, [SCR1_LSU]>;
def : WriteRes<WriteLDD, [SCR1_LSU]>;
}

let Unsupported = true in {
// Atomic memory
def : WriteRes<WriteAtomicW, [SCR1_LSU]>;
def : WriteRes<WriteAtomicD, [SCR1_LSU]>;
def : WriteRes<WriteAtomicLDW, [SCR1_LSU]>;
def : WriteRes<WriteAtomicLDD, [SCR1_LSU]>;
def : WriteRes<WriteAtomicSTW, [SCR1_LSU]>;
def : WriteRes<WriteAtomicSTD, [SCR1_LSU]>;

// FP load/store
def : WriteRes<WriteFST32, [SCR1_LSU]>;
def : WriteRes<WriteFST64, [SCR1_LSU]>;
def : WriteRes<WriteFLD32, [SCR1_LSU]>;
def : WriteRes<WriteFLD64, [SCR1_LSU]>;

// FP instructions
def : WriteRes<WriteFAdd32, []>;
def : WriteRes<WriteFSGNJ32, []>;
def : WriteRes<WriteFMinMax32, []>;
def : WriteRes<WriteFAdd64, []>;
def : WriteRes<WriteFSGNJ64, []>;
def : WriteRes<WriteFMinMax64, []>;
def : WriteRes<WriteFCvtI32ToF32, []>;
def : WriteRes<WriteFCvtI32ToF64, []>;
def : WriteRes<WriteFCvtI64ToF32, []>;
def : WriteRes<WriteFCvtI64ToF64, []>;
def : WriteRes<WriteFCvtF32ToI32, []>;
def : WriteRes<WriteFCvtF32ToI64, []>;
def : WriteRes<WriteFCvtF64ToI32, []>;
def : WriteRes<WriteFCvtF64ToI64, []>;
def : WriteRes<WriteFCvtF32ToF64, []>;
def : WriteRes<WriteFCvtF64ToF32, []>;
def : WriteRes<WriteFClass32, []>;
def : WriteRes<WriteFClass64, []>;
def : WriteRes<WriteFCmp32, []>;
def : WriteRes<WriteFCmp64, []>;
def : WriteRes<WriteFMovF32ToI32, []>;
def : WriteRes<WriteFMovI32ToF32, []>;
def : WriteRes<WriteFMovF64ToI64, []>;
def : WriteRes<WriteFMovI64ToF64, []>;
def : WriteRes<WriteFMul32, []>;
def : WriteRes<WriteFMA32, []>;
def : WriteRes<WriteFMul64, []>;
def : WriteRes<WriteFMA64, []>;
def : WriteRes<WriteFDiv32, []>;
def : WriteRes<WriteFDiv64, []>;
def : WriteRes<WriteFSqrt32, []>;
def : WriteRes<WriteFSqrt64, []>;

def : WriteRes<WriteSFB, []>;
}

// Others
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypasses (none)
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
def : ReadAdvance<ReadSFBJmp, 0>;
def : ReadAdvance<ReadSFBALU, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfh;
}
