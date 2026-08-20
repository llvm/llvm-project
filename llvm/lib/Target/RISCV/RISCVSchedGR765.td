//===-- RISCVSchedGR765.td - GR765 Scheduling Definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduling model for the GR765 RISCV-V configuration.
//
// Timing is taken from the GR765 Data Sheet User's Manual.
//
//===----------------------------------------------------------------------===//

def GR765Model : SchedMachineModel {
  // In-order dual-issue pipeline
  let MicroOpBufferSize = 0;
  let IssueWidth = 2;
  let LoadLatency = 2;
  let MispredictPenalty = 3;
  let CompleteModel = false;
  let UnsupportedFeatures = [HasStdExtZcmt, HasStdExtZksed, HasStdExtZksh,
                             HasVInstructions, HasVInstructionsI64,
                             HasStdExtZkr];
}

//===----------------------------------------------------------------------===//
// Processor resources
//===----------------------------------------------------------------------===//

let BufferSize = 0 in {
def GR765UnitALU        : ProcResource<2>; // Int ALUs
def GR765UnitIMul       : ProcResource<1>; // Int Multiply
def GR765UnitMem        : ProcResource<1>; // Load/Store
def GR765UnitB          : ProcResource<1>; // Branch
def GR765UnitIssue0     : ProcResource<1>; // Lane 0 issue slot
def GR765UnitIssue1     : ProcResource<1>; // Lane 1 issue slot
def GR765UnitCLMUL      : ProcResource<1>; // Carry-less multiply
def GR765UnitCrypto     : ProcResource<1>; // AES / SHA unit

def GR765UnitFPALU      : ProcResource<1>; // FP FPR-result path
def GR765UnitFPTransfer : ProcResource<1>; // FP binary data transfer
def GR765UnitFPToInt    : ProcResource<1>; // FP result to integer pipeline
}

let BufferSize = 1 in {
def GR765UnitIDiv       : ProcResource<1>; // Int Division
def GR765UnitFPDivSqrt  : ProcResource<1>; // FP Divide/Sqrt
}

//===----------------------------------------------------------------------===//

let SchedModel = GR765Model in {

// Instructions may use either issue lane unless their class restricts it.
def GR765UnitIssue : ProcResGroup<[GR765UnitIssue0, GR765UnitIssue1]>;

// Map scheduling writes with identical resource requirements.
multiclass GR765WriteRes<list<SchedWrite> Writes,
                         list<ProcResourceKind> Resources> {
  foreach W = Writes in
    def : WriteRes<W, Resources>;
}

// Conditional branches issue in lane 1. JAL/JALR are single-issue.
let Latency = 2 in {
def : WriteRes<WriteJmp, [GR765UnitB, GR765UnitIssue1]>;
def : WriteRes<WriteJal, [GR765UnitB]> { let SingleIssue = true; }
def : WriteRes<WriteJalr, [GR765UnitB]> { let SingleIssue = true; }
}

// Integer ALU results take two cycles except through the bypasses below.
let Latency = 2 in {
defm : GR765WriteRes<[WriteIALU32, WriteIALU, WriteShiftImm32,
                      WriteShiftImm, WriteShiftReg32, WriteShiftReg,
                      WriteSHXADD, WriteSHXADD32, WriteRotateImm,
                      WriteRotateImm32, WriteRotateReg, WriteRotateReg32,
                      WriteCLZ, WriteCLZ32, WriteCTZ, WriteCTZ32, WriteCPOP,
                      WriteCPOP32, WriteREV8, WriteORCB, WriteIMinMax,
                      WriteSingleBit, WriteSingleBitImm, WriteBEXT,
                      WriteBEXTI, WriteBREV8, WritePACK, WritePACK32, WriteZIP,
                      WriteXPERM], [GR765UnitALU, GR765UnitIssue]>;
}

// Integer multiply uses one shared unit and has two-cycle result latency.
let Latency = 2 in {
defm : GR765WriteRes<[WriteIMul, WriteIMul32],
                     [GR765UnitIMul, GR765UnitIssue]>;
}

// CLMUL is a two-cycle operation restricted to lane 0.
let Latency = 2, SingleIssue = true in {
def : WriteRes<WriteCLMUL, [GR765UnitCLMUL, GR765UnitIssue0]>;
}

// AES/SHA has a two-cycle result latency.
def GR765WriteCrypto : SchedWriteRes<[GR765UnitCrypto, GR765UnitIssue]> {
  let Latency = 2;
}
def : InstRW<[GR765WriteCrypto], (instregex "^AES(32|64)", "^SHA(256|512)")>;

// Integer divide/remainder take 35 cycles. 32-bit forms take 19 cycles.
let Latency = 35, ReleaseAtCycles = [35], SingleIssue = true in {
defm : GR765WriteRes<[WriteIDiv, WriteIRem], [GR765UnitIDiv]>;
}
let Latency = 19, ReleaseAtCycles = [19], SingleIssue = true in {
defm : GR765WriteRes<[WriteIDiv32, WriteIRem32], [GR765UnitIDiv]>;
}

// Integer stores issue through lane 0 with one-cycle latency.
let Latency = 1 in {
defm : GR765WriteRes<[WriteSTB, WriteSTH, WriteSTW, WriteSTD],
                     [GR765UnitMem, GR765UnitIssue0]>;
}

// FP stores transfer register data at two-cycle throughput.
let Latency = 1, ReleaseAtCycles = [1, 2, 1], SingleIssue = true in {
defm : GR765WriteRes<[WriteFST16, WriteFST32, WriteFST64],
                     [GR765UnitMem, GR765UnitFPTransfer,
                      GR765UnitIssue0]>;
}

// Integer loads have two-cycle result latency.
let Latency = 2 in {
defm : GR765WriteRes<[WriteLDB, WriteLDH, WriteLDW, WriteLDD],
                     [GR765UnitMem, GR765UnitIssue0]>;
}

// The seven-cycle latency for a dependent fld-to-fsd pair.
let Latency = 7, SingleIssue = true in {
defm : GR765WriteRes<[WriteFLD16, WriteFLD32, WriteFLD64],
                     [GR765UnitMem, GR765UnitIssue0]>;
}

// Atomic loads and stores have two-cycle latency.
let Latency = 2 in {
defm : GR765WriteRes<[WriteAtomicW, WriteAtomicD, WriteAtomicLDW,
                      WriteAtomicLDD, WriteAtomicSTW, WriteAtomicSTD],
                     [GR765UnitMem, GR765UnitIssue0]>;
}

// Dependent fadd chains take 4 cycles; independent throughput is 1/cycle.
let Latency = 4, SingleIssue = true in {
defm : GR765WriteRes<[WriteFAdd16, WriteFAdd32, WriteFAdd64,
                      WriteFMinMax16, WriteFMinMax32, WriteFMinMax64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;

// FP-result conversions and moves share the four-cycle FP ALU path.
defm : GR765WriteRes<[WriteFCvtI32ToF16, WriteFCvtI32ToF32,
                      WriteFCvtI32ToF64, WriteFCvtI64ToF16,
                      WriteFCvtI64ToF32, WriteFCvtI64ToF64,
                      WriteFCvtF32ToF64, WriteFCvtF64ToF32,
                      WriteFMovI16ToF16, WriteFMovI32ToF32,
                      WriteFMovI64ToF64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;

// FP rounding and immediate loads share the four-cycle FP ALU path.
defm : GR765WriteRes<[WriteFRoundF16, WriteFRoundF32, WriteFRoundF64,
                      WriteFLI16, WriteFLI32, WriteFLI64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;

// Half/single/double FP conversions share the four-cycle FP ALU path.
defm : GR765WriteRes<[WriteFCvtF16ToF32, WriteFCvtF32ToF16,
                      WriteFCvtF16ToF64, WriteFCvtF64ToF16],
                     [GR765UnitFPALU, GR765UnitIssue0]>;
}

// FP-to-integer operations have 4-cycle latency.
let Latency = 4, ReleaseAtCycles = [1, 4, 1], SingleIssue = true in {
defm : GR765WriteRes<[WriteFCvtF16ToI32, WriteFCvtF16ToI64,
                      WriteFCvtF32ToI32, WriteFCvtF32ToI64,
                      WriteFCvtF64ToI32, WriteFCvtF64ToI64, WriteFClass16,
                      WriteFClass32, WriteFClass64, WriteFCmp16, WriteFCmp32,
                      WriteFCmp64],
                     [GR765UnitFPALU, GR765UnitFPToInt, GR765UnitIssue0]>;
}

// FP sign-injection chains have two-cycle latency.
let Latency = 2, SingleIssue = true in {
defm : GR765WriteRes<[WriteFSGNJ16, WriteFSGNJ32, WriteFSGNJ64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;
}

// FP-to-integer bit transfers use the two-cycle transfer path.
let Latency = 2, ReleaseAtCycles = [2, 1], SingleIssue = true in {
defm : GR765WriteRes<[WriteFMovF16ToI16, WriteFMovF32ToI32,
                     WriteFMovF64ToI64],
                     [GR765UnitFPTransfer, GR765UnitIssue0]>;
}

// Dependent FP multiply chains take three cycles.
let Latency = 3, SingleIssue = true in {
defm : GR765WriteRes<[WriteFMul16, WriteFMul32, WriteFMul64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;
}

// Dependent fused multiply-add chains take five cycles.
let Latency = 5, SingleIssue = true in {
defm : GR765WriteRes<[WriteFMA16, WriteFMA32, WriteFMA64],
                     [GR765UnitFPALU, GR765UnitIssue0]>;
}

let Latency = 17, ReleaseAtCycles = [18, 1], SingleIssue = true in {
defm : GR765WriteRes<[WriteFDiv16, WriteFDiv32, WriteFDiv64],
                     [GR765UnitFPDivSqrt, GR765UnitIssue0]>;
}
let Latency = 25, ReleaseAtCycles = [26, 1], SingleIssue = true in {
defm : GR765WriteRes<[WriteFSqrt16, WriteFSqrt32, WriteFSqrt64],
                     [GR765UnitFPDivSqrt, GR765UnitIssue0]>;
}

def : WriteRes<WriteCSR, [GR765UnitIssue0]>;
def : WriteRes<WriteNop, [GR765UnitIssue]>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
//===----------------------------------------------------------------------===//

// Late ALU/branch consumers can use results from multi-cycle operations.
defvar GR765LateBypassWrites = [WriteIALU, WriteIALU32,
                                WriteShiftImm, WriteShiftImm32,
                                WriteShiftReg, WriteShiftReg32,
                                WriteSHXADD, WriteSHXADD32,
                                WriteRotateImm, WriteRotateImm32,
                                WriteRotateReg, WriteRotateReg32,
                                WriteCLZ, WriteCLZ32, WriteCTZ, WriteCTZ32,
                                WriteCPOP, WriteCPOP32,
                                WriteREV8, WriteORCB, WriteIMinMax,
                                WriteSingleBit, WriteSingleBitImm,
                                WriteBEXT, WriteBEXTI,
                                WriteBREV8, WritePACK, WritePACK32, WriteZIP,
                                WriteXPERM,
                                WriteIMul, WriteIMul32,
                                WriteLDB, WriteLDH, WriteLDW, WriteLDD,
                                GR765WriteCrypto,
                                WriteFCvtF16ToI32, WriteFCvtF16ToI64,
                                WriteFCvtF32ToI32, WriteFCvtF32ToI64,
                                WriteFCvtF64ToI32, WriteFCvtF64ToI64,
                                WriteFClass16, WriteFClass32, WriteFClass64,
                                WriteFCmp16, WriteFCmp32, WriteFCmp64,
                                WriteFMovF16ToI16, WriteFMovF32ToI32,
                                WriteFMovF64ToI64];

defvar GR765LateBypassReads = [
  ReadJmp,
  ReadIALU, ReadIALU32, ReadShiftImm, ReadShiftImm32,
  ReadShiftReg, ReadShiftReg32,
  ReadSHXADD, ReadSHXADD32,
  ReadRotateImm, ReadRotateImm32, ReadRotateReg, ReadRotateReg32,
  ReadCLZ, ReadCLZ32, ReadCTZ, ReadCTZ32, ReadCPOP, ReadCPOP32,
  ReadREV8, ReadORCB, ReadIMinMax, ReadSingleBit, ReadSingleBitImm,
  ReadBREV8, ReadPACK, ReadPACK32, ReadZIP, ReadXPERM
];

foreach Read = GR765LateBypassReads in
  def : ReadAdvance<Read, 1, GR765LateBypassWrites>;

// Address generation bypasses ALU results, but not loads or multiplies.
defvar GR765AddrBypassWrites = [
  WriteIALU, WriteIALU32, WriteShiftImm, WriteShiftImm32,
  WriteShiftReg, WriteShiftReg32,
  WriteSHXADD, WriteSHXADD32,
  WriteRotateImm, WriteRotateImm32, WriteRotateReg, WriteRotateReg32,
  WriteCLZ, WriteCLZ32, WriteCTZ, WriteCTZ32, WriteCPOP, WriteCPOP32,
  WriteREV8, WriteORCB, WriteIMinMax, WriteSingleBit, WriteSingleBitImm,
  WriteBEXT, WriteBEXTI, WriteBREV8, WritePACK, WritePACK32, WriteZIP,
  WriteXPERM
];
def : ReadAdvance<ReadMemBase, 1, GR765AddrBypassWrites>;

// 4 cycles for fld -> fop
defvar GR765FPRConsumerReads = [
  ReadFAdd16, ReadFAdd32, ReadFAdd64,
  ReadFMul16, ReadFMul32, ReadFMul64,
  ReadFMA16, ReadFMA16Addend, ReadFMA32, ReadFMA32Addend,
  ReadFMA64, ReadFMA64Addend,
  ReadFDiv16, ReadFDiv32, ReadFDiv64,
  ReadFSqrt16, ReadFSqrt32, ReadFSqrt64,
  ReadFCmp16, ReadFCmp32, ReadFCmp64,
  ReadFSGNJ16, ReadFSGNJ32, ReadFSGNJ64,
  ReadFMinMax16, ReadFMinMax32, ReadFMinMax64,
  ReadFCvtF16ToI32, ReadFCvtF16ToI64,
  ReadFCvtF32ToI32, ReadFCvtF32ToI64,
  ReadFCvtF64ToI32, ReadFCvtF64ToI64,
  ReadFRoundF16, ReadFRoundF32, ReadFRoundF64,
  ReadFCvtF32ToF64, ReadFCvtF64ToF32,
  ReadFMovF32ToI32, ReadFMovF64ToI64,
  ReadFClass32, ReadFClass64,
  ReadFMovF16ToI16, ReadFClass16,
  ReadFCvtF16ToF32, ReadFCvtF32ToF16,
  ReadFCvtF16ToF64, ReadFCvtF64ToF16
];
foreach Read = GR765FPRConsumerReads in
  def : ReadAdvance<Read, 3, [WriteFLD16, WriteFLD32, WriteFLD64]>;

// These operand classes have no modeled bypass or load advance.
defvar GR765NoBypassReads = [
  ReadJalr, ReadCSR, ReadStoreData,
  ReadIDiv, ReadIDiv32, ReadIRem, ReadIRem32, ReadIMul, ReadIMul32,
  ReadAtomicWA, ReadAtomicWD, ReadAtomicDA, ReadAtomicDD,
  ReadAtomicLDW, ReadAtomicLDD, ReadAtomicSTW, ReadAtomicSTD,
  ReadFStoreData, ReadFMemBase,
  ReadFCvtI32ToF32, ReadFCvtI32ToF64, ReadFCvtI32ToF16,
  ReadFCvtI64ToF32, ReadFCvtI64ToF64, ReadFCvtI64ToF16,
  ReadFMovI32ToF32, ReadFMovI64ToF64,
  ReadCLMUL,
  ReadFMovI16ToF16
];

foreach Read = GR765NoBypassReads in
  def : ReadAdvance<Read, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedXsf;
}
