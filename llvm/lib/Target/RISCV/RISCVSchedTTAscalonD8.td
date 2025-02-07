//=- RISCVSchedTTAscalonD8.td - TT Ascalon D8 Sched Defs -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

def TTAscalonD8Model : SchedMachineModel {
  let IssueWidth        =   8; // 8-way decode and dispatch
  let MicroOpBufferSize = 256; // 256 micro-op re-order buffer
  let LoadLatency       =   4; // Optimistic load latency
  let MispredictPenalty =  14; // Fetch + Decode/Rename/Dispatch + Branch

  let CompleteModel = 0;

  // TODO: supported, but haven't added scheduling info yet.
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructions, HasVInstructionsI64];
}

let SchedModel = TTAscalonD8Model in {

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

let BufferSize = 16 in {
  def AscalonLS : ProcResource<3>;
  def AscalonFXA : ProcResource<1>; // ALU, FP/VEC -> INT, MUL, DIV, CSR
  def AscalonFXB : ProcResource<1>; // ALU, INT -> FP/VEC
  def AscalonFXC : ProcResource<2>; // ALU, BR
  def AscalonFXD : ProcResource<2>; // ALU
  def AscalonFP : ProcResource<2>;
  // TODO: two vector units with vector scheduling model.
}

def AscalonFX : ProcResGroup<[AscalonFXA, AscalonFXB, AscalonFXC, AscalonFXD]>;

//===----------------------------------------------------------------------===//

// Branching
def : WriteRes<WriteJmp, [AscalonFXC]>;
def : WriteRes<WriteJal, [AscalonFXC]>;
def : WriteRes<WriteJalr, [AscalonFXC]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [AscalonFX]>;
def : WriteRes<WriteIALU, [AscalonFX]>;
def : WriteRes<WriteShiftImm32, [AscalonFX]>;
def : WriteRes<WriteShiftImm, [AscalonFX]>;
def : WriteRes<WriteShiftReg32, [AscalonFX]>;
def : WriteRes<WriteShiftReg, [AscalonFX]>;

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [AscalonFXA]>;
def : WriteRes<WriteIMul32, [AscalonFXA]>;
}

// Integer division
// Worst case latency is used.

let Latency = 7, ReleaseAtCycles = [7] in {
  def : WriteRes<WriteIDiv32, [AscalonFXA]>;
  def : WriteRes<WriteIDiv, [AscalonFXA]>;
  def : WriteRes<WriteIRem32, [AscalonFXA]>;
  def : WriteRes<WriteIRem, [AscalonFXA]>;
}

// Bitmanip
def : WriteRes<WriteRotateImm, [AscalonFX]>;
def : WriteRes<WriteRotateImm32, [AscalonFX]>;
def : WriteRes<WriteRotateReg, [AscalonFX]>;
def : WriteRes<WriteRotateReg32, [AscalonFX]>;

def : WriteRes<WriteCLZ, [AscalonFX]>;
def : WriteRes<WriteCLZ32, [AscalonFX]>;
def : WriteRes<WriteCTZ, [AscalonFX]>;
def : WriteRes<WriteCTZ32, [AscalonFX]>;

def : WriteRes<WriteCPOP, [AscalonFX]>;
def : WriteRes<WriteCPOP32, [AscalonFX]>;

def : WriteRes<WriteORCB, [AscalonFX]>;

def : WriteRes<WriteIMinMax, [AscalonFX]>;

def : WriteRes<WriteREV8, [AscalonFX]>;

def : WriteRes<WriteSHXADD, [AscalonFX]>;
def : WriteRes<WriteSHXADD32, [AscalonFX]>;

// Single-bit instructions
def : WriteRes<WriteSingleBit, [AscalonFX]>;
def : WriteRes<WriteSingleBitImm, [AscalonFX]>;
def : WriteRes<WriteBEXT, [AscalonFX]>;
def : WriteRes<WriteBEXTI, [AscalonFX]>;

// Memory
def : WriteRes<WriteSTB, [AscalonLS]>;
def : WriteRes<WriteSTH, [AscalonLS]>;
def : WriteRes<WriteSTW, [AscalonLS]>;
def : WriteRes<WriteSTD, [AscalonLS]>;
def : WriteRes<WriteFST16, [AscalonLS]>;
def : WriteRes<WriteFST32, [AscalonLS]>;
def : WriteRes<WriteFST64, [AscalonLS]>;

let Latency = 4 in {
def : WriteRes<WriteLDB, [AscalonLS]>;
def : WriteRes<WriteLDH, [AscalonLS]>;
def : WriteRes<WriteLDW, [AscalonLS]>;
def : WriteRes<WriteLDD, [AscalonLS]>;
def : WriteRes<WriteFLD16, [AscalonLS]>;
def : WriteRes<WriteFLD32, [AscalonLS]>;
def : WriteRes<WriteFLD64, [AscalonLS]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [AscalonLS]>;
def : WriteRes<WriteAtomicSTD, [AscalonLS]>;

let Latency = 4 in {
def : WriteRes<WriteAtomicW, [AscalonLS]>;
def : WriteRes<WriteAtomicD, [AscalonLS]>;
def : WriteRes<WriteAtomicLDW, [AscalonLS]>;
def : WriteRes<WriteAtomicLDD, [AscalonLS]>;
}

// Half precision.
let Latency = 3 in {
def : WriteRes<WriteFAdd16, [AscalonFP]>;
def : WriteRes<WriteFMul16, [AscalonFP]>;
def : WriteRes<WriteFMA16, [AscalonFP]>;
def : WriteRes<WriteFSGNJ16, [AscalonFP]>;
def : WriteRes<WriteFMinMax16, [AscalonFP]>;
}

let Latency = 7, ReleaseAtCycles = [7] in {
def :  WriteRes<WriteFDiv16, [AscalonFP]>;
def :  WriteRes<WriteFSqrt16, [AscalonFP]>;
}

// Single precision.
let Latency = 3 in {
def : WriteRes<WriteFAdd32, [AscalonFP]>;
def : WriteRes<WriteFMul32, [AscalonFP]>;
def : WriteRes<WriteFMA32, [AscalonFP]>;
def : WriteRes<WriteFSGNJ32, [AscalonFP]>;
def : WriteRes<WriteFMinMax32, [AscalonFP]>;
}

let Latency = 7, ReleaseAtCycles = [7] in {
def :  WriteRes<WriteFDiv32, [AscalonFP]>;
def :  WriteRes<WriteFSqrt32, [AscalonFP]>;
}

// Double precision
let Latency = 3 in {
def : WriteRes<WriteFAdd64, [AscalonFP]>;
def : WriteRes<WriteFMul64, [AscalonFP]>;
def : WriteRes<WriteFMA64, [AscalonFP]>;
def : WriteRes<WriteFSGNJ64, [AscalonFP]>;
def : WriteRes<WriteFMinMax64, [AscalonFP]>;
}

let Latency = 12, ReleaseAtCycles = [12] in {
def :  WriteRes<WriteFDiv64, [AscalonFP]>;
def :  WriteRes<WriteFSqrt64, [AscalonFP]>;
}

// Conversions
def : WriteRes<WriteFCvtI32ToF16, [AscalonFXB]>;
def : WriteRes<WriteFCvtI32ToF32, [AscalonFXB]>;
def : WriteRes<WriteFCvtI32ToF64, [AscalonFXB]>;
def : WriteRes<WriteFCvtI64ToF16, [AscalonFXB]>;
def : WriteRes<WriteFCvtI64ToF32, [AscalonFXB]>;
def : WriteRes<WriteFCvtI64ToF64, [AscalonFXB]>;
def : WriteRes<WriteFCvtF16ToI32, [AscalonFXA]>;
def : WriteRes<WriteFCvtF16ToI64, [AscalonFXA]>;
def : WriteRes<WriteFCvtF16ToF32, [AscalonFP]>;
def : WriteRes<WriteFCvtF16ToF64, [AscalonFP]>;
def : WriteRes<WriteFCvtF32ToI32, [AscalonFXA]>;
def : WriteRes<WriteFCvtF32ToI64, [AscalonFXA]>;
def : WriteRes<WriteFCvtF32ToF16, [AscalonFP]>;
def : WriteRes<WriteFCvtF32ToF64, [AscalonFP]>;
def : WriteRes<WriteFCvtF64ToI32, [AscalonFXA]>;
def : WriteRes<WriteFCvtF64ToI64, [AscalonFXA]>;
def : WriteRes<WriteFCvtF64ToF16, [AscalonFP]>;
def : WriteRes<WriteFCvtF64ToF32, [AscalonFP]>;

def : WriteRes<WriteFClass16, [AscalonFP]>;
def : WriteRes<WriteFClass32, [AscalonFP]>;
def : WriteRes<WriteFClass64, [AscalonFP]>;
def : WriteRes<WriteFCmp16, [AscalonFP]>;
def : WriteRes<WriteFCmp32, [AscalonFP]>;
def : WriteRes<WriteFCmp64, [AscalonFP]>;

def : WriteRes<WriteFMovI16ToF16, [AscalonFXB]>;
def : WriteRes<WriteFMovF16ToI16, [AscalonFXA]>;
def : WriteRes<WriteFMovI32ToF32, [AscalonFXB]>;
def : WriteRes<WriteFMovF32ToI32, [AscalonFXA]>;
def : WriteRes<WriteFMovI64ToF64, [AscalonFXB]>;
def : WriteRes<WriteFMovF64ToI64, [AscalonFXA]>;

// Others
def : WriteRes<WriteCSR, [AscalonFXA]>;
def : WriteRes<WriteNop, [AscalonFX]>;

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
// Single-bit instructions
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
}
