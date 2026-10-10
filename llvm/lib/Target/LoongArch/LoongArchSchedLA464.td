//==- LoongArchSchedLA464.td - LA464 Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for LA464 to support instruction
// scheduling and other instruction cost heuristics.
// Based on:
//  * Loongson 3A6000: A Star among Chinese CPUs, by Chips and Cheese
//    https://chipsandcheese.com/p/loongson-3a6000-a-star-among-chinese-cpus/
//  * Loongson-3A5000 instruction latency and throughput document, V0.1,
//    2024/07/24, Loongson Technology. Where the two sources disagree, the
//    measured latency and throughput of the document is used.
//===----------------------------------------------------------------------===//

def LA464Model : SchedMachineModel {
  let IssueWidth = 4;             // 4-way decode and dispatch
  let MicroOpBufferSize = 128;    // Reorder Buffer size
  let LoadLatency = 4;
  let MispredictPenalty = 14;     // Estimated value
  let CompleteModel = 0;
}

let SchedModel = LA464Model in {
// Integer Execution Unit
//
// Execution Pipes
// There are four integer execution pipes and two address generation pipes.
def LA464ALU0 : ProcResource<1>; // ALU, MUL, Branch, ALSL, BSTRINS, BYTEPICK
def LA464ALU1 : ProcResource<1>; // ALU, MUL, Branch, ALSL, BSTRINS, BYTEPICK
def LA464ALU2 : ProcResource<1>; // ALU
def LA464ALU3 : ProcResource<1>; // ALU

def LA464AGU0 : ProcResource<1>; // Load, Store
def LA464AGU1 : ProcResource<1>; // Load

// Execution Units
//
// Integer division runs on a non-pipelined unit behind ALU0. (Estimated)
def LA464Div : ProcResource<1>;

// Execution pipeline grouping
//
// Only ALU0 and ALU1 execute branches, integer multiplies, ALSL, BSTRINS and
// BYTEPICK according to their throughput (IPC is 2 instead of 4). (Estimated)
def LA464ALU01 : ProcResGroup<[LA464ALU0, LA464ALU1]>;
defvar LA464Branch = LA464ALU01;

// Both AGUs generate load addresses, so plain loads reach two per cycle.
// Stores and the indexed / bounded accesses are measured at one per cycle, so
// they are modelled on AGU0 alone. (Estimated)
def LA464AGU : ProcResGroup<[LA464AGU0, LA464AGU1]>;

// Scheduling
//
// LA464 has an unified 32 entry integer scheduler.
def LA464ALU : ProcResGroup<[LA464ALU0, LA464ALU1, LA464ALU2, LA464ALU3]> {
  let BufferSize = 32;
}

// Floating-Point / Vector Unit
//
// Execution pipes
// There are two floating point/vector execution pipes.
def LA464FPU0 : ProcResource<1>; // FALU, FMUL, FMA, FDIV, FMOV, VALU, VNarrow
def LA464FPU1 : ProcResource<1>; // FALU, FMUL, FMA, VALU

// Execution Units
//
// Floating-point division, square root and reciprocal run on a non-pipelined
// unit behind FPU0. It also executes the vector divide, square root and
// reciprocal operations. (Estimated)
def LA464FDiv : ProcResource<1>;

// Only FPU0 moves data between the general purpose or condition flag register
// file and the floating-point / vector register file, and only FPU0 executes
// FSEL, so those are limited to one per cycle. (Estimated)
defvar LA464FMov = LA464FPU0;

// Only FPU0 has the cross-element datapath used by the narrowing vector
// shifts, VSLLWIL and the resizing floating-point conversions; the document
// measures one per cycle for all of them. (Estimated)
defvar LA464VNarrow = LA464FPU0;

// Scheduling
//
// LA464 has an unified 32 entry vector/fp scheduler. Both pipes execute the
// scalar floating-point and the 128-bit / 256-bit vector operations.
def LA464FPU : ProcResGroup<[LA464FPU0, LA464FPU1]> {
  let BufferSize = 32;
}

// Load-Store Unit
//
// The LS unit contains two pipelines. Both can issue a load, so the document
// measures two loads per cycle, while stores are limited to one per cycle.
def LA464Load : ProcResource<2> {
  // The LS unit utilizes a 64-entry load queue (LDQ). (Estimated)
  let BufferSize = 64;
}

def LA464Store : ProcResource<1> {
  // The LS unit utilizes a 44-entry store queue (STQ). (Estimated)
  let BufferSize = 44;
}

//===----------------------------------------------------------------------===//

// Base Instruction
// Integer arithmetic and logic, default latency is 1
def : WriteRes<WriteIALU, [LA464ALU]>;
def : WriteRes<WriteIALU32, [LA464ALU]>;

def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;

// Integer instruction latency ALSL
def : WriteRes<WriteIALSL, [LA464ALU01]>;
def : WriteRes<WriteIALSL32, [LA464ALU01]>;

def : ReadAdvance<ReadIALSL, 0>;
def : ReadAdvance<ReadIALSL32, 0>;

// Integer instruction latency MUL
let Latency = 4 in {
  def : WriteRes<WriteIMUL, [LA464ALU01]>;
  def : WriteRes<WriteIMUL32, [LA464ALU01]>;
}
def : ReadAdvance<ReadIMUL, 0>;
def : ReadAdvance<ReadIMUL32, 0>;

// Integer instruction latency DIV
// Latency is 5-21 and throughput is 0.08-0.25, pick the worst case.
let Latency = 21, ReleaseAtCycles = [1, 11] in {
  def : WriteRes<WriteIDIV32, [LA464ALU0, LA464Div]>;
}
// Latency is 5-36 and throughput is 0.048-0.25, pick the worst case.
let Latency = 36, ReleaseAtCycles = [1, 20] in {
  def : WriteRes<WriteIDIV, [LA464ALU0, LA464Div]>;
}
def : ReadAdvance<ReadIDIV, 0>;
def : ReadAdvance<ReadIDIV32, 0>;

// Integer instruction latency SHIFT
def : WriteRes<WriteIShift, [LA464ALU]>;
def : WriteRes<WriteIShift32, [LA464ALU]>;

def : ReadAdvance<ReadIShift, 0>;
def : ReadAdvance<ReadIShift32, 0>;

// Integer instruction latency EXT
def : WriteRes<WriteIEXT, [LA464ALU]>;

def : ReadAdvance<ReadIEXT, 0>;

// Integer instruction latency CLO / CLZ / CTO / CTZ
def : WriteRes<WriteICL, [LA464ALU]>;
def : WriteRes<WriteICL32, [LA464ALU]>;
def : WriteRes<WriteICT, [LA464ALU]>;
def : WriteRes<WriteICT32, [LA464ALU]>;

def : ReadAdvance<ReadICL, 0>;
def : ReadAdvance<ReadICL32, 0>;
def : ReadAdvance<ReadICT, 0>;
def : ReadAdvance<ReadICT32, 0>;

// Integer instruction latency BytePick
def : WriteRes<WriteIBytePick, [LA464ALU01]>;
def : WriteRes<WriteIBytePick32, [LA464ALU01]>;

def : ReadAdvance<ReadIBytePick, 0>;
def : ReadAdvance<ReadIBytePick32, 0>;

// Integer instruction latency REV
def : WriteRes<WriteIRev, [LA464ALU]>;
def : WriteRes<WriteIRev32, [LA464ALU]>;

def : ReadAdvance<ReadIRev, 0>;
def : ReadAdvance<ReadIRev32, 0>;

// Integer instruction latency BitRev
def : WriteRes<WriteIBitRev, [LA464ALU]>;
def : WriteRes<WriteIBitRev32, [LA464ALU]>;

def : ReadAdvance<ReadIBitRev, 0>;
def : ReadAdvance<ReadIBitRev32, 0>;

// Integer instruction latency BSTRINS
def : WriteRes<WriteIBstrins, [LA464ALU01]>;
def : WriteRes<WriteIBstrins32, [LA464ALU01]>;

def : ReadAdvance<ReadIBstrins, 0>;
def : ReadAdvance<ReadIBstrins32, 0>;

// Integer instruction latency BSTRPICK
def : WriteRes<WriteIBstrPick, [LA464ALU]>;
def : WriteRes<WriteIBstrPick32, [LA464ALU]>;

def : ReadAdvance<ReadIBstrPick, 0>;
def : ReadAdvance<ReadIBstrPick32, 0>;

// Integer instruction latency SELECT
def : WriteRes<WriteIMASK, [LA464ALU]>;

def : ReadAdvance<ReadIMASK, 0>;

// Integer instruction latency Jump
def : WriteRes<WriteIJump, [LA464Branch]>;

def : ReadAdvance<ReadIJump, 0>;

// Integer instruction latency LD ST LDX STX
let Latency = 4 in {
  def : WriteRes<WriteLD, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteST, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteLDX, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteSTX, [LA464AGU0, LA464Store]>;
}
def : ReadAdvance<ReadSTData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadMemOffset, 0>;

// Integer instruction latency LDPTR STPTR
let Latency = 4 in {
  def : WriteRes<WriteLDPTR, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteSTPTR, [LA464AGU0, LA464Store]>;
}

// Integer instruction latency LDGT LDLE STGT STLE
let Latency = 4 in {
  def : WriteRes<WriteLDGT, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteLDLE, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteSTGT, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteSTLE, [LA464AGU0, LA464Store]>;
}
def : ReadAdvance<ReadMemCmp, 0>;

// Integer instruction latency CRC.W.B.W CRCC.W.B.W
let Latency = 5, ReleaseAtCycles = [4] in {
  def : WriteRes<WriteCRCB, [LA464ALU0]>;
}
// Integer instruction latency CRC.W.H.W CRCC.W.H.W
let Latency = 7, ReleaseAtCycles = [5] in {
  def : WriteRes<WriteCRCH, [LA464ALU0]>;
}
// Integer instruction latency CRC.W.W.W CRCC.W.W.W
let Latency = 11, ReleaseAtCycles = [7] in {
  def : WriteRes<WriteCRCW, [LA464ALU0]>;
}
// Integer instruction latency CRC.W.D.W CRCC.W.D.W
let Latency = 19, ReleaseAtCycles = [11] in {
  def : WriteRes<WriteCRCD, [LA464ALU0]>;
}
def : ReadAdvance<ReadCRCB, 0>;
def : ReadAdvance<ReadCRCH, 0>;
def : ReadAdvance<ReadCRCW, 0>;
def : ReadAdvance<ReadCRCD, 0>;

// Floating-point instruction latency FADD FSUB FMUL FDIV FMADD FMINMAX
let Latency = 5 in {
  def : WriteRes<WriteFADD, [LA464FPU]>;
  def : WriteRes<WriteFMUL, [LA464FPU]>;
  def : WriteRes<WriteFMADD, [LA464FPU]>;
}
let Latency = 2 in {
  def : WriteRes<WriteFMinMax, [LA464FPU]>;
}
def : ReadAdvance<ReadFADD, 0>;
def : ReadAdvance<ReadFMUL, 0>;
def : ReadAdvance<ReadFMADD, 0>;
def : ReadAdvance<ReadFMinMax, 0>;
// Latency is 6-14, pick the worst case.
let Latency = 14, ReleaseAtCycles = [1, 7] in {
  def : WriteRes<WriteFDIV32, [LA464FPU0, LA464FDiv]>;
}
// Latency is 6-20, pick the worst case.
let Latency = 20, ReleaseAtCycles = [1, 11] in {
  def : WriteRes<WriteFDIV, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadFDIV32, 0>;
def : ReadAdvance<ReadFDIV, 0>;

// Floating-point instruction latency FABS FNEG FSQRT FRECIP FRSQRT FSCALEB FLOGB FCOPYSIGN FCLASS FCMP
def : WriteRes<WriteFABS, [LA464FPU]>;
def : WriteRes<WriteFNEG, [LA464FPU]>;
def : WriteRes<WriteFCOPYSIGN, [LA464FPU]>;
let Latency = 2 in {
  def : WriteRes<WriteFCLASS, [LA464FPU]>;
}
def : WriteRes<WriteFCMP, [LA464FPU]>;
let Latency = 4 in {
  def : WriteRes<WriteFSCALEB, [LA464FPU]>;
  def : WriteRes<WriteFLOGB, [LA464FPU]>;
}
def : ReadAdvance<ReadFABS, 0>;
def : ReadAdvance<ReadFNEG, 0>;
def : ReadAdvance<ReadFSCALEB, 0>;
def : ReadAdvance<ReadFLOGB, 0>;
def : ReadAdvance<ReadFCOPYSIGN, 0>;
def : ReadAdvance<ReadFCLASS, 0>;
def : ReadAdvance<ReadFCMP, 0>;
// Latency is 8-16, pick the worst case.
let Latency = 16, ReleaseAtCycles = [1, 7] in {
  def : WriteRes<WriteFRECIP32, [LA464FPU0, LA464FDiv]>;
}
// Latency is 8-23, pick the worst case.
let Latency = 23, ReleaseAtCycles = [1, 11] in {
  def : WriteRes<WriteFRECIP, [LA464FPU0, LA464FDiv]>;
}
// Latency is 8-22, pick the worst case.
let Latency = 22, ReleaseAtCycles = [1, 10] in {
  def : WriteRes<WriteFSQRT32, [LA464FPU0, LA464FDiv]>;
}
// Latency is 8-36, pick the worst case.
let Latency = 36, ReleaseAtCycles = [1, 18] in {
  def : WriteRes<WriteFSQRT, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadFSQRT32, 0>;
def : ReadAdvance<ReadFSQRT, 0>;
def : ReadAdvance<ReadFRECIP32, 0>;
def : ReadAdvance<ReadFRECIP, 0>;
// Latency is 11-33, pick the worst case.
let Latency = 33, ReleaseAtCycles = [1, 16] in {
  def : WriteRes<WriteFRSQRT32, [LA464FPU0, LA464FDiv]>;
}
// Latency is 11-54, pick the worst case.
let Latency = 54, ReleaseAtCycles = [1, 27] in {
  def : WriteRes<WriteFRSQRT, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadFRSQRT32, 0>;
def : ReadAdvance<ReadFRSQRT, 0>;

// Floating-point instruction latency FCVT FSEL
let Latency = 2 in {
  def : WriteRes<WriteFCvtStoD, [LA464FPU]>;
  def : WriteRes<WriteFCvtDtoS, [LA464FPU]>;
}
let Latency = 4 in {
  def : WriteRes<WriteFCvtStoW, [LA464FPU]>;
  def : WriteRes<WriteFCvtStoL, [LA464FPU]>;
  def : WriteRes<WriteFCvtDtoW, [LA464FPU]>;
  def : WriteRes<WriteFCvtDtoL, [LA464FPU]>;

  def : WriteRes<WriteFCvtWtoS, [LA464FPU]>;
  def : WriteRes<WriteFCvtWtoD, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoS, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoD, [LA464FPU]>;

  def : WriteRes<WriteFCvtWtoS_RM, [LA464FPU]>;
  def : WriteRes<WriteFCvtWtoD_RM, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoS_RM, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoD_RM, [LA464FPU]>;

  def : WriteRes<WriteFCvtWtoS_RP, [LA464FPU]>;
  def : WriteRes<WriteFCvtWtoD_RP, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoS_RP, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoD_RP, [LA464FPU]>;

  def : WriteRes<WriteFCvtWtoS_RZ, [LA464FPU]>;
  def : WriteRes<WriteFCvtWtoD_RZ, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoS_RZ, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoD_RZ, [LA464FPU]>;

  def : WriteRes<WriteFCvtWtoS_RNE, [LA464FPU]>;
  def : WriteRes<WriteFCvtWtoD_RNE, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoS_RNE, [LA464FPU]>;
  def : WriteRes<WriteFCvtLtoD_RNE, [LA464FPU]>;

  def : WriteRes<WriteFCvtFStoIS, [LA464FPU]>;
  def : WriteRes<WriteFCvtFDtoID, [LA464FPU]>;
}
def : ReadAdvance<ReadFCvtStoD, 0>;
def : ReadAdvance<ReadFCvtDtoS, 0>;

def : ReadAdvance<ReadFCvtStoW, 0>;
def : ReadAdvance<ReadFCvtStoL, 0>;
def : ReadAdvance<ReadFCvtDtoW, 0>;
def : ReadAdvance<ReadFCvtDtoL, 0>;

def : ReadAdvance<ReadFCvtWtoS, 0>;
def : ReadAdvance<ReadFCvtWtoD, 0>;
def : ReadAdvance<ReadFCvtLtoS, 0>;
def : ReadAdvance<ReadFCvtLtoD, 0>;

def : ReadAdvance<ReadFCvtWtoS_RM, 0>;
def : ReadAdvance<ReadFCvtWtoD_RM, 0>;
def : ReadAdvance<ReadFCvtLtoS_RM, 0>;
def : ReadAdvance<ReadFCvtLtoD_RM, 0>;

def : ReadAdvance<ReadFCvtWtoS_RP, 0>;
def : ReadAdvance<ReadFCvtWtoD_RP, 0>;
def : ReadAdvance<ReadFCvtLtoS_RP, 0>;
def : ReadAdvance<ReadFCvtLtoD_RP, 0>;

def : ReadAdvance<ReadFCvtWtoS_RZ, 0>;
def : ReadAdvance<ReadFCvtWtoD_RZ, 0>;
def : ReadAdvance<ReadFCvtLtoS_RZ, 0>;
def : ReadAdvance<ReadFCvtLtoD_RZ, 0>;

def : ReadAdvance<ReadFCvtWtoS_RNE, 0>;
def : ReadAdvance<ReadFCvtWtoD_RNE, 0>;
def : ReadAdvance<ReadFCvtLtoS_RNE, 0>;
def : ReadAdvance<ReadFCvtLtoD_RNE, 0>;

def : ReadAdvance<ReadFCvtFStoIS, 0>;
def : ReadAdvance<ReadFCvtFDtoID, 0>;

def : WriteRes<WriteFSEL, [LA464FMov]>;
def : ReadAdvance<ReadFSEL, 0>;

// Floating-point instruction latency FMOV
def : WriteRes<WriteFMovS, [LA464FPU]>;
def : WriteRes<WriteFMovD, [LA464FPU]>;
def : WriteRes<WriteFMoveFRtoCF, [LA464FPU]>;
def : WriteRes<WriteFMoveCFtoFR, [LA464FPU]>;

def : ReadAdvance<ReadFMovS, 0>;
def : ReadAdvance<ReadFMovD, 0>;
def : ReadAdvance<ReadFMoveFRtoCF, 0>;
def : ReadAdvance<ReadFMoveCFtoFR, 0>;
let Latency = 2 in {
  def : WriteRes<WriteFMovGRtoFRW, [LA464FMov]>;
  def : WriteRes<WriteFMovGRtoFRD, [LA464FMov]>;
  def : WriteRes<WriteFMovFRtoGRS, [LA464FMov]>;
  def : WriteRes<WriteFMovFRHtoGRS, [LA464FMov]>;
  def : WriteRes<WriteFMovFRtoGRD, [LA464FMov]>;
}
let Latency = 3 in {
  def : WriteRes<WriteFMovGRtoFRHW, [LA464FMov]>;
}
def : ReadAdvance<ReadFMovGRtoFRW, 0>;
def : ReadAdvance<ReadFMovGRtoFRHW, 0>;
def : ReadAdvance<ReadFMovGRtoFRD, 0>;
def : ReadAdvance<ReadFMovFRtoGRS, 0>;
def : ReadAdvance<ReadFMovFRHtoGRS, 0>;
def : ReadAdvance<ReadFMovFRtoGRD, 0>;
let Latency = 7, ReleaseAtCycles = [6] in {
  def : WriteRes<WriteFMoveCFtoGR, [LA464FMov]>;
}
let Latency = 15, ReleaseAtCycles = [14] in {
  def : WriteRes<WriteFMoveGRtoCF, [LA464FMov]>;
}
def : ReadAdvance<ReadFMoveCFtoGR, 0>;
def : ReadAdvance<ReadFMoveGRtoCF, 0>;
let Latency = 7, ReleaseAtCycles = [6] in {
  def : WriteRes<WriteFMovFCSRtoGR, [LA464FMov]>;
}
def : ReadAdvance<ReadFMovFCSRtoGR, 0>;
let Latency = 15, ReleaseAtCycles = [14] in {
  def : WriteRes<WriteFMovGRtoFCSR, [LA464FMov]>;
}
def : ReadAdvance<ReadFMovGRtoFCSR, 0>;

// Floating-point instruction latency FJUMP
def : WriteRes<WriteFJump, [LA464Branch]>;

def : ReadAdvance<ReadFJump, 0>;

// Floating-point instruction latency FLD FST FLDX FSTX
let Latency = 5 in {
  def : WriteRes<WriteFLD, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteFST, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteFLDX, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteFSTX, [LA464AGU0, LA464Store]>;
}
def : ReadAdvance<ReadFSTData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFMemOffset, 0>;

// Floating-point instruction latency FLDGT FLDLE FSTGT FSTLE
let Latency = 5 in {
  def : WriteRes<WriteFLDGT, [LA464AGU0, LA464Load, LA464FPU]>;
  def : WriteRes<WriteFLDLE, [LA464AGU0, LA464Load, LA464FPU]>;
  def : WriteRes<WriteFSTGT, [LA464AGU0, LA464Store, LA464FPU]>;
  def : WriteRes<WriteFSTLE, [LA464AGU0, LA464Store, LA464FPU]>;
}
def : ReadAdvance<ReadFMemCmp, 0>;

// Integer instruction latency PCADD
def : WriteRes<WriteIPCAdd, [LA464AGU]>;

def : ReadAdvance<ReadIPCAdd, 0>;

//===----------------------------------------------------------------------===//

// Vector instruction - LSX / LASX
//
// Vector integer arithmetic, default latency is 1.
def : WriteRes<WriteVALU128, [LA464FPU]>;
def : WriteRes<WriteVALU256, [LA464FPU]>;

def : ReadAdvance<ReadVALU128, 0>;
def : ReadAdvance<ReadVALU256, 0>;

let Latency = 3 in {
  def : WriteRes<WriteVALUQ128, [LA464FPU]>;
  def : WriteRes<WriteVALUQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVALUQ128, 0>;
def : ReadAdvance<ReadVALUQ256, 0>;

def : WriteRes<WriteVAVG128, [LA464FPU]>;
def : WriteRes<WriteVAVG256, [LA464FPU]>;

def : ReadAdvance<ReadVAVG128, 0>;
def : ReadAdvance<ReadVAVG256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVAVGD128, [LA464FPU]>;
  def : WriteRes<WriteVAVGD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVAVGD128, 0>;
def : ReadAdvance<ReadVAVGD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVABSD128, [LA464FPU]>;
  def : WriteRes<WriteVABSD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVABSD128, 0>;
def : ReadAdvance<ReadVABSD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVADDA128, [LA464FPU]>;
  def : WriteRes<WriteVADDA256, [LA464FPU]>;
}
def : ReadAdvance<ReadVADDA128, 0>;
def : ReadAdvance<ReadVADDA256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVHADDW128, [LA464FPU]>;
  def : WriteRes<WriteVHADDW256, [LA464FPU]>;
}
def : ReadAdvance<ReadVHADDW128, 0>;
def : ReadAdvance<ReadVHADDW256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVHADDWQ128, [LA464FPU]>;
  def : WriteRes<WriteVHADDWQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVHADDWQ128, 0>;
def : ReadAdvance<ReadVHADDWQ256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVWADD128, [LA464FPU]>;
  def : WriteRes<WriteVWADD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVWADD128, 0>;
def : ReadAdvance<ReadVWADD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVWADDQ128, [LA464FPU]>;
  def : WriteRes<WriteVWADDQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVWADDQ128, 0>;
def : ReadAdvance<ReadVWADDQ256, 0>;

def : WriteRes<WriteVMINMAX128, [LA464FPU]>;
def : WriteRes<WriteVMINMAX256, [LA464FPU]>;

def : ReadAdvance<ReadVMINMAX128, 0>;
def : ReadAdvance<ReadVMINMAX256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVMINMAXD128, [LA464FPU]>;
  def : WriteRes<WriteVMINMAXD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVMINMAXD128, 0>;
def : ReadAdvance<ReadVMINMAXD256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVMUL128, [LA464FPU]>;
  def : WriteRes<WriteVMUL256, [LA464FPU]>;
}
def : ReadAdvance<ReadVMUL128, 0>;
def : ReadAdvance<ReadVMUL256, 0>;
let Latency = 7 in {
  def : WriteRes<WriteVMULQ128, [LA464FPU]>;
  def : WriteRes<WriteVMULQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVMULQ128, 0>;
def : ReadAdvance<ReadVMULQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVMADD128, [LA464FPU]>;
  def : WriteRes<WriteVMADD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVMADD128, 0>;
def : ReadAdvance<ReadVMADD256, 0>;
let Latency = 7 in {
  def : WriteRes<WriteVMADDQ128, [LA464FPU]>;
  def : WriteRes<WriteVMADDQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVMADDQ128, 0>;
def : ReadAdvance<ReadVMADDQ256, 0>;
// VMOD is slower than VDIV on LA464 and shares the same class.
// Latency is 26-63 and throughput is 0.029-0.067, pick the worst case.
let Latency = 63, ReleaseAtCycles = [1, 33] in {
  def : WriteRes<WriteVDIVB128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVDIVB256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVDIVB128, 0>;
def : ReadAdvance<ReadVDIVB256, 0>;
// Latency is 15-41 and throughput is 0.043-0.111, pick the worst case.
let Latency = 41, ReleaseAtCycles = [1, 22] in {
  def : WriteRes<WriteVDIVH128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVDIVH256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVDIVH128, 0>;
def : ReadAdvance<ReadVDIVH256, 0>;
// Latency is 10-30 and throughput is 0.059-0.167, pick the worst case.
let Latency = 30, ReleaseAtCycles = [1, 16] in {
  def : WriteRes<WriteVDIVW128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVDIVW256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVDIVW128, 0>;
def : ReadAdvance<ReadVDIVW256, 0>;
// Latency is 7-25 and throughput is 0.071-0.222, pick the worst case.
let Latency = 25, ReleaseAtCycles = [1, 13] in {
  def : WriteRes<WriteVDIVD128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVDIVD256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVDIVD128, 0>;
def : ReadAdvance<ReadVDIVD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSAT128, [LA464FPU]>;
  def : WriteRes<WriteVSAT256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSAT128, 0>;
def : ReadAdvance<ReadVSAT256, 0>;

def : WriteRes<WriteVSIGNCOV128, [LA464FPU]>;
def : WriteRes<WriteVSIGNCOV256, [LA464FPU]>;

def : ReadAdvance<ReadVSIGNCOV128, 0>;
def : ReadAdvance<ReadVSIGNCOV256, 0>;

def : WriteRes<WriteVEXTH128, [LA464FPU]>;
def : WriteRes<WriteVEXTH256, [LA464FPU]>;

def : ReadAdvance<ReadVEXTH128, 0>;
def : ReadAdvance<ReadVEXTH256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVEXT2XV256, [LA464FPU]>;
}
def : ReadAdvance<ReadVEXT2XV256, 0>;

def : WriteRes<WriteVEXTL128, [LA464FPU]>;
def : WriteRes<WriteVEXTL256, [LA464FPU]>;

def : ReadAdvance<ReadVEXTL128, 0>;
def : ReadAdvance<ReadVEXTL256, 0>;

def : WriteRes<WriteVLDI128, [LA464FPU]>;
def : WriteRes<WriteVLDI256, [LA464FPU]>;

def : ReadAdvance<ReadVLDI128, 0>;
def : ReadAdvance<ReadVLDI256, 0>;

//
// Vector compare, mask and condition flag
def : WriteRes<WriteVCMP128, [LA464FPU]>;
def : WriteRes<WriteVCMP256, [LA464FPU]>;

def : ReadAdvance<ReadVCMP128, 0>;
def : ReadAdvance<ReadVCMP256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVCMPD128, [LA464FPU]>;
  def : WriteRes<WriteVCMPD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVCMPD128, 0>;
def : ReadAdvance<ReadVCMPD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFCMP128, [LA464FPU]>;
  def : WriteRes<WriteVFCMP256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFCMP128, 0>;
def : ReadAdvance<ReadVFCMP256, 0>;

def : WriteRes<WriteVMSK128, [LA464FPU]>;
def : WriteRes<WriteVMSK256, [LA464FPU]>;

def : ReadAdvance<ReadVMSK128, 0>;
def : ReadAdvance<ReadVMSK256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSET128, [LA464FPU]>;
}
let Latency = 4 in {
  def : WriteRes<WriteVSET256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSET128, 0>;
def : ReadAdvance<ReadVSET256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFRSTP128, [LA464FPU]>;
  def : WriteRes<WriteVFRSTP256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFRSTP128, 0>;
def : ReadAdvance<ReadVFRSTP256, 0>;

//
// Vector bit manipulation
def : WriteRes<WriteVBIT128, [LA464FPU]>;
def : WriteRes<WriteVBIT256, [LA464FPU]>;

def : ReadAdvance<ReadVBIT128, 0>;
def : ReadAdvance<ReadVBIT256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVBITOPS128, [LA464FPU]>;
  def : WriteRes<WriteVBITOPS256, [LA464FPU]>;
}
def : ReadAdvance<ReadVBITOPS128, 0>;
def : ReadAdvance<ReadVBITOPS256, 0>;

def : WriteRes<WriteVBITSEL128, [LA464FPU]>;
def : WriteRes<WriteVBITSEL256, [LA464FPU]>;

def : ReadAdvance<ReadVBITSEL128, 0>;
def : ReadAdvance<ReadVBITSEL256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVPCNT128, [LA464FPU]>;
  def : WriteRes<WriteVPCNT256, [LA464FPU]>;
}
def : ReadAdvance<ReadVPCNT128, 0>;
def : ReadAdvance<ReadVPCNT256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVCL128, [LA464FPU]>;
  def : WriteRes<WriteVCL256, [LA464FPU]>;
}
def : ReadAdvance<ReadVCL128, 0>;
def : ReadAdvance<ReadVCL256, 0>;

//
// Vector shift
def : WriteRes<WriteVSHIFT128, [LA464FPU]>;
def : WriteRes<WriteVSHIFT256, [LA464FPU]>;

def : ReadAdvance<ReadVSHIFT128, 0>;
def : ReadAdvance<ReadVSHIFT256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVROTR128, [LA464FPU]>;
  def : WriteRes<WriteVROTR256, [LA464FPU]>;
}
def : ReadAdvance<ReadVROTR128, 0>;
def : ReadAdvance<ReadVROTR256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTR128, [LA464FPU]>;
  def : WriteRes<WriteVSHIFTR256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSHIFTR128, 0>;
def : ReadAdvance<ReadVSHIFTR256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSLLWIL128, [LA464VNarrow]>;
  def : WriteRes<WriteVSLLWIL256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSLLWIL128, 0>;
def : ReadAdvance<ReadVSLLWIL256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSHIFTN128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTN256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTN128, 0>;
def : ReadAdvance<ReadVSHIFTN256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNI128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTNI256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTNI128, 0>;
def : ReadAdvance<ReadVSHIFTNI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTNIQ128, [LA464FPU]>;
  def : WriteRes<WriteVSHIFTNIQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSHIFTNIQ128, 0>;
def : ReadAdvance<ReadVSHIFTNIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRN128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTRN256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTRN128, 0>;
def : ReadAdvance<ReadVSHIFTRN256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNI128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTRNI256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTRNI128, 0>;
def : ReadAdvance<ReadVSHIFTRNI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTRNIQ128, [LA464FPU]>;
  def : WriteRes<WriteVSHIFTRNIQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSHIFTRNIQ128, 0>;
def : ReadAdvance<ReadVSHIFTRNIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNS128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTNS256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTNS128, 0>;
def : ReadAdvance<ReadVSHIFTNS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNSI128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTNSI256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTNSI128, 0>;
def : ReadAdvance<ReadVSHIFTNSI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTNSIQ128, [LA464FPU]>;
  def : WriteRes<WriteVSHIFTNSIQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSHIFTNSIQ128, 0>;
def : ReadAdvance<ReadVSHIFTNSIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNS128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTRNS256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTRNS128, 0>;
def : ReadAdvance<ReadVSHIFTRNS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNSI128, [LA464VNarrow]>;
  def : WriteRes<WriteVSHIFTRNSI256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVSHIFTRNSI128, 0>;
def : ReadAdvance<ReadVSHIFTRNSI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTRNSIQ128, [LA464FPU]>;
  def : WriteRes<WriteVSHIFTRNSIQ256, [LA464FPU]>;
}
def : ReadAdvance<ReadVSHIFTRNSIQ128, 0>;
def : ReadAdvance<ReadVSHIFTRNSIQ256, 0>;

//
// Vector shuffle, permute and element move
def : WriteRes<WriteVPACK128, [LA464FPU]>;
def : WriteRes<WriteVPACK256, [LA464FPU]>;

def : ReadAdvance<ReadVPACK128, 0>;
def : ReadAdvance<ReadVPACK256, 0>;

def : WriteRes<WriteVSHUF128, [LA464FPU]>;
def : WriteRes<WriteVSHUF256, [LA464FPU]>;

def : ReadAdvance<ReadVSHUF128, 0>;
def : ReadAdvance<ReadVSHUF256, 0>;

def : WriteRes<WriteVSHUF4I128, [LA464FPU]>;
def : WriteRes<WriteVSHUF4I256, [LA464FPU]>;

def : ReadAdvance<ReadVSHUF4I128, 0>;
def : ReadAdvance<ReadVSHUF4I256, 0>;

def : WriteRes<WriteVHSELI256, [LA464FPU]>;

def : ReadAdvance<ReadVHSELI256, 0>;

def : WriteRes<WriteVPERMI128, [LA464FPU]>;
def : WriteRes<WriteVPERMI256, [LA464FPU]>;

def : ReadAdvance<ReadVPERMI128, 0>;
def : ReadAdvance<ReadVPERMI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPERM256, [LA464FPU]>;
}
def : ReadAdvance<ReadVPERM256, 0>;

def : WriteRes<WriteVEXTRINS128, [LA464FPU]>;
def : WriteRes<WriteVEXTRINS256, [LA464FPU]>;

def : ReadAdvance<ReadVEXTRINS128, 0>;
def : ReadAdvance<ReadVEXTRINS256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPICKVE256, [LA464FPU]>;
}
def : ReadAdvance<ReadVPICKVE256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVINSVE0256, [LA464FPU]>;
}
def : ReadAdvance<ReadVINSVE0256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLVE0256, [LA464FPU]>;
}
def : ReadAdvance<ReadVREPLVE0256, 0>;

def : WriteRes<WriteVREPLVEI128, [LA464FPU]>;
def : WriteRes<WriteVREPLVEI256, [LA464FPU]>;

def : ReadAdvance<ReadVREPLVEI128, 0>;
def : ReadAdvance<ReadVREPLVEI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLVE128, [LA464FMov]>;
  def : WriteRes<WriteVREPLVE256, [LA464FMov]>;
}
def : ReadAdvance<ReadVREPLVE128, 0>;
def : ReadAdvance<ReadVREPLVE256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLGR2VR128, [LA464FMov]>;
  def : WriteRes<WriteVREPLGR2VR256, [LA464FMov]>;
}
def : ReadAdvance<ReadVREPLGR2VR128, 0>;
def : ReadAdvance<ReadVREPLGR2VR256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVREPLGR2VRWD128, [LA464FMov]>;
  def : WriteRes<WriteVREPLGR2VRWD256, [LA464FMov]>;
}
def : ReadAdvance<ReadVREPLGR2VRWD128, 0>;
def : ReadAdvance<ReadVREPLGR2VRWD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVINSGR2VR128, [LA464FMov]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVINSGR2VR256, [LA464FMov]>;
}
def : ReadAdvance<ReadVINSGR2VR128, 0>;
def : ReadAdvance<ReadVINSGR2VR256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPICKVE2GR128, [LA464FMov]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVPICKVE2GR256, [LA464FMov]>;
}
def : ReadAdvance<ReadVPICKVE2GR128, 0>;
def : ReadAdvance<ReadVPICKVE2GR256, 0>;

//
// Vector floating-point arithmetic
let Latency = 5 in {
  def : WriteRes<WriteVFADD128, [LA464FPU]>;
  def : WriteRes<WriteVFADD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFADD128, 0>;
def : ReadAdvance<ReadVFADD256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFMUL128, [LA464FPU]>;
  def : WriteRes<WriteVFMUL256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFMUL128, 0>;
def : ReadAdvance<ReadVFMUL256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFMADD128, [LA464FPU]>;
  def : WriteRes<WriteVFMADD256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFMADD128, 0>;
def : ReadAdvance<ReadVFMADD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFMINMAX128, [LA464FPU]>;
  def : WriteRes<WriteVFMINMAX256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFMINMAX128, 0>;
def : ReadAdvance<ReadVFMINMAX256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFCLASS128, [LA464FPU]>;
  def : WriteRes<WriteVFCLASS256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFCLASS128, 0>;
def : ReadAdvance<ReadVFCLASS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFLOGB128, [LA464FPU]>;
  def : WriteRes<WriteVFLOGB256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFLOGB128, 0>;
def : ReadAdvance<ReadVFLOGB256, 0>;
// Latency is 9-24 and throughput is 0.0714-0.167, pick the worst case.
let Latency = 24, ReleaseAtCycles = [1, 13] in {
  def : WriteRes<WriteVFDIVS128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFDIVS256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFDIVS128, 0>;
def : ReadAdvance<ReadVFDIVS256, 0>;
// Latency is 6-20 and throughput is 0.0833-0.222, pick the worst case.
let Latency = 20, ReleaseAtCycles = [1, 11] in {
  def : WriteRes<WriteVFDIVD128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFDIVD256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFDIVD128, 0>;
def : ReadAdvance<ReadVFDIVD256, 0>;
// Latency is 11-39 and throughput is 0.05-0.167, pick the worst case.
let Latency = 39, ReleaseAtCycles = [1, 19] in {
  def : WriteRes<WriteVFSQRTS128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFSQRTS256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFSQRTS128, 0>;
def : ReadAdvance<ReadVFSQRTS256, 0>;
// Latency is 8-36 and throughput is 0.054-0.222, pick the worst case.
let Latency = 36, ReleaseAtCycles = [1, 18] in {
  def : WriteRes<WriteVFSQRTD128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFSQRTD256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFSQRTD128, 0>;
def : ReadAdvance<ReadVFSQRTD256, 0>;
// Latency is 11-27 and throughput is 0.071-0.167, pick the worst case.
let Latency = 27, ReleaseAtCycles = [1, 13] in {
  def : WriteRes<WriteVFRECIPS128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFRECIPS256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFRECIPS128, 0>;
def : ReadAdvance<ReadVFRECIPS256, 0>;
// Latency is 8-23 and throughput is 0.083-0.222, pick the worst case.
let Latency = 23, ReleaseAtCycles = [1, 11] in {
  def : WriteRes<WriteVFRECIPD128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFRECIPD256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFRECIPD128, 0>;
def : ReadAdvance<ReadVFRECIPD256, 0>;
// Latency is 17-61 and throughput is 0.032-0.111, pick the worst case.
let Latency = 61, ReleaseAtCycles = [1, 30] in {
  def : WriteRes<WriteVFRSQRTS128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFRSQRTS256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFRSQRTS128, 0>;
def : ReadAdvance<ReadVFRSQRTS256, 0>;
// Latency is 11-54 and throughput is 0.036-0.167, pick the worst case.
let Latency = 54, ReleaseAtCycles = [1, 27] in {
  def : WriteRes<WriteVFRSQRTD128, [LA464FPU0, LA464FDiv]>;
  def : WriteRes<WriteVFRSQRTD256, [LA464FPU0, LA464FDiv]>;
}
def : ReadAdvance<ReadVFRSQRTD128, 0>;
def : ReadAdvance<ReadVFRSQRTD256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFRINT128, [LA464FPU]>;
  def : WriteRes<WriteVFRINT256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFRINT128, 0>;
def : ReadAdvance<ReadVFRINT256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVFCVT128, [LA464VNarrow]>;
  def : WriteRes<WriteVFCVT256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVFCVT128, 0>;
def : ReadAdvance<ReadVFCVT256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFTINT128, [LA464FPU]>;
  def : WriteRes<WriteVFTINT256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFTINT128, 0>;
def : ReadAdvance<ReadVFTINT256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFTINTX128, [LA464VNarrow]>;
  def : WriteRes<WriteVFTINTX256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVFTINTX128, 0>;
def : ReadAdvance<ReadVFTINTX256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFFINT128, [LA464FPU]>;
  def : WriteRes<WriteVFFINT256, [LA464FPU]>;
}
def : ReadAdvance<ReadVFFINT128, 0>;
def : ReadAdvance<ReadVFFINT256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFFINTX128, [LA464VNarrow]>;
  def : WriteRes<WriteVFFINTX256, [LA464VNarrow]>;
}
def : ReadAdvance<ReadVFFINTX128, 0>;
def : ReadAdvance<ReadVFFINTX256, 0>;

//
// Vector memory access
let Latency = 5 in {
  def : WriteRes<WriteVLD128, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteVLD256, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteVLDX128, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteVLDX256, [LA464AGU0, LA464Load]>;
  def : WriteRes<WriteVLDREPL128, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteVLDREPL256, [LA464AGU, LA464Load]>;
  def : WriteRes<WriteVST128, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteVST256, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteVSTX128, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteVSTX256, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteVSTELM128, [LA464AGU0, LA464Store]>;
  def : WriteRes<WriteVSTELM256, [LA464AGU0, LA464Store]>;
}

// Vector memory access operands
def : ReadAdvance<ReadVSTData128, 0>;
def : ReadAdvance<ReadVMemBase128, 0>;
def : ReadAdvance<ReadVMemOffset128, 0>;
def : ReadAdvance<ReadVSTData256, 0>;
def : ReadAdvance<ReadVMemBase256, 0>;
def : ReadAdvance<ReadVMemOffset256, 0>;

// FRECIPE / FRSQRTE are only implemented by LA664.
let Unsupported = true in {
  def : WriteRes<WriteFRECIPE, []>;
  def : ReadAdvance<ReadFRECIPE, 0>;
  def : WriteRes<WriteFRSQRTE, []>;
  def : ReadAdvance<ReadFRSQRTE, 0>;
  def : WriteRes<WriteVFRECIPE128, []>;
  def : ReadAdvance<ReadVFRECIPE128, 0>;
  def : WriteRes<WriteVFRECIPE256, []>;
  def : ReadAdvance<ReadVFRECIPE256, 0>;
  def : WriteRes<WriteVFRSQRTE128, []>;
  def : ReadAdvance<ReadVFRSQRTE128, 0>;
  def : WriteRes<WriteVFRSQRTE256, []>;
  def : ReadAdvance<ReadVFRSQRTE256, 0>;
}

let Unsupported = true in {
  def : WriteRes<WriteAtomic, []>;
  def : ReadAdvance<ReadAtomic, 0>;
}

let Unsupported = true in {
  def : WriteRes<WritePseudo, []>;
  def : ReadAdvance<ReadPseudo, 0>;
}

let Unsupported = true in {
  def : WriteRes<WriteOthers, []>;
  def : ReadAdvance<ReadOthers, 0>;
}

} // SchedModel
