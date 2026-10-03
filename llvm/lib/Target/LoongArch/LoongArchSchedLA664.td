//==- LoongArchSchedLA664.td - LA664 Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for LA664 to support instruction
// scheduling and other instruction cost heuristics.
// Based on:
//  * Loongson 3A6000: A Star among Chinese CPUs, by Chips and Cheese
//    https://chipsandcheese.com/p/loongson-3a6000-a-star-among-chinese-cpus/
//  * Loongson-3A6000 instruction latency and throughput document, V0.1,
//    2024/07/24, Loongson Technology. Where the two sources disagree, the
//    measured latency and throughput of the document is used, e.g. it reports
//    two FMA operations per cycle instead of one.
//===----------------------------------------------------------------------===//

def LA664Model : SchedMachineModel {
  let IssueWidth = 6;             // 6-way decode and dispatch
  let MicroOpBufferSize = 256;    // Reorder Buffer size
  let LoadLatency = 4;
  let MispredictPenalty = 14;     // Estimated value
  let CompleteModel = 0;
}

let SchedModel = LA664Model in {
// Integer Execution Unit
//
// Execution Pipes
def LA664ALU0 : ProcResource<1>; // ALU, MUL, Branch
def LA664ALU1 : ProcResource<1>; // ALU, MUL, Branch
def LA664ALU2 : ProcResource<1>; // ALU
def LA664ALU3 : ProcResource<1>; // ALU

def LA664AGU0 : ProcResource<1>; // Load
def LA664AGU1 : ProcResource<1>; // Load
def LA664AGU2 : ProcResource<1>; // Store
def LA664AGU3 : ProcResource<1>; // Store

// Execution Units
//
// Integer division issued on port 0. (Estimiated)
def LA664Div : ProcResource<1>;

def LA664ALU01 : ProcResGroup<[LA664ALU0, LA664ALU1]>;
def LA664AGU01 : ProcResGroup<[LA664AGU0, LA664AGU1]>;
def LA664AGU23 : ProcResGroup<[LA664AGU2, LA664AGU3]>;

// Execution pipeline grouping
//
defvar LA664Branch = LA664ALU01;
defvar LA664Mul = LA664ALU01;
defvar LA664LoadAGU = LA664AGU01;
defvar LA664StoreAGU = LA664AGU23;

// Scheduling
//
// LA664 has an unified 48 entry Integer scheduler
def LA664ALU : ProcResGroup<[LA664ALU0, LA664ALU1, LA664ALU2, LA664ALU3]> {
  let BufferSize=48;
}

// LA664 has an unified 48 entry AGU scheduler
def LA664AGU : ProcResGroup<[LA664AGU0, LA664AGU1, LA664AGU2, LA664AGU3]> {
  let BufferSize=48;
}

// Floating-Point / Vector Unit
//
// Execution pipes
// There are four floating point/vector execution pipes.
def LA664FPU0 : ProcResource<1>; // FALU, FMUL, FMA, FDIV, FMOV, VALU, VFPU
def LA664FPU1 : ProcResource<1>; // FALU, FMUL, FMA, VALU, VFPU
def LA664FPU2 : ProcResource<1>; // FALU, VALU, VMUL
def LA664FPU3 : ProcResource<1>; // FALU, VALU, VMUL

// Execution Units
//
// FPU0 and FPU1 have float multiplication execution capability. They also
// execute the vector floating-point operations that are limited to two per
// cycle, e.g. VFMUL, VFRINT, VFCVT and the resizing VFTINT / VFFINT
// conversions.
defvar LA664FMul0 = LA664FPU0;
defvar LA664FMul1 = LA664FPU1;

// FP division/sqrt on port 0. (Estimated)
// It also executes the vector divide, square root and reciprocal operations.
def LA664FDiv : ProcResource<1>;

// Only FPU0 has float move execution capability. (Estimated)
// It also executes the vector operations that move data between a general
// purpose register and a vector register, which are limited to one per cycle.
defvar LA664FMov = LA664FPU0;

// FPU0 and FPU1 have float FMA execution capability. (Estimated)
// They also execute the vector floating-point FMA operations.
defvar LA664FMA0 = LA664FPU0;
defvar LA664FMA1 = LA664FPU1;

// FPU0 and FPU1 have float compare execution capability. (Estimated)
defvar LA664FCmp0 = LA664FPU0;
defvar LA664FCmp1 = LA664FPU1;

// FPU2 and FPU3 have vector integer multiplication execution capability. They
// also execute the other vector integer operations that are limited to two
// per cycle, e.g. VBITCLR, VPCNT, VSHUF and the narrowing shifts. (Estimated)
defvar LA664VMul0 = LA664FPU2;
defvar LA664VMul1 = LA664FPU3;

// Execution pipeline grouping
//
// Floating-point multiplies appear to use different pipes than the vector
// integer ones.
def LA664FMul : ProcResGroup<[LA664FMul0, LA664FMul1]>;
def LA664FMA : ProcResGroup<[LA664FMA0, LA664FMA1]>;
def LA664FCmp : ProcResGroup<[LA664FCmp0, LA664FCmp1]>;
def LA664VMul : ProcResGroup<[LA664VMul0, LA664VMul1]>;

// Scheduling
//
// LA664 has an unified 48 entry vector/fp scheduler.
// All four pipes execute the operations that reach a throughput of four per
// cycle, which covers both the scalar floating-point ones and the 128-bit /
// 256-bit vector ones.
def LA664FPU : ProcResGroup<[LA664FPU0, LA664FPU1, LA664FPU2, LA664FPU3]> {
  let BufferSize=48;
}

// Load-Store Unit
//
// The LS unit contains four pipelines, two for load and two for store.
// LA664 can handle two 256-bit vector accesses per cycle.
// These can be any combination of loads and stores.
def LA664Load : ProcResource<2> {
  // The LS unit utilizes a 80-entry load queue (LDQ).
  let BufferSize = 80;
}

def LA664Store : ProcResource<2> {
  // The LS unit utilizes a 64-entry store queue (STQ).
  let BufferSize = 64;
}

//===----------------------------------------------------------------------===//

// Base Instruction
// Integer arithmetic and logic, default latency is 1
def : WriteRes<WriteIALU, [LA664ALU]>;
def : WriteRes<WriteIALU32, [LA664ALU]>;

def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;

// Integer instruction latency ALSL
def : WriteRes<WriteIALSL, [LA664ALU]>;
def : WriteRes<WriteIALSL32, [LA664ALU]>;

def : ReadAdvance<ReadIALSL, 0>;
def : ReadAdvance<ReadIALSL32, 0>;

// Integer instruction latency MUL
let Latency = 4 in {
  def : WriteRes<WriteIMUL, [LA664Mul]>;
  def : WriteRes<WriteIMUL32, [LA664Mul]>;
}
def : ReadAdvance<ReadIMUL, 0>;
def : ReadAdvance<ReadIMUL32, 0>;

// Integer instruction latency DIV
// Latency 4-19, pick the worst case.
let Latency = 19, ReleaseAtCycles = [1,12] in {
  def : WriteRes<WriteIDIV32, [LA664ALU0, LA664Div]>;
}
// Latency 4-32, pick the worst case.
let Latency = 32, ReleaseAtCycles = [1,19] in {
  def : WriteRes<WriteIDIV, [LA664ALU0, LA664Div]>;
}
def : ReadAdvance<ReadIDIV, 0>;
def : ReadAdvance<ReadIDIV32, 0>;

// Integer instruction latency SHIFT
def : WriteRes<WriteIShift, [LA664ALU]>;
def : WriteRes<WriteIShift32, [LA664ALU]>;

def : ReadAdvance<ReadIShift, 0>;
def : ReadAdvance<ReadIShift32, 0>;

// Integer instruction latency EXT
def : WriteRes<WriteIEXT, [LA664ALU]>;

def : ReadAdvance<ReadIEXT, 0>;

// Integer instruction latency CLO / CLZ / CTO / CTZ
def : WriteRes<WriteICL, [LA664ALU]>;
def : WriteRes<WriteICL32, [LA664ALU]>;
def : WriteRes<WriteICT, [LA664ALU]>;
def : WriteRes<WriteICT32, [LA664ALU]>;

def : ReadAdvance<ReadICL, 0>;
def : ReadAdvance<ReadICL32, 0>;
def : ReadAdvance<ReadICT, 0>;
def : ReadAdvance<ReadICT32, 0>;

// Integer instruction latency BytePick
def : WriteRes<WriteIBytePick, [LA664ALU]>;
def : WriteRes<WriteIBytePick32, [LA664ALU]>;

def : ReadAdvance<ReadIBytePick, 0>;
def : ReadAdvance<ReadIBytePick32, 0>;

// Integer instruction latency REV
def : WriteRes<WriteIRev, [LA664ALU]>;
def : WriteRes<WriteIRev32, [LA664ALU]>;

def : ReadAdvance<ReadIRev, 0>;
def : ReadAdvance<ReadIRev32, 0>;

// Integer instruction latency BitRev
def : WriteRes<WriteIBitRev, [LA664ALU]>;
def : WriteRes<WriteIBitRev32, [LA664ALU]>;

def : ReadAdvance<ReadIBitRev, 0>;
def : ReadAdvance<ReadIBitRev32, 0>;

// Integer instruction latency BSTRINS
def : WriteRes<WriteIBstrins, [LA664ALU]>;
def : WriteRes<WriteIBstrins32, [LA664ALU]>;

def : ReadAdvance<ReadIBstrins, 0>;
def : ReadAdvance<ReadIBstrins32, 0>;

// Integer instruction latency BSTRPICK
def : WriteRes<WriteIBstrPick, [LA664ALU]>;
def : WriteRes<WriteIBstrPick32, [LA664ALU]>;

def : ReadAdvance<ReadIBstrPick, 0>;
def : ReadAdvance<ReadIBstrPick32, 0>;

// Integer instruction latency SELECT
def : WriteRes<WriteIMASK, [LA664ALU]>;

def : ReadAdvance<ReadIMASK, 0>;

// Integer instruction latency Jump
def : WriteRes<WriteIJump, [LA664Branch]>;

def : ReadAdvance<ReadIJump, 0>;

// Integer instruction latency LD ST LDX STX
let Latency = 4 in {
  def : WriteRes<WriteLD, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteST, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteLDX, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteSTX, [LA664StoreAGU, LA664Store]>;
}
def : ReadAdvance<ReadSTData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadMemOffset, 0>;

// Integer instruction latency LDPTR STPTR
let Latency = 4 in {
  def : WriteRes<WriteLDPTR, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteSTPTR, [LA664StoreAGU, LA664Store]>;
}

// Integer instruction latency LDGT LDLE STGT STLE
let Latency = 4 in {
  def : WriteRes<WriteLDGT, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteLDLE, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteSTGT, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteSTLE, [LA664StoreAGU, LA664Store]>;
}
def : ReadAdvance<ReadMemCmp, 0>;

// Integer instruction latency CRC.W.B[H/W/D].W CRCC.W.B[H/W/D].W
def : WriteRes<WriteCRCB, [LA664ALU01]>;
def : WriteRes<WriteCRCH, [LA664ALU01]>;
def : WriteRes<WriteCRCW, [LA664ALU01]>;
def : WriteRes<WriteCRCD, [LA664ALU01]>;

def : ReadAdvance<ReadCRCB, 0>;
def : ReadAdvance<ReadCRCH, 0>;
def : ReadAdvance<ReadCRCW, 0>;
def : ReadAdvance<ReadCRCD, 0>;

// Floating-point instruction latency FADD FSUB FMUL FDIV FMADD FMINMAX
let Latency = 3 in {
  def : WriteRes<WriteFADD, [LA664FPU]>;
}
let Latency = 5 in {
  def : WriteRes<WriteFMUL, [LA664FMul]>;
  def : WriteRes<WriteFMADD, [LA664FMA]>;
}
let Latency = 2 in {
  def : WriteRes<WriteFMinMax, [LA664FPU]>;
}
def : ReadAdvance<ReadFADD, 0>;
def : ReadAdvance<ReadFMUL, 0>;
def : ReadAdvance<ReadFMADD, 0>;
def : ReadAdvance<ReadFMinMax, 0>;
// Latency is 6-13, pick the worst case.
let Latency = 13, ReleaseAtCycles = [1, 3] in {
  def : WriteRes<WriteFDIV32, [LA664FPU0, LA664FDiv]>;
}
// Latency is 6-19, pick the worst case.
let Latency = 19, ReleaseAtCycles = [1, 4] in {
  def : WriteRes<WriteFDIV, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadFDIV32, 0>;
def : ReadAdvance<ReadFDIV, 0>;

// Floating-point instruction latency FABS FNEG FSQRT FRECIP FRSQRT FSCALEB FLOGB FCOPYSIGN FCLASS FCMP
def : WriteRes<WriteFABS, [LA664FPU]>;
def : WriteRes<WriteFNEG, [LA664FPU]>;
def : WriteRes<WriteFCOPYSIGN, [LA664FPU]>;
let Latency = 2 in {
  def : WriteRes<WriteFCLASS, [LA664FPU]>;
  def : WriteRes<WriteFCMP, [LA664FCmp]>;
}
let Latency = 4 in {
  def : WriteRes<WriteFSCALEB, [LA664FPU]>;
  def : WriteRes<WriteFLOGB, [LA664FPU]>;
}
def : ReadAdvance<ReadFABS, 0>;
def : ReadAdvance<ReadFNEG, 0>;
def : ReadAdvance<ReadFSCALEB, 0>;
def : ReadAdvance<ReadFLOGB, 0>;
def : ReadAdvance<ReadFCOPYSIGN, 0>;
def : ReadAdvance<ReadFCLASS, 0>;
def : ReadAdvance<ReadFCMP, 0>;
// Latency is 8-16, pick the worst case.
let Latency = 16, ReleaseAtCycles = [1, 3] in {
  def : WriteRes<WriteFRECIP32, [LA664FPU0, LA664FDiv]>;
}
// Latency is 8-23, pick the worst case.
let Latency = 23, ReleaseAtCycles = [1, 4] in {
  def : WriteRes<WriteFRECIP, [LA664FPU0, LA664FDiv]>;
}
// Latency is 8-22, pick the worst case.
let Latency = 22, ReleaseAtCycles = [1, 4] in {
  def : WriteRes<WriteFSQRT32, [LA664FPU0, LA664FDiv]>;
}
// Latency is 8-36, pick the worst case.
let Latency = 36, ReleaseAtCycles = [1, 7] in {
  def : WriteRes<WriteFSQRT, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadFSQRT32, 0>;
def : ReadAdvance<ReadFSQRT, 0>;
def : ReadAdvance<ReadFRECIP32, 0>;
def : ReadAdvance<ReadFRECIP, 0>;
// Latency is 11-33, pick the worst case.
let Latency = 33, ReleaseAtCycles = [1, 7] in {
  def : WriteRes<WriteFRSQRT32, [LA664FPU0, LA664FDiv]>;
}
// Latency is 11-54, pick the worst case.
let Latency = 54, ReleaseAtCycles = [1, 12] in {
  def : WriteRes<WriteFRSQRT, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadFRSQRT32, 0>;
def : ReadAdvance<ReadFRSQRT, 0>;

// Floating-point instruction latency FCVT FSEL
let Latency = 2 in {
  def : WriteRes<WriteFCvtStoD, [LA664FPU]>;
  def : WriteRes<WriteFCvtDtoS, [LA664FPU]>;
}
let Latency = 4 in {
  def : WriteRes<WriteFCvtStoW, [LA664FPU]>;
  def : WriteRes<WriteFCvtStoL, [LA664FPU]>;
  def : WriteRes<WriteFCvtDtoW, [LA664FPU]>;
  def : WriteRes<WriteFCvtDtoL, [LA664FPU]>;

  def : WriteRes<WriteFCvtWtoS, [LA664FPU]>;
  def : WriteRes<WriteFCvtWtoD, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoS, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoD, [LA664FPU]>;

  def : WriteRes<WriteFCvtWtoS_RM, [LA664FPU]>;
  def : WriteRes<WriteFCvtWtoD_RM, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoS_RM, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoD_RM, [LA664FPU]>;

  def : WriteRes<WriteFCvtWtoS_RP, [LA664FPU]>;
  def : WriteRes<WriteFCvtWtoD_RP, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoS_RP, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoD_RP, [LA664FPU]>;

  def : WriteRes<WriteFCvtWtoS_RZ, [LA664FPU]>;
  def : WriteRes<WriteFCvtWtoD_RZ, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoS_RZ, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoD_RZ, [LA664FPU]>;

  def : WriteRes<WriteFCvtWtoS_RNE, [LA664FPU]>;
  def : WriteRes<WriteFCvtWtoD_RNE, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoS_RNE, [LA664FPU]>;
  def : WriteRes<WriteFCvtLtoD_RNE, [LA664FPU]>;

  def : WriteRes<WriteFCvtFStoIS, [LA664FPU]>;
  def : WriteRes<WriteFCvtFDtoID, [LA664FPU]>;
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

def : WriteRes<WriteFSEL, [LA664FCmp]>;
def : ReadAdvance<ReadFSEL, 0>;

// Floating-point instruction latency FMOV
def : WriteRes<WriteFMovS, [LA664FPU]>;
def : WriteRes<WriteFMovD, [LA664FPU]>;
def : WriteRes<WriteFMoveFRtoCF, [LA664FCmp]>;
def : WriteRes<WriteFMoveCFtoFR, [LA664FCmp]>;

def : ReadAdvance<ReadFMovS, 0>;
def : ReadAdvance<ReadFMovD, 0>;
def : ReadAdvance<ReadFMoveFRtoCF, 0>;
def : ReadAdvance<ReadFMoveCFtoFR, 0>;
let Latency = 2 in {
  def : WriteRes<WriteFMovGRtoFRW, [LA664FMov]>;
  def : WriteRes<WriteFMovGRtoFRD, [LA664FMov]>;
  def : WriteRes<WriteFMovFRtoGRS, [LA664FMov]>;
  def : WriteRes<WriteFMovFRHtoGRS, [LA664FMov]>;
  def : WriteRes<WriteFMovFRtoGRD, [LA664FMov]>;
}
let Latency = 3 in {
  def : WriteRes<WriteFMovGRtoFRHW, [LA664FMov]>;
}
def : ReadAdvance<ReadFMovGRtoFRW, 0>;
def : ReadAdvance<ReadFMovGRtoFRHW, 0>;
def : ReadAdvance<ReadFMovGRtoFRD, 0>;
def : ReadAdvance<ReadFMovFRtoGRS, 0>;
def : ReadAdvance<ReadFMovFRHtoGRS, 0>;
def : ReadAdvance<ReadFMovFRtoGRD, 0>;
let Latency = 2 in {
  def : WriteRes<WriteFMoveCFtoGR, [LA664FMov]>;
  def : WriteRes<WriteFMoveGRtoCF, [LA664FMov]>;
}
def : ReadAdvance<ReadFMoveCFtoGR, 0>;
def : ReadAdvance<ReadFMoveGRtoCF, 0>;
let Latency = 7 in {
  def : WriteRes<WriteFMovFCSRtoGR, [LA664FMov]>;
}
def : ReadAdvance<ReadFMovFCSRtoGR, 0>;
let Latency = 15 in {
  def : WriteRes<WriteFMovGRtoFCSR, [LA664FMov]>;
}
def : ReadAdvance<ReadFMovGRtoFCSR, 0>;

// Floating-point instruction latency FJUMP
def : WriteRes<WriteFJump, [LA664Branch]>;

def : ReadAdvance<ReadFJump, 0>;

// Floating-point instruction latency FLD FST FLDX FSTX
let Latency = 5 in {
  def : WriteRes<WriteFLD, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteFST, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteFLDX, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteFSTX, [LA664StoreAGU, LA664Store]>;
}
def : ReadAdvance<ReadFSTData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFMemOffset, 0>;

// Floating-point instruction latency FLDGT FLDLE FSTGT FSTLE
let Latency = 5 in {
  def : WriteRes<WriteFLDGT, [LA664LoadAGU, LA664Load, LA664FCmp]>;
  def : WriteRes<WriteFLDLE, [LA664LoadAGU, LA664Load, LA664FCmp]>;
  def : WriteRes<WriteFSTGT, [LA664StoreAGU, LA664Store, LA664FCmp]>;
  def : WriteRes<WriteFSTLE, [LA664StoreAGU, LA664Store, LA664FCmp]>;
}
def : ReadAdvance<ReadFMemCmp, 0>;

// Floating-point instruction latency FRECIPE/FRSQRTE
let Latency = 5 in {
  def : WriteRes<WriteFRECIPE, [LA664FMul]>;
  def : WriteRes<WriteFRSQRTE, [LA664FMul]>;
}
def : ReadAdvance<ReadFRECIPE, 0>;
def : ReadAdvance<ReadFRSQRTE, 0>;

// Integer instruction latency PCADD
def : WriteRes<WriteIPCAdd, [LA664LoadAGU]>;

def : ReadAdvance<ReadIPCAdd, 0>;

//===----------------------------------------------------------------------===//

// Vector instruction - LSX / LASX
//
// Vector integer arithmetic, default latency is 1.
def : WriteRes<WriteVALU128, [LA664FPU]>;
def : WriteRes<WriteVALU256, [LA664FPU]>;

def : ReadAdvance<ReadVALU128, 0>;
def : ReadAdvance<ReadVALU256, 0>;

let Latency = 3 in {
  def : WriteRes<WriteVALUQ128, [LA664VMul]>;
  def : WriteRes<WriteVALUQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVALUQ128, 0>;
def : ReadAdvance<ReadVALUQ256, 0>;

def : WriteRes<WriteVAVG128, [LA664FPU]>;
def : WriteRes<WriteVAVG256, [LA664FPU]>;

def : ReadAdvance<ReadVAVG128, 0>;
def : ReadAdvance<ReadVAVG256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVAVGD128, [LA664FPU]>;
  def : WriteRes<WriteVAVGD256, [LA664FPU]>;
}
def : ReadAdvance<ReadVAVGD128, 0>;
def : ReadAdvance<ReadVAVGD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVABSD128, [LA664VMul]>;
  def : WriteRes<WriteVABSD256, [LA664VMul]>;
}
def : ReadAdvance<ReadVABSD128, 0>;
def : ReadAdvance<ReadVABSD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVADDA128, [LA664VMul]>;
  def : WriteRes<WriteVADDA256, [LA664VMul]>;
}
def : ReadAdvance<ReadVADDA128, 0>;
def : ReadAdvance<ReadVADDA256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVHADDW128, [LA664VMul]>;
  def : WriteRes<WriteVHADDW256, [LA664VMul]>;
}
def : ReadAdvance<ReadVHADDW128, 0>;
def : ReadAdvance<ReadVHADDW256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVHADDWQ128, [LA664VMul]>;
  def : WriteRes<WriteVHADDWQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVHADDWQ128, 0>;
def : ReadAdvance<ReadVHADDWQ256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVWADD128, [LA664VMul]>;
  def : WriteRes<WriteVWADD256, [LA664VMul]>;
}
def : ReadAdvance<ReadVWADD128, 0>;
def : ReadAdvance<ReadVWADD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVWADDQ128, [LA664VMul]>;
  def : WriteRes<WriteVWADDQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVWADDQ128, 0>;
def : ReadAdvance<ReadVWADDQ256, 0>;

def : WriteRes<WriteVMINMAX128, [LA664FPU]>;
def : WriteRes<WriteVMINMAX256, [LA664FPU]>;

def : ReadAdvance<ReadVMINMAX128, 0>;
def : ReadAdvance<ReadVMINMAX256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVMINMAXD128, [LA664FPU]>;
  def : WriteRes<WriteVMINMAXD256, [LA664FPU]>;
}
def : ReadAdvance<ReadVMINMAXD128, 0>;
def : ReadAdvance<ReadVMINMAXD256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVMUL128, [LA664VMul]>;
  def : WriteRes<WriteVMUL256, [LA664VMul]>;
}
def : ReadAdvance<ReadVMUL128, 0>;
def : ReadAdvance<ReadVMUL256, 0>;
let Latency = 7 in {
  def : WriteRes<WriteVMULQ128, [LA664VMul]>;
  def : WriteRes<WriteVMULQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVMULQ128, 0>;
def : ReadAdvance<ReadVMULQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVMADD128, [LA664VMul]>;
  def : WriteRes<WriteVMADD256, [LA664VMul]>;
}
def : ReadAdvance<ReadVMADD128, 0>;
def : ReadAdvance<ReadVMADD256, 0>;
let Latency = 7 in {
  def : WriteRes<WriteVMADDQ128, [LA664VMul]>;
  def : WriteRes<WriteVMADDQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVMADDQ128, 0>;
def : ReadAdvance<ReadVMADDQ256, 0>;
// Latency is 24-52 and throughput is 0.03-0.074, pick the worst case.
let Latency = 52, ReleaseAtCycles = [1, 32] in {
  def : WriteRes<WriteVDIVB128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVDIVB256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVDIVB128, 0>;
def : ReadAdvance<ReadVDIVB256, 0>;
// Latency is 14-35 and throughput is 0.05-0.13, pick the worst case.
let Latency = 35, ReleaseAtCycles = [1, 19] in {
  def : WriteRes<WriteVDIVH128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVDIVH256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVDIVH128, 0>;
def : ReadAdvance<ReadVDIVH256, 0>;
// Latency is 9-26 and throughput is 0.065-0.22, pick the worst case.
let Latency = 26, ReleaseAtCycles = [1, 14] in {
  def : WriteRes<WriteVDIVW128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVDIVW256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVDIVW128, 0>;
def : ReadAdvance<ReadVDIVW256, 0>;
// Latency is 6-22 and throughput is 0.08-0.33, pick the worst case.
let Latency = 22, ReleaseAtCycles = [1, 12] in {
  def : WriteRes<WriteVDIVD128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVDIVD256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVDIVD128, 0>;
def : ReadAdvance<ReadVDIVD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSAT128, [LA664VMul]>;
  def : WriteRes<WriteVSAT256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSAT128, 0>;
def : ReadAdvance<ReadVSAT256, 0>;

def : WriteRes<WriteVSIGNCOV128, [LA664VMul]>;
def : WriteRes<WriteVSIGNCOV256, [LA664VMul]>;

def : ReadAdvance<ReadVSIGNCOV128, 0>;
def : ReadAdvance<ReadVSIGNCOV256, 0>;

def : WriteRes<WriteVEXTH128, [LA664FPU]>;
def : WriteRes<WriteVEXTH256, [LA664FPU]>;

def : ReadAdvance<ReadVEXTH128, 0>;
def : ReadAdvance<ReadVEXTH256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVEXT2XV256, [LA664FPU]>;
}
def : ReadAdvance<ReadVEXT2XV256, 0>;

def : WriteRes<WriteVEXTL128, [LA664FPU]>;
def : WriteRes<WriteVEXTL256, [LA664FPU]>;

def : ReadAdvance<ReadVEXTL128, 0>;
def : ReadAdvance<ReadVEXTL256, 0>;

def : WriteRes<WriteVLDI128, [LA664FPU]>;
def : WriteRes<WriteVLDI256, [LA664FPU]>;

def : ReadAdvance<ReadVLDI128, 0>;
def : ReadAdvance<ReadVLDI256, 0>;

//
// Vector compare, mask and condition flag
def : WriteRes<WriteVCMP128, [LA664FPU]>;
def : WriteRes<WriteVCMP256, [LA664FPU]>;

def : ReadAdvance<ReadVCMP128, 0>;
def : ReadAdvance<ReadVCMP256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVCMPD128, [LA664FPU]>;
  def : WriteRes<WriteVCMPD256, [LA664FPU]>;
}
def : ReadAdvance<ReadVCMPD128, 0>;
def : ReadAdvance<ReadVCMPD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFCMP128, [LA664FPU]>;
  def : WriteRes<WriteVFCMP256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFCMP128, 0>;
def : ReadAdvance<ReadVFCMP256, 0>;

def : WriteRes<WriteVMSK128, [LA664FPU]>;
def : WriteRes<WriteVMSK256, [LA664FPU]>;

def : ReadAdvance<ReadVMSK128, 0>;
def : ReadAdvance<ReadVMSK256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSET128, [LA664VMul]>;
}
let Latency = 3 in {
  def : WriteRes<WriteVSET256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSET128, 0>;
def : ReadAdvance<ReadVSET256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFRSTP128, [LA664VMul]>;
  def : WriteRes<WriteVFRSTP256, [LA664VMul]>;
}
def : ReadAdvance<ReadVFRSTP128, 0>;
def : ReadAdvance<ReadVFRSTP256, 0>;

//
// Vector bit manipulation
def : WriteRes<WriteVBIT128, [LA664FPU]>;
def : WriteRes<WriteVBIT256, [LA664FPU]>;

def : ReadAdvance<ReadVBIT128, 0>;
def : ReadAdvance<ReadVBIT256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVBITOPS128, [LA664VMul]>;
  def : WriteRes<WriteVBITOPS256, [LA664VMul]>;
}
def : ReadAdvance<ReadVBITOPS128, 0>;
def : ReadAdvance<ReadVBITOPS256, 0>;

def : WriteRes<WriteVBITSEL128, [LA664VMul]>;
def : WriteRes<WriteVBITSEL256, [LA664VMul]>;

def : ReadAdvance<ReadVBITSEL128, 0>;
def : ReadAdvance<ReadVBITSEL256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVPCNT128, [LA664VMul]>;
  def : WriteRes<WriteVPCNT256, [LA664VMul]>;
}
def : ReadAdvance<ReadVPCNT128, 0>;
def : ReadAdvance<ReadVPCNT256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVCL128, [LA664FPU]>;
  def : WriteRes<WriteVCL256, [LA664FPU]>;
}
def : ReadAdvance<ReadVCL128, 0>;
def : ReadAdvance<ReadVCL256, 0>;

//
// Vector shift
def : WriteRes<WriteVSHIFT128, [LA664FPU]>;
def : WriteRes<WriteVSHIFT256, [LA664FPU]>;

def : ReadAdvance<ReadVSHIFT128, 0>;
def : ReadAdvance<ReadVSHIFT256, 0>;

def : WriteRes<WriteVROTR128, [LA664FPU]>;
def : WriteRes<WriteVROTR256, [LA664FPU]>;

def : ReadAdvance<ReadVROTR128, 0>;
def : ReadAdvance<ReadVROTR256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTR128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTR256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTR128, 0>;
def : ReadAdvance<ReadVSHIFTR256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSLLWIL128, [LA664VMul]>;
  def : WriteRes<WriteVSLLWIL256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSLLWIL128, 0>;
def : ReadAdvance<ReadVSLLWIL256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVSHIFTN128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTN256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTN128, 0>;
def : ReadAdvance<ReadVSHIFTN256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNI128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTNI256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTNI128, 0>;
def : ReadAdvance<ReadVSHIFTNI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTNIQ128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTNIQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTNIQ128, 0>;
def : ReadAdvance<ReadVSHIFTNIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRN128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRN256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRN128, 0>;
def : ReadAdvance<ReadVSHIFTRN256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNI128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRNI256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRNI128, 0>;
def : ReadAdvance<ReadVSHIFTRNI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTRNIQ128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRNIQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRNIQ128, 0>;
def : ReadAdvance<ReadVSHIFTRNIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNS128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTNS256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTNS128, 0>;
def : ReadAdvance<ReadVSHIFTNS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTNSI128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTNSI256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTNSI128, 0>;
def : ReadAdvance<ReadVSHIFTNSI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTNSIQ128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTNSIQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTNSIQ128, 0>;
def : ReadAdvance<ReadVSHIFTNSIQ256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNS128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRNS256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRNS128, 0>;
def : ReadAdvance<ReadVSHIFTRNS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVSHIFTRNSI128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRNSI256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRNSI128, 0>;
def : ReadAdvance<ReadVSHIFTRNSI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVSHIFTRNSIQ128, [LA664VMul]>;
  def : WriteRes<WriteVSHIFTRNSIQ256, [LA664VMul]>;
}
def : ReadAdvance<ReadVSHIFTRNSIQ128, 0>;
def : ReadAdvance<ReadVSHIFTRNSIQ256, 0>;

//
// Vector shuffle, permute and element move
def : WriteRes<WriteVPACK128, [LA664FPU]>;
def : WriteRes<WriteVPACK256, [LA664FPU]>;

def : ReadAdvance<ReadVPACK128, 0>;
def : ReadAdvance<ReadVPACK256, 0>;

def : WriteRes<WriteVSHUF128, [LA664VMul]>;
def : WriteRes<WriteVSHUF256, [LA664VMul]>;

def : ReadAdvance<ReadVSHUF128, 0>;
def : ReadAdvance<ReadVSHUF256, 0>;

def : WriteRes<WriteVSHUF4I128, [LA664FPU]>;
def : WriteRes<WriteVSHUF4I256, [LA664FPU]>;

def : ReadAdvance<ReadVSHUF4I128, 0>;
def : ReadAdvance<ReadVSHUF4I256, 0>;

def : WriteRes<WriteVHSELI256, [LA664FPU]>;

def : ReadAdvance<ReadVHSELI256, 0>;

def : WriteRes<WriteVPERMI128, [LA664FPU]>;
def : WriteRes<WriteVPERMI256, [LA664FPU]>;

def : ReadAdvance<ReadVPERMI128, 0>;
def : ReadAdvance<ReadVPERMI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPERM256, [LA664FPU]>;
}
def : ReadAdvance<ReadVPERM256, 0>;

def : WriteRes<WriteVEXTRINS128, [LA664FPU]>;
def : WriteRes<WriteVEXTRINS256, [LA664FPU]>;

def : ReadAdvance<ReadVEXTRINS128, 0>;
def : ReadAdvance<ReadVEXTRINS256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPICKVE256, [LA664FPU]>;
}
def : ReadAdvance<ReadVPICKVE256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVINSVE0256, [LA664FPU]>;
}
def : ReadAdvance<ReadVINSVE0256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLVE0256, [LA664FPU]>;
}
def : ReadAdvance<ReadVREPLVE0256, 0>;

def : WriteRes<WriteVREPLVEI128, [LA664FPU]>;
def : WriteRes<WriteVREPLVEI256, [LA664FPU]>;

def : ReadAdvance<ReadVREPLVEI128, 0>;
def : ReadAdvance<ReadVREPLVEI256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLVE128, [LA664FMov]>;
  def : WriteRes<WriteVREPLVE256, [LA664FMov]>;
}
def : ReadAdvance<ReadVREPLVE128, 0>;
def : ReadAdvance<ReadVREPLVE256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVREPLGR2VR128, [LA664FMov]>;
  def : WriteRes<WriteVREPLGR2VR256, [LA664FMov]>;
}
def : ReadAdvance<ReadVREPLGR2VR128, 0>;
def : ReadAdvance<ReadVREPLGR2VR256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVREPLGR2VRWD128, [LA664FMov]>;
  def : WriteRes<WriteVREPLGR2VRWD256, [LA664FMov]>;
}
def : ReadAdvance<ReadVREPLGR2VRWD128, 0>;
def : ReadAdvance<ReadVREPLGR2VRWD256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVINSGR2VR128, [LA664FMov]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVINSGR2VR256, [LA664FMov]>;
}
def : ReadAdvance<ReadVINSGR2VR128, 0>;
def : ReadAdvance<ReadVINSGR2VR256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVPICKVE2GR128, [LA664FMov]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVPICKVE2GR256, [LA664FMov]>;
}
def : ReadAdvance<ReadVPICKVE2GR128, 0>;
def : ReadAdvance<ReadVPICKVE2GR256, 0>;

//
// Vector floating-point arithmetic
let Latency = 3 in {
  def : WriteRes<WriteVFADD128, [LA664FPU]>;
  def : WriteRes<WriteVFADD256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFADD128, 0>;
def : ReadAdvance<ReadVFADD256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFMUL128, [LA664FMul]>;
  def : WriteRes<WriteVFMUL256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFMUL128, 0>;
def : ReadAdvance<ReadVFMUL256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFMADD128, [LA664FMA]>;
  def : WriteRes<WriteVFMADD256, [LA664FMA]>;
}
def : ReadAdvance<ReadVFMADD128, 0>;
def : ReadAdvance<ReadVFMADD256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFMINMAX128, [LA664FPU]>;
  def : WriteRes<WriteVFMINMAX256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFMINMAX128, 0>;
def : ReadAdvance<ReadVFMINMAX256, 0>;
let Latency = 2 in {
  def : WriteRes<WriteVFCLASS128, [LA664FPU]>;
  def : WriteRes<WriteVFCLASS256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFCLASS128, 0>;
def : ReadAdvance<ReadVFCLASS256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFLOGB128, [LA664FPU]>;
  def : WriteRes<WriteVFLOGB256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFLOGB128, 0>;
def : ReadAdvance<ReadVFLOGB256, 0>;
// Latency is 9-23 and throughput is 0.08-0.22, pick the worst case.
let Latency = 23, ReleaseAtCycles = [1, 12] in {
  def : WriteRes<WriteVFDIVS128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFDIVS256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFDIVS128, 0>;
def : ReadAdvance<ReadVFDIVS256, 0>;
// Latency is 6-19 and throughput is 0.095-0.33, pick the worst case.
let Latency = 19, ReleaseAtCycles = [1, 10] in {
  def : WriteRes<WriteVFDIVD128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFDIVD256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFDIVD128, 0>;
def : ReadAdvance<ReadVFDIVD256, 0>;
// Latency is 11-39 and throughput is 0.054-0.222, pick the worst case.
let Latency = 39, ReleaseAtCycles = [1, 18] in {
  def : WriteRes<WriteVFSQRTS128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFSQRTS256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFSQRTS128, 0>;
def : ReadAdvance<ReadVFSQRTS256, 0>;
// Latency is 8-36 and throughput is 0.059-0.333, pick the worst case.
let Latency = 36, ReleaseAtCycles = [1, 16] in {
  def : WriteRes<WriteVFSQRTD128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFSQRTD256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFSQRTD128, 0>;
def : ReadAdvance<ReadVFSQRTD256, 0>;
// Latency is 11-27 and throughput is 0.08-0.222, pick the worst case.
let Latency = 27, ReleaseAtCycles = [1, 12] in {
  def : WriteRes<WriteVFRECIPS128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFRECIPS256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFRECIPS128, 0>;
def : ReadAdvance<ReadVFRECIPS256, 0>;
// Latency is 8-23 and throughput is 0.095-0.333, pick the worst case.
let Latency = 23, ReleaseAtCycles = [1, 10] in {
  def : WriteRes<WriteVFRECIPD128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFRECIPD256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFRECIPD128, 0>;
def : ReadAdvance<ReadVFRECIPD256, 0>;
// Latency is 17-61 and throughput is 0.034-0.133, pick the worst case.
let Latency = 61, ReleaseAtCycles = [1, 28] in {
  def : WriteRes<WriteVFRSQRTS128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFRSQRTS256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFRSQRTS128, 0>;
def : ReadAdvance<ReadVFRSQRTS256, 0>;
// Latency is 11-54 and throughput is 0.038-0.222, pick the worst case.
let Latency = 54, ReleaseAtCycles = [1, 25] in {
  def : WriteRes<WriteVFRSQRTD128, [LA664FPU0, LA664FDiv]>;
  def : WriteRes<WriteVFRSQRTD256, [LA664FPU0, LA664FDiv]>;
}
def : ReadAdvance<ReadVFRSQRTD128, 0>;
def : ReadAdvance<ReadVFRSQRTD256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFRECIPE128, [LA664FMul]>;
  def : WriteRes<WriteVFRECIPE256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFRECIPE128, 0>;
def : ReadAdvance<ReadVFRECIPE256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFRSQRTE128, [LA664FMul]>;
  def : WriteRes<WriteVFRSQRTE256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFRSQRTE128, 0>;
def : ReadAdvance<ReadVFRSQRTE256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFRINT128, [LA664FMul]>;
  def : WriteRes<WriteVFRINT256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFRINT128, 0>;
def : ReadAdvance<ReadVFRINT256, 0>;
let Latency = 3 in {
  def : WriteRes<WriteVFCVT128, [LA664FMul]>;
  def : WriteRes<WriteVFCVT256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFCVT128, 0>;
def : ReadAdvance<ReadVFCVT256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFTINT128, [LA664FPU]>;
  def : WriteRes<WriteVFTINT256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFTINT128, 0>;
def : ReadAdvance<ReadVFTINT256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFTINTX128, [LA664FMul]>;
  def : WriteRes<WriteVFTINTX256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFTINTX128, 0>;
def : ReadAdvance<ReadVFTINTX256, 0>;
let Latency = 4 in {
  def : WriteRes<WriteVFFINT128, [LA664FPU]>;
  def : WriteRes<WriteVFFINT256, [LA664FPU]>;
}
def : ReadAdvance<ReadVFFINT128, 0>;
def : ReadAdvance<ReadVFFINT256, 0>;
let Latency = 5 in {
  def : WriteRes<WriteVFFINTX128, [LA664FMul]>;
  def : WriteRes<WriteVFFINTX256, [LA664FMul]>;
}
def : ReadAdvance<ReadVFFINTX128, 0>;
def : ReadAdvance<ReadVFFINTX256, 0>;

//
// Vector memory access
let Latency = 5 in {
  def : WriteRes<WriteVLD128, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteVLD256, [LA664LoadAGU, LA664Load]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVLDX128, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteVLDX256, [LA664LoadAGU, LA664Load]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVLDREPL128, [LA664LoadAGU, LA664Load]>;
  def : WriteRes<WriteVLDREPL256, [LA664LoadAGU, LA664Load]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVST128, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteVST256, [LA664StoreAGU, LA664Store]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVSTX128, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteVSTX256, [LA664StoreAGU, LA664Store]>;
}
let Latency = 5 in {
  def : WriteRes<WriteVSTELM128, [LA664StoreAGU, LA664Store]>;
  def : WriteRes<WriteVSTELM256, [LA664StoreAGU, LA664Store]>;
}

// Vector memory access operands
def : ReadAdvance<ReadVSTData128, 0>;
def : ReadAdvance<ReadVMemBase128, 0>;
def : ReadAdvance<ReadVMemOffset128, 0>;
def : ReadAdvance<ReadVSTData256, 0>;
def : ReadAdvance<ReadVMemBase256, 0>;
def : ReadAdvance<ReadVMemOffset256, 0>;

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