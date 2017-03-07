//=- AArch64SchedThunderX2T99.td - Cavium ThunderX T99 Scheduling ---*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for Cavium ThunderX2T99
// processors.
// Based on Broadcom Vulcan.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 2. Pipeline Description.

def ThunderX2T99Model : SchedMachineModel {
  let IssueWidth            =   4; // 4 micro-ops dispatched at a time.
  let MicroOpBufferSize     = 180; // 180 entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  12; // Extra cycles for mispredicted branch.
  // Determined via a mix of micro-arch details and experimentation.
  let LoopMicroOpBufferSize =  32;
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;
}

// Define the issue ports.

// Port 0: ALU, FP/SIMD.
def THX2T99P0 : ProcResource<1>;

// Port 1: ALU, FP/SIMD, integer mul/div.
def THX2T99P1 : ProcResource<1>;

// Port 2: ALU, Branch.
def THX2T99P2 : ProcResource<1>;

// Port 3: Store data.
def THX2T99P3 : ProcResource<1>;

// Port 4: Load/store.
def THX2T99P4 : ProcResource<1>;

// Port 5: Load/store.
def THX2T99P5 : ProcResource<1>;

let SchedModel = ThunderX2T99Model in {

// Define groups for the functional units on each issue port.  Each group
// created will be used by a WriteRes later on.
//
// NOTE: Some groups only contain one member.  This is a way to create names for
// the various functional units that share a single issue port.  For example,
// "THX2T99I1" for ALU ops on port 1 and "THX2T99F1" for FP ops on port 1.

// Integer divide and multiply micro-ops only on port 1.
def THX2T99I1 : ProcResGroup<[THX2T99P1]>;

// Branch micro-ops only on port 2.
def THX2T99I2 : ProcResGroup<[THX2T99P2]>;

// ALU micro-ops on ports 0, 1, and 2.
def THX2T99I012 : ProcResGroup<[THX2T99P0, THX2T99P1, THX2T99P2]>;

// Crypto FP/SIMD micro-ops only on port 1.
def THX2T99F1 : ProcResGroup<[THX2T99P1]>;

// FP/SIMD micro-ops on ports 0 and 1.
def THX2T99F01 : ProcResGroup<[THX2T99P0, THX2T99P1]>;

// Store data micro-ops only on port 3.
def THX2T99SD : ProcResGroup<[THX2T99P3]>;

// Load/store micro-ops on ports 4 and 5.
def THX2T99LS01 : ProcResGroup<[THX2T99P4, THX2T99P5]>;

// 60 entry unified scheduler.
def THX2T99Any : ProcResGroup<[THX2T99P0, THX2T99P1, THX2T99P2,
                              THX2T99P3, THX2T99P4, THX2T99P5]> {
  let BufferSize=60;
}

// Define commonly used write types for InstRW specializations.
// All definitions follow the format: THX2T99Write_<NumCycles>Cyc_<Resources>.

// 3 cycles on I1.
def THX2T99Write_3Cyc_I1 : SchedWriteRes<[THX2T99I1]> { let Latency = 3; }

// 4 cycles on I1.
def THX2T99Write_4Cyc_I1 : SchedWriteRes<[THX2T99I1]> { let Latency = 4; }

// 1 cycle on I0, I1, or I2.
def THX2T99Write_1Cyc_I012 : SchedWriteRes<[THX2T99I012]> { let Latency = 1; }

// 5 cycles on F1.
def THX2T99Write_5Cyc_F1 : SchedWriteRes<[THX2T99F1]> { let Latency = 5; }

// 7 cycles on F1.
def THX2T99Write_7Cyc_F1 : SchedWriteRes<[THX2T99F1]> { let Latency = 7; }

// 4 cycles on F0 or F1.
def THX2T99Write_4Cyc_F01 : SchedWriteRes<[THX2T99F01]> { let Latency = 4; }

// 5 cycles on F0 or F1.
def THX2T99Write_5Cyc_F01 : SchedWriteRes<[THX2T99F01]> { let Latency = 5; }

// 6 cycles on F0 or F1.
def THX2T99Write_6Cyc_F01 : SchedWriteRes<[THX2T99F01]> { let Latency = 6; }

// 7 cycles on F0 or F1.
def THX2T99Write_7Cyc_F01 : SchedWriteRes<[THX2T99F01]> { let Latency = 7; }

// 8 cycles on F0 or F1.
def THX2T99Write_8Cyc_F01 : SchedWriteRes<[THX2T99F01]> { let Latency = 8; }

// 16 cycles on F0 or F1.
def THX2T99Write_16Cyc_F01 : SchedWriteRes<[THX2T99F01]> {
  let Latency = 16;
  let ResourceCycles = [8];
}

// 23 cycles on F0 or F1.
def THX2T99Write_23Cyc_F01 : SchedWriteRes<[THX2T99F01]> {
  let Latency = 23;
  let ResourceCycles = [11];
}

// 1 cycles on LS0 or LS1.
def THX2T99Write_1Cyc_LS01 : SchedWriteRes<[THX2T99LS01]> { let Latency = 1; }

// 4 cycles on LS0 or LS1.
def THX2T99Write_4Cyc_LS01 : SchedWriteRes<[THX2T99LS01]> { let Latency = 4; }

// 5 cycles on LS0 or LS1.
def THX2T99Write_5Cyc_LS01 : SchedWriteRes<[THX2T99LS01]> { let Latency = 5; }

// 6 cycles on LS0 or LS1.
def THX2T99Write_6Cyc_LS01 : SchedWriteRes<[THX2T99LS01]> { let Latency = 6; }

// 5 cycles on LS0 or LS1 and I0, I1, or I2.
def THX2T99Write_5Cyc_LS01_I012 : SchedWriteRes<[THX2T99LS01, THX2T99I012]> {
  let Latency = 5;
  let NumMicroOps = 2;
}

// 5 cycles on LS0 or LS1 and 2 of I0, I1, or I2.
def THX2T99Write_6Cyc_LS01_I012_I012 : 
  SchedWriteRes<[THX2T99LS01, THX2T99I012, THX2T99I012]> {
  let Latency = 6;
  let NumMicroOps = 3;
}

// 1 cycles on LS0 or LS1 and F0 or F1.
def THX2T99Write_1Cyc_LS01_F01 : SchedWriteRes<[THX2T99LS01, THX2T99F01]> {
  let Latency = 1;
  let NumMicroOps = 2;
}

// 5 cycles on LS0 or LS1 and F0 or F1.
def THX2T99Write_5Cyc_LS01_F01 : SchedWriteRes<[THX2T99LS01, THX2T99F01]> {
  let Latency = 5;
  let NumMicroOps = 2;
}

// 6 cycles on LS0 or LS1 and F0 or F1.
def THX2T99Write_6Cyc_LS01_F01 : SchedWriteRes<[THX2T99LS01, THX2T99F01]> {
  let Latency = 6;
  let NumMicroOps = 2;
}

// 7 cycles on LS0 or LS1 and F0 or F1.
def THX2T99Write_7Cyc_LS01_F01 : SchedWriteRes<[THX2T99LS01, THX2T99F01]> {
  let Latency = 7;
  let NumMicroOps = 2;
}

// 8 cycles on LS0 or LS1 and F0 or F1.
def THX2T99Write_8Cyc_LS01_F01 : SchedWriteRes<[THX2T99LS01, THX2T99F01]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

// Define commonly used read types.

// No forwarding is provided for these types.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD,     0>;

}


//===----------------------------------------------------------------------===//
// 3. Instruction Tables.

let SchedModel = ThunderX2T99Model in {

//---
// 3.1 Branch Instructions
//---

// Branch, immed
// Branch and link, immed
// Compare and branch
def : WriteRes<WriteBr,      [THX2T99I2]> { let Latency = 1; }

def : WriteRes<WriteSys,     []> { let Latency = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }

def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }

// Branch, register
// Branch and link, register != LR
// Branch and link, register = LR
def : WriteRes<WriteBrReg,   [THX2T99I2]> { let Latency = 1; }

//---
// 3.2 Arithmetic and Logical Instructions
// 3.3 Move and Shift Instructions
//---

// ALU, basic
// Conditional compare
// Conditional select
// Address generation
def : WriteRes<WriteI,       [THX2T99I012]> { let Latency = 1; }
def : InstRW<[WriteI], (instrs COPY)>;

// ALU, extend and/or shift
def : WriteRes<WriteISReg,   [THX2T99I012]> {
  let Latency = 2;
  let ResourceCycles = [2];
}

def : WriteRes<WriteIEReg,   [THX2T99I012]> {
  let Latency = 2;
  let ResourceCycles = [2];
}

// Move immed
def : WriteRes<WriteImm,     [THX2T99I012]> { let Latency = 1; }

// Variable shift
def : WriteRes<WriteIS,      [THX2T99I012]> { let Latency = 1; }

//---
// 3.4 Divide and Multiply Instructions
//---

// Divide, W-form
// Latency range of 13-23.  Take the average.
def : WriteRes<WriteID32,    [THX2T99I1]> {
  let Latency = 18;
  let ResourceCycles = [18];
}

// Divide, X-form
// Latency range of 13-39.  Take the average.
def : WriteRes<WriteID64,    [THX2T99I1]> {
  let Latency = 26;
  let ResourceCycles = [26];
}

// Multiply accumulate, W-form
def : WriteRes<WriteIM32,    [THX2T99I012]> { let Latency = 5; }

// Multiply accumulate, X-form
def : WriteRes<WriteIM64,    [THX2T99I012]> { let Latency = 5; }

// Bitfield extract, two reg
def : WriteRes<WriteExtr,    [THX2T99I012]> { let Latency = 1; }

// Bitfield move, basic
// Bitfield move, insert
// NOTE: Handled by WriteIS.

// Count leading
def : InstRW<[THX2T99Write_3Cyc_I1], (instregex "^CLS(W|X)r$",
                                               "^CLZ(W|X)r$")>;

// Reverse bits/bytes
// NOTE: Handled by WriteI.

//---
// 3.6 Load Instructions 
// 3.10 FP Load Instructions
//---

// Load register, literal
// Load register, unscaled immed
// Load register, immed unprivileged
// Load register, unsigned immed
def : WriteRes<WriteLD,      [THX2T99LS01]> { let Latency = 4; }

// Load register, immed post-index
// NOTE: Handled by WriteLD, WriteI.
// Load register, immed pre-index
// NOTE: Handled by WriteLD, WriteAdr.
def : WriteRes<WriteAdr,     [THX2T99I012]> { let Latency = 1; }

// Load register offset, basic
// Load register, register offset, scale by 4/8
// Load register, register offset, scale by 2
// Load register offset, extend
// Load register, register offset, extend, scale by 4/8
// Load register, register offset, extend, scale by 2
def THX2T99WriteLDIdx : SchedWriteVariant<[
  SchedVar<ScaledIdxPred, [THX2T99Write_6Cyc_LS01_I012_I012]>,
  SchedVar<NoSchedPred,   [THX2T99Write_5Cyc_LS01_I012]>]>;
def : SchedAlias<WriteLDIdx, THX2T99WriteLDIdx>;

def THX2T99ReadAdrBase : SchedReadVariant<[
  SchedVar<ScaledIdxPred, [ReadDefault]>,
  SchedVar<NoSchedPred,   [ReadDefault]>]>;
def : SchedAlias<ReadAdrBase, THX2T99ReadAdrBase>;

// Load pair, immed offset, normal
// Load pair, immed offset, signed words, base != SP
// Load pair, immed offset signed words, base = SP
// LDP only breaks into *one* LS micro-op.  Thus
// the resources are handling by WriteLD.
def : WriteRes<WriteLDHi,    []> {
  let Latency = 5;
}

// Load pair, immed pre-index, normal
// Load pair, immed pre-index, signed words
// Load pair, immed post-index, normal
// Load pair, immed post-index, signed words
// NOTE: Handled by WriteLD, WriteLDHi, WriteAdr.

//--
// 3.7 Store Instructions 
// 3.11 FP Store Instructions
//--

// Store register, unscaled immed
// Store register, immed unprivileged
// Store register, unsigned immed
def : WriteRes<WriteST,      [THX2T99LS01, THX2T99SD]> {
  let Latency = 1;
  let NumMicroOps = 2;
}

// Store register, immed post-index
// NOTE: Handled by WriteAdr, WriteST, ReadAdrBase

// Store register, immed pre-index
// NOTE: Handled by WriteAdr, WriteST

// Store register, register offset, basic
// Store register, register offset, scaled by 4/8
// Store register, register offset, scaled by 2
// Store register, register offset, extend
// Store register, register offset, extend, scale by 4/8
// Store register, register offset, extend, scale by 1
def : WriteRes<WriteSTIdx, [THX2T99LS01, THX2T99SD, THX2T99I012]> {
  let Latency = 1;
  let NumMicroOps = 3;
}

// Store pair, immed offset, W-form
// Store pair, immed offset, X-form
def : WriteRes<WriteSTP,     [THX2T99LS01, THX2T99SD]> {
  let Latency = 1;
  let NumMicroOps = 2;
}

// Store pair, immed post-index, W-form
// Store pair, immed post-index, X-form
// Store pair, immed pre-index, W-form
// Store pair, immed pre-index, X-form
// NOTE: Handled by WriteAdr, WriteSTP.

//---
// 3.8 FP Data Processing Instructions
//---

// FP absolute value
// FP min/max
// FP negate
def : WriteRes<WriteF,       [THX2T99F01]> { let Latency = 5; }

// FP arithmetic
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FADD", "^FSUB")>;

// FP compare
def : WriteRes<WriteFCmp,    [THX2T99F01]> { let Latency = 5; }

// FP divide, S-form
// FP square root, S-form
def : WriteRes<WriteFDiv,    [THX2T99F01]> {
  let Latency = 16;
  let ResourceCycles = [8];
}

// FP divide, D-form
// FP square root, D-form
def : InstRW<[THX2T99Write_23Cyc_F01], (instrs FDIVDrr, FSQRTDr)>;

// FP multiply
// FP multiply accumulate
def : WriteRes<WriteFMul, [THX2T99F01]> { let Latency = 6; }

// FP round to integral
def : InstRW<[THX2T99Write_7Cyc_F01],
            (instregex "^FRINT(A|I|M|N|P|X|Z)(Sr|Dr)")>;

// FP select
def : InstRW<[THX2T99Write_4Cyc_F01], (instregex "^FCSEL")>;

//---
// 3.9 FP Miscellaneous Instructions
//---

// FP convert, from vec to vec reg
// FP convert, from gen to vec reg
// FP convert, from vec to gen reg
def : WriteRes<WriteFCvt, [THX2T99F01]> { let Latency = 7; }

// FP move, immed
// FP move, register
def : WriteRes<WriteFImm, [THX2T99F01]> { let Latency = 4; }

// FP transfer, from gen to vec reg
// FP transfer, from vec to gen reg
def : WriteRes<WriteFCopy, [THX2T99F01]> { let Latency = 4; }
def : InstRW<[THX2T99Write_5Cyc_F01], (instrs FMOVXDHighr, FMOVDXHighr)>;

//---
// 3.12 ASIMD Integer Instructions
//---

// ASIMD absolute diff, D-form
// ASIMD absolute diff, Q-form
// ASIMD absolute diff accum, D-form
// ASIMD absolute diff accum, Q-form
// ASIMD absolute diff accum long
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD compare
// ASIMD logical (AND, BIC, EOR)
// ASIMD max/min, basic
// ASIMD max/min, reduce, 4H/4S
// ASIMD max/min, reduce, 8B/8H
// ASIMD max/min, reduce, 16B
// ASIMD multiply, D-form
// ASIMD multiply, Q-form
// ASIMD multiply accumulate long
// ASIMD multiply accumulate saturating long
// ASIMD multiply long
// ASIMD pairwise add and accumulate
// ASIMD shift accumulate
// ASIMD shift by immed, basic
// ASIMD shift by immed and insert, basic, D-form
// ASIMD shift by immed and insert, basic, Q-form
// ASIMD shift by immed, complex
// ASIMD shift by register, basic, D-form
// ASIMD shift by register, basic, Q-form
// ASIMD shift by register, complex, D-form
// ASIMD shift by register, complex, Q-form
def : WriteRes<WriteV, [THX2T99F01]> { let Latency = 7; }

// ASIMD arith, reduce, 4H/4S
// ASIMD arith, reduce, 8B/8H
// ASIMD arith, reduce, 16B
def : InstRW<[THX2T99Write_5Cyc_F01], 
            (instregex "^ADDVv", "^SADDLVv", "^UADDLVv")>;

// ASIMD logical (MOV, MVN, ORN, ORR)
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^ORRv", "^ORNv", "^NOTv")>;

// ASIMD polynomial (8x8) multiply long
def : InstRW<[THX2T99Write_5Cyc_F01], (instrs PMULLv8i8, PMULLv16i8)>;

//---
// 3.13 ASIMD Floating-point Instructions
//---

// ASIMD FP absolute value
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FABSv")>;

// ASIMD FP arith, normal, D-form
// ASIMD FP arith, normal, Q-form
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FABDv", "^FADDv", "^FSUBv")>;

// ASIMD FP arith,pairwise, D-form
// ASIMD FP arith, pairwise, Q-form
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FADDPv")>;

// ASIMD FP compare, D-form
// ASIMD FP compare, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FACGEv", "^FACGTv")>;
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FCMEQv", "^FCMGEv",
                                                "^FCMGTv", "^FCMLEv",
                                                "^FCMLTv")>;

// ASIMD FP convert, long
// ASIMD FP convert, narrow
// ASIMD FP convert, other, D-form
// ASIMD FP convert, other, Q-form
// NOTE: Handled by WriteV.

// ASIMD FP divide, D-form, F32
def : InstRW<[THX2T99Write_16Cyc_F01], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[THX2T99Write_16Cyc_F01], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[THX2T99Write_23Cyc_F01], (instrs FDIVv2f64)>;

// ASIMD FP max/min, normal, D-form
// ASIMD FP max/min, normal, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FMAXv", "^FMAXNMv",
                                                "^FMINv", "^FMINNMv")>;

// ASIMD FP max/min, pairwise, D-form
// ASIMD FP max/min, pairwise, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FMAXPv", "^FMAXNMPv",
                                                "^FMINPv", "^FMINNMPv")>;

// ASIMD FP max/min, reduce
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FMAXVv", "^FMAXNMVv",
                                                "^FMINVv", "^FMINNMVv")>;

// ASIMD FP multiply, D-form, FZ
// ASIMD FP multiply, D-form, no FZ
// ASIMD FP multiply, Q-form, FZ
// ASIMD FP multiply, Q-form, no FZ
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FMULv", "^FMULXv")>;

// ASIMD FP multiply accumulate, Dform, FZ
// ASIMD FP multiply accumulate, Dform, no FZ
// ASIMD FP multiply accumulate, Qform, FZ
// ASIMD FP multiply accumulate, Qform, no FZ
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FMLAv", "^FMLSv")>;

// ASIMD FP negate
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FNEGv")>;

// ASIMD FP round, D-form
// ASIMD FP round, Q-form
// NOTE: Handled by WriteV.

//--
// 3.14 ASIMD Miscellaneous Instructions
//--

// ASIMD bit reverse
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^RBITv")>;

// ASIMD bitwise insert, D-form
// ASIMD bitwise insert, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^BIFv", "^BITv", "^BSLv")>;

// ASIMD count, D-form
// ASIMD count, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^CLSv", "^CLZv", "^CNTv")>;

// ASIMD duplicate, gen reg
// ASIMD duplicate, element
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^DUPv")>;

// ASIMD extract
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^EXTv")>;

// ASIMD extract narrow
// ASIMD extract narrow, saturating
// NOTE: Handled by WriteV.

// ASIMD insert, element to element
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^INSv")>;

// ASIMD move, integer immed
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^MOVIv", "^MOVIDv")>;

// ASIMD move, FP immed
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^FMOVv")>;

// ASIMD reciprocal estimate, D-form
// ASIMD reciprocal estimate, Q-form
def : InstRW<[THX2T99Write_5Cyc_F01], 
            (instregex "^FRECPEv", "^FRECPXv", "^URECPEv",
                                   "^FRSQRTEv", "^URSQRTEv")>;

// ASIMD reciprocal step, D-form, FZ
// ASIMD reciprocal step, D-form, no FZ
// ASIMD reciprocal step, Q-form, FZ
// ASIMD reciprocal step, Q-form, no FZ
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^FRECPSv", "^FRSQRTSv")>;

// ASIMD reverse
def : InstRW<[THX2T99Write_5Cyc_F01], 
            (instregex "^REV16v", "^REV32v", "^REV64v")>;

// ASIMD table lookup, D-form
// ASIMD table lookup, Q-form
def : InstRW<[THX2T99Write_8Cyc_F01], (instregex "^TBLv", "^TBXv")>;

// ASIMD transfer, element to word or word
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^UMOVv")>;

// ASIMD transfer, element to gen reg
def : InstRW<[THX2T99Write_6Cyc_F01], (instregex "^SMOVv", "^UMOVv")>;

// ASIMD transfer gen reg to element
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^INSv")>;

// ASIMD transpose
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^TRN1v", "^TRN2v",
                                                "^UZP1v", "^UZP2v")>;

// ASIMD unzip/zip
def : InstRW<[THX2T99Write_5Cyc_F01], (instregex "^ZIP1v", "^ZIP2v")>;

//--
// 3.15 ASIMD Load Instructions 
//--

// ASIMD load, 1 element, multiple, 1 reg, D-form
// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[THX2T99Write_4Cyc_LS01], 
            (instregex "^LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_4Cyc_LS01, WriteAdr], 
            (instregex "^LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[THX2T99Write_4Cyc_LS01], 
            (instregex "^LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_4Cyc_LS01, WriteAdr], 
            (instregex "^LD1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[THX2T99Write_5Cyc_LS01], 
            (instregex "^LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01, WriteAdr], 
            (instregex "^LD1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[THX2T99Write_6Cyc_LS01], 
            (instregex "^LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_6Cyc_LS01, WriteAdr], 
            (instregex "^LD1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[THX2T99Write_5Cyc_LS01_F01], (instregex "^LD1i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[THX2T99Write_5Cyc_LS01_F01], 
            (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_5Cyc_LS01_F01], 
            (instregex "^LD2Twov(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD2Twov(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[THX2T99Write_5Cyc_LS01_F01], (instregex "^LD2i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[THX2T99Write_5Cyc_LS01_F01], 
            (instregex "^LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_5Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_8Cyc_LS01_F01], 
            (instregex "^LD3Threev(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_8Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD3Threev(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, one lone, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[THX2T99Write_7Cyc_LS01_F01], (instregex "^LD3i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_7Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[THX2T99Write_7Cyc_LS01_F01], 
            (instregex "^LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_7Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_8Cyc_LS01_F01], 
            (instregex "^LD4Fourv(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_8Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD4Fourv(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[THX2T99Write_6Cyc_LS01_F01], (instregex "^LD4i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_6Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[THX2T99Write_6Cyc_LS01_F01], 
            (instregex "^LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_6Cyc_LS01_F01, WriteAdr], 
            (instregex "^LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

//--
// 3.16 ASIMD Store Instructions
//--

// ASIMD store, 1 element, multiple, 1 reg, D-form
// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[THX2T99Write_1Cyc_LS01], 
            (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01, WriteAdr], 
            (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[THX2T99Write_1Cyc_LS01], 
            (instregex "^ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01, WriteAdr], 
            (instregex "^ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[THX2T99Write_1Cyc_LS01], 
            (instregex "^ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01, WriteAdr], 
            (instregex "^ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[THX2T99Write_1Cyc_LS01], 
            (instregex "^ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01, WriteAdr], 
            (instregex "^ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], 
            (instregex "^ST1i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], 
            (instregex "^ST2Twov(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST2Twov(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], 
            (instregex "^ST2i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], 
            (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], (instregex "^ST3i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
// ASIMD store, 4 element, multiple, Q-form, B/H/S
// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], 
            (instregex "^ST4Fourv(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST4Fourv(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H
// ASIMD store, 4 element, one lane, S
// ASIMD store, 4 element, one lane, D
def : InstRW<[THX2T99Write_1Cyc_LS01_F01], (instregex "^ST4i(8|16|32|64)$")>;
def : InstRW<[THX2T99Write_1Cyc_LS01_F01, WriteAdr], 
            (instregex "^ST4i(8|16|32|64)_POST$")>;

//--
// 3.17 Cryptography Extensions
//--

// Crypto AES ops
def : InstRW<[THX2T99Write_5Cyc_F1], (instregex "^AES")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[THX2T99Write_5Cyc_F1], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 xor ops
// Crypto SHA1 schedule acceleration ops
// Crypto SHA256 schedule acceleration op (1 u-op)
// Crypto SHA256 schedule acceleration op (2 u-ops)
// Crypto SHA256 hash acceleration ops
def : InstRW<[THX2T99Write_7Cyc_F1], (instregex "^SHA")>;

//--
// 3.18 CRC
//--

// CRC checksum ops
def : InstRW<[THX2T99Write_4Cyc_I1], (instregex "^CRC32")>;

} // SchedModel = ThunderX2T99Model
