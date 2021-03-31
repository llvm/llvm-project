//===-- RISCVSchedule.td - RISCV Scheduling Definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Define scheduler resources associated with def operands.
def WriteIALU       : SchedWrite;    // 32 or 64-bit integer ALU operations
def WriteIALU32     : SchedWrite;    // 32-bit integer ALU operations on RV64I
def WriteShiftImm   : SchedWrite;    // 32 or 64-bit shift by immediate operations
def WriteShiftImm32 : SchedWrite;    // 32-bit shift by immediate operations on RV64Ix
def WriteShiftReg   : SchedWrite;    // 32 or 64-bit shift by immediate operations
def WriteShiftReg32 : SchedWrite;    // 32-bit shift by immediate operations on RV64Ix
def WriteIDiv       : SchedWrite;    // 32-bit or 64-bit divide and remainder
def WriteIDiv32     : SchedWrite;    // 32-bit divide and remainder on RV64I
def WriteIMul       : SchedWrite;    // 32-bit or 64-bit multiply
def WriteIMul32     : SchedWrite;    // 32-bit multiply on RV64I
def WriteJmp        : SchedWrite;    // Jump
def WriteJal        : SchedWrite;    // Jump and link
def WriteJalr       : SchedWrite;    // Jump and link register
def WriteJmpReg     : SchedWrite;    // Jump register
def WriteNop        : SchedWrite;
def WriteLDB        : SchedWrite;    // Load byte
def WriteLDH        : SchedWrite;    // Load half-word
def WriteLDW        : SchedWrite;    // Load word
def WriteLDWU       : SchedWrite;    // Load word unsigned
def WriteLDD        : SchedWrite;    // Load double-word
def WriteCSR        : SchedWrite;    // CSR instructions
def WriteSTB        : SchedWrite;    // Store byte
def WriteSTH        : SchedWrite;    // Store half-word
def WriteSTW        : SchedWrite;    // Store word
def WriteSTD        : SchedWrite;    // Store double-word
def WriteAtomicW    : SchedWrite;    //Atomic memory operation word size
def WriteAtomicD    : SchedWrite;    //Atomic memory operation double word size
def WriteAtomicLDW  : SchedWrite;    // Atomic load word
def WriteAtomicLDD  : SchedWrite;    // Atomic load double word
def WriteAtomicSTW  : SchedWrite;    // Atomic store word
def WriteAtomicSTD  : SchedWrite;    // Atomic store double word
def WriteFALU16     : SchedWrite;    // FP 16-bit computation
def WriteFALU32     : SchedWrite;    // FP 32-bit computation
def WriteFALU64     : SchedWrite;    // FP 64-bit computation
def WriteFMul16     : SchedWrite;    // 16-bit floating point multiply
def WriteFMA16      : SchedWrite;    // 16-bit floating point fused multiply-add
def WriteFMul32     : SchedWrite;    // 32-bit floating point multiply
def WriteFMA32      : SchedWrite;    // 32-bit floating point fused multiply-add
def WriteFMul64     : SchedWrite;    // 64-bit floating point multiply
def WriteFMA64      : SchedWrite;    // 64-bit floating point fused multiply-add
def WriteFDiv16     : SchedWrite;    // 16-bit floating point divide
def WriteFDiv32     : SchedWrite;    // 32-bit floating point divide
def WriteFDiv64     : SchedWrite;    // 64-bit floating point divide
def WriteFSqrt16    : SchedWrite;    // 16-bit floating point sqrt
def WriteFSqrt32    : SchedWrite;    // 32-bit floating point sqrt
def WriteFSqrt64    : SchedWrite;    // 64-bit floating point sqrt

// Integer to float conversions
def WriteFCvtI32ToF16  : SchedWrite;
def WriteFCvtI32ToF32  : SchedWrite;
def WriteFCvtI32ToF64  : SchedWrite;
def WriteFCvtI64ToF16  : SchedWrite;    // RV64I only
def WriteFCvtI64ToF32  : SchedWrite;    // RV64I only
def WriteFCvtI64ToF64  : SchedWrite;    // RV64I only

//Float to integer conversions
def WriteFCvtF16ToI32  : SchedWrite;
def WriteFCvtF16ToI64  : SchedWrite;    // RV64I only
def WriteFCvtF32ToI32  : SchedWrite;
def WriteFCvtF32ToI64  : SchedWrite;    // RV64I only
def WriteFCvtF64ToI32  : SchedWrite;
def WriteFCvtF64ToI64  : SchedWrite;    // RV64I only

// Float to float conversions
def WriteFCvtF32ToF64  : SchedWrite;
def WriteFCvtF64ToF32  : SchedWrite;
def WriteFCvtF16ToF32  : SchedWrite;
def WriteFCvtF32ToF16  : SchedWrite;
def WriteFCvtF16ToF64  : SchedWrite;
def WriteFCvtF64ToF16  : SchedWrite;

def WriteFClass16   : SchedWrite;    // 16-bit floating point classify
def WriteFClass32   : SchedWrite;    // 32-bit floating point classify
def WriteFClass64   : SchedWrite;    // 64-bit floating point classify
def WriteFCmp16     : SchedWrite;    // 16-bit floating point compare
def WriteFCmp32     : SchedWrite;    // 32-bit floating point compare
def WriteFCmp64     : SchedWrite;    // 64-bit floating point compare
def WriteFSGNJ16    : SchedWrite;    // 16-bit floating point sign-injection
def WriteFSGNJ32    : SchedWrite;    // 32-bit floating point sign-injection
def WriteFSGNJ64    : SchedWrite;    // 64-bit floating point sign-injection
def WriteFMinMax16  : SchedWrite;    // 16-bit floating point min or max
def WriteFMinMax32  : SchedWrite;    // 32-bit floating point min or max
def WriteFMinMax64  : SchedWrite;    // 64-bit floating point min or max

def WriteFMovF16ToI16     : SchedWrite;
def WriteFMovI16ToF16     : SchedWrite;
def WriteFMovF32ToI32     : SchedWrite;
def WriteFMovI32ToF32     : SchedWrite;
def WriteFMovF64ToI64     : SchedWrite;    // RV64I only
def WriteFMovI64ToF64     : SchedWrite;    // RV64I only

def WriteFLD16        : SchedWrite;    // Floating point sp load
def WriteFLD32        : SchedWrite;    // Floating point sp load
def WriteFLD64        : SchedWrite;    // Floating point dp load
def WriteFST16        : SchedWrite;    // Floating point sp store
def WriteFST32        : SchedWrite;    // Floating point sp store
def WriteFST64        : SchedWrite;    // Floating point dp store

/// Define scheduler resources associated with use operands.
def ReadJmp         : SchedRead;
def ReadJalr        : SchedRead;
def ReadCSR         : SchedRead;
def ReadMemBase     : SchedRead;
def ReadFMemBase    : SchedRead;
def ReadStoreData   : SchedRead;
def ReadIALU        : SchedRead;
def ReadIALU32      : SchedRead;    // 32-bit integer ALU operations on RV64I
def ReadShiftImm    : SchedRead;
def ReadShiftImm32  : SchedRead;    // 32-bit shift by immediate operations on RV64Ix
def ReadShiftReg    : SchedRead;
def ReadShiftReg32  : SchedRead;    // 32-bit shift by register operations on RV64Ix
def ReadIDiv        : SchedRead;
def ReadIDiv32      : SchedRead;
def ReadIMul        : SchedRead;
def ReadIMul32      : SchedRead;
def ReadAtomicWA    : SchedRead;
def ReadAtomicWD    : SchedRead;
def ReadAtomicDA    : SchedRead;
def ReadAtomicDD    : SchedRead;
def ReadAtomicLDW   : SchedRead;    // Atomic load word
def ReadAtomicLDD   : SchedRead;    // Atomic load double word
def ReadAtomicSTW   : SchedRead;    // Atomic store word
def ReadAtomicSTD   : SchedRead;    // Atomic store double word
def ReadFALU16      : SchedRead;    // FP 16-bit computation
def ReadFALU32      : SchedRead;    // FP 32-bit computation
def ReadFALU64      : SchedRead;    // FP 64-bit computation
def ReadFMul16      : SchedRead;    // 16-bit floating point multiply
def ReadFMA16       : SchedRead;    // 16-bit floating point fused multiply-add
def ReadFMul32      : SchedRead;    // 32-bit floating point multiply
def ReadFMA32       : SchedRead;    // 32-bit floating point fused multiply-add
def ReadFMul64      : SchedRead;    // 64-bit floating point multiply
def ReadFMA64       : SchedRead;    // 64-bit floating point fused multiply-add
def ReadFDiv16      : SchedRead;    // 16-bit floating point divide
def ReadFDiv32      : SchedRead;    // 32-bit floating point divide
def ReadFDiv64      : SchedRead;    // 64-bit floating point divide
def ReadFSqrt16     : SchedRead;    // 16-bit floating point sqrt
def ReadFSqrt32     : SchedRead;    // 32-bit floating point sqrt
def ReadFSqrt64     : SchedRead;    // 64-bit floating point sqrt
def ReadFCmp16      : SchedRead;
def ReadFCmp32      : SchedRead;
def ReadFCmp64      : SchedRead;
def ReadFSGNJ16     : SchedRead;
def ReadFSGNJ32     : SchedRead;
def ReadFSGNJ64     : SchedRead;
def ReadFMinMax16   : SchedRead;
def ReadFMinMax32   : SchedRead;
def ReadFMinMax64   : SchedRead;
def ReadFCvtF16ToI32     : SchedRead;
def ReadFCvtF16ToI64     : SchedRead;
def ReadFCvtF32ToI32     : SchedRead;
def ReadFCvtF32ToI64     : SchedRead;
def ReadFCvtF64ToI32     : SchedRead;
def ReadFCvtF64ToI64     : SchedRead;
def ReadFCvtI32ToF16     : SchedRead;
def ReadFCvtI32ToF32     : SchedRead;
def ReadFCvtI32ToF64     : SchedRead;
def ReadFCvtI64ToF16     : SchedRead;
def ReadFCvtI64ToF32     : SchedRead;
def ReadFCvtI64ToF64     : SchedRead;
def ReadFMovF16ToI16     : SchedRead;
def ReadFMovI16ToF16     : SchedRead;
def ReadFMovF32ToI32     : SchedRead;
def ReadFMovI32ToF32     : SchedRead;
def ReadFMovF64ToI64     : SchedRead;
def ReadFMovI64ToF64     : SchedRead;
def ReadFCvtF32ToF64     : SchedRead;
def ReadFCvtF64ToF32     : SchedRead;
def ReadFCvtF16ToF32     : SchedRead;
def ReadFCvtF32ToF16     : SchedRead;
def ReadFCvtF16ToF64     : SchedRead;
def ReadFCvtF64ToF16     : SchedRead;
def ReadFClass16         : SchedRead;
def ReadFClass32         : SchedRead;
def ReadFClass64         : SchedRead;

// Include the scheduler resources for other instruction extensions.
include "RISCVScheduleB.td"
