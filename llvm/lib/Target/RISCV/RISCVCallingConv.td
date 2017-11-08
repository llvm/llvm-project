//===-- RISCVCallingConv.td - Calling Conventions RISCV ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the RISCV architecture.
//
//===----------------------------------------------------------------------===//

// RISCV 32-bit C return-value convention.
def RetCC_RISCV32 : CallingConv<[CCIfType<[i32], CCAssignToReg<[X10, X11]>>]>;

// RISCV 32-bit C Calling convention.
def CC_RISCV32 : CallingConv<[
  // Promote i8/i16 args to i32
  CCIfType<[ i8, i16 ], CCPromoteToType<i32>>,

  // All arguments get passed in integer registers if there is space.
  CCIfType<[i32], CCAssignToReg<[ X10, X11, X12, X13, X14, X15, X16, X17]>>,

  // Could be assigned to the stack in 8-byte aligned units, but unsupported
  CCAssignToStack<8, 8>
]>;

def CSR : CalleeSavedRegs<(add X1, X3, X4, X8, X9, (sequence "X%u", 18, 27))>;

// Needed for implementation of RISCVRegisterInfo::getNoPreservedMask()
def CSR_NoRegs : CalleeSavedRegs<(add)>;
