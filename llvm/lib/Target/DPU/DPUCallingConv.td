//===-- DPUCallingConv.td - Calling Conventions BPF --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the DPU architecture.
//
//===----------------------------------------------------------------------===//

// DPU 32-bit C return-value convention.
def RetCC_DPU : CallingConv<[
  CCIfType<[i32], CCAssignToReg<[RVAL, RVALHI]>>,
  CCIfType<[i64], CCAssignToReg<[RDVAL]>>
]>;

// DPU C Calling convention.
def CC_DPU : CallingConv<[
  // Promote every type of argument to longword
  CCIfType<[ i8, i16 ], CCPromoteToType<i32>>,

  // All arguments get passed in integer registers if there is space.
  CCIfType<[i32, f32], CCAssignToReg<[ R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 ]>>,

  // Support for 64 bits data: use the double registers:
  CCIfType<[i64, f64], CCAssignToReg<[ D0, D2, D4, D6, D8, D10, D12, D14 ]>>,

  // Could be assigned to the stack in 4-byte aligned units, but unsupported
  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def CSR_NoRegs : CalleeSavedRegs<(add)>;

def CC_DPU_VarArg : CallingConv<[
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,

  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;
