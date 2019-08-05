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
  CCIfType<[ i8, i16 ], CCPromoteToType<i32>>,
  CCIfType<[i32], CCAssignToReg<[R0]>>,
  CCIfType<[i64], CCAssignToReg<[D0]>>
]>;

// DPU C Calling convention.
def CC_DPU : CallingConv<[
  // Promote every type of argument to longword
  CCIfType<[ i8, i16 ], CCPromoteToType<i32>>,

  // All arguments get passed in integer registers if there is space.
  CCIfType<[i32, f32], CCAssignToReg<[ R0, R1, R2, R3, R4, R5, R6, R7 ]>>,

  // Support for 64 bits data: use the double registers:
  CCIfType<[i64, f64], CCAssignToReg<[ D0, D2, D4, D6 ]>>,

  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def CC_DPU_VarArg : CallingConv<[
  CCIfType<[i8, i16], CCPromoteToType<i32>>,

  CCIfType<[i32, f32], CCAssignToStack<4, 4>>,

  CCIfType<[i64, f64], CCAssignToStack<8, 8>>
]>;

def CSR : CalleeSavedRegs<(add D14, D16, D18, D20)>;
