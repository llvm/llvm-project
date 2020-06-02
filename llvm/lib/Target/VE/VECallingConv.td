//===-- VECallingConv.td - Calling Conventions VE ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the VE architectures.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Aurora VE
//===----------------------------------------------------------------------===//
def CC_VE_C_Stack: CallingConv<[
  // float --> need special handling like below.
  //    0      4
  //    +------+------+
  //    | empty| float|
  //    +------+------+
  CCIfType<[f32], CCCustom<"allocateFloat">>,

  // All of the rest are assigned to the stack in 8-byte aligned units.
  CCAssignToStack<0, 8>
]>;

def CC_VE : CallingConv<[
  // All arguments get passed in generic registers if there is space.

  // Promote i1/i8/i16 arguments to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // bool, char, int, enum, long --> generic integer 32 bit registers
  CCIfType<[i32], CCAssignToRegWithShadow<
    [SW0, SW1, SW2, SW3, SW4, SW5, SW6, SW7],
    [SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,

  // float --> generic floating point 32 bit registers
  CCIfType<[f32], CCAssignToRegWithShadow<
    [SF0, SF1, SF2, SF3, SF4, SF5, SF6, SF7],
    [SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,

  // long long/double --> generic 64 bit registers
  CCIfType<[i64, f64],
           CCAssignToReg<[SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,

  // Alternatively, they are assigned to the stack in 8-byte aligned units.
  CCDelegateTo<CC_VE_C_Stack>
]>;

// All arguments get passed in stack for varargs function or non-prototyped
// function.
def CC_VE2 : CallingConv<[
  // float --> need special handling like below.
  //    0      4
  //    +------+------+
  //    | empty| float|
  //    +------+------+
  CCIfType<[f32], CCCustom<"allocateFloat">>,

  CCAssignToStack<0, 8>
]>;

def RetCC_VE : CallingConv<[
  // Promote i1/i8/i16 arguments to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // bool, char, int, enum, long --> generic integer 32 bit registers
  CCIfType<[i32], CCAssignToRegWithShadow<
    [SW0, SW1, SW2, SW3, SW4, SW5, SW6, SW7],
    [SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,

  // float --> generic floating point 32 bit registers
  CCIfType<[f32], CCAssignToRegWithShadow<
    [SF0, SF1, SF2, SF3, SF4, SF5, SF6, SF7],
    [SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,

  // long long/double --> generic 64 bit registers
  CCIfType<[i64, f64],
           CCAssignToReg<[SX0, SX1, SX2, SX3, SX4, SX5, SX6, SX7]>>,
]>;

// Callee-saved registers
def CSR : CalleeSavedRegs<(add (sequence "SX%u", 18, 33))>;
def CSR_NoRegs : CalleeSavedRegs<(add)>;

// PreserveAll (clobbers s62,s63) - used for ve_grow_stack
def CSR_preserve_all : CalleeSavedRegs<(add (sequence "SX%u", 0, 61))>;
