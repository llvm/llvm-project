//===-- P2CallingConv.td - Calling Conventions for P2 --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for P2 architecture.
//===----------------------------------------------------------------------===//

def CC_P2 : CallingConv<[
    // byvals are passed via the stack
    CCIfByVal<CCPassByVal<1, 1>>,
    // Promote i1/i8/i16 arguments to i32
    CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,
    // The first 4 arguments are passed in registers
    CCIfType<[i32], CCAssignToReg<[R0, R1, R2, R3]>>,
    // Fall-back, and use the stack, stack slot is 4 bytes wide and byte aligned.
    CCIfType<[i32], CCAssignToStack<4, 1>>,
    // 64 bit arguments are passed on the stack
    CCIfType<[i64], CCAssignToStack<8, 1>>
]>;

// Calling convention for variadic functions.
def CC_P2_Vararg : CallingConv <[
    // variable args are always passed through the stack with an alignment of 1.
    CCAssignToStack<4, 1>
]>;

def RetCC_P2 : CallingConv <[
    CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,
    // i32 are returned in R31 and R30 (two registers to 64 bit numbers if needed)
    CCIfType<[i32], CCAssignToReg<[R31, R30]>>,

    CCIfByVal<CCPassByVal<1, 1>>
]>;

def CSR : CalleeSavedRegs<(add (sequence "R%u", 0, 29))> {}