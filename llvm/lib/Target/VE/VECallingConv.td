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

// Callee-saved registers
def CSR : CalleeSavedRegs<(add (sequence "SX%u", 18, 33))>;
def CSR_NoRegs : CalleeSavedRegs<(add)>;
