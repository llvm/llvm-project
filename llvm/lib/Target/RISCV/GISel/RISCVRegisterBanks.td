//=-- RISCVRegisterBank.td - Describe the RISC-V Banks -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// General Purpose Registers: X.
def GPRRegBank : RegisterBank<"GPRB", [GPR]>;

/// Floating Point Registers: F.
def FPRRegBank : RegisterBank<"FPRB", [FPR64]>;
