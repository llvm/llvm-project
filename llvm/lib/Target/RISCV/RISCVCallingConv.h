//===-- RISCVCallingConv.h - RISC-V Custom CC Routines ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the custom routines for the RISC-V Calling Convention.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {

/// This is used for assigining arguments to locations when making calls.
CCAssignFn CC_RISCV;

/// This is used for assigning return values to locations when making calls.
CCAssignFn RetCC_RISCV;

namespace RISCV {

ArrayRef<MCPhysReg> getArgGPRs(const RISCVABI::ABI ABI);

} // end namespace RISCV

} // end namespace llvm
