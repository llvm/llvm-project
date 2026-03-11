//===- NVVMIntrinsicDiagnostics.h - NVVM diagnostic provider ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares linkNVVMIntrinsicDiagnostics(), which must be called from
// LLVMInitializeNVPTXTarget() to ensure the NVVM diagnostics provider object
// file is linked in when building with static libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_NVVMINTRINSICDIAGNOSTICS_H
#define LLVM_IR_NVVMINTRINSICDIAGNOSTICS_H

#include "llvm/Support/Compiler.h"

namespace llvm {

/// Force-links the NVVM intrinsic diagnostics provider when building with
/// static libraries.  Call from \c LLVMInitializeNVPTXTarget.
LLVM_ABI void linkNVVMIntrinsicDiagnostics();

} // namespace llvm

#endif // LLVM_IR_NVVMINTRINSICDIAGNOSTICS_H
