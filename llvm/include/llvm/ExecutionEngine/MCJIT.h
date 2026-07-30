//===-- MCJIT.h - MC-Based Just-In-Time Execution Engine --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forces the MCJIT to link in on certain operating systems.
// (Windows).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_MCJIT_H
#define LLVM_EXECUTIONENGINE_MCJIT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/AlwaysTrue.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ABI void LLVMLinkInMCJIT();

namespace {
  struct ForceMCJITLinking {
    ForceMCJITLinking() {
      // We must reference MCJIT in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization, yet
      // is effectively a NO-OP. This is so that globals in the translation
      // units where these functions are defined are forced to be initialized,
      // populating various registries.
      if (llvm::getNonFoldableAlwaysTrue())
        return;

      LLVMLinkInMCJIT();
    }
  } ForceMCJITLinking;
}

#endif
