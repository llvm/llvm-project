//===---- OrcMCJITReplacement.h - Orc-based MCJIT replacement ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forces OrcMCJITReplacement to link in on certain operating systems.
// (Windows).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORCMCJITREPLACEMENT_H
#define LLVM_EXECUTIONENGINE_ORCMCJITREPLACEMENT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include <cstdlib>

extern "C" void LLVMLinkInOrcMCJITReplacement();

namespace {
  struct ForceOrcMCJITReplacementLinking {
    ForceOrcMCJITReplacementLinking() {
      // We must reference OrcMCJITReplacement in such a way that compilers will
      // not delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough to know
      // that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      LLVMLinkInOrcMCJITReplacement();
    }
  } ForceOrcMCJITReplacementLinking;
}

#endif
