//===-- JIT.h - Abstract Execution Engine Interface -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file forces the JIT to link in on certain operating systems.
// (Windows).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_JIT_H
#define LLVM_EXECUTION_ENGINE_JIT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include <cstdlib>

extern "C" void LLVMLinkInJIT();

namespace {
  struct ForceJITLinking {
    ForceJITLinking() {
      // We must reference JIT in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      LLVMLinkInJIT();
    }
  } ForceJITLinking;
}

#endif
