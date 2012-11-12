//===- llvm/Codegen/LinkAllCodegenComponents.h ------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all codegen related passes for tools like lli and
// llc that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H
#define LLVM_CODEGEN_LINKALLCODEGENCOMPONENTS_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/GCs.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdlib>

namespace {
  struct ForceCodegenLinking {
    ForceCodegenLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void) llvm::createFastRegisterAllocator();
      (void) llvm::createBasicRegisterAllocator();
      (void) llvm::createGreedyRegisterAllocator();
      (void) llvm::createDefaultPBQPRegisterAllocator();

      llvm::linkOcamlGC();
      llvm::linkShadowStackGC();

      (void) llvm::createBURRListDAGScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createSourceListDAGScheduler(NULL,llvm::CodeGenOpt::Default);
      (void) llvm::createHybridListDAGScheduler(NULL,llvm::CodeGenOpt::Default);
      (void) llvm::createFastDAGScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createDefaultScheduler(NULL, llvm::CodeGenOpt::Default);
      (void) llvm::createVLIWDAGScheduler(NULL, llvm::CodeGenOpt::Default);

    }
  } ForceCodegenLinking; // Force link by creating a global definition.
}

#endif
