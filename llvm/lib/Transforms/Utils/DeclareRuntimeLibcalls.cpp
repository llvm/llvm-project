//===- DeclareRuntimeLibcalls.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Insert declarations for all runtime library calls known for the target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DeclareRuntimeLibcalls.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/RuntimeLibcalls.h"

using namespace llvm;

PreservedAnalyses DeclareRuntimeLibcallsPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  const RTLIB::RuntimeLibcallsInfo &RTLCI =
      MAM.getResult<RuntimeLibraryAnalysis>(M);

  for (RTLIB::LibcallImpl Impl : RTLIB::libcall_impls()) {
    if (!RTLCI.isAvailable(Impl))
      continue;

    if (RTLIB::RuntimeLibcallsInfo::isVariable(Impl))
      RTLCI.getOrInsertImplGlobal(M, Impl);
    else
      RTLCI.getOrInsertLibcallDecl(M, Impl);
  }

  return PreservedAnalyses::none();
}
