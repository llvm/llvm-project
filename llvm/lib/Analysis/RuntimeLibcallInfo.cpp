//===- RuntimeLibcallInfo.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

AnalysisKey RuntimeLibraryAnalysis::Key;

RTLIB::RuntimeLibcallsInfo
RuntimeLibraryAnalysis::run(const Module &M, ModuleAnalysisManager &) {
  if (!LibcallsInfo)
    LibcallsInfo = RTLIB::RuntimeLibcallsInfo(M);
  return *LibcallsInfo;
}

INITIALIZE_PASS(RuntimeLibraryInfoWrapper, "runtime-library-info",
                "Runtime Library Function Analysis", false, true)

RuntimeLibraryInfoWrapper::RuntimeLibraryInfoWrapper()
    : ImmutablePass(ID), RTLA(RTLIB::RuntimeLibcallsInfo(Triple())) {}

char RuntimeLibraryInfoWrapper::ID = 0;

ModulePass *llvm::createRuntimeLibraryInfoWrapperPass() {
  return new RuntimeLibraryInfoWrapper();
}

void RuntimeLibraryInfoWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

// Assume this is stable unless explicitly invalidated.
bool RTLIB::RuntimeLibcallsInfo::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<RuntimeLibraryAnalysis>();
  return !PAC.preservedWhenStateless();
}
