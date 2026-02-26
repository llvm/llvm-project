//===- LibcallLoweringInfo.cpp - Interface for runtime libcalls -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

LibcallLoweringInfo::LibcallLoweringInfo(
    const RTLIB::RuntimeLibcallsInfo &RTLCI,
    const TargetSubtargetInfo &Subtarget)
    : RTLCI(RTLCI) {
  // TODO: This should be generated with lowering predicates, and assert the
  // call is available.
  for (RTLIB::LibcallImpl Impl : RTLIB::libcall_impls()) {
    if (RTLCI.isAvailable(Impl)) {
      RTLIB::Libcall LC = RTLIB::RuntimeLibcallsInfo::getLibcallFromImpl(Impl);
      // FIXME: Hack, assume the first available libcall wins.
      if (LibcallImpls[LC] == RTLIB::Unsupported)
        LibcallImpls[LC] = Impl;
    }
  }

  Subtarget.initLibcallLoweringInfo(*this);
}

AnalysisKey LibcallLoweringModuleAnalysis::Key;

bool LibcallLoweringModuleAnalysisResult::invalidate(
    Module &, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  // Passes that change the runtime libcall set must explicitly invalidate this
  // pass.
  auto PAC = PA.getChecker<LibcallLoweringModuleAnalysis>();
  return !PAC.preservedWhenStateless();
}

LibcallLoweringModuleAnalysisResult
LibcallLoweringModuleAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  LibcallLoweringMap.init(&MAM.getResult<RuntimeLibraryAnalysis>(M));
  return LibcallLoweringMap;
}

INITIALIZE_PASS_BEGIN(LibcallLoweringInfoWrapper, "libcall-lowering-info",
                      "Library Function Lowering Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(RuntimeLibraryInfoWrapper)
INITIALIZE_PASS_END(LibcallLoweringInfoWrapper, "libcall-lowering-info",
                    "Library Function Lowering Analysis", false, true)

char LibcallLoweringInfoWrapper::ID = 0;

LibcallLoweringInfoWrapper::LibcallLoweringInfoWrapper() : ImmutablePass(ID) {}

void LibcallLoweringInfoWrapper::initializePass() {
  RuntimeLibcallsWrapper = &getAnalysis<RuntimeLibraryInfoWrapper>();
}

void LibcallLoweringInfoWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<RuntimeLibraryInfoWrapper>();
  AU.setPreservesAll();
}

void LibcallLoweringInfoWrapper::releaseMemory() { Result.clear(); }

ModulePass *llvm::createLibcallLoweringInfoWrapper() {
  return new LibcallLoweringInfoWrapper();
}
