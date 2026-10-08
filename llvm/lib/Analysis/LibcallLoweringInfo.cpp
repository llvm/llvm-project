//===- LibcallLoweringInfo.cpp - Runtime libcall lowering info ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LibcallLoweringInfo.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"

using namespace llvm;

LibcallLoweringInfo::LibcallLoweringInfo(
    const RTLIB::RuntimeLibcallsInfo &RTLCI,
    ApplyContextRulesFn ApplyContextRules)
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

  // Apply the caller's context-specific (e.g. subtarget) libcall rules.
  if (ApplyContextRules)
    ApplyContextRules(*this);
}

AnalysisKey LibcallLoweringModuleAnalysis::Key;

bool ModuleLibcallLoweringInfo::invalidate(
    Module &, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  // Passes that change the runtime libcall set must explicitly invalidate this
  // pass.
  auto PAC = PA.getChecker<LibcallLoweringModuleAnalysis>();
  return !PAC.preservedWhenStateless();
}

ModuleLibcallLoweringInfo
LibcallLoweringModuleAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  LibcallLoweringMap.init(&MAM.getResult<RuntimeLibraryAnalysis>(M));
  return LibcallLoweringMap;
}
