//===- CtxProfAnalysis.h - maintain contextual profile info   -*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_CTXPROFANALYSIS_H
#define LLVM_ANALYSIS_CTXPROFANALYSIS_H

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include <map>

namespace llvm {

class CtxProfAnalysis;

/// The instrumented contextual profile, produced by the CtxProfAnalysis.
class PGOContextualProfile {
  std::optional<PGOCtxProfContext::CallTargetMapTy> Profiles;

public:
  explicit PGOContextualProfile(PGOCtxProfContext::CallTargetMapTy &&Profiles)
      : Profiles(std::move(Profiles)) {}
  PGOContextualProfile() = default;
  PGOContextualProfile(const PGOContextualProfile &) = delete;
  PGOContextualProfile(PGOContextualProfile &&) = default;

  operator bool() const { return Profiles.has_value(); }

  const PGOCtxProfContext::CallTargetMapTy &profiles() const {
    return *Profiles;
  }

  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &) {
    // Check whether the analysis has been explicitly invalidated. Otherwise,
    // it's stateless and remains preserved.
    auto PAC = PA.getChecker<CtxProfAnalysis>();
    return !PAC.preservedWhenStateless();
  }
};

class CtxProfAnalysis : public AnalysisInfoMixin<CtxProfAnalysis> {
  StringRef Profile;

public:
  static AnalysisKey Key;
  explicit CtxProfAnalysis(StringRef Profile) : Profile(Profile) {};

  using Result = PGOContextualProfile;

  PGOContextualProfile run(Module &M, ModuleAnalysisManager &MAM);
};

class CtxProfAnalysisPrinterPass
    : public PassInfoMixin<CtxProfAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit CtxProfAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};
} // namespace llvm
#endif // LLVM_ANALYSIS_CTXPROFANALYSIS_H
