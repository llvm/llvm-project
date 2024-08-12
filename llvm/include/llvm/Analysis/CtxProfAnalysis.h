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

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"

namespace llvm {

class CtxProfAnalysis;

/// The instrumented contextual profile, produced by the CtxProfAnalysis.
class PGOContextualProfile {
  friend class CtxProfAnalysis;
  friend class CtxProfAnalysisPrinterPass;
  struct FunctionInfo {
    uint32_t NextCounterIndex = 0;
    uint32_t NextCallsiteIndex = 0;
    const std::string Name;

    FunctionInfo(StringRef Name) : Name(Name) {}
  };
  std::optional<PGOCtxProfContext::CallTargetMapTy> Profiles;
  // For the GUIDs in this module, associate metadata about each function which
  // we'll need when we maintain the profiles during IPO transformations.
  DenseMap<GlobalValue::GUID, FunctionInfo> FuncInfo;

  GlobalValue::GUID getKnownGUID(const Function &F) const;

  // This is meant to be constructed from CtxProfAnalysis, which will also set
  // its state piecemeal.
  PGOContextualProfile() = default;

public:
  PGOContextualProfile(const PGOContextualProfile &) = delete;
  PGOContextualProfile(PGOContextualProfile &&) = default;

  operator bool() const { return Profiles.has_value(); }

  const PGOCtxProfContext::CallTargetMapTy &profiles() const {
    return *Profiles;
  }

  bool isFunctionKnown(const Function &F) const { return getKnownGUID(F) != 0; }

  uint32_t allocateNextCounterIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getKnownGUID(F))->second.NextCounterIndex++;
  }

  uint32_t allocateNextCallsiteIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getKnownGUID(F))->second.NextCallsiteIndex++;
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
