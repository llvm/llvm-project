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

#include "llvm/ADT/DenseMap.h"
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

  /// Get the GUID of this Function if it's defined in this module.
  GlobalValue::GUID getDefinedFunctionGUID(const Function &F) const;

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

  bool isFunctionKnown(const Function &F) const {
    return getDefinedFunctionGUID(F) != 0;
  }

  uint32_t allocateNextCounterIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCounterIndex++;
  }

  uint32_t allocateNextCallsiteIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCallsiteIndex++;
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

/// Assign a GUID to functions as metadata. GUID calculation takes linkage into
/// account, which may change especially through and after thinlto. By
/// pre-computing and assigning as metadata, this mechanism is resilient to such
/// changes (as well as name changes e.g. suffix ".llvm." additions).

// FIXME(mtrofin): we can generalize this mechanism to calculate a GUID early in
// the pass pipeline, associate it with any Global Value, and then use it for
// PGO and ThinLTO.
// At that point, this should be moved elsewhere.
class AssignGUIDPass : public PassInfoMixin<AssignGUIDPass> {
public:
  explicit AssignGUIDPass() = default;

  /// Assign a GUID *if* one is not already assign, as a function metadata named
  /// `GUIDMetadataName`.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static const char *GUIDMetadataName;
  // This should become GlobalValue::getGUID
  static uint64_t getGUID(const Function &F);
};

} // namespace llvm
#endif // LLVM_ANALYSIS_CTXPROFANALYSIS_H
