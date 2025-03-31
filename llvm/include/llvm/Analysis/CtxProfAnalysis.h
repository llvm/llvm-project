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

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include <optional>

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
    PGOCtxProfContext Index;
    FunctionInfo(StringRef Name) : Name(Name) {}
  };
  PGOCtxProfile Profiles;
  // For the GUIDs in this module, associate metadata about each function which
  // we'll need when we maintain the profiles during IPO transformations.
  std::map<GlobalValue::GUID, FunctionInfo> FuncInfo;

  // This is meant to be constructed from CtxProfAnalysis, which will also set
  // its state piecemeal.
  PGOContextualProfile() = default;

  void initIndex();

public:
  PGOContextualProfile(const PGOContextualProfile &) = delete;
  PGOContextualProfile(PGOContextualProfile &&) = default;

  const CtxProfContextualProfiles &contexts() const {
    return Profiles.Contexts;
  }

  const PGOCtxProfile &profiles() const { return Profiles; }

  bool isFunctionKnown(const Function &F) const { return F.getGUID() != 0; }

  StringRef getFunctionName(GlobalValue::GUID GUID) const {
    auto It = FuncInfo.find(GUID);
    if (It == FuncInfo.end())
      return "";
    return It->second.Name;
  }

  uint32_t getNumCounters(const Function &F) const {
    assert(isFunctionKnown(F));
    return FuncInfo.find(F.getGUID())->second.NextCounterIndex;
  }

  uint32_t getNumCallsites(const Function &F) const {
    assert(isFunctionKnown(F));
    return FuncInfo.find(F.getGUID())->second.NextCallsiteIndex;
  }

  uint32_t allocateNextCounterIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(F.getGUID())->second.NextCounterIndex++;
  }

  uint32_t allocateNextCallsiteIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(F.getGUID())->second.NextCallsiteIndex++;
  }

  using ConstVisitor = function_ref<void(const PGOCtxProfContext &)>;
  using Visitor = function_ref<void(PGOCtxProfContext &)>;

  void update(Visitor, const Function &F);
  void visit(ConstVisitor, const Function *F = nullptr) const;

  const CtxProfFlatProfile flatten() const;

  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &) {
    // Check whether the analysis has been explicitly invalidated. Otherwise,
    // it's stateless and remains preserved.
    auto PAC = PA.getChecker<CtxProfAnalysis>();
    return !PAC.preservedWhenStateless();
  }
};

class CtxProfAnalysis : public AnalysisInfoMixin<CtxProfAnalysis> {
  const std::optional<StringRef> Profile;

public:
  static AnalysisKey Key;
  explicit CtxProfAnalysis(std::optional<StringRef> Profile = std::nullopt);

  using Result = PGOContextualProfile;

  PGOContextualProfile run(Module &M, ModuleAnalysisManager &MAM);

  /// Get the instruction instrumenting a callsite, or nullptr if that cannot be
  /// found.
  static InstrProfCallsite *getCallsiteInstrumentation(CallBase &CB);

  /// Get the instruction instrumenting a BB, or nullptr if not present.
  static InstrProfIncrementInst *getBBInstrumentation(BasicBlock &BB);

  /// Get the step instrumentation associated with a `select`
  static InstrProfIncrementInstStep *getSelectInstrumentation(SelectInst &SI);

  // FIXME: refactor to an advisor model, and separate
  static void collectIndirectCallPromotionList(
      CallBase &IC, Result &Profile,
      SetVector<std::pair<CallBase *, Function *>> &Candidates);
};

class CtxProfAnalysisPrinterPass
    : public PassInfoMixin<CtxProfAnalysisPrinterPass> {
public:
  enum class PrintMode { Everything, YAML };
  explicit CtxProfAnalysisPrinterPass(raw_ostream &OS);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
  const PrintMode Mode;
};

} // namespace llvm
#endif // LLVM_ANALYSIS_CTXPROFANALYSIS_H
