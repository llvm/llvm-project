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
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

class CtxProfAnalysis;

using FlatIndirectTargets = DenseMap<GlobalValue::GUID, uint64_t>;
using CtxProfFlatIndirectCallProfile =
    DenseMap<GlobalValue::GUID, DenseMap<uint32_t, FlatIndirectTargets>>;

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

  // True if this module is a post-thinlto module containing just functions
  // participating in one or more contextual profiles.
  bool IsInSpecializedModule = false;

  // For the GUIDs in this module, associate metadata about each function which
  // we'll need when we maintain the profiles during IPO transformations.
  std::map<GlobalValue::GUID, FunctionInfo> FuncInfo;

  /// Get the GUID of this Function if it's defined in this module.
  LLVM_ABI GlobalValue::GUID getDefinedFunctionGUID(const Function &F) const;

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

  LLVM_ABI bool isInSpecializedModule() const;

  bool isFunctionKnown(const Function &F) const {
    return getDefinedFunctionGUID(F) != 0;
  }

  StringRef getFunctionName(GlobalValue::GUID GUID) const {
    auto It = FuncInfo.find(GUID);
    if (It == FuncInfo.end())
      return "";
    return It->second.Name;
  }

  uint32_t getNumCounters(const Function &F) const {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCounterIndex;
  }

  uint32_t getNumCallsites(const Function &F) const {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCallsiteIndex;
  }

  uint32_t allocateNextCounterIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCounterIndex++;
  }

  uint32_t allocateNextCallsiteIndex(const Function &F) {
    assert(isFunctionKnown(F));
    return FuncInfo.find(getDefinedFunctionGUID(F))->second.NextCallsiteIndex++;
  }

  using ConstVisitor = function_ref<void(const PGOCtxProfContext &)>;
  using Visitor = function_ref<void(PGOCtxProfContext &)>;

  LLVM_ABI void update(Visitor, const Function &F);
  LLVM_ABI void visit(ConstVisitor, const Function *F = nullptr) const;

  LLVM_ABI const CtxProfFlatProfile flatten() const;
  LLVM_ABI const CtxProfFlatIndirectCallProfile flattenVirtCalls() const;

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
  LLVM_ABI static AnalysisKey Key;
  LLVM_ABI explicit CtxProfAnalysis(
      std::optional<StringRef> Profile = std::nullopt);

  using Result = PGOContextualProfile;

  LLVM_ABI PGOContextualProfile run(Module &M, ModuleAnalysisManager &MAM);

  /// Get the instruction instrumenting a callsite, or nullptr if that cannot be
  /// found.
  LLVM_ABI static InstrProfCallsite *getCallsiteInstrumentation(CallBase &CB);

  /// Get the instruction instrumenting a BB, or nullptr if not present.
  LLVM_ABI static InstrProfIncrementInst *getBBInstrumentation(BasicBlock &BB);

  /// Get the step instrumentation associated with a `select`
  LLVM_ABI static InstrProfIncrementInstStep *
  getSelectInstrumentation(SelectInst &SI);

  // FIXME: refactor to an advisor model, and separate
  LLVM_ABI static void collectIndirectCallPromotionList(
      CallBase &IC, Result &Profile,
      SetVector<std::pair<CallBase *, Function *>> &Candidates);
};

class CtxProfAnalysisPrinterPass
    : public PassInfoMixin<CtxProfAnalysisPrinterPass> {
public:
  enum class PrintMode { Everything, YAML };
  LLVM_ABI explicit CtxProfAnalysisPrinterPass(raw_ostream &OS);

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }

private:
  raw_ostream &OS;
  const PrintMode Mode;
};

/// Utility that propagates counter values to each basic block and to each edge
/// when a basic block has more than one outgoing edge, using an adaptation of
/// PGOUseFunc::populateCounters.
// FIXME(mtrofin): look into factoring the code to share one implementation.
class ProfileAnnotatorImpl;
class ProfileAnnotator {
  std::unique_ptr<ProfileAnnotatorImpl> PImpl;

public:
  LLVM_ABI ProfileAnnotator(const Function &F, ArrayRef<uint64_t> RawCounters);
  LLVM_ABI uint64_t getBBCount(const BasicBlock &BB) const;

  // Finds the true and false counts for the given select instruction. Returns
  // false if the select doesn't have instrumentation or if the count of the
  // parent BB is 0.
  LLVM_ABI bool getSelectInstrProfile(SelectInst &SI, uint64_t &TrueCount,
                                      uint64_t &FalseCount) const;
  // Clears Profile and populates it with the edge weights, in the same order as
  // they need to appear in the MD_prof metadata. Also computes the max of those
  // weights an returns it in MaxCount. Returs false if:
  //   - the BB has less than 2 successors
  //   - the counts are 0
  LLVM_ABI bool getOutgoingBranchWeights(BasicBlock &BB,
                                         SmallVectorImpl<uint64_t> &Profile,
                                         uint64_t &MaxCount) const;
  LLVM_ABI ~ProfileAnnotator();
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
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  LLVM_ABI static const char *GUIDMetadataName;
  // This should become GlobalValue::getGUID
  LLVM_ABI static uint64_t getGUID(const Function &F);
};

} // namespace llvm
#endif // LLVM_ANALYSIS_CTXPROFANALYSIS_H
