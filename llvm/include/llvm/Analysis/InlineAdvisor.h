//===- InlineAdvisor.h - Inlining decision making abstraction -*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_INLINEADVISOR_H_
#define LLVM_INLINEADVISOR_H_

#include <memory>
#include <unordered_set>
#include <vector>

#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class BasicBlock;
class CallBase;
class Function;
class Module;
class OptimizationRemarkEmitter;

/// There are 3 scenarios we can use the InlineAdvisor:
/// - Default - use manual heuristics.
///
/// - Release mode, the expected mode for production, day to day deployments.
/// In this mode, when building the compiler, we also compile a pre-trained ML
/// model to native code, and link it as a static library. This mode has low
/// overhead and no additional dependencies for the compiler runtime.
///
/// - Development mode, for training new models.
/// In this mode, we trade off runtime performance for flexibility. This mode
/// requires the full C Tensorflow API library, and evaluates models
/// dynamically. This mode also permits generating training logs, for offline
/// training.
enum class InliningAdvisorMode : int { Default, Release, Development };

class InlineAdvisor;
/// Capture state between an inlining decision having had been made, and
/// its impact being observable. When collecting model training data, this
/// allows recording features/decisions/partial reward data sets.
///
/// Derivations of this type are expected to be tightly coupled with their
/// InliningAdvisors. The base type implements the minimal contractual
/// obligations.
class InlineAdvice {
public:
  InlineAdvice(InlineAdvisor *Advisor, CallBase &CB,
               OptimizationRemarkEmitter &ORE, bool IsInliningRecommended);

  InlineAdvice(InlineAdvice &&) = delete;
  InlineAdvice(const InlineAdvice &) = delete;
  virtual ~InlineAdvice() {
    assert(Recorded && "InlineAdvice should have been informed of the "
                       "inliner's decision in all cases");
  }

  /// Exactly one of the record* APIs must be called. Implementers may extend
  /// behavior by implementing the corresponding record*Impl.
  ///
  /// Call after inlining succeeded, and did not result in deleting the callee.
  void recordInlining() {
    markRecorded();
    recordInliningImpl();
  }

  /// Call after inlining succeeded, and resulted in deleting the callee.
  void recordInliningWithCalleeDeleted();

  /// Call after the decision for a call site was to not inline.
  void recordUnsuccessfulInlining(const InlineResult &Result) {
    markRecorded();
    recordUnsuccessfulInliningImpl(Result);
  }

  /// Call to indicate inlining was not attempted.
  void recordUnattemptedInlining() {
    markRecorded();
    recordUnattemptedInliningImpl();
  }

  /// Get the inlining recommendation.
  bool isInliningRecommended() const { return IsInliningRecommended; }
  const DebugLoc &getOriginalCallSiteDebugLoc() const { return DLoc; }
  const BasicBlock *getOriginalCallSiteBasicBlock() const { return Block; }

protected:
  virtual void recordInliningImpl() {}
  virtual void recordInliningWithCalleeDeletedImpl() {}
  virtual void recordUnsuccessfulInliningImpl(const InlineResult &Result) {}
  virtual void recordUnattemptedInliningImpl() {}

  InlineAdvisor *const Advisor;
  /// Caller and Callee are pre-inlining.
  Function *const Caller;
  Function *const Callee;

  // Capture the context of CB before inlining, as a successful inlining may
  // change that context, and we want to report success or failure in the
  // original context.
  const DebugLoc DLoc;
  const BasicBlock *const Block;
  OptimizationRemarkEmitter &ORE;
  const bool IsInliningRecommended;

private:
  void markRecorded() {
    assert(!Recorded && "Recording should happen exactly once");
    Recorded = true;
  }

  bool Recorded = false;
};

/// Interface for deciding whether to inline a call site or not.
class InlineAdvisor {
public:
  InlineAdvisor(InlineAdvisor &&) = delete;
  virtual ~InlineAdvisor() { freeDeletedFunctions(); }

  /// Get an InlineAdvice containing a recommendation on whether to
  /// inline or not. \p CB is assumed to be a direct call. \p FAM is assumed to
  /// be up-to-date wrt previous inlining decisions.
  /// Returns an InlineAdvice with the inlining recommendation.
  virtual std::unique_ptr<InlineAdvice> getAdvice(CallBase &CB) = 0;

  /// This must be called when the Inliner pass is entered, to allow the
  /// InlineAdvisor update internal state, as result of function passes run
  /// between Inliner pass runs (for the same module).
  virtual void onPassEntry() {}

  /// This must be called when the Inliner pass is exited, as function passes
  /// may be run subsequently. This allows an implementation of InlineAdvisor
  /// to prepare for a partial update.
  virtual void onPassExit() {}

protected:
  InlineAdvisor(FunctionAnalysisManager &FAM) : FAM(FAM) {}

  FunctionAnalysisManager &FAM;

  /// We may want to defer deleting functions to after the inlining for a whole
  /// module has finished. This allows us to reliably use function pointers as
  /// unique identifiers, as an efficient implementation detail of the
  /// InlineAdvisor. Otherwise, it is possible the memory allocator
  /// re-allocate Function objects at the same address of a deleted Function;
  /// and Functions are potentially created during the function passes called
  /// after each SCC inlining (e.g. argument promotion does that).
  void freeDeletedFunctions();

  bool isFunctionDeleted(const Function *F) const {
    return DeletedFunctions.count(F);
  }

private:
  friend class InlineAdvice;
  void markFunctionAsDeleted(Function *F);
  std::unordered_set<const Function *> DeletedFunctions;
};

/// The default (manual heuristics) implementation of the InlineAdvisor. This
/// implementation does not need to keep state between inliner pass runs, and is
/// reusable as-is for inliner pass test scenarios, as well as for regular use.
class DefaultInlineAdvisor : public InlineAdvisor {
public:
  DefaultInlineAdvisor(FunctionAnalysisManager &FAM, InlineParams Params)
      : InlineAdvisor(FAM), Params(Params) {}

private:
  std::unique_ptr<InlineAdvice> getAdvice(CallBase &CB) override;

  void onPassExit() override { freeDeletedFunctions(); }

  InlineParams Params;
};

/// The InlineAdvisorAnalysis is a module pass because the InlineAdvisor
/// needs to capture state right before inlining commences over a module.
class InlineAdvisorAnalysis : public AnalysisInfoMixin<InlineAdvisorAnalysis> {
public:
  static AnalysisKey Key;
  InlineAdvisorAnalysis() = default;
  struct Result {
    Result(Module &M, ModuleAnalysisManager &MAM) : M(M), MAM(MAM) {}
    bool invalidate(Module &, const PreservedAnalyses &,
                    ModuleAnalysisManager::Invalidator &) {
      // InlineAdvisor must be preserved across analysis invalidations.
      return false;
    }
    bool tryCreate(InlineParams Params, InliningAdvisorMode Mode);
    InlineAdvisor *getAdvisor() const { return Advisor.get(); }
    void clear() { Advisor.reset(); }

  private:
    Module &M;
    ModuleAnalysisManager &MAM;
    std::unique_ptr<InlineAdvisor> Advisor;
  };

  Result run(Module &M, ModuleAnalysisManager &MAM) { return Result(M, MAM); }
};

// Default (manual policy) decision making helper APIs. Shared with the legacy
// pass manager inliner.

/// Return the cost only if the inliner should attempt to inline at the given
/// CallSite. If we return the cost, we will emit an optimisation remark later
/// using that cost, so we won't do so from this function. Return None if
/// inlining should not be attempted.
Optional<InlineCost>
shouldInline(CallBase &CB, function_ref<InlineCost(CallBase &CB)> GetInlineCost,
             OptimizationRemarkEmitter &ORE, bool EnableDeferral = true);

/// Emit ORE message.
void emitInlinedInto(OptimizationRemarkEmitter &ORE, DebugLoc DLoc,
                     const BasicBlock *Block, const Function &Callee,
                     const Function &Caller, const InlineCost &IC,
                     bool ForProfileContext = false,
                     const char *PassName = nullptr);

/// Add location info to ORE message.
void addLocationToRemarks(OptimizationRemark &Remark, DebugLoc DLoc);

/// Set the inline-remark attribute.
void setInlineRemark(CallBase &CB, StringRef Message);

/// Utility for extracting the inline cost message to a string.
std::string inlineCostStr(const InlineCost &IC);
} // namespace llvm
#endif // LLVM_INLINEADVISOR_H_
