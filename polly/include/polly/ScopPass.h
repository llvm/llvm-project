//===--------- ScopPass.h - Pass for Static Control Parts --------*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScopPass class.  ScopPasses are just RegionPasses,
// except they operate on Polly IR (Scop and ScopStmt) built by ScopInfo Pass.
// Because they operate on Polly IR, not the LLVM IR, ScopPasses are not allowed
// to modify the LLVM IR. Due to this limitation, the ScopPass class takes
// care of declaring that no LLVM passes are invalidated.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_PASS_H
#define POLLY_SCOP_PASS_H

#include "polly/ScopInfo.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerImpl.h"

namespace polly {
using llvm::AllAnalysesOn;
using llvm::AnalysisManager;
using llvm::DominatorTreeAnalysis;
using llvm::InnerAnalysisManagerProxy;
using llvm::LoopAnalysis;
using llvm::OuterAnalysisManagerProxy;
using llvm::PassManager;
using llvm::RegionInfoAnalysis;
using llvm::ScalarEvolutionAnalysis;
using llvm::SmallPriorityWorklist;
using llvm::TargetIRAnalysis;
using llvm::TargetTransformInfo;

class Scop;
class SPMUpdater;
struct ScopStandardAnalysisResults;

using ScopAnalysisManager =
    AnalysisManager<Scop, ScopStandardAnalysisResults &>;
using ScopAnalysisManagerFunctionProxy =
    InnerAnalysisManagerProxy<ScopAnalysisManager, Function>;
using FunctionAnalysisManagerScopProxy =
    OuterAnalysisManagerProxy<FunctionAnalysisManager, Scop,
                              ScopStandardAnalysisResults &>;
} // namespace polly

namespace llvm {
using polly::Scop;
using polly::ScopAnalysisManager;
using polly::ScopAnalysisManagerFunctionProxy;
using polly::ScopInfo;
using polly::ScopStandardAnalysisResults;
using polly::SPMUpdater;

template <>
class InnerAnalysisManagerProxy<ScopAnalysisManager, Function>::Result {
public:
  explicit Result(ScopAnalysisManager &InnerAM, ScopInfo &SI)
      : InnerAM(&InnerAM), SI(&SI) {}
  Result(Result &&R) : InnerAM(std::move(R.InnerAM)), SI(R.SI) {
    R.InnerAM = nullptr;
  }
  Result &operator=(Result &&RHS) {
    InnerAM = RHS.InnerAM;
    SI = RHS.SI;
    RHS.InnerAM = nullptr;
    return *this;
  }
  ~Result() {
    if (!InnerAM)
      return;
    InnerAM->clear();
  }

  ScopAnalysisManager &getManager() { return *InnerAM; }

  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  ScopAnalysisManager *InnerAM;
  ScopInfo *SI;
};

// A partial specialization of the require analysis template pass to handle
// extra parameters
template <typename AnalysisT>
struct RequireAnalysisPass<AnalysisT, Scop, ScopAnalysisManager,
                           ScopStandardAnalysisResults &, SPMUpdater &>
    : PassInfoMixin<
          RequireAnalysisPass<AnalysisT, Scop, ScopAnalysisManager,
                              ScopStandardAnalysisResults &, SPMUpdater &>> {
  PreservedAnalyses run(Scop &L, ScopAnalysisManager &AM,
                        ScopStandardAnalysisResults &AR, SPMUpdater &) {
    (void)AM.template getResult<AnalysisT>(L, AR);
    return PreservedAnalyses::all();
  }
};

template <>
InnerAnalysisManagerProxy<ScopAnalysisManager, Function>::Result
InnerAnalysisManagerProxy<ScopAnalysisManager, Function>::run(
    Function &F, FunctionAnalysisManager &FAM);

template <>
PreservedAnalyses
PassManager<Scop, ScopAnalysisManager, ScopStandardAnalysisResults &,
            SPMUpdater &>::run(Scop &InitialS, ScopAnalysisManager &AM,
                               ScopStandardAnalysisResults &, SPMUpdater &);
extern template class PassManager<Scop, ScopAnalysisManager,
                                  ScopStandardAnalysisResults &, SPMUpdater &>;
extern template class InnerAnalysisManagerProxy<ScopAnalysisManager, Function>;
extern template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Scop,
                                                ScopStandardAnalysisResults &>;
} // namespace llvm

namespace polly {

template <typename AnalysisManagerT, typename IRUnitT, typename... ExtraArgTs>
class OwningInnerAnalysisManagerProxy final
    : public InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT> {
public:
  OwningInnerAnalysisManagerProxy()
      : InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT>(InnerAM) {}
  using Result = typename InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT,
                                                    ExtraArgTs...>::Result;
  Result run(IRUnitT &IR, AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
             ExtraArgTs...) {
    return Result(InnerAM);
  }

  AnalysisManagerT &getManager() { return InnerAM; }

private:
  AnalysisManagerT InnerAM;
};

template <>
OwningInnerAnalysisManagerProxy<ScopAnalysisManager, Function>::Result
OwningInnerAnalysisManagerProxy<ScopAnalysisManager, Function>::run(
    Function &F, FunctionAnalysisManager &FAM);
extern template class OwningInnerAnalysisManagerProxy<ScopAnalysisManager,
                                                      Function>;

using OwningScopAnalysisManagerFunctionProxy =
    OwningInnerAnalysisManagerProxy<ScopAnalysisManager, Function>;
using ScopPassManager =
    PassManager<Scop, ScopAnalysisManager, ScopStandardAnalysisResults &,
                SPMUpdater &>;

/// ScopPass - This class adapts the RegionPass interface to allow convenient
/// creation of passes that operate on the Polly IR. Instead of overriding
/// runOnRegion, subclasses override runOnScop.
class ScopPass : public RegionPass {
  Scop *S;

protected:
  explicit ScopPass(char &ID) : RegionPass(ID), S(nullptr) {}

  /// runOnScop - This method must be overloaded to perform the
  /// desired Polyhedral transformation or analysis.
  ///
  virtual bool runOnScop(Scop &S) = 0;

  /// Print method for SCoPs.
  virtual void printScop(raw_ostream &OS, Scop &S) const {}

  /// getAnalysisUsage - Subclasses that override getAnalysisUsage
  /// must call this.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool runOnRegion(Region *R, RGPassManager &RGM) override;
  void print(raw_ostream &OS, const Module *) const override;
};

struct ScopStandardAnalysisResults {
  DominatorTree &DT;
  ScopInfo &SI;
  ScalarEvolution &SE;
  LoopInfo &LI;
  RegionInfo &RI;
  TargetTransformInfo &TTI;
};

class SPMUpdater final {
public:
  SPMUpdater(SmallPriorityWorklist<Region *, 4> &Worklist,
             ScopAnalysisManager &SAM)
      : InvalidateCurrentScop(false), Worklist(Worklist), SAM(SAM) {}

  bool invalidateCurrentScop() const { return InvalidateCurrentScop; }

  void invalidateScop(Scop &S) {
    if (&S == CurrentScop)
      InvalidateCurrentScop = true;

    Worklist.erase(&S.getRegion());
    SAM.clear(S, S.getName());
  }

private:
  Scop *CurrentScop;
  bool InvalidateCurrentScop;
  SmallPriorityWorklist<Region *, 4> &Worklist;
  ScopAnalysisManager &SAM;
  template <typename ScopPassT> friend struct FunctionToScopPassAdaptor;
};

template <typename ScopPassT>
struct FunctionToScopPassAdaptor final
    : PassInfoMixin<FunctionToScopPassAdaptor<ScopPassT>> {
  explicit FunctionToScopPassAdaptor(ScopPassT Pass) : Pass(std::move(Pass)) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    ScopDetection &SD = AM.getResult<ScopAnalysis>(F);
    ScopInfo &SI = AM.getResult<ScopInfoAnalysis>(F);
    if (SI.empty()) {
      // With no scops having been detected, no IR changes have been made and
      // therefore all analyses are preserved. However, we must still free the
      // Scop analysis results which may hold AssertingVH that cause an error
      // if its value is destroyed.
      PreservedAnalyses PA = PreservedAnalyses::all();
      PA.abandon<ScopInfoAnalysis>();
      PA.abandon<ScopAnalysis>();
      AM.invalidate(F, PA);
      return PreservedAnalyses::all();
    }

    SmallPriorityWorklist<Region *, 4> Worklist;
    for (auto &S : SI)
      if (S.second)
        Worklist.insert(S.first);

    ScopStandardAnalysisResults AR = {AM.getResult<DominatorTreeAnalysis>(F),
                                      AM.getResult<ScopInfoAnalysis>(F),
                                      AM.getResult<ScalarEvolutionAnalysis>(F),
                                      AM.getResult<LoopAnalysis>(F),
                                      AM.getResult<RegionInfoAnalysis>(F),
                                      AM.getResult<TargetIRAnalysis>(F)};

    ScopAnalysisManager &SAM =
        AM.getResult<ScopAnalysisManagerFunctionProxy>(F).getManager();

    SPMUpdater Updater{Worklist, SAM};

    while (!Worklist.empty()) {
      Region *R = Worklist.pop_back_val();
      if (!SD.isMaxRegionInScop(*R, /*Verify=*/false))
        continue;
      Scop *scop = SI.getScop(R);
      if (!scop)
        continue;
      Updater.CurrentScop = scop;
      Updater.InvalidateCurrentScop = false;
      PreservedAnalyses PassPA = Pass.run(*scop, SAM, AR, Updater);

      SAM.invalidate(*scop, PassPA);
      if (Updater.invalidateCurrentScop())
        SI.recompute();
    };

    // FIXME: For the same reason as we add a BarrierNoopPass in the legacy pass
    // manager, do not preserve any analyses. While CodeGeneration may preserve
    // IR analyses sufficiently to process another Scop in the same function (it
    // has to, otherwise the ScopDetection result itself would need to be
    // invalidated), it is not sufficient for other purposes. For instance,
    // CodeGeneration does not inform LoopInfo about new loops in the
    // Polly-generated IR.
    return PreservedAnalyses::none();
  }

private:
  ScopPassT Pass;
};

template <typename ScopPassT>
FunctionToScopPassAdaptor<ScopPassT>
createFunctionToScopPassAdaptor(ScopPassT Pass) {
  return FunctionToScopPassAdaptor<ScopPassT>(std::move(Pass));
}
} // namespace polly

#endif
