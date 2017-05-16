//===--------- ScopPass.h - Pass for Static Control Parts --------*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/IR/PassManager.h"

using namespace llvm;

namespace polly {
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
using polly::ScopInfo;
using polly::ScopAnalysisManager;
using polly::ScopStandardAnalysisResults;
using polly::ScopAnalysisManagerFunctionProxy;
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
using ScopPassManager =
    PassManager<Scop, ScopAnalysisManager, ScopStandardAnalysisResults &,
                SPMUpdater &>;

/// ScopPass - This class adapts the RegionPass interface to allow convenient
/// creation of passes that operate on the Polly IR. Instead of overriding
/// runOnRegion, subclasses override runOnScop.
class ScopPass : public RegionPass {
  Scop *S;

protected:
  explicit ScopPass(char &ID) : RegionPass(ID), S(0) {}

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
  ScalarEvolution &SE;
  LoopInfo &LI;
  RegionInfo &RI;
};

class SPMUpdater {
public:
  SPMUpdater(SmallPriorityWorklist<Scop *, 4> &Worklist,
             ScopAnalysisManager &SAM)
      : Worklist(Worklist), SAM(SAM) {}

  void SkipScop(Scop &S) {
    if (Worklist.erase(&S))
      SAM.clear(S);
  }

private:
  SmallPriorityWorklist<Scop *, 4> &Worklist;
  ScopAnalysisManager &SAM;
};

template <typename ScopPassT>
class FunctionToScopPassAdaptor
    : public PassInfoMixin<FunctionToScopPassAdaptor<ScopPassT>> {
public:
  explicit FunctionToScopPassAdaptor(ScopPassT Pass) : Pass(std::move(Pass)) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    PreservedAnalyses PA = PreservedAnalyses::all();
    auto &Scops = AM.getResult<ScopInfoAnalysis>(F);
    if (Scops.empty())
      return PA;

    ScopAnalysisManager &SAM =
        AM.getResult<ScopAnalysisManagerFunctionProxy>(F).getManager();

    ScopStandardAnalysisResults AR = {AM.getResult<DominatorTreeAnalysis>(F),
                                      AM.getResult<ScalarEvolutionAnalysis>(F),
                                      AM.getResult<LoopAnalysis>(F),
                                      AM.getResult<RegionInfoAnalysis>(F)};

    SmallPriorityWorklist<Scop *, 4> Worklist;
    SPMUpdater Updater{Worklist, SAM};

    for (auto &S : Scops)
      if (auto *scop = S.second.get())
        Worklist.insert(scop);

    while (!Worklist.empty()) {
      Scop *scop = Worklist.pop_back_val();
      PreservedAnalyses PassPA = Pass.run(*scop, SAM, AR, Updater);

      SAM.invalidate(*scop, PassPA);
      PA.intersect(std::move(PassPA));
    };

    PA.preserveSet<AllAnalysesOn<Scop>>();
    PA.preserve<ScopAnalysisManagerFunctionProxy>();
    return PA;
  }

private:
  ScopPassT Pass;
}; // namespace polly

template <typename ScopPassT>
FunctionToScopPassAdaptor<ScopPassT>
createFunctionToScopPassAdaptor(ScopPassT Pass) {
  return FunctionToScopPassAdaptor<ScopPassT>(std::move(Pass));
}

} // namespace polly

#endif
