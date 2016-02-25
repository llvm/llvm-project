//===- LoopPassManager.h - Loop pass management -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides classes for managing passes over loops in LLVM IR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPPASSMANAGER_H
#define LLVM_ANALYSIS_LOOPPASSMANAGER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \brief The loop pass manager.
///
/// See the documentation for the PassManager template for details. It runs a
/// sequency of loop passes over each loop that the manager is run over. This
/// typedef serves as a convenient way to refer to this construct.
typedef PassManager<Loop> LoopPassManager;

/// \brief The loop analysis manager.
///
/// See the documentation for the AnalysisManager template for detail
/// documentation. This typedef serves as a convenient way to refer to this
/// construct in the adaptors and proxies used to integrate this into the larger
/// pass manager infrastructure.
typedef AnalysisManager<Loop> LoopAnalysisManager;

/// \brief A function analysis which acts as a proxy for a loop analysis
/// manager.
///
/// This primarily proxies invalidation information from the function analysis
/// manager and function pass manager to a loop analysis manager. You should
/// never use a loop analysis manager from within (transitively) a function
/// pass manager unless your parent function pass has received a proxy result
/// object for it.
class LoopAnalysisManagerFunctionProxy {
public:
  class Result {
  public:
    explicit Result(LoopAnalysisManager &LAM) : LAM(&LAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : LAM(Arg.LAM) {}
    Result(Result &&Arg) : LAM(std::move(Arg.LAM)) {}
    Result &operator=(Result RHS) {
      std::swap(LAM, RHS.LAM);
      return *this;
    }
    ~Result();

    /// \brief Accessor for the \c LoopAnalysisManager.
    LoopAnalysisManager &getManager() { return *LAM; }

    /// \brief Handler for invalidation of the function.
    ///
    /// If this analysis itself is preserved, then we assume that the function
    /// hasn't changed and thus we don't need to invalidate *all* cached data
    /// associated with a \c Loop* in the \c LoopAnalysisManager.
    ///
    /// Regardless of whether this analysis is marked as preserved, all of the
    /// analyses in the \c LoopAnalysisManager are potentially invalidated based
    /// on the set of preserved analyses.
    bool invalidate(Function &F, const PreservedAnalyses &PA);

  private:
    LoopAnalysisManager *LAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "LoopAnalysisManagerFunctionProxy"; }

  explicit LoopAnalysisManagerFunctionProxy(LoopAnalysisManager &LAM)
      : LAM(&LAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  LoopAnalysisManagerFunctionProxy(const LoopAnalysisManagerFunctionProxy &Arg)
      : LAM(Arg.LAM) {}
  LoopAnalysisManagerFunctionProxy(LoopAnalysisManagerFunctionProxy &&Arg)
      : LAM(std::move(Arg.LAM)) {}
  LoopAnalysisManagerFunctionProxy &
  operator=(LoopAnalysisManagerFunctionProxy RHS) {
    std::swap(LAM, RHS.LAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work, it is primarily used to insert our
  /// proxy result object into the function analysis cache so that we can proxy
  /// invalidation to the loop analysis manager.
  ///
  /// In debug builds, it will also assert that the analysis manager is empty as
  /// no queries should arrive at the loop analysis manager prior to this
  /// analysis being requested.
  Result run(Function &F);

private:
  static char PassID;

  LoopAnalysisManager *LAM;
};

/// \brief A loop analysis which acts as a proxy for a function analysis
/// manager.
///
/// This primarily provides an accessor to a parent function analysis manager to
/// loop passes. Only the const interface of the function analysis manager is
/// provided to indicate that once inside of a loop analysis pass you cannot
/// request a function analysis to actually run. Instead, the user must rely on
/// the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
class FunctionAnalysisManagerLoopProxy {
public:
  /// \brief Result proxy object for \c FunctionAnalysisManagerLoopProxy.
  class Result {
  public:
    explicit Result(const FunctionAnalysisManager &FAM) : FAM(&FAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : FAM(Arg.FAM) {}
    Result(Result &&Arg) : FAM(std::move(Arg.FAM)) {}
    Result &operator=(Result RHS) {
      std::swap(FAM, RHS.FAM);
      return *this;
    }

    const FunctionAnalysisManager &getManager() const { return *FAM; }

    /// \brief Handle invalidation by ignoring it, this pass is immutable.
    bool invalidate(Function &) { return false; }

  private:
    const FunctionAnalysisManager *FAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "FunctionAnalysisManagerLoopProxy"; }

  FunctionAnalysisManagerLoopProxy(const FunctionAnalysisManager &FAM)
      : FAM(&FAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionAnalysisManagerLoopProxy(const FunctionAnalysisManagerLoopProxy &Arg)
      : FAM(Arg.FAM) {}
  FunctionAnalysisManagerLoopProxy(FunctionAnalysisManagerLoopProxy &&Arg)
      : FAM(std::move(Arg.FAM)) {}
  FunctionAnalysisManagerLoopProxy &
  operator=(FunctionAnalysisManagerLoopProxy RHS) {
    std::swap(FAM, RHS.FAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c FAM reference into the
  /// result.
  Result run(Loop &) { return Result(*FAM); }

private:
  static char PassID;

  const FunctionAnalysisManager *FAM;
};

/// \brief Adaptor that maps from a function to its loops.
///
/// Designed to allow composition of a LoopPass(Manager) and a
/// FunctionPassManager. Note that if this pass is constructed with a \c
/// FunctionAnalysisManager it will run the \c LoopAnalysisManagerFunctionProxy
/// analysis prior to running the loop passes over the function to enable a \c
/// LoopAnalysisManager to be used within this run safely.
template <typename LoopPassT> class FunctionToLoopPassAdaptor {
public:
  explicit FunctionToLoopPassAdaptor(LoopPassT Pass)
      : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionToLoopPassAdaptor(const FunctionToLoopPassAdaptor &Arg)
      : Pass(Arg.Pass) {}
  FunctionToLoopPassAdaptor(FunctionToLoopPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)) {}
  friend void swap(FunctionToLoopPassAdaptor &LHS,
                   FunctionToLoopPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  FunctionToLoopPassAdaptor &operator=(FunctionToLoopPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the loop passes across every loop in the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager *AM) {
    assert(AM && "We need analyses to compute the loop structure!");

    // Setup the loop analysis manager from its proxy.
    LoopAnalysisManager *LAM =
        &AM->getResult<LoopAnalysisManagerFunctionProxy>(F).getManager();
    // Get the loop structure for this function
    LoopInfo &LI = AM->getResult<LoopAnalysis>(F);

    PreservedAnalyses PA = PreservedAnalyses::all();

    // We want to visit the loops in reverse post-order. We'll build the stack
    // of loops to visit in Loops by first walking the loops in pre-order.
    SmallVector<Loop *, 2> Loops;
    SmallVector<Loop *, 2> WorkList(LI.begin(), LI.end());
    while (!WorkList.empty()) {
      Loop *L = WorkList.pop_back_val();
      WorkList.insert(WorkList.end(), L->begin(), L->end());
      Loops.push_back(L);
    }

    // Now pop each element off of the stack to visit the loops in reverse
    // post-order.
    for (auto *L : reverse(Loops)) {
      PreservedAnalyses PassPA = Pass.run(*L, LAM);

      // We know that the loop pass couldn't have invalidated any other loop's
      // analyses (that's the contract of a loop pass), so directly handle the
      // loop analysis manager's invalidation here.  Also, update the
      // preserved analyses to reflect that once invalidated these can again
      // be preserved.
      PassPA = LAM->invalidate(*L, std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation of
    // loop analyses by the proxy, but that's OK because we've taken care to
    // invalidate analyses in the loop analysis manager incrementally above.
    PA.preserve<LoopAnalysisManagerFunctionProxy>();
    return PA;
  }

  static StringRef name() { return "FunctionToLoopPassAdaptor"; }

private:
  LoopPassT Pass;
};

/// \brief A function to deduce a loop pass type and wrap it in the templated
/// adaptor.
template <typename LoopPassT>
FunctionToLoopPassAdaptor<LoopPassT>
createFunctionToLoopPassAdaptor(LoopPassT Pass) {
  return FunctionToLoopPassAdaptor<LoopPassT>(std::move(Pass));
}
}

#endif // LLVM_ANALYSIS_LOOPPASSMANAGER_H
