//===- CGSCCPassManager.h - Call graph pass management ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides classes for managing passes over SCCs of the call
/// graph. These passes form an important component of LLVM's interprocedural
/// optimizations. Because they operate on the SCCs of the call graph, and they
/// traverse the graph in post order, they can effectively do pair-wise
/// interprocedural optimizations for all call edges in the program. At each
/// call site edge, the callee has already been optimized as much as is
/// possible. This in turn allows very accurate analysis of it for IPO.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CGSCCPASSMANAGER_H
#define LLVM_ANALYSIS_CGSCCPASSMANAGER_H

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

extern template class PassManager<LazyCallGraph::SCC>;
/// \brief The CGSCC pass manager.
///
/// See the documentation for the PassManager template for details. It runs
/// a sequency of SCC passes over each SCC that the manager is run over. This
/// typedef serves as a convenient way to refer to this construct.
typedef PassManager<LazyCallGraph::SCC> CGSCCPassManager;

extern template class AnalysisManager<LazyCallGraph::SCC>;
/// \brief The CGSCC analysis manager.
///
/// See the documentation for the AnalysisManager template for detail
/// documentation. This typedef serves as a convenient way to refer to this
/// construct in the adaptors and proxies used to integrate this into the larger
/// pass manager infrastructure.
typedef AnalysisManager<LazyCallGraph::SCC> CGSCCAnalysisManager;

extern template class InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>;
/// A proxy from a \c CGSCCAnalysisManager to a \c Module.
typedef InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>
    CGSCCAnalysisManagerModuleProxy;

extern template class OuterAnalysisManagerProxy<ModuleAnalysisManager,
                                                LazyCallGraph::SCC>;
/// A proxy from a \c ModuleAnalysisManager to an \c SCC.
typedef OuterAnalysisManagerProxy<ModuleAnalysisManager, LazyCallGraph::SCC>
    ModuleAnalysisManagerCGSCCProxy;

/// \brief The core module pass which does a post-order walk of the SCCs and
/// runs a CGSCC pass over each one.
///
/// Designed to allow composition of a CGSCCPass(Manager) and
/// a ModulePassManager. Note that this pass must be run with a module analysis
/// manager as it uses the LazyCallGraph analysis. It will also run the
/// \c CGSCCAnalysisManagerModuleProxy analysis prior to running the CGSCC
/// pass over the module to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
template <typename CGSCCPassT>
class ModuleToPostOrderCGSCCPassAdaptor
    : public PassInfoMixin<ModuleToPostOrderCGSCCPassAdaptor<CGSCCPassT>> {
public:
  explicit ModuleToPostOrderCGSCCPassAdaptor(CGSCCPassT Pass, bool DebugLogging = false)
      : Pass(std::move(Pass)), DebugLogging(DebugLogging) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleToPostOrderCGSCCPassAdaptor(
      const ModuleToPostOrderCGSCCPassAdaptor &Arg)
      : Pass(Arg.Pass), DebugLogging(Arg.DebugLogging) {}
  ModuleToPostOrderCGSCCPassAdaptor(ModuleToPostOrderCGSCCPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)), DebugLogging(Arg.DebugLogging) {}
  friend void swap(ModuleToPostOrderCGSCCPassAdaptor &LHS,
                   ModuleToPostOrderCGSCCPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
    swap(LHS.DebugLogging, RHS.DebugLogging);
  }
  ModuleToPostOrderCGSCCPassAdaptor &
  operator=(ModuleToPostOrderCGSCCPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the CGSCC pass across every SCC in the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    // Setup the CGSCC analysis manager from its proxy.
    CGSCCAnalysisManager &CGAM =
        AM.getResult<CGSCCAnalysisManagerModuleProxy>(M).getManager();

    // Get the call graph for this module.
    LazyCallGraph &CG = AM.getResult<LazyCallGraphAnalysis>(M);

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::RefSCC &RC : CG.postorder_ref_sccs()) {
      if (DebugLogging)
        dbgs() << "Running an SCC pass across the RefSCC: " << RC << "\n";

      for (LazyCallGraph::SCC &C : RC) {
        PreservedAnalyses PassPA = Pass.run(C, CGAM);

        // We know that the CGSCC pass couldn't have invalidated any other
        // SCC's analyses (that's the contract of a CGSCC pass), so
        // directly handle the CGSCC analysis manager's invalidation here. We
        // also update the preserved set of analyses to reflect that invalidated
        // analyses are now safe to preserve.
        // FIXME: This isn't quite correct. We need to handle the case where the
        // pass updated the CG, particularly some child of the current SCC, and
        // invalidate its analyses.
        PassPA = CGAM.invalidate(C, std::move(PassPA));

        // Then intersect the preserved set so that invalidation of module
        // analyses will eventually occur when the module pass completes.
        PA.intersect(std::move(PassPA));
      }
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of CGSCC analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the CGSCC analysis manager
    // incrementally above.
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    return PA;
  }

private:
  CGSCCPassT Pass;
  bool DebugLogging;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename CGSCCPassT>
ModuleToPostOrderCGSCCPassAdaptor<CGSCCPassT>
createModuleToPostOrderCGSCCPassAdaptor(CGSCCPassT Pass, bool DebugLogging = false) {
  return ModuleToPostOrderCGSCCPassAdaptor<CGSCCPassT>(std::move(Pass), DebugLogging);
}

extern template class InnerAnalysisManagerProxy<FunctionAnalysisManager,
                                                LazyCallGraph::SCC>;
/// A proxy from a \c FunctionAnalysisManager to an \c SCC.
typedef InnerAnalysisManagerProxy<FunctionAnalysisManager, LazyCallGraph::SCC>
    FunctionAnalysisManagerCGSCCProxy;

extern template class OuterAnalysisManagerProxy<CGSCCAnalysisManager, Function>;
/// A proxy from a \c CGSCCAnalysisManager to a \c Function.
typedef OuterAnalysisManagerProxy<CGSCCAnalysisManager, Function>
    CGSCCAnalysisManagerFunctionProxy;

/// \brief Adaptor that maps from a SCC to its functions.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a CGSCCPassManager. Note that if this pass is constructed with a pointer
/// to a \c CGSCCAnalysisManager it will run the
/// \c FunctionAnalysisManagerCGSCCProxy analysis prior to running the function
/// pass over the SCC to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
template <typename FunctionPassT>
class CGSCCToFunctionPassAdaptor
    : public PassInfoMixin<CGSCCToFunctionPassAdaptor<FunctionPassT>> {
public:
  explicit CGSCCToFunctionPassAdaptor(FunctionPassT Pass, bool DebugLogging = false)
      : Pass(std::move(Pass)), DebugLogging(DebugLogging) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  CGSCCToFunctionPassAdaptor(const CGSCCToFunctionPassAdaptor &Arg)
      : Pass(Arg.Pass), DebugLogging(Arg.DebugLogging) {}
  CGSCCToFunctionPassAdaptor(CGSCCToFunctionPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)), DebugLogging(Arg.DebugLogging) {}
  friend void swap(CGSCCToFunctionPassAdaptor &LHS,
                   CGSCCToFunctionPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
    swap(LHS.DebugLogging, RHS.DebugLogging);
  }
  CGSCCToFunctionPassAdaptor &operator=(CGSCCToFunctionPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the function pass across every function in the module.
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM) {
    // Setup the function analysis manager from its proxy.
    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C).getManager();

    if (DebugLogging)
      dbgs() << "Running function passes across an SCC: " << C << "\n";

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::Node &N : C) {
      PreservedAnalyses PassPA = Pass.run(N.getFunction(), FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here.
      // Also, update the preserved analyses to reflect that once invalidated
      // these can again be preserved.
      PassPA = FAM.invalidate(N.getFunction(), std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of function analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the function analysis manager
    // incrementally above.
    // FIXME: We need to update the call graph here to account for any deleted
    // edges!
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    return PA;
  }

private:
  FunctionPassT Pass;
  bool DebugLogging;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
CGSCCToFunctionPassAdaptor<FunctionPassT>
createCGSCCToFunctionPassAdaptor(FunctionPassT Pass, bool DebugLogging = false) {
  return CGSCCToFunctionPassAdaptor<FunctionPassT>(std::move(Pass),
                                                   DebugLogging);
}
}

#endif
