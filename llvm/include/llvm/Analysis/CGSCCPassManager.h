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
/// traverse the graph in post-order, they can effectively do pair-wise
/// interprocedural optimizations for all call edges in the program while
/// incrementally refining it and improving the context of these pair-wise
/// optimizations. At each call site edge, the callee has already been
/// optimized as much as is possible. This in turn allows very accurate
/// analysis of it for IPO.
///
/// A secondary more general goal is to be able to isolate optimization on
/// unrelated parts of the IR module. This is useful to ensure our
/// optimizations are principled and don't miss oportunities where refinement
/// of one part of the module influence transformations in another part of the
/// module. But this is also useful if we want to parallelize the optimizations
/// across common large module graph shapes which tend to be very wide and have
/// large regions of unrelated cliques.
///
/// To satisfy these goals, we use the LazyCallGraph which provides two graphs
/// nested inside each other (and built lazily from the bottom-up): the call
/// graph proper, and a reference graph. The reference graph is super set of
/// the call graph and is a conservative approximation of what could through
/// scalar or CGSCC transforms *become* the call graph. Using this allows us to
/// ensure we optimize functions prior to them being introduced into the call
/// graph by devirtualization or other technique, and thus ensures that
/// subsequent pair-wise interprocedural optimizations observe the optimized
/// form of these functions. The (potentially transitive) reference
/// reachability used by the reference graph is a conservative approximation
/// that still allows us to have independent regions of the graph.
///
/// FIXME: There is one major drawback of the reference graph: in its naive
/// form it is quadratic because it contains a distinct edge for each
/// (potentially indirect) reference, even if are all through some common
/// global table of function pointers. This can be fixed in a number of ways
/// that essentially preserve enough of the normalization. While it isn't
/// expected to completely preclude the usability of this, it will need to be
/// addressed.
///
///
/// All of these issues are made substantially more complex in the face of
/// mutations to the call graph while optimization passes are being run. When
/// mutations to the call graph occur we want to achieve two different things:
///
/// - We need to update the call graph in-flight and invalidate analyses
///   cached on entities in the graph. Because of the cache-based analysis
///   design of the pass manager, it is essential to have stable identities for
///   the elements of the IR that passes traverse, and to invalidate any
///   analyses cached on these elements as the mutations take place.
///
/// - We want to preserve the incremental and post-order traversal of the
///   graph even as it is refined and mutated. This means we want optimization
///   to observe the most refined form of the call graph and to do so in
///   post-order.
///
/// To address this, the CGSCC manager uses both worklists that can be expanded
/// by passes which transform the IR, and provides invalidation tests to skip
/// entries that become dead. This extra data is provided to every SCC pass so
/// that it can carefully update the manager's traversal as the call graph
/// mutates.
///
/// We also provide support for running function passes within the CGSCC walk,
/// and there we provide automatic update of the call graph including of the
/// pass manager to reflect call graph changes that fall out naturally as part
/// of scalar transformations.
///
/// The patterns used to ensure the goals of post-order visitation of the fully
/// refined graph:
///
/// 1) Sink toward the "bottom" as the graph is refined. This means that any
///    iteration continues in some valid post-order sequence after the mutation
///    has altered the structure.
///
/// 2) Enqueue in post-order, including the current entity. If the current
///    entity's shape changes, it and everything after it in post-order needs
///    to be visited to observe that shape.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CGSCCPASSMANAGER_H
#define LLVM_ANALYSIS_CGSCCPASSMANAGER_H

#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

struct CGSCCUpdateResult;

extern template class AnalysisManager<LazyCallGraph::SCC, LazyCallGraph &>;
/// \brief The CGSCC analysis manager.
///
/// See the documentation for the AnalysisManager template for detail
/// documentation. This typedef serves as a convenient way to refer to this
/// construct in the adaptors and proxies used to integrate this into the larger
/// pass manager infrastructure.
typedef AnalysisManager<LazyCallGraph::SCC, LazyCallGraph &>
    CGSCCAnalysisManager;

// Explicit specialization and instantiation declarations for the pass manager.
// See the comments on the definition of the specialization for details on how
// it differs from the primary template.
template <>
PreservedAnalyses
PassManager<LazyCallGraph::SCC, CGSCCAnalysisManager, LazyCallGraph &,
            CGSCCUpdateResult &>::run(LazyCallGraph::SCC &InitialC,
                                      CGSCCAnalysisManager &AM,
                                      LazyCallGraph &G, CGSCCUpdateResult &UR);
extern template class PassManager<LazyCallGraph::SCC, CGSCCAnalysisManager,
                                  LazyCallGraph &, CGSCCUpdateResult &>;

/// \brief The CGSCC pass manager.
///
/// See the documentation for the PassManager template for details. It runs
/// a sequency of SCC passes over each SCC that the manager is run over. This
/// typedef serves as a convenient way to refer to this construct.
typedef PassManager<LazyCallGraph::SCC, CGSCCAnalysisManager, LazyCallGraph &,
                    CGSCCUpdateResult &>
    CGSCCPassManager;

/// An explicit specialization of the require analysis template pass.
template <typename AnalysisT>
struct RequireAnalysisPass<AnalysisT, LazyCallGraph::SCC, CGSCCAnalysisManager,
                           LazyCallGraph &, CGSCCUpdateResult &>
    : PassInfoMixin<RequireAnalysisPass<AnalysisT, LazyCallGraph::SCC,
                                        CGSCCAnalysisManager, LazyCallGraph &,
                                        CGSCCUpdateResult &>> {
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &) {
    (void)AM.template getResult<AnalysisT>(C, CG);
    return PreservedAnalyses::all();
  }
};

extern template class InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>;
/// A proxy from a \c CGSCCAnalysisManager to a \c Module.
typedef InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>
    CGSCCAnalysisManagerModuleProxy;

extern template class OuterAnalysisManagerProxy<
    ModuleAnalysisManager, LazyCallGraph::SCC, LazyCallGraph &>;
/// A proxy from a \c ModuleAnalysisManager to an \c SCC.
typedef OuterAnalysisManagerProxy<ModuleAnalysisManager, LazyCallGraph::SCC,
                                  LazyCallGraph &>
    ModuleAnalysisManagerCGSCCProxy;

/// Support structure for SCC passes to communicate updates the call graph back
/// to the CGSCC pass manager infrsatructure.
///
/// The CGSCC pass manager runs SCC passes which are allowed to update the call
/// graph and SCC structures. This means the structure the pass manager works
/// on is mutating underneath it. In order to support that, there needs to be
/// careful communication about the precise nature and ramifications of these
/// updates to the pass management infrastructure.
///
/// All SCC passes will have to accept a reference to the management layer's
/// update result struct and use it to reflect the results of any CG updates
/// performed.
///
/// Passes which do not change the call graph structure in any way can just
/// ignore this argument to their run method.
struct CGSCCUpdateResult {
  /// Worklist of the RefSCCs queued for processing.
  ///
  /// When a pass refines the graph and creates new RefSCCs or causes them to
  /// have a different shape or set of component SCCs it should add the RefSCCs
  /// to this worklist so that we visit them in the refined form.
  ///
  /// This worklist is in reverse post-order, as we pop off the back in order
  /// to observe RefSCCs in post-order. When adding RefSCCs, clients should add
  /// them in reverse post-order.
  SmallPriorityWorklist<LazyCallGraph::RefSCC *, 1> &RCWorklist;

  /// Worklist of the SCCs queued for processing.
  ///
  /// When a pass refines the graph and creates new SCCs or causes them to have
  /// a different shape or set of component functions it should add the SCCs to
  /// this worklist so that we visit them in the refined form.
  ///
  /// Note that if the SCCs are part of a RefSCC that is added to the \c
  /// RCWorklist, they don't need to be added here as visiting the RefSCC will
  /// be sufficient to re-visit the SCCs within it.
  ///
  /// This worklist is in reverse post-order, as we pop off the back in order
  /// to observe SCCs in post-order. When adding SCCs, clients should add them
  /// in reverse post-order.
  SmallPriorityWorklist<LazyCallGraph::SCC *, 1> &CWorklist;

  /// The set of invalidated RefSCCs which should be skipped if they are found
  /// in \c RCWorklist.
  ///
  /// This is used to quickly prune out RefSCCs when they get deleted and
  /// happen to already be on the worklist. We use this primarily to avoid
  /// scanning the list and removing entries from it.
  SmallPtrSetImpl<LazyCallGraph::RefSCC *> &InvalidatedRefSCCs;

  /// The set of invalidated SCCs which should be skipped if they are found
  /// in \c CWorklist.
  ///
  /// This is used to quickly prune out SCCs when they get deleted and happen
  /// to already be on the worklist. We use this primarily to avoid scanning
  /// the list and removing entries from it.
  SmallPtrSetImpl<LazyCallGraph::SCC *> &InvalidatedSCCs;

  /// If non-null, the updated current \c RefSCC being processed.
  ///
  /// This is set when a graph refinement takes place an the "current" point in
  /// the graph moves "down" or earlier in the post-order walk. This will often
  /// cause the "current" RefSCC to be a newly created RefSCC object and the
  /// old one to be added to the above worklist. When that happens, this
  /// pointer is non-null and can be used to continue processing the "top" of
  /// the post-order walk.
  LazyCallGraph::RefSCC *UpdatedRC;

  /// If non-null, the updated current \c SCC being processed.
  ///
  /// This is set when a graph refinement takes place an the "current" point in
  /// the graph moves "down" or earlier in the post-order walk. This will often
  /// cause the "current" SCC to be a newly created SCC object and the old one
  /// to be added to the above worklist. When that happens, this pointer is
  /// non-null and can be used to continue processing the "top" of the
  /// post-order walk.
  LazyCallGraph::SCC *UpdatedC;
};

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

    // We keep worklists to allow us to push more work onto the pass manager as
    // the passes are run.
    SmallPriorityWorklist<LazyCallGraph::RefSCC *, 1> RCWorklist;
    SmallPriorityWorklist<LazyCallGraph::SCC *, 1> CWorklist;

    // Keep sets for invalidated SCCs and RefSCCs that should be skipped when
    // iterating off the worklists.
    SmallPtrSet<LazyCallGraph::RefSCC *, 4> InvalidRefSCCSet;
    SmallPtrSet<LazyCallGraph::SCC *, 4> InvalidSCCSet;

    CGSCCUpdateResult UR = {RCWorklist,    CWorklist, InvalidRefSCCSet,
                            InvalidSCCSet, nullptr,   nullptr};

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::RefSCC &InitialRC : CG.postorder_ref_sccs()) {
      assert(RCWorklist.empty() &&
             "Should always start with an empty RefSCC worklist");
      // The postorder_ref_sccs range we are walking is lazily constructed, so
      // we only push the first one onto the worklist. The worklist allows us
      // to capture *new* RefSCCs created during transformations.
      //
      // We really want to form RefSCCs lazily because that makes them cheaper
      // to update as the program is simplified and allows us to have greater
      // cache locality as forming a RefSCC touches all the parts of all the
      // functions within that RefSCC.
      RCWorklist.insert(&InitialRC);

      do {
        LazyCallGraph::RefSCC *RC = RCWorklist.pop_back_val();
        if (InvalidRefSCCSet.count(RC))
          continue;

        assert(CWorklist.empty() &&
               "Should always start with an empty SCC worklist");

        if (DebugLogging)
          dbgs() << "Running an SCC pass across the RefSCC: " << *RC << "\n";

        // Push the initial SCCs in reverse post-order as we'll pop off the the
        // back and so see this in post-order.
        for (LazyCallGraph::SCC &C : reverse(*RC))
          CWorklist.insert(&C);

        do {
          LazyCallGraph::SCC *C = CWorklist.pop_back_val();
          // Due to call graph mutations, we may have invalid SCCs or SCCs from
          // other RefSCCs in the worklist. The invalid ones are dead and the
          // other RefSCCs should be queued above, so we just need to skip both
          // scenarios here.
          if (InvalidSCCSet.count(C) || &C->getOuterRefSCC() != RC)
            continue;

          do {
            // Check that we didn't miss any update scenario.
            assert(!InvalidSCCSet.count(C) && "Processing an invalid SCC!");
            assert(C->begin() != C->end() && "Cannot have an empty SCC!");
            assert(&C->getOuterRefSCC() == RC &&
                   "Processing an SCC in a different RefSCC!");

            UR.UpdatedRC = nullptr;
            UR.UpdatedC = nullptr;
            PreservedAnalyses PassPA = Pass.run(*C, CGAM, CG, UR);

            // We handle invalidating the CGSCC analysis manager's information
            // for the (potentially updated) SCC here. Note that any other SCCs
            // whose structure has changed should have been invalidated by
            // whatever was updating the call graph. This SCC gets invalidated
            // late as it contains the nodes that were actively being
            // processed.
            PassPA = CGAM.invalidate(*(UR.UpdatedC ? UR.UpdatedC : C),
                                     std::move(PassPA));

            // Then intersect the preserved set so that invalidation of module
            // analyses will eventually occur when the module pass completes.
            PA.intersect(std::move(PassPA));

            // The pass may have restructured the call graph and refined the
            // current SCC and/or RefSCC. We need to update our current SCC and
            // RefSCC pointers to follow these. Also, when the current SCC is
            // refined, re-run the SCC pass over the newly refined SCC in order
            // to observe the most precise SCC model available. This inherently
            // cannot cycle excessively as it only happens when we split SCCs
            // apart, at most converging on a DAG of single nodes.
            // FIXME: If we ever start having RefSCC passes, we'll want to
            // iterate there too.
            RC = UR.UpdatedRC ? UR.UpdatedRC : RC;
            C = UR.UpdatedC ? UR.UpdatedC : C;
            if (DebugLogging && UR.UpdatedC)
              dbgs() << "Re-running SCC passes after a refinement of the "
                        "current SCC: "
                     << *UR.UpdatedC << "\n";
          } while (UR.UpdatedC);

        } while (!CWorklist.empty());
      } while (!RCWorklist.empty());
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

extern template class InnerAnalysisManagerProxy<
    FunctionAnalysisManager, LazyCallGraph::SCC, LazyCallGraph &>;
/// A proxy from a \c FunctionAnalysisManager to an \c SCC.
typedef InnerAnalysisManagerProxy<FunctionAnalysisManager, LazyCallGraph::SCC,
                                  LazyCallGraph &>
    FunctionAnalysisManagerCGSCCProxy;

extern template class OuterAnalysisManagerProxy<CGSCCAnalysisManager, Function>;
/// A proxy from a \c CGSCCAnalysisManager to a \c Function.
typedef OuterAnalysisManagerProxy<CGSCCAnalysisManager, Function>
    CGSCCAnalysisManagerFunctionProxy;

/// Helper to update the call graph after running a function pass.
///
/// Function passes can only mutate the call graph in specific ways. This
/// routine provides a helper that updates the call graph in those ways
/// including returning whether any changes were made and populating a CG
/// update result struct for the overall CGSCC walk.
LazyCallGraph::SCC &updateCGAndAnalysisManagerForFunctionPass(
    LazyCallGraph &G, LazyCallGraph::SCC &C, LazyCallGraph::Node &N,
    CGSCCAnalysisManager &AM, CGSCCUpdateResult &UR, bool DebugLogging = false);

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
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR) {
    // Setup the function analysis manager from its proxy.
    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

    SmallVector<LazyCallGraph::Node *, 4> Nodes;
    for (LazyCallGraph::Node &N : C)
      Nodes.push_back(&N);

    // The SCC may get split while we are optimizing functions due to deleting
    // edges. If this happens, the current SCC can shift, so keep track of
    // a pointer we can overwrite.
    LazyCallGraph::SCC *CurrentC = &C;

    if (DebugLogging)
      dbgs() << "Running function passes across an SCC: " << C << "\n";

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::Node *N : Nodes) {
      // Skip nodes from other SCCs. These may have been split out during
      // processing. We'll eventually visit those SCCs and pick up the nodes
      // there.
      if (CG.lookupSCC(*N) != CurrentC)
        continue;

      PreservedAnalyses PassPA = Pass.run(N->getFunction(), FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here.
      // Also, update the preserved analyses to reflect that once invalidated
      // these can again be preserved.
      PassPA = FAM.invalidate(N->getFunction(), std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));

      // Update the call graph based on this function pass. This may also
      // update the current SCC to point to a smaller, more refined SCC.
      CurrentC = &updateCGAndAnalysisManagerForFunctionPass(
          CG, *CurrentC, *N, AM, UR, DebugLogging);
      assert(CG.lookupSCC(*N) == CurrentC &&
             "Current SCC not updated to the SCC containing the current node!");
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of function analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the function analysis manager
    // incrementally above.
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();

    // We've also ensured that we updated the call graph along the way.
    PA.preserve<LazyCallGraphAnalysis>();

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
