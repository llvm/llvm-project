//===- CGSCCPassManager.cpp - Managing & running CGSCC passes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/CallSite.h"

using namespace llvm;

namespace llvm {

// Explicit instantiations for the core proxy templates.
template class AnalysisManager<LazyCallGraph::SCC, LazyCallGraph &>;
template class PassManager<LazyCallGraph::SCC, CGSCCAnalysisManager,
                           LazyCallGraph &, CGSCCUpdateResult &>;
template class InnerAnalysisManagerProxy<CGSCCAnalysisManager, Module>;
template class OuterAnalysisManagerProxy<ModuleAnalysisManager,
                                         LazyCallGraph::SCC, LazyCallGraph &>;
template class InnerAnalysisManagerProxy<FunctionAnalysisManager,
                                         LazyCallGraph::SCC, LazyCallGraph &>;
template class OuterAnalysisManagerProxy<CGSCCAnalysisManager, Function>;

/// Explicitly specialize the pass manager run method to handle call graph
/// updates.
template <>
PreservedAnalyses
PassManager<LazyCallGraph::SCC, CGSCCAnalysisManager, LazyCallGraph &,
            CGSCCUpdateResult &>::run(LazyCallGraph::SCC &InitialC,
                                      CGSCCAnalysisManager &AM,
                                      LazyCallGraph &G, CGSCCUpdateResult &UR) {
  PreservedAnalyses PA = PreservedAnalyses::all();

  if (DebugLogging)
    dbgs() << "Starting CGSCC pass manager run.\n";

  // The SCC may be refined while we are running passes over it, so set up
  // a pointer that we can update.
  LazyCallGraph::SCC *C = &InitialC;

  for (auto &Pass : Passes) {
    if (DebugLogging)
      dbgs() << "Running pass: " << Pass->name() << " on " << *C << "\n";

    PreservedAnalyses PassPA = Pass->run(*C, AM, G, UR);

    // Update the SCC if necessary.
    C = UR.UpdatedC ? UR.UpdatedC : C;

    // Check that we didn't miss any update scenario.
    assert(!UR.InvalidatedSCCs.count(C) && "Processing an invalid SCC!");
    assert(C->begin() != C->end() && "Cannot have an empty SCC!");

    // Update the analysis manager as each pass runs and potentially
    // invalidates analyses. We also update the preserved set of analyses
    // based on what analyses we have already handled the invalidation for
    // here and don't need to invalidate when finished.
    PassPA = AM.invalidate(*C, std::move(PassPA));

    // Finally, we intersect the final preserved analyses to compute the
    // aggregate preserved set for this pass manager.
    PA.intersect(std::move(PassPA));

    // FIXME: Historically, the pass managers all called the LLVM context's
    // yield function here. We don't have a generic way to acquire the
    // context and it isn't yet clear what the right pattern is for yielding
    // in the new pass manager so it is currently omitted.
    // ...getContext().yield();
  }

  if (DebugLogging)
    dbgs() << "Finished CGSCC pass manager run.\n";

  return PA;
}

} // End llvm namespace

namespace {
/// Helper function to update both the \c CGSCCAnalysisManager \p AM and the \c
/// CGSCCPassManager's \c CGSCCUpdateResult \p UR based on a range of newly
/// added SCCs.
///
/// The range of new SCCs must be in postorder already. The SCC they were split
/// out of must be provided as \p C. The current node being mutated and
/// triggering updates must be passed as \p N.
///
/// This function returns the SCC containing \p N. This will be either \p C if
/// no new SCCs have been split out, or it will be the new SCC containing \p N.
template <typename SCCRangeT>
LazyCallGraph::SCC *
incorporateNewSCCRange(const SCCRangeT &NewSCCRange, LazyCallGraph &G,
                       LazyCallGraph::Node &N, LazyCallGraph::SCC *C,
                       CGSCCAnalysisManager &AM, CGSCCUpdateResult &UR,
                       bool DebugLogging = false) {
  typedef LazyCallGraph::SCC SCC;

  if (NewSCCRange.begin() == NewSCCRange.end())
    return C;

  // Invalidate the analyses of the current SCC and add it to the worklist since
  // it has changed its shape.
  AM.invalidate(*C, PreservedAnalyses::none());
  UR.CWorklist.insert(C);
  if (DebugLogging)
    dbgs() << "Enqueuing the existing SCC in the worklist:" << *C << "\n";

  SCC *OldC = C;
  (void)OldC;

  // Update the current SCC. Note that if we have new SCCs, this must actually
  // change the SCC.
  assert(C != &*NewSCCRange.begin() &&
         "Cannot insert new SCCs without changing current SCC!");
  C = &*NewSCCRange.begin();
  assert(G.lookupSCC(N) == C && "Failed to update current SCC!");

  for (SCC &NewC :
       reverse(make_range(std::next(NewSCCRange.begin()), NewSCCRange.end()))) {
    assert(C != &NewC && "No need to re-visit the current SCC!");
    assert(OldC != &NewC && "Already handled the original SCC!");
    UR.CWorklist.insert(&NewC);
    if (DebugLogging)
      dbgs() << "Enqueuing a newly formed SCC:" << NewC << "\n";
  }
  return C;
}
}

LazyCallGraph::SCC &llvm::updateCGAndAnalysisManagerForFunctionPass(
    LazyCallGraph &G, LazyCallGraph::SCC &InitialC, LazyCallGraph::Node &N,
    CGSCCAnalysisManager &AM, CGSCCUpdateResult &UR, bool DebugLogging) {
  typedef LazyCallGraph::Node Node;
  typedef LazyCallGraph::Edge Edge;
  typedef LazyCallGraph::SCC SCC;
  typedef LazyCallGraph::RefSCC RefSCC;

  RefSCC &InitialRC = InitialC.getOuterRefSCC();
  SCC *C = &InitialC;
  RefSCC *RC = &InitialRC;
  Function &F = N.getFunction();

  // Walk the function body and build up the set of retained, promoted, and
  // demoted edges.
  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Constant *, 16> Visited;
  SmallPtrSet<Function *, 16> RetainedEdges;
  SmallSetVector<Function *, 4> PromotedRefTargets;
  SmallSetVector<Function *, 4> DemotedCallTargets;
  // First walk the function and handle all called functions. We do this first
  // because if there is a single call edge, whether there are ref edges is
  // irrelevant.
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto CS = CallSite(&I))
        if (Function *Callee = CS.getCalledFunction())
          if (Visited.insert(Callee).second && !Callee->isDeclaration()) {
            const Edge *E = N.lookup(*Callee);
            // FIXME: We should really handle adding new calls. While it will
            // make downstream usage more complex, there is no fundamental
            // limitation and it will allow passes within the CGSCC to be a bit
            // more flexible in what transforms they can do. Until then, we
            // verify that new calls haven't been introduced.
            assert(E && "No function transformations should introduce *new* "
                        "call edges! Any new calls should be modeled as "
                        "promoted existing ref edges!");
            RetainedEdges.insert(Callee);
            if (!E->isCall())
              PromotedRefTargets.insert(Callee);
          }

  // Now walk all references.
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      for (Value *Op : I.operand_values())
        if (Constant *C = dyn_cast<Constant>(Op))
          if (Visited.insert(C).second)
            Worklist.push_back(C);

      LazyCallGraph::visitReferences(Worklist, Visited, [&](Function &Referee) {
        // Skip declarations.
        if (Referee.isDeclaration())
          return;

        const Edge *E = N.lookup(Referee);
        // FIXME: Similarly to new calls, we also currently preclude
        // introducing new references. See above for details.
        assert(E && "No function transformations should introduce *new* ref "
                    "edges! Any new ref edges would require IPO which "
                    "function passes aren't allowed to do!");
        RetainedEdges.insert(&Referee);
        if (E->isCall())
          DemotedCallTargets.insert(&Referee);
      });
    }

  // First remove all of the edges that are no longer present in this function.
  // We have to build a list of dead targets first and then remove them as the
  // data structures will all be invalidated by removing them.
  SmallVector<PointerIntPair<Node *, 1, Edge::Kind>, 4> DeadTargets;
  for (Edge &E : N)
    if (!RetainedEdges.count(&E.getFunction()))
      DeadTargets.push_back({E.getNode(), E.getKind()});
  for (auto DeadTarget : DeadTargets) {
    Node &TargetN = *DeadTarget.getPointer();
    bool IsCall = DeadTarget.getInt() == Edge::Call;
    SCC &TargetC = *G.lookupSCC(TargetN);
    RefSCC &TargetRC = TargetC.getOuterRefSCC();

    if (&TargetRC != RC) {
      RC->removeOutgoingEdge(N, TargetN);
      if (DebugLogging)
        dbgs() << "Deleting outgoing edge from '" << N << "' to '" << TargetN
               << "'\n";
      continue;
    }
    if (DebugLogging)
      dbgs() << "Deleting internal " << (IsCall ? "call" : "ref")
             << " edge from '" << N << "' to '" << TargetN << "'\n";

    if (IsCall)
      C = incorporateNewSCCRange(RC->switchInternalEdgeToRef(N, TargetN), G, N,
                                 C, AM, UR, DebugLogging);

    auto NewRefSCCs = RC->removeInternalRefEdge(N, TargetN);
    if (!NewRefSCCs.empty()) {
      // Note that we don't bother to invalidate analyses as ref-edge
      // connectivity is not really observable in any way and is intended
      // exclusively to be used for ordering of transforms rather than for
      // analysis conclusions.

      // The RC worklist is in reverse postorder, so we first enqueue the
      // current RefSCC as it will remain the parent of all split RefSCCs, then
      // we enqueue the new ones in RPO except for the one which contains the
      // source node as that is the "bottom" we will continue processing in the
      // bottom-up walk.
      UR.RCWorklist.insert(RC);
      if (DebugLogging)
        dbgs() << "Enqueuing the existing RefSCC in the update worklist: "
               << *RC << "\n";
      // Update the RC to the "bottom".
      assert(G.lookupSCC(N) == C && "Changed the SCC when splitting RefSCCs!");
      RC = &C->getOuterRefSCC();
      assert(G.lookupRefSCC(N) == RC && "Failed to update current RefSCC!");
      for (RefSCC *NewRC : reverse(NewRefSCCs))
        if (NewRC != RC) {
          UR.RCWorklist.insert(NewRC);
          if (DebugLogging)
            dbgs() << "Enqueuing a new RefSCC in the update worklist: "
                   << *NewRC << "\n";
        }
    }
  }

  // Next demote all the call edges that are now ref edges. This helps make
  // the SCCs small which should minimize the work below as we don't want to
  // form cycles that this would break.
  for (Function *RefTarget : DemotedCallTargets) {
    Node &TargetN = *G.lookup(*RefTarget);
    SCC &TargetC = *G.lookupSCC(TargetN);
    RefSCC &TargetRC = TargetC.getOuterRefSCC();

    // The easy case is when the target RefSCC is not this RefSCC. This is
    // only supported when the target RefSCC is a child of this RefSCC.
    if (&TargetRC != RC) {
      assert(RC->isAncestorOf(TargetRC) &&
             "Cannot potentially form RefSCC cycles here!");
      RC->switchOutgoingEdgeToRef(N, TargetN);
      if (DebugLogging)
        dbgs() << "Switch outgoing call edge to a ref edge from '" << N
               << "' to '" << TargetN << "'\n";
      continue;
    }

    // Otherwise we are switching an internal call edge to a ref edge. This
    // may split up some SCCs.
    C = incorporateNewSCCRange(RC->switchInternalEdgeToRef(N, TargetN), G, N, C,
                               AM, UR, DebugLogging);
  }

  // Now promote ref edges into call edges.
  for (Function *CallTarget : PromotedRefTargets) {
    Node &TargetN = *G.lookup(*CallTarget);
    SCC &TargetC = *G.lookupSCC(TargetN);
    RefSCC &TargetRC = TargetC.getOuterRefSCC();

    // The easy case is when the target RefSCC is not this RefSCC. This is
    // only supported when the target RefSCC is a child of this RefSCC.
    if (&TargetRC != RC) {
      assert(RC->isAncestorOf(TargetRC) &&
             "Cannot potentially form RefSCC cycles here!");
      RC->switchOutgoingEdgeToCall(N, TargetN);
      if (DebugLogging)
        dbgs() << "Switch outgoing ref edge to a call edge from '" << N
               << "' to '" << TargetN << "'\n";
      continue;
    }
    if (DebugLogging)
      dbgs() << "Switch an internal ref edge to a call edge from '" << N
             << "' to '" << TargetN << "'\n";

    // Otherwise we are switching an internal ref edge to a call edge. This
    // may merge away some SCCs, and we add those to the UpdateResult. We also
    // need to make sure to update the worklist in the event SCCs have moved
    // before the current one in the post-order sequence.
    auto InitialSCCIndex = RC->find(*C) - RC->begin();
    auto InvalidatedSCCs = RC->switchInternalEdgeToCall(N, TargetN);
    if (!InvalidatedSCCs.empty()) {
      C = &TargetC;
      assert(G.lookupSCC(N) == C && "Failed to update current SCC!");

      // Any analyses cached for this SCC are no longer precise as the shape
      // has changed by introducing this cycle.
      AM.invalidate(*C, PreservedAnalyses::none());

      for (SCC *InvalidatedC : InvalidatedSCCs) {
        assert(InvalidatedC != C && "Cannot invalidate the current SCC!");
        UR.InvalidatedSCCs.insert(InvalidatedC);

        // Also clear any cached analyses for the SCCs that are dead. This
        // isn't really necessary for correctness but can release memory.
        AM.clear(*InvalidatedC);
      }
    }
    auto NewSCCIndex = RC->find(*C) - RC->begin();
    if (InitialSCCIndex < NewSCCIndex) {
      // Put our current SCC back onto the worklist as we'll visit other SCCs
      // that are now definitively ordered prior to the current one in the
      // post-order sequence, and may end up observing more precise context to
      // optimize the current SCC.
      UR.CWorklist.insert(C);
      if (DebugLogging)
        dbgs() << "Enqueuing the existing SCC in the worklist: " << *C << "\n";
      // Enqueue in reverse order as we pop off the back of the worklist.
      for (SCC &MovedC : reverse(make_range(RC->begin() + InitialSCCIndex,
                                            RC->begin() + NewSCCIndex))) {
        UR.CWorklist.insert(&MovedC);
        if (DebugLogging)
          dbgs() << "Enqueuing a newly earlier in post-order SCC: " << MovedC
                 << "\n";
      }
    }
  }

  assert(!UR.InvalidatedSCCs.count(C) && "Invalidated the current SCC!");
  assert(!UR.InvalidatedRefSCCs.count(RC) && "Invalidated the current RefSCC!");
  assert(&C->getOuterRefSCC() == RC && "Current SCC not in current RefSCC!");

  // Record the current RefSCC and SCC for higher layers of the CGSCC pass
  // manager now that all the updates have been applied.
  if (RC != &InitialRC)
    UR.UpdatedRC = RC;
  if (C != &InitialC)
    UR.UpdatedC = C;

  return *C;
}
