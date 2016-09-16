//===- LazyCallGraph.cpp - Analysis of a Module's call graph --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

using namespace llvm;

#define DEBUG_TYPE "lcg"

static void addEdge(SmallVectorImpl<LazyCallGraph::Edge> &Edges,
                    DenseMap<Function *, int> &EdgeIndexMap, Function &F,
                    LazyCallGraph::Edge::Kind EK) {
  // Note that we consider *any* function with a definition to be a viable
  // edge. Even if the function's definition is subject to replacement by
  // some other module (say, a weak definition) there may still be
  // optimizations which essentially speculate based on the definition and
  // a way to check that the specific definition is in fact the one being
  // used. For example, this could be done by moving the weak definition to
  // a strong (internal) definition and making the weak definition be an
  // alias. Then a test of the address of the weak function against the new
  // strong definition's address would be an effective way to determine the
  // safety of optimizing a direct call edge.
  if (!F.isDeclaration() &&
      EdgeIndexMap.insert({&F, Edges.size()}).second) {
    DEBUG(dbgs() << "    Added callable function: " << F.getName() << "\n");
    Edges.emplace_back(LazyCallGraph::Edge(F, EK));
  }
}

LazyCallGraph::Node::Node(LazyCallGraph &G, Function &F)
    : G(&G), F(F), DFSNumber(0), LowLink(0) {
  DEBUG(dbgs() << "  Adding functions called by '" << F.getName()
               << "' to the graph.\n");

  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Function *, 4> Callees;
  SmallPtrSet<Constant *, 16> Visited;

  // Find all the potential call graph edges in this function. We track both
  // actual call edges and indirect references to functions. The direct calls
  // are trivially added, but to accumulate the latter we walk the instructions
  // and add every operand which is a constant to the worklist to process
  // afterward.
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      if (auto CS = CallSite(&I))
        if (Function *Callee = CS.getCalledFunction())
          if (Callees.insert(Callee).second) {
            Visited.insert(Callee);
            addEdge(Edges, EdgeIndexMap, *Callee, LazyCallGraph::Edge::Call);
          }

      for (Value *Op : I.operand_values())
        if (Constant *C = dyn_cast<Constant>(Op))
          if (Visited.insert(C).second)
            Worklist.push_back(C);
    }

  // We've collected all the constant (and thus potentially function or
  // function containing) operands to all of the instructions in the function.
  // Process them (recursively) collecting every function found.
  visitReferences(Worklist, Visited, [&](Function &F) {
    addEdge(Edges, EdgeIndexMap, F, LazyCallGraph::Edge::Ref);
  });
}

void LazyCallGraph::Node::insertEdgeInternal(Function &Target, Edge::Kind EK) {
  if (Node *N = G->lookup(Target))
    return insertEdgeInternal(*N, EK);

  EdgeIndexMap.insert({&Target, Edges.size()});
  Edges.emplace_back(Target, EK);
}

void LazyCallGraph::Node::insertEdgeInternal(Node &TargetN, Edge::Kind EK) {
  EdgeIndexMap.insert({&TargetN.getFunction(), Edges.size()});
  Edges.emplace_back(TargetN, EK);
}

void LazyCallGraph::Node::setEdgeKind(Function &TargetF, Edge::Kind EK) {
  Edges[EdgeIndexMap.find(&TargetF)->second].setKind(EK);
}

void LazyCallGraph::Node::removeEdgeInternal(Function &Target) {
  auto IndexMapI = EdgeIndexMap.find(&Target);
  assert(IndexMapI != EdgeIndexMap.end() &&
         "Target not in the edge set for this caller?");

  Edges[IndexMapI->second] = Edge();
  EdgeIndexMap.erase(IndexMapI);
}

void LazyCallGraph::Node::dump() const {
  dbgs() << *this << '\n';
}

LazyCallGraph::LazyCallGraph(Module &M) : NextDFSNumber(0) {
  DEBUG(dbgs() << "Building CG for module: " << M.getModuleIdentifier()
               << "\n");
  for (Function &F : M)
    if (!F.isDeclaration() && !F.hasLocalLinkage())
      if (EntryIndexMap.insert({&F, EntryEdges.size()}).second) {
        DEBUG(dbgs() << "  Adding '" << F.getName()
                     << "' to entry set of the graph.\n");
        EntryEdges.emplace_back(F, Edge::Ref);
      }

  // Now add entry nodes for functions reachable via initializers to globals.
  SmallVector<Constant *, 16> Worklist;
  SmallPtrSet<Constant *, 16> Visited;
  for (GlobalVariable &GV : M.globals())
    if (GV.hasInitializer())
      if (Visited.insert(GV.getInitializer()).second)
        Worklist.push_back(GV.getInitializer());

  DEBUG(dbgs() << "  Adding functions referenced by global initializers to the "
                  "entry set.\n");
  visitReferences(Worklist, Visited, [&](Function &F) {
    addEdge(EntryEdges, EntryIndexMap, F, LazyCallGraph::Edge::Ref);
  });

  for (const Edge &E : EntryEdges)
    RefSCCEntryNodes.push_back(&E.getFunction());
}

LazyCallGraph::LazyCallGraph(LazyCallGraph &&G)
    : BPA(std::move(G.BPA)), NodeMap(std::move(G.NodeMap)),
      EntryEdges(std::move(G.EntryEdges)),
      EntryIndexMap(std::move(G.EntryIndexMap)), SCCBPA(std::move(G.SCCBPA)),
      SCCMap(std::move(G.SCCMap)), LeafRefSCCs(std::move(G.LeafRefSCCs)),
      DFSStack(std::move(G.DFSStack)),
      RefSCCEntryNodes(std::move(G.RefSCCEntryNodes)),
      NextDFSNumber(G.NextDFSNumber) {
  updateGraphPtrs();
}

LazyCallGraph &LazyCallGraph::operator=(LazyCallGraph &&G) {
  BPA = std::move(G.BPA);
  NodeMap = std::move(G.NodeMap);
  EntryEdges = std::move(G.EntryEdges);
  EntryIndexMap = std::move(G.EntryIndexMap);
  SCCBPA = std::move(G.SCCBPA);
  SCCMap = std::move(G.SCCMap);
  LeafRefSCCs = std::move(G.LeafRefSCCs);
  DFSStack = std::move(G.DFSStack);
  RefSCCEntryNodes = std::move(G.RefSCCEntryNodes);
  NextDFSNumber = G.NextDFSNumber;
  updateGraphPtrs();
  return *this;
}

void LazyCallGraph::SCC::dump() const {
  dbgs() << *this << '\n';
}

#ifndef NDEBUG
void LazyCallGraph::SCC::verify() {
  assert(OuterRefSCC && "Can't have a null RefSCC!");
  assert(!Nodes.empty() && "Can't have an empty SCC!");

  for (Node *N : Nodes) {
    assert(N && "Can't have a null node!");
    assert(OuterRefSCC->G->lookupSCC(*N) == this &&
           "Node does not map to this SCC!");
    assert(N->DFSNumber == -1 &&
           "Must set DFS numbers to -1 when adding a node to an SCC!");
    assert(N->LowLink == -1 &&
           "Must set low link to -1 when adding a node to an SCC!");
    for (Edge &E : *N)
      assert(E.getNode() && "Can't have an edge to a raw function!");
  }
}
#endif

LazyCallGraph::RefSCC::RefSCC(LazyCallGraph &G) : G(&G) {}

void LazyCallGraph::RefSCC::dump() const {
  dbgs() << *this << '\n';
}

#ifndef NDEBUG
void LazyCallGraph::RefSCC::verify() {
  assert(G && "Can't have a null graph!");
  assert(!SCCs.empty() && "Can't have an empty SCC!");

  // Verify basic properties of the SCCs.
  SmallPtrSet<SCC *, 4> SCCSet;
  for (SCC *C : SCCs) {
    assert(C && "Can't have a null SCC!");
    C->verify();
    assert(&C->getOuterRefSCC() == this &&
           "SCC doesn't think it is inside this RefSCC!");
    bool Inserted = SCCSet.insert(C).second;
    assert(Inserted && "Found a duplicate SCC!");
  }

  // Check that our indices map correctly.
  for (auto &SCCIndexPair : SCCIndices) {
    SCC *C = SCCIndexPair.first;
    int i = SCCIndexPair.second;
    assert(C && "Can't have a null SCC in the indices!");
    assert(SCCSet.count(C) && "Found an index for an SCC not in the RefSCC!");
    assert(SCCs[i] == C && "Index doesn't point to SCC!");
  }

  // Check that the SCCs are in fact in post-order.
  for (int i = 0, Size = SCCs.size(); i < Size; ++i) {
    SCC &SourceSCC = *SCCs[i];
    for (Node &N : SourceSCC)
      for (Edge &E : N) {
        if (!E.isCall())
          continue;
        SCC &TargetSCC = *G->lookupSCC(*E.getNode());
        if (&TargetSCC.getOuterRefSCC() == this) {
          assert(SCCIndices.find(&TargetSCC)->second <= i &&
                 "Edge between SCCs violates post-order relationship.");
          continue;
        }
        assert(TargetSCC.getOuterRefSCC().Parents.count(this) &&
               "Edge to a RefSCC missing us in its parent set.");
      }
  }
}
#endif

bool LazyCallGraph::RefSCC::isDescendantOf(const RefSCC &C) const {
  // Walk up the parents of this SCC and verify that we eventually find C.
  SmallVector<const RefSCC *, 4> AncestorWorklist;
  AncestorWorklist.push_back(this);
  do {
    const RefSCC *AncestorC = AncestorWorklist.pop_back_val();
    if (AncestorC->isChildOf(C))
      return true;
    for (const RefSCC *ParentC : AncestorC->Parents)
      AncestorWorklist.push_back(ParentC);
  } while (!AncestorWorklist.empty());

  return false;
}

/// Generic helper that updates a postorder sequence of SCCs for a potentially
/// cycle-introducing edge insertion.
///
/// A postorder sequence of SCCs of a directed graph has one fundamental
/// property: all deges in the DAG of SCCs point "up" the sequence. That is,
/// all edges in the SCC DAG point to prior SCCs in the sequence.
///
/// This routine both updates a postorder sequence and uses that sequence to
/// compute the set of SCCs connected into a cycle. It should only be called to
/// insert a "downward" edge which will require changing the sequence to
/// restore it to a postorder.
///
/// When inserting an edge from an earlier SCC to a later SCC in some postorder
/// sequence, all of the SCCs which may be impacted are in the closed range of
/// those two within the postorder sequence. The algorithm used here to restore
/// the state is as follows:
///
/// 1) Starting from the source SCC, construct a set of SCCs which reach the
///    source SCC consisting of just the source SCC. Then scan toward the
///    target SCC in postorder and for each SCC, if it has an edge to an SCC
///    in the set, add it to the set. Otherwise, the source SCC is not
///    a successor, move it in the postorder sequence to immediately before
///    the source SCC, shifting the source SCC and all SCCs in the set one
///    position toward the target SCC. Stop scanning after processing the
///    target SCC.
/// 2) If the source SCC is now past the target SCC in the postorder sequence,
///    and thus the new edge will flow toward the start, we are done.
/// 3) Otherwise, starting from the target SCC, walk all edges which reach an
///    SCC between the source and the target, and add them to the set of
///    connected SCCs, then recurse through them. Once a complete set of the
///    SCCs the target connects to is known, hoist the remaining SCCs between
///    the source and the target to be above the target. Note that there is no
///    need to process the source SCC, it is already known to connect.
/// 4) At this point, all of the SCCs in the closed range between the source
///    SCC and the target SCC in the postorder sequence are connected,
///    including the target SCC and the source SCC. Inserting the edge from
///    the source SCC to the target SCC will form a cycle out of precisely
///    these SCCs. Thus we can merge all of the SCCs in this closed range into
///    a single SCC.
///
/// This process has various important properties:
/// - Only mutates the SCCs when adding the edge actually changes the SCC
///   structure.
/// - Never mutates SCCs which are unaffected by the change.
/// - Updates the postorder sequence to correctly satisfy the postorder
///   constraint after the edge is inserted.
/// - Only reorders SCCs in the closed postorder sequence from the source to
///   the target, so easy to bound how much has changed even in the ordering.
/// - Big-O is the number of edges in the closed postorder range of SCCs from
///   source to target.
///
/// This helper routine, in addition to updating the postorder sequence itself
/// will also update a map from SCCs to indices within that sequecne.
///
/// The sequence and the map must operate on pointers to the SCC type.
///
/// Two callbacks must be provided. The first computes the subset of SCCs in
/// the postorder closed range from the source to the target which connect to
/// the source SCC via some (transitive) set of edges. The second computes the
/// subset of the same range which the target SCC connects to via some
/// (transitive) set of edges. Both callbacks should populate the set argument
/// provided.
template <typename SCCT, typename PostorderSequenceT, typename SCCIndexMapT,
          typename ComputeSourceConnectedSetCallableT,
          typename ComputeTargetConnectedSetCallableT>
static iterator_range<typename PostorderSequenceT::iterator>
updatePostorderSequenceForEdgeInsertion(
    SCCT &SourceSCC, SCCT &TargetSCC, PostorderSequenceT &SCCs,
    SCCIndexMapT &SCCIndices,
    ComputeSourceConnectedSetCallableT ComputeSourceConnectedSet,
    ComputeTargetConnectedSetCallableT ComputeTargetConnectedSet) {
  int SourceIdx = SCCIndices[&SourceSCC];
  int TargetIdx = SCCIndices[&TargetSCC];
  assert(SourceIdx < TargetIdx && "Cannot have equal indices here!");

  SmallPtrSet<SCCT *, 4> ConnectedSet;

  // Compute the SCCs which (transitively) reach the source.
  ComputeSourceConnectedSet(ConnectedSet);

  // Partition the SCCs in this part of the port-order sequence so only SCCs
  // connecting to the source remain between it and the target. This is
  // a benign partition as it preserves postorder.
  auto SourceI = std::stable_partition(
      SCCs.begin() + SourceIdx, SCCs.begin() + TargetIdx + 1,
      [&ConnectedSet](SCCT *C) { return !ConnectedSet.count(C); });
  for (int i = SourceIdx, e = TargetIdx + 1; i < e; ++i)
    SCCIndices.find(SCCs[i])->second = i;

  // If the target doesn't connect to the source, then we've corrected the
  // post-order and there are no cycles formed.
  if (!ConnectedSet.count(&TargetSCC)) {
    assert(SourceI > (SCCs.begin() + SourceIdx) &&
           "Must have moved the source to fix the post-order.");
    assert(*std::prev(SourceI) == &TargetSCC &&
           "Last SCC to move should have bene the target.");

    // Return an empty range at the target SCC indicating there is nothing to
    // merge.
    return make_range(std::prev(SourceI), std::prev(SourceI));
  }

  assert(SCCs[TargetIdx] == &TargetSCC &&
         "Should not have moved target if connected!");
  SourceIdx = SourceI - SCCs.begin();
  assert(SCCs[SourceIdx] == &SourceSCC &&
         "Bad updated index computation for the source SCC!");


  // See whether there are any remaining intervening SCCs between the source
  // and target. If so we need to make sure they all are reachable form the
  // target.
  if (SourceIdx + 1 < TargetIdx) {
    ConnectedSet.clear();
    ComputeTargetConnectedSet(ConnectedSet);

    // Partition SCCs so that only SCCs reached from the target remain between
    // the source and the target. This preserves postorder.
    auto TargetI = std::stable_partition(
        SCCs.begin() + SourceIdx + 1, SCCs.begin() + TargetIdx + 1,
        [&ConnectedSet](SCCT *C) { return ConnectedSet.count(C); });
    for (int i = SourceIdx + 1, e = TargetIdx + 1; i < e; ++i)
      SCCIndices.find(SCCs[i])->second = i;
    TargetIdx = std::prev(TargetI) - SCCs.begin();
    assert(SCCs[TargetIdx] == &TargetSCC &&
           "Should always end with the target!");
  }

  // At this point, we know that connecting source to target forms a cycle
  // because target connects back to source, and we know that all of the SCCs
  // between the source and target in the postorder sequence participate in that
  // cycle.
  return make_range(SCCs.begin() + SourceIdx, SCCs.begin() + TargetIdx);
}

SmallVector<LazyCallGraph::SCC *, 1>
LazyCallGraph::RefSCC::switchInternalEdgeToCall(Node &SourceN, Node &TargetN) {
  assert(!SourceN[TargetN].isCall() && "Must start with a ref edge!");
  SmallVector<SCC *, 1> DeletedSCCs;

#ifndef NDEBUG
  // In a debug build, verify the RefSCC is valid to start with and when this
  // routine finishes.
  verify();
  auto VerifyOnExit = make_scope_exit([&]() { verify(); });
#endif

  SCC &SourceSCC = *G->lookupSCC(SourceN);
  SCC &TargetSCC = *G->lookupSCC(TargetN);

  // If the two nodes are already part of the same SCC, we're also done as
  // we've just added more connectivity.
  if (&SourceSCC == &TargetSCC) {
    SourceN.setEdgeKind(TargetN.getFunction(), Edge::Call);
    return DeletedSCCs;
  }

  // At this point we leverage the postorder list of SCCs to detect when the
  // insertion of an edge changes the SCC structure in any way.
  //
  // First and foremost, we can eliminate the need for any changes when the
  // edge is toward the beginning of the postorder sequence because all edges
  // flow in that direction already. Thus adding a new one cannot form a cycle.
  int SourceIdx = SCCIndices[&SourceSCC];
  int TargetIdx = SCCIndices[&TargetSCC];
  if (TargetIdx < SourceIdx) {
    SourceN.setEdgeKind(TargetN.getFunction(), Edge::Call);
    return DeletedSCCs;
  }

  // Compute the SCCs which (transitively) reach the source.
  auto ComputeSourceConnectedSet = [&](SmallPtrSetImpl<SCC *> &ConnectedSet) {
#ifndef NDEBUG
    // Check that the RefSCC is still valid before computing this as the
    // results will be nonsensical of we've broken its invariants.
    verify();
#endif
    ConnectedSet.insert(&SourceSCC);
    auto IsConnected = [&](SCC &C) {
      for (Node &N : C)
        for (Edge &E : N.calls()) {
          assert(E.getNode() && "Must have formed a node within an SCC!");
          if (ConnectedSet.count(G->lookupSCC(*E.getNode())))
            return true;
        }

      return false;
    };

    for (SCC *C :
         make_range(SCCs.begin() + SourceIdx + 1, SCCs.begin() + TargetIdx + 1))
      if (IsConnected(*C))
        ConnectedSet.insert(C);
  };

  // Use a normal worklist to find which SCCs the target connects to. We still
  // bound the search based on the range in the postorder list we care about,
  // but because this is forward connectivity we just "recurse" through the
  // edges.
  auto ComputeTargetConnectedSet = [&](SmallPtrSetImpl<SCC *> &ConnectedSet) {
#ifndef NDEBUG
    // Check that the RefSCC is still valid before computing this as the
    // results will be nonsensical of we've broken its invariants.
    verify();
#endif
    ConnectedSet.insert(&TargetSCC);
    SmallVector<SCC *, 4> Worklist;
    Worklist.push_back(&TargetSCC);
    do {
      SCC &C = *Worklist.pop_back_val();
      for (Node &N : C)
        for (Edge &E : N) {
          assert(E.getNode() && "Must have formed a node within an SCC!");
          if (!E.isCall())
            continue;
          SCC &EdgeC = *G->lookupSCC(*E.getNode());
          if (&EdgeC.getOuterRefSCC() != this)
            // Not in this RefSCC...
            continue;
          if (SCCIndices.find(&EdgeC)->second <= SourceIdx)
            // Not in the postorder sequence between source and target.
            continue;

          if (ConnectedSet.insert(&EdgeC).second)
            Worklist.push_back(&EdgeC);
        }
    } while (!Worklist.empty());
  };

  // Use a generic helper to update the postorder sequence of SCCs and return
  // a range of any SCCs connected into a cycle by inserting this edge. This
  // routine will also take care of updating the indices into the postorder
  // sequence.
  auto MergeRange = updatePostorderSequenceForEdgeInsertion(
      SourceSCC, TargetSCC, SCCs, SCCIndices, ComputeSourceConnectedSet,
      ComputeTargetConnectedSet);

  // If the merge range is empty, then adding the edge didn't actually form any
  // new cycles. We're done.
  if (MergeRange.begin() == MergeRange.end()) {
    // Now that the SCC structure is finalized, flip the kind to call.
    SourceN.setEdgeKind(TargetN.getFunction(), Edge::Call);
    return DeletedSCCs;
  }

#ifndef NDEBUG
  // Before merging, check that the RefSCC remains valid after all the
  // postorder updates.
  verify();
#endif

  // Otherwise we need to merge all of the SCCs in the cycle into a single
  // result SCC.
  //
  // NB: We merge into the target because all of these functions were already
  // reachable from the target, meaning any SCC-wide properties deduced about it
  // other than the set of functions within it will not have changed.
  for (SCC *C : MergeRange) {
    assert(C != &TargetSCC &&
           "We merge *into* the target and shouldn't process it here!");
    SCCIndices.erase(C);
    TargetSCC.Nodes.append(C->Nodes.begin(), C->Nodes.end());
    for (Node *N : C->Nodes)
      G->SCCMap[N] = &TargetSCC;
    C->clear();
    DeletedSCCs.push_back(C);
  }

  // Erase the merged SCCs from the list and update the indices of the
  // remaining SCCs.
  int IndexOffset = MergeRange.end() - MergeRange.begin();
  auto EraseEnd = SCCs.erase(MergeRange.begin(), MergeRange.end());
  for (SCC *C : make_range(EraseEnd, SCCs.end()))
    SCCIndices[C] -= IndexOffset;

  // Now that the SCC structure is finalized, flip the kind to call.
  SourceN.setEdgeKind(TargetN.getFunction(), Edge::Call);

  // And we're done!
  return DeletedSCCs;
}

iterator_range<LazyCallGraph::RefSCC::iterator>
LazyCallGraph::RefSCC::switchInternalEdgeToRef(Node &SourceN, Node &TargetN) {
  assert(SourceN[TargetN].isCall() && "Must start with a call edge!");

#ifndef NDEBUG
  // In a debug build, verify the RefSCC is valid to start with and when this
  // routine finishes.
  verify();
  auto VerifyOnExit = make_scope_exit([&]() { verify(); });
#endif

  SCC &SourceSCC = *G->lookupSCC(SourceN);
  SCC &TargetSCC = *G->lookupSCC(TargetN);

  assert(&SourceSCC.getOuterRefSCC() == this &&
         "Source must be in this RefSCC.");
  assert(&TargetSCC.getOuterRefSCC() == this &&
         "Target must be in this RefSCC.");

  // Set the edge kind.
  SourceN.setEdgeKind(TargetN.getFunction(), Edge::Ref);

  // If this call edge is just connecting two separate SCCs within this RefSCC,
  // there is nothing to do.
  if (&SourceSCC != &TargetSCC)
    return make_range(SCCs.end(), SCCs.end());

  // Otherwise we are removing a call edge from a single SCC. This may break
  // the cycle. In order to compute the new set of SCCs, we need to do a small
  // DFS over the nodes within the SCC to form any sub-cycles that remain as
  // distinct SCCs and compute a postorder over the resulting SCCs.
  //
  // However, we specially handle the target node. The target node is known to
  // reach all other nodes in the original SCC by definition. This means that
  // we want the old SCC to be replaced with an SCC contaning that node as it
  // will be the root of whatever SCC DAG results from the DFS. Assumptions
  // about an SCC such as the set of functions called will continue to hold,
  // etc.

  SCC &OldSCC = TargetSCC;
  SmallVector<std::pair<Node *, call_edge_iterator>, 16> DFSStack;
  SmallVector<Node *, 16> PendingSCCStack;
  SmallVector<SCC *, 4> NewSCCs;

  // Prepare the nodes for a fresh DFS.
  SmallVector<Node *, 16> Worklist;
  Worklist.swap(OldSCC.Nodes);
  for (Node *N : Worklist) {
    N->DFSNumber = N->LowLink = 0;
    G->SCCMap.erase(N);
  }

  // Force the target node to be in the old SCC. This also enables us to take
  // a very significant short-cut in the standard Tarjan walk to re-form SCCs
  // below: whenever we build an edge that reaches the target node, we know
  // that the target node eventually connects back to all other nodes in our
  // walk. As a consequence, we can detect and handle participants in that
  // cycle without walking all the edges that form this connection, and instead
  // by relying on the fundamental guarantee coming into this operation (all
  // nodes are reachable from the target due to previously forming an SCC).
  TargetN.DFSNumber = TargetN.LowLink = -1;
  OldSCC.Nodes.push_back(&TargetN);
  G->SCCMap[&TargetN] = &OldSCC;

  // Scan down the stack and DFS across the call edges.
  for (Node *RootN : Worklist) {
    assert(DFSStack.empty() &&
           "Cannot begin a new root with a non-empty DFS stack!");
    assert(PendingSCCStack.empty() &&
           "Cannot begin a new root with pending nodes for an SCC!");

    // Skip any nodes we've already reached in the DFS.
    if (RootN->DFSNumber != 0) {
      assert(RootN->DFSNumber == -1 &&
             "Shouldn't have any mid-DFS root nodes!");
      continue;
    }

    RootN->DFSNumber = RootN->LowLink = 1;
    int NextDFSNumber = 2;

    DFSStack.push_back({RootN, RootN->call_begin()});
    do {
      Node *N;
      call_edge_iterator I;
      std::tie(N, I) = DFSStack.pop_back_val();
      auto E = N->call_end();
      while (I != E) {
        Node &ChildN = *I->getNode();
        if (ChildN.DFSNumber == 0) {
          // We haven't yet visited this child, so descend, pushing the current
          // node onto the stack.
          DFSStack.push_back({N, I});

          assert(!G->SCCMap.count(&ChildN) &&
                 "Found a node with 0 DFS number but already in an SCC!");
          ChildN.DFSNumber = ChildN.LowLink = NextDFSNumber++;
          N = &ChildN;
          I = N->call_begin();
          E = N->call_end();
          continue;
        }

        // Check for the child already being part of some component.
        if (ChildN.DFSNumber == -1) {
          if (G->lookupSCC(ChildN) == &OldSCC) {
            // If the child is part of the old SCC, we know that it can reach
            // every other node, so we have formed a cycle. Pull the entire DFS
            // and pending stacks into it. See the comment above about setting
            // up the old SCC for why we do this.
            int OldSize = OldSCC.size();
            OldSCC.Nodes.push_back(N);
            OldSCC.Nodes.append(PendingSCCStack.begin(), PendingSCCStack.end());
            PendingSCCStack.clear();
            while (!DFSStack.empty())
              OldSCC.Nodes.push_back(DFSStack.pop_back_val().first);
            for (Node &N : make_range(OldSCC.begin() + OldSize, OldSCC.end())) {
              N.DFSNumber = N.LowLink = -1;
              G->SCCMap[&N] = &OldSCC;
            }
            N = nullptr;
            break;
          }

          // If the child has already been added to some child component, it
          // couldn't impact the low-link of this parent because it isn't
          // connected, and thus its low-link isn't relevant so skip it.
          ++I;
          continue;
        }

        // Track the lowest linked child as the lowest link for this node.
        assert(ChildN.LowLink > 0 && "Must have a positive low-link number!");
        if (ChildN.LowLink < N->LowLink)
          N->LowLink = ChildN.LowLink;

        // Move to the next edge.
        ++I;
      }
      if (!N)
        // Cleared the DFS early, start another round.
        break;

      // We've finished processing N and its descendents, put it on our pending
      // SCC stack to eventually get merged into an SCC of nodes.
      PendingSCCStack.push_back(N);

      // If this node is linked to some lower entry, continue walking up the
      // stack.
      if (N->LowLink != N->DFSNumber)
        continue;

      // Otherwise, we've completed an SCC. Append it to our post order list of
      // SCCs.
      int RootDFSNumber = N->DFSNumber;
      // Find the range of the node stack by walking down until we pass the
      // root DFS number.
      auto SCCNodes = make_range(
          PendingSCCStack.rbegin(),
          find_if(reverse(PendingSCCStack), [RootDFSNumber](const Node *N) {
            return N->DFSNumber < RootDFSNumber;
          }));

      // Form a new SCC out of these nodes and then clear them off our pending
      // stack.
      NewSCCs.push_back(G->createSCC(*this, SCCNodes));
      for (Node &N : *NewSCCs.back()) {
        N.DFSNumber = N.LowLink = -1;
        G->SCCMap[&N] = NewSCCs.back();
      }
      PendingSCCStack.erase(SCCNodes.end().base(), PendingSCCStack.end());
    } while (!DFSStack.empty());
  }

  // Insert the remaining SCCs before the old one. The old SCC can reach all
  // other SCCs we form because it contains the target node of the removed edge
  // of the old SCC. This means that we will have edges into all of the new
  // SCCs, which means the old one must come last for postorder.
  int OldIdx = SCCIndices[&OldSCC];
  SCCs.insert(SCCs.begin() + OldIdx, NewSCCs.begin(), NewSCCs.end());

  // Update the mapping from SCC* to index to use the new SCC*s, and remove the
  // old SCC from the mapping.
  for (int Idx = OldIdx, Size = SCCs.size(); Idx < Size; ++Idx)
    SCCIndices[SCCs[Idx]] = Idx;

  return make_range(SCCs.begin() + OldIdx,
                    SCCs.begin() + OldIdx + NewSCCs.size());
}

void LazyCallGraph::RefSCC::switchOutgoingEdgeToCall(Node &SourceN,
                                                     Node &TargetN) {
  assert(!SourceN[TargetN].isCall() && "Must start with a ref edge!");

  assert(G->lookupRefSCC(SourceN) == this && "Source must be in this RefSCC.");
  assert(G->lookupRefSCC(TargetN) != this &&
         "Target must not be in this RefSCC.");
  assert(G->lookupRefSCC(TargetN)->isDescendantOf(*this) &&
         "Target must be a descendant of the Source.");

  // Edges between RefSCCs are the same regardless of call or ref, so we can
  // just flip the edge here.
  SourceN.setEdgeKind(TargetN.getFunction(), Edge::Call);

#ifndef NDEBUG
  // Check that the RefSCC is still valid.
  verify();
#endif
}

void LazyCallGraph::RefSCC::switchOutgoingEdgeToRef(Node &SourceN,
                                                    Node &TargetN) {
  assert(SourceN[TargetN].isCall() && "Must start with a call edge!");

  assert(G->lookupRefSCC(SourceN) == this && "Source must be in this RefSCC.");
  assert(G->lookupRefSCC(TargetN) != this &&
         "Target must not be in this RefSCC.");
  assert(G->lookupRefSCC(TargetN)->isDescendantOf(*this) &&
         "Target must be a descendant of the Source.");

  // Edges between RefSCCs are the same regardless of call or ref, so we can
  // just flip the edge here.
  SourceN.setEdgeKind(TargetN.getFunction(), Edge::Ref);

#ifndef NDEBUG
  // Check that the RefSCC is still valid.
  verify();
#endif
}

void LazyCallGraph::RefSCC::insertInternalRefEdge(Node &SourceN,
                                                  Node &TargetN) {
  assert(G->lookupRefSCC(SourceN) == this && "Source must be in this RefSCC.");
  assert(G->lookupRefSCC(TargetN) == this && "Target must be in this RefSCC.");

  SourceN.insertEdgeInternal(TargetN, Edge::Ref);

#ifndef NDEBUG
  // Check that the RefSCC is still valid.
  verify();
#endif
}

void LazyCallGraph::RefSCC::insertOutgoingEdge(Node &SourceN, Node &TargetN,
                                               Edge::Kind EK) {
  // First insert it into the caller.
  SourceN.insertEdgeInternal(TargetN, EK);

  assert(G->lookupRefSCC(SourceN) == this && "Source must be in this RefSCC.");

  RefSCC &TargetC = *G->lookupRefSCC(TargetN);
  assert(&TargetC != this && "Target must not be in this RefSCC.");
  assert(TargetC.isDescendantOf(*this) &&
         "Target must be a descendant of the Source.");

  // The only change required is to add this SCC to the parent set of the
  // callee.
  TargetC.Parents.insert(this);

#ifndef NDEBUG
  // Check that the RefSCC is still valid.
  verify();
#endif
}

SmallVector<LazyCallGraph::RefSCC *, 1>
LazyCallGraph::RefSCC::insertIncomingRefEdge(Node &SourceN, Node &TargetN) {
  assert(G->lookupRefSCC(TargetN) == this && "Target must be in this RefSCC.");
  RefSCC &SourceC = *G->lookupRefSCC(SourceN);
  assert(&SourceC != this && "Source must not be in this RefSCC.");
  assert(SourceC.isDescendantOf(*this) &&
         "Source must be a descendant of the Target.");

  SmallVector<RefSCC *, 1> DeletedRefSCCs;

#ifndef NDEBUG
  // In a debug build, verify the RefSCC is valid to start with and when this
  // routine finishes.
  verify();
  auto VerifyOnExit = make_scope_exit([&]() { verify(); });
#endif

  int SourceIdx = G->RefSCCIndices[&SourceC];
  int TargetIdx = G->RefSCCIndices[this];
  assert(SourceIdx < TargetIdx &&
         "Postorder list doesn't see edge as incoming!");

  // Compute the RefSCCs which (transitively) reach the source. We do this by
  // working backwards from the source using the parent set in each RefSCC,
  // skipping any RefSCCs that don't fall in the postorder range. This has the
  // advantage of walking the sparser parent edge (in high fan-out graphs) but
  // more importantly this removes examining all forward edges in all RefSCCs
  // within the postorder range which aren't in fact connected. Only connected
  // RefSCCs (and their edges) are visited here.
  auto ComputeSourceConnectedSet = [&](SmallPtrSetImpl<RefSCC *> &Set) {
    Set.insert(&SourceC);
    SmallVector<RefSCC *, 4> Worklist;
    Worklist.push_back(&SourceC);
    do {
      RefSCC &RC = *Worklist.pop_back_val();
      for (RefSCC &ParentRC : RC.parents()) {
        // Skip any RefSCCs outside the range of source to target in the
        // postorder sequence.
        int ParentIdx = G->getRefSCCIndex(ParentRC);
        assert(ParentIdx > SourceIdx && "Parent cannot precede source in postorder!");
        if (ParentIdx > TargetIdx)
          continue;
        if (Set.insert(&ParentRC).second)
          // First edge connecting to this parent, add it to our worklist.
          Worklist.push_back(&ParentRC);
      }
    } while (!Worklist.empty());
  };

  // Use a normal worklist to find which SCCs the target connects to. We still
  // bound the search based on the range in the postorder list we care about,
  // but because this is forward connectivity we just "recurse" through the
  // edges.
  auto ComputeTargetConnectedSet = [&](SmallPtrSetImpl<RefSCC *> &Set) {
    Set.insert(this);
    SmallVector<RefSCC *, 4> Worklist;
    Worklist.push_back(this);
    do {
      RefSCC &RC = *Worklist.pop_back_val();
      for (SCC &C : RC)
        for (Node &N : C)
          for (Edge &E : N) {
            assert(E.getNode() && "Must have formed a node!");
            RefSCC &EdgeRC = *G->lookupRefSCC(*E.getNode());
            if (G->getRefSCCIndex(EdgeRC) <= SourceIdx)
              // Not in the postorder sequence between source and target.
              continue;

            if (Set.insert(&EdgeRC).second)
              Worklist.push_back(&EdgeRC);
          }
    } while (!Worklist.empty());
  };

  // Use a generic helper to update the postorder sequence of RefSCCs and return
  // a range of any RefSCCs connected into a cycle by inserting this edge. This
  // routine will also take care of updating the indices into the postorder
  // sequence.
  iterator_range<SmallVectorImpl<RefSCC *>::iterator> MergeRange =
      updatePostorderSequenceForEdgeInsertion(
          SourceC, *this, G->PostOrderRefSCCs, G->RefSCCIndices,
          ComputeSourceConnectedSet, ComputeTargetConnectedSet);

  // Build a set so we can do fast tests for whether a merge is occuring.
  SmallPtrSet<RefSCC *, 16> MergeSet(MergeRange.begin(), MergeRange.end());

  // Now that we have identified all of the SCCs which need to be merged into
  // a connected set with the inserted edge, merge all of them into this SCC.
  SmallVector<SCC *, 16> MergedSCCs;
  int SCCIndex = 0;
  for (RefSCC *RC : MergeRange) {
    assert(RC != this && "We're merging into the target RefSCC, so it "
                         "shouldn't be in the range.");

    // Merge the parents which aren't part of the merge into the our parents.
    for (RefSCC *ParentRC : RC->Parents)
      if (!MergeSet.count(ParentRC))
        Parents.insert(ParentRC);
    RC->Parents.clear();

    // Walk the inner SCCs to update their up-pointer and walk all the edges to
    // update any parent sets.
    // FIXME: We should try to find a way to avoid this (rather expensive) edge
    // walk by updating the parent sets in some other manner.
    for (SCC &InnerC : *RC) {
      InnerC.OuterRefSCC = this;
      SCCIndices[&InnerC] = SCCIndex++;
      for (Node &N : InnerC) {
        G->SCCMap[&N] = &InnerC;
        for (Edge &E : N) {
          assert(E.getNode() &&
                 "Cannot have a null node within a visited SCC!");
          RefSCC &ChildRC = *G->lookupRefSCC(*E.getNode());
          if (MergeSet.count(&ChildRC))
            continue;
          ChildRC.Parents.erase(RC);
          ChildRC.Parents.insert(this);
        }
      }
    }

    // Now merge in the SCCs. We can actually move here so try to reuse storage
    // the first time through.
    if (MergedSCCs.empty())
      MergedSCCs = std::move(RC->SCCs);
    else
      MergedSCCs.append(RC->SCCs.begin(), RC->SCCs.end());
    RC->SCCs.clear();
    DeletedRefSCCs.push_back(RC);
  }

  // Append our original SCCs to the merged list and move it into place.
  for (SCC &InnerC : *this)
    SCCIndices[&InnerC] = SCCIndex++;
  MergedSCCs.append(SCCs.begin(), SCCs.end());
  SCCs = std::move(MergedSCCs);

  // Remove the merged away RefSCCs from the post order sequence.
  for (RefSCC *RC : MergeRange)
    G->RefSCCIndices.erase(RC);
  int IndexOffset = MergeRange.end() - MergeRange.begin();
  auto EraseEnd =
      G->PostOrderRefSCCs.erase(MergeRange.begin(), MergeRange.end());
  for (RefSCC *RC : make_range(EraseEnd, G->PostOrderRefSCCs.end()))
    G->RefSCCIndices[RC] -= IndexOffset;

  // At this point we have a merged RefSCC with a post-order SCCs list, just
  // connect the nodes to form the new edge.
  SourceN.insertEdgeInternal(TargetN, Edge::Ref);

  // We return the list of SCCs which were merged so that callers can
  // invalidate any data they have associated with those SCCs. Note that these
  // SCCs are no longer in an interesting state (they are totally empty) but
  // the pointers will remain stable for the life of the graph itself.
  return DeletedRefSCCs;
}

void LazyCallGraph::RefSCC::removeOutgoingEdge(Node &SourceN, Node &TargetN) {
  assert(G->lookupRefSCC(SourceN) == this &&
         "The source must be a member of this RefSCC.");

  RefSCC &TargetRC = *G->lookupRefSCC(TargetN);
  assert(&TargetRC != this && "The target must not be a member of this RefSCC");

  assert(!is_contained(G->LeafRefSCCs, this) &&
         "Cannot have a leaf RefSCC source.");

#ifndef NDEBUG
  // In a debug build, verify the RefSCC is valid to start with and when this
  // routine finishes.
  verify();
  auto VerifyOnExit = make_scope_exit([&]() { verify(); });
#endif

  // First remove it from the node.
  SourceN.removeEdgeInternal(TargetN.getFunction());

  bool HasOtherEdgeToChildRC = false;
  bool HasOtherChildRC = false;
  for (SCC *InnerC : SCCs) {
    for (Node &N : *InnerC) {
      for (Edge &E : N) {
        assert(E.getNode() && "Cannot have a missing node in a visited SCC!");
        RefSCC &OtherChildRC = *G->lookupRefSCC(*E.getNode());
        if (&OtherChildRC == &TargetRC) {
          HasOtherEdgeToChildRC = true;
          break;
        }
        if (&OtherChildRC != this)
          HasOtherChildRC = true;
      }
      if (HasOtherEdgeToChildRC)
        break;
    }
    if (HasOtherEdgeToChildRC)
      break;
  }
  // Because the SCCs form a DAG, deleting such an edge cannot change the set
  // of SCCs in the graph. However, it may cut an edge of the SCC DAG, making
  // the source SCC no longer connected to the target SCC. If so, we need to
  // update the target SCC's map of its parents.
  if (!HasOtherEdgeToChildRC) {
    bool Removed = TargetRC.Parents.erase(this);
    (void)Removed;
    assert(Removed &&
           "Did not find the source SCC in the target SCC's parent list!");

    // It may orphan an SCC if it is the last edge reaching it, but that does
    // not violate any invariants of the graph.
    if (TargetRC.Parents.empty())
      DEBUG(dbgs() << "LCG: Update removing " << SourceN.getFunction().getName()
                   << " -> " << TargetN.getFunction().getName()
                   << " edge orphaned the callee's SCC!\n");

    // It may make the Source SCC a leaf SCC.
    if (!HasOtherChildRC)
      G->LeafRefSCCs.push_back(this);
  }
}

SmallVector<LazyCallGraph::RefSCC *, 1>
LazyCallGraph::RefSCC::removeInternalRefEdge(Node &SourceN, Node &TargetN) {
  assert(!SourceN[TargetN].isCall() &&
         "Cannot remove a call edge, it must first be made a ref edge");

#ifndef NDEBUG
  // In a debug build, verify the RefSCC is valid to start with and when this
  // routine finishes.
  verify();
  auto VerifyOnExit = make_scope_exit([&]() { verify(); });
#endif

  // First remove the actual edge.
  SourceN.removeEdgeInternal(TargetN.getFunction());

  // We return a list of the resulting *new* RefSCCs in post-order.
  SmallVector<RefSCC *, 1> Result;

  // Direct recursion doesn't impact the SCC graph at all.
  if (&SourceN == &TargetN)
    return Result;

  // We build somewhat synthetic new RefSCCs by providing a postorder mapping
  // for each inner SCC. We also store these associated with *nodes* rather
  // than SCCs because this saves a round-trip through the node->SCC map and in
  // the common case, SCCs are small. We will verify that we always give the
  // same number to every node in the SCC such that these are equivalent.
  const int RootPostOrderNumber = 0;
  int PostOrderNumber = RootPostOrderNumber + 1;
  SmallDenseMap<Node *, int> PostOrderMapping;

  // Every node in the target SCC can already reach every node in this RefSCC
  // (by definition). It is the only node we know will stay inside this RefSCC.
  // Everything which transitively reaches Target will also remain in the
  // RefSCC. We handle this by pre-marking that the nodes in the target SCC map
  // back to the root post order number.
  //
  // This also enables us to take a very significant short-cut in the standard
  // Tarjan walk to re-form RefSCCs below: whenever we build an edge that
  // references the target node, we know that the target node eventually
  // references all other nodes in our walk. As a consequence, we can detect
  // and handle participants in that cycle without walking all the edges that
  // form the connections, and instead by relying on the fundamental guarantee
  // coming into this operation.
  SCC &TargetC = *G->lookupSCC(TargetN);
  for (Node &N : TargetC)
    PostOrderMapping[&N] = RootPostOrderNumber;

  // Reset all the other nodes to prepare for a DFS over them, and add them to
  // our worklist.
  SmallVector<Node *, 8> Worklist;
  for (SCC *C : SCCs) {
    if (C == &TargetC)
      continue;

    for (Node &N : *C)
      N.DFSNumber = N.LowLink = 0;

    Worklist.append(C->Nodes.begin(), C->Nodes.end());
  }

  auto MarkNodeForSCCNumber = [&PostOrderMapping](Node &N, int Number) {
    N.DFSNumber = N.LowLink = -1;
    PostOrderMapping[&N] = Number;
  };

  SmallVector<std::pair<Node *, edge_iterator>, 4> DFSStack;
  SmallVector<Node *, 4> PendingRefSCCStack;
  do {
    assert(DFSStack.empty() &&
           "Cannot begin a new root with a non-empty DFS stack!");
    assert(PendingRefSCCStack.empty() &&
           "Cannot begin a new root with pending nodes for an SCC!");

    Node *RootN = Worklist.pop_back_val();
    // Skip any nodes we've already reached in the DFS.
    if (RootN->DFSNumber != 0) {
      assert(RootN->DFSNumber == -1 &&
             "Shouldn't have any mid-DFS root nodes!");
      continue;
    }

    RootN->DFSNumber = RootN->LowLink = 1;
    int NextDFSNumber = 2;

    DFSStack.push_back({RootN, RootN->begin()});
    do {
      Node *N;
      edge_iterator I;
      std::tie(N, I) = DFSStack.pop_back_val();
      auto E = N->end();

      assert(N->DFSNumber != 0 && "We should always assign a DFS number "
                                  "before processing a node.");

      while (I != E) {
        Node &ChildN = I->getNode(*G);
        if (ChildN.DFSNumber == 0) {
          // Mark that we should start at this child when next this node is the
          // top of the stack. We don't start at the next child to ensure this
          // child's lowlink is reflected.
          DFSStack.push_back({N, I});

          // Continue, resetting to the child node.
          ChildN.LowLink = ChildN.DFSNumber = NextDFSNumber++;
          N = &ChildN;
          I = ChildN.begin();
          E = ChildN.end();
          continue;
        }
        if (ChildN.DFSNumber == -1) {
          // Check if this edge's target node connects to the deleted edge's
          // target node. If so, we know that every node connected will end up
          // in this RefSCC, so collapse the entire current stack into the root
          // slot in our SCC numbering. See above for the motivation of
          // optimizing the target connected nodes in this way.
          auto PostOrderI = PostOrderMapping.find(&ChildN);
          if (PostOrderI != PostOrderMapping.end() &&
              PostOrderI->second == RootPostOrderNumber) {
            MarkNodeForSCCNumber(*N, RootPostOrderNumber);
            while (!PendingRefSCCStack.empty())
              MarkNodeForSCCNumber(*PendingRefSCCStack.pop_back_val(),
                                   RootPostOrderNumber);
            while (!DFSStack.empty())
              MarkNodeForSCCNumber(*DFSStack.pop_back_val().first,
                                   RootPostOrderNumber);
            // Ensure we break all the way out of the enclosing loop.
            N = nullptr;
            break;
          }

          // If this child isn't currently in this RefSCC, no need to process
          // it.
          // However, we do need to remove this RefSCC from its RefSCC's parent
          // set.
          RefSCC &ChildRC = *G->lookupRefSCC(ChildN);
          ChildRC.Parents.erase(this);
          ++I;
          continue;
        }

        // Track the lowest link of the children, if any are still in the stack.
        // Any child not on the stack will have a LowLink of -1.
        assert(ChildN.LowLink != 0 &&
               "Low-link must not be zero with a non-zero DFS number.");
        if (ChildN.LowLink >= 0 && ChildN.LowLink < N->LowLink)
          N->LowLink = ChildN.LowLink;
        ++I;
      }
      if (!N)
        // We short-circuited this node.
        break;

      // We've finished processing N and its descendents, put it on our pending
      // stack to eventually get merged into a RefSCC.
      PendingRefSCCStack.push_back(N);

      // If this node is linked to some lower entry, continue walking up the
      // stack.
      if (N->LowLink != N->DFSNumber) {
        assert(!DFSStack.empty() &&
               "We never found a viable root for a RefSCC to pop off!");
        continue;
      }

      // Otherwise, form a new RefSCC from the top of the pending node stack.
      int RootDFSNumber = N->DFSNumber;
      // Find the range of the node stack by walking down until we pass the
      // root DFS number.
      auto RefSCCNodes = make_range(
          PendingRefSCCStack.rbegin(),
          find_if(reverse(PendingRefSCCStack), [RootDFSNumber](const Node *N) {
            return N->DFSNumber < RootDFSNumber;
          }));

      // Mark the postorder number for these nodes and clear them off the
      // stack. We'll use the postorder number to pull them into RefSCCs at the
      // end. FIXME: Fuse with the loop above.
      int RefSCCNumber = PostOrderNumber++;
      for (Node *N : RefSCCNodes)
        MarkNodeForSCCNumber(*N, RefSCCNumber);

      PendingRefSCCStack.erase(RefSCCNodes.end().base(),
                               PendingRefSCCStack.end());
    } while (!DFSStack.empty());

    assert(DFSStack.empty() && "Didn't flush the entire DFS stack!");
    assert(PendingRefSCCStack.empty() && "Didn't flush all pending nodes!");
  } while (!Worklist.empty());

  // We now have a post-order numbering for RefSCCs and a mapping from each
  // node in this RefSCC to its final RefSCC. We create each new RefSCC node
  // (re-using this RefSCC node for the root) and build a radix-sort style map
  // from postorder number to the RefSCC. We then append SCCs to each of these
  // RefSCCs in the order they occured in the original SCCs container.
  for (int i = 1; i < PostOrderNumber; ++i)
    Result.push_back(G->createRefSCC(*G));

  // Insert the resulting postorder sequence into the global graph postorder
  // sequence before the current RefSCC in that sequence. The idea being that
  // this RefSCC is the target of the reference edge removed, and thus has
  // a direct or indirect edge to every other RefSCC formed and so must be at
  // the end of any postorder traversal.
  //
  // FIXME: It'd be nice to change the APIs so that we returned an iterator
  // range over the global postorder sequence and generally use that sequence
  // rather than building a separate result vector here.
  if (!Result.empty()) {
    int Idx = G->getRefSCCIndex(*this);
    G->PostOrderRefSCCs.insert(G->PostOrderRefSCCs.begin() + Idx,
                               Result.begin(), Result.end());
    for (int i : seq<int>(Idx, G->PostOrderRefSCCs.size()))
      G->RefSCCIndices[G->PostOrderRefSCCs[i]] = i;
    assert(G->PostOrderRefSCCs[G->getRefSCCIndex(*this)] == this &&
           "Failed to update this RefSCC's index after insertion!");
  }

  for (SCC *C : SCCs) {
    auto PostOrderI = PostOrderMapping.find(&*C->begin());
    assert(PostOrderI != PostOrderMapping.end() &&
           "Cannot have missing mappings for nodes!");
    int SCCNumber = PostOrderI->second;
#ifndef NDEBUG
    for (Node &N : *C)
      assert(PostOrderMapping.find(&N)->second == SCCNumber &&
             "Cannot have different numbers for nodes in the same SCC!");
#endif
    if (SCCNumber == 0)
      // The root node is handled separately by removing the SCCs.
      continue;

    RefSCC &RC = *Result[SCCNumber - 1];
    int SCCIndex = RC.SCCs.size();
    RC.SCCs.push_back(C);
    SCCIndices[C] = SCCIndex;
    C->OuterRefSCC = &RC;
  }

  // FIXME: We re-walk the edges in each RefSCC to establish whether it is
  // a leaf and connect it to the rest of the graph's parents lists. This is
  // really wasteful. We should instead do this during the DFS to avoid yet
  // another edge walk.
  for (RefSCC *RC : Result)
    G->connectRefSCC(*RC);

  // Now erase all but the root's SCCs.
  SCCs.erase(remove_if(SCCs,
                       [&](SCC *C) {
                         return PostOrderMapping.lookup(&*C->begin()) !=
                                RootPostOrderNumber;
                       }),
             SCCs.end());
  SCCIndices.clear();
  for (int i = 0, Size = SCCs.size(); i < Size; ++i)
    SCCIndices[SCCs[i]] = i;

#ifndef NDEBUG
  // Now we need to reconnect the current (root) SCC to the graph. We do this
  // manually because we can special case our leaf handling and detect errors.
  bool IsLeaf = true;
#endif
  for (SCC *C : SCCs)
    for (Node &N : *C) {
      for (Edge &E : N) {
        assert(E.getNode() && "Cannot have a missing node in a visited SCC!");
        RefSCC &ChildRC = *G->lookupRefSCC(*E.getNode());
        if (&ChildRC == this)
          continue;
        ChildRC.Parents.insert(this);
#ifndef NDEBUG
        IsLeaf = false;
#endif
      }
    }
#ifndef NDEBUG
  if (!Result.empty())
    assert(!IsLeaf && "This SCC cannot be a leaf as we have split out new "
                      "SCCs by removing this edge.");
  if (none_of(G->LeafRefSCCs, [&](RefSCC *C) { return C == this; }))
    assert(!IsLeaf && "This SCC cannot be a leaf as it already had child "
                      "SCCs before we removed this edge.");
#endif
  // If this SCC stopped being a leaf through this edge removal, remove it from
  // the leaf SCC list. Note that this DTRT in the case where this was never
  // a leaf.
  // FIXME: As LeafRefSCCs could be very large, we might want to not walk the
  // entire list if this RefSCC wasn't a leaf before the edge removal.
  if (!Result.empty())
    G->LeafRefSCCs.erase(
        std::remove(G->LeafRefSCCs.begin(), G->LeafRefSCCs.end(), this),
        G->LeafRefSCCs.end());

  // Return the new list of SCCs.
  return Result;
}

void LazyCallGraph::insertEdge(Node &SourceN, Function &Target, Edge::Kind EK) {
  assert(SCCMap.empty() && DFSStack.empty() &&
         "This method cannot be called after SCCs have been formed!");

  return SourceN.insertEdgeInternal(Target, EK);
}

void LazyCallGraph::removeEdge(Node &SourceN, Function &Target) {
  assert(SCCMap.empty() && DFSStack.empty() &&
         "This method cannot be called after SCCs have been formed!");

  return SourceN.removeEdgeInternal(Target);
}

LazyCallGraph::Node &LazyCallGraph::insertInto(Function &F, Node *&MappedN) {
  return *new (MappedN = BPA.Allocate()) Node(*this, F);
}

void LazyCallGraph::updateGraphPtrs() {
  // Process all nodes updating the graph pointers.
  {
    SmallVector<Node *, 16> Worklist;
    for (Edge &E : EntryEdges)
      if (Node *EntryN = E.getNode())
        Worklist.push_back(EntryN);

    while (!Worklist.empty()) {
      Node *N = Worklist.pop_back_val();
      N->G = this;
      for (Edge &E : N->Edges)
        if (Node *TargetN = E.getNode())
          Worklist.push_back(TargetN);
    }
  }

  // Process all SCCs updating the graph pointers.
  {
    SmallVector<RefSCC *, 16> Worklist(LeafRefSCCs.begin(), LeafRefSCCs.end());

    while (!Worklist.empty()) {
      RefSCC &C = *Worklist.pop_back_val();
      C.G = this;
      for (RefSCC &ParentC : C.parents())
        Worklist.push_back(&ParentC);
    }
  }
}

/// Build the internal SCCs for a RefSCC from a sequence of nodes.
///
/// Appends the SCCs to the provided vector and updates the map with their
/// indices. Both the vector and map must be empty when passed into this
/// routine.
void LazyCallGraph::buildSCCs(RefSCC &RC, node_stack_range Nodes) {
  assert(RC.SCCs.empty() && "Already built SCCs!");
  assert(RC.SCCIndices.empty() && "Already mapped SCC indices!");

  for (Node *N : Nodes) {
    assert(N->LowLink >= (*Nodes.begin())->LowLink &&
           "We cannot have a low link in an SCC lower than its root on the "
           "stack!");

    // This node will go into the next RefSCC, clear out its DFS and low link
    // as we scan.
    N->DFSNumber = N->LowLink = 0;
  }

  // Each RefSCC contains a DAG of the call SCCs. To build these, we do
  // a direct walk of the call edges using Tarjan's algorithm. We reuse the
  // internal storage as we won't need it for the outer graph's DFS any longer.

  SmallVector<std::pair<Node *, call_edge_iterator>, 16> DFSStack;
  SmallVector<Node *, 16> PendingSCCStack;

  // Scan down the stack and DFS across the call edges.
  for (Node *RootN : Nodes) {
    assert(DFSStack.empty() &&
           "Cannot begin a new root with a non-empty DFS stack!");
    assert(PendingSCCStack.empty() &&
           "Cannot begin a new root with pending nodes for an SCC!");

    // Skip any nodes we've already reached in the DFS.
    if (RootN->DFSNumber != 0) {
      assert(RootN->DFSNumber == -1 &&
             "Shouldn't have any mid-DFS root nodes!");
      continue;
    }

    RootN->DFSNumber = RootN->LowLink = 1;
    int NextDFSNumber = 2;

    DFSStack.push_back({RootN, RootN->call_begin()});
    do {
      Node *N;
      call_edge_iterator I;
      std::tie(N, I) = DFSStack.pop_back_val();
      auto E = N->call_end();
      while (I != E) {
        Node &ChildN = *I->getNode();
        if (ChildN.DFSNumber == 0) {
          // We haven't yet visited this child, so descend, pushing the current
          // node onto the stack.
          DFSStack.push_back({N, I});

          assert(!lookupSCC(ChildN) &&
                 "Found a node with 0 DFS number but already in an SCC!");
          ChildN.DFSNumber = ChildN.LowLink = NextDFSNumber++;
          N = &ChildN;
          I = N->call_begin();
          E = N->call_end();
          continue;
        }

        // If the child has already been added to some child component, it
        // couldn't impact the low-link of this parent because it isn't
        // connected, and thus its low-link isn't relevant so skip it.
        if (ChildN.DFSNumber == -1) {
          ++I;
          continue;
        }

        // Track the lowest linked child as the lowest link for this node.
        assert(ChildN.LowLink > 0 && "Must have a positive low-link number!");
        if (ChildN.LowLink < N->LowLink)
          N->LowLink = ChildN.LowLink;

        // Move to the next edge.
        ++I;
      }

      // We've finished processing N and its descendents, put it on our pending
      // SCC stack to eventually get merged into an SCC of nodes.
      PendingSCCStack.push_back(N);

      // If this node is linked to some lower entry, continue walking up the
      // stack.
      if (N->LowLink != N->DFSNumber)
        continue;

      // Otherwise, we've completed an SCC. Append it to our post order list of
      // SCCs.
      int RootDFSNumber = N->DFSNumber;
      // Find the range of the node stack by walking down until we pass the
      // root DFS number.
      auto SCCNodes = make_range(
          PendingSCCStack.rbegin(),
          find_if(reverse(PendingSCCStack), [RootDFSNumber](const Node *N) {
            return N->DFSNumber < RootDFSNumber;
          }));
      // Form a new SCC out of these nodes and then clear them off our pending
      // stack.
      RC.SCCs.push_back(createSCC(RC, SCCNodes));
      for (Node &N : *RC.SCCs.back()) {
        N.DFSNumber = N.LowLink = -1;
        SCCMap[&N] = RC.SCCs.back();
      }
      PendingSCCStack.erase(SCCNodes.end().base(), PendingSCCStack.end());
    } while (!DFSStack.empty());
  }

  // Wire up the SCC indices.
  for (int i = 0, Size = RC.SCCs.size(); i < Size; ++i)
    RC.SCCIndices[RC.SCCs[i]] = i;
}

// FIXME: We should move callers of this to embed the parent linking and leaf
// tracking into their DFS in order to remove a full walk of all edges.
void LazyCallGraph::connectRefSCC(RefSCC &RC) {
  // Walk all edges in the RefSCC (this remains linear as we only do this once
  // when we build the RefSCC) to connect it to the parent sets of its
  // children.
  bool IsLeaf = true;
  for (SCC &C : RC)
    for (Node &N : C)
      for (Edge &E : N) {
        assert(E.getNode() &&
               "Cannot have a missing node in a visited part of the graph!");
        RefSCC &ChildRC = *lookupRefSCC(*E.getNode());
        if (&ChildRC == &RC)
          continue;
        ChildRC.Parents.insert(&RC);
        IsLeaf = false;
      }

  // For the SCCs where we fine no child SCCs, add them to the leaf list.
  if (IsLeaf)
    LeafRefSCCs.push_back(&RC);
}

bool LazyCallGraph::buildNextRefSCCInPostOrder() {
  if (DFSStack.empty()) {
    Node *N;
    do {
      // If we've handled all candidate entry nodes to the SCC forest, we're
      // done.
      if (RefSCCEntryNodes.empty())
        return false;

      N = &get(*RefSCCEntryNodes.pop_back_val());
    } while (N->DFSNumber != 0);

    // Found a new root, begin the DFS here.
    N->LowLink = N->DFSNumber = 1;
    NextDFSNumber = 2;
    DFSStack.push_back({N, N->begin()});
  }

  for (;;) {
    Node *N;
    edge_iterator I;
    std::tie(N, I) = DFSStack.pop_back_val();

    assert(N->DFSNumber > 0 && "We should always assign a DFS number "
                               "before placing a node onto the stack.");

    auto E = N->end();
    while (I != E) {
      Node &ChildN = I->getNode(*this);
      if (ChildN.DFSNumber == 0) {
        // We haven't yet visited this child, so descend, pushing the current
        // node onto the stack.
        DFSStack.push_back({N, N->begin()});

        assert(!SCCMap.count(&ChildN) &&
               "Found a node with 0 DFS number but already in an SCC!");
        ChildN.LowLink = ChildN.DFSNumber = NextDFSNumber++;
        N = &ChildN;
        I = N->begin();
        E = N->end();
        continue;
      }

      // If the child has already been added to some child component, it
      // couldn't impact the low-link of this parent because it isn't
      // connected, and thus its low-link isn't relevant so skip it.
      if (ChildN.DFSNumber == -1) {
        ++I;
        continue;
      }

      // Track the lowest linked child as the lowest link for this node.
      assert(ChildN.LowLink > 0 && "Must have a positive low-link number!");
      if (ChildN.LowLink < N->LowLink)
        N->LowLink = ChildN.LowLink;

      // Move to the next edge.
      ++I;
    }

    // We've finished processing N and its descendents, put it on our pending
    // SCC stack to eventually get merged into an SCC of nodes.
    PendingRefSCCStack.push_back(N);

    // If this node is linked to some lower entry, continue walking up the
    // stack.
    if (N->LowLink != N->DFSNumber) {
      assert(!DFSStack.empty() &&
             "We never found a viable root for an SCC to pop off!");
      continue;
    }

    // Otherwise, form a new RefSCC from the top of the pending node stack.
    int RootDFSNumber = N->DFSNumber;
    // Find the range of the node stack by walking down until we pass the
    // root DFS number.
    auto RefSCCNodes = node_stack_range(
        PendingRefSCCStack.rbegin(),
        find_if(reverse(PendingRefSCCStack), [RootDFSNumber](const Node *N) {
          return N->DFSNumber < RootDFSNumber;
        }));
    // Form a new RefSCC out of these nodes and then clear them off our pending
    // stack.
    RefSCC *NewRC = createRefSCC(*this);
    buildSCCs(*NewRC, RefSCCNodes);
    connectRefSCC(*NewRC);
    PendingRefSCCStack.erase(RefSCCNodes.end().base(),
                             PendingRefSCCStack.end());

    // Push the new node into the postorder list and return true indicating we
    // successfully grew the postorder sequence by one.
    bool Inserted =
        RefSCCIndices.insert({NewRC, PostOrderRefSCCs.size()}).second;
    (void)Inserted;
    assert(Inserted && "Cannot already have this RefSCC in the index map!");
    PostOrderRefSCCs.push_back(NewRC);
    return true;
  }
}

char LazyCallGraphAnalysis::PassID;

LazyCallGraphPrinterPass::LazyCallGraphPrinterPass(raw_ostream &OS) : OS(OS) {}

static void printNode(raw_ostream &OS, LazyCallGraph::Node &N) {
  OS << "  Edges in function: " << N.getFunction().getName() << "\n";
  for (const LazyCallGraph::Edge &E : N)
    OS << "    " << (E.isCall() ? "call" : "ref ") << " -> "
       << E.getFunction().getName() << "\n";

  OS << "\n";
}

static void printSCC(raw_ostream &OS, LazyCallGraph::SCC &C) {
  ptrdiff_t Size = std::distance(C.begin(), C.end());
  OS << "    SCC with " << Size << " functions:\n";

  for (LazyCallGraph::Node &N : C)
    OS << "      " << N.getFunction().getName() << "\n";
}

static void printRefSCC(raw_ostream &OS, LazyCallGraph::RefSCC &C) {
  ptrdiff_t Size = std::distance(C.begin(), C.end());
  OS << "  RefSCC with " << Size << " call SCCs:\n";

  for (LazyCallGraph::SCC &InnerC : C)
    printSCC(OS, InnerC);

  OS << "\n";
}

PreservedAnalyses LazyCallGraphPrinterPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  LazyCallGraph &G = AM.getResult<LazyCallGraphAnalysis>(M);

  OS << "Printing the call graph for module: " << M.getModuleIdentifier()
     << "\n\n";

  for (Function &F : M)
    printNode(OS, G.get(F));

  for (LazyCallGraph::RefSCC &C : G.postorder_ref_sccs())
    printRefSCC(OS, C);

  return PreservedAnalyses::all();
}

LazyCallGraphDOTPrinterPass::LazyCallGraphDOTPrinterPass(raw_ostream &OS)
    : OS(OS) {}

static void printNodeDOT(raw_ostream &OS, LazyCallGraph::Node &N) {
  std::string Name = "\"" + DOT::EscapeString(N.getFunction().getName()) + "\"";

  for (const LazyCallGraph::Edge &E : N) {
    OS << "  " << Name << " -> \""
       << DOT::EscapeString(E.getFunction().getName()) << "\"";
    if (!E.isCall()) // It is a ref edge.
      OS << " [style=dashed,label=\"ref\"]";
    OS << ";\n";
  }

  OS << "\n";
}

PreservedAnalyses LazyCallGraphDOTPrinterPass::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  LazyCallGraph &G = AM.getResult<LazyCallGraphAnalysis>(M);

  OS << "digraph \"" << DOT::EscapeString(M.getModuleIdentifier()) << "\" {\n";

  for (Function &F : M)
    printNodeDOT(OS, G.get(F));

  OS << "}\n";

  return PreservedAnalyses::all();
}
