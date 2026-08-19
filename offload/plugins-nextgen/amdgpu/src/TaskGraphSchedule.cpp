//===-- TaskGraphSchedule.cpp - Lower a taskgraph onto in-order lanes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A single recursive pass over the region tree that places each leaf on a lane
// and works out, for each one, the fewest completions it must wait for.
//
// What makes the waits few is that lanes are in-order, so knowing that one item
// has completed is knowing that everything before it on its lane has too.  Each
// placed item therefore carries a *clock*: per lane, how many of that lane's
// items are known to have completed by the time the item itself has.  An item's
// clock is the join of what it waits for and of the item before it on its own
// lane, so knowledge flows along every ordering the lanes enforce.  A producer
// then needs no wait when what the consumer already knows covers it, which
// subsumes three reductions:
//   * a dependence on something earlier on the *same* lane needs no wait;
//   * of several producers on one lane, waiting for the latest covers the rest;
//   * a producer that another producer, or the lane's earlier items, already
//     waited for needs no wait of its own.
//
// Correctness rests on three invariants, each of which is load-bearing (each
// was a bug first, caught by the randomized oracle in
// offload/unittests/TaskGraph):
//   1. Positions are per-lane and only ever compared with positions, or clock
//      entries, for the same lane.  An earlier producer does not imply a later
//      one has finished.
//   2. Inside an irreducible region, predecessor signals are gathered
//      *unfiltered*.  Reduction may only be applied where a leaf is actually
//      placed and its lane is therefore known: reducing against a guessed lane
//      drops real dependences for a successor sub-region whose several entry
//      leaves get spread over different lanes.
//   3. Every per-lane structure is sized to the real lane count.
//
// Deadlock freedom follows from the single pass: an item can only be made to
// wait for an already-placed producer, so the happens-before graph can only
// ever point backwards in placement order and cannot contain a cycle.
//
//===----------------------------------------------------------------------===//

#include "TaskGraphSchedule.h"

#include "PluginInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <utility>
#include <vector>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

namespace {

/// "This item has completed", carried from the item that produces it to the
/// items that consume it, by item number.
using SignalsTy = SmallVector<uint32_t, 8>;

/// Per lane, how many of the lane's items are known to have completed.
using ClockTy = SmallVector<uint32_t, 8>;

/// An intra-region dependence edge, by child ordinal.
using EdgeTy = std::pair<uint32_t, uint32_t>;

/// Topologically sort [0, NumChildren) under \p Edges, predecessors first
/// (Kahn's algorithm).  A cycle is a hard error: the region tree is required to
/// be a DAG, and silently picking some order for a cycle would produce a
/// lowering that does not implement the graph.
Error topoSort(uint32_t NumChildren, ArrayRef<EdgeTy> Edges,
               SmallVectorImpl<uint32_t> &Order) {
  SmallVector<uint32_t, 16> InDegree(NumChildren, 0);
  for (const EdgeTy &E : Edges)
    ++InDegree[E.second];

  SmallVector<uint32_t, 16> Ready;
  for (uint32_t I = 0; I < NumChildren; ++I)
    if (InDegree[I] == 0)
      Ready.push_back(I);

  // Ready is consumed as a queue, not a stack, so the order comes out level by
  // level.  Any topological order is correct, but a breadth-first one emits a
  // layer of mutually independent children consecutively, and the round-robin
  // placement below then spreads that layer across the lanes.  Depth-first
  // instead emits a child right after its own predecessor, so the round robin
  // puts it on the next lane where it can only wait: the lanes run in lockstep
  // and the lowering extracts no concurrency at all.
  for (uint32_t Head = 0; Head < Ready.size(); ++Head) {
    uint32_t Idx = Ready[Head];
    Order.push_back(Idx);
    for (const EdgeTy &E : Edges)
      if (E.first == Idx && --InDegree[E.second] == 0)
        Ready.push_back(E.second);
  }

  if (Order.size() != NumChildren)
    return Plugin::error(error::ErrorCode::INVALID_ARGUMENT,
                         "cyclic irreducible region in taskgraph");
  return Plugin::success();
}

class LoweringTy {
public:
  LoweringTy(uint32_t NumLanes, TaskGraphLaneSinkTy &Sink)
      : NumLanes(NumLanes), Sink(Sink), Tails(NumLanes, NoItem),
        Lengths(NumLanes, 0) {}

  /// Lower \p Elem, which must not start until every signal in \p In has fired.
  /// \p Lane is the lane cursor, advanced in place; \p Out receives the
  /// element's output signals, i.e. what a successor of it must wait for.
  Error lower(const TaskGraphElement &Elem, ArrayRef<uint32_t> In,
              uint32_t &Lane, SignalsTy &Out);

private:
  static constexpr uint32_t NoItem = ~0U;

  /// What the lowering remembers of an item once it is placed.
  struct PlacedTy {
    uint32_t Lane;
    /// 0-based position within Lane.
    uint32_t Pos;
    ClockTy Clock;
  };

  Error lowerIrreducible(const TaskGraphIrreducibleRegion &Region,
                         ArrayRef<uint32_t> In, uint32_t &Lane, SignalsTy &Out);

  /// Place \p Node on \p Lane, waiting for whatever \p In reduces to.
  /// Returns its item number.
  Expected<uint32_t> place(const TaskGraphNode &Node, uint32_t Lane,
                           ArrayRef<uint32_t> In);

  static void join(ClockTy &Into, const ClockTy &From) {
    for (auto [I, F] : llvm::zip_equal(Into, From))
      I = std::max(I, F);
  }

  const uint32_t NumLanes;
  TaskGraphLaneSinkTy &Sink;
  std::vector<PlacedTy> Placed;
  /// Per lane, the last item placed on it, or NoItem.  Sized to the real lane
  /// count (invariant 3), as is every clock.
  SmallVector<uint32_t, 8> Tails;
  /// Per lane, how many items have been placed on it.
  SmallVector<uint32_t, 8> Lengths;
};

Expected<uint32_t> LoweringTy::place(const TaskGraphNode &Node, uint32_t Lane,
                                     ArrayRef<uint32_t> In) {
  // What the item knows to have completed before it starts, without waiting
  // for anything: on its own lane, everything placed there before it.
  ClockTy Known(NumLanes, 0);
  if (Tails[Lane] != NoItem)
    Known = Placed[Tails[Lane]].Clock;

  // Of the producers, the latest per lane not already known to have completed.
  // Invariant 1: positions are only compared within a lane.
  SmallVector<uint32_t, 8> Latest(NumLanes, NoItem);
  for (uint32_t Sig : In) {
    const PlacedTy &P = Placed[Sig];
    if (Known[P.Lane] > P.Pos)
      continue;
    if (Latest[P.Lane] == NoItem || Placed[Latest[P.Lane]].Pos < P.Pos)
      Latest[P.Lane] = Sig;
  }

  // Less any that another of them implies.  Two cannot imply each other, which
  // would make them each complete before the other, so dropping every implied
  // one leaves a set that still implies them all.
  SmallVector<uint32_t, 4> Waits;
  for (uint32_t L = 0; L < NumLanes; ++L) {
    if (Latest[L] == NoItem)
      continue;
    const uint32_t Pos = Placed[Latest[L]].Pos;
    bool Implied = llvm::any_of(Latest, [&](uint32_t Other) {
      return Other != NoItem && Other != Latest[L] &&
             Placed[Other].Clock[L] > Pos;
    });
    if (!Implied)
      Waits.push_back(Latest[L]);
  }

  // Everything in In completes before the item starts, waited for or implied,
  // so its clock is all of theirs joined, plus the item itself.
  ClockTy Clock = std::move(Known);
  for (uint32_t Sig : In)
    join(Clock, Placed[Sig].Clock);

  const uint32_t Item = Placed.size();
  const uint32_t Pos = Lengths[Lane]++;
  Clock[Lane] = Pos + 1;
  Tails[Lane] = Item;
  Placed.push_back({Lane, Pos, std::move(Clock)});

  if (Error Err = Sink.place(Item, Node, Lane, Waits))
    return std::move(Err);
  return Item;
}

Error LoweringTy::lower(const TaskGraphElement &Elem, ArrayRef<uint32_t> In,
                        uint32_t &Lane, SignalsTy &Out) {
  // A leaf is the only thing that emits work.  It stays on the current lane
  // (the cursor is advanced by the enclosing region, not here) and its output
  // signal is its own completion.
  if (const auto *Node = dyn_cast<TaskGraphNode>(&Elem)) {
    Expected<uint32_t> Item = place(*Node, Lane, In);
    if (!Item)
      return Item.takeError();
    Out.push_back(*Item);
    return Plugin::success();
  }

  const auto *Region = cast<TaskGraphRegion>(&Elem);
  switch (Region->getKind()) {
  case TGKind::ParRegion:
    // Children are independent: each gets the region's own incoming
    // dependences, and each starts on a fresh lane.  With more children than
    // lanes the surplus shares lanes and is serialized by in-order execution --
    // a loss of concurrency, not of correctness.
    for (const TaskGraphElement *Child : Region->children()) {
      if (Error Err = lower(*Child, In, Lane, Out))
        return Err;
      Lane = (Lane + 1) % NumLanes;
    }
    break;

  case TGKind::SeqRegion:
  case TGKind::ExclRegion: {
    // Thread the signals through the children left to right, deliberately
    // leaving the lane cursor alone: a run of leaves then lands in order on one
    // lane, where in-order execution orders them for free.
    //
    // An exclusive region is serialized exactly like a sequential one.  A total
    // order over the children trivially satisfies mutual exclusion; it gives up
    // the freedom to run them in any order, which this backend has no way to
    // exploit anyway.
    SignalsTy Cur(In.begin(), In.end());
    for (const TaskGraphElement *Child : Region->children()) {
      SignalsTy Next;
      if (Error Err = lower(*Child, Cur, Lane, Next))
        return Err;
      Cur = std::move(Next);
    }
    Out.append(Cur.begin(), Cur.end());
    break;
  }

  case TGKind::IrreducibleRegion:
    if (Error Err = lowerIrreducible(cast<TaskGraphIrreducibleRegion>(*Region),
                                     In, Lane, Out))
      return Err;
    break;

  default:
    return Plugin::error(error::ErrorCode::INVALID_ARGUMENT,
                         "unexpected element kind in taskgraph region tree");
  }

  // A region that emitted no work at all is transparent to dependences: it
  // passes its incoming signals straight through, so a successor still waits
  // for whatever preceded it.  Without this, an empty region would cut the
  // dependence chain wherever the work either side of it landed on different
  // lanes.
  if (Out.empty())
    Out.append(In.begin(), In.end());
  return Plugin::success();
}

Error LoweringTy::lowerIrreducible(const TaskGraphIrreducibleRegion &Region,
                                   ArrayRef<uint32_t> In, uint32_t &Lane,
                                   SignalsTy &Out) {
  const uint32_t NumChildren = Region.NumChildren;

  // Both endpoints name a child of this region; the region's own entries and
  // exits are derived below from which children have no incoming or no outgoing
  // edge, so the stream does not carry them.  An out-of-range ordinal is a
  // malformed stream, and ignoring the edge would silently drop a dependence,
  // so decline the graph and let it run on the host instead.
  SmallVector<EdgeTy, 16> Edges;
  for (const TaskGraphEdge &E : Region.edges()) {
    if (E.Src < 0 || E.Dst < 0 || static_cast<uint32_t>(E.Src) >= NumChildren ||
        static_cast<uint32_t>(E.Dst) >= NumChildren)
      return Plugin::error(
          error::ErrorCode::INVALID_ARGUMENT,
          "irreducible region edge does not name a child of the region");
    Edges.push_back(
        {static_cast<uint32_t>(E.Src), static_cast<uint32_t>(E.Dst)});
  }

  SmallVector<uint32_t, 16> Order;
  if (Error Err = topoSort(NumChildren, Edges, Order))
    return Err;

  // Visiting in topological order means a child's predecessors have already
  // been lowered, so their output signals are available when it needs them.
  std::vector<SignalsTy> Outgoing(NumChildren);
  for (uint32_t Idx : Order) {
    SignalsTy Intra;
    for (const EdgeTy &E : Edges)
      if (E.second == Idx)
        Intra.append(Outgoing[E.first].begin(), Outgoing[E.first].end());

    // An entry child (no predecessor in this region) takes the region's own
    // incoming dependences; every other child reaches them transitively through
    // the entries.  Note Intra is passed on unfiltered -- see invariant 2.
    ArrayRef<uint32_t> Use = Intra.empty() ? In : ArrayRef<uint32_t>(Intra);
    if (Error Err = lower(*Region.children()[Idx], Use, Lane, Outgoing[Idx]))
      return Err;
    Lane = (Lane + 1) % NumLanes;
  }

  // The region has completed once its exits have; a DAG always has at least
  // one.
  for (uint32_t Idx = 0; Idx < NumChildren; ++Idx) {
    bool HasSuccessor = false;
    for (const EdgeTy &E : Edges)
      if (E.first == Idx) {
        HasSuccessor = true;
        break;
      }
    if (!HasSuccessor)
      Out.append(Outgoing[Idx].begin(), Outgoing[Idx].end());
  }
  return Plugin::success();
}

} // namespace

Error lowerTaskGraphOntoLanes(const TaskGraphTy &Graph, uint32_t NumLanes,
                              TaskGraphLaneSinkTy &Sink) {
  if (NumLanes == 0)
    return Plugin::error(error::ErrorCode::INVALID_ARGUMENT,
                         "taskgraph lowering needs at least one lane");
  if (!Graph.getRoot())
    return Plugin::error(error::ErrorCode::INVALID_ARGUMENT,
                         "taskgraph has no root region");

  LoweringTy Lowering(NumLanes, Sink);
  SignalsTy Out;
  uint32_t Lane = 0;
  return Lowering.lower(*Graph.getRoot(), {}, Lane, Out);
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
