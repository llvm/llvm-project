//===-- TaskGraphScheduleTest.cpp - Tests for taskgraph lane lowering -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What has to be true of lowerTaskGraphOntoLanes's output is that the
// dependence order the region tree asks for is really enforced by the lanes it
// places leaves on.  Reading the placements and convincing yourself of that by
// eye does not scale, so each test here does it mechanically:
//
//   1. Build a region tree and lower it onto N lanes, into a sink that records
//      every placement.
//   2. Reconstruct the happens-before relation the placements actually
//      implement: consecutive items on a lane (in-order execution), plus each
//      item's waits.
//   3. Derive the *intended* partial order over leaves straight from the tree,
//      with an oracle that shares no code with the lowering.
//   4. Assert every intended dependence is reachable in the reconstructed
//      relation, that the relation is acyclic (i.e. cannot deadlock), and that
//      no wait is one the lanes already imply.
//
// The interesting cases are the ones where the reductions the lowering relies
// on could drop a real dependence: irreducible regions, irreducible regions
// nested inside each other, and multi-entry / multi-exit sub-regions as edge
// endpoints.  Every such case is run at several lane counts, including one lane
// and more lanes than the tree can use, because the reductions behave
// differently as items collide on lanes.  A randomized sweep over generated
// trees covers the shapes nobody thought to write down.
//
//===----------------------------------------------------------------------===//

#include "TaskGraphSchedule.h"

#include "Shared/TaskGraphBuilder.h"

#include "llvm/ADT/DenseMap.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::omp::target;
using namespace llvm::omp::target::plugin;

namespace {

//===----------------------------------------------------------------------===//
// Writing region trees
//===----------------------------------------------------------------------===//

/// A region tree as a test writes it.  Leaves carry a small integer identity so
/// the assertions can talk about "leaf 3" rather than about pointers.
struct SpecTy {
  enum KindTy {
    Target,
    EnterData,
    ExitData,
    UpdateData,
    Par,
    Seq,
    Excl,
    Irred
  };

  KindTy Kind;
  int Id = 0;                             // leaves only
  std::vector<SpecTy> Children;           // regions only
  std::vector<std::pair<int, int>> Edges; // Irred only, by child ordinal

  bool isLeaf() const { return Kind <= UpdateData; }
};

SpecTy target(int Id) { return {SpecTy::Target, Id, {}, {}}; }
SpecTy enterData(int Id) { return {SpecTy::EnterData, Id, {}, {}}; }
SpecTy exitData(int Id) { return {SpecTy::ExitData, Id, {}, {}}; }
SpecTy updateData(int Id) { return {SpecTy::UpdateData, Id, {}, {}}; }
SpecTy par(std::vector<SpecTy> Children) {
  return {SpecTy::Par, 0, std::move(Children), {}};
}
SpecTy seq(std::vector<SpecTy> Children) {
  return {SpecTy::Seq, 0, std::move(Children), {}};
}
SpecTy excl(std::vector<SpecTy> Children) {
  return {SpecTy::Excl, 0, std::move(Children), {}};
}
SpecTy irred(std::vector<SpecTy> Children,
             std::vector<std::pair<int, int>> Edges) {
  return {SpecTy::Irred, 0, std::move(Children), std::move(Edges)};
}

/// A TaskGraphTy materialized from a SpecTy, streamed in exactly as libomp
/// streams a recorded graph into libomptarget: a measuring pass that only sizes
/// the block, then a build pass into a block of that size.  Going through the
/// real construction path means these tests would also catch a tree that is
/// mis-sized or mis-linked, not just a mis-scheduled one.
class BuiltGraphTy {
public:
  explicit BuiltGraphTy(const SpecTy &Spec) {
    Graph = TaskGraphBuilder::start(/*DeviceId=*/0, /*GraphId=*/0,
                                    /*ByteSize=*/0);
    stream(Spec);
    size_t Bytes = TaskGraphBuilder(*Graph).finish();
    delete Graph;

    Graph = TaskGraphBuilder::start(/*DeviceId=*/0, /*GraphId=*/0, Bytes);
    stream(Spec);
    TaskGraphBuilder(*Graph).finish();
  }
  ~BuiltGraphTy() { delete Graph; }

  const TaskGraphTy &graph() const { return *Graph; }

  /// Why the graph is not well formed, or null.
  const char *whyMalformed() const {
    return TaskGraphBuilder::whyMalformed(*Graph);
  }

  /// The spec-level identity of a leaf, or -1 if it is not one of ours.
  int idOf(const TaskGraphNode *Node) const {
    auto It = Ids.find(Node);
    return It == Ids.end() ? -1 : It->second;
  }

private:
  template <class NodeT> void streamLeaf(int Id) {
    TaskGraphBuilder B(*Graph);
    if (NodeT *Node = B.makeLeaf<NodeT>()) {
      B.linkChild(Node);
      Ids[Node] = Id;
    }
  }

  template <class RegionT> void streamRegion(const SpecTy &Spec) {
    TaskGraphBuilder B(*Graph);
    auto *Region =
        B.makeRegion<RegionT>(static_cast<int32_t>(Spec.Children.size()));
    if (Region)
      B.pushRegion(Region);
    for (const SpecTy &Child : Spec.Children)
      stream(Child);
    if (Region)
      B.popRegion();
  }

  void stream(const SpecTy &Spec) {
    switch (Spec.Kind) {
    case SpecTy::Target:
      return streamLeaf<TaskGraphTargetNode>(Spec.Id);
    case SpecTy::EnterData:
      return streamLeaf<TaskGraphTargetEnterDataNode>(Spec.Id);
    case SpecTy::ExitData:
      return streamLeaf<TaskGraphTargetExitDataNode>(Spec.Id);
    case SpecTy::UpdateData:
      return streamLeaf<TaskGraphTargetUpdateDataNode>(Spec.Id);
    case SpecTy::Par:
      return streamRegion<TaskGraphParRegion>(Spec);
    case SpecTy::Seq:
      return streamRegion<TaskGraphSeqRegion>(Spec);
    case SpecTy::Excl:
      return streamRegion<TaskGraphExclRegion>(Spec);
    case SpecTy::Irred:
      break;
    }

    // An irreducible region's edge array is sized with the region, so the
    // measuring pass accounts for it before any edge is emitted.
    TaskGraphBuilder B(*Graph);
    auto *Region =
        B.makeIrreducibleRegion(static_cast<int32_t>(Spec.Children.size()),
                                static_cast<int32_t>(Spec.Edges.size()));
    if (Region)
      B.pushRegion(Region);
    for (const SpecTy &Child : Spec.Children)
      stream(Child);
    if (Region) {
      for (const std::pair<int, int> &Edge : Spec.Edges)
        B.addEdge(Edge.first, Edge.second);
      B.popRegion();
    }
  }

  TaskGraphTy *Graph = nullptr;
  DenseMap<const TaskGraphNode *, int> Ids;
};

//===----------------------------------------------------------------------===//
// What the lowering actually enforces
//===----------------------------------------------------------------------===//

/// Every placement a lowering made, as a sink saw them.
struct PlacementsTy {
  struct ItemTy {
    const TaskGraphNode *Node = nullptr;
    uint32_t Lane = 0;
    std::vector<uint32_t> Waits;
  };
  std::vector<ItemTy> Items;
  /// Per lane, the items placed on it in order.
  std::vector<std::vector<uint32_t>> Lanes;
};

/// A sink that only records what it is given, and checks the numbering it is
/// promised.
class RecordingSinkTy final : public TaskGraphLaneSinkTy {
public:
  explicit RecordingSinkTy(uint32_t NumLanes) { P.Lanes.resize(NumLanes); }

  Error place(uint32_t Item, const TaskGraphNode &Leaf, uint32_t Lane,
              ArrayRef<uint32_t> Waits) override {
    EXPECT_EQ(Item, P.Items.size()) << "items are numbered as they are placed";
    P.Items.push_back(
        {&Leaf, Lane, std::vector<uint32_t>(Waits.begin(), Waits.end())});
    P.Lanes[Lane].push_back(Item);
    return Error::success();
  }

  PlacementsTy P;
};

/// Lower \p Built onto \p NumLanes lanes and return what was placed.
Expected<PlacementsTy> lowerGraph(const TaskGraphTy &Graph, uint32_t NumLanes) {
  RecordingSinkTy Sink(NumLanes);
  if (Error Err = lowerTaskGraphOntoLanes(Graph, NumLanes, Sink))
    return std::move(Err);
  return std::move(Sink.P);
}

/// The happens-before relation implied by a lowering, over its items.
struct HappensBeforeTy {
  std::vector<std::vector<uint32_t>> Succs;
  /// Per item, its predecessor on its own lane, or ~0U.
  std::vector<uint32_t> LanePred;
  /// Leaf identity -> item, for talking about the lowering in spec terms.
  std::map<int, uint32_t> ItemOfLeaf;

  size_t size() const { return Succs.size(); }
};

HappensBeforeTy buildHappensBefore(const PlacementsTy &P,
                                   const BuiltGraphTy &Built) {
  HappensBeforeTy HB;
  HB.Succs.resize(P.Items.size());
  HB.LanePred.assign(P.Items.size(), ~0U);

  // In-order execution within a lane.
  for (const std::vector<uint32_t> &Lane : P.Lanes)
    for (size_t I = 1; I < Lane.size(); ++I) {
      HB.Succs[Lane[I - 1]].push_back(Lane[I]);
      HB.LanePred[Lane[I]] = Lane[I - 1];
    }

  // Explicit waits, cross-lane or on an item placed on no lane.
  for (uint32_t I = 0; I < P.Items.size(); ++I)
    for (uint32_t Wait : P.Items[I].Waits)
      HB.Succs[Wait].push_back(I);

  for (uint32_t I = 0; I < P.Items.size(); ++I)
    HB.ItemOfLeaf[Built.idOf(P.Items[I].Node)] = I;
  return HB;
}

bool reachable(const HappensBeforeTy &HB, uint32_t From, uint32_t To) {
  if (From == To)
    return true;
  std::vector<bool> Seen(HB.size(), false);
  std::vector<uint32_t> Stack{From};
  while (!Stack.empty()) {
    uint32_t Node = Stack.back();
    Stack.pop_back();
    if (Seen[Node])
      continue;
    Seen[Node] = true;
    for (uint32_t Succ : HB.Succs[Node]) {
      if (Succ == To)
        return true;
      Stack.push_back(Succ);
    }
  }
  return false;
}

/// Does the schedule guarantee leaf \p A completes before leaf \p B starts?
bool happensBefore(const HappensBeforeTy &HB, int A, int B) {
  auto ItA = HB.ItemOfLeaf.find(A), ItB = HB.ItemOfLeaf.find(B);
  if (ItA == HB.ItemOfLeaf.end() || ItB == HB.ItemOfLeaf.end())
    return false;
  return reachable(HB, ItA->second, ItB->second);
}

/// A cycle here would be a schedule that cannot make progress.
bool isAcyclic(const HappensBeforeTy &HB) {
  enum { White, Grey, Black };
  std::vector<uint8_t> Colour(HB.size(), White);
  // Iterative DFS over (node, next successor index) so a deep chain cannot
  // overflow the stack.
  for (uint32_t Root = 0; Root < HB.size(); ++Root) {
    if (Colour[Root] != White)
      continue;
    std::vector<std::pair<uint32_t, size_t>> Stack{{Root, 0}};
    Colour[Root] = Grey;
    while (!Stack.empty()) {
      auto &[Node, Next] = Stack.back();
      if (Next == HB.Succs[Node].size()) {
        Colour[Node] = Black;
        Stack.pop_back();
        continue;
      }
      uint32_t Succ = HB.Succs[Node][Next++];
      if (Colour[Succ] == Grey)
        return false;
      if (Colour[Succ] == White) {
        Colour[Succ] = Grey;
        Stack.push_back({Succ, 0});
      }
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// What the tree asks for (independent oracle)
//===----------------------------------------------------------------------===//

/// The intended dependences of a sub-tree, plus the leaves at its boundary:
/// Entry leaves are those nothing inside precedes (so an incoming dependence
/// must precede all of them), Exit leaves those nothing inside follows.
///
/// This reads the region model's *meaning* directly and shares no code with the
/// scheduler, which is what makes it worth checking against.
struct AnalysisTy {
  std::vector<int> Entry;
  std::vector<int> Exit;
  std::set<std::pair<int, int>> Edges;
};

AnalysisTy analyze(const SpecTy &Spec) {
  AnalysisTy Result;
  if (Spec.isLeaf()) {
    Result.Entry = {Spec.Id};
    Result.Exit = {Spec.Id};
    return Result;
  }

  std::vector<AnalysisTy> Children;
  for (const SpecTy &Child : Spec.Children)
    Children.push_back(analyze(Child));
  for (const AnalysisTy &Child : Children)
    Result.Edges.insert(Child.Edges.begin(), Child.Edges.end());

  auto cross = [&](const std::vector<int> &From, const std::vector<int> &To) {
    for (int A : From)
      for (int B : To)
        Result.Edges.insert({A, B});
  };

  switch (Spec.Kind) {
  case SpecTy::Par:
    // Independent children: the region's boundary is the union of theirs.
    for (const AnalysisTy &Child : Children) {
      Result.Entry.insert(Result.Entry.end(), Child.Entry.begin(),
                          Child.Entry.end());
      Result.Exit.insert(Result.Exit.end(), Child.Exit.begin(),
                         Child.Exit.end());
    }
    break;

  case SpecTy::Seq:
  case SpecTy::Excl: {
    // A chain: each child's exits precede the next child's entries.  An
    // exclusive region is required to serialize too, so it is analyzed the same
    // way -- a total order over the children satisfies mutual exclusion.
    //
    // A child with no leaves at all is skipped rather than treated as a break
    // in the chain, so the dependence either side of it is still required.
    std::vector<int> PrevExits;
    for (const AnalysisTy &Child : Children) {
      if (Child.Entry.empty())
        continue;
      cross(PrevExits, Child.Entry);
      if (Result.Entry.empty())
        Result.Entry = Child.Entry;
      PrevExits = Child.Exit;
    }
    Result.Exit = PrevExits;
    break;
  }

  default: {
    // Irreducible: the explicit edges are between whole children, so they
    // become every (exit of source, entry of destination) pair.  A child with
    // no incoming edge is on the region's entry boundary, one with no outgoing
    // edge on its exit boundary.
    const size_t N = Children.size();
    std::vector<bool> HasPred(N, false), HasSucc(N, false);
    for (const std::pair<int, int> &Edge : Spec.Edges) {
      cross(Children[Edge.first].Exit, Children[Edge.second].Entry);
      HasSucc[Edge.first] = true;
      HasPred[Edge.second] = true;
    }
    for (size_t I = 0; I < N; ++I) {
      if (!HasPred[I])
        Result.Entry.insert(Result.Entry.end(), Children[I].Entry.begin(),
                            Children[I].Entry.end());
      if (!HasSucc[I])
        Result.Exit.insert(Result.Exit.end(), Children[I].Exit.begin(),
                           Children[I].Exit.end());
    }
    break;
  }
  }
  return Result;
}

void collectLeaves(const SpecTy &Spec, std::vector<int> &Out) {
  if (Spec.isLeaf()) {
    Out.push_back(Spec.Id);
    return;
  }
  for (const SpecTy &Child : Spec.Children)
    collectLeaves(Child, Out);
}

//===----------------------------------------------------------------------===//
// The assertion every test is built from
//===----------------------------------------------------------------------===//

/// Check the shape of every item's waits: they name only items placed before
/// it, none on its own lane, at most one per other lane, and none that the
/// item's lane or its other waits already imply.  That last one is what makes
/// the waits minimal, and is the property the lowering's clocks exist for.
void checkWaits(const PlacementsTy &P, const HappensBeforeTy &HB,
                const std::string &Context) {
  for (uint32_t I = 0; I < P.Items.size(); ++I) {
    const PlacementsTy::ItemTy &Item = P.Items[I];
    std::set<uint32_t> LanesWaited;
    for (uint32_t Wait : Item.Waits) {
      EXPECT_LT(Wait, I) << Context << ": item " << I
                         << " waits for an item placed after it";
      if (Wait >= I)
        continue;
      const uint32_t WaitLane = P.Items[Wait].Lane;
      EXPECT_NE(WaitLane, Item.Lane)
          << Context << ": item " << I << " waits on its own lane";
      EXPECT_TRUE(LanesWaited.insert(WaitLane).second)
          << Context << ": item " << I << " waits twice on lane " << WaitLane;

      // Implied by the item before it on its lane, or by another wait.
      std::vector<uint32_t> Others(Item.Waits.begin(), Item.Waits.end());
      if (HB.LanePred[I] != ~0U)
        Others.push_back(HB.LanePred[I]);
      for (uint32_t Other : Others)
        EXPECT_TRUE(Other == Wait || !reachable(HB, Wait, Other))
            << Context << ": item " << I << " waits for " << Wait << ", which "
            << Other << " already implies";
    }
  }
}

/// Lower \p Spec onto \p NumLanes lanes and check the result against the
/// oracle.  \p Context names the case in any failure message.
HappensBeforeTy checkSchedule(const SpecTy &Spec, uint32_t NumLanes,
                              const std::string &Context) {
  BuiltGraphTy Built(Spec);
  EXPECT_EQ(Built.whyMalformed(), nullptr)
      << Context << ": well-formed graph rejected";
  auto POrErr = lowerGraph(Built.graph(), NumLanes);
  EXPECT_TRUE(static_cast<bool>(POrErr)) << Context << ": lowering failed";
  if (!POrErr) {
    consumeError(POrErr.takeError());
    return {};
  }

  HappensBeforeTy HB = buildHappensBefore(*POrErr, Built);

  std::vector<int> Leaves;
  collectLeaves(Spec, Leaves);
  EXPECT_EQ(POrErr->Items.size(), Leaves.size())
      << Context << ": every leaf must be placed exactly once";
  for (int Leaf : Leaves)
    EXPECT_TRUE(HB.ItemOfLeaf.count(Leaf))
        << Context << ": leaf " << Leaf << " was not placed";

  for (const std::pair<int, int> &Edge : analyze(Spec).Edges)
    EXPECT_TRUE(happensBefore(HB, Edge.first, Edge.second))
        << Context << ": dependence " << Edge.first << " -> " << Edge.second
        << " is not enforced on " << NumLanes << " lanes";

  EXPECT_TRUE(isAcyclic(HB)) << Context << ": lowering can deadlock";
  checkWaits(*POrErr, HB, Context);
  return HB;
}

/// Same, over a spread of lane counts: one lane (everything collides), a couple
/// of lanes (items collide selectively), and more lanes than the tree can use.
void checkScheduleAcrossLanes(const SpecTy &Spec, const std::string &Context,
                              std::vector<uint32_t> LaneCounts = {1, 2, 3, 4, 5,
                                                                  8}) {
  for (uint32_t NumLanes : LaneCounts)
    checkSchedule(Spec, NumLanes, Context + "/" + std::to_string(NumLanes));
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(TaskGraphSchedule, LeavesAreScheduledOnce) {
  SpecTy Spec = seq({target(1), enterData(2), exitData(3), updateData(4)});
  BuiltGraphTy Built(Spec);
  auto POrErr = lowerGraph(Built.graph(), 4);
  ASSERT_TRUE(static_cast<bool>(POrErr));
  EXPECT_EQ(POrErr->Items.size(), 4u);

  // Every leaf kind reaches the sink as the leaf it was recorded as.
  std::map<int, TGKind> Kinds;
  for (const PlacementsTy::ItemTy &Item : POrErr->Items)
    Kinds[Built.idOf(Item.Node)] = Item.Node->getKind();
  EXPECT_EQ(Kinds[1], TGKind::TargetNode);
  EXPECT_EQ(Kinds[2], TGKind::TargetEnterDataNode);
  EXPECT_EQ(Kinds[3], TGKind::TargetExitDataNode);
  EXPECT_EQ(Kinds[4], TGKind::TargetUpdateDataNode);
}

TEST(TaskGraphSchedule, SingleLeafNeedsNoWaits) {
  SpecTy Spec = target(1);
  BuiltGraphTy Built(Spec);
  auto POrErr = lowerGraph(Built.graph(), 4);
  ASSERT_TRUE(static_cast<bool>(POrErr));
  ASSERT_EQ(POrErr->Items.size(), 1u);
  EXPECT_TRUE(POrErr->Items[0].Waits.empty());
}

TEST(TaskGraphSchedule, SequentialRegionIsTotallyOrdered) {
  SpecTy Spec = seq({target(1), target(2), target(3), target(4)});
  HappensBeforeTy HB = checkSchedule(Spec, 4, "seq");
  EXPECT_TRUE(happensBefore(HB, 1, 4));
}

TEST(TaskGraphSchedule, ExclusiveRegionSerializes) {
  SpecTy Spec = excl({target(1), target(2), target(3)});
  HappensBeforeTy HB = checkSchedule(Spec, 4, "excl");
  EXPECT_TRUE(happensBefore(HB, 1, 3));
}

TEST(TaskGraphSchedule, ParallelRegionIsUnordered) {
  SpecTy Spec = par({target(1), target(2), target(3), target(4)});
  EXPECT_TRUE(analyze(Spec).Edges.empty());
  HappensBeforeTy HB = checkSchedule(Spec, 4, "par");
  // Unordered means unordered: nothing is forced either way.
  EXPECT_FALSE(happensBefore(HB, 1, 2));
  EXPECT_FALSE(happensBefore(HB, 2, 1));
}

TEST(TaskGraphSchedule, EveryParallelChildPrecedesASuccessor) {
  // The join has to wait for children spread over several lanes.
  SpecTy Spec = seq({par({target(1), target(2), target(3)}), target(4)});
  checkScheduleAcrossLanes(Spec, "par-then-seq");
  HappensBeforeTy HB = checkSchedule(Spec, 4, "par-then-seq/4");
  for (int Child : {1, 2, 3})
    EXPECT_TRUE(happensBefore(HB, Child, 4));
}

TEST(TaskGraphSchedule, IrreducibleDiamond) {
  SpecTy Spec = irred({target(1), target(2), target(3), target(4)},
                      {{0, 1}, {0, 2}, {1, 3}, {2, 3}});
  checkScheduleAcrossLanes(Spec, "diamond");
  HappensBeforeTy HB = checkSchedule(Spec, 4, "diamond/4");
  EXPECT_TRUE(happensBefore(HB, 1, 4));
  // The two middle leaves are concurrent and must not be ordered.
  EXPECT_FALSE(happensBefore(HB, 2, 3));
  EXPECT_FALSE(happensBefore(HB, 3, 2));
}

TEST(TaskGraphSchedule, IrreducibleMultiPredecessorJoin) {
  // Three producers into one consumer, then a further consumer: with fewer
  // lanes than producers some of them share a lane, which is what the
  // latest-producer-per-lane reduction has to get right.
  SpecTy Spec = irred({target(1), target(2), target(3), target(4), target(5)},
                      {{0, 3}, {1, 3}, {2, 3}, {3, 4}});
  checkScheduleAcrossLanes(Spec, "join");
  HappensBeforeTy HB = checkSchedule(Spec, 2, "join/2");
  for (int Producer : {1, 2, 3})
    EXPECT_TRUE(happensBefore(HB, Producer, 4));
  EXPECT_TRUE(happensBefore(HB, 1, 5));
}

TEST(TaskGraphSchedule, IrreducibleWithinIrreducible) {
  // The case the same-lane elision is most likely to get wrong: each child of
  // the outer region is itself irreducible, so an outer edge's endpoints are
  // whole sets of leaves that end up on different lanes.
  SpecTy A = irred({target(10), target(11), target(12), target(13)},
                   {{0, 1}, {0, 2}, {1, 3}, {2, 3}});
  SpecTy B = irred({target(20), target(21), target(22)}, {{0, 1}, {0, 2}});
  SpecTy C = irred({target(30), target(31), target(32)}, {{0, 2}, {1, 2}});
  SpecTy Spec = irred({A, B, C}, {{0, 1}, {1, 2}});

  checkScheduleAcrossLanes(Spec, "nested");
  HappensBeforeTy HB = checkSchedule(Spec, 3, "nested/3");
  EXPECT_TRUE(happensBefore(HB, 13, 20));
  for (int Exit : {21, 22})
    for (int Entry : {30, 31})
      EXPECT_TRUE(happensBefore(HB, Exit, Entry));
  EXPECT_TRUE(happensBefore(HB, 10, 32));
}

TEST(TaskGraphSchedule, DeepHeterogeneousNesting) {
  auto stage = [](int Base) {
    return irred({enterData(Base), target(Base + 1),
                  irred({target(Base + 2), target(Base + 3)}, {{0, 1}}),
                  exitData(Base + 4)},
                 {{0, 1}, {1, 2}, {2, 3}});
  };
  SpecTy Spec = irred({stage(40), stage(50)}, {{0, 1}});

  checkScheduleAcrossLanes(Spec, "deep");
  HappensBeforeTy HB = checkSchedule(Spec, 4, "deep/4");
  EXPECT_TRUE(happensBefore(HB, 44, 50));
  EXPECT_TRUE(happensBefore(HB, 40, 44));
  EXPECT_TRUE(happensBefore(HB, 40, 54));
}

TEST(TaskGraphSchedule, ChainLongerThanLaneCount) {
  // A chain of ten leaves on two lanes: consecutive leaves land on alternating
  // lanes, so nearly every dependence needs a real wait.
  std::vector<SpecTy> Children;
  std::vector<std::pair<int, int>> Edges;
  for (int I = 0; I < 10; ++I) {
    Children.push_back(target(I + 1));
    if (I)
      Edges.push_back({I - 1, I});
  }
  SpecTy Spec = irred(std::move(Children), std::move(Edges));

  HappensBeforeTy HB = checkSchedule(Spec, 2, "chain/2");
  for (int I = 1; I < 10; ++I)
    EXPECT_TRUE(happensBefore(HB, I, I + 1));
  EXPECT_TRUE(happensBefore(HB, 1, 10));
}

TEST(TaskGraphSchedule, ParallelRegionAsEdgeSource) {
  // The source of an irreducible edge is a region whose exits span lanes, so a
  // single wait cannot cover it.
  SpecTy Spec =
      irred({par({target(1), target(2), target(3)}), target(4)}, {{0, 1}});
  checkScheduleAcrossLanes(Spec, "par-source");
  HappensBeforeTy HB = checkSchedule(Spec, 4, "par-source/4");
  for (int Producer : {1, 2, 3})
    EXPECT_TRUE(happensBefore(HB, Producer, 4));
}

TEST(TaskGraphSchedule, IrreducibleInParallelInIrreducible) {
  SpecTy Inner = irred({target(10), target(11)}, {{0, 1}});
  SpecTy Spec = irred({par({Inner, target(12)}), target(13)}, {{0, 1}});
  checkScheduleAcrossLanes(Spec, "mixed");
  HappensBeforeTy HB = checkSchedule(Spec, 5, "mixed/5");
  EXPECT_TRUE(happensBefore(HB, 10, 11));
  EXPECT_TRUE(happensBefore(HB, 11, 13));
  EXPECT_TRUE(happensBefore(HB, 12, 13));
}

TEST(TaskGraphSchedule, SequenceOfMultiExitIrreducibleRegions) {
  SpecTy First = irred({target(20), target(21), target(22)}, {{0, 1}, {0, 2}});
  SpecTy Second = irred({target(30), target(31), target(32)}, {{0, 2}, {1, 2}});
  SpecTy Spec = seq({First, Second});
  checkScheduleAcrossLanes(Spec, "seq-of-irreds");
  HappensBeforeTy HB = checkSchedule(Spec, 3, "seq-of-irreds/3");
  for (int Exit : {21, 22})
    for (int Entry : {30, 31})
      EXPECT_TRUE(happensBefore(HB, Exit, Entry));
}

TEST(TaskGraphSchedule, WideFanIn) {
  // Six producers into one consumer at lane counts that force several producers
  // per lane, exercising the per-lane latest-producer reduction.
  const int NumProducers = 6;
  std::vector<SpecTy> Children;
  std::vector<std::pair<int, int>> Edges;
  for (int I = 0; I <= NumProducers; ++I)
    Children.push_back(target(I + 1));
  for (int I = 0; I < NumProducers; ++I)
    Edges.push_back({I, NumProducers});
  SpecTy Spec = irred(std::move(Children), std::move(Edges));

  for (uint32_t NumLanes : {2u, 3u, 4u, 5u}) {
    HappensBeforeTy HB =
        checkSchedule(Spec, NumLanes, "fan-in/" + std::to_string(NumLanes));
    for (int Producer = 1; Producer <= NumProducers; ++Producer)
      EXPECT_TRUE(happensBefore(HB, Producer, NumProducers + 1));
  }
}

TEST(TaskGraphSchedule, EmptyRegionPassesDependencesThrough) {
  // A region that emits nothing must not cut the chain around it.  The leaves
  // either side have to be on different lanes for this to test anything: on one
  // lane in-order execution would order them whatever the schedule says, so the
  // predecessors here are a parallel region, which spreads them over lanes.
  SpecTy Spec = seq({par({target(1), target(2)}), par({}), target(3)});
  HappensBeforeTy HB = checkSchedule(Spec, 4, "empty-region");
  EXPECT_TRUE(happensBefore(HB, 1, 3));
  EXPECT_TRUE(happensBefore(HB, 2, 3));
}

TEST(TaskGraphSchedule, WaitImpliedByAnotherWaitIsDropped) {
  // A triangle over three lanes: 2 waits for 1, and 3 for both.  Keeping the
  // latest producer per lane would leave 3 waiting for 1 and 2, though 2 has
  // already waited for 1; one wait is all 3 needs.
  SpecTy Spec =
      irred({target(1), target(2), target(3)}, {{0, 1}, {0, 2}, {1, 2}});
  BuiltGraphTy Built(Spec);
  auto POrErr = lowerGraph(Built.graph(), 3);
  ASSERT_TRUE(static_cast<bool>(POrErr));
  HappensBeforeTy HB = buildHappensBefore(*POrErr, Built);
  const PlacementsTy::ItemTy &Third = POrErr->Items[HB.ItemOfLeaf[3]];
  ASSERT_EQ(Third.Lane, 2u) << "the triangle should span all three lanes";
  EXPECT_EQ(Third.Waits, std::vector<uint32_t>{HB.ItemOfLeaf[2]});
  checkScheduleAcrossLanes(Spec, "triangle");
}

TEST(TaskGraphSchedule, CyclicIrreducibleRegionIsRejected) {
  // Picking some order for a cycle would silently produce a schedule that does
  // not implement the graph, so this has to be an error.
  for (const SpecTy &Spec :
       {irred({target(1), target(2)}, {{0, 1}, {1, 0}}),
        irred({target(1), target(2), target(3)}, {{0, 1}, {1, 2}, {2, 0}})}) {
    BuiltGraphTy Built(Spec);
    auto POrErr = lowerGraph(Built.graph(), 4);
    EXPECT_FALSE(static_cast<bool>(POrErr));
    if (!POrErr)
      consumeError(POrErr.takeError());
  }
}

TEST(TaskGraphStructure, WellFormedGraphIsAccepted) {
  BuiltGraphTy Built(seq(
      {target(1), irred({target(2), target(3), target(4)}, {{0, 1}, {0, 2}})}));
  EXPECT_EQ(Built.whyMalformed(), nullptr);
}

TEST(TaskGraphStructure, MalformedGraphIsRejected) {
  // libomptarget checks these once so that no backend has to, which is what
  // keeps a backend from silently dropping a dependence it cannot place.
  for (const SpecTy &Spec :
       {irred({target(1), target(2)}, {{0, 1}, {1, 0}}),
        irred({target(1), target(2), target(3)}, {{0, 1}, {1, 2}, {2, 0}}),
        irred({target(1), target(2)}, {{0, 2}}),
        irred({target(1), target(2)}, {{-1, 1}})}) {
    BuiltGraphTy Built(Spec);
    EXPECT_NE(Built.whyMalformed(), nullptr);
  }
}

TEST(TaskGraphSchedule, ZeroLanesIsRejected) {
  SpecTy Spec = target(1);
  BuiltGraphTy Built(Spec);
  auto POrErr = lowerGraph(Built.graph(), 0);
  EXPECT_FALSE(static_cast<bool>(POrErr));
  if (!POrErr)
    consumeError(POrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Randomized sweep
//===----------------------------------------------------------------------===//

/// Generate an arbitrary region tree.  Irreducible edge sets only go from lower
/// to higher child ordinals, which keeps them acyclic (cycles are covered by
/// their own test) while still producing every DAG shape.
SpecTy generateSpec(std::mt19937 &Rng, int Depth, int &NextId) {
  const int LeafPercent = Depth <= 0 ? 100 : 35;
  if (static_cast<int>(Rng() % 100) < LeafPercent) {
    int Id = ++NextId;
    switch (Rng() % 4) {
    case 0:
      return target(Id);
    case 1:
      return enterData(Id);
    case 2:
      return updateData(Id);
    default:
      return exitData(Id);
    }
  }

  const size_t NumChildren = 2 + Rng() % 3;
  std::vector<SpecTy> Children;
  for (size_t I = 0; I < NumChildren; ++I)
    Children.push_back(generateSpec(Rng, Depth - 1, NextId));

  switch (Rng() % 4) {
  case 0:
    return seq(std::move(Children));
  case 1:
    return par(std::move(Children));
  case 2:
    return excl(std::move(Children));
  default: {
    std::vector<std::pair<int, int>> Edges;
    for (size_t I = 0; I < NumChildren; ++I)
      for (size_t J = I + 1; J < NumChildren; ++J)
        if (Rng() % 100 < 45)
          Edges.push_back({static_cast<int>(I), static_cast<int>(J)});
    return irred(std::move(Children), std::move(Edges));
  }
  }
}

TEST(TaskGraphSchedule, RandomizedTreesAgainstOracle) {
  const unsigned NumTrees = 400;
  const unsigned BaseSeed = 0xC0FFEE;
  for (unsigned I = 0; I < NumTrees; ++I) {
    const unsigned Seed = BaseSeed + I;
    std::mt19937 Rng(Seed);
    int NextId = 0;
    SpecTy Spec = generateSpec(Rng, /*Depth=*/4, NextId);
    for (uint32_t NumLanes : {1u, 2u, 3u, 5u, 8u})
      checkSchedule(Spec, NumLanes, "seed=" + std::to_string(Seed));
  }
}

} // namespace
