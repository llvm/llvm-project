//===--------- WaitingOnGraphTest.cpp - Test WaitingOnGraph APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/WaitingOnGraph.h"
#include "gtest/gtest.h"

namespace llvm::orc::detail {

class WaitingOnGraphTest : public testing::Test {
public:
  using TestGraph = WaitingOnGraph<uintptr_t, uintptr_t>;

protected:
  using SuperNode = TestGraph::SuperNode;
  using SuperNodeBuilder = TestGraph::SuperNodeBuilder;
  using ContainerElementsMap = TestGraph::ContainerElementsMap;
  using ElemToSuperNodeMap = TestGraph::ElemToSuperNodeMap;
  using SimplifyResult = TestGraph::SimplifyResult;
  using EmitResult = TestGraph::EmitResult;

  static const ContainerElementsMap &getDefs(SuperNode &SN) { return SN.Defs; }

  static const ContainerElementsMap &getDeps(SuperNode &SN) { return SN.Deps; }

  static std::vector<std::unique_ptr<SuperNode>> &getSNs(SimplifyResult &SR) {
    return SR.SNs;
  }

  static ElemToSuperNodeMap &getElemToSN(SimplifyResult &SR) {
    return SR.ElemToSN;
  }

  static std::vector<std::unique_ptr<SuperNode>> &getPendingSNs(TestGraph &G) {
    return G.PendingSNs;
  }

  static ContainerElementsMap merge(ContainerElementsMap M1,
                                    const ContainerElementsMap &M2) {
    ContainerElementsMap Result = std::move(M1);
    for (auto &[Container, Elems] : M2)
      Result[Container].insert(Elems.begin(), Elems.end());
    return Result;
  }

  ContainerElementsMap
  collapseDefs(std::vector<std::unique_ptr<SuperNode>> &SNs,
               bool DepsMustMatch = true) {
    if (SNs.empty())
      return ContainerElementsMap();

    ContainerElementsMap Result = SNs[0]->defs();
#ifndef NDEBUG
    const ContainerElementsMap &Deps = SNs[0]->deps();
#endif // NDEBUG

    for (size_t I = 1; I != SNs.size(); ++I) {
      assert(!DepsMustMatch || SNs[I]->deps() == Deps);
      Result = merge(std::move(Result), SNs[I]->defs());
    }

    return Result;
  }

  EmitResult integrate(EmitResult ER) {
    for (auto &SN : ER.Ready)
      for (auto &[Container, Elems] : SN->defs())
        Ready[Container].insert(Elems.begin(), Elems.end());
    for (auto &SN : ER.Failed)
      for (auto &[Container, Elems] : SN->defs())
        Failed[Container].insert(Elems.begin(), Elems.end());
    return ER;
  }

  EmitResult emit(SimplifyResult SR) {
    return integrate(G.emit(std::move(SR), GetExternalState));
  }

  TestGraph G;
  ContainerElementsMap Ready;
  ContainerElementsMap Failed;

  class ExternalStateGetter {
  public:
    ExternalStateGetter(WaitingOnGraphTest &T) : T(T) {}
    TestGraph::ExternalState operator()(TestGraph::ContainerId C,
                                        TestGraph::ElementId E) {
      {
        auto I = T.Failed.find(C);
        if (I != T.Failed.end())
          if (I->second.count(E))
            return TestGraph::ExternalState::Failed;
      }

      {
        auto I = T.Ready.find(C);
        if (I != T.Ready.end())
          if (I->second.count(E))
            return TestGraph::ExternalState::Ready;
      }

      return TestGraph::ExternalState::None;
    }

  private:
    WaitingOnGraphTest &T;
  };

  ExternalStateGetter GetExternalState{*this};
};

} // namespace llvm::orc::detail

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::detail;

TEST_F(WaitingOnGraphTest, ConstructAndDestroyEmpty) {
  // Nothing to do here -- we're just testing construction and destruction
  // of the WaitingOnGraphTest::G member.
}

TEST_F(WaitingOnGraphTest, Build_TrivialSingleSuperNode) {
  // Add one set of trivial defs and empty deps to the builder, make sure that
  // they're passed through to the resulting super-node.
  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {0}}});
  ContainerElementsMap Deps;
  B.add(Defs, Deps);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs[0]), Defs);
  EXPECT_EQ(getDeps(*SNs[0]), Deps);
}

TEST_F(WaitingOnGraphTest, Build_EmptyDefs) {
  // Adding empty def sets is ok, but should not result in creation of a
  // SuperNode.
  SuperNodeBuilder B;
  ContainerElementsMap Empty;
  B.add(Empty, Empty);
  auto SNs = B.takeSuperNodes();
  EXPECT_TRUE(SNs.empty());
}

TEST_F(WaitingOnGraphTest, Build_NonTrivialSingleSuperNode) {
  // Add one non-trivwial set of defs and deps. Make sure that they're passed
  // through to the resulting super-node.
  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {0, 1, 2}}});
  ContainerElementsMap Deps({{1, {3, 4, 5}}});
  B.add(Defs, Deps);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs[0]), Defs);
  EXPECT_EQ(getDeps(*SNs[0]), Deps);
}

TEST_F(WaitingOnGraphTest, Build_CoalesceEmptyDeps) {
  // Add two trivial defs both with empty deps to the builder. Check that
  // they're coalesced into a single super-node.
  SuperNodeBuilder B;
  ContainerElementsMap Defs1({{0, {0}}});
  ContainerElementsMap Defs2({{0, {1}}});
  ContainerElementsMap Deps;
  B.add(Defs1, Deps);
  B.add(Defs2, Deps);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs[0]), merge(Defs1, Defs2));
  EXPECT_EQ(getDeps(*SNs[0]), Deps);
}

TEST_F(WaitingOnGraphTest, Build_CoalesceNonEmptyDeps) {
  // Add two sets trivial of trivial defs with empty deps to the builder. Check
  // that the two coalesce into a single super node.
  SuperNodeBuilder B;
  ContainerElementsMap Defs1({{0, {0}}});
  ContainerElementsMap Defs2({{0, {1}}});
  ContainerElementsMap Deps({{1, {1}}});
  B.add(Defs1, Deps);
  B.add(Defs2, Deps);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs[0]), merge(Defs1, Defs2));
  EXPECT_EQ(getDeps(*SNs[0]), Deps);
}

TEST_F(WaitingOnGraphTest, Build_CoalesceInterleaved) {
  // Add multiple sets of defs, some with the same dep sets. Check that nodes
  // are still coalesced as expected.
  SuperNodeBuilder B;

  ContainerElementsMap DefsA1({{0, {0}}});
  ContainerElementsMap DefsA2({{0, {1}}});
  ContainerElementsMap DefsB1({{1, {0}}});
  ContainerElementsMap DefsB2({{1, {1}}});
  ContainerElementsMap DepsA({{2, {0}}, {3, {0}}});
  ContainerElementsMap DepsB({{4, {0}}, {5, {0}}});
  B.add(DefsA1, DepsA);
  B.add(DefsB1, DepsB);
  B.add(DefsA2, DepsA);
  B.add(DefsB2, DepsB);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 2U);
  EXPECT_EQ(getDefs(*SNs[0]), merge(DefsA1, DefsA2));
  EXPECT_EQ(getDeps(*SNs[0]), DepsA);
  EXPECT_EQ(getDefs(*SNs[1]), merge(DefsB1, DefsB2));
  EXPECT_EQ(getDeps(*SNs[1]), DepsB);
}

TEST_F(WaitingOnGraphTest, Build_SelfDepRemoval) {
  // Add multiple sets of defs, some with the same dep sets. Check that nodes
  // are still coalesced as expected.
  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {0, 1}}});
  ContainerElementsMap Deps({{0, {1}}});
  ContainerElementsMap Empty;
  B.add(Defs, Deps);
  auto SNs = B.takeSuperNodes();
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs[0]), Defs);
  EXPECT_EQ(getDeps(*SNs[0]), Empty);
}

TEST_F(WaitingOnGraphTest, Simplification_EmptySimplification) {
  auto SR = TestGraph::simplify({});
  auto &SNs = getSNs(SR);
  EXPECT_EQ(SNs.size(), 0U);
  EXPECT_EQ(getElemToSN(SR), ElemToSuperNodeMap());
}

TEST_F(WaitingOnGraphTest, Simplification_TrivialSingleSuperNode) {
  // Test trivial call to simplify.
  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {0}}});
  ContainerElementsMap Deps({{0, {0}}});
  B.add(Defs, Deps);
  auto SR = TestGraph::simplify(B.takeSuperNodes());
  ContainerElementsMap Empty;

  // Check SNs.
  auto &SNs = getSNs(SR);
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs.at(0)), Defs);
  EXPECT_EQ(getDeps(*SNs.at(0)), Empty);

  // Check ElemToSNs.
  ElemToSuperNodeMap ExpectedElemToSNs;
  ExpectedElemToSNs[0][0] = SNs[0].get();
  EXPECT_EQ(getElemToSN(SR), ExpectedElemToSNs);
}

TEST_F(WaitingOnGraphTest, Simplification_SimplifySingleContainerSimpleCycle) {
  // Test trivial simplification call with two nodes and one internal
  // dependence cycle within a single container:
  // N0: (0, 0) -> (0, 1)
  // N1: (0, 1) -> (0, 0)
  // We expect intra-simplify cycle elimination to clear both dependence sets,
  // and coalescing to join them into one supernode covering both defs.
  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{0, {1}}});
  B.add(Defs0, Deps0);
  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Deps1({{0, {0}}});
  B.add(Defs1, Deps1);
  auto SR = TestGraph::simplify(B.takeSuperNodes());

  // Check SNs.
  auto &SNs = getSNs(SR);
  ContainerElementsMap Empty;
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs.at(0)), merge(Defs0, Defs1));
  EXPECT_EQ(getDeps(*SNs.at(0)), Empty);

  // Check ElemToSNs.
  ElemToSuperNodeMap ExpectedElemToSNs;
  ExpectedElemToSNs[0][0] = SNs[0].get();
  ExpectedElemToSNs[0][1] = SNs[0].get();

  EXPECT_EQ(getElemToSN(SR), ExpectedElemToSNs);
}

TEST_F(WaitingOnGraphTest,
       Simplification_SimplifySingleContainerNElementCycle) {
  // Test trivial simplification call with M nodes and one internal
  // dependence cycle within a single container:
  // N0: (0, 0) -> (0, 1)
  // N1: (0, 1) -> (0, 2)
  // ...
  // NM: (0, M) -> (0, 0)
  // We expect intra-simplify cycle elimination to clear all dependence sets,
  // and coalescing to join them into one supernode covering all defs.
  SuperNodeBuilder B;
  constexpr size_t M = 10;
  for (size_t I = 0; I != M; ++I) {
    ContainerElementsMap Defs({{0, {I}}});
    ContainerElementsMap Deps({{0, {(I + 1) % M}}});
    B.add(Defs, Deps);
  }
  auto InitSNs = B.takeSuperNodes();
  EXPECT_EQ(InitSNs.size(), M);

  auto SR = TestGraph::simplify(std::move(InitSNs));

  // Check SNs.
  auto &SNs = getSNs(SR);
  ContainerElementsMap ExpectedDefs;
  for (size_t I = 0; I != M; ++I)
    ExpectedDefs[0].insert(I);
  ContainerElementsMap Empty;
  EXPECT_EQ(SNs.size(), 1U);
  EXPECT_EQ(getDefs(*SNs.at(0)), ExpectedDefs);
  EXPECT_EQ(getDeps(*SNs.at(0)), Empty);

  // Check ElemToSNs.
  ElemToSuperNodeMap ExpectedElemToSNs;
  for (size_t I = 0; I != M; ++I)
    ExpectedElemToSNs[0][I] = SNs[0].get();

  EXPECT_EQ(getElemToSN(SR), ExpectedElemToSNs);
}

TEST_F(WaitingOnGraphTest, Simplification_SimplifyIntraSimplifyPropagateDeps) {
  // Test trivial simplification call with two nodes and one internal
  // dependence cycle within a single container:
  // N0: (0, 0) -> (0, {1, 2})
  // N1: (0, 1) -> (0, {3})
  // We expect intra-simplify cycle elimination to replace the dependence of
  // (0, 0) on (0, 1) with a dependence on (0, 3) instead.
  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{0, {1, 2}}});
  B.add(Defs0, Deps0);
  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Deps1({{0, {3}}});
  B.add(Defs1, Deps1);
  auto SR = TestGraph::simplify(B.takeSuperNodes());

  // Check SNs.
  auto &SNs = getSNs(SR);
  EXPECT_EQ(SNs.size(), 2U);

  // ContainerElemenstMap ExpectedDefs0({{0, {0}}});
  // ContainerElemenstMap ExpectedDeps0({{0, {1, 3}}});
  EXPECT_EQ(getDefs(*SNs.at(0)), ContainerElementsMap({{0, {0}}}));
  EXPECT_EQ(getDeps(*SNs.at(0)), ContainerElementsMap({{0, {2, 3}}}));

  EXPECT_EQ(getDefs(*SNs.at(1)), ContainerElementsMap({{0, {1}}}));
  EXPECT_EQ(getDeps(*SNs.at(1)), ContainerElementsMap({{0, {3}}}));

  // Check ElemToSNs.
  ElemToSuperNodeMap ExpectedElemToSNs;
  ExpectedElemToSNs[0][0] = SNs[0].get();
  ExpectedElemToSNs[0][1] = SNs[1].get();

  EXPECT_EQ(getElemToSN(SR), ExpectedElemToSNs);
}

TEST_F(WaitingOnGraphTest, Emit_EmptyEmit) {
  // Check that empty emits work as expected.
  auto ER = G.emit(TestGraph::simplify({}), GetExternalState);

  EXPECT_EQ(ER.Ready.size(), 0U);
  EXPECT_EQ(ER.Failed.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Emit_TrivialSingleNode) {
  // Check that emitting a single node behaves as expected.
  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {0}}});
  B.add(Defs, ContainerElementsMap());
  auto ER = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(collapseDefs(ER.Ready), Defs);
  EXPECT_EQ(ER.Failed.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Emit_TrivialSequence) {
  // Perform a sequence of two emits where the second emit depends on the
  // first. Check that nodes become ready after each emit.
  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Empty;
  B.add(Defs0, Empty);
  auto ER0 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(collapseDefs(ER0.Ready), Defs0);
  EXPECT_EQ(ER0.Failed.size(), 0U);

  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Deps1({{0, {0}}});
  B.add(Defs1, Deps1);
  auto ER1 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(collapseDefs(ER1.Ready), Defs1);
  EXPECT_EQ(ER1.Failed.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Emit_TrivialReverseSequence) {
  // Perform a sequence of two emits where the first emit depends on the
  // second. Check that both nodes become ready after the second emit.
  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{0, {1}}});
  B.add(Defs0, Deps0);
  auto ER0 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER0.Ready.size(), 0U);
  EXPECT_EQ(ER0.Failed.size(), 0U);

  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Empty;
  B.add(Defs1, Empty);
  auto ER1 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(collapseDefs(ER1.Ready), merge(Defs0, Defs1));
  EXPECT_EQ(ER1.Failed.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Emit_Coalescing) {
  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{1, {0}}});
  B.add(Defs0, Deps0);
  auto ER0 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER0.Ready.size(), 0U);
  EXPECT_EQ(ER0.Failed.size(), 0U);

  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Deps1({{1, {0}}});
  B.add(Defs1, Deps1);
  auto ER1 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER1.Ready.size(), 0U);
  EXPECT_EQ(ER1.Failed.size(), 0U);

  // Check that after emitting two nodes with the same dep set we have only one
  // pending supernode whose defs are the union of the defs in the two emits.
  auto &PendingSNs = getPendingSNs(G);
  EXPECT_EQ(PendingSNs.size(), 1U);
  EXPECT_EQ(getDefs(*PendingSNs.at(0)), merge(Defs0, Defs1));

  ContainerElementsMap Defs2({{1, {0}}});
  ContainerElementsMap Empty;
  B.add(Defs2, Empty);
  auto ER2 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(collapseDefs(ER2.Ready), merge(merge(Defs0, Defs1), Defs2));
  EXPECT_EQ(ER2.Failed.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Emit_ZigZag) {
  // Perform a sequence of four emits, where the first three contain a zig-zag
  // pattern:
  // 1. (0, 0) -> (0, 1)
  // 2. (0, 2) -> (0, 3)
  //   ^ -- At this point we expect two pending supernodes.
  // 3. (0, 1) -> (0, 2)
  //   ^ -- Resolution of (0, 1) should cause all three emitted nodes to coalsce
  //        into one supernode defining (0, {1, 2, 3}).
  // 4. (0, 3)
  //   ^ -- Should cause all four nodes to become ready.

  SuperNodeBuilder B;
  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{0, {1}}});
  B.add(Defs0, Deps0);
  auto ER0 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER0.Ready.size(), 0U);
  EXPECT_EQ(ER0.Failed.size(), 0U);

  ContainerElementsMap Defs1({{0, {2}}});
  ContainerElementsMap Deps1({{0, {3}}});
  B.add(Defs1, Deps1);
  auto ER1 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER1.Ready.size(), 0U);
  EXPECT_EQ(ER1.Failed.size(), 0U);

  // Check that after emitting two nodes with the same dep set we have only one
  // pending supernode whose defs are the union of the defs in the two emits.
  auto &PendingSNs = getPendingSNs(G);
  EXPECT_EQ(PendingSNs.size(), 2U);
  EXPECT_EQ(getDefs(*PendingSNs.at(0)), Defs0);
  EXPECT_EQ(getDeps(*PendingSNs.at(0)), Deps0);
  EXPECT_EQ(getDefs(*PendingSNs.at(1)), Defs1);
  EXPECT_EQ(getDeps(*PendingSNs.at(1)), Deps1);

  ContainerElementsMap Defs2({{0, {1}}});
  ContainerElementsMap Deps2({{0, {2}}});
  B.add(Defs2, Deps2);
  auto ER2 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER2.Ready.size(), 0U);
  EXPECT_EQ(ER2.Failed.size(), 0U);

  // Check that after emitting the third node we've coalesced all three.
  EXPECT_EQ(PendingSNs.size(), 1U);
  EXPECT_EQ(getDefs(*PendingSNs.at(0)), merge(merge(Defs0, Defs1), Defs2));
  EXPECT_EQ(getDeps(*PendingSNs.at(0)), Deps1);

  ContainerElementsMap Defs3({{0, {3}}});
  ContainerElementsMap Empty;
  B.add(Defs3, Empty);
  auto ER3 = emit(TestGraph::simplify(B.takeSuperNodes()));

  EXPECT_EQ(collapseDefs(ER3.Ready),
            merge(merge(merge(Defs0, Defs1), Defs2), Defs3));
  EXPECT_EQ(ER2.Failed.size(), 0U);
  EXPECT_TRUE(PendingSNs.empty());
}

TEST_F(WaitingOnGraphTest, Fail_Empty) {
  // Check that failing an empty set is a no-op.
  auto FR = G.fail(ContainerElementsMap());
  EXPECT_EQ(FR.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Fail_Single) {
  // Check that failing a set with no existing dependencies works.
  auto FR = G.fail({{0, {0}}});
  EXPECT_EQ(FR.size(), 0U);
}

TEST_F(WaitingOnGraphTest, Fail_EmitDependenceOnFailure) {
  // Check that emitted nodes that directly depend on failed nodes also fail.
  Failed = {{0, {0}}};

  SuperNodeBuilder B;
  ContainerElementsMap Defs({{0, {1}}});
  ContainerElementsMap Deps({{0, {0}}});
  B.add(Defs, Deps);
  auto ER = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER.Ready.size(), 0U);
  EXPECT_EQ(collapseDefs(ER.Failed, false), Defs);
}

TEST_F(WaitingOnGraphTest, Fail_ZigZag) {
  // Check that if an emit introduces a transitive dependence of a failed
  // node, then all nodes that depend on the failed node are also failed.
  SuperNodeBuilder B;

  ContainerElementsMap Defs0({{0, {0}}});
  ContainerElementsMap Deps0({{0, {1}}});
  B.add(Defs0, Deps0);
  auto ER0 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER0.Ready.size(), 0U);
  EXPECT_EQ(ER0.Failed.size(), 0U);

  Failed = {{0, {2}}};

  ContainerElementsMap Defs1({{0, {1}}});
  ContainerElementsMap Deps1({{0, {2}}});
  B.add(Defs1, Deps1);
  auto ER1 = emit(TestGraph::simplify(B.takeSuperNodes()));
  EXPECT_EQ(ER1.Ready.size(), 0U);
  EXPECT_EQ(collapseDefs(ER1.Failed, false), merge(Defs0, Defs1));
}
