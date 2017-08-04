//===- llvm/unittest/XRay/GraphTest.cpp - XRay Graph unit tests -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/XRay/Graph.h"
#include "gtest/gtest.h"
#include <iostream>
#include <set>
#include <type_traits>

using namespace llvm;
using namespace xray;

namespace {
struct VAttr {
  unsigned VA;
};
struct EAttr {
  unsigned EA;
};
typedef Graph<VAttr, EAttr, unsigned> GraphT;
typedef typename GraphT::VertexIdentifier VI;
typedef typename GraphT::EdgeIdentifier EI;

// Test Fixture
template <typename T> class GraphTest : public testing::Test {
protected:
  T Graph = getTestGraph();

private:
  static T getTestGraph() {
    using std::make_pair;
    typename std::remove_const<T>::type G;
    G.insert(make_pair(1u, VAttr({3u})));
    G.insert(make_pair(2u, VAttr({5u})));
    G.insert(make_pair(3u, VAttr({7u})));
    G.insert(make_pair(4u, VAttr({11u})));
    G.insert(make_pair(5u, VAttr({13u})));
    G.insert(make_pair(6u, VAttr({17u})));

    G.insert(std::make_pair(EI(1u, 2u), EAttr({3u * 5u})));
    G.insert(std::make_pair(EI(2u, 3u), EAttr({5u * 7u})));
    G.insert(std::make_pair(EI(6u, 3u), EAttr({2u * 7u * 17u})));
    G.insert(std::make_pair(EI(4u, 6u), EAttr({11u * 17u})));
    G.insert(std::make_pair(EI(2u, 4u), EAttr({5u * 11u})));
    G.insert(std::make_pair(EI(2u, 5u), EAttr({5u * 13u})));
    G.insert(std::make_pair(EI(4u, 5u), EAttr({11u * 13u})));

    return G;
  }
};

typedef ::testing::Types<GraphT, const GraphT> GraphTestTypes;

using VVT = typename GraphT::VertexValueType;
using EVT = typename GraphT::EdgeValueType;

TYPED_TEST_CASE(GraphTest, GraphTestTypes);

template <typename T> void graphVertexTester(T &G) {
  std::set<unsigned> V({1u, 2u, 3u, 4u, 5u, 6u});
  std::vector<unsigned> VA({0u, 3u, 5u, 7u, 11u, 13u, 17u});

  EXPECT_EQ(V.size(), G.vertices().size());
  EXPECT_FALSE(G.vertices().empty());
  for (unsigned u : V) {
    auto EVV = G.at(u);
    ASSERT_TRUE(!!EVV);
    EXPECT_EQ(1u, G.count(u));
    EXPECT_EQ(VA[u], EVV->VA);
    EXPECT_NE(G.vertices().end(),
              std::find_if(G.vertices().begin(), G.vertices().end(),
                           [&](const VVT &VV) { return VV.first == u; }));
    consumeError(EVV.takeError());
  }

  for (auto &VVT : G.vertices()) {
    EXPECT_EQ(1u, V.count(VVT.first));
    EXPECT_EQ(VA[VVT.first], VVT.second.VA);
  }
}

template <typename T> void graphEdgeTester(T &G) {
  std::set<unsigned> V({1u, 2u, 3u, 4u, 5u, 6u});

  std::set<std::pair<unsigned, unsigned>> E(
      {{1u, 2u}, {2u, 3u}, {6u, 3u}, {4u, 6u}, {2u, 4u}, {2u, 5u}, {4u, 5u}});
  std::vector<unsigned> VA({0u, 3u, 5u, 7u, 11u, 13u, 17u});

  EXPECT_EQ(E.size(), G.edges().size());
  EXPECT_FALSE(G.edges().empty());
  for (std::pair<unsigned, unsigned> u : E) {
    auto EEV = G.at(u);
    ASSERT_TRUE(!!EEV);
    EXPECT_EQ(1u, G.count(u));
    EXPECT_EQ(VA[u.first] * VA[u.second] * ((u.first > u.second) ? 2 : 1),
              EEV->EA);
    auto Pred = [&](const EVT &EV) { return EV.first == u; };
    EXPECT_NE(G.edges().end(),
              std::find_if(G.edges().begin(), G.edges().end(), Pred));
    consumeError(EEV.takeError());
  }

  for (auto &EV : G.edges()) {
    EXPECT_EQ(1u, E.count(EV.first));
    EXPECT_EQ(VA[EV.first.first] * VA[EV.first.second] *
                  ((EV.first.first > EV.first.second) ? 2 : 1),
              EV.second.EA);
    const auto &IE = G.inEdges(EV.first.second);
    const auto &OE = G.outEdges(EV.first.first);
    EXPECT_NE(IE.size(), 0u);
    EXPECT_NE(OE.size(), 0u);
    EXPECT_NE(IE.begin(), IE.end());
    EXPECT_NE(OE.begin(), OE.end());
    {
      auto It = std::find_if(
          G.inEdges(EV.first.second).begin(), G.inEdges(EV.first.second).end(),
          [&](const EVT &EVI) { return EVI.first == EV.first; });
      EXPECT_NE(G.inEdges(EV.first.second).end(), It);
    }
    {
      auto It = std::find_if(
          G.inEdges(EV.first.first).begin(), G.inEdges(EV.first.first).end(),
          [&](const EVT &EVI) { return EVI.first == EV.first; });
      EXPECT_EQ(G.inEdges(EV.first.first).end(), It);
    }
    {
      auto It =
          std::find_if(G.outEdges(EV.first.second).begin(),
                       G.outEdges(EV.first.second).end(),
                       [&](const EVT &EVI) { return EVI.first == EV.first; });
      EXPECT_EQ(G.outEdges(EV.first.second).end(), It);
    }
    {
      auto It = std::find_if(
          G.outEdges(EV.first.first).begin(), G.outEdges(EV.first.first).end(),
          [&](const EVT &EVI) { return EVI.first == EV.first; });
      EXPECT_NE(G.outEdges(EV.first.first).end(), It);
    }
  }
}

TYPED_TEST(GraphTest, TestGraphEdge) {
  auto &G = this->Graph;

  graphEdgeTester(G);
}

TYPED_TEST(GraphTest, TestGraphVertex) {
  auto &G = this->Graph;

  graphVertexTester(G);
}

TYPED_TEST(GraphTest, TestCopyConstructor) {
  TypeParam G(this->Graph);

  graphEdgeTester(G);
  graphVertexTester(G);
}

TYPED_TEST(GraphTest, TestCopyAssign) {
  TypeParam G = this->Graph;

  graphEdgeTester(G);
  graphVertexTester(G);
}

TYPED_TEST(GraphTest, TestMoveConstructor) {
  TypeParam G(std::move(this->Graph));

  graphEdgeTester(G);
  graphVertexTester(G);
}

// Tests the incremental Construction of a graph
TEST(GraphTest, TestConstruction) {
  GraphT MG;
  const GraphT &G = MG;
  EXPECT_EQ(0u, G.count(0u));
  EXPECT_EQ(0u, G.count({0u, 1u}));
  auto VE = G.at(0);
  auto EE = G.at({0, 0});
  EXPECT_FALSE(VE); // G.at[0] returns an error
  EXPECT_FALSE(EE); // G.at[{0,0}] returns an error
  consumeError(VE.takeError());
  consumeError(EE.takeError());
  EXPECT_TRUE(G.vertices().empty());
  EXPECT_TRUE(G.edges().empty());
  EXPECT_EQ(G.vertices().begin(), G.vertices().end());
  EXPECT_EQ(G.edges().begin(), G.edges().end());
}

TEST(GraphTest, TestiVertexAccessOperator) {
  GraphT MG;
  const GraphT &G = MG;

  MG[0u] = {1u};
  EXPECT_EQ(1u, MG[0u].VA);
  EXPECT_EQ(1u, G.count(0u));
  EXPECT_EQ(0u, G.count(1u));
  EXPECT_EQ(1u, MG[0u].VA);
  auto T = G.at(0u);
  EXPECT_TRUE(!!T);
  EXPECT_EQ(1u, T->VA);

  EXPECT_EQ(1u, G.vertices().size());
  EXPECT_EQ(0u, G.edges().size());
  EXPECT_FALSE(G.vertices().empty());
  EXPECT_TRUE(G.edges().empty());
  EXPECT_NE(G.vertices().begin(), G.vertices().end());
  EXPECT_EQ(G.edges().begin(), G.edges().end());
  EXPECT_EQ(1u, G.vertices().begin()->second.VA);
  EXPECT_EQ(0u, G.vertices().begin()->first);
  EXPECT_EQ(0u, G.outEdges(0u).size());
  EXPECT_TRUE(G.outEdges(0u).empty());
  EXPECT_EQ(G.outEdges(0u).begin(), G.outEdges(0u).end());
  EXPECT_EQ(0u, G.inEdges(0u).size());
  EXPECT_TRUE(G.inEdges(0u).empty());
  EXPECT_EQ(G.inEdges(0u).begin(), G.inEdges(0u).end());
}

TEST(GraphTest, TestEdgeAccessOperator) {
  GraphT MG;
  const GraphT &G = MG;

  MG[{0u, 0u}] = {2u};
  EI EdgeIdent({0u, 0u});
  EXPECT_EQ(2u, MG[EdgeIdent].EA);
  EXPECT_EQ(1u, G.count({0u, 0u}));
  EXPECT_EQ(0u, G.count({0u, 1u}));
  EXPECT_EQ(1u, G.count(0u));
  EXPECT_NE(1u, G.count(1u));
  auto T = G.at({0u, 0u});
  EXPECT_TRUE(T && T->EA == 2u);
  EXPECT_EQ(1u, G.edges().size());
  EXPECT_EQ(1u, G.vertices().size());
  EXPECT_FALSE(G.edges().empty());
  EXPECT_FALSE(G.vertices().empty());
  EXPECT_NE(G.edges().begin(), G.edges().end());
  EXPECT_EQ(EI(0u, 0u), G.edges().begin()->first);
  EXPECT_EQ(2u, G.edges().begin()->second.EA);
  EXPECT_EQ(1u, G.outEdges(0u).size());
  EXPECT_FALSE(G.outEdges(0u).empty());
  EXPECT_NE(G.outEdges(0u).begin(), G.outEdges(0u).end());
  EXPECT_EQ(EI(0u, 0u), G.outEdges(0u).begin()->first);
  EXPECT_EQ(2u, G.outEdges(0u).begin()->second.EA);
  EXPECT_EQ(++(G.outEdges(0u).begin()), G.outEdges(0u).end());
  EXPECT_EQ(1u, G.inEdges(0u).size());
  EXPECT_FALSE(G.inEdges(0u).empty());
  EXPECT_NE(G.inEdges(0u).begin(), G.inEdges(0u).end());
  EXPECT_EQ(EI(0u, 0u), G.inEdges(0u).begin()->first);
  EXPECT_EQ(2u, G.inEdges(0u).begin()->second.EA);
  EXPECT_EQ(++(G.inEdges(0u).begin()), G.inEdges(0u).end());
}
}
