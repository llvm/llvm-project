//===- RootOrderingTest.cpp - unit tests for optimal branching ------------===//
//
// Part of the LLVM Project, under the Apache License v[1].0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Conversion/PDLToPDLInterp/RootOrdering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::pdl_to_pdl_interp;

namespace {

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

/// The test fixture for constructing root ordering tests and verifying results.
/// This fixture constructs the test values v. The test populates the graph
/// with the desired costs and then calls check(), passing the expected optimal
/// cost and the list of edges in the preorder traversal of the optimal
/// branching.
class RootOrderingTest : public ::testing::Test {
protected:
  RootOrderingTest() {
    context.loadDialect<ArithDialect>();
    createValues();
  }

  /// Creates the test values. These values simply act as vertices / vertex IDs
  /// in the cost graph, rather than being a part of an IR.
  void createValues() {
    OpBuilder builder(&context);
    builder.setInsertionPointToStart(&block);
    for (int i = 0; i < 4; ++i)
      // Ops will be deleted when `block` is destroyed.
      v[i] = builder.create<ConstantIntOp>(builder.getUnknownLoc(), i, 32);
  }

  /// Checks that optimal branching on graph has the given cost and
  /// its preorder traversal results in the specified edges.
  void check(unsigned cost, const OptimalBranching::EdgeList &edges) {
    OptimalBranching opt(graph, v[0]);
    EXPECT_EQ(opt.solve(), cost);
    EXPECT_EQ(opt.preOrderTraversal({v, v + edges.size()}), edges);
    for (std::pair<Value, Value> edge : edges)
      EXPECT_EQ(opt.getRootOrderingParents().lookup(edge.first), edge.second);
  }

protected:
  /// The context for creating the values.
  MLIRContext context;

  /// Block holding all the operations.
  Block block;

  /// Values used in the graph definition. We always use leading `n` values.
  Value v[4];

  /// The graph being tested on.
  RootOrderingGraph graph;
};

//===----------------------------------------------------------------------===//
// Simple 3-node graphs
//===----------------------------------------------------------------------===//

TEST_F(RootOrderingTest, simpleA) {
  graph[v[1]][v[0]].cost = {1, 10};
  graph[v[2]][v[0]].cost = {1, 11};
  graph[v[1]][v[2]].cost = {2, 12};
  graph[v[2]][v[1]].cost = {2, 13};
  check(2, {{v[0], {}}, {v[1], v[0]}, {v[2], v[0]}});
}

TEST_F(RootOrderingTest, simpleB) {
  graph[v[1]][v[0]].cost = {1, 10};
  graph[v[2]][v[0]].cost = {2, 11};
  graph[v[1]][v[2]].cost = {1, 12};
  graph[v[2]][v[1]].cost = {1, 13};
  check(2, {{v[0], {}}, {v[1], v[0]}, {v[2], v[1]}});
}

TEST_F(RootOrderingTest, simpleC) {
  graph[v[1]][v[0]].cost = {2, 10};
  graph[v[2]][v[0]].cost = {2, 11};
  graph[v[1]][v[2]].cost = {1, 12};
  graph[v[2]][v[1]].cost = {1, 13};
  check(3, {{v[0], {}}, {v[1], v[0]}, {v[2], v[1]}});
}

//===----------------------------------------------------------------------===//
// Graph for testing contraction
//===----------------------------------------------------------------------===//

TEST_F(RootOrderingTest, contraction) {
  graph[v[1]][v[0]].cost = {10, 0};
  graph[v[2]][v[0]].cost = {5, 0};
  graph[v[2]][v[1]].cost = {1, 0};
  graph[v[3]][v[2]].cost = {2, 0};
  graph[v[1]][v[3]].cost = {3, 0};
  check(10, {{v[0], {}}, {v[2], v[0]}, {v[3], v[2]}, {v[1], v[3]}});
}

} // namespace
