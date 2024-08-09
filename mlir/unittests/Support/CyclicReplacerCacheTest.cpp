//===- CyclicReplacerCacheTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/CyclicReplacerCache.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include "gmock/gmock.h"
#include <map>
#include <set>

using namespace mlir;

TEST(CachedCyclicReplacerTest, testNoRecursion) {
  CachedCyclicReplacer<int, bool> replacer(
      /*replacer=*/[](int n) { return static_cast<bool>(n); },
      /*cycleBreaker=*/[](int n) { return std::nullopt; });

  EXPECT_EQ(replacer(3), true);
  EXPECT_EQ(replacer(0), false);
}

TEST(CachedCyclicReplacerTest, testInPlaceRecursionPruneAnywhere) {
  // Replacer cycles through ints 0 -> 1 -> 2 -> 0 -> ...
  std::optional<CachedCyclicReplacer<int, int>> replacer;
  replacer.emplace(
      /*replacer=*/[&](int n) { return (*replacer)((n + 1) % 3); },
      /*cycleBreaker=*/[&](int n) { return -1; });

  // Starting at 0.
  EXPECT_EQ((*replacer)(0), -1);
  // Starting at 2.
  EXPECT_EQ((*replacer)(2), -1);
}

//===----------------------------------------------------------------------===//
// CachedCyclicReplacer: ChainRecursion
//===----------------------------------------------------------------------===//

/// This set of tests uses a replacer function that maps ints into vectors of
/// ints.
///
/// The replacement result for input `n` is the replacement result of `(n+1)%3`
/// appended with an element `42`. Theoretically, this will produce an
/// infinitely long vector. The cycle-breaker function prunes this infinite
/// recursion in the replacer logic by returning an empty vector upon the first
/// re-occurrence of an input value.
namespace {
class CachedCyclicReplacerChainRecursionPruningTest : public ::testing::Test {
public:
  // N ==> (N+1) % 3
  // This will create a chain of infinite length without recursion pruning.
  CachedCyclicReplacerChainRecursionPruningTest()
      : replacer(
            [&](int n) {
              ++invokeCount;
              std::vector<int> result = replacer((n + 1) % 3);
              result.push_back(42);
              return result;
            },
            [&](int n) -> std::optional<std::vector<int>> {
              return baseCase.value_or(n) == n
                         ? std::make_optional(std::vector<int>{})
                         : std::nullopt;
            }) {}

  std::vector<int> getChain(unsigned N) { return std::vector<int>(N, 42); };

  CachedCyclicReplacer<int, std::vector<int>> replacer;
  int invokeCount = 0;
  std::optional<int> baseCase = std::nullopt;
};
} // namespace

TEST_F(CachedCyclicReplacerChainRecursionPruningTest, testPruneAnywhere0) {
  // Starting at 0. Cycle length is 3.
  EXPECT_EQ(replacer(0), getChain(3));
  EXPECT_EQ(invokeCount, 3);

  // Starting at 1. Cycle length is 5 now because of a cached replacement at 0.
  invokeCount = 0;
  EXPECT_EQ(replacer(1), getChain(5));
  EXPECT_EQ(invokeCount, 2);

  // Starting at 2. Cycle length is 4. Entire result is cached.
  invokeCount = 0;
  EXPECT_EQ(replacer(2), getChain(4));
  EXPECT_EQ(invokeCount, 0);
}

TEST_F(CachedCyclicReplacerChainRecursionPruningTest, testPruneAnywhere1) {
  // Starting at 1. Cycle length is 3.
  EXPECT_EQ(replacer(1), getChain(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CachedCyclicReplacerChainRecursionPruningTest, testPruneSpecific0) {
  baseCase = 0;

  // Starting at 0. Cycle length is 3.
  EXPECT_EQ(replacer(0), getChain(3));
  EXPECT_EQ(invokeCount, 3);
}

TEST_F(CachedCyclicReplacerChainRecursionPruningTest, testPruneSpecific1) {
  baseCase = 0;

  // Starting at 1. Cycle length is 5 (1 -> 2 -> 0 -> 1 -> 2 -> Prune).
  EXPECT_EQ(replacer(1), getChain(5));
  EXPECT_EQ(invokeCount, 5);

  // Starting at 0. Cycle length is 3. Entire result is cached.
  invokeCount = 0;
  EXPECT_EQ(replacer(0), getChain(3));
  EXPECT_EQ(invokeCount, 0);
}

//===----------------------------------------------------------------------===//
// CachedCyclicReplacer: GraphReplacement
//===----------------------------------------------------------------------===//

/// This set of tests uses a replacer function that maps from cyclic graphs to
/// trees, pruning out cycles in the process.
///
/// It consists of two helper classes:
/// - Graph
///   - A directed graph where nodes are non-negative integers.
/// - PrunedGraph
///   - A Graph where edges that used to cause cycles are now represented with
///     an indirection (a recursionId).
namespace {
class CachedCyclicReplacerGraphReplacement : public ::testing::Test {
public:
  /// A directed graph where nodes are non-negative integers.
  struct Graph {
    using Node = int64_t;

    /// Use ordered containers for deterministic output.
    /// Nodes without outgoing edges are considered nonexistent.
    std::map<Node, std::set<Node>> edges;

    void addEdge(Node src, Node sink) { edges[src].insert(sink); }

    bool isCyclic() const {
      DenseSet<Node> visited;
      for (Node root : llvm::make_first_range(edges)) {
        if (visited.contains(root))
          continue;

        SetVector<Node> path;
        SmallVector<Node> workstack;
        workstack.push_back(root);
        while (!workstack.empty()) {
          Node curr = workstack.back();
          workstack.pop_back();

          if (curr < 0) {
            // A negative node signals the end of processing all of this node's
            // children. Remove self from path.
            assert(path.back() == -curr && "internal inconsistency");
            path.pop_back();
            continue;
          }

          if (path.contains(curr))
            return true;

          visited.insert(curr);
          auto edgesIter = edges.find(curr);
          if (edgesIter == edges.end() || edgesIter->second.empty())
            continue;

          path.insert(curr);
          // Push negative node to signify recursion return.
          workstack.push_back(-curr);
          workstack.insert(workstack.end(), edgesIter->second.begin(),
                           edgesIter->second.end());
        }
      }
      return false;
    }

    /// Deterministic output for testing.
    std::string serialize() const {
      std::ostringstream oss;
      for (const auto &[src, neighbors] : edges) {
        oss << src << ":";
        for (Graph::Node neighbor : neighbors)
          oss << " " << neighbor;
        oss << "\n";
      }
      return oss.str();
    }
  };

  /// A Graph where edges that used to cause cycles (back-edges) are now
  /// represented with an indirection (a recursionId).
  ///
  /// In addition to each node having an integer ID, each node also tracks the
  /// original integer ID it had in the original graph. This way for every
  /// back-edge, we can represent it as pointing to a new instance of the
  /// original node. Then we mark the original node and the new instance with
  /// a new unique recursionId to indicate that they're supposed to be the same
  /// node.
  struct PrunedGraph {
    using Node = Graph::Node;
    struct NodeInfo {
      Graph::Node originalId;
      /// A negative recursive index means not recursive. Otherwise nodes with
      /// the same originalId & recursionId are the same node in the original
      /// graph.
      int64_t recursionId;
    };

    /// Add a regular non-recursive-self node.
    Node addNode(Graph::Node originalId, int64_t recursionIndex = -1) {
      Node id = nextConnectionId++;
      info[id] = {originalId, recursionIndex};
      return id;
    }
    /// Add a recursive-self-node, i.e. a duplicate of the original node that is
    /// meant to represent an indirection to it.
    std::pair<Node, int64_t> addRecursiveSelfNode(Graph::Node originalId) {
      return {addNode(originalId, nextRecursionId), nextRecursionId++};
    }
    void addEdge(Node src, Node sink) { connections.addEdge(src, sink); }

    /// Deterministic output for testing.
    std::string serialize() const {
      std::ostringstream oss;
      oss << "nodes\n";
      for (const auto &[nodeId, nodeInfo] : info) {
        oss << nodeId << ": n" << nodeInfo.originalId;
        if (nodeInfo.recursionId >= 0)
          oss << '<' << nodeInfo.recursionId << '>';
        oss << "\n";
      }
      oss << "edges\n";
      oss << connections.serialize();
      return oss.str();
    }

    bool isCyclic() const { return connections.isCyclic(); }

  private:
    Graph connections;
    int64_t nextRecursionId = 0;
    int64_t nextConnectionId = 0;
    /// Use ordered map for deterministic output.
    std::map<Graph::Node, NodeInfo> info;
  };

  PrunedGraph breakCycles(const Graph &input) {
    assert(input.isCyclic() && "input graph is not cyclic");

    PrunedGraph output;

    DenseMap<Graph::Node, int64_t> recMap;
    auto cycleBreaker = [&](Graph::Node inNode) -> std::optional<Graph::Node> {
      auto [node, recId] = output.addRecursiveSelfNode(inNode);
      recMap[inNode] = recId;
      return node;
    };

    CyclicReplacerCache<Graph::Node, Graph::Node> cache(cycleBreaker);

    std::function<Graph::Node(Graph::Node)> replaceNode =
        [&](Graph::Node inNode) {
          auto cacheEntry = cache.lookupOrInit(inNode);
          if (std::optional<Graph::Node> result = cacheEntry.get())
            return *result;

          // Recursively replace its neighbors.
          SmallVector<Graph::Node> neighbors;
          if (auto it = input.edges.find(inNode); it != input.edges.end())
            neighbors = SmallVector<Graph::Node>(
                llvm::map_range(it->second, replaceNode));

          // Create a new node in the output graph.
          int64_t recursionIndex =
              cacheEntry.wasRepeated() ? recMap.lookup(inNode) : -1;
          Graph::Node result = output.addNode(inNode, recursionIndex);

          for (Graph::Node neighbor : neighbors)
            output.addEdge(result, neighbor);

          cacheEntry.resolve(result);
          return result;
        };

    /// Translate starting from each node.
    for (Graph::Node root : llvm::make_first_range(input.edges))
      replaceNode(root);

    return output;
  }

  /// Helper for serialization tests that allow putting comments in the
  /// serialized format. Every line that begins with a `;` is considered a
  /// comment. The entire line, incl. the terminating `\n` is removed.
  std::string trimComments(StringRef input) {
    std::ostringstream oss;
    bool isNewLine = false;
    bool isComment = false;
    for (char c : input) {
      // Lines beginning with ';' are comments.
      if (isNewLine && c == ';')
        isComment = true;

      if (!isComment)
        oss << c;

      if (c == '\n') {
        isNewLine = true;
        isComment = false;
      }
    }
    return oss.str();
  }
};
} // namespace

TEST_F(CachedCyclicReplacerGraphReplacement, testSingleLoop) {
  // 0 -> 1 -> 2
  // ^         |
  // +---------+
  Graph input = {{{0, {1}}, {1, {2}}, {2, {0}}}};
  PrunedGraph output = breakCycles(input);
  ASSERT_FALSE(output.isCyclic()) << output.serialize();
  EXPECT_EQ(output.serialize(), trimComments(R"(nodes
; root 0
0: n0<0>
1: n2
2: n1
3: n0<0>
; root 1
4: n2
; root 2
5: n1
edges
1: 0
2: 1
3: 2
4: 3
5: 4
)"));
}

TEST_F(CachedCyclicReplacerGraphReplacement, testDualLoop) {
  // +----> 1 -----+
  // |             v
  // 0 <---------- 3
  // |             ^
  // +----> 2 -----+
  //
  // Two loops:
  // 0 -> 1 -> 3 -> 0
  // 0 -> 2 -> 3 -> 0
  Graph input = {{{0, {1, 2}}, {1, {3}}, {2, {3}}, {3, {0}}}};
  PrunedGraph output = breakCycles(input);
  ASSERT_FALSE(output.isCyclic()) << output.serialize();
  EXPECT_EQ(output.serialize(), trimComments(R"(nodes
; root 0
0: n0<0>
1: n3
2: n1
3: n2
4: n0<0>
; root 1
5: n3
6: n1
; root 2
7: n2
edges
1: 0
2: 1
3: 1
4: 2 3
5: 4
6: 5
7: 5
)"));
}

TEST_F(CachedCyclicReplacerGraphReplacement, testNestedLoops) {
  // +----> 1 -----+
  // |      ^      v
  // 0 <----+----- 2
  //
  // Two nested loops:
  // 0 -> 1 -> 2 -> 0
  //      1 -> 2 -> 1
  Graph input = {{{0, {1}}, {1, {2}}, {2, {0, 1}}}};
  PrunedGraph output = breakCycles(input);
  ASSERT_FALSE(output.isCyclic()) << output.serialize();
  EXPECT_EQ(output.serialize(), trimComments(R"(nodes
; root 0
0: n0<0>
1: n1<1>
2: n2
3: n1<1>
4: n0<0>
; root 1
5: n1<2>
6: n2
7: n1<2>
; root 2
8: n2
edges
2: 0 1
3: 2
4: 3
6: 4 5
7: 6
8: 4 7
)"));
}

TEST_F(CachedCyclicReplacerGraphReplacement, testDualNestedLoops) {
  // +----> 1 -----+
  // |      ^      v
  // 0 <----+----- 3
  // |      v      ^
  // +----> 2 -----+
  //
  // Two sets of nested loops:
  // 0 -> 1 -> 3 -> 0
  //      1 -> 3 -> 1
  // 0 -> 2 -> 3 -> 0
  //      2 -> 3 -> 2
  Graph input = {{{0, {1, 2}}, {1, {3}}, {2, {3}}, {3, {0, 1, 2}}}};
  PrunedGraph output = breakCycles(input);
  ASSERT_FALSE(output.isCyclic()) << output.serialize();
  EXPECT_EQ(output.serialize(), trimComments(R"(nodes
; root 0
0: n0<0>
1: n1<1>
2: n3<2>
3: n2
4: n3<2>
5: n1<1>
6: n2<3>
7: n3
8: n2<3>
9: n0<0>
; root 1
10: n1<4>
11: n3<5>
12: n2
13: n3<5>
14: n1<4>
; root 2
15: n2<6>
16: n3
17: n2<6>
; root 3
18: n3
edges
; root 0
3: 2
4: 0 1 3
5: 4
7: 0 5 6
8: 7
9: 5 8
; root 1
12: 11
13: 9 10 12
14: 13
; root 2
16: 9 14 15
17: 16
; root 3
18: 9 14 17
)"));
}
