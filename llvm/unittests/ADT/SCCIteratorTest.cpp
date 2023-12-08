//===----- llvm/unittest/ADT/SCCIteratorTest.cpp - SCCIterator tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SCCIterator.h"
#include "TestGraph.h"
#include "gtest/gtest.h"
#include <limits.h>

using namespace llvm;

namespace llvm {

TEST(SCCIteratorTest, AllSmallGraphs) {
  // Test SCC computation against every graph with NUM_NODES nodes or less.
  // Since SCC considers every node to have an implicit self-edge, we only
  // create graphs for which every node has a self-edge.
#define NUM_NODES 4
#define NUM_GRAPHS (NUM_NODES * (NUM_NODES - 1))
  typedef Graph<NUM_NODES> GT;

  /// Enumerate all graphs using NUM_GRAPHS bits.
  static_assert(NUM_GRAPHS < sizeof(unsigned) * CHAR_BIT, "Too many graphs!");
  for (unsigned GraphDescriptor = 0; GraphDescriptor < (1U << NUM_GRAPHS);
       ++GraphDescriptor) {
    GT G;

    // Add edges as specified by the descriptor.
    unsigned DescriptorCopy = GraphDescriptor;
    for (unsigned i = 0; i != NUM_NODES; ++i)
      for (unsigned j = 0; j != NUM_NODES; ++j) {
        // Always add a self-edge.
        if (i == j) {
          G.AddEdge(i, j);
          continue;
        }
        if (DescriptorCopy & 1)
          G.AddEdge(i, j);
        DescriptorCopy >>= 1;
      }

    // Test the SCC logic on this graph.

    /// NodesInSomeSCC - Those nodes which are in some SCC.
    GT::NodeSubset NodesInSomeSCC;

    for (scc_iterator<GT> I = scc_begin(G), E = scc_end(G); I != E; ++I) {
      const std::vector<GT::NodeType *> &SCC = *I;

      // Get the nodes in this SCC as a NodeSubset rather than a vector.
      GT::NodeSubset NodesInThisSCC;
      for (unsigned i = 0, e = SCC.size(); i != e; ++i)
        NodesInThisSCC.AddNode(SCC[i]->first);

      // There should be at least one node in every SCC.
      EXPECT_FALSE(NodesInThisSCC.isEmpty());

      // Check that every node in the SCC is reachable from every other node in
      // the SCC.
      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i)) {
          EXPECT_TRUE(NodesInThisSCC.isSubsetOf(G.NodesReachableFrom(i)));
        }

      // OK, now that we now that every node in the SCC is reachable from every
      // other, this means that the set of nodes reachable from any node in the
      // SCC is the same as the set of nodes reachable from every node in the
      // SCC.  Check that for every node N not in the SCC but reachable from the
      // SCC, no element of the SCC is reachable from N.
      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i)) {
          GT::NodeSubset NodesReachableFromSCC = G.NodesReachableFrom(i);
          GT::NodeSubset ReachableButNotInSCC =
            NodesReachableFromSCC.Meet(NodesInThisSCC.Complement());

          for (unsigned j = 0; j != NUM_NODES; ++j)
            if (ReachableButNotInSCC.count(j)) {
              EXPECT_TRUE(G.NodesReachableFrom(j).Meet(NodesInThisSCC).isEmpty());
            }

          // The result must be the same for all other nodes in this SCC, so
          // there is no point in checking them.
          break;
        }

      // This is indeed a SCC: a maximal set of nodes for which each node is
      // reachable from every other.

      // Check that we didn't already see this SCC.
      EXPECT_TRUE(NodesInSomeSCC.Meet(NodesInThisSCC).isEmpty());

      NodesInSomeSCC = NodesInSomeSCC.Join(NodesInThisSCC);

      // Check a property that is specific to the LLVM SCC iterator and
      // guaranteed by it: if a node in SCC S1 has an edge to a node in
      // SCC S2, then S1 is visited *after* S2.  This means that the set
      // of nodes reachable from this SCC must be contained either in the
      // union of this SCC and all previously visited SCC's.

      for (unsigned i = 0; i != NUM_NODES; ++i)
        if (NodesInThisSCC.count(i)) {
          GT::NodeSubset NodesReachableFromSCC = G.NodesReachableFrom(i);
          EXPECT_TRUE(NodesReachableFromSCC.isSubsetOf(NodesInSomeSCC));
          // The result must be the same for all other nodes in this SCC, so
          // there is no point in checking them.
          break;
        }
    }

    // Finally, check that the nodes in some SCC are exactly those that are
    // reachable from the initial node.
    EXPECT_EQ(NodesInSomeSCC, G.NodesReachableFrom(0));
  }
}

}
