//=== llvm/unittest/ADT/DepthFirstIteratorTest.cpp - DFS iterator tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DepthFirstIterator.h"
#include "TestGraph.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {

template <typename T> struct CountedSet {
  typedef typename SmallPtrSet<T, 4>::iterator iterator;

  SmallPtrSet<T, 4> S;
  int InsertVisited = 0;

  std::pair<iterator, bool> insert(const T &Item) {
    InsertVisited++;
    return S.insert(Item);
  }

  size_t count(const T &Item) const { return S.count(Item); }
  
  void completed(T) { }
};

template <typename T> class df_iterator_storage<CountedSet<T>, true> {
public:
  df_iterator_storage(CountedSet<T> &VSet) : Visited(VSet) {}

  CountedSet<T> &Visited;
};

TEST(DepthFirstIteratorTest, ActuallyUpdateIterator) {
  typedef CountedSet<Graph<3>::NodeType *> StorageT;
  typedef df_iterator<Graph<3>, StorageT, true> DFIter;

  Graph<3> G;
  G.AddEdge(0, 1);
  G.AddEdge(0, 2);
  StorageT S;
  for (auto N : make_range(DFIter::begin(G, S), DFIter::end(G, S)))
    (void)N;

  EXPECT_EQ(3, S.InsertVisited);
}
}
