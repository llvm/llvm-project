//=======- CallGraphTest.cpp - Unit tests for the CG analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <typename Ty> void canSpecializeGraphTraitsIterators(Ty *G) {
  typedef typename GraphTraits<Ty *>::NodeRef NodeRef;

  auto I = GraphTraits<Ty *>::nodes_begin(G);
  auto E = GraphTraits<Ty *>::nodes_end(G);
  auto X = ++I;

  // Should be able to iterate over all nodes of the graph.
  static_assert(std::is_same<decltype(*I), NodeRef>::value,
                "Node type does not match");
  static_assert(std::is_same<decltype(*X), NodeRef>::value,
                "Node type does not match");
  static_assert(std::is_same<decltype(*E), NodeRef>::value,
                "Node type does not match");

  NodeRef N = GraphTraits<Ty *>::getEntryNode(G);

  auto S = GraphTraits<NodeRef>::child_begin(N);
  auto F = GraphTraits<NodeRef>::child_end(N);

  // Should be able to iterate over immediate successors of a node.
  static_assert(std::is_same<decltype(*S), NodeRef>::value,
                "Node type does not match");
  static_assert(std::is_same<decltype(*F), NodeRef>::value,
                "Node type does not match");
}

TEST(CallGraphTest, GraphTraitsSpecialization) {
  LLVMContext Context;
  Module M("", Context);
  CallGraph CG(M);

  canSpecializeGraphTraitsIterators(&CG);
}

TEST(CallGraphTest, GraphTraitsConstSpecialization) {
  LLVMContext Context;
  Module M("", Context);
  CallGraph CG(M);

  canSpecializeGraphTraitsIterators(const_cast<const CallGraph *>(&CG));
}
}
