//===-- OperandGraphTest.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OperandGraph.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::ElementsAre;
using testing::IsEmpty;
using testing::Not;

namespace exegesis {
namespace graph {
namespace {

static const auto In = Node::In();
static const auto Out = Node::Out();

TEST(OperandGraphTest, NoPath) {
  Graph TheGraph;
  EXPECT_THAT(TheGraph.getPathFrom(In, Out), IsEmpty());
}

TEST(OperandGraphTest, Connecting) {
  Graph TheGraph;
  TheGraph.connect(In, Out);
  EXPECT_THAT(TheGraph.getPathFrom(In, Out), Not(IsEmpty()));
  EXPECT_THAT(TheGraph.getPathFrom(In, Out), ElementsAre(In, Out));
}

TEST(OperandGraphTest, ConnectingThroughVariable) {
  const Node Var = Node::Var(1);
  Graph TheGraph;
  TheGraph.connect(In, Var);
  TheGraph.connect(Var, Out);
  EXPECT_THAT(TheGraph.getPathFrom(In, Out), Not(IsEmpty()));
  EXPECT_THAT(TheGraph.getPathFrom(In, Out), ElementsAre(In, Var, Out));
}

} // namespace
} // namespace graph
} // namespace exegesis
