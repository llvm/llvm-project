//===- llvm/unittest/ADT/ScopeExit.cpp - Scope exit unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(ScopeExitTest, Basic) {
  struct Callable {
    bool &Called;
    Callable(bool &Called) : Called(Called) {}
    Callable(Callable &&RHS) : Called(RHS.Called) {}
    void operator()() { Called = true; }
  };
  bool Called = false;
  {
    auto g = make_scope_exit(Callable(Called));
    EXPECT_FALSE(Called);
  }
  EXPECT_TRUE(Called);
}

TEST(ScopeExitTest, Release) {
  int Count = 0;
  auto Increment = [&] { ++Count; };
  {
    auto G = make_scope_exit(Increment);
    auto H = std::move(G);
    auto I = std::move(G);
    EXPECT_EQ(0, Count);
  }
  EXPECT_EQ(1, Count);
  {
    auto G = make_scope_exit(Increment);
    G.release();
  }
  EXPECT_EQ(1, Count);
}

} // end anonymous namespace
