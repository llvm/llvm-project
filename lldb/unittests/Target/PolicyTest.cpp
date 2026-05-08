//===-- PolicyTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Policy.h"
#include "lldb/Utility/StreamString.h"
#include "gtest/gtest.h"

#include <thread>

using namespace lldb_private;

TEST(PolicyTest, DefaultIsPublicWithAllCapabilities) {
  Policy p;
  EXPECT_EQ(p.view, Policy::View::Public);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_TRUE(p.capabilities.can_load_frame_providers);
  EXPECT_TRUE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PublicState) {
  Policy p = Policy::PublicState();
  EXPECT_EQ(p.view, Policy::View::Public);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_TRUE(p.capabilities.can_load_frame_providers);
  EXPECT_TRUE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PrivateState) {
  Policy p = Policy::PrivateState();
  EXPECT_EQ(p.view, Policy::View::Private);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_FALSE(p.capabilities.can_load_frame_providers);
  EXPECT_FALSE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PublicStateRunningExpression) {
  Policy p = Policy::PublicStateRunningExpression();
  EXPECT_EQ(p.view, Policy::View::Public);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_FALSE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_TRUE(p.capabilities.can_load_frame_providers);
  EXPECT_TRUE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, StackDefaultIsPublicState) {
  PolicyStack &stack = PolicyStack::GetForCurrentThread();
  Policy current = stack.Current();
  EXPECT_EQ(current.view, Policy::View::Public);
  EXPECT_TRUE(current.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(current.capabilities.can_load_frame_providers);
}

TEST(PolicyTest, StackPushPop) {
  PolicyStack &stack = PolicyStack::GetForCurrentThread();

  stack.Push(Policy::PrivateState());
  EXPECT_EQ(stack.Current().view, Policy::View::Private);
  EXPECT_FALSE(stack.Current().capabilities.can_load_frame_providers);

  stack.Push(Policy::PublicStateRunningExpression());
  EXPECT_EQ(stack.Current().view, Policy::View::Public);
  EXPECT_FALSE(stack.Current().capabilities.can_run_breakpoint_actions);

  stack.Pop();
  EXPECT_EQ(stack.Current().view, Policy::View::Private);

  stack.Pop();
  EXPECT_EQ(stack.Current().view, Policy::View::Public);
}

TEST(PolicyTest, GuardRAII) {
  PolicyStack &stack = PolicyStack::GetForCurrentThread();
  EXPECT_EQ(stack.Current().view, Policy::View::Public);

  {
    PolicyStack::Guard guard(Policy::PrivateState());
    EXPECT_EQ(stack.Current().view, Policy::View::Private);
    EXPECT_FALSE(stack.Current().capabilities.can_load_frame_providers);

    {
      PolicyStack::Guard inner(Policy::PublicStateRunningExpression());
      EXPECT_EQ(stack.Current().view, Policy::View::Public);
      EXPECT_FALSE(stack.Current().capabilities.can_run_breakpoint_actions);
    }

    EXPECT_EQ(stack.Current().view, Policy::View::Private);
  }

  EXPECT_EQ(stack.Current().view, Policy::View::Public);
}

TEST(PolicyTest, StackIsPerThread) {
  PolicyStack &main_stack = PolicyStack::GetForCurrentThread();
  main_stack.Push(Policy::PrivateState());

  Policy::View other_thread_view;
  std::thread t([&other_thread_view]() {
    PolicyStack &stack = PolicyStack::GetForCurrentThread();
    other_thread_view = stack.Current().view;
  });
  t.join();

  EXPECT_EQ(main_stack.Current().view, Policy::View::Private);
  EXPECT_EQ(other_thread_view, Policy::View::Public);

  main_stack.Pop();
}

TEST(PolicyTest, DumpPublicState) {
  StreamString s;
  Policy::PublicState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "view=public, capabilities={"
            "eval_expr=1 run_all=1 try_all=1 "
            "bp_actions=1 frame_providers=1 frame_recognizers=1}");
}

TEST(PolicyTest, DumpPrivateState) {
  StreamString s;
  Policy::PrivateState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "view=private, capabilities={"
            "eval_expr=1 run_all=1 try_all=1 "
            "bp_actions=1 frame_providers=0 frame_recognizers=0}");
}

TEST(PolicyTest, DumpStack) {
  PolicyStack &stack = PolicyStack::GetForCurrentThread();
  stack.Push(Policy::PrivateState());

  StreamString s;
  stack.Dump(s);
  EXPECT_NE(s.GetString().find("depth=2"), std::string::npos);
  EXPECT_NE(s.GetString().find("[0] view=public"), std::string::npos);
  EXPECT_NE(s.GetString().find("[1] view=private"), std::string::npos);

  stack.Pop();
}
