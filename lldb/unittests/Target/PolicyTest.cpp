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
  Policy current = PolicyStack::Get().Current();
  EXPECT_EQ(current.view, Policy::View::Public);
  EXPECT_TRUE(current.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(current.capabilities.can_load_frame_providers);
}

TEST(PolicyTest, StackPushPop) {
  PolicyStack::Get().Push(Policy::PrivateState());
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  EXPECT_FALSE(
      PolicyStack::Get().Current().capabilities.can_load_frame_providers);

  PolicyStack::Get().Push(Policy::PublicStateRunningExpression());
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
  EXPECT_FALSE(
      PolicyStack::Get().Current().capabilities.can_run_breakpoint_actions);

  PolicyStack::Get().Pop();
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);

  PolicyStack::Get().Pop();
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
}

TEST(PolicyTest, GuardRAII) {
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);

  {
    PolicyStack::Guard guard(Policy::PrivateState());
    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
    EXPECT_FALSE(
        PolicyStack::Get().Current().capabilities.can_load_frame_providers);

    {
      PolicyStack::Guard inner(Policy::PublicStateRunningExpression());
      EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
      EXPECT_FALSE(
          PolicyStack::Get().Current().capabilities.can_run_breakpoint_actions);
    }

    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  }

  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
}

TEST(PolicyTest, StackIsPerThread) {
  PolicyStack::Get().Push(Policy::PrivateState());

  Policy::View other_thread_view;
  std::thread t([&other_thread_view]() {
    other_thread_view = PolicyStack::Get().Current().view;
  });
  t.join();

  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  EXPECT_EQ(other_thread_view, Policy::View::Public);

  PolicyStack::Get().Pop();
}

TEST(PolicyTest, DumpPublicState) {
  StreamString s;
  Policy::PublicState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "policy: view=public, capabilities={"
            "eval_expr=true run_all=true try_all=true "
            "bp_actions=true frame_providers=true frame_recognizers=true}");
}

TEST(PolicyTest, DumpPrivateState) {
  StreamString s;
  Policy::PrivateState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "policy: view=private, capabilities={"
            "eval_expr=true run_all=true try_all=true "
            "bp_actions=true frame_providers=false frame_recognizers=false}");
}

TEST(PolicyTest, DumpStack) {
  PolicyStack::Get().Push(Policy::PrivateState());

  StreamString s;
  PolicyStack::Get().Dump(s);
  EXPECT_NE(s.GetString().find("depth=2"), std::string::npos);
  EXPECT_NE(s.GetString().find("[0] policy: view=public"), std::string::npos);
  EXPECT_NE(s.GetString().find("[1] policy: view=private"), std::string::npos);

  PolicyStack::Get().Pop();
}
