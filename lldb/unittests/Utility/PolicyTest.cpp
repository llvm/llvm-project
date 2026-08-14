//===-- PolicyTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Policy.h"
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
  Policy p = Policy::CreatePublicState();
  EXPECT_EQ(p.view, Policy::View::Public);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_TRUE(p.capabilities.can_load_frame_providers);
  EXPECT_TRUE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PrivateState) {
  Policy p = Policy::CreatePrivateState();
  EXPECT_EQ(p.view, Policy::View::Private);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_TRUE(p.capabilities.can_load_frame_providers);
  EXPECT_TRUE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PrivateStateRunningExpression) {
  Policy p = Policy::CreatePrivateState(
      Policy::PrivateStatePurpose::RunningExpression);
  EXPECT_EQ(p.view, Policy::View::Private);
  EXPECT_TRUE(p.capabilities.can_evaluate_expressions);
  EXPECT_TRUE(p.capabilities.can_run_all_threads);
  EXPECT_TRUE(p.capabilities.can_try_all_threads);
  EXPECT_TRUE(p.capabilities.can_run_breakpoint_actions);
  EXPECT_FALSE(p.capabilities.can_load_frame_providers);
  EXPECT_FALSE(p.capabilities.can_run_frame_recognizers);
}

TEST(PolicyTest, PublicStateRunningExpression) {
  Policy p = Policy::CreatePublicStateRunningExpression();
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
  {
    PolicyStack::Guard guard = PolicyStack::Get().PushPrivateState(
        Policy::PrivateStatePurpose::RunningExpression);
    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
    EXPECT_FALSE(
        PolicyStack::Get().Current().capabilities.can_load_frame_providers);

    {
      PolicyStack::Guard inner =
          PolicyStack::Get().PushPublicStateRunningExpression();
      // PushPublicStateRunningExpression inherits from Current() and only
      // toggles bp_actions; view stays Private.
      EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
      EXPECT_FALSE(
          PolicyStack::Get().Current().capabilities.can_run_breakpoint_actions);
    }

    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  }

  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
}

TEST(PolicyTest, GuardRAII) {
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);

  {
    PolicyStack::Guard guard = PolicyStack::Get().PushPrivateState(
        Policy::PrivateStatePurpose::RunningExpression);
    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
    EXPECT_FALSE(
        PolicyStack::Get().Current().capabilities.can_load_frame_providers);

    {
      PolicyStack::Guard inner =
          PolicyStack::Get().PushPublicStateRunningExpression();
      // Inherits Private view from outer guard.
      EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
      EXPECT_FALSE(
          PolicyStack::Get().Current().capabilities.can_run_breakpoint_actions);
    }

    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  }

  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
}

TEST(PolicyTest, StackIsPerThread) {
  PolicyStack::Guard guard = PolicyStack::Get().PushPrivateState();

  Policy::View other_thread_view;
  std::thread t([&other_thread_view]() {
    other_thread_view = PolicyStack::Get().Current().view;
  });
  t.join();

  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  EXPECT_EQ(other_thread_view, Policy::View::Public);
}

TEST(PolicyTest, DumpPublicState) {
  StreamString s;
  Policy::CreatePublicState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "policy: view=public, capabilities={"
            "eval_expr=true run_all=true try_all=true "
            "bp_actions=true frame_providers=true frame_recognizers=true}");
}

TEST(PolicyTest, DumpPrivateState) {
  StreamString s;
  Policy::CreatePrivateState().Dump(s);
  EXPECT_EQ(s.GetString(),
            "policy: view=private, capabilities={"
            "eval_expr=true run_all=true try_all=true "
            "bp_actions=true frame_providers=true frame_recognizers=true}");
}

TEST(PolicyTest, DumpStack) {
  PolicyStack::Guard guard = PolicyStack::Get().PushPrivateState();

  StreamString s;
  PolicyStack::Get().Dump(s);
  EXPECT_NE(s.GetString().find("depth=2"), std::string::npos);
  EXPECT_NE(s.GetString().find("[0] policy: view=public"), std::string::npos);
  EXPECT_NE(s.GetString().find("[1] policy: view=private"), std::string::npos);
}

TEST(PolicyTest, GuardSameThreadMove) {
  // Move on the same thread is fine; the moved-into Guard still pops on
  // destruction and the moved-from Guard becomes a no-op.
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
  {
    PolicyStack::Guard outer = PolicyStack::Get().PushPrivateState();
    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);

    PolicyStack::Guard moved = std::move(outer);
    EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  }
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Public);
}

#if GTEST_HAS_DEATH_TEST
TEST(PolicyStackDeathTest, GuardDestroyedOnDifferentThread) {
  PolicyStack::Guard outer = PolicyStack::Get().PushPrivateState();
  // The move into the closure happens here, on the constructing thread, so
  // it doesn't trip the thread-affinity check. The closure (and thus the
  // Guard) is destroyed on the worker thread once it returns, which is
  // where the violation is detected.
  EXPECT_DEATH(
      {
        std::thread t([guard = std::move(outer)]() mutable { (void)guard; });
        t.join();
      },
      "PolicyStack::Guard");
}
#endif

TEST(PolicyTest, PushInheritsFromCurrent) {
  // Push* methods inherit from Current() rather than starting from a
  // default Policy: stacking PushPublicStateRunningExpression on top of
  // PushPrivateState(RunningExpression) must preserve the Private view.
  PolicyStack::Guard outer = PolicyStack::Get().PushPrivateState(
      Policy::PrivateStatePurpose::RunningExpression);
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);

  PolicyStack::Guard inner =
      PolicyStack::Get().PushPublicStateRunningExpression();
  // Capability from inner push.
  EXPECT_FALSE(
      PolicyStack::Get().Current().capabilities.can_run_breakpoint_actions);
  // View inherited from outer push (would be Public if Push reset state).
  EXPECT_EQ(PolicyStack::Get().Current().view, Policy::View::Private);
  // Capabilities from outer push also inherited.
  EXPECT_FALSE(
      PolicyStack::Get().Current().capabilities.can_load_frame_providers);
}
