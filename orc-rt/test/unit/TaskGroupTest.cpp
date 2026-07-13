//===- TaskGroupTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's TaskGroup.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/TaskGroup.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace orc_rt;

TEST(TaskGroupTest, TrivialConstructionAndDestruction) {
  auto TG = TaskGroup::Create();
}

TEST(TaskGroupTest, SingleTokenThenClose) {
  bool Completed = false;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Completed = true; });

  {
    TaskGroup::Token T(TG);
    EXPECT_TRUE(T);
    TG->close();
    EXPECT_FALSE(Completed);
  }
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, CloseWithNoTokens) {
  bool Completed = false;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Completed = true; });
  TG->close();
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, TokenFromClosedGroup) {
  auto TG = TaskGroup::Create();
  TG->close();
  TaskGroup::Token T(TG);
  EXPECT_FALSE(T);
}

TEST(TaskGroupTest, TokenFromNullSharedPtr) {
  std::shared_ptr<TaskGroup> TG;
  TaskGroup::Token T(TG);
  EXPECT_FALSE(T);
}

TEST(TaskGroupTest, CopyToken) {
  bool Completed = false;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Completed = true; });

  {
    TaskGroup::Token T1(TG);
    EXPECT_TRUE(T1);
    {
      TaskGroup::Token T2(T1); // Copy increments count
      EXPECT_TRUE(T2);
      TG->close();
      EXPECT_FALSE(Completed);
    }
    EXPECT_FALSE(Completed); // T1 still alive
  }
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, MoveToken) {
  bool Completed = false;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Completed = true; });

  TaskGroup::Token T1(TG);
  TaskGroup::Token T2(std::move(T1));
  EXPECT_FALSE(T1);
  EXPECT_TRUE(T2);

  TG->close();
  EXPECT_FALSE(Completed);

  T2 = TaskGroup::Token(); // Release
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, CopyAssignmentReleasesOld) {
  bool Completed1 = false;
  bool Completed2 = false;
  auto TG1 = TaskGroup::Create();
  auto TG2 = TaskGroup::Create();
  TG1->addOnComplete([&]() { Completed1 = true; });
  TG2->addOnComplete([&]() { Completed2 = true; });

  TaskGroup::Token T1(TG1);
  TaskGroup::Token T2(TG2);

  TG1->close();
  TG2->close();

  EXPECT_FALSE(Completed1);
  EXPECT_FALSE(Completed2);

  T1 = T2; // Releases TG1, acquires TG2

  EXPECT_TRUE(Completed1);  // TG1 should complete
  EXPECT_FALSE(Completed2); // TG2 still has T1 and T2
}

TEST(TaskGroupTest, CopyAssignmentFromClosedGroup) {
  bool Completed = false;
  auto TG1 = TaskGroup::Create();
  auto TG2 = TaskGroup::Create();
  TG1->addOnComplete([&]() { Completed = true; });
  TG2->close();

  TaskGroup::Token T1(TG1);
  TaskGroup::Token T2(TG2);
  EXPECT_TRUE(T1);
  EXPECT_FALSE(T2);

  TG1->close();
  EXPECT_FALSE(Completed);

  T1 = T2; // Assign from empty, releases TG1
  EXPECT_FALSE(T1);
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, MoveAssignmentReleasesOld) {
  bool Completed1 = false;
  bool Completed2 = false;
  auto TG1 = TaskGroup::Create();
  auto TG2 = TaskGroup::Create();
  TG1->addOnComplete([&]() { Completed1 = true; });
  TG2->addOnComplete([&]() { Completed2 = true; });

  TaskGroup::Token T1(TG1);
  TaskGroup::Token T2(TG2);

  TG1->close();
  TG2->close();

  EXPECT_FALSE(Completed1);
  EXPECT_FALSE(Completed2);

  T1 = std::move(T2); // Releases TG1, takes TG2 from T2

  EXPECT_TRUE(Completed1);  // TG1 should complete
  EXPECT_FALSE(Completed2); // TG2 now held by T1
  EXPECT_FALSE(T2);         // T2 is now empty
}

TEST(TaskGroupTest, SelfCopyAssignment) {
  auto TG = TaskGroup::Create();
  TaskGroup::Token T(TG);
  EXPECT_TRUE(T);

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
  T = T; // Self-assign
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  EXPECT_TRUE(T); // Should still be valid

  TG->close();
}

TEST(TaskGroupTest, SelfMoveAssignment) {
  auto TG = TaskGroup::Create();
  TaskGroup::Token T(TG);
  EXPECT_TRUE(T);

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
#endif
  T = std::move(T); // Self-move-assign
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  EXPECT_TRUE(T); // Should still be valid

  TG->close();
}

TEST(TaskGroupTest, AddOnCompleteAfterCompletion) {
  auto TG = TaskGroup::Create();
  TG->close();

  bool Completed = false;
  TG->addOnComplete([&]() { Completed = true; });
  EXPECT_TRUE(Completed); // Runs immediately
}

TEST(TaskGroupTest, AddOnCompleteWhileTasksPending) {
  auto TG = TaskGroup::Create();
  TaskGroup::Token T(TG);
  TG->close();

  bool Completed = false;
  TG->addOnComplete([&]() { Completed = true; });
  EXPECT_FALSE(Completed); // Queued, not run yet

  T = TaskGroup::Token(); // Release
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, MultipleCallbacks) {
  std::vector<int> Order;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Order.push_back(1); });
  TG->addOnComplete([&]() { Order.push_back(2); });
  TG->addOnComplete([&]() { Order.push_back(3); });
  TG->close();

  ASSERT_EQ(Order.size(), 3u);
  EXPECT_EQ(Order[0], 1);
  EXPECT_EQ(Order[1], 2);
  EXPECT_EQ(Order[2], 3);
}

TEST(TaskGroupTest, MultipleTokens) {
  int CompletionCount = 0;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { CompletionCount++; });

  {
    TaskGroup::Token T1(TG);
    TaskGroup::Token T2(TG);
    TaskGroup::Token T3(TG);
    TG->close();
    EXPECT_EQ(CompletionCount, 0);
  }
  EXPECT_EQ(CompletionCount, 1); // Only fires once
}

TEST(TaskGroupTest, CloseIsIdempotent) {
  int CompletionCount = 0;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { CompletionCount++; });

  TG->close();
  TG->close();
  TG->close();

  EXPECT_EQ(CompletionCount, 1);
}

TEST(TaskGroupTest, AcquireAfterCloseViaDirectAPI) {
  auto TG = TaskGroup::Create();
  EXPECT_TRUE(TG->acquireToken());
  TG->close();
  EXPECT_FALSE(TG->acquireToken());
  TG->releaseToken(); // Release the one we acquired
}

TEST(TaskGroupTest, DirectAPIMatchesRAII) {
  bool Completed = false;
  auto TG = TaskGroup::Create();
  TG->addOnComplete([&]() { Completed = true; });

  TG->acquireToken();
  TG->acquireToken();
  TG->close();

  EXPECT_FALSE(Completed);
  TG->releaseToken();
  EXPECT_FALSE(Completed);
  TG->releaseToken();
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, TokenKeepsTaskGroupAlive) {
  TaskGroup::Token T;
  bool Completed = false;

  {
    auto TG = TaskGroup::Create();
    TG->addOnComplete([&]() { Completed = true; });
    T = TaskGroup::Token(TG);
    TG->close();
    // TG goes out of scope here, but T holds a shared_ptr
  }

  EXPECT_FALSE(Completed); // Still pending - T keeps TG alive
  T = TaskGroup::Token();  // Release
  EXPECT_TRUE(Completed);
}

TEST(TaskGroupTest, ConcurrentTokens) {
  for (int Iter = 0; Iter < 100; ++Iter) {
    std::atomic<int> Count{0};
    auto TG = TaskGroup::Create();
    TG->addOnComplete([&]() { Count++; });

    std::vector<std::thread> Threads;
    for (int I = 0; I < 10; ++I) {
      TaskGroup::Token T(TG);
      Threads.emplace_back([T = std::move(T)]() {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      });
    }

    TG->close();
    for (auto &T : Threads)
      T.join();

    EXPECT_EQ(Count, 1);
  }
}

TEST(TaskGroupTest, ConcurrentAddOnCompleteAndClose) {
  for (int Iter = 0; Iter < 100; ++Iter) {
    std::atomic<int> Count{0};
    auto TG = TaskGroup::Create();

    std::thread Closer([&]() { TG->close(); });

    std::thread Registerer([&]() { TG->addOnComplete([&]() { Count++; }); });

    Closer.join();
    Registerer.join();

    // Callback should have run exactly once regardless of order
    EXPECT_EQ(Count, 1);
  }
}

TEST(TaskGroupTest, ConcurrentAcquireAndClose) {
  for (int Iter = 0; Iter < 100; ++Iter) {
    std::atomic<int> SuccessfulAcquires{0};
    std::atomic<int> CompletionCount{0};
    auto TG = TaskGroup::Create();
    TG->addOnComplete([&]() { CompletionCount++; });

    std::vector<std::thread> Threads;

    // Multiple threads trying to acquire
    for (int I = 0; I < 5; ++I) {
      Threads.emplace_back([&, T = TaskGroup::Token(TG)]() {
        if (T)
          SuccessfulAcquires++;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      });
    }

    // One thread closing
    Threads.emplace_back([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      TG->close();
    });

    for (auto &T : Threads)
      T.join();

    // Should complete exactly once
    EXPECT_EQ(CompletionCount, 1);
    // At least some acquires should have succeeded before close
    EXPECT_EQ(SuccessfulAcquires, 5);
  }
}
