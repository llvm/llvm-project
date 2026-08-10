//===-- SharedClusterTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/SharedCluster.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
class DestructNotifier {
public:
  DestructNotifier(std::vector<int> &Queue, int Key) : Queue(Queue), Key(Key) {}
  ~DestructNotifier() { Queue.push_back(Key); }

  std::vector<int> &Queue;
  const int Key;
};
} // namespace

TEST(SharedCluster, ClusterManager) {
  std::vector<int> Queue;
  {
    auto CM = ClusterManager<DestructNotifier>::Create();
    auto *One = new DestructNotifier(Queue, 1);
    auto *Two = new DestructNotifier(Queue, 2);
    CM->ManageObject(One);
    CM->ManageObject(Two);

    ASSERT_THAT(Queue, testing::IsEmpty());
    {
      std::shared_ptr<DestructNotifier> OnePtr = CM->GetSharedPointer(One);
      ASSERT_EQ(OnePtr->Key, 1);
      ASSERT_THAT(Queue, testing::IsEmpty());

      {
        std::shared_ptr<DestructNotifier> OnePtrCopy = OnePtr;
        ASSERT_EQ(OnePtrCopy->Key, 1);
        ASSERT_THAT(Queue, testing::IsEmpty());
      }

      {
        std::shared_ptr<DestructNotifier> TwoPtr = CM->GetSharedPointer(Two);
        ASSERT_EQ(TwoPtr->Key, 2);
        ASSERT_THAT(Queue, testing::IsEmpty());
      }

      ASSERT_THAT(Queue, testing::IsEmpty());
    }
    ASSERT_THAT(Queue, testing::IsEmpty());
  }
  ASSERT_THAT(Queue, testing::ElementsAre(1, 2));
}
