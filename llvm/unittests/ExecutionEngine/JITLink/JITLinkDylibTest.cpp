//===--- JITLinkDylibTest.cpp - Test JITLinkDylib and notifications ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;

namespace {

class RecordingMemoryManager : public JITLinkMemoryManager {
public:
  void allocate(const JITLinkDylib *JD, LinkGraph &G,
                OnAllocatedFunction OnAllocated) override {
    llvm_unreachable("Not used by this test");
  }

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override {
    llvm_unreachable("Not used by this test");
  }

  using JITLinkMemoryManager::deallocate;

  void notifyDestroying(JITLinkDylib &JD) override {
    NotifiedName = JD.getName();
    ++NotifyCount;
  }

  unsigned NotifyCount = 0;
  std::string NotifiedName;
};

} // namespace

TEST(JITLinkDylibTest, GetName) {
  JITLinkDylib JD("foo");
  EXPECT_EQ(JD.getName(), "foo");
}

TEST(JITLinkDylibTest, MemoryManagerNotifiedOnDestruction) {
  RecordingMemoryManager MemMgr;
  {
    JITLinkDylib JD("foo");
    JD.notifyOnDestruction(MemMgr);
    EXPECT_EQ(MemMgr.NotifyCount, 0U);
  }
  EXPECT_EQ(MemMgr.NotifyCount, 1U);
  EXPECT_EQ(MemMgr.NotifiedName, "foo");
}

TEST(JITLinkDylibTest, MultipleMemoryManagersAllNotified) {
  RecordingMemoryManager MemMgr1, MemMgr2;
  {
    JITLinkDylib JD("foo");
    JD.notifyOnDestruction(MemMgr1);
    JD.notifyOnDestruction(MemMgr2);
  }
  EXPECT_EQ(MemMgr1.NotifyCount, 1U);
  EXPECT_EQ(MemMgr2.NotifyCount, 1U);
}
