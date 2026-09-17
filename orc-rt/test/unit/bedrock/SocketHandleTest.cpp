//===- SocketHandleTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These cover the ownership bookkeeping in SocketHandle.h, which every platform
// compiles verbatim, so they are shared rather than written per system.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketHandle.h"
#include "gtest/gtest.h"

#include "SocketTestUtils.h"

#include <utility>

using namespace orc_rt;

namespace {

/// A socket for a test to own, failing the test if the system refuses one.
NativeSocketHandle testSocket() {
  auto H = makeNativeSocket();
  EXPECT_TRUE(H.has_value()) << "could not create a socket for the test";
  return H.value_or(InvalidNativeSocketHandle);
}

TEST(SocketHandleTest, DefaultConstructedOwnsNothing) {
  SocketHandle H;
  EXPECT_FALSE(H);
  EXPECT_EQ(H.get(), InvalidNativeSocketHandle);
  // Must not try to close the sentinel.
  H.reset();
  EXPECT_FALSE(H);
  EXPECT_EQ(H.release(), InvalidNativeSocketHandle);
}

TEST(SocketHandleTest, AdoptingAFailedSocketCallGivesAnEmptyHandle) {
  // So that a socket call's result can be adopted before it is checked.
  SocketHandle H(InvalidNativeSocketHandle);
  EXPECT_FALSE(H);
}

TEST(SocketHandleTest, ResetCloses) {
  NativeSocketHandle Raw = testSocket();
  SocketHandle H(Raw);
  ASSERT_TRUE(isNativeSocketOpen(Raw));

  H.reset();
  EXPECT_FALSE(H);
  EXPECT_FALSE(isNativeSocketOpen(Raw));
}

TEST(SocketHandleTest, DestructorCloses) {
  NativeSocketHandle Raw = testSocket();
  {
    SocketHandle H(Raw);
    ASSERT_TRUE(isNativeSocketOpen(Raw));
  }
  EXPECT_FALSE(isNativeSocketOpen(Raw));
}

TEST(SocketHandleTest, ReleaseGivesUpOwnership) {
  NativeSocketHandle Raw = testSocket();
  {
    SocketHandle H(Raw);
    EXPECT_EQ(H.release(), Raw);
    EXPECT_FALSE(H);
  }
  EXPECT_TRUE(isNativeSocketOpen(Raw));
  closeNativeSocket(Raw);
}

TEST(SocketHandleTest, MoveConstructionTransfersOwnership) {
  NativeSocketHandle Raw = testSocket();
  SocketHandle Source(Raw);

  SocketHandle Moved(std::move(Source));
  EXPECT_FALSE(Source) << "moved-from handle still owns a socket";
  ASSERT_TRUE(Moved);
  EXPECT_EQ(Moved.get(), Raw);
  EXPECT_TRUE(isNativeSocketOpen(Raw));
}

TEST(SocketHandleTest, MoveAssignmentClosesTheOldSocket) {
  NativeSocketHandle Replaced = testSocket();
  NativeSocketHandle Kept = testSocket();

  SocketHandle Target(Replaced);
  SocketHandle Source(Kept);
  Target = std::move(Source);

  EXPECT_FALSE(isNativeSocketOpen(Replaced)) << "overwritten socket was leaked";
  EXPECT_TRUE(isNativeSocketOpen(Kept));
  EXPECT_EQ(Target.get(), Kept);
  EXPECT_FALSE(Source);
}

TEST(SocketHandleTest, SelfMoveAssignmentKeepsTheSocket) {
  NativeSocketHandle Raw = testSocket();
  SocketHandle H(Raw);

  // Through a reference because -Wself-move rejects the direct form, which is
  // what a caller reaching this via a template or a swap would write.
  SocketHandle &Alias = H;
  H = std::move(Alias);

  EXPECT_TRUE(isNativeSocketOpen(Raw));
  EXPECT_EQ(H.get(), Raw);
}

} // namespace
