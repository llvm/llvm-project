//===- VettedPeerTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for VettedPeer.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/VettedPeer.h"

#include "gtest/gtest.h"

#include <memory>
#include <type_traits>

using namespace orc_rt;

namespace {

// A stand-in for a move-only channel such as SocketHandle.
using Channel = std::unique_ptr<int>;

// The point of VettedPeer: a bare channel does not become one by accident.
// Only the named factories make one.
static_assert(!std::is_constructible_v<VettedPeer<Channel>, Channel>);
static_assert(!std::is_convertible_v<Channel, VettedPeer<Channel>>);

TEST(VettedPeerTest, TakeYieldsTheChannel) {
  auto Check = [](VettedPeer<Channel> (*Make)(Channel)) {
    auto C = std::make_unique<int>(42);
    int *Raw = C.get();
    Channel Out = Make(std::move(C)).take();
    EXPECT_EQ(Out.get(), Raw);
  };
  Check(VettedPeer<Channel>::inherited);
  Check(VettedPeer<Channel>::checked);
  Check(VettedPeer<Channel>::unchecked);
}

} // namespace
