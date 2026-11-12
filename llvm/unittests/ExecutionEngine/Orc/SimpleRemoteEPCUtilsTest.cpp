//===--- SimpleRemoteEPCUtilsTest.cpp - Test SimpleRemoteEPC utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

static ExecutorAddr tagFor(SimpleRemoteEPCResultKind K) {
  return ExecutorAddr(static_cast<uint64_t>(K));
}

TEST(SimpleRemoteEPCUtilsTest, ResultMessageValueRoundTrip) {
  constexpr StringRef Content = "result bytes";
  auto [TagAddr, Payload] = encodeResultMessage(
      shared::WrapperFunctionBuffer::copyFrom(Content.data(), Content.size()));

  EXPECT_EQ(TagAddr, tagFor(SimpleRemoteEPCResultKind::Value));

  auto Decoded = decodeResultMessage(TagAddr, std::move(Payload));
  ASSERT_THAT_EXPECTED(Decoded, Succeeded());
  EXPECT_EQ(Decoded->getOutOfBandError(), nullptr);
  EXPECT_EQ(StringRef(Decoded->data(), Decoded->size()), Content);
}

TEST(SimpleRemoteEPCUtilsTest, ResultMessageEmptyValueRoundTrip) {
  // An empty buffer is what a void-returning wrapper produces. It shares
  // 'Size == 0' with an out-of-band error, so check it is not mistaken for one.
  auto [TagAddr, Payload] =
      encodeResultMessage(shared::WrapperFunctionBuffer());

  EXPECT_EQ(TagAddr, tagFor(SimpleRemoteEPCResultKind::Value));

  auto Decoded = decodeResultMessage(TagAddr, std::move(Payload));
  ASSERT_THAT_EXPECTED(Decoded, Succeeded());
  EXPECT_TRUE(Decoded->empty());
}

TEST(SimpleRemoteEPCUtilsTest, ResultMessageOutOfBandErrorRoundTrip) {
  constexpr const char *Msg = "Could not deserialize wrapper function arg data";
  auto [TagAddr, Payload] = encodeResultMessage(
      shared::WrapperFunctionBuffer::createOutOfBandError(Msg));

  EXPECT_EQ(TagAddr, tagFor(SimpleRemoteEPCResultKind::OutOfBandError));

  auto Decoded = decodeResultMessage(TagAddr, std::move(Payload));
  ASSERT_THAT_EXPECTED(Decoded, Succeeded());
  ASSERT_NE(Decoded->getOutOfBandError(), nullptr);
  EXPECT_STREQ(Decoded->getOutOfBandError(), Msg);
}

TEST(SimpleRemoteEPCUtilsTest, ResultMessageUnknownKindIsProtocolError) {
  // An unrecognized kind means the peer is speaking a dialect this build does
  // not know, so it ends the session rather than guessing at the payload.
  auto Decoded = decodeResultMessage(
      ExecutorAddr(
          static_cast<uint64_t>(SimpleRemoteEPCResultKind::LastResultKind) + 1),
      shared::WrapperFunctionBuffer());
  EXPECT_THAT_EXPECTED(Decoded, Failed());
}

TEST(SimpleRemoteEPCUtilsTest, ResultMessageMalformedOOBPayloadUnblocksCaller) {
  // A payload that will not decode is not fatal: the call waiting on it still
  // has to be completed, so it comes back as an out-of-band error about itself.
  auto Decoded =
      decodeResultMessage(tagFor(SimpleRemoteEPCResultKind::OutOfBandError),
                          shared::WrapperFunctionBuffer::allocate(4));
  ASSERT_THAT_EXPECTED(Decoded, Succeeded());
  EXPECT_NE(Decoded->getOutOfBandError(), nullptr);
}
