//===- ConnectionSpecTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ConnectionSpec.h"
#include "orc-rt/support/Error.h"
#include "gtest/gtest.h"

using namespace orc_rt;

namespace {

TEST(ConnectionSpecTest, TransportActionAndDescriptor) {
  auto CS = ConnectionSpec::parse("tcp:connect=localhost:20000");
  ASSERT_TRUE(!!CS);
  EXPECT_EQ(CS->transport(), "tcp");
  EXPECT_EQ(CS->action(), "connect");
  EXPECT_EQ(CS->descriptor(), "localhost:20000");
}

TEST(ConnectionSpecTest, TransportActionAndMultiPartDescriptor) {
  auto CS = ConnectionSpec::parse("pipe:adopt=3,4");
  ASSERT_TRUE(!!CS);
  EXPECT_EQ(CS->transport(), "pipe");
  EXPECT_EQ(CS->action(), "adopt");
  EXPECT_EQ(CS->descriptor(), "3,4");
}

TEST(ConnectionSpecTest, EmptyDescriptorIsAllowed) {
  // Whether an empty descriptor means anything is the transport's business, so
  // it isn't a syntax error here.
  auto CS = ConnectionSpec::parse("socket:adopt=");
  ASSERT_TRUE(!!CS);
  EXPECT_EQ(CS->transport(), "socket");
  EXPECT_EQ(CS->descriptor(), "");
}

TEST(ConnectionSpecTest, DescriptorKeepsColonsAndEquals) {
  auto CS = ConnectionSpec::parse("tcp:listen=[::1]:0");
  ASSERT_TRUE(!!CS);
  EXPECT_EQ(CS->transport(), "tcp");
  EXPECT_EQ(CS->action(), "listen");
  EXPECT_EQ(CS->descriptor(), "[::1]:0");

  // Only the first '=' is structural.
  auto Odd = ConnectionSpec::parse("unix:listen=/tmp/a=b.sock");
  ASSERT_TRUE(!!Odd);
  EXPECT_EQ(Odd->descriptor(), "/tmp/a=b.sock");
}

TEST(ConnectionSpecTest, MissingEqualsFails) {
  auto CS = ConnectionSpec::parse("tcp:connect");
  EXPECT_FALSE(!!CS);
  consumeError(CS.takeError());
}

TEST(ConnectionSpecTest, EmptyTransportFails) {
  auto CS = ConnectionSpec::parse("=localhost:20000");
  EXPECT_FALSE(!!CS);
  consumeError(CS.takeError());

  auto WithAction = ConnectionSpec::parse(":connect=localhost:20000");
  EXPECT_FALSE(!!WithAction);
  consumeError(WithAction.takeError());
}

TEST(ConnectionSpecTest, EmptyActionFails) {
  auto CS = ConnectionSpec::parse("tcp:=localhost:20000");
  EXPECT_FALSE(!!CS);
  consumeError(CS.takeError());
}

TEST(ConnectionSpecTest, ErrorMessageQuotesTheSpec) {
  auto CS = ConnectionSpec::parse("tcp:connect");
  ASSERT_FALSE(!!CS);
  EXPECT_NE(toString(CS.takeError()).find("'tcp:connect'"), std::string::npos);
}

} // namespace
