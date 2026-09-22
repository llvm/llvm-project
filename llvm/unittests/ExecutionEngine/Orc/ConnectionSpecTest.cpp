//===- ConnectionSpecTest.cpp - Unit tests for ConnectionSpec -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/ConnectionSpec.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(ConnectionSpecTest, TransportAndDescriptorOnly) {
  // The grammar permits a missing action, though no transport defines one.
  auto Spec = ConnectionSpec::parse("socket=3");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "socket");
  EXPECT_EQ(Spec->getAction(), "");
  EXPECT_EQ(Spec->getDescriptor(), "3");
  EXPECT_EQ(Spec->str(), "socket=3");
}

TEST(ConnectionSpecTest, MultiPartDescriptor) {
  // Descriptor syntax is the transport's business, so a comma-separated pair
  // passes through untouched.
  auto Spec = ConnectionSpec::parse("pipe:adopt=3,4");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "pipe");
  EXPECT_EQ(Spec->getAction(), "adopt");
  EXPECT_EQ(Spec->getDescriptor(), "3,4");
  EXPECT_EQ(Spec->str(), "pipe:adopt=3,4");
}

TEST(ConnectionSpecTest, TransportActionAndDescriptor) {
  auto Spec = ConnectionSpec::parse("tcp:connect=localhost:20000");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "tcp");
  EXPECT_EQ(Spec->getAction(), "connect");
  EXPECT_EQ(Spec->getDescriptor(), "localhost:20000");
  EXPECT_EQ(Spec->str(), "tcp:connect=localhost:20000");
}

TEST(ConnectionSpecTest, EmptyDescriptorIsLegal) {
  auto Spec = ConnectionSpec::parse("socket:adopt=");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "socket");
  EXPECT_EQ(Spec->getAction(), "adopt");
  EXPECT_EQ(Spec->getDescriptor(), "");
  EXPECT_EQ(Spec->str(), "socket:adopt=");
}

TEST(ConnectionSpecTest, ColonInDescriptorAfterAction) {
  // The ':' search must be confined to the text before the first '=', so a
  // descriptor may itself contain ':' unescaped.
  auto Spec = ConnectionSpec::parse("tcp:listen=[::1]:0");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "tcp");
  EXPECT_EQ(Spec->getAction(), "listen");
  EXPECT_EQ(Spec->getDescriptor(), "[::1]:0");
  EXPECT_EQ(Spec->str(), "tcp:listen=[::1]:0");
}

TEST(ConnectionSpecTest, EqualsInDescriptor) {
  // Only the *first* '=' delimits transport/action from the descriptor, so a
  // descriptor may itself contain '=' unescaped.
  auto Spec = ConnectionSpec::parse("unix:listen=/tmp/a=b.sock");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "unix");
  EXPECT_EQ(Spec->getAction(), "listen");
  EXPECT_EQ(Spec->getDescriptor(), "/tmp/a=b.sock");
  EXPECT_EQ(Spec->str(), "unix:listen=/tmp/a=b.sock");
}

TEST(ConnectionSpecTest, ColonInActionIsLegal) {
  // Only the *first* ':' before the '=' delimits the transport from the
  // action; the action is an opaque token, so the parser passes any
  // remaining ':' through rather than rejecting it. Transports are free to
  // reject actions they don't recognize.
  auto Spec = ConnectionSpec::parse("tcp::listen=[::1]:0");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "tcp");
  EXPECT_EQ(Spec->getAction(), ":listen");
  EXPECT_EQ(Spec->getDescriptor(), "[::1]:0");
  EXPECT_EQ(Spec->str(), "tcp::listen=[::1]:0");
}

TEST(ConnectionSpecTest, NamesArePreservedVerbatim) {
  // The parser does not case-normalize: callers match case-sensitively.
  auto Spec = ConnectionSpec::parse("TCP:Connect=localhost:20000");
  ASSERT_THAT_EXPECTED(Spec, Succeeded());
  EXPECT_EQ(Spec->getTransport(), "TCP");
  EXPECT_EQ(Spec->getAction(), "Connect");
}

TEST(ConnectionSpecTest, MissingEqualsIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse("tcp:connect"), Failed());
}

TEST(ConnectionSpecTest, EmptySpecIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse(""), Failed());
}

TEST(ConnectionSpecTest, EmptyTransportIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse("=3"), Failed());
}

TEST(ConnectionSpecTest, DescriptorOnlyIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse("="), Failed());
}

TEST(ConnectionSpecTest, EmptyActionIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse("tcp:=localhost:20000"), Failed());
}

TEST(ConnectionSpecTest, EmptyTransportAndActionIsError) {
  EXPECT_THAT_EXPECTED(ConnectionSpec::parse(":=3"), Failed());
}

} // namespace
