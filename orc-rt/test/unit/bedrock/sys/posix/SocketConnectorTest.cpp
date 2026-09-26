//===- SocketConnectorTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the socket:adopt connector's validation of its descriptor.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SocketConnector.h"

#include "ErrorMatchers.h"
#include "bedrock/SocketTestUtils.h"

#include "gtest/gtest.h"

#include <fcntl.h>
#include <string>
#include <unistd.h>

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::HasSubstr;

namespace {

/// Runs "socket:adopt=<FD>" through the socket connector. The GetAttachInfo it
/// supplies records that it was called and then fails, so a descriptor that
/// passes validation stops there rather than being attached.
Error adoptFD(int FD, bool &AttachInfoRequested) {
  ConnectorRegistry R;
  if (auto Err = registerSocketConnector(R))
    return Err;
  auto CS = ConnectionSpec::parse("socket:adopt=" + std::to_string(FD));
  if (!CS)
    return CS.takeError();
  return R.connect(
      [&]() noexcept -> Expected<ConnectorRegistry::AttachInfo> {
        AttachInfoRequested = true;
        return make_error<StringError>("attach info requested");
      },
      *CS);
}

bool isOpen(int FD) { return ::fcntl(FD, F_GETFD) != -1; }

TEST(SocketConnectorTest, RejectsPipe) {
  int P[2];
  ASSERT_EQ(::pipe(P), 0);

  bool AttachInfoRequested = false;
  EXPECT_THAT_ERROR(adoptFD(P[0], AttachInfoRequested),
                    FailedWithMessage(HasSubstr("is not a socket")));
  EXPECT_FALSE(AttachInfoRequested);
  EXPECT_TRUE(isOpen(P[0])) << "a rejected descriptor must be left open";

  ::close(P[0]);
  ::close(P[1]);
}

TEST(SocketConnectorTest, RejectsClosedDescriptor) {
  int P[2];
  ASSERT_EQ(::pipe(P), 0);
  ::close(P[0]);
  ::close(P[1]);

  bool AttachInfoRequested = false;
  EXPECT_THAT_ERROR(adoptFD(P[0], AttachInfoRequested),
                    FailedWithMessage(HasSubstr("is not a socket")));
  EXPECT_FALSE(AttachInfoRequested);
}

TEST(SocketConnectorTest, TakesOwnershipOfASocketEvenOnFailure) {
  auto H = makeNativeSocket();
  ASSERT_TRUE(H.has_value());

  bool AttachInfoRequested = false;
  EXPECT_THAT_ERROR(adoptFD(*H, AttachInfoRequested),
                    FailedWithMessage("attach info requested"));
  EXPECT_TRUE(AttachInfoRequested);
  EXPECT_FALSE(isNativeSocketOpen(*H))
      << "an adopted socket must be closed when the connection fails";
}

} // namespace
