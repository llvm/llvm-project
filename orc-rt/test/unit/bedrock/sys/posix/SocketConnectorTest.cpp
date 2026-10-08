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
#include "orc-rt/bedrock/Session.h"

#include "BedrockTestUtils.h"
#include "CommonTestUtils.h"
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

/// Runs "socket:adopt=<FD>" through the socket connector, targeting a fresh
/// Session. Every case below fails before attach, so the Session never
/// connects (and noErrors would catch an unexpected attach failure).
Error adoptFD(int FD) {
  ConnectorRegistry R;
  if (auto Err = registerSocketConnector(R))
    return Err;
  auto CS = ConnectionSpec::parse("socket:adopt=" + std::to_string(FD));
  if (!CS)
    return CS.takeError();
  Session S(mockExecutorProcessInfo(), noDispatch, noErrors);
  return R.connect(*CS, S, BootstrapInfo(S));
}

bool isOpen(int FD) { return ::fcntl(FD, F_GETFD) != -1; }

TEST(SocketConnectorTest, RejectsPipe) {
  int P[2];
  ASSERT_EQ(::pipe(P), 0);

  EXPECT_THAT_ERROR(adoptFD(P[0]),
                    FailedWithMessage(HasSubstr("is not a socket")));
  EXPECT_TRUE(isOpen(P[0])) << "a rejected descriptor must be left open";

  ::close(P[0]);
  ::close(P[1]);
}

TEST(SocketConnectorTest, RejectsClosedDescriptor) {
  int P[2];
  ASSERT_EQ(::pipe(P), 0);
  ::close(P[0]);
  ::close(P[1]);

  EXPECT_THAT_ERROR(adoptFD(P[0]),
                    FailedWithMessage(HasSubstr("is not a socket")));
}

TEST(SocketConnectorTest, TakesOwnershipOfASocketEvenOnFailure) {
  // A non-stream socket passes the connector's is-a-socket check, so the
  // connector takes ownership of it, but is then rejected when creating the
  // ControllerAccess, which requires a stream socket.
  auto H = makeNativeNonStreamSocket();
  ASSERT_TRUE(H.has_value()) << "could not create a socket for the test";

  EXPECT_THAT_ERROR(adoptFD(*H),
                    FailedWithMessage(HasSubstr("requires a stream socket")));
  EXPECT_FALSE(isNativeSocketOpen(*H))
      << "an adopted socket must be closed when the connection fails";
}

} // namespace
