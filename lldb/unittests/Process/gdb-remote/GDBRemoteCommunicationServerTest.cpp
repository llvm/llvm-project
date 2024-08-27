//===-- GDBRemoteCommunicationServerTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "GDBRemoteTestUtils.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "lldb/Utility/Connection.h"
#include "lldb/Utility/UnimplementedError.h"

namespace lldb_private {
namespace process_gdb_remote {

TEST(GDBRemoteCommunicationServerTest, SendErrorResponse_ErrorNumber) {
  MockServerWithMockConnection server;
  server.SendErrorResponse(0x42);

  EXPECT_THAT(server.GetPackets(), testing::ElementsAre("$E42#ab"));
}

TEST(GDBRemoteCommunicationServerTest, SendErrorResponse_Status) {
  MockServerWithMockConnection server;
  Status status(0x42, lldb::eErrorTypeGeneric, "Test error message");
  server.SendErrorResponse(status);

  EXPECT_THAT(
      server.GetPackets(),
      testing::ElementsAre("$E42;54657374206572726f72206d657373616765#ad"));
}

TEST(GDBRemoteCommunicationServerTest, SendErrorResponse_UnimplementedError) {
  MockServerWithMockConnection server;

  auto error = llvm::make_error<UnimplementedError>();
  server.SendErrorResponse(std::move(error));

  EXPECT_THAT(server.GetPackets(), testing::ElementsAre("$#00"));
}

TEST(GDBRemoteCommunicationServerTest, SendErrorResponse_StringError) {
  MockServerWithMockConnection server;

  auto error = llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "String error test");
  server.SendErrorResponse(std::move(error));

  EXPECT_THAT(
      server.GetPackets(),
      testing::ElementsAre("$Eff;537472696e67206572726f722074657374#b0"));
}

TEST(GDBRemoteCommunicationServerTest, SendErrorResponse_ErrorList) {
  MockServerWithMockConnection server;

  auto error = llvm::joinErrors(llvm::make_error<UnimplementedError>(),
                                llvm::make_error<UnimplementedError>());

  server.SendErrorResponse(std::move(error));
  // Make sure only one packet is sent even when there are multiple errors.
  EXPECT_EQ(server.GetPackets().size(), 1UL);
}

} // namespace process_gdb_remote
} // namespace lldb_private
