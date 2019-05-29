//===-- ConnectionFileDescriptorTest.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SocketTestUtilities.h"
#include "gtest/gtest.h"

#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Utility/UriParser.h"

using namespace lldb_private;

class ConnectionFileDescriptorTest : public testing::Test {
public:
  void SetUp() override {
    ASSERT_THAT_ERROR(Socket::Initialize(), llvm::Succeeded());
  }

  void TearDown() override { Socket::Terminate(); }

  void TestGetURI(std::string ip) {
    std::unique_ptr<TCPSocket> socket_a_up;
    std::unique_ptr<TCPSocket> socket_b_up;
    if (!IsAddressFamilySupported(ip)) {
      GTEST_LOG_(WARNING) << "Skipping test due to missing IPv"
                          << (IsIPv4(ip) ? "4" : "6") << " support.";
      return;
    }
    CreateTCPConnectedSockets(ip, &socket_a_up, &socket_b_up);
    auto socket = socket_a_up.release();
    ConnectionFileDescriptor connection_file_descriptor(socket);

    llvm::StringRef scheme;
    llvm::StringRef hostname;
    int port;
    llvm::StringRef path;
    std::string uri(connection_file_descriptor.GetURI());
    EXPECT_TRUE(UriParser::Parse(uri, scheme, hostname, port, path));
    EXPECT_EQ(ip, hostname);
    EXPECT_EQ(socket->GetRemotePortNumber(), port);
  }
};

TEST_F(ConnectionFileDescriptorTest, TCPGetURIv4) { TestGetURI("127.0.0.1"); }

TEST_F(ConnectionFileDescriptorTest, TCPGetURIv6) { TestGetURI("::1"); }