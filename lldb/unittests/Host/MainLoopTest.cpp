//===-- MainLoopTest.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/common/TCPSocket.h"
#include "gtest/gtest.h"
#include <future>

using namespace lldb_private;

namespace {
class MainLoopTest : public testing::Test {
public:
  static void SetUpTestCase() {
#ifdef _MSC_VER
    WSADATA data;
    ASSERT_EQ(0, WSAStartup(MAKEWORD(2, 2), &data));
#endif
  }

  static void TearDownTestCase() {
#ifdef _MSC_VER
    ASSERT_EQ(0, WSACleanup());
#endif
  }

  void SetUp() override {
    bool child_processes_inherit = false;
    Status error;
    std::unique_ptr<TCPSocket> listen_socket_up(
        new TCPSocket(true, child_processes_inherit));
    ASSERT_TRUE(error.Success());
    error = listen_socket_up->Listen("localhost:0", 5);
    ASSERT_TRUE(error.Success());

    Socket *accept_socket;
    std::future<Status> accept_error = std::async(std::launch::async, [&] {
      return listen_socket_up->Accept(accept_socket);
    });

    std::unique_ptr<TCPSocket> connect_socket_up(
        new TCPSocket(true, child_processes_inherit));
    error = connect_socket_up->Connect(
        llvm::formatv("localhost:{0}", listen_socket_up->GetLocalPortNumber())
            .str());
    ASSERT_TRUE(error.Success());
    ASSERT_TRUE(accept_error.get().Success());

    callback_count = 0;
    socketpair[0] = std::move(connect_socket_up);
    socketpair[1].reset(accept_socket);
  }

  void TearDown() override {
    socketpair[0].reset();
    socketpair[1].reset();
  }

protected:
  MainLoop::Callback make_callback() {
    return [&](MainLoopBase &loop) {
      ++callback_count;
      loop.RequestTermination();
    };
  }
  std::shared_ptr<Socket> socketpair[2];
  unsigned callback_count;
};
} // namespace

TEST_F(MainLoopTest, ReadObject) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());

  MainLoop loop;

  Status error;
  auto handle = loop.RegisterReadObject(socketpair[1], make_callback(), error);
  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(handle);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

TEST_F(MainLoopTest, TerminatesImmediately) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());
  ASSERT_TRUE(socketpair[1]->Write(&X, len).Success());

  MainLoop loop;
  Status error;
  auto handle0 = loop.RegisterReadObject(socketpair[0], make_callback(), error);
  ASSERT_TRUE(error.Success());
  auto handle1 = loop.RegisterReadObject(socketpair[1], make_callback(), error);
  ASSERT_TRUE(error.Success());

  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

#ifdef LLVM_ON_UNIX
TEST_F(MainLoopTest, DetectsEOF) {
  PseudoTerminal term;
  ASSERT_TRUE(term.OpenFirstAvailableMaster(O_RDWR, nullptr, 0));
  ASSERT_TRUE(term.OpenSlave(O_RDWR | O_NOCTTY, nullptr, 0));
  auto conn = llvm::make_unique<ConnectionFileDescriptor>(
      term.ReleaseMasterFileDescriptor(), true);

  Status error;
  MainLoop loop;
  auto handle =
      loop.RegisterReadObject(conn->GetReadObject(), make_callback(), error);
  ASSERT_TRUE(error.Success());
  term.CloseSlaveFileDescriptor();

  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

TEST_F(MainLoopTest, Signal) {
  MainLoop loop;
  Status error;

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());
  kill(getpid(), SIGUSR1);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}
#endif
