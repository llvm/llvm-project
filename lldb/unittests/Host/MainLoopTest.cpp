//===-- MainLoopTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoop.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/common/TCPSocket.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <future>
#include <thread>

using namespace lldb_private;

namespace {
class MainLoopTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, Socket> subsystems;

  void SetUp() override {
    Status error;
    auto listen_socket_up = std::make_unique<TCPSocket>(true);
    ASSERT_TRUE(error.Success());
    error = listen_socket_up->Listen("localhost:0", 5);
    ASSERT_TRUE(error.Success());

    Socket *accept_socket;
    auto connect_socket_up = std::make_unique<TCPSocket>(true);
    error = connect_socket_up->Connect(
        llvm::formatv("localhost:{0}", listen_socket_up->GetLocalPortNumber())
            .str());
    ASSERT_TRUE(error.Success());
    ASSERT_TRUE(listen_socket_up->Accept(std::chrono::seconds(1), accept_socket)
                    .Success());

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

TEST_F(MainLoopTest, NoSpuriousReads) {
  // Write one byte into the socket.
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());

  MainLoop loop;

  Status error;
  auto handle = loop.RegisterReadObject(
      socketpair[1],
      [this](MainLoopBase &) {
        if (callback_count == 0) {
          // Read the byte back the first time we're called. After that, the
          // socket is empty, and we should not be called anymore.
          char X;
          size_t len = sizeof(X);
          EXPECT_THAT_ERROR(socketpair[1]->Read(&X, len).ToError(),
                            llvm::Succeeded());
          EXPECT_EQ(len, sizeof(X));
        }
        ++callback_count;
      },
      error);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  // Terminate the loop after one second.
  loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                   std::chrono::seconds(1));
  ASSERT_THAT_ERROR(loop.Run().ToError(), llvm::Succeeded());

  // Make sure the callback was called only once.
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

TEST_F(MainLoopTest, PendingCallback) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());

  MainLoop loop;
  Status error;
  auto handle = loop.RegisterReadObject(
      socketpair[1],
      [&](MainLoopBase &loop) {
        // Both callbacks should be called before the loop terminates.
        loop.AddPendingCallback(make_callback());
        loop.AddPendingCallback(make_callback());
        loop.RequestTermination();
      },
      error);
  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(handle);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(2u, callback_count);
}

TEST_F(MainLoopTest, PendingCallbackCalledOnlyOnce) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());

  MainLoop loop;
  Status error;
  auto handle = loop.RegisterReadObject(
      socketpair[1],
      [&](MainLoopBase &loop) {
        // Add one pending callback on the first iteration.
        if (callback_count == 0) {
          loop.AddPendingCallback([&](MainLoopBase &loop) {
            callback_count++;
          });
        }
        // Terminate the loop on second iteration.
        if (callback_count++ >= 1)
          loop.RequestTermination();
      },
      error);
  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(handle);
  ASSERT_TRUE(loop.Run().Success());
  // 2 iterations of read callback + 1 call of pending callback.
  ASSERT_EQ(3u, callback_count);
}

TEST_F(MainLoopTest, PendingCallbackTrigger) {
  MainLoop loop;
  std::promise<void> add_callback2;
  bool callback1_called = false;
  loop.AddPendingCallback([&](MainLoopBase &loop) {
    callback1_called = true;
    add_callback2.set_value();
  });
  Status error;
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  bool callback2_called = false;
  std::thread callback2_adder([&]() {
    add_callback2.get_future().get();
    loop.AddPendingCallback([&](MainLoopBase &loop) {
      callback2_called = true;
      loop.RequestTermination();
    });
  });
  ASSERT_THAT_ERROR(loop.Run().ToError(), llvm::Succeeded());
  callback2_adder.join();
  ASSERT_TRUE(callback1_called);
  ASSERT_TRUE(callback2_called);
}

TEST_F(MainLoopTest, ManyPendingCallbacks) {
  MainLoop loop;
  Status error;
  // Try to fill up the pipe buffer and make sure bad things don't happen. This
  // is a regression test for the case where writing to the interrupt pipe
  // caused a deadlock when the pipe filled up (either because the main loop was
  // not running, because it was slow, or because it was busy/blocked doing
  // something else).
  for (int i = 0; i < 65536; ++i)
    loop.AddPendingCallback(
        [&](MainLoopBase &loop) { loop.RequestTermination(); });
  ASSERT_TRUE(loop.Run().Success());
}

TEST_F(MainLoopTest, CallbackWithTimeout) {
  MainLoop loop;
  loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                   std::chrono::seconds(2));
  auto start = std::chrono::steady_clock::now();
  ASSERT_THAT_ERROR(loop.Run().takeError(), llvm::Succeeded());
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::seconds(2));
}

TEST_F(MainLoopTest, TimedCallbacksRunInOrder) {
  MainLoop loop;
  auto start = std::chrono::steady_clock::now();
  std::chrono::milliseconds epsilon(10);
  std::vector<int> order;
  auto add_cb = [&](int id) {
    loop.AddCallback([&order, id](MainLoopBase &) { order.push_back(id); },
                     start + id * epsilon);
  };
  add_cb(3);
  add_cb(2);
  add_cb(4);
  add_cb(1);
  loop.AddCallback([](MainLoopBase &loop) { loop.RequestTermination(); },
                   start + 5 * epsilon);
  ASSERT_THAT_ERROR(loop.Run().takeError(), llvm::Succeeded());
  EXPECT_GE(std::chrono::steady_clock::now() - start, 5 * epsilon);
  ASSERT_THAT(order, testing::ElementsAre(1, 2, 3, 4));
}

TEST_F(MainLoopTest, TimedCallbackShortensSleep) {
  MainLoop loop;
  auto start = std::chrono::steady_clock::now();
  bool long_callback_called = false;
  loop.AddCallback(
      [&](MainLoopBase &loop) {
        long_callback_called = true;
        loop.RequestTermination();
      },
      std::chrono::seconds(30));
  std::future<Status> async_run =
      std::async(std::launch::async, &MainLoop::Run, std::ref(loop));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  bool short_callback_called = false;
  loop.AddCallback(
      [&](MainLoopBase &loop) {
        short_callback_called = true;
        loop.RequestTermination();
      },
      std::chrono::seconds(1));
  ASSERT_THAT_ERROR(async_run.get().takeError(), llvm::Succeeded());
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(10));
  EXPECT_TRUE(short_callback_called);
  EXPECT_FALSE(long_callback_called);
}

#ifdef LLVM_ON_UNIX
TEST_F(MainLoopTest, DetectsEOF) {

  PseudoTerminal term;
  ASSERT_THAT_ERROR(term.OpenFirstAvailablePrimary(O_RDWR), llvm::Succeeded());
  ASSERT_THAT_ERROR(term.OpenSecondary(O_RDWR | O_NOCTTY), llvm::Succeeded());
  auto conn = std::make_unique<ConnectionFileDescriptor>(
      term.ReleasePrimaryFileDescriptor(), true);

  Status error;
  MainLoop loop;
  auto handle =
      loop.RegisterReadObject(conn->GetReadObject(), make_callback(), error);
  ASSERT_TRUE(error.Success());
  term.CloseSecondaryFileDescriptor();

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

TEST_F(MainLoopTest, SignalOnOtherThread) {
  MainLoop loop;
  Status error;

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());
  std::thread([] { pthread_kill(pthread_self(), SIGUSR1); }).join();
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

// Test that a signal which is not monitored by the MainLoop does not
// cause a premature exit.
TEST_F(MainLoopTest, UnmonitoredSignal) {
  MainLoop loop;
  Status error;
  struct sigaction sa;
  sa.sa_sigaction = [](int, siginfo_t *, void *) { };
  sa.sa_flags = SA_SIGINFO; // important: no SA_RESTART
  sigemptyset(&sa.sa_mask);
  ASSERT_EQ(0, sigaction(SIGUSR2, &sa, nullptr));

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());
  kill(getpid(), SIGUSR2);
  kill(getpid(), SIGUSR1);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

// Test that two callbacks can be registered for the same signal
// and unregistered independently.
TEST_F(MainLoopTest, TwoSignalCallbacks) {
  MainLoop loop;
  Status error;
  unsigned callback2_count = 0;
  unsigned callback3_count = 0;

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());

  {
    // Run a single iteration with two callbacks enabled.
    auto handle2 = loop.RegisterSignal(
        SIGUSR1, [&](MainLoopBase &loop) { ++callback2_count; }, error);
    ASSERT_TRUE(error.Success());

    kill(getpid(), SIGUSR1);
    ASSERT_TRUE(loop.Run().Success());
    ASSERT_EQ(1u, callback_count);
    ASSERT_EQ(1u, callback2_count);
    ASSERT_EQ(0u, callback3_count);
  }

  {
    // Make sure that remove + add new works.
    auto handle3 = loop.RegisterSignal(
        SIGUSR1, [&](MainLoopBase &loop) { ++callback3_count; }, error);
    ASSERT_TRUE(error.Success());

    kill(getpid(), SIGUSR1);
    ASSERT_TRUE(loop.Run().Success());
    ASSERT_EQ(2u, callback_count);
    ASSERT_EQ(1u, callback2_count);
    ASSERT_EQ(1u, callback3_count);
  }

  // Both extra callbacks should be unregistered now.
  kill(getpid(), SIGUSR1);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(3u, callback_count);
  ASSERT_EQ(1u, callback2_count);
  ASSERT_EQ(1u, callback3_count);
}
#endif
