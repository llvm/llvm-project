//===-- CommunicationTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Communication.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/Pipe.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include "TestingSupport/Host/SocketTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"

#include <thread>

#if LLDB_ENABLE_POSIX
#include <fcntl.h>
#endif

using namespace lldb_private;

class CommunicationTest : public testing::Test {
private:
  SubsystemRAII<Socket> m_subsystems;
};

static void CommunicationReadTest(bool use_read_thread) {
  std::unique_ptr<TCPSocket> a, b;
  ASSERT_TRUE(CreateTCPConnectedSockets("localhost", &a, &b));

  size_t num_bytes = 4;
  ASSERT_THAT_ERROR(a->Write("test", num_bytes).ToError(), llvm::Succeeded());
  ASSERT_EQ(num_bytes, 4U);

  Communication comm("test");
  comm.SetConnection(std::make_unique<ConnectionFileDescriptor>(b.release()));
  comm.SetCloseOnEOF(true);

  if (use_read_thread) {
    ASSERT_TRUE(comm.StartReadThread());
  }

  // This read should wait for the data to become available and return it.
  lldb::ConnectionStatus status = lldb::eConnectionStatusSuccess;
  char buf[16];
  Status error;
  EXPECT_EQ(
      comm.Read(buf, sizeof(buf), std::chrono::seconds(5), status, &error), 4U);
  EXPECT_EQ(status, lldb::eConnectionStatusSuccess);
  EXPECT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  buf[4] = 0;
  EXPECT_STREQ(buf, "test");

  // These reads should time out as there is no more data.
  error.Clear();
  EXPECT_EQ(comm.Read(buf, sizeof(buf), std::chrono::microseconds(10), status,
                      &error),
            0U);
  EXPECT_EQ(status, lldb::eConnectionStatusTimedOut);
  EXPECT_THAT_ERROR(error.ToError(), llvm::Failed());

  // 0 is special-cased, so we test it separately.
  error.Clear();
  EXPECT_EQ(
      comm.Read(buf, sizeof(buf), std::chrono::seconds(0), status, &error), 0U);
  EXPECT_EQ(status, lldb::eConnectionStatusTimedOut);
  EXPECT_THAT_ERROR(error.ToError(), llvm::Failed());

  // This read should return EOF.
  ASSERT_THAT_ERROR(a->Close().ToError(), llvm::Succeeded());
  error.Clear();
  EXPECT_EQ(
      comm.Read(buf, sizeof(buf), std::chrono::seconds(5), status, &error), 0U);
  EXPECT_EQ(status, lldb::eConnectionStatusEndOfFile);
  EXPECT_THAT_ERROR(error.ToError(), llvm::Succeeded());

  // JoinReadThread() should just return immediately since there was no read
  // thread started.
  EXPECT_TRUE(comm.JoinReadThread());
}

TEST_F(CommunicationTest, Read) {
  CommunicationReadTest(/*use_thread=*/false);
}

TEST_F(CommunicationTest, ReadThread) {
  CommunicationReadTest(/*use_thread=*/true);
}

TEST_F(CommunicationTest, SynchronizeWhileClosing) {
  std::unique_ptr<TCPSocket> a, b;
  ASSERT_TRUE(CreateTCPConnectedSockets("localhost", &a, &b));

  Communication comm("test");
  comm.SetConnection(std::make_unique<ConnectionFileDescriptor>(b.release()));
  comm.SetCloseOnEOF(true);
  ASSERT_TRUE(comm.StartReadThread());

  // Ensure that we can safely synchronize with the read thread while it is
  // closing the read end (in response to us closing the write end).
  ASSERT_THAT_ERROR(a->Close().ToError(), llvm::Succeeded());
  comm.SynchronizeWithReadThread();

  ASSERT_TRUE(comm.StopReadThread());
}

#if LLDB_ENABLE_POSIX
TEST_F(CommunicationTest, WriteAll) {
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(/*child_process_inherit=*/false).ToError(),
                    llvm::Succeeded());

  // Make the write end non-blocking in order to easily reproduce a partial
  // write.
  int write_fd = pipe.ReleaseWriteFileDescriptor();
  int flags = fcntl(write_fd, F_GETFL);
  ASSERT_NE(flags, -1);
  ASSERT_NE(fcntl(write_fd, F_SETFL, flags | O_NONBLOCK), -1);

  ConnectionFileDescriptor read_conn{pipe.ReleaseReadFileDescriptor(),
                                     /*owns_fd=*/true};
  Communication write_comm("test");
  write_comm.SetConnection(
      std::make_unique<ConnectionFileDescriptor>(write_fd, /*owns_fd=*/true));

  std::thread read_thread{[&read_conn]() {
    // Read using a smaller buffer to increase chances of partial write.
    char buf[128 * 1024];
    lldb::ConnectionStatus conn_status;

    do {
      read_conn.Read(buf, sizeof(buf), std::chrono::seconds(1), conn_status,
                     nullptr);
    } while (conn_status != lldb::eConnectionStatusEndOfFile);
  }};

  // Write 1 MiB of data into the pipe.
  lldb::ConnectionStatus conn_status;
  Status error;
  std::vector<uint8_t> data(1024 * 1024, 0x80);
  EXPECT_EQ(write_comm.WriteAll(data.data(), data.size(), conn_status, &error),
            data.size());
  EXPECT_EQ(conn_status, lldb::eConnectionStatusSuccess);
  EXPECT_FALSE(error.Fail());

  // Close the write end in order to trigger EOF.
  write_comm.Disconnect();
  read_thread.join();
}
#endif
