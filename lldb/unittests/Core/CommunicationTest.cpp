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

#include <thread>

#if LLDB_ENABLE_POSIX
#include <fcntl.h>
#endif

using namespace lldb_private;

#ifndef _WIN32
TEST(CommunicationTest, SynchronizeWhileClosing) {
  // Set up a communication object reading from a pipe.
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(/*child_process_inherit=*/false).ToError(),
                    llvm::Succeeded());

  Communication comm("test");
  comm.SetConnection(std::make_unique<ConnectionFileDescriptor>(
      pipe.ReleaseReadFileDescriptor(), /*owns_fd=*/true));
  comm.SetCloseOnEOF(true);
  ASSERT_TRUE(comm.StartReadThread());

  // Ensure that we can safely synchronize with the read thread while it is
  // closing the read end (in response to us closing the write end).
  pipe.CloseWriteFileDescriptor();
  comm.SynchronizeWithReadThread();

  ASSERT_TRUE(comm.StopReadThread());
}
#endif

#if LLDB_ENABLE_POSIX
TEST(CommunicationTest, WriteAll) {
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
  std::vector<char> data(1024 * 1024, 0x80);
  EXPECT_EQ(write_comm.WriteAll(data.data(), data.size(), conn_status, &error),
            data.size());
  EXPECT_EQ(conn_status, lldb::eConnectionStatusSuccess);
  EXPECT_FALSE(error.Fail());

  // Close the write end in order to trigger EOF.
  write_comm.Disconnect();
  read_thread.join();
}
#endif
