//===-- PipeTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Pipe.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <chrono>
#include <fcntl.h>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

using namespace lldb_private;

class PipeTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

TEST_F(PipeTest, CreateWithUniqueName) {
  Pipe pipe;
  llvm::SmallString<0> name;
  ASSERT_THAT_ERROR(pipe.CreateWithUniqueName("PipeTest-CreateWithUniqueName",
                                              /*child_process_inherit=*/false,
                                              name)
                        .ToError(),
                    llvm::Succeeded());
}

// Test broken
#ifndef _WIN32
TEST_F(PipeTest, OpenAsReader) {
  Pipe pipe;
  llvm::SmallString<0> name;
  ASSERT_THAT_ERROR(pipe.CreateWithUniqueName("PipeTest-OpenAsReader",
                                              /*child_process_inherit=*/false,
                                              name)
                        .ToError(),
                    llvm::Succeeded());

  // Ensure name is not null-terminated
  size_t name_len = name.size();
  name += "foobar";
  llvm::StringRef name_ref(name.data(), name_len);
  ASSERT_THAT_ERROR(
      pipe.OpenAsReader(name_ref, /*child_process_inherit=*/false).ToError(),
      llvm::Succeeded());

  ASSERT_TRUE(pipe.CanRead());
}
#endif

// Tests flaky on Windows
#ifndef _WIN32
TEST_F(PipeTest, WriteWithTimeout) {
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(false).ToError(), llvm::Succeeded());

  // The pipe buffer is 1024 for PipeWindows and at least 512 on Darwin.
  // In Linux versions before 2.6.11, the capacity of a pipe was the same as the
  // system page size (e.g., 4096 bytes on i386).
  // Since Linux 2.6.11, the pipe capacity is 16 pages (i.e., 65,536 bytes in a
  // system with a page size of 4096 bytes).
  // Since Linux 2.6.35, the default pipe capacity is 16 pages, but the capacity
  // can be queried and set using the fcntl(2) F_GETPIPE_SZ and F_SETPIPE_SZ
  // operations:

#if !defined(_WIN32) && defined(F_SETPIPE_SZ)
  ::fcntl(pipe.GetWriteFileDescriptor(), F_SETPIPE_SZ, 4096);
#endif

  const size_t buf_size = 66000;

  // Note write_chunk_size must be less than the pipe buffer.
  const size_t write_chunk_size = 234;

  std::vector<int32_t> write_buf(buf_size / sizeof(int32_t));
  std::iota(write_buf.begin(), write_buf.end(), 0);
  std::vector<int32_t> read_buf(write_buf.size() + 100, -1);

  char *write_ptr = reinterpret_cast<char *>(write_buf.data());
  char *read_ptr = reinterpret_cast<char *>(read_buf.data());
  size_t write_bytes = 0;
  size_t read_bytes = 0;

  // Write to the pipe until it is full.
  while (write_bytes + write_chunk_size <= buf_size) {
    llvm::Expected<size_t> num_bytes =
        pipe.Write(write_ptr + write_bytes, write_chunk_size,
                   std::chrono::milliseconds(10));
    if (num_bytes) {
      write_bytes += *num_bytes;
    } else {
      ASSERT_THAT_ERROR(num_bytes.takeError(), llvm::Failed());
      break; // The write buffer is full.
    }
  }
  ASSERT_LE(write_bytes + write_chunk_size, buf_size)
      << "Pipe buffer larger than expected";

  // Attempt a write with a long timeout.
  auto start_time = std::chrono::steady_clock::now();
  // TODO: Assert a specific error (EAGAIN?) here.
  ASSERT_THAT_EXPECTED(pipe.Write(write_ptr + write_bytes, write_chunk_size,
                                  std::chrono::seconds(2)),
                       llvm::Failed());
  auto dur = std::chrono::steady_clock::now() - start_time;
  ASSERT_GE(dur, std::chrono::seconds(2));

  // Attempt a write with a short timeout.
  start_time = std::chrono::steady_clock::now();
  ASSERT_THAT_EXPECTED(pipe.Write(write_ptr + write_bytes, write_chunk_size,
                                  std::chrono::milliseconds(200)),
                       llvm::Failed());
  dur = std::chrono::steady_clock::now() - start_time;
  ASSERT_GE(dur, std::chrono::milliseconds(200));
  ASSERT_LT(dur, std::chrono::seconds(2));

  // Drain the pipe.
  while (read_bytes < write_bytes) {
    llvm::Expected<size_t> num_bytes =
        pipe.Read(read_ptr + read_bytes, write_bytes - read_bytes,
                  std::chrono::milliseconds(10));
    ASSERT_THAT_EXPECTED(num_bytes, llvm::Succeeded());
    read_bytes += *num_bytes;
  }

  // Be sure the pipe is empty.
  ASSERT_THAT_EXPECTED(
      pipe.Read(read_ptr + read_bytes, 100, std::chrono::milliseconds(10)),
      llvm::Failed());

  // Check that we got what we wrote.
  ASSERT_EQ(write_bytes, read_bytes);
  ASSERT_TRUE(std::equal(write_buf.begin(),
                         write_buf.begin() + write_bytes / sizeof(uint32_t),
                         read_buf.begin()));

  // Write to the pipe again and check that it succeeds.
  ASSERT_THAT_EXPECTED(
      pipe.Write(write_ptr, write_chunk_size, std::chrono::milliseconds(10)),
      llvm::Succeeded());
}

TEST_F(PipeTest, ReadWithTimeout) {
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(false).ToError(), llvm::Succeeded());

  char buf[100];
  // The pipe is initially empty. A polling read returns immediately.
  ASSERT_THAT_EXPECTED(pipe.Read(buf, sizeof(buf), std::chrono::seconds(0)),
                       llvm::Failed());

  // With a timeout, we should wait for at least this amount of time (but not
  // too much).
  auto start = std::chrono::steady_clock::now();
  ASSERT_THAT_EXPECTED(
      pipe.Read(buf, sizeof(buf), std::chrono::milliseconds(200)),
      llvm::Failed());
  auto dur = std::chrono::steady_clock::now() - start;
  EXPECT_GT(dur, std::chrono::milliseconds(200));
  EXPECT_LT(dur, std::chrono::seconds(2));

  // Write something into the pipe, and read it back. The blocking read call
  // should return even though it hasn't filled the buffer.
  llvm::StringRef hello_world("Hello world!");
  ASSERT_THAT_EXPECTED(pipe.Write(hello_world.data(), hello_world.size()),
                       llvm::HasValue(hello_world.size()));
  ASSERT_THAT_EXPECTED(pipe.Read(buf, sizeof(buf)),
                       llvm::HasValue(hello_world.size()));
  EXPECT_EQ(llvm::StringRef(buf, hello_world.size()), hello_world);

  // Now write something and try to read it in chunks.
  memset(buf, 0, sizeof(buf));
  ASSERT_THAT_EXPECTED(pipe.Write(hello_world.data(), hello_world.size()),
                       llvm::HasValue(hello_world.size()));
  ASSERT_THAT_EXPECTED(pipe.Read(buf, 4), llvm::HasValue(4));
  ASSERT_THAT_EXPECTED(pipe.Read(buf + 4, sizeof(buf) - 4),
                       llvm::HasValue(hello_world.size() - 4));
  EXPECT_EQ(llvm::StringRef(buf, hello_world.size()), hello_world);

  // A blocking read should wait until the data arrives.
  memset(buf, 0, sizeof(buf));
  std::future<llvm::Expected<size_t>> future_num_bytes = std::async(
      std::launch::async, [&] { return pipe.Read(buf, sizeof(buf)); });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_THAT_EXPECTED(pipe.Write(hello_world.data(), hello_world.size()),
                       llvm::HasValue(hello_world.size()));
  ASSERT_THAT_EXPECTED(future_num_bytes.get(),
                       llvm::HasValue(hello_world.size()));
  EXPECT_EQ(llvm::StringRef(buf, hello_world.size()), hello_world);
}
#endif /*ifndef _WIN32*/
