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
#include "gtest/gtest.h"

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
  // Note OpenAsReader() do nothing on Windows, the pipe is already opened for
  // read and write.
  ASSERT_THAT_ERROR(
      pipe.OpenAsReader(name_ref, /*child_process_inherit=*/false).ToError(),
      llvm::Succeeded());

  ASSERT_TRUE(pipe.CanRead());
}

TEST_F(PipeTest, WriteWithTimeout) {
  Pipe pipe;
  ASSERT_THAT_ERROR(pipe.CreateNew(false).ToError(), llvm::Succeeded());
  // Note write_chunk_size must be less than the pipe buffer.
  // The pipe buffer is 1024 for PipeWindows and 4096 for PipePosix.
  const size_t buf_size = 8192;
  const size_t write_chunk_size = 256;
  const size_t read_chunk_size = 300;
  std::unique_ptr<int32_t[]> write_buf_ptr(
      new int32_t[buf_size / sizeof(int32_t)]);
  int32_t *write_buf = write_buf_ptr.get();
  std::unique_ptr<int32_t[]> read_buf_ptr(
      new int32_t[(buf_size + 100) / sizeof(int32_t)]);
  int32_t *read_buf = read_buf_ptr.get();
  for (int i = 0; i < buf_size / sizeof(int32_t); ++i) {
    write_buf[i] = i;
    read_buf[i] = -i;
  }

  char *write_ptr = (char *)write_buf;
  size_t write_bytes = 0;
  char *read_ptr = (char *)read_buf;
  size_t read_bytes = 0;
  size_t num_bytes = 0;
  Status error;
  while (write_bytes < buf_size) {
    error = pipe.WriteWithTimeout(write_ptr + write_bytes, write_chunk_size,
                                  std::chrono::milliseconds(10), num_bytes);
    if (error.Fail()) {
      ASSERT_TRUE(read_bytes < buf_size);
      error = pipe.ReadWithTimeout(read_ptr + read_bytes, read_chunk_size,
                                   std::chrono::milliseconds(10), num_bytes);
      if (error.Fail())
        FAIL();
      else
        read_bytes += num_bytes;
    } else
      write_bytes += num_bytes;
  }
  // Read the rest data.
  while (read_bytes < buf_size) {
    error = pipe.ReadWithTimeout(read_ptr + read_bytes, buf_size - read_bytes,
                                 std::chrono::milliseconds(10), num_bytes);
    if (error.Fail())
      FAIL();
    else
      read_bytes += num_bytes;
  }

  // Be sure the pipe is empty.
  error = pipe.ReadWithTimeout(read_ptr + read_bytes, 100,
                               std::chrono::milliseconds(10), num_bytes);
  ASSERT_TRUE(error.Fail());

  // Compare the data.
  ASSERT_EQ(write_bytes, read_bytes);
  ASSERT_EQ(memcmp(write_buf, read_buf, buf_size), 0);
}
