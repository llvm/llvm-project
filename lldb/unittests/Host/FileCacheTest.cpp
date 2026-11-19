//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileCache.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/Pipe.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

class FileCacheTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

// Named pipes are files only on POSIX.
#ifndef _WIN32
TEST_F(FileCacheTest, WriteToPipe) {
  Pipe pipe;
  llvm::SmallString<0> name;
  ASSERT_THAT_ERROR(
      pipe.CreateWithUniqueName("FileCacheTest-WriteToPipe", name).ToError(),
      llvm::Succeeded());
  llvm::scope_exit delete_pipe([&] { pipe.Delete(name); });
  // With the read end open, opening the write end does not block.
  ASSERT_THAT_ERROR(pipe.OpenAsReader(name).ToError(), llvm::Succeeded());

  FileCache &cache = FileCache::GetInstance();
  Status error;
  lldb::user_id_t fd =
      cache.OpenFile(FileSpec(name), File::eOpenOptionWriteOnly, 0, error);
  ASSERT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  // A pipe cannot seek, so the bytes go to its end whatever the offset.
  EXPECT_EQ(cache.WriteFile(fd, /*offset=*/1, "abc", 3, error), 3u);
  EXPECT_THAT_ERROR(error.ToError(), llvm::Succeeded());
  EXPECT_TRUE(cache.CloseFile(fd, error));

  char buffer[3];
  ASSERT_THAT_EXPECTED(pipe.Read(buffer, sizeof(buffer)), llvm::HasValue(3));
  EXPECT_EQ(llvm::StringRef(buffer, sizeof(buffer)), "abc");
}
#endif
