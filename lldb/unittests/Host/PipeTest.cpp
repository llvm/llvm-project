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
}
#endif
