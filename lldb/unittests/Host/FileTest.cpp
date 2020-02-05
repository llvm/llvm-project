//===-- FileTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/File.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(File, GetWaitableHandleFileno) {
  const auto *Info = testing::UnitTest::GetInstance()->current_test_info();

  llvm::SmallString<128> name;
  int fd;
  llvm::sys::fs::createTemporaryFile(llvm::Twine(Info->test_case_name()) + "-" +
                                         Info->name(),
                                     "test", fd, name);
  llvm::FileRemover remover(name);
  ASSERT_GE(fd, 0);

  FILE *stream = fdopen(fd, "r");
  ASSERT_TRUE(stream);

  NativeFile file(stream, true);
  EXPECT_EQ(file.GetWaitableHandle(), fd);
}

TEST(File, GetStreamFromDescriptor) {
  const auto *Info = testing::UnitTest::GetInstance()->current_test_info();
  llvm::SmallString<128> name;
  int fd;
  llvm::sys::fs::createTemporaryFile(llvm::Twine(Info->test_case_name()) + "-" +
                                         Info->name(),
                                     "test", fd, name);

  llvm::FileRemover remover(name);
  ASSERT_GE(fd, 0);

  NativeFile file(fd, File::eOpenOptionWrite, true);
  ASSERT_TRUE(file.IsValid());

  FILE *stream = file.GetStream();
  ASSERT_TRUE(stream != NULL);

  EXPECT_EQ(file.GetDescriptor(), fd);
  EXPECT_EQ(file.GetWaitableHandle(), fd);
}
