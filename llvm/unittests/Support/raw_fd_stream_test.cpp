//===- llvm/unittest/Support/raw_fd_stream_test.cpp - raw_fd_stream tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// While these are marked as "internal" APIs, they seem to work and be pretty
// widely used for their exact documented behavior.
using ::testing::internal::CaptureStderr;
using ::testing::internal::CaptureStdout;
using ::testing::internal::GetCapturedStderr;
using ::testing::internal::GetCapturedStdout;

TEST(raw_fd_streamTest, ReadAfterWrite) {
  SmallString<64> Path;
  int FD;
  ASSERT_FALSE(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
  FileRemover Cleanup(Path);
  std::error_code EC;
  raw_fd_stream OS(Path, EC);
  EXPECT_TRUE(!EC);

  char Bytes[8];

  OS.write("01234567", 8);

  OS.seek(3);
  EXPECT_EQ(OS.read(Bytes, 2), 2);
  EXPECT_EQ(Bytes[0], '3');
  EXPECT_EQ(Bytes[1], '4');

  OS.seek(4);
  OS.write("xyz", 3);

  OS.seek(0);
  EXPECT_EQ(OS.read(Bytes, 8), 8);
  EXPECT_EQ(Bytes[0], '0');
  EXPECT_EQ(Bytes[1], '1');
  EXPECT_EQ(Bytes[2], '2');
  EXPECT_EQ(Bytes[3], '3');
  EXPECT_EQ(Bytes[4], 'x');
  EXPECT_EQ(Bytes[5], 'y');
  EXPECT_EQ(Bytes[6], 'z');
  EXPECT_EQ(Bytes[7], '7');
}

TEST(raw_fd_streamTest, DynCast) {
  {
    std::error_code EC;
    raw_fd_stream OS("-", EC);
    EXPECT_TRUE(dyn_cast<raw_fd_stream>(&OS));
  }
  {
    std::error_code EC;
    raw_fd_ostream OS("-", EC);
    EXPECT_FALSE(dyn_cast<raw_fd_stream>(&OS));
  }
}

TEST(raw_fd_streamTest, OverrideOutsAndErrs) {
  SmallString<64> Path;
  int FD;
  ASSERT_FALSE(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
  FileRemover Cleanup(Path);
  std::error_code EC;
  raw_fd_stream OS(Path, EC);
  ASSERT_TRUE(!EC);

  ScopedOutsAndErrsOverride Overrides(&OS, &OS);

  // First test `outs`.
  llvm::outs() << "outs";
  llvm::outs().flush();
  char Buffer[4];
  OS.seek(0);
  OS.read(Buffer, sizeof(Buffer));
  EXPECT_EQ("outs", StringRef(Buffer, sizeof(Buffer)));

  // Now test `errs`.
  OS.seek(0);
  llvm::errs() << "errs";
  llvm::errs().flush();
  OS.seek(0);
  OS.read(Buffer, sizeof(Buffer));
  EXPECT_EQ("errs", StringRef(Buffer, sizeof(Buffer)));

  // Seek back before overrides to improve the restore test.
  OS.seek(0);

  // Now try nesting a new set of redirects.
  {
    SmallString<64> Path2;
    int FD2;
    ASSERT_FALSE(sys::fs::createTemporaryFile("foo2", "bar2", FD2, Path2));
    FileRemover Cleanup(Path2);
    raw_fd_stream OS2(Path, EC);
    ASSERT_TRUE(!EC);

    ScopedOutsAndErrsOverride Overrides(&OS2, &OS2);

    llvm::outs() << "te";
    llvm::outs().flush();
    llvm::errs() << "st";
    llvm::errs().flush();
    OS2.seek(0);
    OS2.read(Buffer, sizeof(Buffer));
    EXPECT_EQ("test", StringRef(Buffer, sizeof(Buffer)));
  }

  // Nest and un-override to ensure we can reach stdout and stderr again.
  {
    ScopedOutsAndErrsOverride Overrides(nullptr, nullptr);

    CaptureStdout();
    CaptureStderr();
    llvm::outs() << "llvm";
    llvm::outs().flush();
    llvm::errs() << "clang";
    llvm::errs().flush();
    std::string OutResult = GetCapturedStdout();
    std::string ErrResult = GetCapturedStderr();

    EXPECT_EQ("llvm", OutResult);
    EXPECT_EQ("clang", ErrResult);
  }

  // Make sure the prior override is restored.
  llvm::outs() << "sw";
  llvm::outs().flush();
  llvm::errs() << "im";
  llvm::errs().flush();
  OS.seek(0);
  OS.read(Buffer, sizeof(Buffer));
  EXPECT_EQ("swim", StringRef(Buffer, sizeof(Buffer)));

  // Last but not least, make sure our overrides are propagated into the crash
  // recovery context.
  OS.seek(0);
  CrashRecoveryContext CRC;

  ASSERT_TRUE(CRC.RunSafelyOnThread([] {
    llvm::outs() << "bo";
    llvm::outs().flush();
    llvm::errs() << "om";
    llvm::errs().flush();
  }));
  OS.seek(0);
  OS.read(Buffer, sizeof(Buffer));
  EXPECT_EQ("boom", StringRef(Buffer, sizeof(Buffer)));
}

} // namespace
