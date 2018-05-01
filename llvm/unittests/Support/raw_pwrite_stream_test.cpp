//===- raw_pwrite_stream_test.cpp - raw_pwrite_stream tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

namespace {

TEST(raw_pwrite_ostreamTest, TestSVector) {
  SmallVector<char, 0> Buffer;
  raw_svector_ostream OS(Buffer);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(Test, OS.str());

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif
}

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

TEST(raw_pwrite_ostreamTest, TestFD) {
  SmallString<64> Path;
  int FD;

  // If we want to clean up from a death test, we have to remove the file from
  // the parent process. Have the parent create the file, pass it via
  // environment variable to the child, let the child crash, and then remove it
  // in the parent.
  const char *ParentPath = getenv("RAW_PWRITE_TEST_FILE");
  if (ParentPath) {
    Path = ParentPath;
    ASSERT_NO_ERROR(sys::fs::openFileForRead(Path, FD));
  } else {
    ASSERT_NO_ERROR(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
    setenv("RAW_PWRITE_TEST_FILE", Path.c_str(), true);
  }
  FileRemover Cleanup(Path);

  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif
}

#ifdef LLVM_ON_UNIX
TEST(raw_pwrite_ostreamTest, TestDevNull) {
  int FD;
  sys::fs::openFileForWrite("/dev/null", FD, sys::fs::F_None);
  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);
}
#endif
}
