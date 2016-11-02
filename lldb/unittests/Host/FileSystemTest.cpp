//===-- FileSystemTest.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/FileSystem.h"

extern const char *TestMainArgv0;

using namespace lldb_private;

TEST(FileSystemTest, FileAndDirectoryComponents) {
  using namespace std::chrono;

  const bool resolve = true;
#ifdef _WIN32
  FileSpec fs1("C:\\FILE\\THAT\\DOES\\NOT\\EXIST.TXT", !resolve);
#else
  FileSpec fs1("/file/that/does/not/exist.txt", !resolve);
#endif
  FileSpec fs2(TestMainArgv0, resolve);

  EXPECT_EQ(system_clock::time_point(), FileSystem::GetModificationTime(fs1));
  EXPECT_LT(system_clock::time_point() + hours(24 * 365 * 20),
            FileSystem::GetModificationTime(fs2));
}
