//===- llvm/unittest/Support/ReplaceFileTest.cpp - unit tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x)                                                 \
  do {                                                                     \
    if (std::error_code ASSERT_NO_ERROR_ec = x) {                          \
      errs() << #x ": did not return errc::success.\n"                     \
             << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"     \
             << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n"; \
    }                                                                      \
  } while (false)

namespace {
std::error_code CreateFileWithContent(const SmallString<128> &FilePath,
                                      const StringRef &content) {
  int FD = 0;
  if (std::error_code ec = fs::openFileForWrite(FilePath, FD))
    return ec;

  const bool ShouldClose = true;
  raw_fd_ostream OS(FD, ShouldClose);
  OS << content;

  return std::error_code();
}

class ScopedFD {
  int FD;

  ScopedFD(const ScopedFD &) = delete;
  ScopedFD &operator=(const ScopedFD &) = delete;

 public:
  explicit ScopedFD(int Descriptor) : FD(Descriptor) {}
  ~ScopedFD() { Process::SafelyCloseFileDescriptor(FD); }
};

bool FDHasContent(int FD, StringRef Content) {
  auto Buffer = MemoryBuffer::getOpenFile(FD, "", -1);
  assert(Buffer);
  return Buffer.get()->getBuffer() == Content;
}

bool FileHasContent(StringRef File, StringRef Content) {
  int FD = 0;
  auto EC = fs::openFileForRead(File, FD);
  (void)EC;
  assert(!EC);
  ScopedFD EventuallyCloseIt(FD);
  return FDHasContent(FD, Content);
}

TEST(rename, FileOpenedForReadingCanBeReplaced) {
  // Create unique temporary directory for this test.
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(fs::createUniqueDirectory(
      "FileOpenedForReadingCanBeReplaced-test", TestDirectory));

  // Add a couple of files to the test directory.
  SmallString<128> SourceFileName(TestDirectory);
  path::append(SourceFileName, "source");

  SmallString<128> TargetFileName(TestDirectory);
  path::append(TargetFileName, "target");

  ASSERT_NO_ERROR(CreateFileWithContent(SourceFileName, "!!source!!"));
  ASSERT_NO_ERROR(CreateFileWithContent(TargetFileName, "!!target!!"));

  {
    // Open the target file for reading.
    int ReadFD = 0;
    ASSERT_NO_ERROR(fs::openFileForRead(TargetFileName, ReadFD));
    ScopedFD EventuallyCloseIt(ReadFD);

    // Confirm we can replace the file while it is open.
    EXPECT_TRUE(!fs::rename(SourceFileName, TargetFileName));

    // We should still be able to read the old data through the existing
    // descriptor.
    EXPECT_TRUE(FDHasContent(ReadFD, "!!target!!"));

    // The source file should no longer exist
    EXPECT_FALSE(fs::exists(SourceFileName));
  }

  // If we obtain a new descriptor for the target file, we should find that it
  // contains the content that was in the source file.
  EXPECT_TRUE(FileHasContent(TargetFileName, "!!source!!"));

  // Rename the target file back to the source file name to confirm that rename
  // still works if the destination does not already exist.
  EXPECT_TRUE(!fs::rename(TargetFileName, SourceFileName));
  EXPECT_FALSE(fs::exists(TargetFileName));
  ASSERT_TRUE(fs::exists(SourceFileName));

  // Clean up.
  ASSERT_NO_ERROR(fs::remove(SourceFileName));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}

TEST(rename, ExistingTemp) {
  // Test that existing .tmpN files don't get deleted by the Windows
  // sys::fs::rename implementation.
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(
      fs::createUniqueDirectory("ExistingTemp-test", TestDirectory));

  SmallString<128> SourceFileName(TestDirectory);
  path::append(SourceFileName, "source");

  SmallString<128> TargetFileName(TestDirectory);
  path::append(TargetFileName, "target");

  SmallString<128> TargetTmp0FileName(TestDirectory);
  path::append(TargetTmp0FileName, "target.tmp0");

  SmallString<128> TargetTmp1FileName(TestDirectory);
  path::append(TargetTmp1FileName, "target.tmp1");

  ASSERT_NO_ERROR(CreateFileWithContent(SourceFileName, "!!source!!"));
  ASSERT_NO_ERROR(CreateFileWithContent(TargetFileName, "!!target!!"));
  ASSERT_NO_ERROR(CreateFileWithContent(TargetTmp0FileName, "!!target.tmp0!!"));

  {
    // Use mapped_file_region to make sure that the destination file is mmap'ed.
    // This will cause SetInformationByHandle to fail when renaming to the
    // destination, and we will follow the code path that tries to give target
    // a temporary name.
    int TargetFD;
    std::error_code EC;
    ASSERT_NO_ERROR(fs::openFileForRead(TargetFileName, TargetFD));
    ScopedFD X(TargetFD);
    sys::fs::mapped_file_region MFR(
        TargetFD, sys::fs::mapped_file_region::readonly, 10, 0, EC);
    ASSERT_FALSE(EC);

    ASSERT_NO_ERROR(fs::rename(SourceFileName, TargetFileName));

#ifdef _WIN32
    // Make sure that target was temporarily renamed to target.tmp1 on Windows.
    // This is signified by a permission denied error as opposed to no such file
    // or directory when trying to open it.
    int Tmp1FD;
    EXPECT_EQ(errc::permission_denied,
              fs::openFileForRead(TargetTmp1FileName, Tmp1FD));
#endif
  }

  EXPECT_TRUE(FileHasContent(TargetTmp0FileName, "!!target.tmp0!!"));

  ASSERT_NO_ERROR(fs::remove(TargetFileName));
  ASSERT_NO_ERROR(fs::remove(TargetTmp0FileName));
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}

}  // anonymous namespace
