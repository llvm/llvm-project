//===- llvm/unittest/Support/FileOutputBuffer.cpp - unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

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
TEST(FileOutputBuffer, Test) {
  // Create unique temporary directory for these tests
  SmallString<128> TestDirectory;
  {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("FileOutputBuffer-test", TestDirectory));
  }

  // TEST 1: Verify commit case.
  SmallString<128> File1(TestDirectory);
  File1.append("/file1");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
        FileOutputBuffer::create(File1, 8192);
    ASSERT_NO_ERROR(errorToErrorCode(BufferOrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> &Buffer = *BufferOrErr;
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Write to end of buffer to verify it is writable.
    memcpy(Buffer->getBufferEnd() - 20, "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer.
    ASSERT_NO_ERROR(errorToErrorCode(Buffer->commit()));
  }

  // Verify file is correct size.
  uint64_t File1Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File1), File1Size));
  ASSERT_EQ(File1Size, 8192ULL);
  ASSERT_NO_ERROR(fs::remove(File1.str()));

  // TEST 2: Verify abort case.
  SmallString<128> File2(TestDirectory);
  File2.append("/file2");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> Buffer2OrErr =
        FileOutputBuffer::create(File2, 8192);
    ASSERT_NO_ERROR(errorToErrorCode(Buffer2OrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> &Buffer2 = *Buffer2OrErr;
    // Fill buffer with special header.
    memcpy(Buffer2->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Do *not* commit buffer.
  }
  // Verify file does not exist (because buffer not committed).
  ASSERT_EQ(fs::access(Twine(File2), fs::AccessMode::Exist),
            errc::no_such_file_or_directory);
  ASSERT_NO_ERROR(fs::remove(File2.str()));

  // TEST 3: Verify sizing down case.
  SmallString<128> File3(TestDirectory);
  File3.append("/file3");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
        FileOutputBuffer::create(File3, 8192000);
    ASSERT_NO_ERROR(errorToErrorCode(BufferOrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> &Buffer = *BufferOrErr;
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Write to end of buffer to verify it is writable.
    memcpy(Buffer->getBufferEnd() - 20, "AABBCCDDEEFFGGHHIIJJ", 20);
    ASSERT_NO_ERROR(errorToErrorCode(Buffer->commit()));
  }

  // Verify file is correct size.
  uint64_t File3Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File3), File3Size));
  ASSERT_EQ(File3Size, 8192000ULL);
  ASSERT_NO_ERROR(fs::remove(File3.str()));

  // TEST 4: Verify file can be made executable.
  SmallString<128> File4(TestDirectory);
  File4.append("/file4");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
        FileOutputBuffer::create(File4, 8192, FileOutputBuffer::F_executable);
    ASSERT_NO_ERROR(errorToErrorCode(BufferOrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> &Buffer = *BufferOrErr;
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer.
    ASSERT_NO_ERROR(errorToErrorCode(Buffer->commit()));
  }
  // Verify file exists and is executable.
  fs::file_status Status;
  ASSERT_NO_ERROR(fs::status(Twine(File4), Status));
  bool IsExecutable = (Status.permissions() & fs::owner_exe);
  EXPECT_TRUE(IsExecutable);
  ASSERT_NO_ERROR(fs::remove(File4.str()));

  // TEST 5: In-memory buffer works as expected.
  SmallString<128> File5(TestDirectory);
  File5.append("/file5");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
        FileOutputBuffer::create(File5, 8000, FileOutputBuffer::F_no_mmap);
    ASSERT_NO_ERROR(errorToErrorCode(BufferOrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> &Buffer = *BufferOrErr;
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    ASSERT_NO_ERROR(errorToErrorCode(Buffer->commit()));
    // Write to end of buffer to verify it is writable.
    memcpy(Buffer->getBufferEnd() - 20, "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer.
    ASSERT_NO_ERROR(errorToErrorCode(Buffer->commit()));
  }

  // Verify file is correct size.
  uint64_t File5Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File5), File5Size));
  ASSERT_EQ(File5Size, 8000ULL);
  ASSERT_NO_ERROR(fs::remove(File5.str()));

  // TEST 6: Create an empty file.
  SmallString<128> File6(TestDirectory);
  File6.append("/file6");
  {
    Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
        FileOutputBuffer::create(File6, 0);
    ASSERT_NO_ERROR(errorToErrorCode(BufferOrErr.takeError()));
    ASSERT_NO_ERROR(errorToErrorCode((*BufferOrErr)->commit()));
  }
  uint64_t File6Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File6), File6Size));
  ASSERT_EQ(File6Size, 0ULL);
  ASSERT_NO_ERROR(fs::remove(File6.str()));

  // Clean up.
  ASSERT_NO_ERROR(fs::remove(TestDirectory.str()));
}
} // anonymous namespace
