//===- llvm/unittest/Support/FileOutputBuffer.cpp - unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PathV2.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x) \
  if (error_code ASSERT_NO_ERROR_ec = x) { \
    errs() << #x ": did not return errc::success.\n" \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n" \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n"; \
  } else {}

namespace {


// NOTE: Temporarily run this test on unix only.  Once the file mapping
// routines are ported to Windows, this conditional can be removed.
#if LLVM_ON_UNIX


TEST(FileOutputBuffer, Test) {
  // Create unique temporary directory for these tests
  SmallString<128> TestDirectory;
  {
    int fd;
    ASSERT_NO_ERROR(
      fs::unique_file("FileOutputBuffer-test-%%-%%-%%-%%/dir", fd,
                      TestDirectory));
    ::close(fd);
    TestDirectory = path::parent_path(TestDirectory);
  }
     
  // TEST 1: Verify commit case.
  SmallString<128> File1(TestDirectory);
	File1.append("/file1");
  {
    OwningPtr<FileOutputBuffer> Buffer;
    ASSERT_NO_ERROR(FileOutputBuffer::create(File1, 8192, Buffer));
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Write to end of buffer to verify it is writable.
    memcpy(Buffer->getBufferEnd() - 20, "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer.
    ASSERT_NO_ERROR(Buffer->commit());
  }
  // Verify file exists and starts with special header.
  bool MagicMatches = false;
  ASSERT_NO_ERROR(fs::has_magic(Twine(File1), Twine("AABBCCDDEEFFGGHHIIJJ"), 
                                                                MagicMatches));
  EXPECT_TRUE(MagicMatches);
  // Verify file is correct size.
  uint64_t File1Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File1), File1Size));
  ASSERT_EQ(File1Size, 8192ULL);

 	// TEST 2: Verify abort case.
  SmallString<128> File2(TestDirectory);
	File2.append("/file2");
  {
    OwningPtr<FileOutputBuffer> Buffer2;
    ASSERT_NO_ERROR(FileOutputBuffer::create(File2, 8192, Buffer2));
    // Fill buffer with special header.
    memcpy(Buffer2->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Do *not* commit buffer.
  }
  // Verify file does not exist (because buffer not commited).
  bool Exists = false;
  ASSERT_NO_ERROR(fs::exists(Twine(File2), Exists));
  EXPECT_FALSE(Exists);  


  // TEST 3: Verify sizing down case.
  SmallString<128> File3(TestDirectory);
	File3.append("/file3");
  {
    OwningPtr<FileOutputBuffer> Buffer;
    ASSERT_NO_ERROR(FileOutputBuffer::create(File3, 8192000, Buffer));
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Write to end of buffer to verify it is writable.
    memcpy(Buffer->getBufferEnd() - 20, "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer, but size down to smaller size
    ASSERT_NO_ERROR(Buffer->commit(5000));
  }
  // Verify file exists and starts with special header.
  bool MagicMatches3 = false;
  ASSERT_NO_ERROR(fs::has_magic(Twine(File3), Twine("AABBCCDDEEFFGGHHIIJJ"), 
                                                              MagicMatches3));
  EXPECT_TRUE(MagicMatches3);
  // Verify file is correct size.
  uint64_t File3Size;
  ASSERT_NO_ERROR(fs::file_size(Twine(File3), File3Size));
  ASSERT_EQ(File3Size, 5000ULL);


  // TEST 4: Verify file can be made executable.
  SmallString<128> File4(TestDirectory);
	File4.append("/file4");
  {
    OwningPtr<FileOutputBuffer> Buffer;
    ASSERT_NO_ERROR(FileOutputBuffer::create(File4, 8192, Buffer, 
                                              FileOutputBuffer::F_executable));
    // Start buffer with special header.
    memcpy(Buffer->getBufferStart(), "AABBCCDDEEFFGGHHIIJJ", 20);
    // Commit buffer.
    ASSERT_NO_ERROR(Buffer->commit());
  }
  // Verify file exists and is executable.
  fs::file_status Status;
  ASSERT_NO_ERROR(fs::status(Twine(File4), Status));
  bool IsExecutable = (Status.permissions() & fs::owner_exe);
  EXPECT_TRUE(IsExecutable);

  // Clean up.
  uint32_t RemovedCount;
  ASSERT_NO_ERROR(fs::remove_all(TestDirectory.str(), RemovedCount));
}

#endif // LLVM_ON_UNIX

} // anonymous namespace
