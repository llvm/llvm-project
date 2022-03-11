//===- llvm/unittest/Support/MemoryBufferTest.cpp - MemoryBuffer tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MemoryBuffer support class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#if LLVM_ENABLE_THREADS
#include <thread>
#endif
#if LLVM_ON_UNIX
#include <unistd.h>
#endif
#if _WIN32
#include <windows.h>
#endif

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

#define ASSERT_ERROR(x)                                                        \
  if (!x) {                                                                    \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return a failure error code.\n";                  \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  }

namespace {

class MemoryBufferTest : public testing::Test {
protected:
  MemoryBufferTest()
  : data("this is some data")
  { }

  void SetUp() override {}

  /// Common testing for different modes of getOpenFileSlice.
  /// Creates a temporary file with known contents, and uses
  /// MemoryBuffer::getOpenFileSlice to map it.
  /// If \p Reopen is true, the file is closed after creating and reopened
  /// anew before using MemoryBuffer.
  void testGetOpenFileSlice(bool Reopen);

  typedef std::unique_ptr<MemoryBuffer> OwningBuffer;

  std::string data;
};

TEST_F(MemoryBufferTest, get) {
  // Default name and null-terminator flag
  OwningBuffer MB1(MemoryBuffer::getMemBuffer(data));
  EXPECT_NE(nullptr, MB1.get());

  // RequiresNullTerminator = false
  OwningBuffer MB2(MemoryBuffer::getMemBuffer(data, "one", false));
  EXPECT_NE(nullptr, MB2.get());

  // RequiresNullTerminator = true
  OwningBuffer MB3(MemoryBuffer::getMemBuffer(data, "two", true));
  EXPECT_NE(nullptr, MB3.get());

  // verify all 3 buffers point to the same address
  EXPECT_EQ(MB1->getBufferStart(), MB2->getBufferStart());
  EXPECT_EQ(MB2->getBufferStart(), MB3->getBufferStart());

  // verify the original data is unmodified after deleting the buffers
  MB1.reset();
  MB2.reset();
  MB3.reset();
  EXPECT_EQ("this is some data", data);
}

TEST_F(MemoryBufferTest, getOpenFile) {
  int FD;
  SmallString<64> TestPath;
  ASSERT_EQ(sys::fs::createTemporaryFile("MemoryBufferTest_getOpenFile", "temp",
                                         FD, TestPath),
            std::error_code());

  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(FD, /*shouldClose*/ true);
  OF << "12345678";
  OF.close();

  {
    Expected<sys::fs::file_t> File = sys::fs::openNativeFileForRead(TestPath);
    ASSERT_THAT_EXPECTED(File, Succeeded());
    auto OnExit =
        make_scope_exit([&] { ASSERT_NO_ERROR(sys::fs::closeFile(*File)); });
    ErrorOr<OwningBuffer> MB = MemoryBuffer::getOpenFile(*File, TestPath, 6);
    ASSERT_NO_ERROR(MB.getError());
    EXPECT_EQ("123456", MB.get()->getBuffer());
  }
  {
    Expected<sys::fs::file_t> File = sys::fs::openNativeFileForWrite(
        TestPath, sys::fs::CD_OpenExisting, sys::fs::OF_None);
    ASSERT_THAT_EXPECTED(File, Succeeded());
    auto OnExit =
        make_scope_exit([&] { ASSERT_NO_ERROR(sys::fs::closeFile(*File)); });
    ASSERT_ERROR(MemoryBuffer::getOpenFile(*File, TestPath, 6).getError());
  }
}

TEST_F(MemoryBufferTest, NullTerminator4K) {
  // Test that a file with size that is a multiple of the page size can be null
  // terminated correctly by MemoryBuffer.
  int TestFD;
  SmallString<64> TestPath;
  sys::fs::createTemporaryFile("MemoryBufferTest_NullTerminator4K", "temp",
                               TestFD, TestPath);
  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(TestFD, true, /*unbuffered=*/true);
  for (unsigned i = 0; i < 4096 / 16; ++i) {
    OF << "0123456789abcdef";
  }
  OF.close();

  ErrorOr<OwningBuffer> MB = MemoryBuffer::getFile(TestPath.c_str());
  std::error_code EC = MB.getError();
  ASSERT_FALSE(EC);

  const char *BufData = MB.get()->getBufferStart();
  EXPECT_EQ('f', BufData[4095]);
  EXPECT_EQ('\0', BufData[4096]);
}

TEST_F(MemoryBufferTest, copy) {
  // copy with no name
  OwningBuffer MBC1(MemoryBuffer::getMemBufferCopy(data));
  EXPECT_NE(nullptr, MBC1.get());

  // copy with a name
  OwningBuffer MBC2(MemoryBuffer::getMemBufferCopy(data, "copy"));
  EXPECT_NE(nullptr, MBC2.get());

  // verify the two copies do not point to the same place
  EXPECT_NE(MBC1->getBufferStart(), MBC2->getBufferStart());
}

#if LLVM_ENABLE_THREADS
TEST_F(MemoryBufferTest, createFromPipe) {
  sys::fs::file_t pipes[2];
#if LLVM_ON_UNIX
  ASSERT_EQ(::pipe(pipes), 0) << strerror(errno);
#else
  ASSERT_TRUE(::CreatePipe(&pipes[0], &pipes[1], nullptr, 0))
      << ::GetLastError();
#endif
  auto ReadCloser = make_scope_exit([&] { sys::fs::closeFile(pipes[0]); });
  std::thread Writer([&] {
    auto WriteCloser = make_scope_exit([&] { sys::fs::closeFile(pipes[1]); });
    for (unsigned i = 0; i < 5; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
#if LLVM_ON_UNIX
      ASSERT_EQ(::write(pipes[1], "foo", 3), 3) << strerror(errno);
#else
      DWORD Written;
      ASSERT_TRUE(::WriteFile(pipes[1], "foo", 3, &Written, nullptr))
          << ::GetLastError();
      ASSERT_EQ(Written, 3u);
#endif
    }
  });
  ErrorOr<OwningBuffer> MB =
      MemoryBuffer::getOpenFile(pipes[0], "pipe", /*FileSize*/ -1);
  Writer.join();
  ASSERT_NO_ERROR(MB.getError());
  EXPECT_EQ(MB.get()->getBuffer(), "foofoofoofoofoo");
}
#endif

TEST_F(MemoryBufferTest, make_new) {
  // 0-sized buffer
  OwningBuffer Zero(WritableMemoryBuffer::getNewUninitMemBuffer(0));
  EXPECT_NE(nullptr, Zero.get());

  // uninitialized buffer with no name
  OwningBuffer One(WritableMemoryBuffer::getNewUninitMemBuffer(321));
  EXPECT_NE(nullptr, One.get());

  // uninitialized buffer with name
  OwningBuffer Two(WritableMemoryBuffer::getNewUninitMemBuffer(123, "bla"));
  EXPECT_NE(nullptr, Two.get());

  // 0-initialized buffer with no name
  OwningBuffer Three(WritableMemoryBuffer::getNewMemBuffer(321, data));
  EXPECT_NE(nullptr, Three.get());
  for (size_t i = 0; i < 321; ++i)
    EXPECT_EQ(0, Three->getBufferStart()[0]);

  // 0-initialized buffer with name
  OwningBuffer Four(WritableMemoryBuffer::getNewMemBuffer(123, "zeros"));
  EXPECT_NE(nullptr, Four.get());
  for (size_t i = 0; i < 123; ++i)
    EXPECT_EQ(0, Four->getBufferStart()[0]);

  // uninitialized buffer with rollover size
  OwningBuffer Five(
      WritableMemoryBuffer::getNewUninitMemBuffer(SIZE_MAX, "huge"));
  EXPECT_EQ(nullptr, Five.get());
}

void MemoryBufferTest::testGetOpenFileSlice(bool Reopen) {
  // Test that MemoryBuffer::getOpenFile works properly when no null
  // terminator is requested and the size is large enough to trigger
  // the usage of memory mapping.
  int TestFD;
  SmallString<64> TestPath;
  // Create a temporary file and write data into it.
  sys::fs::createTemporaryFile("prefix", "temp", TestFD, TestPath);
  FileRemover Cleanup(TestPath);
  // OF is responsible for closing the file; If the file is not
  // reopened, it will be unbuffered so that the results are
  // immediately visible through the fd.
  raw_fd_ostream OF(TestFD, true, !Reopen);
  for (int i = 0; i < 60000; ++i) {
    OF << "0123456789";
  }

  if (Reopen) {
    OF.close();
    EXPECT_FALSE(sys::fs::openFileForRead(TestPath.c_str(), TestFD));
  }

  ErrorOr<OwningBuffer> Buf = MemoryBuffer::getOpenFileSlice(
      sys::fs::convertFDToNativeFile(TestFD), TestPath.c_str(),
      40000, // Size
      80000  // Offset
  );

  std::error_code EC = Buf.getError();
  EXPECT_FALSE(EC);

  StringRef BufData = Buf.get()->getBuffer();
  EXPECT_EQ(BufData.size(), 40000U);
  EXPECT_EQ(BufData[0], '0');
  EXPECT_EQ(BufData[9], '9');
}

TEST_F(MemoryBufferTest, getOpenFileNoReopen) {
  testGetOpenFileSlice(false);
}

TEST_F(MemoryBufferTest, getOpenFileReopened) {
  testGetOpenFileSlice(true);
}

TEST_F(MemoryBufferTest, slice) {
  // Create a file that is six pages long with different data on each page.
  int FD;
  SmallString<64> TestPath;
  sys::fs::createTemporaryFile("MemoryBufferTest_Slice", "temp", FD, TestPath);
  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(FD, true, /*unbuffered=*/true);
  for (unsigned i = 0; i < 0x2000 / 8; ++i) {
    OF << "12345678";
  }
  for (unsigned i = 0; i < 0x2000 / 8; ++i) {
    OF << "abcdefgh";
  }
  for (unsigned i = 0; i < 0x2000 / 8; ++i) {
    OF << "ABCDEFGH";
  }
  OF.close();

  // Try offset of one page.
  ErrorOr<OwningBuffer> MB = MemoryBuffer::getFileSlice(TestPath.str(),
                                                        0x4000, 0x1000);
  std::error_code EC = MB.getError();
  ASSERT_FALSE(EC);
  EXPECT_EQ(0x4000UL, MB.get()->getBufferSize());
 
  StringRef BufData = MB.get()->getBuffer();
  EXPECT_TRUE(BufData.substr(0x0000,8).equals("12345678"));
  EXPECT_TRUE(BufData.substr(0x0FF8,8).equals("12345678"));
  EXPECT_TRUE(BufData.substr(0x1000,8).equals("abcdefgh"));
  EXPECT_TRUE(BufData.substr(0x2FF8,8).equals("abcdefgh"));
  EXPECT_TRUE(BufData.substr(0x3000,8).equals("ABCDEFGH"));
  EXPECT_TRUE(BufData.substr(0x3FF8,8).equals("ABCDEFGH"));
   
  // Try non-page aligned.
  ErrorOr<OwningBuffer> MB2 = MemoryBuffer::getFileSlice(TestPath.str(),
                                                         0x3000, 0x0800);
  EC = MB2.getError();
  ASSERT_FALSE(EC);
  EXPECT_EQ(0x3000UL, MB2.get()->getBufferSize());
  
  StringRef BufData2 = MB2.get()->getBuffer();
  EXPECT_TRUE(BufData2.substr(0x0000,8).equals("12345678"));
  EXPECT_TRUE(BufData2.substr(0x17F8,8).equals("12345678"));
  EXPECT_TRUE(BufData2.substr(0x1800,8).equals("abcdefgh"));
  EXPECT_TRUE(BufData2.substr(0x2FF8,8).equals("abcdefgh"));
}

TEST_F(MemoryBufferTest, writableSlice) {
  // Create a file initialized with some data
  int FD;
  SmallString<64> TestPath;
  sys::fs::createTemporaryFile("MemoryBufferTest_WritableSlice", "temp", FD,
                               TestPath);
  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(FD, true);
  for (unsigned i = 0; i < 0x1000; ++i)
    OF << "0123456789abcdef";
  OF.close();

  {
    auto MBOrError =
        WritableMemoryBuffer::getFileSlice(TestPath.str(), 0x6000, 0x2000);
    ASSERT_FALSE(MBOrError.getError());
    // Write some data.  It should be mapped private, so that upon completion
    // the original file contents are not modified.
    WritableMemoryBuffer &MB = **MBOrError;
    ASSERT_EQ(0x6000u, MB.getBufferSize());
    char *Start = MB.getBufferStart();
    ASSERT_EQ(MB.getBufferEnd(), MB.getBufferStart() + MB.getBufferSize());
    ::memset(Start, 'x', MB.getBufferSize());
  }

  auto MBOrError = MemoryBuffer::getFile(TestPath);
  ASSERT_FALSE(MBOrError.getError());
  auto &MB = **MBOrError;
  ASSERT_EQ(0x10000u, MB.getBufferSize());
  for (size_t i = 0; i < MB.getBufferSize(); i += 0x10)
    EXPECT_EQ("0123456789abcdef", MB.getBuffer().substr(i, 0x10)) << "i: " << i;
}

TEST_F(MemoryBufferTest, writeThroughFile) {
  // Create a file initialized with some data
  int FD;
  SmallString<64> TestPath;
  sys::fs::createTemporaryFile("MemoryBufferTest_WriteThrough", "temp", FD,
                               TestPath);
  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(FD, true);
  OF << "0123456789abcdef";
  OF.close();
  {
    auto MBOrError = WriteThroughMemoryBuffer::getFile(TestPath);
    ASSERT_FALSE(MBOrError.getError());
    // Write some data.  It should be mapped readwrite, so that upon completion
    // the original file contents are modified.
    WriteThroughMemoryBuffer &MB = **MBOrError;
    ASSERT_EQ(16u, MB.getBufferSize());
    char *Start = MB.getBufferStart();
    ASSERT_EQ(MB.getBufferEnd(), MB.getBufferStart() + MB.getBufferSize());
    ::memset(Start, 'x', MB.getBufferSize());
  }

  auto MBOrError = MemoryBuffer::getFile(TestPath);
  ASSERT_FALSE(MBOrError.getError());
  auto &MB = **MBOrError;
  ASSERT_EQ(16u, MB.getBufferSize());
  EXPECT_EQ("xxxxxxxxxxxxxxxx", MB.getBuffer());
}

TEST_F(MemoryBufferTest, mmapVolatileNoNull) {
  // Verify that `MemoryBuffer::getOpenFile` will use mmap when
  // `RequiresNullTerminator = false`, `IsVolatile = true`, and the file is
  // large enough to use mmap.
  //
  // This is done because Clang should use this mode to open module files, and
  // falling back to malloc for them causes a huge memory usage increase.

  int FD;
  SmallString<64> TestPath;
  ASSERT_NO_ERROR(sys::fs::createTemporaryFile(
      "MemoryBufferTest_mmapVolatileNoNull", "temp", FD, TestPath));
  FileRemover Cleanup(TestPath);
  raw_fd_ostream OF(FD, true);
  // Create a file large enough to mmap. 4 pages should be enough.
  unsigned PageSize = sys::Process::getPageSizeEstimate();
  unsigned FileWrites = (PageSize * 4) / 8;
  for (unsigned i = 0; i < FileWrites; ++i)
    OF << "01234567";
  OF.close();

  Expected<sys::fs::file_t> File = sys::fs::openNativeFileForRead(TestPath);
  ASSERT_THAT_EXPECTED(File, Succeeded());
  auto OnExit =
      make_scope_exit([&] { ASSERT_NO_ERROR(sys::fs::closeFile(*File)); });

  auto MBOrError = MemoryBuffer::getOpenFile(*File, TestPath,
      /*FileSize=*/-1, /*RequiresNullTerminator=*/false, /*IsVolatile=*/true);
  ASSERT_NO_ERROR(MBOrError.getError())
  OwningBuffer MB = std::move(*MBOrError);
  EXPECT_EQ(MB->getBufferKind(), MemoryBuffer::MemoryBuffer_MMap);
  EXPECT_EQ(MB->getBufferSize(), std::size_t(FileWrites * 8));
  EXPECT_TRUE(MB->getBuffer().startswith("01234567"));
}

// Test that SmallVector without a null terminator gets one.
TEST(SmallVectorMemoryBufferTest, WithoutNullTerminatorRequiresNullTerminator) {
  SmallString<0> Data("some data");

  SmallVectorMemoryBuffer MB(std::move(Data),
                             /*RequiresNullTerminator=*/true);
  EXPECT_EQ(MB.getBufferSize(), 9u);
  EXPECT_EQ(MB.getBufferEnd()[0], '\0');
}

// Test that SmallVector with a null terminator keeps it.
TEST(SmallVectorMemoryBufferTest, WithNullTerminatorRequiresNullTerminator) {
  SmallString<0> Data("some data");
  Data.push_back('\0');
  Data.pop_back();

  SmallVectorMemoryBuffer MB(std::move(Data),
                             /*RequiresNullTerminator=*/true);
  EXPECT_EQ(MB.getBufferSize(), 9u);
  EXPECT_EQ(MB.getBufferEnd()[0], '\0');
}

} // namespace
