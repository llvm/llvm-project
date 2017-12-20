//===- llvm/unittest/Support/MemoryBufferTest.cpp - MemoryBuffer tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MemoryBuffer support class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

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
  EXPECT_TRUE(nullptr != MB1.get());

  // RequiresNullTerminator = false
  OwningBuffer MB2(MemoryBuffer::getMemBuffer(data, "one", false));
  EXPECT_TRUE(nullptr != MB2.get());

  // RequiresNullTerminator = true
  OwningBuffer MB3(MemoryBuffer::getMemBuffer(data, "two", true));
  EXPECT_TRUE(nullptr != MB3.get());

  // verify all 3 buffers point to the same address
  EXPECT_EQ(MB1->getBufferStart(), MB2->getBufferStart());
  EXPECT_EQ(MB2->getBufferStart(), MB3->getBufferStart());

  // verify the original data is unmodified after deleting the buffers
  MB1.reset();
  MB2.reset();
  MB3.reset();
  EXPECT_EQ("this is some data", data);
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
  EXPECT_TRUE(nullptr != MBC1.get());

  // copy with a name
  OwningBuffer MBC2(MemoryBuffer::getMemBufferCopy(data, "copy"));
  EXPECT_TRUE(nullptr != MBC2.get());

  // verify the two copies do not point to the same place
  EXPECT_NE(MBC1->getBufferStart(), MBC2->getBufferStart());
}

TEST_F(MemoryBufferTest, make_new) {
  // 0-sized buffer
  OwningBuffer Zero(MemoryBuffer::getNewUninitMemBuffer(0));
  EXPECT_TRUE(nullptr != Zero.get());

  // uninitialized buffer with no name
  OwningBuffer One(MemoryBuffer::getNewUninitMemBuffer(321));
  EXPECT_TRUE(nullptr != One.get());

  // uninitialized buffer with name
  OwningBuffer Two(MemoryBuffer::getNewUninitMemBuffer(123, "bla"));
  EXPECT_TRUE(nullptr != Two.get());

  // 0-initialized buffer with no name
  OwningBuffer Three(MemoryBuffer::getNewMemBuffer(321, data));
  EXPECT_TRUE(nullptr != Three.get());
  for (size_t i = 0; i < 321; ++i)
    EXPECT_EQ(0, Three->getBufferStart()[0]);

  // 0-initialized buffer with name
  OwningBuffer Four(MemoryBuffer::getNewMemBuffer(123, "zeros"));
  EXPECT_TRUE(nullptr != Four.get());
  for (size_t i = 0; i < 123; ++i)
    EXPECT_EQ(0, Four->getBufferStart()[0]);
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

  ErrorOr<OwningBuffer> Buf =
      MemoryBuffer::getOpenFileSlice(TestFD, TestPath.c_str(),
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

TEST_F(MemoryBufferTest, reference) {
  OwningBuffer MB(MemoryBuffer::getMemBuffer(data));
  MemoryBufferRef MBR(*MB);

  EXPECT_EQ(MB->getBufferStart(), MBR.getBufferStart());
  EXPECT_EQ(MB->getBufferIdentifier(), MBR.getBufferIdentifier());
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
}
