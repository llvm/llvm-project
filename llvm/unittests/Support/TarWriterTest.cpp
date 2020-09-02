//===- llvm/unittest/Support/TarWriterTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TarWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using llvm::unittest::TempFile;

namespace {

struct UstarHeader {
  char Name[100];
  char Mode[8];
  char Uid[8];
  char Gid[8];
  char Size[12];
  char Mtime[12];
  char Checksum[8];
  char TypeFlag;
  char Linkname[100];
  char Magic[6];
  char Version[2];
  char Uname[32];
  char Gname[32];
  char DevMajor[8];
  char DevMinor[8];
  char Prefix[155];
  char Pad[12];
};

class TarWriterTest : public ::testing::Test {};

static std::vector<uint8_t> createTar(StringRef Base, StringRef Filename) {
  TempFile TarWriterTest("TarWriterTest", "tar", "", /*Unique*/ true);

  // Create a tar file.
  Expected<std::unique_ptr<TarWriter>> TarOrErr =
      TarWriter::create(TarWriterTest.path(), Base);
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append(Filename, "contents");
  Tar.reset();

  // Read the tar file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(TarWriterTest.path());
  EXPECT_TRUE((bool)MBOrErr);
  std::unique_ptr<MemoryBuffer> MB = std::move(*MBOrErr);
  std::vector<uint8_t> Buf((const uint8_t *)MB->getBufferStart(),
                           (const uint8_t *)MB->getBufferEnd());

  // Windows does not allow us to remove a mmap'ed files, so
  // unmap first and then remove the temporary file.
  MB = nullptr;

  return Buf;
}

static UstarHeader createUstar(StringRef Base, StringRef Filename) {
  std::vector<uint8_t> Buf = createTar(Base, Filename);
  EXPECT_TRUE(Buf.size() >= sizeof(UstarHeader));
  return *reinterpret_cast<const UstarHeader *>(Buf.data());
}

TEST_F(TarWriterTest, Basics) {
  UstarHeader Hdr = createUstar("base", "file");
  EXPECT_EQ("ustar", StringRef(Hdr.Magic));
  EXPECT_EQ("00", StringRef(Hdr.Version, 2));
  EXPECT_EQ("base/file", StringRef(Hdr.Name));
  EXPECT_EQ("00000000010", StringRef(Hdr.Size));
}

TEST_F(TarWriterTest, LongFilename) {
  // The prefix is prefixed by an additional '/' so it's one longer than the
  // number of x's here.
  std::string x136(136, 'x');
  std::string x137(137, 'x');
  std::string y99(99, 'y');
  std::string y100(100, 'y');

  UstarHeader Hdr1 = createUstar("", x136 + "/" + y99);
  EXPECT_EQ("/" + x136, StringRef(Hdr1.Prefix));
  EXPECT_EQ(y99, StringRef(Hdr1.Name));

  UstarHeader Hdr2 = createUstar("", x137 + "/" + y99);
  EXPECT_EQ("", StringRef(Hdr2.Prefix));
  EXPECT_EQ("", StringRef(Hdr2.Name));

  UstarHeader Hdr3 = createUstar("", x136 + "/" + y100);
  EXPECT_EQ("", StringRef(Hdr3.Prefix));
  EXPECT_EQ("", StringRef(Hdr3.Name));

  UstarHeader Hdr4 = createUstar("", x137 + "/" + y100);
  EXPECT_EQ("", StringRef(Hdr4.Prefix));
  EXPECT_EQ("", StringRef(Hdr4.Name));

  std::string yz = "yyyyyyyyyyyyyyyyyyyy/zzzzzzzzzzzzzzzzzzzz";
  UstarHeader Hdr5 = createUstar("", x136 + "/" + yz);
  EXPECT_EQ("/" + x136, StringRef(Hdr5.Prefix));
  EXPECT_EQ(yz, StringRef(Hdr5.Name));
}

TEST_F(TarWriterTest, Pax) {
  std::vector<uint8_t> Buf = createTar("", std::string(200, 'x'));
  EXPECT_TRUE(Buf.size() >= 1024);

  auto *Hdr = reinterpret_cast<const UstarHeader *>(Buf.data());
  EXPECT_EQ("", StringRef(Hdr->Prefix));
  EXPECT_EQ("", StringRef(Hdr->Name));

  StringRef Pax = StringRef((char *)(Buf.data() + 512), 512);
  EXPECT_TRUE(Pax.startswith("211 path=/" + std::string(200, 'x')));
}

TEST_F(TarWriterTest, SingleFile) {
  TempFile TarWriterTest("TarWriterTest", "tar", "", /*Unique*/ true);

  Expected<std::unique_ptr<TarWriter>> TarOrErr =
      TarWriter::create(TarWriterTest.path(), "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar.reset();

  uint64_t TarSize;
  std::error_code EC = sys::fs::file_size(TarWriterTest.path(), TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 2048ULL);
}

TEST_F(TarWriterTest, NoDuplicate) {
  TempFile TarWriterTest("TarWriterTest", "tar", "", /*Unique*/ true);

  Expected<std::unique_ptr<TarWriter>> TarOrErr =
      TarWriter::create(TarWriterTest.path(), "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar->append("BarPath", "bar");
  Tar.reset();

  uint64_t TarSize;
  std::error_code EC = sys::fs::file_size(TarWriterTest.path(), TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 3072ULL);
}

TEST_F(TarWriterTest, Duplicate) {
  TempFile TarWriterTest("TarWriterTest", "tar", "", /*Unique*/ true);

  Expected<std::unique_ptr<TarWriter>> TarOrErr =
      TarWriter::create(TarWriterTest.path(), "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar->append("FooPath", "bar");
  Tar.reset();

  uint64_t TarSize;
  std::error_code EC = sys::fs::file_size(TarWriterTest.path(), TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 2048ULL);
}
} // namespace
