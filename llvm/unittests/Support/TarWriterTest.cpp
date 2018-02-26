//===- llvm/unittest/Support/TarWriterTest.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TarWriter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
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
  // Create a temporary file.
  SmallString<128> Path;
  std::error_code EC =
      sys::fs::createTemporaryFile("TarWriterTest", "tar", Path);
  EXPECT_FALSE((bool)EC);

  // Create a tar file.
  Expected<std::unique_ptr<TarWriter>> TarOrErr = TarWriter::create(Path, Base);
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append(Filename, "contents");
  Tar.reset();

  // Read the tar file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFile(Path);
  EXPECT_TRUE((bool)MBOrErr);
  std::unique_ptr<MemoryBuffer> MB = std::move(*MBOrErr);
  std::vector<uint8_t> Buf((const uint8_t *)MB->getBufferStart(),
                           (const uint8_t *)MB->getBufferEnd());

  // Windows does not allow us to remove a mmap'ed files, so
  // unmap first and then remove the temporary file.
  MB = nullptr;
  sys::fs::remove(Path);

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
  std::string x154(154, 'x');
  std::string x155(155, 'x');
  std::string y99(99, 'y');
  std::string y100(100, 'y');

  UstarHeader Hdr1 = createUstar("", x154 + "/" + y99);
  EXPECT_EQ("/" + x154, StringRef(Hdr1.Prefix));
  EXPECT_EQ(y99, StringRef(Hdr1.Name));

  UstarHeader Hdr2 = createUstar("", x155 + "/" + y99);
  EXPECT_EQ("", StringRef(Hdr2.Prefix));
  EXPECT_EQ("", StringRef(Hdr2.Name));

  UstarHeader Hdr3 = createUstar("", x154 + "/" + y100);
  EXPECT_EQ("", StringRef(Hdr3.Prefix));
  EXPECT_EQ("", StringRef(Hdr3.Name));

  UstarHeader Hdr4 = createUstar("", x155 + "/" + y100);
  EXPECT_EQ("", StringRef(Hdr4.Prefix));
  EXPECT_EQ("", StringRef(Hdr4.Name));

  std::string yz = "yyyyyyyyyyyyyyyyyyyy/zzzzzzzzzzzzzzzzzzzz";
  UstarHeader Hdr5 = createUstar("", x154 + "/" + yz);
  EXPECT_EQ("/" + x154, StringRef(Hdr5.Prefix));
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
  SmallString<128> Path;
  std::error_code EC =
      sys::fs::createTemporaryFile("TarWriterTest", "tar", Path);
  EXPECT_FALSE((bool)EC);

  Expected<std::unique_ptr<TarWriter>> TarOrErr = TarWriter::create(Path, "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar.reset();

  uint64_t TarSize;
  EC = sys::fs::file_size(Path, TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 2048ULL);
}

TEST_F(TarWriterTest, NoDuplicate) {
  SmallString<128> Path;
  std::error_code EC =
      sys::fs::createTemporaryFile("TarWriterTest", "tar", Path);
  EXPECT_FALSE((bool)EC);

  Expected<std::unique_ptr<TarWriter>> TarOrErr = TarWriter::create(Path, "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar->append("BarPath", "bar");
  Tar.reset();

  uint64_t TarSize;
  EC = sys::fs::file_size(Path, TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 3072ULL);
}

TEST_F(TarWriterTest, Duplicate) {
  SmallString<128> Path;
  std::error_code EC =
      sys::fs::createTemporaryFile("TarWriterTest", "tar", Path);
  EXPECT_FALSE((bool)EC);

  Expected<std::unique_ptr<TarWriter>> TarOrErr = TarWriter::create(Path, "");
  EXPECT_TRUE((bool)TarOrErr);
  std::unique_ptr<TarWriter> Tar = std::move(*TarOrErr);
  Tar->append("FooPath", "foo");
  Tar->append("FooPath", "bar");
  Tar.reset();

  uint64_t TarSize;
  EC = sys::fs::file_size(Path, TarSize);
  EXPECT_FALSE((bool)EC);
  EXPECT_EQ(TarSize, 2048ULL);
}
} // namespace
