//===- llvm/unittest/Support/TarWriterTest.cpp ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TarWriter.h"
#include "gtest/gtest.h"

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

static UstarHeader create(StringRef Base, StringRef Filename) {
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
  sys::fs::remove(Path);
  return *reinterpret_cast<const UstarHeader *>(MB->getBufferStart());
}

TEST_F(TarWriterTest, Basics) {
  UstarHeader Hdr = create("base", "file");
  EXPECT_EQ("ustar", StringRef(Hdr.Magic));
  EXPECT_EQ("00", StringRef(Hdr.Version, 2));
  EXPECT_EQ("base/file", StringRef(Hdr.Name));
  EXPECT_EQ("00000000010", StringRef(Hdr.Size));
}

TEST_F(TarWriterTest, LongFilename) {
  UstarHeader Hdr1 = create(
      "012345678", std::string(99, 'x') + "/" + std::string(44, 'x') + "/foo");
  EXPECT_EQ("foo", StringRef(Hdr1.Name));
  EXPECT_EQ("012345678/" + std::string(99, 'x') + "/" + std::string(44, 'x'),
            StringRef(Hdr1.Prefix));

  UstarHeader Hdr2 = create(
      "012345678", std::string(99, 'x') + "/" + std::string(45, 'x') + "/foo");
  EXPECT_EQ("foo", StringRef(Hdr2.Name));
  EXPECT_EQ("012345678/" + std::string(99, 'x') + "/" + std::string(45, 'x'),
            StringRef(Hdr2.Prefix));

  UstarHeader Hdr3 = create(
      "012345678", std::string(99, 'x') + "/" + std::string(46, 'x') + "/foo");
  EXPECT_EQ(std::string(46, 'x') + "/foo", StringRef(Hdr3.Name));
  EXPECT_EQ("012345678/" + std::string(99, 'x'), StringRef(Hdr3.Prefix));
}
}
