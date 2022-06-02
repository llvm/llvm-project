//===- DXContainerTest.cpp - Tests for DXContainerFile --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/DXContainer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

template <std::size_t X> MemoryBufferRef getMemoryBuffer(uint8_t Data[X]) {
  StringRef Obj(reinterpret_cast<char *>(&Data[0]), X);
  return MemoryBufferRef(Obj, "");
}

TEST(DXCFile, IdentifyMagic) {
  {
    StringRef Buffer("DXBC");
    EXPECT_EQ(identify_magic(Buffer), file_magic::dxcontainer_object);
  }
  {
    StringRef Buffer("DXBCBlahBlahBlah");
    EXPECT_EQ(identify_magic(Buffer), file_magic::dxcontainer_object);
  }
}

TEST(DXCFile, ParseHeaderErrors) {
  uint8_t Buffer[] = {0x44, 0x58, 0x42, 0x43};
  EXPECT_THAT_EXPECTED(
      DXContainer::create(getMemoryBuffer<4>(Buffer)),
      FailedWithMessage("Reading structure out of file bounds"));
}

TEST(DXCFile, EmptyFile) {
  EXPECT_THAT_EXPECTED(
      DXContainer::create(MemoryBufferRef(StringRef("", 0), "")),
      FailedWithMessage("Reading structure out of file bounds"));
}

TEST(DXCFile, ParseHeader) {
  uint8_t Buffer[] = {0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                      0x70, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  DXContainer C =
      llvm::cantFail(DXContainer::create(getMemoryBuffer<32>(Buffer)));
  EXPECT_TRUE(memcmp(C.getHeader().Magic, "DXBC", 4) == 0);
  EXPECT_TRUE(memcmp(C.getHeader().FileHash.Digest,
                     "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16) == 0);
  EXPECT_EQ(C.getHeader().Version.Major, 1u);
  EXPECT_EQ(C.getHeader().Version.Minor, 0u);
}

TEST(DXCFile, ParsePartMissingOffsets) {
  uint8_t Buffer[] = {
      0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x70, 0x0D, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  };
  EXPECT_THAT_EXPECTED(
      DXContainer::create(getMemoryBuffer<32>(Buffer)),
      FailedWithMessage("Reading structure out of file bounds"));
}

#if defined(__ARM__)
TEST(DXCFile, DISABLED_ParsePartInvalidOffsets) {
#else
TEST(DXCFile, ParsePartInvalidOffsets) {
#endif
  uint8_t Buffer[] = {
      0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x70, 0x0D, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
  };
  EXPECT_THAT_EXPECTED(
      DXContainer::create(getMemoryBuffer<36>(Buffer)),
      FailedWithMessage("Part offset points beyond boundary of the file"));
}

TEST(DXCFile, ParseEmptyParts) {
  uint8_t Buffer[] = {
      0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
      0x70, 0x0D, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00,
      0x44, 0x00, 0x00, 0x00, 0x4C, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
      0x5C, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x6C, 0x00, 0x00, 0x00,
      0x53, 0x46, 0x49, 0x30, 0x00, 0x00, 0x00, 0x00, 0x49, 0x53, 0x47, 0x31,
      0x00, 0x00, 0x00, 0x00, 0x4F, 0x53, 0x47, 0x31, 0x00, 0x00, 0x00, 0x00,
      0x50, 0x53, 0x56, 0x30, 0x00, 0x00, 0x00, 0x00, 0x53, 0x54, 0x41, 0x54,
      0x00, 0x00, 0x00, 0x00, 0x43, 0x58, 0x49, 0x4C, 0x00, 0x00, 0x00, 0x00,
      0x44, 0x45, 0x41, 0x44, 0x00, 0x00, 0x00, 0x00,
  };
  DXContainer C =
      llvm::cantFail(DXContainer::create(getMemoryBuffer<116>(Buffer)));
  EXPECT_EQ(C.getHeader().PartCount, 7u);

  // All the part sizes are 0, which makes a nice test of the range based for
  int ElementsVisited = 0;
  for (auto Part : C) {
    EXPECT_EQ(Part.Part.Size, 0u);
    EXPECT_EQ(Part.Data.size(), 0u);
    ++ElementsVisited;
  }
  EXPECT_EQ(ElementsVisited, 7);

  {
    auto It = C.begin();
    EXPECT_TRUE(memcmp(It->Part.Name, "SFI0", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "ISG1", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "OSG1", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "PSV0", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "STAT", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "CXIL", 4) == 0);
    ++It;
    EXPECT_TRUE(memcmp(It->Part.Name, "DEAD", 4) == 0);
    ++It; // Don't increment past the end
    EXPECT_TRUE(memcmp(It->Part.Name, "DEAD", 4) == 0);
  }
}
