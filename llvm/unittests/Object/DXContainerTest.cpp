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
  StringRef Buffer("DXBC");
  EXPECT_EQ(identify_magic(Buffer), file_magic::dxcontainer_object);
}

TEST(DXCFile, ParseHeaderErrors) {
  uint8_t Buffer[] = {0x44, 0x58, 0x42, 0x43};
  EXPECT_THAT_EXPECTED(
      DXContainer::create(getMemoryBuffer<4>(Buffer)),
      FailedWithMessage("Reading structure out of file bounds"));
}

TEST(DXCFile, ParseHeader) {
  uint8_t Buffer[] = {0x44, 0x58, 0x42, 0x43, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                      0x70, 0x0D, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00};
  DXContainer C =
      llvm::cantFail(DXContainer::create(getMemoryBuffer<32>(Buffer)));
  EXPECT_TRUE(memcmp(C.getHeader().Magic, "DXBC", 4) == 0);
  EXPECT_TRUE(memcmp(C.getHeader().FileHash.Digest,
                     "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16) == 0);
  EXPECT_EQ(C.getHeader().Version.Major, 1u);
  EXPECT_EQ(C.getHeader().Version.Minor, 0u);
}
