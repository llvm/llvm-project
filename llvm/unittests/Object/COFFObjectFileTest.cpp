//===- COFFObjectFileTest.cpp - Tests for COFFObjectFile ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFF.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace llvm::object;

TEST(COFFObjectFileTest, CHPERangeEntry) {
  chpe_range_entry range;

  range.StartOffset = 0x1000;
  EXPECT_EQ(range.getStart(), 0x1000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64);

  range.StartOffset = 0x2000 | chpe_range_type::Arm64EC;
  EXPECT_EQ(range.getStart(), 0x2000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Arm64EC);

  range.StartOffset = 0x3000 | chpe_range_type::Amd64;
  EXPECT_EQ(range.getStart(), 0x3000u);
  EXPECT_EQ(range.getType(), chpe_range_type::Amd64);
}

TEST(COFFObjectFileTest, PESignatureOffsetOutOfBounds) {
  const uint32_t SignatureOffsets[] = {sizeof(dos_header) + 1, 0x2020,
                                       UINT32_MAX};
  for (uint32_t SignatureOffset : SignatureOffsets) {
    SCOPED_TRACE(SignatureOffset);
    dos_header DH = {};
    DH.Magic[0] = 'M';
    DH.Magic[1] = 'Z';
    DH.AddressOfNewExeHeader = SignatureOffset;

    std::vector<char> Buf(sizeof(DH) + sizeof(COFF::PEMagic));
    memcpy(Buf.data(), &DH, sizeof(DH));
    Expected<std::unique_ptr<COFFObjectFile>> ObjOrErr = COFFObjectFile::create(
        MemoryBufferRef(StringRef(Buf.data(), Buf.size()), "dos-header-only"));
    EXPECT_THAT_ERROR(
        ObjOrErr.takeError(),
        FailedWithMessage("The end of the file was unexpectedly encountered"));
  }
}
