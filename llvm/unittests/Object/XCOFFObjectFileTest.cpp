//===- XCOFFObjectFileTest.cpp - Tests for XCOFFObjectFile ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

TEST(XCOFFObjectFileTest, XCOFFObjectType) {
  // Create an arbitrary object of a non-XCOFF type and test that
  // dyn_cast<XCOFFObjectFile> returns null for it.
  char Buf[sizeof(typename ELF64LE::Ehdr)] = {};
  memcpy(Buf, "\177ELF", 4);

  auto *EHdr = reinterpret_cast<typename ELF64LE::Ehdr *>(Buf);
  EHdr->e_ident[llvm::ELF::EI_CLASS] = llvm::ELF::ELFCLASS64;
  EHdr->e_ident[llvm::ELF::EI_DATA] = llvm::ELF::ELFDATA2LSB;

  MemoryBufferRef Source(StringRef(Buf, sizeof(Buf)), "non-XCOFF");
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createObjectFile(Source);
  ASSERT_THAT_EXPECTED(ObjOrErr, Succeeded());

  EXPECT_TRUE(dyn_cast<XCOFFObjectFile>((*ObjOrErr).get()) == nullptr);
}

TEST(XCOFFObjectFileTest, doesXCOFFTracebackTableBegin) {
  EXPECT_TRUE(doesXCOFFTracebackTableBegin({0, 0, 0, 0}));
  EXPECT_TRUE(doesXCOFFTracebackTableBegin({0, 0, 0, 0, 1}));
  EXPECT_FALSE(doesXCOFFTracebackTableBegin({0, 0, 0, 1}));
  EXPECT_FALSE(doesXCOFFTracebackTableBegin({0, 0, 0}));
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIGeneral) {
  uint8_t V[] = {0x00, 0x00, 0x22, 0x40, 0x80, 0x00, 0x01, 0x05, 0x58, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x07, 0x61, 0x64,
                 0x64, 0x5f, 0x61, 0x6c, 0x6c, 0x00, 0x00, 0x00};
  uint64_t Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr = XCOFFTracebackTable::create(V, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;

  EXPECT_EQ(TT.getVersion(), 0);

  EXPECT_EQ(TT.getLanguageID(), 0);

  EXPECT_FALSE(TT.isGlobalLinkage());
  EXPECT_FALSE(TT.isOutOfLineEpilogOrPrologue());
  EXPECT_TRUE(TT.hasTraceBackTableOffset());
  EXPECT_FALSE(TT.isInternalProcedure());
  EXPECT_FALSE(TT.hasControlledStorage());
  EXPECT_FALSE(TT.isTOCless());
  EXPECT_TRUE(TT.isFloatingPointPresent());
  EXPECT_FALSE(TT.isFloatingPointOperationLogOrAbortEnabled());

  EXPECT_FALSE(TT.isInterruptHandler());
  EXPECT_TRUE(TT.isFuncNamePresent());
  EXPECT_FALSE(TT.isAllocaUsed());
  EXPECT_EQ(TT.getOnConditionDirective(), 0);
  EXPECT_FALSE(TT.isCRSaved());
  EXPECT_FALSE(TT.isLRSaved());

  EXPECT_TRUE(TT.isBackChainStored());
  EXPECT_FALSE(TT.isFixup());
  EXPECT_EQ(TT.getNumOfFPRsSaved(), 0);

  EXPECT_FALSE(TT.hasExtensionTable());
  EXPECT_FALSE(TT.hasVectorInfo());
  EXPECT_EQ(TT.getNumofGPRsSaved(), 0);

  EXPECT_EQ(TT.getNumberOfFixedParms(), 1);

  EXPECT_EQ(TT.getNumberOfFPParms(), 2);
  EXPECT_TRUE(TT.hasParmsOnStack());

  ASSERT_TRUE(TT.getParmsType());
  EXPECT_EQ(TT.getParmsType().getValue(), "i, f, d");

  ASSERT_TRUE(TT.getTraceBackTableOffset());
  EXPECT_EQ(TT.getTraceBackTableOffset().getValue(), 64u);

  EXPECT_FALSE(TT.getHandlerMask());

  ASSERT_TRUE(TT.getFunctionName());
  EXPECT_EQ(TT.getFunctionName().getValue(), "add_all");
  EXPECT_EQ(TT.getFunctionName().getValue().size(), 7u);

  EXPECT_FALSE(TT.getAllocaRegister());
  EXPECT_EQ(Size, 25u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIParmsType) {
  uint8_t V[] = {0x01, 0x02, 0xA2, 0x40, 0x80, 0x00, 0x02, 0x07, 0x2B, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x07, 0x61, 0x64,
                 0x64, 0x5f, 0x61, 0x6c, 0x6c, 0x00, 0x00, 0x00};
  uint64_t Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr = XCOFFTracebackTable::create(V, Size);

  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;
  EXPECT_EQ(TT.getVersion(), 1);
  EXPECT_EQ(TT.getLanguageID(), 2);

  EXPECT_TRUE(TT.isGlobalLinkage());
  EXPECT_EQ(TT.getNumberOfFixedParms(), 2);

  EXPECT_EQ(TT.getNumberOfFPParms(), 3);

  ASSERT_TRUE(TT.getParmsType());
  EXPECT_EQ(TT.getParmsType().getValue(), "i, i, f, f, d");

  V[8] = 0xAC;
  Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr1 = XCOFFTracebackTable::create(V, Size);
  ASSERT_THAT_EXPECTED(TTOrErr1, Succeeded());
  XCOFFTracebackTable TT1 = *TTOrErr1;
  ASSERT_TRUE(TT1.getParmsType());
  EXPECT_EQ(TT1.getParmsType().getValue(), "f, f, d, i, i");

  V[8] = 0xD4;
  Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr2 = XCOFFTracebackTable::create(V, Size);
  ASSERT_THAT_EXPECTED(TTOrErr2, Succeeded());
  XCOFFTracebackTable TT2 = *TTOrErr2;
  ASSERT_TRUE(TT2.getParmsType());
  EXPECT_EQ(TT2.getParmsType().getValue(), "d, i, f, f, i");

  V[6] = 0x01;
  Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr3 = XCOFFTracebackTable::create(V, Size);
  ASSERT_THAT_EXPECTED(TTOrErr3, Succeeded());
  XCOFFTracebackTable TT3 = *TTOrErr3;
  ASSERT_TRUE(TT3.getParmsType());
  EXPECT_EQ(TT3.getParmsType().getValue(), "d, i, f, f");
}

const uint8_t TBTableData[] = {0x00, 0x00, 0x2A, 0x40, 0x80, 0x40, 0x01, 0x05,
                               0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
                               0x00, 0x00, 0x00, 0x02, 0x05, 0x05, 0x00, 0x00,
                               0x06, 0x06, 0x00, 0x00, 0x00, 0x07, 0x61, 0x64,
                               0x64, 0x5f, 0x61, 0x6c, 0x6c, 0x00, 0x00, 0x00};

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIControlledStorageInfoDisp) {
  uint64_t Size = 40;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;
  EXPECT_TRUE(TT.hasControlledStorage());
  ASSERT_TRUE(TT.getNumOfCtlAnchors());
  EXPECT_EQ(TT.getNumOfCtlAnchors().getValue(), 2u);

  ASSERT_TRUE(TT.getControlledStorageInfoDisp());

  SmallVector<uint32_t, 8> Disp = TT.getControlledStorageInfoDisp().getValue();

  ASSERT_EQ(Disp.size(), 2UL);
  EXPECT_EQ(Disp[0], 0x05050000u);
  EXPECT_EQ(Disp[1], 0x06060000u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIHasVectorInfo) {
  uint64_t Size = 40;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;

  EXPECT_TRUE(TT.hasVectorInfo());
  EXPECT_FALSE(TT.getParmsType());
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtMandatory) {
  uint64_t Size = 6;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x6 while reading [0x0, 0x8)"));
  EXPECT_EQ(Size, 0UL);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtParmsType) {
  uint64_t Size = 9;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x9 while reading [0x8, 0xc)"));
  EXPECT_EQ(Size, 8u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtTBOffset) {
  uint64_t Size = 14;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0xe while reading [0xc, 0x10)"));
  EXPECT_EQ(Size, 12u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtHandlerMask) {
  uint8_t V[] = {0x00, 0x00, 0x22, 0xC0, 0x80, 0x00, 0x01, 0x05, 0x58,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x07};
  uint64_t Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr = XCOFFTracebackTable::create(V, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x12 while reading [0x10, 0x14)"));
  EXPECT_EQ(Size, 16u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtNumOfCtlAnchors) {
  uint64_t Size = 19;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x13 while reading [0x10, 0x14)"));
  EXPECT_EQ(Size, 16u);
}

TEST(XCOFFObjectFileTest,
     XCOFFTracebackTableTruncatedAtControlledStorageInfoDisp) {
  uint64_t Size = 21;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x15 while reading [0x14, 0x18)"));
  EXPECT_EQ(Size, 20u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtNameLen) {
  uint64_t Size = 29;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x1d while reading [0x1c, 0x1e)"));
  EXPECT_EQ(Size, 28u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtFunctionName) {
  uint64_t Size = 36;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x24 while reading [0x1e, 0x25)"));
  EXPECT_EQ(Size, 30u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtAllocaUsed) {
  uint8_t V[] = {0x00, 0x00, 0x2A, 0x60, 0x80, 0x00, 0x01, 0x05, 0x58, 0x00,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02,
                 0x05, 0x05, 0x00, 0x00, 0x06, 0x06, 0x00, 0x00, 0x00, 0x07,
                 0x61, 0x64, 0x64, 0x5f, 0x61, 0x6c, 0x6c};
  uint64_t Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr = XCOFFTracebackTable::create(V, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x25 while reading [0x25, 0x26)"));
  EXPECT_EQ(Size, 37u);
}
