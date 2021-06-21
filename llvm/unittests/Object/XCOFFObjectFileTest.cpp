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
using namespace llvm::XCOFF;

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
  EXPECT_EQ(TT.getNumOfGPRsSaved(), 0);

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

  V[6] = 0x04;
  V[7] = 0x1E;
  V[8] = 0xAC;
  V[9] = 0xAA;
  V[10] = 0xAA;
  V[11] = 0xAA;
  Size = sizeof(V);
  Expected<XCOFFTracebackTable> TTOrErr4 = XCOFFTracebackTable::create(V, Size);
  ASSERT_THAT_EXPECTED(TTOrErr4, Succeeded());
  XCOFFTracebackTable TT4 = *TTOrErr4;
  ASSERT_TRUE(TT4.getParmsType());
  EXPECT_EQ(TT4.getParmsType().getValue(),
            "f, f, d, i, i, f, f, f, f, f, f, f, f, f, f, f, f, ...");
}

const uint8_t TBTableData[] = {
    0x00, 0x00, 0x2A, 0x60, 0x80, 0xc0, 0x03, 0x05, 0x48, 0xc4, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x05, 0x05, 0x00, 0x00,
    0x06, 0x06, 0x00, 0x00, 0x00, 0x07, 0x61, 0x64, 0x64, 0x5f, 0x61, 0x6c,
    0x6c, 0x1f, 0x02, 0x05, 0xf0, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIControlledStorageInfoDisp) {
  uint64_t Size = sizeof(TBTableData);
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
  EXPECT_EQ(Size, 45u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIAllocaRegister) {
  uint64_t Size = sizeof(TBTableData);
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;
  ASSERT_TRUE(TT.getAllocaRegister());
  EXPECT_EQ(TT.getAllocaRegister().getValue(), 31u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIHasVectorInfo) {

  uint64_t Size = sizeof(TBTableData);
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;

  EXPECT_EQ(TT.getNumberOfFixedParms(), 3);
  EXPECT_EQ(TT.getNumberOfFPParms(), 2);
  EXPECT_TRUE(TT.hasVectorInfo());
  EXPECT_TRUE(TT.hasExtensionTable());

  ASSERT_TRUE(TT.getParmsType());
  EXPECT_EQ(TT.getParmsType().getValue(), "v, i, f, i, d, i, v");

  ASSERT_TRUE(TT.getVectorExt());
  TBVectorExt VecExt = TT.getVectorExt().getValue();

  EXPECT_EQ(VecExt.getNumberOfVRSaved(), 0);
  EXPECT_TRUE(VecExt.isVRSavedOnStack());
  EXPECT_FALSE(VecExt.hasVarArgs());

  EXPECT_EQ(VecExt.getNumberOfVectorParms(), 2u);
  EXPECT_TRUE(VecExt.hasVMXInstruction());

  EXPECT_EQ(VecExt.getVectorParmsInfo(), "vf, vf");

  ASSERT_TRUE(TT.getExtensionTable());
  EXPECT_EQ(TT.getExtensionTable().getValue(),
            ExtendedTBTableFlag::TB_SSP_CANARY);

  EXPECT_EQ(Size, 45u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableAPIHasVectorInfo1) {
  const uint8_t TBTableData[] = {
      0x00, 0x00, 0x2A, 0x40, 0x80, 0xc0, 0x03, 0x05, 0x48, 0xc5, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x05, 0x05, 0x00, 0x00,
      0x06, 0x06, 0x00, 0x00, 0x00, 0x07, 0x61, 0x64, 0x64, 0x5f, 0x61, 0x6c,
      0x6c, 0x11, 0x07, 0x90, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  uint64_t Size = sizeof(TBTableData);
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  ASSERT_THAT_EXPECTED(TTOrErr, Succeeded());
  XCOFFTracebackTable TT = *TTOrErr;

  ASSERT_TRUE(TT.getParmsType());
  EXPECT_EQ(TT.getParmsType().getValue(), "v, i, f, i, d, i, v, v");

  ASSERT_TRUE(TT.getVectorExt());
  TBVectorExt VecExt = TT.getVectorExt().getValue();

  EXPECT_EQ(VecExt.getNumberOfVRSaved(), 4);
  EXPECT_FALSE(VecExt.isVRSavedOnStack());
  EXPECT_TRUE(VecExt.hasVarArgs());

  EXPECT_EQ(VecExt.getNumberOfVectorParms(), 3u);
  EXPECT_TRUE(VecExt.hasVMXInstruction());

  EXPECT_EQ(VecExt.getVectorParmsInfo(), "vi, vs, vc");

  ASSERT_TRUE(TT.getExtensionTable());
  EXPECT_EQ(TT.getExtensionTable().getValue(),
            ExtendedTBTableFlag::TB_SSP_CANARY);

  EXPECT_EQ(Size, 44u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtMandatory) {
  uint64_t Size = 6;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x6 while reading [0x0, 0x8)"));
  EXPECT_EQ(Size, 0u);
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
  uint64_t Size = 37;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);
  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x25 while reading [0x25, 0x26)"));
  EXPECT_EQ(Size, 37u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtVectorInfoData) {
  uint64_t Size = 39;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x27 while reading [0x26, 0x2c)"));
  EXPECT_EQ(Size, 38u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtVectorInfoParmsInfo) {
  uint64_t Size = 43;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x2b while reading [0x26, 0x2c)"));
  EXPECT_EQ(Size, 38u);
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableTruncatedAtExtLongTBTable) {
  uint64_t Size = 44;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage(
          "unexpected end of data at offset 0x2c while reading [0x2c, 0x2d)"));
  EXPECT_EQ(Size, 44u);
}

TEST(XCOFFObjectFileTest, XCOFFGetCsectAuxRef32) {
  uint8_t XCOFF32Binary[] = {
      // File header.
      0x01, 0xdf, 0x00, 0x01, 0x5f, 0x58, 0xf8, 0x95, 0x00, 0x00, 0x00, 0x3c,
      0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00,

      // Section header for empty .data section.
      0x2e, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x40,

      // Start of symbol table.
      // C_File symbol.
      0x2e, 0x66, 0x69, 0x6c, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xff, 0xfe, 0x00, 0x03, 0x67, 0x01,
      // File Auxiliary Entry.
      0x61, 0x2e, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

      // Csect symbol.
      0x2e, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x00, 0x6b, 0x01,
      // Csect auxiliary entry.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x05,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  ArrayRef<uint8_t> XCOFF32Ref(XCOFF32Binary, sizeof(XCOFF32Binary));
  Expected<std::unique_ptr<ObjectFile>> XCOFFObjOrErr =
      object::ObjectFile::createObjectFile(
          MemoryBufferRef(toStringRef(XCOFF32Ref), "dummyXCOFF"),
          file_magic::xcoff_object_32);
  ASSERT_THAT_EXPECTED(XCOFFObjOrErr, Succeeded());

  const XCOFFObjectFile &File = *cast<XCOFFObjectFile>((*XCOFFObjOrErr).get());
  DataRefImpl Ref;
  Ref.p = File.getSymbolEntryAddressByIndex(2);
  XCOFFSymbolRef SymRef = File.toSymbolRef(Ref);
  Expected<XCOFFCsectAuxRef> CsectRefOrErr = SymRef.getXCOFFCsectAuxRef();
  ASSERT_THAT_EXPECTED(CsectRefOrErr, Succeeded());

  // Set csect symbol's auxiliary entry count to 0.
  XCOFF32Binary[113] = 0;
  Expected<XCOFFCsectAuxRef> ExpectErr = SymRef.getXCOFFCsectAuxRef();
  EXPECT_THAT_ERROR(
      ExpectErr.takeError(),
      FailedWithMessage("csect symbol \".data\" contains no auxiliary entry"));
}

TEST(XCOFFObjectFileTest, XCOFFGetCsectAuxRef64) {
  uint8_t XCOFF64Binary[] = {
      // File header.
      0x01, 0xf7, 0x00, 0x01, 0x5f, 0x59, 0x25, 0xeb, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,

      // Section header for empty .data section.
      0x2e, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00,

      // Start of symbol table.
      // C_File symbol.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
      0xff, 0xfe, 0x00, 0x02, 0x67, 0x01,
      // File Auxiliary Entry.
      0x61, 0x2e, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0xfc,

      // Csect symbol.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a,
      0x00, 0x01, 0x00, 0x00, 0x6b, 0x01,
      // Csect auxiliary entry.
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x05,
      0x00, 0x00, 0x00, 0x00, 0x00, 0xfb,

      // String table.
      0x00, 0x00, 0x00, 0x10, 0x2e, 0x66, 0x69, 0x6c, 0x65, 0x00, 0x2e, 0x64,
      0x61, 0x74, 0x61, 0x00};

  ArrayRef<uint8_t> XCOFF64Ref(XCOFF64Binary, sizeof(XCOFF64Binary));
  Expected<std::unique_ptr<ObjectFile>> XCOFFObjOrErr =
      object::ObjectFile::createObjectFile(
          MemoryBufferRef(toStringRef(XCOFF64Ref), "dummyXCOFF"),
          file_magic::xcoff_object_64);
  ASSERT_THAT_EXPECTED(XCOFFObjOrErr, Succeeded());

  const XCOFFObjectFile &File = *cast<XCOFFObjectFile>((*XCOFFObjOrErr).get());
  DataRefImpl Ref;
  Ref.p = File.getSymbolEntryAddressByIndex(2);
  XCOFFSymbolRef SymRef = File.toSymbolRef(Ref);
  Expected<XCOFFCsectAuxRef> CsectRefOrErr = SymRef.getXCOFFCsectAuxRef();
  ASSERT_THAT_EXPECTED(CsectRefOrErr, Succeeded());

  // Inject incorrect auxiliary type value.
  XCOFF64Binary[167] = static_cast<uint8_t>(XCOFF::AUX_SYM);
  Expected<XCOFFCsectAuxRef> NotFoundErr = SymRef.getXCOFFCsectAuxRef();
  EXPECT_THAT_ERROR(
      NotFoundErr.takeError(),
      FailedWithMessage(
          "a csect auxiliary entry is not found for symbol \".data\""));

  // Set csect symbol's auxiliary entry count to 0.
  XCOFF64Binary[149] = 0;
  Expected<XCOFFCsectAuxRef> ExpectErr = SymRef.getXCOFFCsectAuxRef();
  EXPECT_THAT_ERROR(
      ExpectErr.takeError(),
      FailedWithMessage("csect symbol \".data\" contains no auxiliary entry"));
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableErrorAtParameterType) {
  const uint8_t TBTableData[] = {0x00, 0x00, 0x22, 0x40, 0x80, 0x00, 0x01,
                                 0x05, 0x58, 0x00, 0x10, 0x00, 0x00, 0x00,
                                 0x00, 0x40, 0x00, 0x07, 0x61, 0x64, 0x64,
                                 0x5f, 0x61, 0x6c, 0x6c, 0x00, 0x00, 0x00};
  uint64_t Size = 28;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage("ParmsType encodes can not map to ParmsNum parameters "
                        "in parseParmsType."));
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableErrorAtParameterTypeWithVecInfo) {
  const uint8_t TBTableData[] = {
      0x00, 0x00, 0x2A, 0x40, 0x80, 0xc0, 0x03, 0x05, 0x48, 0xc0, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x05, 0x05, 0x00, 0x00,
      0x06, 0x06, 0x00, 0x00, 0x00, 0x07, 0x61, 0x64, 0x64, 0x5f, 0x61, 0x6c,
      0x6c, 0x11, 0x07, 0x90, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  uint64_t Size = 44;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(
      TTOrErr.takeError(),
      FailedWithMessage("ParmsType encodes can not map to ParmsNum parameters "
                        "in parseParmsTypeWithVecInfo."));
}

TEST(XCOFFObjectFileTest, XCOFFTracebackTableErrorAtVecParameterType) {
  const uint8_t TBTableData[] = {
      0x00, 0x00, 0x2A, 0x40, 0x80, 0xc0, 0x03, 0x05, 0x48, 0xc0, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x05, 0x05, 0x00, 0x00,
      0x06, 0x06, 0x00, 0x00, 0x00, 0x07, 0x61, 0x64, 0x64, 0x5f, 0x61, 0x6c,
      0x6c, 0x11, 0x07, 0x90, 0x00, 0x00, 0x20, 0x20, 0x00, 0x00, 0x00};
  uint64_t Size = 44;
  Expected<XCOFFTracebackTable> TTOrErr =
      XCOFFTracebackTable::create(TBTableData, Size);

  EXPECT_THAT_ERROR(TTOrErr.takeError(),
                    FailedWithMessage("ParmsType encodes more than ParmsNum "
                                      "parameters in parseVectorParmsType."));
}
