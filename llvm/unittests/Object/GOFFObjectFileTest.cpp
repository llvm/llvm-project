//===- GOFFObjectFileTest.cpp - Tests for GOFFObjectFile ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::GOFF;

namespace {
size_t newRecord(std::vector<char> &Data) {
  size_t Pos = Data.size();
  Data.resize(Pos + GOFF::RecordLength);
  return Pos;
}

void addEndRecord(std::vector<char> &GOFFData, uint8_t RecordCount = 0) {
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;
  GOFFData[Pos + 11] = (char)RecordCount;
}

void addHdrRecord(std::vector<char> &GOFFData, uint8_t ArchLevel = 0) {
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;
  GOFFData[Pos + 50] = (char)ArchLevel;
}

void addEsdRecord(std::vector<char> &GOFFData, uint8_t Type, uint8_t ESDID,
                  const std::vector<uint8_t> &Name, uint8_t ParentESDID = 0,
                  uint8_t BindingScope = 0, uint8_t NameSpaceID = 0,
                  uint8_t AdditionalFlags = 0,
                  uint8_t BehavioralAttributes[10] = nullptr,
                  uint32_t Length = 0) {
  size_t Pos = GOFFData.size();
  GOFFData.resize(GOFFData.size() + GOFF::RecordLength);

  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)Type;
  GOFFData[Pos + 7] = (char)ESDID;          // ESDID.
  GOFFData[Pos + 11] = (char)ParentESDID;   // Parent ESDID.
  GOFFData[Pos + 24] = (char)(Length >> 24); // Length (big-endian).
  GOFFData[Pos + 25] = (char)(Length >> 16);
  GOFFData[Pos + 26] = (char)(Length >> 8);
  GOFFData[Pos + 27] = (char)(Length);
  GOFFData[Pos + 40] = (char)NameSpaceID;   // Name Space ID
  GOFFData[Pos + 41] = (char)AdditionalFlags; // Additional Flags

  if (BehavioralAttributes) {
    for (size_t Offset=0; Offset < 10; Offset++)
      GOFFData[Pos + 60 + Offset] = (char)BehavioralAttributes[Offset];
  }

  GOFFData[Pos + 71] = (char)(Name.size()); // Size of symbol name.
  size_t StringOffset = Pos + 72; // Start of Symbol name
  for (uint8_t C : Name) {
    GOFFData[StringOffset] = (char)C;
    StringOffset++;

    if (StringOffset == Pos + GOFF::RecordLength) {
      // If we reach the end of the current record, we need to start a new one.
      GOFFData[Pos + 1] |= 0x01; // set continuation bit in the current record.

      // start a new continuation record
      Pos = GOFFData.size();
      GOFFData.resize(GOFFData.size() + GOFF::RecordLength);
      GOFFData[Pos] = (char)0x03;
      GOFFData[Pos + 1] = (char)0x02; // continuation record

      StringOffset = Pos + 3;
    }
  }
}

void constructValidGOFF(const char *Data, size_t Size) {
  StringRef ValidSize(Data, Size);
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(ValidSize, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

void constructInvalidGOFF(const char *Data, size_t Size) {
  // Construct GOFFObject with record of length != multiple of 80.
  StringRef InvalidData(Data, Size);
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(InvalidData, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("object file is not the right size. Must be a multiple "
                        "of 80 bytes, but is " +
                        std::to_string(Size) + " bytes"));
}
} // namespace

TEST(GOFFObjectFileTest, createObjectFile) {
  const uint8_t GOFFData[] = {
      0x03, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x40, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00,
  };
  ArrayRef<uint8_t> GOFFRef(GOFFData, sizeof(GOFFData));
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createObjectFile(
          MemoryBufferRef(toStringRef(GOFFRef), "dummyGOFF"),
          file_magic::goff_object);
  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST(GOFFObjectFileTest, ConstructGOFFObjectValidSize) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // END record.
  addEndRecord(GOFFData);

  constructValidGOFF(GOFFData.data(), GOFFData.size());
  constructValidGOFF(GOFFData.data(), 0);
}

TEST(GOFFObjectFileTest, ConstructGOFFObjectInvalidSize) {
  std::vector<char> GOFFData;
  GOFFData.resize(GOFF::RecordLength * 3);
  constructInvalidGOFF(GOFFData.data(), 70);
  constructInvalidGOFF(GOFFData.data(), 79);
  constructInvalidGOFF(GOFFData.data(), 81);
}

TEST(GOFFObjectFileTest, MissingHDR) {
  std::vector<char> GOFFData;

  // ESD record.
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;

  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("object file must start with HDR record"));
}

TEST(GOFFObjectFileTest, MissingEND) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // ESD record.
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr, FailedWithMessage("object file must end with END record"));
}

TEST(GOFFObjectFileTest, GetSymbolName) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // ESD record. Symbol name is Hello.
  addEsdRecord(GOFFData, 0x02, 0x01,
               {0xC8, // H
                0x85, // e
                0x93, // l
                0x93, // l
                0x96},// o
               0x01);

  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();

    EXPECT_EQ(SymbolName, "Hello");
  }
}

TEST(GOFFObjectFileTest, ConcatenatedGOFFFile) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);
  // ESD record.
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  // END record.
  addEndRecord(GOFFData);
  // HDR record.
  addHdrRecord(GOFFData);
  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST(GOFFObjectFileTest, ContinuationGetSymbolName) {
  std::vector<char> GOFFContData;

  // HDR record.
  addHdrRecord(GOFFContData);

  // ESD record with continuation. Symbol name is Helloworld.
  addEsdRecord(GOFFContData, 0x02, 0x01,
               {0xC8, // H
                0x85, // e
                0x93, // l
                0x93, // l
                0x96, // o
                0xA6, // w
                0x96, // o
                0x99, // r
                0x93, // l
                0x84},// d
               0x01);

  // END record.
  addEndRecord(GOFFContData);

  StringRef Data(GOFFContData.data(), GOFFContData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();
    EXPECT_EQ(SymbolName, "Helloworld");
  }
}

TEST(GOFFObjectFileTest, ContinuationBitNotSet) {
  std::vector<char> GOFFContData;

  // HDR record.
  addHdrRecord(GOFFContData);

  // ESD record.
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x01;
  GOFFContData[Pos + 3] = (char)0x02;
  GOFFContData[Pos + 7] = (char)0x01;
  GOFFContData[Pos + 11] = (char)0x01;
  GOFFContData[Pos + 71] = (char)0x0A; // Size of symbol name.
  GOFFContData[Pos + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFContData[Pos + 73] = (char)0x85;
  GOFFContData[Pos + 74] = (char)0x93;
  GOFFContData[Pos + 75] = (char)0x93;
  GOFFContData[Pos + 76] = (char)0x96;
  GOFFContData[Pos + 77] = (char)0xA6;
  GOFFContData[Pos + 78] = (char)0x96;
  GOFFContData[Pos + 79] = (char)0x99;

  // ESD continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x00;
  GOFFContData[Pos + 3] = (char)0x93;
  GOFFContData[Pos + 4] = (char)0x84;

  // END record.
  addEndRecord(GOFFContData);

  StringRef Data(GOFFContData.data(), GOFFContData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));
  EXPECT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is not a continuation record but the "
                        "preceding record is continued"));
}

TEST(GOFFObjectFileTest, ContinuationRecordNotTerminated) {
  std::vector<char> GOFFContData;

  // HDR record.
  addHdrRecord(GOFFContData);

  // ESD record.
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x01;
  GOFFContData[Pos + 3] = (char)0x02;
  GOFFContData[Pos + 7] = (char)0x01;
  GOFFContData[Pos + 11] = (char)0x01;
  GOFFContData[Pos + 71] = (char)0x0A; // Size of symbol name.
  GOFFContData[Pos + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFContData[Pos + 73] = (char)0x85;
  GOFFContData[Pos + 74] = (char)0x93;
  GOFFContData[Pos + 75] = (char)0x93;
  GOFFContData[Pos + 76] = (char)0x96;
  GOFFContData[Pos + 77] = (char)0xA6;
  GOFFContData[Pos + 78] = (char)0x96;
  GOFFContData[Pos + 79] = (char)0x99;

  // ESD continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x03; // Continued bit set.
  GOFFContData[Pos + 3] = (char)0x93;
  GOFFContData[Pos + 4] = (char)0x84;

  // END record.
  addEndRecord(GOFFContData);

  StringRef Data(GOFFContData.data(), GOFFContData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));
  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    EXPECT_THAT_EXPECTED(SymbolNameOrErr,
                         FailedWithMessage("continued bit should not be set"));
  }
}

TEST(GOFFObjectFileTest, PrevNotContinued) {
  std::vector<char> GOFFContData;

  // HDR record.
  addHdrRecord(GOFFContData);

  // ESD record, with continued bit not set.
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;

  // ESD continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x02;

  // END record.
  addEndRecord(GOFFContData);

  StringRef Data(GOFFContData.data(), GOFFContData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that is not "
                        "preceded by a continued record"));
}

TEST(GOFFObjectFileTest, ContinuationTypeMismatch) {
  std::vector<char> GOFFContData;

  // HDR record.
  addHdrRecord(GOFFContData);

  // ESD record.
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x01; // Continued to next record.

  // END continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x42;

  // END record.
  addEndRecord(GOFFContData);

  StringRef Data(GOFFContData.data(), GOFFContData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that does not match "
                        "the type of the previous record"));
}

TEST(GOFFObjectFileTest, TwoSymbols) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // ESD record 1. Symbol name is x.
  addEsdRecord(GOFFData, 0x00, 0x01,
               {0xa7}); // x

  // ESD record 2. Symbol name is Hello.
  addEsdRecord(GOFFData, 0x03, 0x02,
               {0xC8, // H
                0x85, // e
                0x93, // l
                0x93, // l
                0x96},// o
               0x01);

  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();
    EXPECT_EQ(SymbolName, "Hello");
  }
}

TEST(GOFFObjectFileTest, InvalidSymbolType) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // ESD record with invalid symbol type 0x05.
  addEsdRecord(GOFFData, 0x05, 0x01,
               {0xC8}, // H
               0x01);

  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<SymbolRef::Type> SymbolType = Symbol.getType();
    EXPECT_THAT_EXPECTED(
        SymbolType,
        FailedWithMessage("ESD record 1 has invalid symbol type 0x05"));

    Expected<section_iterator> SymSI = Symbol.getSection();
    ASSERT_THAT_EXPECTED(
        SymSI,
        FailedWithMessage(
            "symbol with ESD id 1 refers to invalid section with ESD id 1"));
  }
}

TEST(GOFFObjectFileTest, InvalidERSymbolType) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData);

  // ESD record.
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x04;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 63] = (char)0x03; // Unknown executable type.
  GOFFData[Pos + 71] = (char)0x01; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name.

  // END record.
  addEndRecord(GOFFData);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<SymbolRef::Type> SymbolType = Symbol.getType();
    EXPECT_THAT_EXPECTED(
        SymbolType,
        FailedWithMessage("ESD record 1 has unknown Executable type 0x03"));
  }
}

TEST(GOFFObjectFileTest, TXTConstruct) {
  std::vector<char> GOFFData;

  // HDR record.
  addHdrRecord(GOFFData, 0x01);

  // ESD record. Symbol name is var#c.
  addEsdRecord(GOFFData, 0x00, 0x01,
               {0xa5, // v
                0x81, // a
                0x99, // r
                0x7b, // #
                0x83});// c

  // ESD record. Symbol name is c_CoDE64.
  uint8_t BehavioralAttributes[] = {0x04, 0x04, 0x00, 0x0a,
                                     0x00, 0x00, 0x03, 0x00,
                                     0x00, 0x00};
  addEsdRecord(GOFFData, 0x01, 0x02,
               {0xc3, // c
                0x6d, // _
                0xc3, // c
                0xd6, // o
                0xc4, // D
                0xc5, // E
                0xf6, // 6
                0xf4},// 4
               0x01, 0x00, 0x01, 0x80, BehavioralAttributes, 0x08);

  // ESD record. Symbol name is var#c.
  addEsdRecord(GOFFData, 0x02, 0x03,
               {0xa5, // v
                0x81, // a
                0x99, // r
                0x7b, // #
                0x83},// c
               0x02);

  // TXT record.
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x10;
  GOFFData[Pos + 7] = (char)0x02;
  GOFFData[Pos + 23] = (char)0x08; // Data Length.
  GOFFData[Pos + 24] = (char)0x12;
  GOFFData[Pos + 25] = (char)0x34;
  GOFFData[Pos + 26] = (char)0x56;
  GOFFData[Pos + 27] = (char)0x78;
  GOFFData[Pos + 28] = (char)0x9a;
  GOFFData[Pos + 29] = (char)0xbc;
  GOFFData[Pos + 30] = (char)0xde;
  GOFFData[Pos + 31] = (char)0xf0;

  // END record.
  addEndRecord(GOFFData, 0x06);

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());
  auto Symbols = GOFFObj->symbols();
  ASSERT_EQ(std::distance(Symbols.begin(), Symbols.end()), 1);
  SymbolRef Symbol = *Symbols.begin();
  Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
  ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
  StringRef SymbolName = SymbolNameOrErr.get();
  EXPECT_EQ(SymbolName, "var#c");

  auto Sections = GOFFObj->sections();
  ASSERT_EQ(std::distance(Sections.begin(), Sections.end()), 1);
  SectionRef Section = *Sections.begin();
  Expected<StringRef> SectionContent = Section.getContents();
  ASSERT_THAT_EXPECTED(SectionContent, Succeeded());
  StringRef Contents = SectionContent.get();
  EXPECT_EQ(Contents, "\x12\x34\x56\x78\x9a\xbc\xde\xf0");
}
