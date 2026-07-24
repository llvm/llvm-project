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

using namespace llvm;
using namespace llvm::object;
using namespace llvm::GOFF;

namespace {
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

class GOFFObjectFileTest : public ::testing::Test {
protected:
  std::vector<char> GOFFData;
  uint8_t RecordCount = 0;

  size_t addNewRecord() {
    size_t Pos = GOFFData.size();
    GOFFData.resize(Pos + GOFF::RecordLength);
    ++RecordCount;
    return Pos;
  }

  void addEndRecord() {
    size_t Pos = addNewRecord();
    GOFFData[Pos] = (char)0x03;
    GOFFData[Pos + 1] = (char)0x40; // END record, non-continued.
    GOFFData[Pos + 11] = (char)RecordCount;
  }

  void addHdrRecord(uint8_t ArchLevel = 0) {
    size_t Pos = addNewRecord();
    GOFFData[Pos] = (char)0x03;
    GOFFData[Pos + 1] = (char)0xF0; // HDR record, non-continued.
    GOFFData[Pos + 50] = (char)ArchLevel;
  }

  void addEsdRecord(uint8_t Type, uint8_t ESDID,
                    const std::vector<uint8_t> &Name, uint8_t ParentESDID = 0,
                    uint8_t BindingScope = 0, uint8_t NameSpaceID = 0,
                    uint8_t AdditionalFlags = 0,
                    std::array<uint8_t, 10> BehavioralAttributes = {},
                    uint32_t Length = 0) {
    size_t Pos = GOFFData.size();
    GOFFData.resize(GOFFData.size() + GOFF::RecordLength);
    ++RecordCount;

    GOFFData[Pos] = (char)0x03;
    GOFFData[Pos + 3] = (char)Type;
    GOFFData[Pos + 7] = (char)ESDID;           // ESDID.
    GOFFData[Pos + 11] = (char)ParentESDID;    // Parent ESDID.
    GOFFData[Pos + 24] = (char)(Length >> 24); // Length (big-endian).
    GOFFData[Pos + 25] = (char)(Length >> 16);
    GOFFData[Pos + 26] = (char)(Length >> 8);
    GOFFData[Pos + 27] = (char)(Length);
    GOFFData[Pos + 40] = (char)NameSpaceID;     // Name Space ID
    GOFFData[Pos + 41] = (char)AdditionalFlags; // Additional Flags

    for (size_t Offset = 0; Offset < 10; Offset++)
      GOFFData[Pos + 60 + Offset] = (char)BehavioralAttributes[Offset];

    GOFFData[Pos + 71] = (char)(Name.size()); // Size of symbol name.
    size_t StringOffset = Pos + 72;           // Start of Symbol name
    for (uint8_t C : Name) {
      GOFFData[StringOffset] = (char)C;
      ++StringOffset;

      if (StringOffset == Pos + GOFF::RecordLength) {
        // If we reach the end of the current record, we need to start a new
        // one.
        GOFFData[Pos + 1] |= 0x01; // Set continuation bit in the current
                                   // record.

        // start a new continuation record
        Pos = GOFFData.size();
        GOFFData.resize(GOFFData.size() + GOFF::RecordLength);
        ++RecordCount;
        GOFFData[Pos] = (char)0x03;
        GOFFData[Pos + 1] = (char)0x02; // continuation record

        StringOffset = Pos + 3;
      }
    }
  }
};
} // namespace

TEST_F(GOFFObjectFileTest, createObjectFile) {
  const uint8_t Data[] = {
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
  ArrayRef<uint8_t> GOFFRef(Data, sizeof(Data));
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createObjectFile(
          MemoryBufferRef(toStringRef(GOFFRef), "dummyGOFF"),
          file_magic::goff_object);
  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST_F(GOFFObjectFileTest, ConstructGOFFObjectValidSize) {
  // HDR record.
  addHdrRecord();

  // END record.
  addEndRecord();

  constructValidGOFF(GOFFData.data(), GOFFData.size());
  constructValidGOFF(GOFFData.data(), 0);
}

TEST_F(GOFFObjectFileTest, ConstructGOFFObjectInvalidSize) {
  GOFFData.resize(GOFF::RecordLength * 3);
  constructInvalidGOFF(GOFFData.data(), 70);
  constructInvalidGOFF(GOFFData.data(), 79);
  constructInvalidGOFF(GOFFData.data(), 81);
}

TEST_F(GOFFObjectFileTest, MissingHDR) {
  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("object file must start with HDR record"));
}

TEST_F(GOFFObjectFileTest, MissingEND) {
  // HDR record.
  addHdrRecord();

  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr, FailedWithMessage("object file must end with END record"));
}

TEST_F(GOFFObjectFileTest, GetSymbolName) {
  // HDR record.
  addHdrRecord();

  // ESD record. Symbol name is Hello.
  addEsdRecord(0x02, 0x01,
               {0xC8,  // H
                0x85,  // e
                0x93,  // l
                0x93,  // l
                0x96}, // o
               0x01);

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();

    EXPECT_EQ(SymbolName, "Hello");
  }
}

TEST_F(GOFFObjectFileTest, ConcatenatedGOFFFile) {
  // HDR record.
  addHdrRecord();
  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  // END record.
  addEndRecord();
  // HDR record.
  addHdrRecord();
  // ESD record.
  Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST_F(GOFFObjectFileTest, ContinuationGetSymbolName) {
  // HDR record.
  addHdrRecord();

  // ESD record with continuation. Symbol name is Helloworld.
  addEsdRecord(0x02, 0x01,
               {0xC8,  // H
                0x85,  // e
                0x93,  // l
                0x93,  // l
                0x96,  // o
                0xA6,  // w
                0x96,  // o
                0x99,  // r
                0x93,  // l
                0x84}, // d
               0x01);

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();
    EXPECT_EQ(SymbolName, "Helloworld");
  }
}

TEST_F(GOFFObjectFileTest, ContinuationBitNotSet) {
  // HDR record.
  addHdrRecord();

  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x01;
  GOFFData[Pos + 3] = (char)0x02;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 71] = (char)0x0A; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFData[Pos + 73] = (char)0x85;
  GOFFData[Pos + 74] = (char)0x93;
  GOFFData[Pos + 75] = (char)0x93;
  GOFFData[Pos + 76] = (char)0x96;
  GOFFData[Pos + 77] = (char)0xA6;
  GOFFData[Pos + 78] = (char)0x96;
  GOFFData[Pos + 79] = (char)0x99;

  // ESD continuation record.
  Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x00;
  GOFFData[Pos + 3] = (char)0x93;
  GOFFData[Pos + 4] = (char)0x84;

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));
  EXPECT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is not a continuation record but the "
                        "preceding record is continued"));
}

TEST_F(GOFFObjectFileTest, ContinuationRecordNotTerminated) {
  // HDR record.
  addHdrRecord();

  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x01;
  GOFFData[Pos + 3] = (char)0x02;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 71] = (char)0x0A; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFData[Pos + 73] = (char)0x85;
  GOFFData[Pos + 74] = (char)0x93;
  GOFFData[Pos + 75] = (char)0x93;
  GOFFData[Pos + 76] = (char)0x96;
  GOFFData[Pos + 77] = (char)0xA6;
  GOFFData[Pos + 78] = (char)0x96;
  GOFFData[Pos + 79] = (char)0x99;

  // ESD continuation record.
  Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x03; // Continued bit set.
  GOFFData[Pos + 3] = (char)0x93;
  GOFFData[Pos + 4] = (char)0x84;

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));
  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    EXPECT_THAT_EXPECTED(SymbolNameOrErr,
                         FailedWithMessage("continued bit should not be set"));
  }
}

TEST_F(GOFFObjectFileTest, PrevNotContinued) {
  // HDR record.
  addHdrRecord();

  // ESD record, with continued bit not set.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;

  // ESD continuation record.
  Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x02;

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that is not "
                        "preceded by a continued record"));
}

TEST_F(GOFFObjectFileTest, ContinuationTypeMismatch) {
  // HDR record.
  addHdrRecord();

  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x01; // Continued to next record.

  // END continuation record.
  Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x42;

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that does not match "
                        "the type of the previous record"));
}

TEST_F(GOFFObjectFileTest, TwoSymbols) {
  // HDR record.
  addHdrRecord();

  // ESD record 1. Symbol name is x.
  addEsdRecord(0x00, 0x01, {0xa7}); // x

  // ESD record 2. Symbol name is Hello.
  addEsdRecord(0x03, 0x02,
               {0xC8,  // H
                0x85,  // e
                0x93,  // l
                0x93,  // l
                0x96}, // o
               0x01);

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();
    EXPECT_EQ(SymbolName, "Hello");
  }
}

TEST_F(GOFFObjectFileTest, InvalidSymbolType) {
  // HDR record.
  addHdrRecord();

  // ESD record with invalid symbol type 0x05.
  addEsdRecord(0x05, 0x01, {0xC8}, // H
               0x01);

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

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

TEST_F(GOFFObjectFileTest, InvalidERSymbolType) {
  // HDR record.
  addHdrRecord();

  // ESD record.
  size_t Pos = addNewRecord();
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x04;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 63] = (char)0x03; // Unknown executable type.
  GOFFData[Pos + 71] = (char)0x01; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name.

  // END record.
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());

  for (SymbolRef Symbol : GOFFObj->symbols()) {
    Expected<SymbolRef::Type> SymbolType = Symbol.getType();
    EXPECT_THAT_EXPECTED(
        SymbolType,
        FailedWithMessage("ESD record 1 has unknown Executable type 0x03"));
  }
}

TEST_F(GOFFObjectFileTest, TXTConstruct) {
  // HDR record.
  addHdrRecord(0x01);

  // ESD record. Symbol name is var#c.
  addEsdRecord(0x00, 0x01,
               {0xa5,   // v
                0x81,   // a
                0x99,   // r
                0x7b,   // #
                0x83}); // c

  // ESD record. Symbol name is c_CoDE64.
  std::array<uint8_t, 10> BehavioralAttributes = {0x04, 0x04, 0x00, 0x0a, 0x00,
                                                  0x00, 0x03, 0x00, 0x00, 0x00};
  addEsdRecord(0x01, 0x02,
               {0xc3,  // c
                0x6d,  // _
                0xc3,  // c
                0xd6,  // o
                0xc4,  // D
                0xc5,  // E
                0xf6,  // 6
                0xf4}, // 4
               0x01, 0x00, 0x01, 0x80, BehavioralAttributes, 0x08);

  // ESD record. Symbol name is var#c.
  addEsdRecord(0x02, 0x03,
               {0xa5,  // v
                0x81,  // a
                0x99,  // r
                0x7b,  // #
                0x83}, // c
               0x02);

  // TXT record.
  size_t Pos = addNewRecord();
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
  addEndRecord();

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj =
      static_cast<GOFFObjectFile *>((*GOFFObjOrErr).get());
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

TEST(GOFFObjectFileTest, GlobalSymbols) {
  char GOFFData[GOFF::RecordLength * 12] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record 1: type SD
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 3] = (char)0x00;  // Type: SD
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;  // ESDID.
  GOFFData[GOFF::RecordLength + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xC1; // Symbol name is A.

  // ESD record 2: type ED
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 3] = (char)0x01;  // Type: ED
  GOFFData[GOFF::RecordLength * 2 + 7] = (char)0x02;  // ESDID.
  GOFFData[GOFF::RecordLength * 2 + 11] = (char)0x01; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 2 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 72] = (char)0xC2; // Symbol name is B.

  // ESD record 3: type LD
  GOFFData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFData[GOFF::RecordLength * 3 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 3 + 7] = (char)0x03;  // ESDID.
  GOFFData[GOFF::RecordLength * 3 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 3 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 3 + 72] = (char)0xC3; // Symbol name is C.

  // ESD record 4: type PR
  GOFFData[GOFF::RecordLength * 4] = (char)0x03;
  GOFFData[GOFF::RecordLength * 4 + 3] = (char)0x03;  // Type: PR
  GOFFData[GOFF::RecordLength * 4 + 7] = (char)0x04;  // ESDID.
  GOFFData[GOFF::RecordLength * 4 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 4 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 4 + 72] = (char)0xC4; // Symbol name is D.

  // ESD record 5: type ErWx
  GOFFData[GOFF::RecordLength * 5] = (char)0x03;
  GOFFData[GOFF::RecordLength * 5 + 3] = (char)0x04;  // Type: ErWx
  GOFFData[GOFF::RecordLength * 5 + 7] = (char)0x05;  // ESDID.
  GOFFData[GOFF::RecordLength * 5 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 5 + 72] = (char)0xC5; // Symbol name is E.

  // ESD record 6: type LD + Section binding scope
  GOFFData[GOFF::RecordLength * 6] = (char)0x03;
  GOFFData[GOFF::RecordLength * 6 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 6 + 7] = (char)0x06;  // ESDID.
  GOFFData[GOFF::RecordLength * 6 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 6 + 65] = (char)0x01; // Binding Scope: Section.
  GOFFData[GOFF::RecordLength * 6 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 6 + 72] = (char)0xC6; // Symbol name is F.

  // ESD record 7: type LD + Module binding scope
  GOFFData[GOFF::RecordLength * 7] = (char)0x03;
  GOFFData[GOFF::RecordLength * 7 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 7 + 7] = (char)0x07;  // ESDID.
  GOFFData[GOFF::RecordLength * 7 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 7 + 65] = (char)0x02; // Binding Scope: Module.
  GOFFData[GOFF::RecordLength * 7 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 7 + 72] = (char)0xC7; // Symbol name is G.

  // ESD record 8: type LD + Library binding scope
  GOFFData[GOFF::RecordLength * 8] = (char)0x03;
  GOFFData[GOFF::RecordLength * 8 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 8 + 7] = (char)0x08;  // ESDID.
  GOFFData[GOFF::RecordLength * 8 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 8 + 65] = (char)0x03; // Binding Scope: Library.
  GOFFData[GOFF::RecordLength * 8 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 8 + 72] = (char)0xC8; // Symbol name is H.

  // ESD record 9: type LD + Import-Export binding scope
  GOFFData[GOFF::RecordLength * 9] = (char)0x03;
  GOFFData[GOFF::RecordLength * 9 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 9 + 7] = (char)0x09;  // ESDID.
  GOFFData[GOFF::RecordLength * 9 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 9 + 65] =
      (char)0x04; // Binding Scope: ImportExport.
  GOFFData[GOFF::RecordLength * 9 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 9 + 72] = (char)0xC9; // Symbol name is I.

  // ESD record 10: type LD + blank name
  GOFFData[GOFF::RecordLength * 10] = (char)0x03;
  GOFFData[GOFF::RecordLength * 10 + 3] = (char)0x02;  // Type: LD
  GOFFData[GOFF::RecordLength * 10 + 7] = (char)0x0A;  // ESDID.
  GOFFData[GOFF::RecordLength * 10 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 10 + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 10 + 72] = (char)0x40; // Symbol name is ' '.

  // END record.
  GOFFData[GOFF::RecordLength * 11] = (char)0x03;
  GOFFData[GOFF::RecordLength * 11 + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 12);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());

  auto SymbolRange = GOFFObj->symbols();
  auto Symbol = SymbolRange.begin();
  auto ValidateGlobal = [&](StringRef Name, bool IsGlobal) {
    ASSERT_TRUE(Symbol != SymbolRange.end());

    // Check Name.
    Expected<StringRef> SymbolNameOrErr = GOFFObj->getSymbolName(*Symbol);
    ASSERT_THAT_EXPECTED(SymbolNameOrErr, Succeeded());
    StringRef SymbolName = SymbolNameOrErr.get();
    EXPECT_EQ(SymbolName, Name);

    // Check flags.
    Expected<uint32_t> SymbolFlagsOrErr = Symbol->getFlags();
    ASSERT_THAT_EXPECTED(SymbolFlagsOrErr, Succeeded());
    uint32_t SymbolFlags = SymbolFlagsOrErr.get();
    if (IsGlobal) {
      EXPECT_TRUE(SymbolFlags & SymbolRef::SF_Global);
    } else {
      EXPECT_FALSE(SymbolFlags & SymbolRef::SF_Global);
    }

    ++Symbol;
  };

  // ESD records 'A' and 'B' shouldn't be considered symbols.
  ValidateGlobal("C", true);
  ValidateGlobal("D", true);
  ValidateGlobal("E", true);
  ValidateGlobal("F", false);
  ValidateGlobal("G", false);
  ValidateGlobal("H", true);
  ValidateGlobal("I", true);
  ValidateGlobal(" ", false);
}
