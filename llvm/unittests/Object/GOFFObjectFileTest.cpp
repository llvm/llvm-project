//===- GOFFObjectFileTest.cpp - Tests for GOFFObjectFile ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::GOFF;

namespace {
struct TestData {
  std::vector<uint8_t> Data;

  TestData() = default;
  TestData(size_t Records) : Data(Records * GOFF::RecordLength, 0) {}
  MemoryBufferRef getMemoryBuffer() const {
    return MemoryBufferRef(toStringRef(Data), "dummyGOFF");
  }
  MutableArrayRef<uint8_t> operator[](size_t Index) {
    return MutableArrayRef(Data.data() + (GOFF::RecordLength * Index),
                           GOFF::RecordLength);
  }
};

void assertValidGOFF(TestData Data) {
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(Data.getMemoryBuffer());

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

void assertInvalidGOFF(TestData Data) {
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(Data.getMemoryBuffer());

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("object file is not the right size. Must be a multiple "
                        "of 80 bytes, but is " +
                        std::to_string(Data.Data.size()) + " bytes"));
}
} // namespace

TEST(GOFFObjectFileTest, ConstructGOFFObjectValidSize) {
  TestData Records(2);
  Records[0][0] = 0x03;
  Records[0][1] = 0xF0;
  Records[1][0] = 0x03;
  Records[1][1] = 0x40;
  assertValidGOFF(Records);
  Records.Data.resize(0);
  assertValidGOFF(Records);
}

TEST(GOFFObjectFileTest, ConstructGOFFObjectInvalidSize) {
  TestData Records;
  Records.Data.resize(70);
  assertInvalidGOFF(Records);
  Records.Data.resize(79);
  assertInvalidGOFF(Records);
  Records.Data.resize(81);
  assertInvalidGOFF(Records);
}

TEST(GOFFObjectFileTest, MissingHDR) {
  char GOFFData[GOFF::RecordLength * 2] = {0x00};

  // ESD record.
  GOFFData[0] = (char)0x03;

  // END record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 2);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("object file must start with HDR record"));
}

TEST(GOFFObjectFileTest, MissingEND) {
  char GOFFData[GOFF::RecordLength * 2] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;

  StringRef Data(GOFFData, GOFF::RecordLength * 2);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr, FailedWithMessage("object file must end with END record"));
}

TEST(GOFFObjectFileTest, GetSymbolName) {
  char GOFFData[GOFF::RecordLength * 3] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 3] = (char)0x02;
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFData[GOFF::RecordLength + 71] = (char)0x05; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name is Hello.
  GOFFData[GOFF::RecordLength + 73] = (char)0x85;
  GOFFData[GOFF::RecordLength + 74] = (char)0x93;
  GOFFData[GOFF::RecordLength + 75] = (char)0x93;
  GOFFData[GOFF::RecordLength + 76] = (char)0x96;

  // END record.
  GOFFData[GOFF::RecordLength * 2] = 0x03;
  GOFFData[GOFF::RecordLength * 2 + 1] = 0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 3);

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
  char GOFFData[GOFF::RecordLength * 6] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;
  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  // END record.
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 1] = (char)0x40;
  // HDR record.
  GOFFData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFData[GOFF::RecordLength * 3 + 1] = (char)0xF0;
  // ESD record.
  GOFFData[GOFF::RecordLength * 4] = (char)0x03;
  // END record.
  GOFFData[GOFF::RecordLength * 5] = (char)0x03;
  GOFFData[GOFF::RecordLength * 5 + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 6);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST(GOFFObjectFileTest, ContinuationGetSymbolName) {
  char GOFFContData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFContData[0] = (char)0x03;
  GOFFContData[1] = (char)0xF0;

  // ESD record.
  GOFFContData[GOFF::RecordLength] = (char)0x03;
  GOFFContData[GOFF::RecordLength + 1] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 3] = (char)0x02;
  GOFFContData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 71] = (char)0x0A; // Size of symbol name.
  GOFFContData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFContData[GOFF::RecordLength + 73] = (char)0x85;
  GOFFContData[GOFF::RecordLength + 74] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 75] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 76] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 77] = (char)0xA6;
  GOFFContData[GOFF::RecordLength + 78] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 79] = (char)0x99;

  // ESD continuation record.
  GOFFContData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 2 + 1] = (char)0x02; // No further continuations.
  GOFFContData[GOFF::RecordLength * 2 + 3] = (char)0x93;
  GOFFContData[GOFF::RecordLength * 2 + 4] = (char)0x84;

  // END record.
  GOFFContData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFContData, GOFF::RecordLength * 4);

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
  char GOFFContData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFContData[0] = (char)0x03;
  GOFFContData[1] = (char)0xF0;

  // ESD record.
  GOFFContData[GOFF::RecordLength] = (char)0x03;
  GOFFContData[GOFF::RecordLength + 1] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 3] = (char)0x02;
  GOFFContData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 71] = (char)0x0A; // Size of symbol name.
  GOFFContData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFContData[GOFF::RecordLength + 73] = (char)0x85;
  GOFFContData[GOFF::RecordLength + 74] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 75] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 76] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 77] = (char)0xA6;
  GOFFContData[GOFF::RecordLength + 78] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 79] = (char)0x99;

  // ESD continuation record.
  GOFFContData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 2 + 1] = (char)0x00;
  GOFFContData[GOFF::RecordLength * 2 + 3] = (char)0x93;
  GOFFContData[GOFF::RecordLength * 2 + 4] = (char)0x84;

  // END record.
  GOFFContData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFContData, GOFF::RecordLength * 4);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));
  EXPECT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is not a continuation record but the "
                        "preceding record is continued"));
}

TEST(GOFFObjectFileTest, ContinuationRecordNotTerminated) {
  char GOFFContData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFContData[0] = (char)0x03;
  GOFFContData[1] = (char)0xF0;

  // ESD record.
  GOFFContData[GOFF::RecordLength] = (char)0x03;
  GOFFContData[GOFF::RecordLength + 1] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 3] = (char)0x02;
  GOFFContData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFContData[GOFF::RecordLength + 71] = (char)0x0A; // Size of symbol name.
  GOFFContData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name is HelloWorld.
  GOFFContData[GOFF::RecordLength + 73] = (char)0x85;
  GOFFContData[GOFF::RecordLength + 74] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 75] = (char)0x93;
  GOFFContData[GOFF::RecordLength + 76] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 77] = (char)0xA6;
  GOFFContData[GOFF::RecordLength + 78] = (char)0x96;
  GOFFContData[GOFF::RecordLength + 79] = (char)0x99;

  // ESD continuation record.
  GOFFContData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 2 + 1] = (char)0x03; // Continued bit set.
  GOFFContData[GOFF::RecordLength * 2 + 3] = (char)0x93;
  GOFFContData[GOFF::RecordLength * 2 + 4] = (char)0x84;

  // END record.
  GOFFContData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFContData, GOFF::RecordLength * 4);

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
  char GOFFContData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFContData[0] = (char)0x03;
  GOFFContData[1] = (char)0xF0;

  // ESD record, with continued bit not set.
  GOFFContData[GOFF::RecordLength] = (char)0x03;

  // ESD continuation record.
  GOFFContData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 2 + 1] = (char)0x02;

  // END record.
  GOFFContData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFContData, GOFF::RecordLength * 4);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that is not "
                        "preceded by a continued record"));
}

TEST(GOFFObjectFileTest, ContinuationTypeMismatch) {
  char GOFFContData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFContData[0] = (char)0x03;
  GOFFContData[1] = (char)0xF0;

  // ESD record.
  GOFFContData[GOFF::RecordLength] = (char)0x03;
  GOFFContData[GOFF::RecordLength + 1] = (char)0x01; // Continued to next record.

  // END continuation record.
  GOFFContData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 2 + 1] = (char)0x42;

  // END record.
  GOFFContData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFContData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFContData, GOFF::RecordLength * 4);

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(
      GOFFObjOrErr,
      FailedWithMessage("record 2 is a continuation record that does not match "
                        "the type of the previous record"));
}

TEST(GOFFObjectFileTest, TwoSymbols) {
  char GOFFData[GOFF::RecordLength * 4] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record 1.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 3] = (char)0x00;
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;  // ESDID.
  GOFFData[GOFF::RecordLength + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xa7; // Symbol name is x.

  // ESD record 2.
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 3] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 7] = (char)0x02;  // ESDID.
  GOFFData[GOFF::RecordLength * 2 + 11] = (char)0x01; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 2 + 71] = (char)0x05; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 72] = (char)0xC8; // Symbol name is Hello.
  GOFFData[GOFF::RecordLength * 2 + 73] = (char)0x85;
  GOFFData[GOFF::RecordLength * 2 + 74] = (char)0x93;
  GOFFData[GOFF::RecordLength * 2 + 75] = (char)0x93;
  GOFFData[GOFF::RecordLength * 2 + 76] = (char)0x96;

  // END record.
  GOFFData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFData[GOFF::RecordLength * 3 + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 4);

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
  char GOFFData[GOFF::RecordLength * 3] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 3] = (char)0x05;
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFData[GOFF::RecordLength + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name.

  // END record.
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 3);

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
  char GOFFData[GOFF::RecordLength * 3] = {0x00};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;

  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 3] = (char)0x04;
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;
  GOFFData[GOFF::RecordLength + 11] = (char)0x01;
  GOFFData[GOFF::RecordLength + 63] = (char)0x03; // Unknown executable type.
  GOFFData[GOFF::RecordLength + 71] = (char)0x01; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xC8; // Symbol name.

  // END record.
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 1] = (char)0x40;

  StringRef Data(GOFFData, GOFF::RecordLength * 3);

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
  char GOFFData[GOFF::RecordLength * 6] = {};

  // HDR record.
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;
  GOFFData[50] = (char)0x01;

  // ESD record.
  GOFFData[GOFF::RecordLength] = (char)0x03;
  GOFFData[GOFF::RecordLength + 7] = (char)0x01;  // ESDID.
  GOFFData[GOFF::RecordLength + 71] = (char)0x05; // Size of symbol name.
  GOFFData[GOFF::RecordLength + 72] = (char)0xa5; // Symbol name is v.
  GOFFData[GOFF::RecordLength + 73] = (char)0x81; // Symbol name is a.
  GOFFData[GOFF::RecordLength + 74] = (char)0x99; // Symbol name is r.
  GOFFData[GOFF::RecordLength + 75] = (char)0x7b; // Symbol name is #.
  GOFFData[GOFF::RecordLength + 76] = (char)0x83; // Symbol name is c.

  // ESD record.
  GOFFData[GOFF::RecordLength * 2] = (char)0x03;
  GOFFData[GOFF::RecordLength * 2 + 3] = (char)0x01;
  GOFFData[GOFF::RecordLength * 2 + 7] = (char)0x02;  // ESDID.
  GOFFData[GOFF::RecordLength * 2 + 11] = (char)0x01; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 2 + 27] = (char)0x08; // Length.
  GOFFData[GOFF::RecordLength * 2 + 40] = (char)0x01; // Name Space ID.
  GOFFData[GOFF::RecordLength * 2 + 41] = (char)0x80;
  GOFFData[GOFF::RecordLength * 2 + 60] = (char)0x04; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 61] = (char)0x04; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 63] = (char)0x0a; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 66] = (char)0x03; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 71] = (char)0x08; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 2 + 72] = (char)0xc3; // Symbol name is c.
  GOFFData[GOFF::RecordLength * 2 + 73] = (char)0x6d; // Symbol name is _.
  GOFFData[GOFF::RecordLength * 2 + 74] = (char)0xc3; // Symbol name is c.
  GOFFData[GOFF::RecordLength * 2 + 75] = (char)0xd6; // Symbol name is o.
  GOFFData[GOFF::RecordLength * 2 + 76] = (char)0xc4; // Symbol name is D.
  GOFFData[GOFF::RecordLength * 2 + 77] = (char)0xc5; // Symbol name is E.
  GOFFData[GOFF::RecordLength * 2 + 78] = (char)0xf6; // Symbol name is 6.
  GOFFData[GOFF::RecordLength * 2 + 79] = (char)0xf4; // Symbol name is 4.

  // ESD record.
  GOFFData[GOFF::RecordLength * 3] = (char)0x03;
  GOFFData[GOFF::RecordLength * 3 + 3] = (char)0x02;
  GOFFData[GOFF::RecordLength * 3 + 7] = (char)0x03;  // ESDID.
  GOFFData[GOFF::RecordLength * 3 + 11] = (char)0x02; // Parent ESDID.
  GOFFData[GOFF::RecordLength * 3 + 71] = (char)0x05; // Size of symbol name.
  GOFFData[GOFF::RecordLength * 3 + 72] = (char)0xa5; // Symbol name is v.
  GOFFData[GOFF::RecordLength * 3 + 73] = (char)0x81; // Symbol name is a.
  GOFFData[GOFF::RecordLength * 3 + 74] = (char)0x99; // Symbol name is r.
  GOFFData[GOFF::RecordLength * 3 + 75] = (char)0x7b; // Symbol name is #.
  GOFFData[GOFF::RecordLength * 3 + 76] = (char)0x83; // Symbol name is c.

  // TXT record.
  GOFFData[GOFF::RecordLength * 4] = (char)0x03;
  GOFFData[GOFF::RecordLength * 4 + 1] = (char)0x10;
  GOFFData[GOFF::RecordLength * 4 + 7] = (char)0x02;
  GOFFData[GOFF::RecordLength * 4 + 23] = (char)0x08; // Data Length.
  GOFFData[GOFF::RecordLength * 4 + 24] = (char)0x12;
  GOFFData[GOFF::RecordLength * 4 + 25] = (char)0x34;
  GOFFData[GOFF::RecordLength * 4 + 26] = (char)0x56;
  GOFFData[GOFF::RecordLength * 4 + 27] = (char)0x78;
  GOFFData[GOFF::RecordLength * 4 + 28] = (char)0x9a;
  GOFFData[GOFF::RecordLength * 4 + 29] = (char)0xbc;
  GOFFData[GOFF::RecordLength * 4 + 30] = (char)0xde;
  GOFFData[GOFF::RecordLength * 4 + 31] = (char)0xf0;

  // END record.
  GOFFData[GOFF::RecordLength * 5] = (char)0x03;
  GOFFData[GOFF::RecordLength * 5 + 1] = (char)0x40;
  GOFFData[GOFF::RecordLength * 5 + 11] = (char)0x06;

  StringRef Data(GOFFData, GOFF::RecordLength * 6);

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

// Test elements with an offset as produced by HLASM.
TEST(GOFFObjectFileTest, EdWithOffset) {
  TestData Records(5);

  // HDR record.
  auto Hdr = Records[0];
  Hdr[0] = 0x03;
  Hdr[1] = 0xF0;
  Hdr[50] = 0x01;

  // SD ESD record.
  auto Sd = Records[1];
  Sd[0] = 0x03;
  Sd[3] = 0x00;  // SD symbol type.
  Sd[7] = 0x01;  // ESDID.
  Sd[71] = 0x04; // Symbol name length.
  Sd[72] = 0xa5; // Symbol name is T.
  Sd[73] = 0x81; // Symbol name is E.
  Sd[74] = 0x99; // Symbol name is S.
  Sd[75] = 0x7b; // Symbol name is T.

  // ED ESD record.
  auto Ed = Records[2];
  Ed[0] = 0x03;
  Ed[3] = 0x01;  // ED symbol type.
  Ed[7] = 0x02;  // ESDID.
  Ed[11] = 0x01; // Parent ESDID.
  Ed[18] = 0x01; // Offset is 4096.
  Ed[19] = 0x00; // Offset is 4096.
  Ed[27] = 0x08; // Element length is 8.
  Ed[40] = 0x01; // Name Space ID.
  Ed[41] = 0x80;
  Ed[60] = 0x04;  // Addressing mode is AMODE(64).
  Ed[61] = 0x03;  // Residence mode is AMODE(31).
  Ed[63] |= 0x60; // Tasking behaviour is RENT.
  Ed[63] |= 0x08; // Element is read-only.
  Ed[63] |= 0x02; // Element is executable.
  Ed[66] = 0x03;  // Element has doubleword alignment.
  Ed[71] = 0x06;  // Size of symbol name.
  Ed[72] = 0xc3;  // Symbol name is B.
  Ed[73] = 0x6d;  // Symbol name is _.
  Ed[74] = 0xc3;  // Symbol name is T.
  Ed[75] = 0xd6;  // Symbol name is E.
  Ed[76] = 0xc4;  // Symbol name is X.
  Ed[77] = 0xc5;  // Symbol name is T.

  // TXT record.
  auto Txt = Records[3];
  Txt[0] = 0x03;
  Txt[1] = 0x10;
  Txt[7] = 0x02;  // Owning ESDID is the ED.
  Txt[14] = 0x01; // Offset is 4096.
  Txt[15] = 0x00; // Offset is 4096.
  Txt[23] = 0x08; // Data Length.
  Txt[24] = 0x12;
  Txt[25] = 0x34;
  Txt[26] = 0x56;
  Txt[27] = 0x78;
  Txt[28] = 0x9a;
  Txt[29] = 0xbc;
  Txt[30] = 0xde;
  Txt[31] = 0xf0;

  // END record.
  auto End = Records[4];
  End[0] = 0x03;
  End[1] = 0x40;
  End[11] = 0x05;

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(Records.getMemoryBuffer());

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());

  GOFFObjectFile *GOFFObj = dyn_cast<GOFFObjectFile>((*GOFFObjOrErr).get());
  auto Symbols = GOFFObj->symbols();
  ASSERT_EQ(std::distance(Symbols.begin(), Symbols.end()), 0);

  auto Sections = GOFFObj->sections();
  ASSERT_EQ(std::distance(Sections.begin(), Sections.end()), 1);
  SectionRef Section = *Sections.begin();
  Expected<StringRef> SectionContent = Section.getContents();
  ASSERT_THAT_EXPECTED(SectionContent, Succeeded());
  StringRef Contents = SectionContent.get();
  EXPECT_EQ(Contents, "\x12\x34\x56\x78\x9a\xbc\xde\xf0");
}
