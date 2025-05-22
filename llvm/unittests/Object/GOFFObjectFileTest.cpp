//===- GOFFObjectFileTest.cpp - Tests for GOFFObjectFile ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::GOFF;

namespace {
char GOFFData[GOFF::RecordLength * 3] = {0x00};

void constructValidGOFF(size_t Size) {
  StringRef ValidSize(GOFFData, Size);
  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(ValidSize, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

void constructInvalidGOFF(size_t Size) {
  // Construct GOFFObject with record of length != multiple of 80.
  StringRef InvalidData(GOFFData, Size);
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

TEST(GOFFObjectFileTest, ConstructGOFFObjectValidSize) {
  GOFFData[0] = (char)0x03;
  GOFFData[1] = (char)0xF0;
  GOFFData[80] = (char)0x03;
  GOFFData[81] = (char)0x40;
  constructValidGOFF(160);
  constructValidGOFF(0);
}

TEST(GOFFObjectFileTest, ConstructGOFFObjectInvalidSize) {
  constructInvalidGOFF(70);
  constructInvalidGOFF(79);
  constructInvalidGOFF(81);
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
