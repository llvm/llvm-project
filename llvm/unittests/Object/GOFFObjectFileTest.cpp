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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

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
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFData);
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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x02;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 71] = (char)0x05; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name is Hello.
  GOFFData[Pos + 73] = (char)0x85;
  GOFFData[Pos + 74] = (char)0x93;
  GOFFData[Pos + 75] = (char)0x93;
  GOFFData[Pos + 76] = (char)0x96;

  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = 0x03;
  GOFFData[Pos + 1] = 0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;
  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;
  // HDR record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;
  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

  StringRef Data(GOFFData.data(), GOFFData.size());

  Expected<std::unique_ptr<ObjectFile>> GOFFObjOrErr =
      object::ObjectFile::createGOFFObjectFile(
          MemoryBufferRef(Data, "dummyGOFF"));

  ASSERT_THAT_EXPECTED(GOFFObjOrErr, Succeeded());
}

TEST(GOFFObjectFileTest, ContinuationGetSymbolName) {
  std::vector<char> GOFFContData;

  // HDR record.
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFContData);
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
  GOFFContData[Pos + 1] = (char)0x02; // No further continuations.
  GOFFContData[Pos + 3] = (char)0x93;
  GOFFContData[Pos + 4] = (char)0x84;

  // END record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFContData);
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
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFContData);
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
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0xF0;

  // ESD record, with continued bit not set.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;

  // ESD continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x02;

  // END record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x01; // Continued to next record.

  // END continuation record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x42;

  // END record.
  Pos = newRecord(GOFFContData);
  GOFFContData[Pos] = (char)0x03;
  GOFFContData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // ESD record 1.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x00;
  GOFFData[Pos + 7] = (char)0x01;  // ESDID.
  GOFFData[Pos + 71] = (char)0x01; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xa7; // Symbol name is x.

  // ESD record 2.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x03;
  GOFFData[Pos + 7] = (char)0x02;  // ESDID.
  GOFFData[Pos + 11] = (char)0x01; // Parent ESDID.
  GOFFData[Pos + 71] = (char)0x05; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name is Hello.
  GOFFData[Pos + 73] = (char)0x85;
  GOFFData[Pos + 74] = (char)0x93;
  GOFFData[Pos + 75] = (char)0x93;
  GOFFData[Pos + 76] = (char)0x96;

  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x05;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 71] = (char)0x01; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name.

  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x04;
  GOFFData[Pos + 7] = (char)0x01;
  GOFFData[Pos + 11] = (char)0x01;
  GOFFData[Pos + 63] = (char)0x03; // Unknown executable type.
  GOFFData[Pos + 71] = (char)0x01; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xC8; // Symbol name.

  // END record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;

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
  size_t Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0xF0;
  GOFFData[Pos + 50] = (char)0x01;

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 7] = (char)0x01;  // ESDID.
  GOFFData[Pos + 71] = (char)0x05; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xa5; // Symbol name is v.
  GOFFData[Pos + 73] = (char)0x81; // Symbol name is a.
  GOFFData[Pos + 74] = (char)0x99; // Symbol name is r.
  GOFFData[Pos + 75] = (char)0x7b; // Symbol name is #.
  GOFFData[Pos + 76] = (char)0x83; // Symbol name is c.

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x01;
  GOFFData[Pos + 7] = (char)0x02;  // ESDID.
  GOFFData[Pos + 11] = (char)0x01; // Parent ESDID.
  GOFFData[Pos + 27] = (char)0x08; // Length.
  GOFFData[Pos + 40] = (char)0x01; // Name Space ID.
  GOFFData[Pos + 41] = (char)0x80;
  GOFFData[Pos + 60] = (char)0x04; // Size of symbol name.
  GOFFData[Pos + 61] = (char)0x04; // Size of symbol name.
  GOFFData[Pos + 63] = (char)0x0a; // Size of symbol name.
  GOFFData[Pos + 66] = (char)0x03; // Size of symbol name.
  GOFFData[Pos + 71] = (char)0x08; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xc3; // Symbol name is c.
  GOFFData[Pos + 73] = (char)0x6d; // Symbol name is _.
  GOFFData[Pos + 74] = (char)0xc3; // Symbol name is c.
  GOFFData[Pos + 75] = (char)0xd6; // Symbol name is o.
  GOFFData[Pos + 76] = (char)0xc4; // Symbol name is D.
  GOFFData[Pos + 77] = (char)0xc5; // Symbol name is E.
  GOFFData[Pos + 78] = (char)0xf6; // Symbol name is 6.
  GOFFData[Pos + 79] = (char)0xf4; // Symbol name is 4.

  // ESD record.
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 3] = (char)0x02;
  GOFFData[Pos + 7] = (char)0x03;  // ESDID.
  GOFFData[Pos + 11] = (char)0x02; // Parent ESDID.
  GOFFData[Pos + 71] = (char)0x05; // Size of symbol name.
  GOFFData[Pos + 72] = (char)0xa5; // Symbol name is v.
  GOFFData[Pos + 73] = (char)0x81; // Symbol name is a.
  GOFFData[Pos + 74] = (char)0x99; // Symbol name is r.
  GOFFData[Pos + 75] = (char)0x7b; // Symbol name is #.
  GOFFData[Pos + 76] = (char)0x83; // Symbol name is c.

  // TXT record.
  Pos = newRecord(GOFFData);
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
  Pos = newRecord(GOFFData);
  GOFFData[Pos] = (char)0x03;
  GOFFData[Pos + 1] = (char)0x40;
  GOFFData[Pos + 11] = (char)0x06;

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
