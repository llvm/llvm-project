// llvm/unittest/Support/SYCLPropertySetIOTest.cpp - SYCL Property set I/O tests
// //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SYCLPropertySetIO.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::util;

namespace {

TEST(SYCLPropertySet, IncorrectValuesReadIO) {
  auto Content = "Staff/Ages]\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("property category missing")));

  Content = "[Staff/Ages\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()),
                    FailedWithMessage(testing::HasSubstr("invalid line")));

  Content = "[Staff/Ages]\n"
            "person1=\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("invalid property line")));

  Content = "[Staff/Ages]\n"
            "person1=|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("invalid property value")));

  Content = "[Staff/Ages]\n"
            "person1=abc|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("invalid property type")));

  Content = "[Staff/Ages]\n"
            "person1=1|IAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("invalid property value")));

  Content = "[Staff/Ages]\n"
            "person1=2|IAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr(
          "Base64 encoded strings must be a multiple of 4 bytes in length")));

  Content = "[Staff/Ages]\n"
            "person1=100|10\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(std::move(PropSetsPtr.takeError()),
                    FailedWithMessage(testing::HasSubstr("bad property type")));

  Content = "[Opt/Param]\n"
            "kernel1=2|IAAAAAAAAAQA\tIAAAAAAAAAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("Invalid Base64 character")));

  Content = "[Opt/Param]\n"
            "kernel1=2|IAAAAAAAAAQA\nIAAAAAAAAAQ\n";
  MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());
  EXPECT_THAT_ERROR(
      std::move(PropSetsPtr.takeError()),
      FailedWithMessage(testing::HasSubstr("invalid property line")));
}

TEST(SYCLPropertySet, DuplicateKeysRead) {
  // '1' in '1|20' means 'integer property'
  auto Content = "[Staff/Ages]\n"
                 "person1=1|20\n"
                 "person1=1|25\n";
  // For duplicate keys, the latest key is selected
  auto ExpectedContent = "[Staff/Ages]\n"
                         "person1=1|25\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "SYCLPropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  EXPECT_EQ(Serialized, ExpectedContent);
}

// Test read-then-write of a list of integer values
TEST(SYCLPropertySet, IntValuesIO) {
  // '1' in '1|20' means 'integer property'
  auto Content = "[Staff/Ages]\n"
                 "person1=1|20\n"
                 "person2=1|25\n"
                 "[Staff/Experience]\n"
                 "person1=1|1\n"
                 "person2=1|2\n"
                 "person3=1|12\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "SYCLPropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  EXPECT_EQ(Serialized, Content);
}

// Test read-then-write of a list of byte arrays
TEST(SYCLPropertySet, ByteArrayValuesIO) {
  // '2' in '2|...' means 'byte array property', Base64-encoded
  // encodes the following byte arrays:
  //   { 8, 0, 0, 0, 0, 0, 0, 0, 0x1 };
  //   { 40, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0x7F, 0xFF, 0x70 };
  // first 8 bytes are the size in bits (40) of what follows (5 bytes).

  auto Content = "[Opt/Param]\n"
                 "kernel1=2|IAAAAAAAAAQA\n"
                 "kernel2=2|oAAAAAAAAAAAw///3/wB\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto PropSetsPtr = SYCLPropertySetRegistry::read(MemBuf.get());

  if (!PropSetsPtr)
    FAIL() << "SYCLPropertySetRegistry::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSetsPtr->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  EXPECT_EQ(Serialized, Content);
}

// Test write-then-read of boolean values
TEST(SYCLPropertySet, Mask) {
  SYCLPropertySetRegistry PropSet;
  uint32_t Mask = 0;
  std::map<StringRef, uint32_t> DevMask = {{"Mask", Mask}};
  PropSet.add("MaskCategory", DevMask);
  std::string Serialized;
  std::string Expected("[MaskCategory]\nMask=1|0\n");
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSet.write(OS);
  }
  llvm::errs() << Serialized << "\n";
  EXPECT_EQ(Serialized, Expected);
}

// Test write-then-read of std::map<StringRef, SYCLPropertyValue>
// SYClPropertyValue is a class which contains one member, which is
// std::variant<uint32_t, std::unique_ptr<std::byte, Deleter>> Val;
TEST(SYCLPropertySet, SYCLPropertyValues) {
  std::map<StringRef, SYCLPropertyValue> PropValues;
  std::vector<uint32_t> Values = {1, 2, 3, 4};
  uint32_t Size = 4;
  PropValues["Values"] = std::move(Values);
  PropValues["Size"] = Size;
  SYCLPropertySetRegistry PropSet;
  PropSet.add("Property Values", PropValues);
  std::string Serialized;
  std::string Expected(
      "[Property Values]\nSize=1|4\nValues=2|AQAAAAIAAAADAAAABAAAAA==\n");
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSet.write(OS);
  }
  llvm::errs() << Serialized << "\n";
  EXPECT_EQ(Serialized, Expected);
}

// Test write-then-read of MapVector of StringRef to a simple struct datatype.
// Example of simple data structure:
// struct SimpleDS {
//   unsigned ID;
//   unsigned Offset;
//   unsigned Size;
// };
struct SimpleDS {
  unsigned ID;
  unsigned Offset;
  unsigned Size;
};
using MapTy = MapVector<StringRef, std::vector<SimpleDS>>;
TEST(SYCLPropertySet, MapToStruct) {
  MapTy Values;
  std::vector<SimpleDS> SimpleDSVal(2);
  unsigned Start = 0;
  for (unsigned I = Start; I < Start + 2; ++I) {
    SimpleDSVal[I - Start].ID = I;
    SimpleDSVal[I - Start].Offset = I * 4;
    SimpleDSVal[I - Start].Size = 4;
  }
  Values["Value0"] = SimpleDSVal;
  Start += 8;
  for (unsigned I = Start; I < Start + 2; ++I) {
    SimpleDSVal[I - Start].ID = I;
    SimpleDSVal[I - Start].Offset = I * 4;
    SimpleDSVal[I - Start].Size = 4;
  }
  Values["Value1"] = SimpleDSVal;
  SYCLPropertySetRegistry PropSet;
  PropSet.add("Values", Values);
  std::string Serialized;
  std::string Expected(
      "[Values]\nValue0=2|AAAAAAAAAAAEAAAAAQAAAAQAAAAEAAAA\nValue1=2|"
      "CAAAACAAAAAEAAAACQAAACQAAAAEAAAA\n");
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSet.write(OS);
  }
  llvm::errs() << Serialized << "\n";
  EXPECT_EQ(Serialized, Expected);
}

// Test write-then-read of vector of chars
TEST(SYCLPropertySet, VectorOfChars) {
  std::vector<char> Values;
  for (unsigned I = 0; I < 8; ++I)
    Values.push_back((char)(I + '0'));
  SYCLPropertySetRegistry PropSet;
  PropSet.add("Values", "all", Values);
  std::string Serialized;
  std::string Expected("[Values]\nall=2|MDEyMzQ1Njc=\n");
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSet.write(OS);
  }
  llvm::errs() << Serialized << "\n";
  EXPECT_EQ(Serialized, Expected);
}

// Test write-then-read of a list of Name-Value pairs
TEST(SYCLPropertySet, ListOfNameValuePairs) {
  SYCLPropertySetRegistry PropSet;
  std::vector<std::string> Names = {"Name0", "Name1", "Name2", "Name3"};
  for (unsigned I = 0; I < 4; ++I) {
    auto Value = I * 8;
    PropSet.add("Values", Names[I], Value);
  }
  std::string Serialized;
  std::string Expected(
      "[Values]\nName0=1|0\nName1=1|8\nName2=1|16\nName3=1|24\n");
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    PropSet.write(OS);
  }
  llvm::errs() << Serialized << "\n";
  EXPECT_EQ(Serialized, Expected);
}
} // namespace
