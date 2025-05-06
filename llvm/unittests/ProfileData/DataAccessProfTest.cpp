
//===- unittests/Support/DataAccessProfTest.cpp
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock-more-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace data_access_prof {
namespace {

using ::llvm::StringRef;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

static std::string ErrorToString(Error E) {
  std::string ErrMsg;
  llvm::raw_string_ostream OS(ErrMsg);
  llvm::logAllUnhandledErrors(std::move(E), OS);
  return ErrMsg;
}

// Test the various scenarios when DataAccessProfData should return error on
// invalid input.
TEST(MemProf, DataAccessProfileError) {
  // Returns error if the input symbol name is empty.
  DataAccessProfData Data;
  EXPECT_THAT(ErrorToString(Data.setDataAccessProfile("", 100)),
              HasSubstr("Empty symbol name"));

  // Returns error when the same symbol gets added twice.
  ASSERT_FALSE(Data.setDataAccessProfile("foo", 100));
  EXPECT_THAT(ErrorToString(Data.setDataAccessProfile("foo", 100)),
              HasSubstr("Duplicate symbol or string literal added"));

  // Returns error when the same string content hash gets added twice.
  ASSERT_FALSE(Data.setDataAccessProfile((uint64_t)135246, 1000));
  EXPECT_THAT(ErrorToString(Data.setDataAccessProfile((uint64_t)135246, 1000)),
              HasSubstr("Duplicate symbol or string literal added"));
}

// Test the following operations on DataAccessProfData:
// - Profile record look up.
// - Serialization and de-serialization.
TEST(MemProf, DataAccessProfile) {
  DataAccessProfData Data;

  // In the bool conversion, Error is true if it's in a failure state and false
  // if it's in an accept state. Use ASSERT_FALSE or EXPECT_FALSE for no error.
  ASSERT_FALSE(Data.setDataAccessProfile("foo.llvm.123", 100));
  ASSERT_FALSE(Data.addKnownSymbolWithoutSamples((uint64_t)789));
  ASSERT_FALSE(Data.addKnownSymbolWithoutSamples("sym2"));
  ASSERT_FALSE(Data.setDataAccessProfile("bar.__uniq.321", 123,
                                         {
                                             DataLocation{"file2", 3},
                                         }));
  ASSERT_FALSE(Data.addKnownSymbolWithoutSamples("sym1"));
  ASSERT_FALSE(Data.addKnownSymbolWithoutSamples((uint64_t)678));
  ASSERT_FALSE(Data.setDataAccessProfile(
      (uint64_t)135246, 1000,
      {DataLocation{"file1", 1}, DataLocation{"file2", 2}}));

  {
    // Test that symbol names and file names are stored in the input order.
    EXPECT_THAT(llvm::to_vector(Data.getStrings()),
                ElementsAre("foo", "bar.__uniq.321", "file2", "file1"));
    EXPECT_THAT(Data.getKnownColdSymbols(), ElementsAre("sym2", "sym1"));
    EXPECT_THAT(Data.getKnownColdHashes(), ElementsAre(789, 678));

    // Look up profiles.
    EXPECT_TRUE(Data.isKnownColdSymbol((uint64_t)789));
    EXPECT_TRUE(Data.isKnownColdSymbol((uint64_t)678));
    EXPECT_TRUE(Data.isKnownColdSymbol("sym2"));
    EXPECT_TRUE(Data.isKnownColdSymbol("sym1"));

    EXPECT_EQ(Data.getProfileRecord("non-existence"), nullptr);
    EXPECT_EQ(Data.getProfileRecord((uint64_t)789987), nullptr);

    EXPECT_THAT(
        *Data.getProfileRecord("foo.llvm.123"),
        AllOf(testing::Field(&DataAccessProfRecord::SymbolID, 0),
              testing::Field(&DataAccessProfRecord::AccessCount, 100),
              testing::Field(&DataAccessProfRecord::IsStringLiteral, false),
              testing::Field(&DataAccessProfRecord::Locations,
                             testing::IsEmpty())));
    EXPECT_THAT(
        *Data.getProfileRecord("bar.__uniq.321"),
        AllOf(
            testing::Field(&DataAccessProfRecord::SymbolID, 1),
            testing::Field(&DataAccessProfRecord::AccessCount, 123),
            testing::Field(&DataAccessProfRecord::IsStringLiteral, false),
            testing::Field(&DataAccessProfRecord::Locations,
                           ElementsAre(AllOf(
                               testing::Field(&DataLocation::FileName, "file2"),
                               testing::Field(&DataLocation::Line, 3))))));
    EXPECT_THAT(
        *Data.getProfileRecord((uint64_t)135246),
        AllOf(testing::Field(&DataAccessProfRecord::SymbolID, 135246),
              testing::Field(&DataAccessProfRecord::AccessCount, 1000),
              testing::Field(&DataAccessProfRecord::IsStringLiteral, true),
              testing::Field(
                  &DataAccessProfRecord::Locations,
                  ElementsAre(
                      AllOf(testing::Field(&DataLocation::FileName, "file1"),
                            testing::Field(&DataLocation::Line, 1)),
                      AllOf(testing::Field(&DataLocation::FileName, "file2"),
                            testing::Field(&DataLocation::Line, 2))))));
  }

  // Tests serialization and de-serialization.
  DataAccessProfData deserializedData;
  {
    std::string serializedData;
    llvm::raw_string_ostream OS(serializedData);
    llvm::ProfOStream POS(OS);

    EXPECT_FALSE(Data.serialize(POS));

    const unsigned char *p =
        reinterpret_cast<const unsigned char *>(serializedData.data());
    ASSERT_THAT(llvm::to_vector(deserializedData.getStrings()),
                testing::IsEmpty());
    EXPECT_FALSE(deserializedData.deserialize(p));

    EXPECT_THAT(llvm::to_vector(deserializedData.getStrings()),
                ElementsAre("foo", "bar.__uniq.321", "file2", "file1"));
    EXPECT_THAT(deserializedData.getKnownColdSymbols(),
                ElementsAre("sym2", "sym1"));
    EXPECT_THAT(deserializedData.getKnownColdHashes(), ElementsAre(789, 678));

    // Look up profiles after deserialization.
    EXPECT_TRUE(deserializedData.isKnownColdSymbol((uint64_t)789));
    EXPECT_TRUE(deserializedData.isKnownColdSymbol((uint64_t)678));
    EXPECT_TRUE(deserializedData.isKnownColdSymbol("sym2"));
    EXPECT_TRUE(deserializedData.isKnownColdSymbol("sym1"));

    auto Records =
        llvm::to_vector(llvm::make_second_range(deserializedData.getRecords()));

    EXPECT_THAT(
        Records,
        ElementsAre(
            AllOf(testing::Field(&DataAccessProfRecord::SymbolID, 0),
                  testing::Field(&DataAccessProfRecord::AccessCount, 100),
                  testing::Field(&DataAccessProfRecord::IsStringLiteral, false),
                  testing::Field(&DataAccessProfRecord::Locations,
                                 testing::IsEmpty())),
            AllOf(testing::Field(&DataAccessProfRecord::SymbolID, 1),
                  testing::Field(&DataAccessProfRecord::AccessCount, 123),
                  testing::Field(&DataAccessProfRecord::IsStringLiteral, false),
                  testing::Field(
                      &DataAccessProfRecord::Locations,
                      ElementsAre(AllOf(
                          testing::Field(&DataLocation::FileName, "file2"),
                          testing::Field(&DataLocation::Line, 3))))),
            AllOf(
                testing::Field(&DataAccessProfRecord::SymbolID, 135246),
                testing::Field(&DataAccessProfRecord::AccessCount, 1000),
                testing::Field(&DataAccessProfRecord::IsStringLiteral, true),
                testing::Field(
                    &DataAccessProfRecord::Locations,
                    ElementsAre(
                        AllOf(testing::Field(&DataLocation::FileName, "file1"),
                              testing::Field(&DataLocation::Line, 1)),
                        AllOf(testing::Field(&DataLocation::FileName, "file2"),
                              testing::Field(&DataLocation::Line, 2)))))));
  }
}
} // namespace
} // namespace data_access_prof
} // namespace llvm
