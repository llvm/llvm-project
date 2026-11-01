//===- SimpleSymbolTableTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's SimpleSymbolTable.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SimpleSymbolTable.h"

#include "ErrorMatchers.h"
#include "gtest/gtest.h"

#include <set>
#include <string>

using namespace orc_rt;
using namespace orc_rt::test;

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Property;

TEST(SimpleSymbolTableTest, EmptyByDefault) {
  SimpleSymbolTable ST;
  EXPECT_TRUE(ST.empty());
  EXPECT_EQ(ST.size(), 0U);
  EXPECT_EQ(ST.begin(), ST.end());
}

TEST(SimpleSymbolTableTest, AddSymbolsUnique) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X},
      {SymbolNameSpec::linker("orc_rt_B"), &Y}};

  EXPECT_THAT_ERROR(ST.addUnique(Syms), Succeeded())
      << "Unexpected error adding unique symbols";

  EXPECT_EQ(ST.size(), 2U);
  EXPECT_FALSE(ST.empty());
  EXPECT_TRUE(ST.count(SymbolNameSpec::linker("orc_rt_A")));
  EXPECT_TRUE(ST.count(SymbolNameSpec::linker("orc_rt_B")));
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_B")), &Y);
}

TEST(SimpleSymbolTableTest, AddConstPointers) {
  SimpleSymbolTable ST;
  const int X = 42;
  const int Y = 7;
  std::pair<SymbolNameSpec, const void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X},
      {SymbolNameSpec::linker("orc_rt_B"), &Y}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_B")), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueMultipleCalls) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;

  std::pair<SymbolNameSpec, void *> First[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X}};
  std::pair<SymbolNameSpec, void *> Second[] = {
      {SymbolNameSpec::linker("orc_rt_B"), &Y}};

  ASSERT_THAT_ERROR(ST.addUnique(First), Succeeded());
  ASSERT_THAT_ERROR(ST.addUnique(Second), Succeeded());

  EXPECT_EQ(ST.size(), 2U);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_B")), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueDuplicateRejected) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;

  std::pair<SymbolNameSpec, void *> First[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(First), Succeeded());

  std::pair<SymbolNameSpec, void *> Second[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &Y}};
  EXPECT_THAT_ERROR(ST.addUnique(Second),
                    Failed<StringError>(Property(&StringError::toString,
                                                 HasSubstr("orc_rt_A"))))
      << "Error message should mention the duplicate symbol name";

  // Original not overwritten.
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueMultipleDuplicates) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;

  std::pair<SymbolNameSpec, void *> First[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X},
      {SymbolNameSpec::linker("orc_rt_B"), &Y}};
  ASSERT_THAT_ERROR(ST.addUnique(First), Succeeded());

  std::pair<SymbolNameSpec, void *> Second[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &Z},
      {SymbolNameSpec::linker("orc_rt_B"), &Z}};
  EXPECT_THAT_ERROR(ST.addUnique(Second),
                    Failed<StringError>(Property(
                        &StringError::toString,
                        AllOf(HasSubstr("orc_rt_A"), HasSubstr("orc_rt_B")))));

  // Originals not overwritten.
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_B")), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueAllOrNothing) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;

  std::pair<SymbolNameSpec, void *> First[] = {
      {SymbolNameSpec::linker("orc_rt_existing"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(First), Succeeded());

  // One new, one incompatible — neither should be added.
  std::pair<SymbolNameSpec, void *> Second[] = {
      {SymbolNameSpec::linker("orc_rt_new"), &Y},
      {SymbolNameSpec::linker("orc_rt_existing"), &Z}};
  EXPECT_THAT_ERROR(ST.addUnique(Second), Failed<StringError>());

  EXPECT_EQ(ST.size(), 1U);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_existing")), &X);
  EXPECT_FALSE(ST.count(SymbolNameSpec::linker("orc_rt_new")));
}

TEST(SimpleSymbolTableTest, AddUniqueSameAddressSucceeds) {
  SimpleSymbolTable ST;
  int X = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());
  // Same name, same address — should succeed.
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());
  EXPECT_EQ(ST.size(), 1U);
  EXPECT_EQ(ST.at(SymbolNameSpec::linker("orc_rt_A")), &X);
}

TEST(SimpleSymbolTableTest, Iteration) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X},
      {SymbolNameSpec::linker("orc_rt_B"), &Y},
      {SymbolNameSpec::linker("orc_rt_C"), &Z}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  std::set<std::string> Names;
  for (auto &[Name, Addr] : ST)
    Names.insert(Name);

  EXPECT_EQ(Names.size(), 3U);
  EXPECT_TRUE(Names.count("orc_rt_A"));
  EXPECT_TRUE(Names.count("orc_rt_B"));
  EXPECT_TRUE(Names.count("orc_rt_C"));
}

TEST(SimpleSymbolTableTest, LookupEmptySet) {
  SimpleSymbolTable ST;
  auto R = ST.lookup(SymbolLookupSet());
  EXPECT_TRUE(R.empty());
}

TEST(SimpleSymbolTableTest, LookupRequiredPresentPreservesOrder) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X},
      {SymbolNameSpec::linker("orc_rt_B"), &Y},
      {SymbolNameSpec::linker("orc_rt_C"), &Z}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  // Deliberately not in insertion order (the table is unordered).
  auto R = ST.lookup({{"orc_rt_C", SymbolLookupFlags::RequiredSymbol},
                      {"orc_rt_A", SymbolLookupFlags::RequiredSymbol},
                      {"orc_rt_B", SymbolLookupFlags::RequiredSymbol}});
  ASSERT_EQ(R.size(), 3U);
  EXPECT_EQ(R[0], &Z);
  EXPECT_EQ(R[1], &X);
  EXPECT_EQ(R[2], &Y);
}

TEST(SimpleSymbolTableTest, LookupWeakPresentReturnsAddress) {
  SimpleSymbolTable ST;
  int X = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  auto R = ST.lookup({{"orc_rt_A", SymbolLookupFlags::WeaklyReferencedSymbol}});
  ASSERT_EQ(R.size(), 1U);
  EXPECT_EQ(R[0], &X);
}

TEST(SimpleSymbolTableTest, LookupWeakMissingIsPresentNull) {
  SimpleSymbolTable ST;
  auto R =
      ST.lookup({{"orc_rt_absent", SymbolLookupFlags::WeaklyReferencedSymbol}});
  ASSERT_EQ(R.size(), 1U);
  // Must be an engaged optional holding null, not an empty optional.
  ASSERT_TRUE(R[0].has_value());
  EXPECT_EQ(*R[0], nullptr);
}

TEST(SimpleSymbolTableTest, LookupRequiredMissingIsEmptyOptional) {
  SimpleSymbolTable ST;
  auto R = ST.lookup({{"orc_rt_absent", SymbolLookupFlags::RequiredSymbol}});
  ASSERT_EQ(R.size(), 1U);
  EXPECT_FALSE(R[0].has_value());
}

TEST(SimpleSymbolTableTest, LookupMixedPresentAndMissing) {
  SimpleSymbolTable ST;
  int X = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::linker("orc_rt_A"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  auto R =
      ST.lookup({{"orc_rt_Z", SymbolLookupFlags::RequiredSymbol},
                 {"orc_rt_A", SymbolLookupFlags::RequiredSymbol},
                 {"orc_rt_weak", SymbolLookupFlags::WeaklyReferencedSymbol},
                 {"orc_rt_A", SymbolLookupFlags::WeaklyReferencedSymbol}});
  ASSERT_EQ(R.size(), 4U);
  EXPECT_FALSE(R[0].has_value());
  EXPECT_EQ(R[1], &X);
  ASSERT_TRUE(R[2].has_value());
  EXPECT_EQ(*R[2], nullptr);
  EXPECT_EQ(R[3], &X);
}

TEST(SimpleSymbolTableTest, LookupUsesLinkerLevelNames) {
  SimpleSymbolTable ST;
  int X = 0;
  std::pair<SymbolNameSpec, void *> Syms[] = {
      {SymbolNameSpec::c("orc_rt_cname"), &X}};
  ASSERT_THAT_ERROR(ST.addUnique(Syms), Succeeded());

  // C names are mangled on the way in, so lookup must use the mangled form.
  auto R = ST.lookup({{mangledCopy(SymbolNameSpec::c("orc_rt_cname")),
                       SymbolLookupFlags::RequiredSymbol}});
  ASSERT_EQ(R.size(), 1U);
  EXPECT_EQ(R[0], &X);
}
