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

#include "orc-rt/SimpleSymbolTable.h"
#include "gtest/gtest.h"

#include <set>
#include <string>

using namespace orc_rt;

TEST(SimpleSymbolTableTest, EmptyByDefault) {
  SimpleSymbolTable ST;
  EXPECT_TRUE(ST.empty());
  EXPECT_EQ(ST.size(), 0U);
  EXPECT_EQ(ST.begin(), ST.end());
}

TEST(SimpleSymbolTableTest, AddSymbolsUnique) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_A", &X}, {"orc_rt_B", &Y}};

  auto Err = ST.addUnique(Syms);
  EXPECT_FALSE(Err) << "Unexpected error adding unique symbols";

  EXPECT_EQ(ST.size(), 2U);
  EXPECT_FALSE(ST.empty());
  EXPECT_TRUE(ST.count("orc_rt_A"));
  EXPECT_TRUE(ST.count("orc_rt_B"));
  EXPECT_EQ(ST.at("orc_rt_A"), &X);
  EXPECT_EQ(ST.at("orc_rt_B"), &Y);
}

TEST(SimpleSymbolTableTest, AddConstPointers) {
  SimpleSymbolTable ST;
  const int X = 42;
  const int Y = 7;
  std::pair<const char *, const void *> Syms[] = {{"orc_rt_A", &X},
                                                  {"orc_rt_B", &Y}};
  cantFail(ST.addUnique(Syms));

  EXPECT_EQ(ST.at("orc_rt_A"), &X);
  EXPECT_EQ(ST.at("orc_rt_B"), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueMultipleCalls) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X}};
  std::pair<const char *, void *> Second[] = {{"orc_rt_B", &Y}};

  cantFail(ST.addUnique(First));
  cantFail(ST.addUnique(Second));

  EXPECT_EQ(ST.size(), 2U);
  EXPECT_EQ(ST.at("orc_rt_A"), &X);
  EXPECT_EQ(ST.at("orc_rt_B"), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueDuplicateRejected) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X}};
  cantFail(ST.addUnique(First));

  std::pair<const char *, void *> Second[] = {{"orc_rt_A", &Y}};
  auto Err = ST.addUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());

  auto ErrMsg = toString(std::move(Err));
  EXPECT_NE(ErrMsg.find("orc_rt_A"), std::string::npos)
      << "Error message should mention the duplicate symbol name";

  // Original not overwritten.
  EXPECT_EQ(ST.at("orc_rt_A"), &X);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueMultipleDuplicates) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X},
                                             {"orc_rt_B", &Y}};
  cantFail(ST.addUnique(First));

  std::pair<const char *, void *> Second[] = {{"orc_rt_A", &Z},
                                              {"orc_rt_B", &Z}};
  auto Err = ST.addUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());

  auto ErrMsg = toString(std::move(Err));
  EXPECT_NE(ErrMsg.find("orc_rt_A"), std::string::npos);
  EXPECT_NE(ErrMsg.find("orc_rt_B"), std::string::npos);

  // Originals not overwritten.
  EXPECT_EQ(ST.at("orc_rt_A"), &X);
  EXPECT_EQ(ST.at("orc_rt_B"), &Y);
}

TEST(SimpleSymbolTableTest, AddSymbolsUniqueAllOrNothing) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_existing", &X}};
  cantFail(ST.addUnique(First));

  // One new, one incompatible — neither should be added.
  std::pair<const char *, void *> Second[] = {{"orc_rt_new", &Y},
                                              {"orc_rt_existing", &Z}};
  auto Err = ST.addUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());
  consumeError(std::move(Err));

  EXPECT_EQ(ST.size(), 1U);
  EXPECT_EQ(ST.at("orc_rt_existing"), &X);
  EXPECT_FALSE(ST.count("orc_rt_new"));
}

TEST(SimpleSymbolTableTest, AddUniqueSameAddressSucceeds) {
  SimpleSymbolTable ST;
  int X = 0;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_A", &X}};
  cantFail(ST.addUnique(Syms));
  cantFail(ST.addUnique(Syms)); // Same name, same address — should succeed.
  EXPECT_EQ(ST.size(), 1U);
  EXPECT_EQ(ST.at("orc_rt_A"), &X);
}

TEST(SimpleSymbolTableTest, Iteration) {
  SimpleSymbolTable ST;
  int X = 0, Y = 0, Z = 0;
  std::pair<const char *, void *> Syms[] = {
      {"orc_rt_A", &X}, {"orc_rt_B", &Y}, {"orc_rt_C", &Z}};
  cantFail(ST.addUnique(Syms));

  std::set<std::string> Names;
  for (auto &[Name, Addr] : ST)
    Names.insert(Name);

  EXPECT_EQ(Names.size(), 3U);
  EXPECT_TRUE(Names.count("orc_rt_A"));
  EXPECT_TRUE(Names.count("orc_rt_B"));
  EXPECT_TRUE(Names.count("orc_rt_C"));
}
