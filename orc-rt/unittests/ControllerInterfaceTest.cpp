//===- ControllerInterfaceTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for orc-rt's ControllerInterface.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/ControllerInterface.h"
#include "gtest/gtest.h"

#include <set>
#include <string>

using namespace orc_rt;

TEST(ControllerInterfaceTest, EmptyByDefault) {
  ControllerInterface CI;
  EXPECT_TRUE(CI.empty());
  EXPECT_EQ(CI.size(), 0U);
  EXPECT_EQ(CI.begin(), CI.end());
}

TEST(ControllerInterfaceTest, AddSymbolsUnique) {
  ControllerInterface CI;
  int X = 0, Y = 0;
  std::pair<const char *, void *> Syms[] = {{"orc_rt_A", &X}, {"orc_rt_B", &Y}};

  auto Err = CI.addSymbolsUnique(Syms);
  EXPECT_FALSE(Err) << "Unexpected error adding unique symbols";

  EXPECT_EQ(CI.size(), 2U);
  EXPECT_FALSE(CI.empty());
  EXPECT_TRUE(CI.count("orc_rt_A"));
  EXPECT_TRUE(CI.count("orc_rt_B"));
  EXPECT_EQ(CI.at("orc_rt_A"), &X);
  EXPECT_EQ(CI.at("orc_rt_B"), &Y);
}

TEST(ControllerInterfaceTest, AddSymbolsUniqueMultipleCalls) {
  ControllerInterface CI;
  int X = 0, Y = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X}};
  std::pair<const char *, void *> Second[] = {{"orc_rt_B", &Y}};

  cantFail(CI.addSymbolsUnique(First));
  cantFail(CI.addSymbolsUnique(Second));

  EXPECT_EQ(CI.size(), 2U);
  EXPECT_EQ(CI.at("orc_rt_A"), &X);
  EXPECT_EQ(CI.at("orc_rt_B"), &Y);
}

TEST(ControllerInterfaceTest, AddSymbolsUniqueDuplicateRejected) {
  ControllerInterface CI;
  int X = 0, Y = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X}};
  cantFail(CI.addSymbolsUnique(First));

  std::pair<const char *, void *> Second[] = {{"orc_rt_A", &Y}};
  auto Err = CI.addSymbolsUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());

  auto ErrMsg = toString(std::move(Err));
  EXPECT_NE(ErrMsg.find("orc_rt_A"), std::string::npos)
      << "Error message should mention the duplicate symbol name";

  // Original not overwritten.
  EXPECT_EQ(CI.at("orc_rt_A"), &X);
}

TEST(ControllerInterfaceTest, AddSymbolsUniqueMultipleDuplicates) {
  ControllerInterface CI;
  int X = 0, Y = 0, Z = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_A", &X},
                                             {"orc_rt_B", &Y}};
  cantFail(CI.addSymbolsUnique(First));

  std::pair<const char *, void *> Second[] = {{"orc_rt_A", &Z},
                                              {"orc_rt_B", &Z}};
  auto Err = CI.addSymbolsUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());

  auto ErrMsg = toString(std::move(Err));
  EXPECT_NE(ErrMsg.find("orc_rt_A"), std::string::npos);
  EXPECT_NE(ErrMsg.find("orc_rt_B"), std::string::npos);

  // Originals not overwritten.
  EXPECT_EQ(CI.at("orc_rt_A"), &X);
  EXPECT_EQ(CI.at("orc_rt_B"), &Y);
}

TEST(ControllerInterfaceTest, AddSymbolsUniqueAllOrNothing) {
  ControllerInterface CI;
  int X = 0, Y = 0, Z = 0;

  std::pair<const char *, void *> First[] = {{"orc_rt_existing", &X}};
  cantFail(CI.addSymbolsUnique(First));

  // One new, one duplicate — neither should be added.
  std::pair<const char *, void *> Second[] = {{"orc_rt_new", &Y},
                                              {"orc_rt_existing", &Z}};
  auto Err = CI.addSymbolsUnique(Second);
  EXPECT_TRUE(Err.isA<StringError>());
  consumeError(std::move(Err));

  EXPECT_EQ(CI.size(), 1U);
  EXPECT_EQ(CI.at("orc_rt_existing"), &X);
  EXPECT_FALSE(CI.count("orc_rt_new"));
}

TEST(ControllerInterfaceTest, Iteration) {
  ControllerInterface CI;
  int X = 0, Y = 0, Z = 0;
  std::pair<const char *, void *> Syms[] = {
      {"orc_rt_A", &X}, {"orc_rt_B", &Y}, {"orc_rt_C", &Z}};
  cantFail(CI.addSymbolsUnique(Syms));

  std::set<std::string> Names;
  for (auto &[Name, Addr] : CI)
    Names.insert(Name);

  EXPECT_EQ(Names.size(), 3U);
  EXPECT_TRUE(Names.count("orc_rt_A"));
  EXPECT_TRUE(Names.count("orc_rt_B"));
  EXPECT_TRUE(Names.count("orc_rt_C"));
}
