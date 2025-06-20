//===- ImportIDTableTests.cpp - Unit tests for ImportIDTable --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionImport.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <set>
#include <type_traits>

using namespace llvm;

TEST(ImportIDTableTests, Basic) {
  FunctionImporter::ImportIDTable Table;

  auto [Def, Decl] = Table.createImportIDs("mod", 123U);
  auto [Def2, Decl2] = Table.createImportIDs("stuff", 456U);

  // Def and Decl must be of the same unsigned integer type.
  static_assert(
      std::is_unsigned_v<FunctionImporter::ImportIDTable::ImportIDTy>);
  static_assert(std::is_same_v<FunctionImporter::ImportIDTable::ImportIDTy,
                               decltype(Def)>);
  static_assert(std::is_same_v<FunctionImporter::ImportIDTable::ImportIDTy,
                               decltype(Decl)>);

  // Check that all IDs are unique.
  std::set<FunctionImporter::ImportIDTable::ImportIDTy> IDs = {Def, Decl, Def2,
                                                               Decl2};
  EXPECT_THAT(IDs, ::testing::SizeIs(4));

  // Verify what Def maps to.
  auto DefTuple = Table.lookup(Def);
  EXPECT_EQ(std::get<0>(DefTuple), StringRef("mod"));
  EXPECT_EQ(std::get<1>(DefTuple), 123U);
  EXPECT_EQ(std::get<2>(DefTuple), GlobalValueSummary::Definition);

  // Verify what Decl maps to.
  auto DeclTuple = Table.lookup(Decl);
  EXPECT_EQ(std::get<0>(DeclTuple), StringRef("mod"));
  EXPECT_EQ(std::get<1>(DeclTuple), 123U);
  EXPECT_EQ(std::get<2>(DeclTuple), GlobalValueSummary::Declaration);

  // Verify what Def2 maps to.
  auto Def2Tuple = Table.lookup(Def2);
  EXPECT_EQ(std::get<0>(Def2Tuple), StringRef("stuff"));
  EXPECT_EQ(std::get<1>(Def2Tuple), 456U);
  EXPECT_EQ(std::get<2>(Def2Tuple), GlobalValueSummary::Definition);

  // Verify what Decl2 maps to.
  auto Decl2Tuple = Table.lookup(Decl2);
  EXPECT_EQ(std::get<0>(Decl2Tuple), StringRef("stuff"));
  EXPECT_EQ(std::get<1>(Decl2Tuple), 456U);
  EXPECT_EQ(std::get<2>(Decl2Tuple), GlobalValueSummary::Declaration);
}

TEST(ImportIDTableTests, Duplicates) {
  FunctionImporter::ImportIDTable Table;

  auto [Def1, Decl1] = Table.createImportIDs("mod", 123U);
  auto [Def2, Decl2] = Table.createImportIDs("mod", 123U);

  // Verify we get the same IDs back.
  EXPECT_EQ(Def1, Def2);
  EXPECT_EQ(Decl1, Decl2);
}

TEST(ImportIDTableTests, Present) {
  FunctionImporter::ImportIDTable Table;

  auto [Def, Decl] = Table.createImportIDs("mod", 123U);
  auto Result = Table.getImportIDs("mod", 123U);

  // Verify that we get the same IDs back.
  ASSERT_NE(Result, std::nullopt);
  EXPECT_EQ(Result->first, Def);
  EXPECT_EQ(Result->second, Decl);
}

TEST(ImportIDTableTests, Missing) {
  FunctionImporter::ImportIDTable Table;

  auto Result = Table.getImportIDs("mod", 123U);

  // Verify that we get std::nullopt for a non-existent pair.
  EXPECT_EQ(Result, std::nullopt);
}
