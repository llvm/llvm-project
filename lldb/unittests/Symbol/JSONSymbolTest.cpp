//===-- JSONSymbolTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Section.h"
#include "lldb/Symbol/Symbol.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace llvm;
using namespace lldb_private;

static std::string g_error_no_section_list = "no section list provided";
static std::string g_error_both_value_and_address =
    "symbol cannot contain both a value and an address";
static std::string g_error_neither_value_or_address =
    "symbol must contain either a value or an address";

TEST(JSONSymbolTest, DeserializeCodeAddress) {
  std::string text = R"(
{
  "name": "foo",
  "type": "code",
  "size": 32,
  "address": 4096
})";

  Expected<json::Value> json = json::parse(text);
  ASSERT_TRUE(static_cast<bool>(json));

  json::Path::Root root;
  JSONSymbol json_symbol;
  ASSERT_TRUE(fromJSON(*json, json_symbol, root));

  SectionSP sect_sp(new Section(
      /*module_sp=*/ModuleSP(),
      /*obj_file=*/nullptr,
      /*sect_id=*/1,
      /*name=*/ConstString(".text"),
      /*sect_type=*/eSectionTypeCode,
      /*file_vm_addr=*/0x1000,
      /*vm_size=*/0x1000,
      /*file_offset=*/0,
      /*file_size=*/0,
      /*log2align=*/5,
      /*flags=*/0x10203040));
  SectionList sect_list;
  sect_list.AddSection(sect_sp);

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, &sect_list);
  EXPECT_THAT_EXPECTED(symbol, llvm::Succeeded());
  EXPECT_EQ(symbol->GetName(), ConstString("foo"));
  EXPECT_EQ(symbol->GetFileAddress(), static_cast<lldb::addr_t>(0x1000));
  EXPECT_EQ(symbol->GetType(), eSymbolTypeCode);
}

TEST(JSONSymbolTest, DeserializeCodeValue) {
  std::string text = R"(
{
  "name": "foo",
  "type": "code",
  "size": 32,
  "value": 4096
})";

  Expected<json::Value> json = json::parse(text);
  EXPECT_THAT_EXPECTED(json, llvm::Succeeded());

  json::Path::Root root;
  JSONSymbol json_symbol;
  ASSERT_TRUE(fromJSON(*json, json_symbol, root));

  SectionList sect_list;

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, &sect_list);
  EXPECT_THAT_EXPECTED(symbol, llvm::Succeeded());
  EXPECT_EQ(symbol->GetName(), ConstString("foo"));
  EXPECT_EQ(symbol->GetRawValue(), static_cast<lldb::addr_t>(0x1000));
  EXPECT_EQ(symbol->GetType(), eSymbolTypeCode);
}

TEST(JSONSymbolTest, JSONInvalidValueAndAddress) {
  std::string text = R"(
{
  "name": "foo",
  "type": "code",
  "size": 32,
  "value": 4096,
  "address": 4096
})";

  Expected<json::Value> json = json::parse(text);
  EXPECT_THAT_EXPECTED(json, llvm::Succeeded());

  json::Path::Root root;
  JSONSymbol json_symbol;
  ASSERT_FALSE(fromJSON(*json, json_symbol, root));
}

TEST(JSONSymbolTest, JSONInvalidNoValueOrAddress) {
  std::string text = R"(
{
  "name": "foo",
  "type": "code",
  "size": 32
})";

  Expected<json::Value> json = json::parse(text);
  EXPECT_THAT_EXPECTED(json, llvm::Succeeded());

  json::Path::Root root;
  JSONSymbol json_symbol;
  ASSERT_FALSE(fromJSON(*json, json_symbol, root));
}

TEST(JSONSymbolTest, JSONInvalidType) {
  std::string text = R"(
{
  "name": "foo",
  "type": "bogus",
  "value": 4096,
  "size": 32
})";

  Expected<json::Value> json = json::parse(text);
  EXPECT_THAT_EXPECTED(json, llvm::Succeeded());

  json::Path::Root root;
  JSONSymbol json_symbol;
  ASSERT_FALSE(fromJSON(*json, json_symbol, root));
}

TEST(JSONSymbolTest, SymbolInvalidNoSectionList) {
  JSONSymbol json_symbol;
  json_symbol.value = 0x1;

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, nullptr);
  EXPECT_THAT_EXPECTED(symbol,
                       llvm::FailedWithMessage(g_error_no_section_list));
}

TEST(JSONSymbolTest, SymbolInvalidValueAndAddress) {
  JSONSymbol json_symbol;
  json_symbol.value = 0x1;
  json_symbol.address = 0x2;

  SectionList sect_list;

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, &sect_list);
  EXPECT_THAT_EXPECTED(symbol,
                       llvm::FailedWithMessage(g_error_both_value_and_address));
}

TEST(JSONSymbolTest, SymbolInvalidNoValueOrAddress) {
  JSONSymbol json_symbol;

  SectionList sect_list;

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, &sect_list);
  EXPECT_THAT_EXPECTED(
      symbol, llvm::FailedWithMessage(g_error_neither_value_or_address));
}

TEST(JSONSymbolTest, SymbolInvalidAddressNotInSection) {
  JSONSymbol json_symbol;
  json_symbol.address = 0x0fff;

  SectionSP sect_sp(new Section(
      /*module_sp=*/ModuleSP(),
      /*obj_file=*/nullptr,
      /*sect_id=*/1,
      /*name=*/ConstString(".text"),
      /*sect_type=*/eSectionTypeCode,
      /*file_vm_addr=*/0x1000,
      /*vm_size=*/0x1000,
      /*file_offset=*/0,
      /*file_size=*/0,
      /*log2align=*/5,
      /*flags=*/0x10203040));
  SectionList sect_list;
  sect_list.AddSection(sect_sp);

  Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, &sect_list);
  EXPECT_THAT_EXPECTED(
      symbol, llvm::FailedWithMessage("no section found for address: 0xfff"));
}
