//===-- DAPTypesTest.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/DAPTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::roundtripJSON;

TEST(DAPTypesTest, SourceLLDBData) {
  SourceLLDBData source_data;
  source_data.persistenceData =
      PersistenceData{"module_path123", "symbol_name456"};

  llvm::Expected<SourceLLDBData> deserialized_data = roundtripJSON(source_data);
  ASSERT_THAT_EXPECTED(deserialized_data, llvm::Succeeded());

  EXPECT_EQ(source_data.persistenceData->module_path,
            deserialized_data->persistenceData->module_path);
  EXPECT_EQ(source_data.persistenceData->symbol_name,
            deserialized_data->persistenceData->symbol_name);
}

TEST(DAPTypesTest, DAPSymbol) {
  Symbol symbol;
  symbol.id = 42;
  symbol.isDebug = true;
  symbol.isExternal = false;
  symbol.isSynthetic = true;
  symbol.type = lldb::eSymbolTypeTrampoline;
  symbol.fileAddress = 0x12345678;
  symbol.loadAddress = 0x87654321;
  symbol.size = 64;
  symbol.name = "testSymbol";

  llvm::Expected<Symbol> deserialized_symbol = roundtripJSON(symbol);
  ASSERT_THAT_EXPECTED(deserialized_symbol, llvm::Succeeded());

  EXPECT_EQ(symbol.id, deserialized_symbol->id);
  EXPECT_EQ(symbol.isDebug, deserialized_symbol->isDebug);
  EXPECT_EQ(symbol.isExternal, deserialized_symbol->isExternal);
  EXPECT_EQ(symbol.isSynthetic, deserialized_symbol->isSynthetic);
  EXPECT_EQ(symbol.type, deserialized_symbol->type);
  EXPECT_EQ(symbol.fileAddress, deserialized_symbol->fileAddress);
  EXPECT_EQ(symbol.loadAddress, deserialized_symbol->loadAddress);
  EXPECT_EQ(symbol.size, deserialized_symbol->size);
  EXPECT_EQ(symbol.name, deserialized_symbol->name);
}
