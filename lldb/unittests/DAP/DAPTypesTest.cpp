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

TEST(DapTypesTest, DAP_var_ref_t) {
  // Check the masked ref constructor.
  const uint32_t scope_masked_var_ref = 0x2000001;
  const var_ref_t scope_ref(scope_masked_var_ref);
  EXPECT_EQ(scope_ref.AsUInt32(), scope_masked_var_ref);
  EXPECT_EQ(scope_ref.Reference(), 1U);
  EXPECT_EQ(scope_ref.Kind(), eReferenceKindScope);

  const uint32_t temp_masked_var_ref = 0x0000021;
  const var_ref_t temp_ref(temp_masked_var_ref);
  EXPECT_EQ(temp_ref.AsUInt32(), temp_masked_var_ref);
  EXPECT_EQ(temp_ref.Reference(), 0x21U);
  EXPECT_EQ(temp_ref.Kind(), eReferenceKindTemporary);

  const uint32_t perm_masked_var_ref = 0x1000032;
  const var_ref_t perm_ref(perm_masked_var_ref);
  EXPECT_EQ(perm_ref.AsUInt32(), perm_masked_var_ref);
  EXPECT_EQ(perm_ref.Reference(), 0x32U);
  EXPECT_EQ(perm_ref.Kind(), eReferenceKindPermanent);

  const var_ref_t invalid_ref{};
  EXPECT_EQ(invalid_ref.AsUInt32(), var_ref_t::k_invalid_var_ref);
  EXPECT_EQ(invalid_ref.Kind(), eReferenceKindInvalid);

  // Check unknown reference kind.
  const uint32_t unknown_masked_var_ref = 0x14000032;
  const var_ref_t unknown_ref(unknown_masked_var_ref);
  EXPECT_EQ(unknown_ref.AsUInt32(), unknown_masked_var_ref);
  EXPECT_EQ(unknown_ref.Reference(), 0x32U);
  EXPECT_EQ(unknown_ref.Kind(), eReferenceKindInvalid);

  const var_ref_t no_child_ref(var_ref_t::k_no_child);
  EXPECT_EQ(no_child_ref.AsUInt32(), 0U);
  EXPECT_EQ(no_child_ref.Reference(), 0U);
  EXPECT_EQ(no_child_ref.Kind(), eReferenceKindTemporary);

  // Check the refkind constructor.
  const uint32_t scope2_masked_ref = 0x2000003;
  const var_ref_t scope_ref2(3, eReferenceKindScope);
  EXPECT_EQ(scope_ref2.AsUInt32(), scope2_masked_ref);
  EXPECT_EQ(scope_ref2.Reference(), 3U);
  EXPECT_EQ(scope_ref2.Kind(), eReferenceKindScope);

  EXPECT_EQ(var_ref_t().AsUInt32(),
            var_ref_t{var_ref_t::k_invalid_var_ref}.AsUInt32());
}
