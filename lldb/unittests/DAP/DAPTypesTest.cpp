//===-- DAPTypesTest.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol/DAPTypes.h"
#include "TestingSupport/TestUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap::protocol;
using lldb_private::roundtripJSON;

TEST(DAPTypesTest, SourceLLDBData) {
  SourceLLDBData source_data;
  source_data.persistenceData = PersistenceData{"module_path123", "symbol_name456"};

  llvm::Expected<SourceLLDBData> deserialized_data = roundtripJSON(source_data);
  ASSERT_THAT_EXPECTED(deserialized_data, llvm::Succeeded());

  EXPECT_EQ(source_data.persistenceData->module_path,
            deserialized_data->persistenceData->module_path);
  EXPECT_EQ(source_data.persistenceData->symbol_name,
            deserialized_data->persistenceData->symbol_name);
}
