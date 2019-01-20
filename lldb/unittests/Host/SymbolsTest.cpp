//===-- SymbolsTest.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/Symbols.h"

using namespace lldb_private;

namespace {
class SymbolsTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
  }
  void TearDown() override {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

TEST_F(
    SymbolsTest,
    TerminateLocateExecutableSymbolFileForUnknownExecutableAndUnknownSymbolFile) {
  ModuleSpec module_spec;
  FileSpec symbol_file_spec = Symbols::LocateExecutableSymbolFile(module_spec);
  EXPECT_TRUE(symbol_file_spec.GetFilename().IsEmpty());
}

TEST_F(SymbolsTest,
       LocateExecutableSymbolFileForUnknownExecutableAndMissingSymbolFile) {
  ModuleSpec module_spec;
  // using a GUID here because the symbol file shouldn't actually exist on disk
  module_spec.GetSymbolFileSpec().SetFile(
      "4A524676-B24B-4F4E-968A-551D465EBAF1.so", FileSpec::Style::native);
  FileSpec symbol_file_spec = Symbols::LocateExecutableSymbolFile(module_spec);
  EXPECT_TRUE(symbol_file_spec.GetFilename().IsEmpty());
}
