//===- COFFImportFileTest.cpp - COFF short import tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/COFFImportFile.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::object;

namespace {

static std::string makeShortImport(StringRef SymbolName,
                                   COFF::ImportNameType NameType,
                                   StringRef ExportName = {}) {
  coff_import_header Header{};
  Header.Sig2 = 0xffff;
  Header.Machine = COFF::IMAGE_FILE_MACHINE_AMD64;
  Header.TypeInfo = (NameType << 2) | COFF::IMPORT_DATA;

  std::string Buffer(reinterpret_cast<const char *>(&Header), sizeof(Header));
  Buffer.append(SymbolName);
  Buffer.push_back('\0');
  Buffer.append("test.dll");
  Buffer.push_back('\0');
  if (!ExportName.empty()) {
    Buffer.append(ExportName);
    Buffer.push_back('\0');
  }
  return Buffer;
}

TEST(COFFImportFileTest, DistinguishesSymbolNameFromExportName) {
  struct TestCase {
    COFF::ImportNameType NameType;
    StringRef SymbolName;
    StringRef ExportAs;
    StringRef ExpectedExportName;
  };

  const TestCase Cases[] = {
      {COFF::IMPORT_NAME, "_name@8", {}, "_name@8"},
      {COFF::IMPORT_NAME_NOPREFIX, "_name@8", {}, "name@8"},
      {COFF::IMPORT_NAME_UNDECORATE, "_name@8", {}, "name"},
      {COFF::IMPORT_NAME_EXPORTAS, "_name@8", "exported", "exported"},
      {COFF::IMPORT_ORDINAL, "_name@8", {}, {}},
  };

  for (const auto &C : Cases) {
    SCOPED_TRACE(C.NameType);
    std::string Buffer = makeShortImport(C.SymbolName, C.NameType, C.ExportAs);
    COFFImportFile File(MemoryBufferRef(Buffer, "test"));
    EXPECT_EQ(File.getSymbolName(), C.SymbolName);
    EXPECT_EQ(File.getExportName(), C.ExpectedExportName);
  }
}

} // namespace
