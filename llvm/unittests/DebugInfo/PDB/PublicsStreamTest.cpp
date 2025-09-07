//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/PublicsStream.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/MemoryBuffer.h"

#include "llvm/Testing/Support/SupportHelpers.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;

extern const char *TestMainArgv0;

static std::string getExePath() {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "PublicSymbols.pdb");
  return std::string(InputsDir);
}

TEST(PublicsStreamTest, FindByAddress) {
  std::string ExePath = getExePath();
  auto Buffer = MemoryBuffer::getFile(ExePath, /*IsText=*/false,
                                      /*RequiresNullTerminator=*/false);
  ASSERT_TRUE(bool(Buffer));
  auto Stream = std::make_unique<MemoryBufferByteStream>(
      std::move(*Buffer), llvm::endianness::little);

  BumpPtrAllocator Alloc;
  PDBFile File(ExePath, std::move(Stream), Alloc);
  ASSERT_FALSE(bool(File.parseFileHeaders()));
  ASSERT_FALSE(bool(File.parseStreamData()));

  auto Publics = File.getPDBPublicsStream();
  ASSERT_TRUE(bool(Publics));
  auto Symbols = File.getPDBSymbolStream();
  ASSERT_TRUE(bool(Symbols));

  auto VTableDerived = Publics->findByAddress(*Symbols, 2, 8);
  ASSERT_TRUE(VTableDerived.has_value());
  // both derived and derived2 have their vftables there - but derived2 is first
  // (due to ICF)
  ASSERT_EQ(VTableDerived->first.Name, "??_7Derived2@@6B@");
  ASSERT_EQ(VTableDerived->second, 26u);

  ASSERT_FALSE(Publics->findByAddress(*Symbols, 2, 7).has_value());
  ASSERT_FALSE(Publics->findByAddress(*Symbols, 2, 9).has_value());

  auto GlobalSym = Publics->findByAddress(*Symbols, 3, 0);
  ASSERT_TRUE(GlobalSym.has_value());
  ASSERT_EQ(GlobalSym->first.Name, "?AGlobal@@3HA");
  ASSERT_EQ(GlobalSym->second, 30u);
}
