//===- llvm/unittest/DebugInfo/PDB/NativeSessionTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/Support/Path.h"

#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <vector>

using namespace llvm;
using namespace llvm::pdb;

extern const char *TestMainArgv0;

static std::string getExePath() {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "SimpleTest.exe");
  return std::string(InputsDir);
}

TEST(NativeSessionTest, TestCreateFromExe) {
  std::unique_ptr<IPDBSession> S;
  std::string ExePath = getExePath();
  Expected<std::string> PdbPath = NativeSession::searchForPdb({ExePath});
  ASSERT_TRUE((bool)PdbPath);

  Error E = NativeSession::createFromPdbPath(PdbPath.get(), S);
  ASSERT_THAT_ERROR(std::move(E), Succeeded());
}

TEST(NativeSessionTest, TestSetLoadAddress) {
  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForEXE(PDB_ReaderType::Native, getExePath(), S);
  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  S->setLoadAddress(123);
  EXPECT_EQ(S->getLoadAddress(), 123U);
}

TEST(NativeSessionTest, TestAddressForVA) {
  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForEXE(PDB_ReaderType::Native, getExePath(), S);
  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  uint64_t LoadAddr = S->getLoadAddress();
  uint32_t Section;
  uint32_t Offset;
  ASSERT_TRUE(S->addressForVA(LoadAddr + 5000, Section, Offset));
  EXPECT_EQ(1U, Section);
  EXPECT_EQ(904U, Offset);

  ASSERT_TRUE(S->addressForVA(-1, Section, Offset));
  EXPECT_EQ(0U, Section);
  EXPECT_EQ(0U, Offset);

  ASSERT_TRUE(S->addressForVA(4, Section, Offset));
  EXPECT_EQ(0U, Section);
  EXPECT_EQ(4U, Offset);

  ASSERT_TRUE(S->addressForVA(LoadAddr + 100000, Section, Offset));
  EXPECT_EQ(3U, Section);
  EXPECT_EQ(83616U, Offset);
}

TEST(NativeSessionTest, TestAddressForRVA) {
  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForEXE(PDB_ReaderType::Native, getExePath(), S);
  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  uint32_t Section;
  uint32_t Offset;
  ASSERT_TRUE(S->addressForVA(5000, Section, Offset));
  EXPECT_EQ(1U, Section);
  EXPECT_EQ(904U, Offset);

  ASSERT_TRUE(S->addressForVA(-1, Section, Offset));
  EXPECT_EQ(0U, Section);
  EXPECT_EQ(0U, Offset);

  ASSERT_TRUE(S->addressForVA(4, Section, Offset));
  EXPECT_EQ(0U, Section);
  EXPECT_EQ(4U, Offset);

  ASSERT_TRUE(S->addressForVA(100000, Section, Offset));
  EXPECT_EQ(3U, Section);
  EXPECT_EQ(83616U, Offset);
}
