//===- NativeSymbolReuseTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDB.h"

#include "llvm/DebugInfo/PDB/ConcreteSymbolEnumerator.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/Support/Path.h"

#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::pdb;

extern const char *TestMainArgv0;

TEST(NativeSymbolReuseTest, GlobalSymbolReuse) {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "empty.pdb");

  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForPDB(PDB_ReaderType::Native, InputsDir, S);

  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  SymIndexId GlobalId;
  {
    auto GS1 = S->getGlobalScope();
    auto GS2 = S->getGlobalScope();

    GlobalId = GS1->getSymIndexId();
    SymIndexId Id2 = GS1->getSymIndexId();
    EXPECT_EQ(GlobalId, Id2);
  }

  {
    auto GS3 = S->getGlobalScope();

    SymIndexId Id3 = GS3->getSymIndexId();
    EXPECT_EQ(GlobalId, Id3);
  }
}

TEST(NativeSymbolReuseTest, CompilandSymbolReuse) {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "empty.pdb");

  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForPDB(PDB_ReaderType::Native, InputsDir, S);

  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  auto GS = S->getGlobalScope();

  std::vector<SymIndexId> CompilandIds;
  {
    auto Compilands = GS->findAllChildren<PDBSymbolCompiland>();
    ASSERT_NE(nullptr, Compilands);
    ASSERT_EQ(2U, Compilands->getChildCount());
    std::vector<SymIndexId> Ids2;

    // First try resetting the enumerator, then try destroying the enumerator
    // and constructing another one.
    while (auto Compiland = Compilands->getNext())
      CompilandIds.push_back(Compiland->getSymIndexId());
    Compilands->reset();
    while (auto Compiland = Compilands->getNext())
      Ids2.push_back(Compiland->getSymIndexId());

    EXPECT_EQ(CompilandIds, Ids2);
  }

  {
    auto Compilands = GS->findAllChildren<PDBSymbolCompiland>();
    ASSERT_NE(nullptr, Compilands);
    ASSERT_EQ(2U, Compilands->getChildCount());

    std::vector<SymIndexId> Ids3;
    while (auto Compiland = Compilands->getNext())
      Ids3.push_back(Compiland->getSymIndexId());

    EXPECT_EQ(CompilandIds, Ids3);
  }
}

TEST(NativeSymbolReuseTest, CompilandSymbolReuseBackwards) {
  SmallString<128> InputsDir = unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(InputsDir, "empty.pdb");

  std::unique_ptr<IPDBSession> S;
  Error E = pdb::loadDataForPDB(PDB_ReaderType::Native, InputsDir, S);

  ASSERT_THAT_ERROR(std::move(E), Succeeded());

  auto GS = S->getGlobalScope();

  // This time do the first iteration backwards, and make sure that when you
  // then iterate them forwards, the IDs come out in reverse.
  std::vector<SymIndexId> CompilandIds;
  {
    auto Compilands = GS->findAllChildren<PDBSymbolCompiland>();
    ASSERT_NE(nullptr, Compilands);
    ASSERT_EQ(2U, Compilands->getChildCount());

    std::vector<SymIndexId> Ids2;

    for (int I = Compilands->getChildCount() - 1; I >= 0; --I) {
      auto Compiland = Compilands->getChildAtIndex(I);
      CompilandIds.push_back(Compiland->getSymIndexId());
    }

    while (auto Compiland = Compilands->getNext())
      Ids2.push_back(Compiland->getSymIndexId());

    auto ReversedIter = llvm::reverse(Ids2);
    std::vector<SymIndexId> Reversed{ReversedIter.begin(), ReversedIter.end()};
    EXPECT_EQ(CompilandIds, Reversed);
  }
}
