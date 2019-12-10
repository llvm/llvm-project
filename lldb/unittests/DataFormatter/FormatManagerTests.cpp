//===-- FormatManagerTests.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/SymbolContext.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(FormatManagerTests, CompatibleLangs) {
  std::vector<LanguageType> candidates = {eLanguageTypeC_plus_plus,
                                          eLanguageTypeObjC};
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC), candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC89), candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC99), candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC11), candidates);

  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC_plus_plus),
            candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC_plus_plus_03),
            candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC_plus_plus_11),
            candidates);
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeC_plus_plus_14),
            candidates);

  candidates = {eLanguageTypeObjC};
  EXPECT_EQ(FormatManager::GetCandidateLanguages(eLanguageTypeObjC),
            candidates);
}
