//===-- ObjCPlusPlusLanguageTest.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/ObjCPlusPlus/ObjCPlusPlusLanguage.h"
#include "lldb/lldb-enumerations.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(ObjCPlusPlusLanguage, IsCategoryApplicable) {
  ObjCPlusPlusLanguage language;
  for (lldb::LanguageType component :
       {lldb::eLanguageTypeC89, lldb::eLanguageTypeC, lldb::eLanguageTypeC99,
        lldb::eLanguageTypeC_plus_plus, lldb::eLanguageTypeObjC,
        lldb::eLanguageTypeObjC_plus_plus})
    EXPECT_TRUE(language.IsCategoryApplicable(lldb::eLanguageTypeObjC_plus_plus,
                                              component));
  EXPECT_FALSE(language.IsCategoryApplicable(lldb::eLanguageTypeObjC_plus_plus,
                                             lldb::eLanguageTypeSwift));
}
