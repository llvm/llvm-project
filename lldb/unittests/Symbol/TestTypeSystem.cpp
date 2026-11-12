//===-- TestTypeSystem.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/TypeSystem.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

class TestTypeSystemMap : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

TEST_F(TestTypeSystemMap, GetTypeSystemForLanguageWithInvalidModule) {
  // GetTypeSystemForLanguage called with an invalid Module.
  TypeSystemMap map;
  Module module{ModuleSpec()};
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, &module,
                                   /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, &module,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, &module,
                                   /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, &module,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
}

TEST_F(TestTypeSystemMap, GetTypeSystemForLanguageWithNoModule) {
  // GetTypeSystemForLanguage called with no Module.
  TypeSystemMap map;
  Module *module = nullptr;
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, module,
                                   /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, module,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, module, /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, module,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
}

TEST_F(TestTypeSystemMap, GetTypeSystemForLanguageWithNoTarget) {
  // GetTypeSystemForLanguage called with no Target.
  TypeSystemMap map;
  Target *target = nullptr;
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, target,
                                   /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeUnknown, target,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language unknown doesn't exist"));

  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, target, /*can_create=*/true),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
  EXPECT_THAT_EXPECTED(
      map.GetTypeSystemForLanguage(eLanguageTypeC, target,
                                   /*can_create=*/false),
      llvm::FailedWithMessage("TypeSystem for language c doesn't exist"));
}
