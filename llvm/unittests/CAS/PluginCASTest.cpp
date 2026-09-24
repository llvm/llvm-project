//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests the plugin-backed \c ObjectStore and \c ActionCache against the mock
/// plugin implementation in \c llvm/tools/libCASPluginTest.
///
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::unittest::cas;

// HWASan does not tag the globals of a dlopen'ed library with glibc, so the
// plugin faults as soon as it touches one of its own globals.
// FIXME: Re-enable once https://github.com/llvm/llvm-project/issues/57206 is
// fixed.
#if !LLVM_HWADDRESS_SANITIZER_BUILD

extern const char *TestMainArgv0;
static std::string TestStringArg1("castest-string-arg1");

/// \returns the path of the libCASPluginTest dynamic library, which implements
/// the CAS plugin API for testing purposes.
static std::string getCASPluginPath() {
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &TestStringArg1);
  llvm::SmallString<256> PathBuf(sys::path::parent_path(Executable));
#if !defined(_WIN32) || defined(__MINGW32__)
  sys::path::append(PathBuf, "libCASPluginTest" LLVM_PLUGIN_EXT);
#else
  sys::path::append(PathBuf, "CASPluginTest" LLVM_PLUGIN_EXT);
#endif
  return std::string(PathBuf);
}

static CASTestingEnv createPlugin(int I) {
  unittest::TempDir Temp("plugin-cas", /*Unique=*/true);
  std::optional<
      std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
      DBs;
  EXPECT_THAT_ERROR(createPluginCASDatabases(getCASPluginPath(), Temp.path(),
                                             /*PluginArgs=*/{})
                        .moveInto(DBs),
                    Succeeded());
  if (!DBs)
    return CASTestingEnv{nullptr, nullptr, std::move(Temp)};
  return CASTestingEnv{std::move(DBs->first), std::move(DBs->second),
                       std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(PluginCAS, CASTest, ::testing::Values(createPlugin));

TEST(PluginCASTest, isMaterialized) {
  unittest::TempDir Temp("plugin-cas", /*Unique=*/true);
  std::string UpDir(Temp.path("up"));
  std::string DownDir(Temp.path("down"));
  std::pair<std::string, std::string> PluginOpts[] = {
      {"upstream-path", std::string(UpDir)}};

  {
    std::optional<
        std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
        DBs;
    ASSERT_THAT_ERROR(
        createPluginCASDatabases(getCASPluginPath(), DownDir, PluginOpts)
            .moveInto(DBs),
        Succeeded());
    std::shared_ptr<ObjectStore> CAS;
    std::shared_ptr<ActionCache> AC;
    std::tie(CAS, AC) = std::move(*DBs);

    std::optional<CASID> ID1, ID2;
    ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
    std::optional<ObjectRef> ID2Ref = CAS->getReference(*ID2);
    ASSERT_TRUE(ID2Ref);
    bool IsMaterialized = false;
    ASSERT_THAT_ERROR(CAS->isMaterialized(*ID2Ref).moveInto(IsMaterialized),
                      Succeeded());
    EXPECT_TRUE(IsMaterialized);
    ASSERT_THAT_ERROR(AC->put(*ID1, *ID2, /*CanBeDistributed=*/true),
                      Succeeded());
  }

  // Clear "local" cache.
  sys::fs::remove_directories(DownDir);

  {
    std::optional<
        std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
        DBs;
    ASSERT_THAT_ERROR(
        createPluginCASDatabases(getCASPluginPath(), DownDir, PluginOpts)
            .moveInto(DBs),
        Succeeded());
    std::shared_ptr<ObjectStore> CAS;
    std::shared_ptr<ActionCache> AC;
    std::tie(CAS, AC) = std::move(*DBs);

    std::optional<CASID> ID1, ID2;
    ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(AC->get(*ID1, /*CanBeDistributed=*/true).moveInto(ID2),
                      Succeeded());
    std::optional<ObjectRef> ID2Ref = CAS->getReference(*ID2);
    ASSERT_TRUE(ID2Ref);
    bool IsMaterialized = false;
    ASSERT_THAT_ERROR(CAS->isMaterialized(*ID2Ref).moveInto(IsMaterialized),
                      Succeeded());
    EXPECT_FALSE(IsMaterialized);

    std::optional<ObjectProxy> Obj;
    ASSERT_THAT_ERROR(CAS->getProxy(*ID2Ref).moveInto(Obj), Succeeded());
    ASSERT_THAT_ERROR(CAS->isMaterialized(*ID2Ref).moveInto(IsMaterialized),
                      Succeeded());
    EXPECT_TRUE(IsMaterialized);
  }
}

TEST(PluginCASTest, validate) {
  unittest::TempDir Temp("plugin-cas", /*Unique=*/true);

  auto validateIfNeeded = [&](bool Force) {
    return validatePluginCASDatabasesIfNeeded(getCASPluginPath(), Temp.path(),
                                              /*PluginArgs=*/{},
                                              /*CheckHash=*/true, Force);
  };
  auto recover = [&]() {
    return recoverPluginCASDatabases(getCASPluginPath(), Temp.path(),
                                     /*PluginArgs=*/{});
  };
  auto openCAS = [&]() {
    std::optional<
        std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
        DBs;
    EXPECT_THAT_ERROR(createPluginCASDatabases(getCASPluginPath(), Temp.path(),
                                               /*PluginArgs=*/{})
                          .moveInto(DBs),
                      Succeeded());
    return DBs;
  };

  std::optional<ValidationResult> Result;
  ASSERT_THAT_ERROR(validateIfNeeded(/*Force=*/false).moveInto(Result),
                    Succeeded());
  EXPECT_EQ(Result, ValidationResult::Valid);

  {
    auto DBs = openCAS();
    ASSERT_TRUE(DBs);
    auto &[CAS, AC] = *DBs;

    std::optional<CASID> ID1, ID2;
    ASSERT_THAT_ERROR(CAS->createProxy({}, "1").moveInto(ID1), Succeeded());
    ASSERT_THAT_ERROR(CAS->createProxy({}, "2").moveInto(ID2), Succeeded());
    ASSERT_THAT_ERROR(AC->put(*ID1, *ID2), Succeeded());

    EXPECT_THAT_ERROR(CAS->validate(/*CheckHash=*/true), Succeeded());
    EXPECT_THAT_ERROR(AC->validate(), Succeeded());
  }

  // Already validated since boot.
  ASSERT_THAT_ERROR(validateIfNeeded(/*Force=*/false).moveInto(Result),
                    Succeeded());
  EXPECT_EQ(Result, ValidationResult::Skipped);

  ASSERT_THAT_ERROR(validateIfNeeded(/*Force=*/true).moveInto(Result),
                    Succeeded());
  EXPECT_EQ(Result, ValidationResult::Valid);

  // Nothing to recover from after a successful validation.
  ASSERT_THAT_ERROR(recover().moveInto(Result), Succeeded());
  EXPECT_EQ(Result, ValidationResult::Skipped);
  EXPECT_TRUE(sys::fs::exists(Temp.path("v1.1")));

  // Invalidate the data.
  ASSERT_FALSE(sys::fs::remove(Temp.path("v1.1/data.v1")));
  EXPECT_THAT_EXPECTED(validateIfNeeded(/*Force=*/true), Failed());

  // Recovery requires exclusive access, and fails rather than waiting for it.
  {
    auto DBs = openCAS();
    ASSERT_TRUE(DBs);
    EXPECT_THAT_EXPECTED(recover(), Failed());
  }

  ASSERT_THAT_ERROR(recover().moveInto(Result), Succeeded());
  EXPECT_EQ(Result, ValidationResult::Recovered);
  EXPECT_FALSE(sys::fs::exists(Temp.path("v1.1")));

  // A concurrent recovery for the same failed validation is skipped.
  ASSERT_THAT_ERROR(recover().moveInto(Result), Succeeded());
  EXPECT_EQ(Result, ValidationResult::Skipped);

  // Recovery counts as validation for this boot.
  ASSERT_THAT_ERROR(validateIfNeeded(/*Force=*/false).moveInto(Result),
                    Succeeded());
  EXPECT_EQ(Result, ValidationResult::Skipped);

  std::pair<std::string, std::string> BadOpts[] = {{"bogus", ""}};
  EXPECT_THAT_EXPECTED(
      validatePluginCASDatabasesIfNeeded(getCASPluginPath(), Temp.path(),
                                         BadOpts, /*CheckHash=*/false,
                                         /*ForceValidation=*/true),
      Failed());
  EXPECT_THAT_EXPECTED(
      recoverPluginCASDatabases(getCASPluginPath(), Temp.path(), BadOpts),
      Failed());
}

#endif /* !LLVM_HWADDRESS_SANITIZER_BUILD */
