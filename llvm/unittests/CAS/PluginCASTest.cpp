//===- llvm/unittest/CAS/PluginCASTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#if LLVM_ENABLE_ONDISK_CAS

using namespace llvm;
using namespace llvm::cas;

// See llvm/utils/unittest/UnitTestMain/TestMain.cpp
extern const char *TestMainArgv0;

// Just a reachable symbol to ease resolving of the executable's path.
static std::string TestStringArg1("plugincas-test-string-arg1");

static std::string getCASPluginPath() {
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &TestStringArg1);
  llvm::SmallString<256> PathBuf(sys::path::parent_path(
      sys::path::parent_path(sys::path::parent_path(Executable))));
#ifndef _WIN32
  std::string LibName = "libCASPluginTest";
  sys::path::append(PathBuf, "lib", LibName + LLVM_PLUGIN_EXT);
#else
  std::string LibName = "CASPluginTest";
  sys::path::append(PathBuf, "bin", LibName + LLVM_PLUGIN_EXT);
#endif
  return std::string(PathBuf);
}

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
    ASSERT_THAT_ERROR(AC->put(*ID1, *ID2, /*Globally=*/true), Succeeded());
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
    ASSERT_THAT_ERROR(AC->get(*ID1, /*Globally=*/true).moveInto(ID2),
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

#endif // LLVM_ENABLE_ONDISK_CAS
