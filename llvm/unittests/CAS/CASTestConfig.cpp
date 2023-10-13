//===- CASTestConfig.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/PluginCAS.h"
#include "llvm/Config/config.h"
#include "gtest/gtest.h"
#include <memory>
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

// See llvm/utils/unittest/UnitTestMain/TestMain.cpp
extern const char *TestMainArgv0;

// Just a reachable symbol to ease resolving of the executable's path.
static std::string TestStringArg1("plugincas-test-string-arg1");

std::string llvm::unittest::cas::getCASPluginPath() {
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &TestStringArg1);
  llvm::SmallString<256> PathBuf(sys::path::parent_path(
      sys::path::parent_path(sys::path::parent_path(Executable))));
  std::string LibName = "libCASPluginTest";
  sys::path::append(PathBuf, "lib", LibName + LLVM_PLUGIN_EXT);
  return std::string(PathBuf);
}


TestingAndDir createInMemory(int I) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  std::unique_ptr<ActionCache> Cache = createInMemoryActionCache();
  return TestingAndDir{std::move(CAS), std::move(Cache), std::nullopt};
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));

#if LLVM_ENABLE_ONDISK_CAS
#ifndef _WIN32
__attribute__((constructor)) static void configureCASTestEnv() {
  // Restrict the size of the on-disk CAS for tests. This allows testing in
  // constrained environments (e.g. small TMPDIR). It also prevents leaving
  // behind large files on file systems that do not support sparse files if a
  // test  crashes before resizing the file.
  static std::once_flag Flag;
  std::call_once(Flag, [] {
    size_t Limit = 100 * 1024 * 1024;
    std::string LimitStr = std::to_string(Limit);
    setenv("LLVM_CAS_MAX_MAPPING_SIZE", LimitStr.c_str(), /*overwrite=*/false);
  });
}
#endif

TestingAndDir createOnDisk(int I) {
  unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<ObjectStore> CAS;
  EXPECT_THAT_ERROR(createOnDiskCAS(Temp.path()).moveInto(CAS), Succeeded());
  std::unique_ptr<ActionCache> Cache;
  EXPECT_THAT_ERROR(createOnDiskActionCache(Temp.path()).moveInto(Cache),
                    Succeeded());
  return TestingAndDir{std::move(CAS), std::move(Cache), std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(OnDiskCAS, CASTest, ::testing::Values(createOnDisk));


TestingAndDir createPluginCASImpl(int I) {
  using namespace llvm::unittest::cas;
  unittest::TempDir Temp("plugin-cas", /*Unique=*/true);
  std::optional<
      std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
      DBs;
  EXPECT_THAT_ERROR(
      createPluginCASDatabases(getCASPluginPath(), Temp.path(), {})
          .moveInto(DBs),
      Succeeded());
  return TestingAndDir{std::move(DBs->first), std::move(DBs->second),
                       std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(PluginCAS, CASTest,
                         ::testing::Values(createPluginCASImpl));
#endif /* LLVM_ENABLE_ONDISK_CAS */
