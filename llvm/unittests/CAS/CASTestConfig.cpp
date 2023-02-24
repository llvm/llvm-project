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
#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "gtest/gtest.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

namespace llvm::unittest::cas {
void MockEnv::anchor() {}
MockEnv::~MockEnv() {}
} // namespace llvm::unittest::cas

TestingAndDir createInMemory(int I) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  std::unique_ptr<ActionCache> Cache = createInMemoryActionCache();
  return TestingAndDir{std::move(CAS), std::move(Cache), nullptr, None};
}

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

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));

#if LLVM_ENABLE_ONDISK_CAS
TestingAndDir createOnDisk(int I) {
  unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<ObjectStore> CAS;
  EXPECT_THAT_ERROR(createOnDiskCAS(Temp.path()).moveInto(CAS), Succeeded());
  std::unique_ptr<ActionCache> Cache;
  EXPECT_THAT_ERROR(createOnDiskActionCache(Temp.path()).moveInto(Cache),
                    Succeeded());
  return TestingAndDir{std::move(CAS), std::move(Cache), nullptr,
                       std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(OnDiskCAS, CASTest, ::testing::Values(createOnDisk));
#endif /* LLVM_ENABLE_ONDISK_CAS */

#if LLVM_CAS_ENABLE_REMOTE_CACHE
std::unique_ptr<unittest::cas::MockEnv> createGRPCEnv(StringRef Socket,
                                                      StringRef TempDir);

static TestingAndDir createGRPCCAS(int I) {
  std::unique_ptr<ObjectStore> CAS;
  unittest::TempDir Temp("daemon", /*Unique=*/true);
  SmallString<100> DaemonPath(Temp.path());
  sys::path::append(DaemonPath, "grpc");
  auto Env = createGRPCEnv(DaemonPath, Temp.path());
  EXPECT_THAT_ERROR(createGRPCRelayCAS(DaemonPath).moveInto(CAS), Succeeded());
  std::unique_ptr<ActionCache> Cache;
  EXPECT_THAT_ERROR(createGRPCActionCache(DaemonPath).moveInto(Cache),
                    Succeeded());
  return TestingAndDir{std::move(CAS), std::move(Cache), std::move(Env),
                       std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(GRPCRelayCAS, CASTest,
                         ::testing::Values(createGRPCCAS));
#endif /* LLVM_CAS_ENABLE_REMOTE_CACHE */
