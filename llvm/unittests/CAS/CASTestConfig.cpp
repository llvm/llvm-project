//===- CASTestConfig.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASDB.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TestingAndDir createInMemory(int I) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  return TestingAndDir{std::move(CAS), createInMemoryActionCache, None};
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));

#if LLVM_ENABLE_ONDISK_CAS
TestingAndDir createOnDisk(int I) {
  unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<CASDB> CAS;
  EXPECT_THAT_ERROR(createOnDiskCAS(Temp.path()).moveInto(CAS), Succeeded());
  std::string TempPath = Temp.path().str();
  auto CreateFn = [&, TempPath](CASDB &CAS) -> std::unique_ptr<ActionCache> {
    std::unique_ptr<ActionCache> Cache;
    EXPECT_THAT_ERROR(createOnDiskActionCache(CAS, TempPath).moveInto(Cache),
                      Succeeded());
    return Cache;
  };
  return TestingAndDir{std::move(CAS), CreateFn, std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(OnDiskCAS, CASTest, ::testing::Values(createOnDisk));
#endif /* LLVM_ENABLE_ONDISK_CAS */
