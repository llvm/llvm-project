//===- CASTestConfig.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;

static CASTestingEnv createInMemory(int I) {
  return CASTestingEnv{createInMemoryCAS(), createInMemoryActionCache(),
                       std::nullopt};
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));

#if LLVM_ENABLE_ONDISK_CAS
namespace llvm::cas::ondisk {
void setMaxMappingSize(uint64_t Size);
} // namespace llvm::cas::ondisk

void setMaxOnDiskCASMappingSize() {
  static std::once_flag Flag;
  std::call_once(
      Flag, [] { llvm::cas::ondisk::setMaxMappingSize(100 * 1024 * 1024); });
}

static CASTestingEnv createOnDisk(int I) {
  unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<ObjectStore> CAS;
  EXPECT_THAT_ERROR(createOnDiskCAS(Temp.path()).moveInto(CAS), Succeeded());
  std::unique_ptr<ActionCache> Cache;
  EXPECT_THAT_ERROR(createOnDiskActionCache(Temp.path()).moveInto(Cache),
                    Succeeded());
  return CASTestingEnv{std::move(CAS), std::move(Cache), std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(OnDiskCAS, CASTest, ::testing::Values(createOnDisk));
#else
void setMaxOnDiskCASMappingSize() {}
#endif /* LLVM_ENABLE_ONDISK_CAS */
