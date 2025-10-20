//===- CASTestConfig.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "llvm/CAS/ObjectStore.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

static CASTestingEnv createInMemory(int I) {
  return CASTestingEnv{createInMemoryCAS(), createInMemoryActionCache()};
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));

#if LLVM_ENABLE_ONDISK_CAS
namespace llvm::cas::ondisk {
extern void setMaxMappingSize(uint64_t Size);
} // namespace llvm::cas::ondisk

void setMaxOnDiskCASMappingSize() {
  static std::once_flag Flag;
  std::call_once(
      Flag, [] { llvm::cas::ondisk::setMaxMappingSize(100 * 1024 * 1024); });
}
#else
void setMaxOnDiskCASMappingSize() {}
#endif /* LLVM_ENABLE_ONDISK_CAS */
