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

CASTestingEnv createInMemory(int I) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  return CASTestingEnv{std::move(CAS)};
}

INSTANTIATE_TEST_SUITE_P(InMemoryCAS, CASTest,
                         ::testing::Values(createInMemory));
