//===- CASTestConfig.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "gtest/gtest.h"

#ifndef LLVM_UNITTESTS_CASTESTCONFIG_H
#define LLVM_UNITTESTS_CASTESTCONFIG_H

struct CASTestingEnv {
  std::unique_ptr<llvm::cas::ObjectStore> CAS;
  std::unique_ptr<llvm::cas::ActionCache> Cache;
};

class CASTest
    : public testing::TestWithParam<std::function<CASTestingEnv(int)>> {
protected:
  std::optional<int> NextCASIndex;

  std::unique_ptr<llvm::cas::ObjectStore> createObjectStore() {
    auto TD = GetParam()(++(*NextCASIndex));
    return std::move(TD.CAS);
  }
  std::unique_ptr<llvm::cas::ActionCache> createActionCache() {
    auto TD = GetParam()(++(*NextCASIndex));
    return std::move(TD.Cache);
  }
  void SetUp() { NextCASIndex = 0; }
  void TearDown() { NextCASIndex = std::nullopt; }
};

#endif
