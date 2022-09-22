//===- CASTestConfig.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#ifndef LLVM_UNITTESTS_CASTESTCONFIG_H
#define LLVM_UNITTESTS_CASTESTCONFIG_H

struct TestingAndDir {
  std::unique_ptr<llvm::cas::ObjectStore> CAS;
  std::unique_ptr<llvm::cas::ActionCache> Cache;
  llvm::Optional<llvm::unittest::TempDir> Temp;
};

class CASTest
    : public testing::TestWithParam<std::function<TestingAndDir(int)>> {
protected:
  llvm::Optional<int> NextCASIndex;

  llvm::SmallVector<llvm::unittest::TempDir> Dirs;

  std::unique_ptr<llvm::cas::ObjectStore> createObjectStore() {
    auto TD = GetParam()(++(*NextCASIndex));
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    return std::move(TD.CAS);
  }
  std::unique_ptr<llvm::cas::ActionCache> createActionCache() {
    auto TD = GetParam()(++(*NextCASIndex));
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    return std::move(TD.Cache);
  }
  void SetUp() { NextCASIndex = 0; }
  void TearDown() {
    NextCASIndex = llvm::None;
    Dirs.clear();
  }
};

#endif
