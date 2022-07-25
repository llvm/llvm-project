//===- CASTestConfig.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <memory>

#ifndef LLVM_UNITTESTS_CASTESTCONFIG_H
#define LLVM_UNITTESTS_CASTESTCONFIG_H

namespace llvm::unittest::cas {
class MockEnv {
  void anchor();

public:
  virtual ~MockEnv();
};
} // namespace llvm::unittest::cas

struct TestingAndDir {
  std::unique_ptr<llvm::cas::ObjectStore> CAS;
  std::unique_ptr<llvm::cas::ActionCache> Cache;
  std::unique_ptr<llvm::unittest::cas::MockEnv> Env;
  llvm::Optional<llvm::unittest::TempDir> Temp;
};

class CASTest
    : public testing::TestWithParam<std::function<TestingAndDir(int)>> {
protected:
  llvm::Optional<int> NextCASIndex;

  llvm::SmallVector<llvm::unittest::TempDir> Dirs;

  llvm::SmallVector<std::unique_ptr<llvm::unittest::cas::MockEnv>> Envs;

  std::unique_ptr<llvm::cas::ObjectStore> createObjectStore() {
    auto TD = GetParam()(++(*NextCASIndex));
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    if (TD.Env)
      Envs.emplace_back(std::move(TD.Env));
    return std::move(TD.CAS);
  }
  std::unique_ptr<llvm::cas::ActionCache> createActionCache() {
    auto TD = GetParam()(++(*NextCASIndex));
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    if (TD.Env)
      Envs.emplace_back(std::move(TD.Env));
    return std::move(TD.Cache);
  }
  void SetUp() { NextCASIndex = 0; }
  void TearDown() {
    NextCASIndex = llvm::None;
    Dirs.clear();
    Envs.clear();
  }
};

#endif
