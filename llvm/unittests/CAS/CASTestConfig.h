//===- CASTestConfig.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASDB.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

struct TestingAndDir {
  std::unique_ptr<llvm::cas::CASDB> DB;
  llvm::Optional<llvm::unittest::TempDir> Temp;
};

class CASDBTest
    : public testing::TestWithParam<std::function<TestingAndDir(int)>> {
protected:
  llvm::Optional<int> NextCASIndex;

  llvm::SmallVector<llvm::unittest::TempDir> Dirs;

  std::unique_ptr<llvm::cas::CASDB> createCAS() {
    auto TD = GetParam()((*NextCASIndex)++);
    if (TD.Temp)
      Dirs.push_back(std::move(*TD.Temp));
    return std::move(TD.DB);
  }
  void SetUp() { NextCASIndex = 0; }
  void TearDown() {
    NextCASIndex = llvm::None;
    Dirs.clear();
  }
};

#if LLVM_ENABLE_ONDISK_CAS
static TestingAndDir createOnDisk(int I) {
  llvm::unittest::TempDir Temp("on-disk-cas", /*Unique=*/true);
  std::unique_ptr<llvm::cas::CASDB> CAS;
  EXPECT_THAT_ERROR(llvm::cas::createOnDiskCAS(Temp.path()).moveInto(CAS),
                    llvm::Succeeded());
  return TestingAndDir{std::move(CAS), std::move(Temp)};
}
#endif /* LLVM_ENABLE_ONDISK_CAS */
