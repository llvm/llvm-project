//===- CASTestConfig.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_CASTESTCONFIG_H
#define LLVM_UNITTESTS_CASTESTCONFIG_H

#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <memory>

#ifdef _WIN32
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

namespace llvm::cas::ondisk {
class ObjectID;
class OnDiskGraphDB;
} // namespace llvm::cas::ondisk

namespace llvm::unittest::cas {

using namespace llvm::cas::ondisk;

class MockEnv {
  void anchor();

public:
  virtual ~MockEnv();
};

struct CASTestingEnv {
  std::unique_ptr<llvm::cas::ObjectStore> CAS;
  std::unique_ptr<llvm::cas::ActionCache> Cache;
  std::optional<llvm::unittest::TempDir> Temp;
};

void setMaxOnDiskCASMappingSize();

// Test fixture for on-disk data base tests.
class OnDiskCASTest : public ::testing::Test {
protected:
  void SetUp() override {
#if !LLVM_ENABLE_ONDISK_CAS
    GTEST_SKIP() << "OnDiskCAS is not enabled";
#endif
    // Use a smaller database size for testing to conserve disk space.
    setMaxOnDiskCASMappingSize();
  }
};

struct CustomHasherParam {
  std::function<void(ArrayRef<ArrayRef<uint8_t>>, ArrayRef<char>,
                     SmallVectorImpl<uint8_t> &)>
      HashFn;
  std::string HashName;
  size_t HashSize;
};

// Test fixture for on-disk data base tests with custom hasher.
class CustomHasherOnDiskCASTest
    : public ::testing::TestWithParam<CustomHasherParam> {
protected:
  void SetUp() override {
#if !LLVM_ENABLE_ONDISK_CAS
    GTEST_SKIP() << "OnDiskCAS is not enabled";
#endif
    // Use a smaller database size for testing to conserve disk space.
    setMaxOnDiskCASMappingSize();
  }

  using HashType = SmallVector<uint8_t, 32>;

  Expected<ObjectID> store(OnDiskGraphDB &DB, StringRef Data,
                           ArrayRef<ObjectID> Refs);

  ObjectID digest(OnDiskGraphDB &DB, StringRef Data, ArrayRef<ObjectID> Refs);
  HashType digest(StringRef Data, ArrayRef<ArrayRef<uint8_t>> RefHashes);
  HashType digest(StringRef Data);
};

// Parametered test fixture for ObjectStore and ActionCache tests.
class CASTest
    : public testing::TestWithParam<std::function<CASTestingEnv(int)>> {
protected:
  std::optional<int> NextCASIndex;

  llvm::SmallVector<llvm::unittest::TempDir> Dirs;

  llvm::SmallVector<std::unique_ptr<llvm::unittest::cas::MockEnv>> Envs;

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

  void SetUp() override {
#ifdef _WIN32
    // Temporarily disable CAS tests on pre windows 11 OS.
    if (llvm::GetWindowsOSVersion() < llvm::VersionTuple(10, 0, 0, 22000))
      GTEST_SKIP() << "CAS tests skipped on older windows version";
#endif
    NextCASIndex = 0;
    setMaxOnDiskCASMappingSize();
  }

  void TearDown() override {
    NextCASIndex = std::nullopt;
    Dirs.clear();
    Envs.clear();
  }
};

} // namespace llvm::unittest::cas

#endif
