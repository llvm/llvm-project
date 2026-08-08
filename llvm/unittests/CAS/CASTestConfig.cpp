//===- CASTestConfig.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CASTestConfig.h"
#include "OnDiskCommonUtils.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <mutex>

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::unittest::cas;

// See llvm/utils/unittest/UnitTestMain/TestMain.cpp
extern const char *TestMainArgv0;

// Just a reachable symbol to ease resolving of the executable's path.
static std::string TestStringArg1("castest-string-arg1");

std::string unittest::cas::getCASPluginPath() {
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &TestStringArg1);
  llvm::SmallString<256> PathBuf(sys::path::parent_path(
      sys::path::parent_path(sys::path::parent_path(Executable))));
#ifndef _WIN32
  std::string LibName = "libCASPluginTest";
  sys::path::append(PathBuf, "lib", LibName + LLVM_PLUGIN_EXT);
#else
  std::string LibName = "CASPluginTest";
  sys::path::append(PathBuf, "bin", LibName + LLVM_PLUGIN_EXT);
#endif
  return std::string(PathBuf);
}

Expected<ObjectID> CustomHasherOnDiskCASTest::store(OnDiskGraphDB &DB,
                                                    StringRef Data,
                                                    ArrayRef<ObjectID> Refs) {
  ObjectID ID = digest(DB, Data, Refs);
  if (Error E = DB.store(ID, Refs, arrayRefFromStringRef<char>(Data)))
    return std::move(E);
  return ID;
}

CustomHasherOnDiskCASTest::HashType
CustomHasherOnDiskCASTest::digest(StringRef Data,
                                  ArrayRef<ArrayRef<uint8_t>> RefHashes) {
  HashType Digest;
  auto HashFn = GetParam().HashFn;
  HashFn(RefHashes, arrayRefFromStringRef<char>(Data), Digest);
  return Digest;
}

ObjectID CustomHasherOnDiskCASTest::digest(OnDiskGraphDB &DB, StringRef Data,
                                           ArrayRef<ObjectID> Refs) {
  SmallVector<ArrayRef<uint8_t>, 8> RefHashes;
  for (ObjectID Ref : Refs)
    RefHashes.push_back(DB.getDigest(Ref));
  HashType Digest = digest(Data, RefHashes);
  std::optional<ObjectID> ID;
  EXPECT_THAT_ERROR(DB.getReference(Digest).moveInto(ID), Succeeded());
  return *ID;
}

CustomHasherOnDiskCASTest::HashType
CustomHasherOnDiskCASTest::digest(StringRef Data) {
  return digest(Data, {});
}

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

void unittest::cas::setMaxOnDiskCASMappingSize() {
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

static void builtinDigest(ArrayRef<ArrayRef<uint8_t>> Refs, ArrayRef<char> Data,
                          SmallVectorImpl<uint8_t> &Result) {
  llvm::unittest::cas::HashType Digest =
      llvm::unittest::cas::digest(toStringRef(Data), Refs);
  Result.assign(Digest.begin(), Digest.end());
}

INSTANTIATE_TEST_SUITE_P(Builtin, CustomHasherOnDiskCASTest,
                         ::testing::Values(CustomHasherParam{
                             builtinDigest, "builtin",
                             sizeof(llvm::unittest::cas::HashType)}));

using SHA1HashType = decltype(SHA1::hash(std::declval<ArrayRef<uint8_t> &>()));

static void sha1Digest(ArrayRef<ArrayRef<uint8_t>> Refs, ArrayRef<char> Data,
                       SmallVectorImpl<uint8_t> &Result) {
  SHA1HashType Digest = BuiltinObjectHasher<SHA1>::hashObject(Refs, Data);
  Result.assign(Digest.begin(), Digest.end());
}

INSTANTIATE_TEST_SUITE_P(SHA1, CustomHasherOnDiskCASTest,
                         ::testing::Values(CustomHasherParam{
                             sha1Digest, "SHA1", sizeof(SHA1HashType)}));

// HWASan does not tag the globals of a dlopen'ed library with glibc, so the
// plugin faults as soon as it touches one of its own globals.
// FIXME: Re-enable once https://github.com/llvm/llvm-project/issues/57206 is
// fixed.
#if !LLVM_HWADDRESS_SANITIZER_BUILD
static CASTestingEnv createPlugin(int I) {
  unittest::TempDir Temp("plugin-cas", /*Unique=*/true);
  std::optional<
      std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
      DBs;
  EXPECT_THAT_ERROR(createPluginCASDatabases(getCASPluginPath(), Temp.path(),
                                             /*PluginArgs=*/{})
                        .moveInto(DBs),
                    Succeeded());
  if (!DBs)
    return CASTestingEnv{nullptr, nullptr, std::move(Temp)};
  return CASTestingEnv{std::move(DBs->first), std::move(DBs->second),
                       std::move(Temp)};
}
INSTANTIATE_TEST_SUITE_P(PluginCAS, CASTest, ::testing::Values(createPlugin));
#endif

#else
void unittest::cas::setMaxOnDiskCASMappingSize() {}
#endif /* LLVM_ENABLE_ONDISK_CAS */
