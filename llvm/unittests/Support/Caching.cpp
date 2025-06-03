//===- Caching.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Caching.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

char data[] = "some data";

TEST(Caching, Normal) {
  SmallString<256> CacheDir;
  sys::fs::createUniquePath("llvm_test_cache-%%%%%%", CacheDir, true);

  sys::fs::remove_directories(CacheDir.str());

  std::unique_ptr<MemoryBuffer> CachedBuffer;
  auto AddBuffer = [&CachedBuffer](unsigned Task, const Twine &ModuleName,
                                   std::unique_ptr<MemoryBuffer> M) {
    CachedBuffer = std::move(M);
  };
  auto CacheOrErr =
      localCache("LLVMTestCache", "LLVMTest", CacheDir, AddBuffer);
  ASSERT_TRUE(bool(CacheOrErr));

  FileCache &Cache = *CacheOrErr;

  {
    auto AddStreamOrErr = Cache(1, "foo", "");
    ASSERT_TRUE(bool(AddStreamOrErr));

    AddStreamFn &AddStream = *AddStreamOrErr;
    ASSERT_TRUE(AddStream);

    auto FileOrErr = AddStream(1, "");
    ASSERT_TRUE(bool(FileOrErr));

    CachedFileStream *CFS = FileOrErr->get();
    (*CFS->OS).write(data, sizeof(data));
    ASSERT_THAT_ERROR(CFS->commit(), Succeeded());
  }

  {
    auto AddStreamOrErr = Cache(1, "foo", "");
    ASSERT_TRUE(bool(AddStreamOrErr));

    AddStreamFn &AddStream = *AddStreamOrErr;
    ASSERT_FALSE(AddStream);

    ASSERT_TRUE(CachedBuffer->getBufferSize() == sizeof(data));
    StringRef readData = CachedBuffer->getBuffer();
    ASSERT_TRUE(memcmp(data, readData.data(), sizeof(data)) == 0);
  }

  ASSERT_NO_ERROR(sys::fs::remove_directories(CacheDir.str()));
}

TEST(Caching, WriteAfterCommit) {
  SmallString<256> CacheDir;
  sys::fs::createUniquePath("llvm_test_cache-%%%%%%", CacheDir, true);

  sys::fs::remove_directories(CacheDir.str());

  std::unique_ptr<MemoryBuffer> CachedBuffer;
  auto AddBuffer = [&CachedBuffer](unsigned Task, const Twine &ModuleName,
                                   std::unique_ptr<MemoryBuffer> M) {
    CachedBuffer = std::move(M);
  };
  auto CacheOrErr =
      localCache("LLVMTestCache", "LLVMTest", CacheDir, AddBuffer);
  ASSERT_TRUE(bool(CacheOrErr));

  FileCache &Cache = *CacheOrErr;

  auto AddStreamOrErr = Cache(1, "foo", "");
  ASSERT_TRUE(bool(AddStreamOrErr));

  AddStreamFn &AddStream = *AddStreamOrErr;
  ASSERT_TRUE(AddStream);

  auto FileOrErr = AddStream(1, "");
  ASSERT_TRUE(bool(FileOrErr));

  CachedFileStream *CFS = FileOrErr->get();
  (*CFS->OS).write(data, sizeof(data));
  ASSERT_THAT_ERROR(CFS->commit(), Succeeded());

  EXPECT_DEATH(
      { (*CFS->OS).write(data, sizeof(data)); }, "")
      << "Write after commit did not cause abort";

  ASSERT_NO_ERROR(sys::fs::remove_directories(CacheDir.str()));
}

TEST(Caching, NoCommit) {
  SmallString<256> CacheDir;
  sys::fs::createUniquePath("llvm_test_cache-%%%%%%", CacheDir, true);

  sys::fs::remove_directories(CacheDir.str());

  std::unique_ptr<MemoryBuffer> CachedBuffer;
  auto AddBuffer = [&CachedBuffer](unsigned Task, const Twine &ModuleName,
                                   std::unique_ptr<MemoryBuffer> M) {
    CachedBuffer = std::move(M);
  };
  auto CacheOrErr =
      localCache("LLVMTestCache", "LLVMTest", CacheDir, AddBuffer);
  ASSERT_TRUE(bool(CacheOrErr));

  FileCache &Cache = *CacheOrErr;

  auto AddStreamOrErr = Cache(1, "foo", "");
  ASSERT_TRUE(bool(AddStreamOrErr));

  AddStreamFn &AddStream = *AddStreamOrErr;
  ASSERT_TRUE(AddStream);

  auto FileOrErr = AddStream(1, "");
  ASSERT_TRUE(bool(FileOrErr));

  CachedFileStream *CFS = FileOrErr->get();
  (*CFS->OS).write(data, sizeof(data));
  ASSERT_THAT_ERROR(CFS->commit(), Succeeded());

  EXPECT_DEATH(
      {
        auto FileOrErr = AddStream(1, "");
        ASSERT_TRUE(bool(FileOrErr));

        CachedFileStream *CFS = FileOrErr->get();
        (*CFS->OS).write(data, sizeof(data));
      },
      "")
      << "destruction without commit did not cause error";

  ASSERT_NO_ERROR(sys::fs::remove_directories(CacheDir.str()));
}
