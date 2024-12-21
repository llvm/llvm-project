//===-- llvm/unittest/Support/DebuginfodTests.cpp - unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

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

using namespace llvm;

// Check that the Debuginfod client can find locally cached artifacts.
TEST(DebuginfodClient, CacheHit) {
  int FD;
  SmallString<64> CachedFilePath;
  sys::fs::createTemporaryFile("llvmcache-key", "temp", FD, CachedFilePath);
  StringRef CacheDir = sys::path::parent_path(CachedFilePath);
  StringRef UniqueKey = sys::path::filename(CachedFilePath);
  llvm::CachePruningPolicy policy;
  EXPECT_TRUE(UniqueKey.consume_front("llvmcache-"));
  raw_fd_ostream OF(FD, true, /*unbuffered=*/true);
  OF << "contents\n";
  OF << CacheDir << "\n";
  OF.close();
  Expected<std::string> PathOrErr = getCachedOrDownloadArtifact(
      UniqueKey, /*UrlPath=*/"/null", CacheDir,
      /*DebuginfodUrls=*/{}, /*Timeout=*/std::chrono::milliseconds(1), policy);
  EXPECT_THAT_EXPECTED(PathOrErr, HasValue(CachedFilePath));
}

// Check that the Debuginfod client returns an Error when it fails to find an
// artifact.
TEST(DebuginfodClient, CacheMiss) {
  SmallString<32> CacheDir;
  ASSERT_NO_ERROR(
      sys::fs::createUniqueDirectory("debuginfod-unittest", CacheDir));
  sys::path::append(CacheDir, "cachedir");
  ASSERT_FALSE(sys::fs::exists(CacheDir));
  setenv("DEBUGINFOD_CACHE_PATH", CacheDir.c_str(),
         /*replace=*/1);
  // Ensure there are no urls to guarantee a cache miss.
  setenv("DEBUGINFOD_URLS", "", /*replace=*/1);
  HTTPClient::initialize();
  Expected<std::string> PathOrErr = getCachedOrDownloadArtifact(
      /*UniqueKey=*/"nonexistent-key", /*UrlPath=*/"/null");
  EXPECT_THAT_EXPECTED(PathOrErr, Failed<StringError>());
  // A cache miss with no possible URLs should not create the cache directory.
  EXPECT_FALSE(sys::fs::exists(CacheDir));
}
