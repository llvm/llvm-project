//===-- llvm/Debuginfod/Debuginfod.cpp - Debuginfod client library --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file defines the fetchInfo function, which retrieves
/// any of the three supported artifact types: (executable, debuginfo, source
/// file) associated with a build-id from debuginfod servers. If a source file
/// is to be fetched, its absolute path must be specified in the Description
/// argument to fetchInfo.
///
//===----------------------------------------------------------------------===//

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/xxhash.h"

namespace llvm {
static std::string uniqueKey(llvm::StringRef S) { return utostr(xxHash64(S)); }

// Returns a binary BuildID as a normalized hex string.
// Uses lowercase for compatibility with common debuginfod servers.
static std::string buildIDToString(BuildIDRef ID) {
  return llvm::toHex(ID, /*LowerCase=*/true);
}

Expected<SmallVector<StringRef>> getDefaultDebuginfodUrls() {
  const char *DebuginfodUrlsEnv = std::getenv("DEBUGINFOD_URLS");
  if (DebuginfodUrlsEnv == nullptr)
    return SmallVector<StringRef>();

  SmallVector<StringRef> DebuginfodUrls;
  StringRef(DebuginfodUrlsEnv).split(DebuginfodUrls, " ");
  return DebuginfodUrls;
}

Expected<std::string> getDefaultDebuginfodCacheDirectory() {
  if (const char *CacheDirectoryEnv = std::getenv("DEBUGINFOD_CACHE_PATH"))
    return CacheDirectoryEnv;

  SmallString<64> CacheDirectory;
  if (!sys::path::cache_directory(CacheDirectory))
    return createStringError(
        errc::io_error, "Unable to determine appropriate cache directory.");
  sys::path::append(CacheDirectory, "llvm-debuginfod", "client");
  return std::string(CacheDirectory);
}

std::chrono::milliseconds getDefaultDebuginfodTimeout() {
  long Timeout;
  const char *DebuginfodTimeoutEnv = std::getenv("DEBUGINFOD_TIMEOUT");
  if (DebuginfodTimeoutEnv &&
      to_integer(StringRef(DebuginfodTimeoutEnv).trim(), Timeout, 10))
    return std::chrono::milliseconds(Timeout * 1000);

  return std::chrono::milliseconds(90 * 1000);
}

/// The following functions fetch a debuginfod artifact to a file in a local
/// cache and return the cached file path. They first search the local cache,
/// followed by the debuginfod servers.

Expected<std::string> getCachedOrDownloadSource(BuildIDRef ID,
                                                StringRef SourceFilePath) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "source",
                    sys::path::convert_to_slash(SourceFilePath));
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

Expected<std::string> getCachedOrDownloadExecutable(BuildIDRef ID) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "executable");
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

Expected<std::string> getCachedOrDownloadDebuginfo(BuildIDRef ID) {
  SmallString<64> UrlPath;
  sys::path::append(UrlPath, sys::path::Style::posix, "buildid",
                    buildIDToString(ID), "debuginfo");
  return getCachedOrDownloadArtifact(uniqueKey(UrlPath), UrlPath);
}

// General fetching function.
Expected<std::string> getCachedOrDownloadArtifact(StringRef UniqueKey,
                                                  StringRef UrlPath) {
  SmallString<10> CacheDir;

  Expected<std::string> CacheDirOrErr = getDefaultDebuginfodCacheDirectory();
  if (!CacheDirOrErr)
    return CacheDirOrErr.takeError();
  CacheDir = *CacheDirOrErr;

  Expected<SmallVector<StringRef>> DebuginfodUrlsOrErr =
      getDefaultDebuginfodUrls();
  if (!DebuginfodUrlsOrErr)
    return DebuginfodUrlsOrErr.takeError();
  SmallVector<StringRef> &DebuginfodUrls = *DebuginfodUrlsOrErr;
  return getCachedOrDownloadArtifact(UniqueKey, UrlPath, CacheDir,
                                     DebuginfodUrls,
                                     getDefaultDebuginfodTimeout());
}

Expected<std::string> getCachedOrDownloadArtifact(
    StringRef UniqueKey, StringRef UrlPath, StringRef CacheDirectoryPath,
    ArrayRef<StringRef> DebuginfodUrls, std::chrono::milliseconds Timeout) {
  SmallString<64> AbsCachedArtifactPath;
  sys::path::append(AbsCachedArtifactPath, CacheDirectoryPath,
                    "llvmcache-" + UniqueKey);

  Expected<FileCache> CacheOrErr =
      localCache("Debuginfod-client", ".debuginfod-client", CacheDirectoryPath);
  if (!CacheOrErr)
    return CacheOrErr.takeError();

  FileCache Cache = *CacheOrErr;
  // We choose an arbitrary Task parameter as we do not make use of it.
  unsigned Task = 0;
  Expected<AddStreamFn> CacheAddStreamOrErr = Cache(Task, UniqueKey);
  if (!CacheAddStreamOrErr)
    return CacheAddStreamOrErr.takeError();
  AddStreamFn &CacheAddStream = *CacheAddStreamOrErr;
  if (!CacheAddStream)
    return std::string(AbsCachedArtifactPath);
  // The artifact was not found in the local cache, query the debuginfod
  // servers.
  if (!HTTPClient::isAvailable())
    return createStringError(errc::io_error,
                             "No working HTTP client is available.");

  if (!HTTPClient::IsInitialized)
    return createStringError(
        errc::io_error,
        "A working HTTP client is available, but it is not initialized. To "
        "allow Debuginfod to make HTTP requests, call HTTPClient::initialize() "
        "at the beginning of main.");

  HTTPClient Client;
  Client.setTimeout(Timeout);
  for (StringRef ServerUrl : DebuginfodUrls) {
    SmallString<64> ArtifactUrl;
    sys::path::append(ArtifactUrl, sys::path::Style::posix, ServerUrl, UrlPath);

    Expected<HTTPResponseBuffer> ResponseOrErr = Client.get(ArtifactUrl);
    if (!ResponseOrErr)
      return ResponseOrErr.takeError();

    HTTPResponseBuffer &Response = *ResponseOrErr;
    if (Response.Code != 200)
      continue;

    // We have retrieved the artifact from this server, and now add it to the
    // file cache.
    Expected<std::unique_ptr<CachedFileStream>> FileStreamOrErr =
        CacheAddStream(Task);
    if (!FileStreamOrErr)
      return FileStreamOrErr.takeError();
    std::unique_ptr<CachedFileStream> &FileStream = *FileStreamOrErr;
    if (!Response.Body)
      return createStringError(
          errc::io_error, "Unallocated MemoryBuffer in HTTPResponseBuffer.");

    *FileStream->OS << StringRef(Response.Body->getBufferStart(),
                                 Response.Body->getBufferSize());

    // Return the path to the artifact on disk.
    return std::string(AbsCachedArtifactPath);
  }

  return createStringError(errc::argument_out_of_domain, "build id not found");
}
} // namespace llvm
