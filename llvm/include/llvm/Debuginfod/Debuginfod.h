//===-- llvm/Debuginfod/Debuginfod.h - Debuginfod client --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of getCachedOrDownloadArtifact and
/// several convenience functions for specific artifact types:
/// getCachedOrDownloadSource, getCachedOrDownloadExecutable, and
/// getCachedOrDownloadDebuginfo. This file also declares
/// getDefaultDebuginfodUrls and getDefaultDebuginfodCacheDirectory.
///
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFOD_DEBUGINFOD_H
#define LLVM_DEBUGINFOD_DEBUGINFOD_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

typedef ArrayRef<uint8_t> BuildIDRef;

typedef SmallVector<uint8_t, 10> BuildID;

/// Finds default array of Debuginfod server URLs by checking DEBUGINFOD_URLS
/// environment variable.
Expected<SmallVector<StringRef>> getDefaultDebuginfodUrls();

/// Finds a default local file caching directory for the debuginfod client,
/// first checking DEBUGINFOD_CACHE_PATH.
Expected<std::string> getDefaultDebuginfodCacheDirectory();

/// Finds a default timeout for debuginfod HTTP requests. Checks
/// DEBUGINFOD_TIMEOUT environment variable, default is 90 seconds (90000 ms).
std::chrono::milliseconds getDefaultDebuginfodTimeout();

/// Fetches a specified source file by searching the default local cache
/// directory and server URLs.
Expected<std::string> getCachedOrDownloadSource(BuildIDRef ID,
                                                StringRef SourceFilePath);

/// Fetches an executable by searching the default local cache directory and
/// server URLs.
Expected<std::string> getCachedOrDownloadExecutable(BuildIDRef ID);

/// Fetches a debug binary by searching the default local cache directory and
/// server URLs.
Expected<std::string> getCachedOrDownloadDebuginfo(BuildIDRef ID);

/// Fetches any debuginfod artifact using the default local cache directory and
/// server URLs.
Expected<std::string> getCachedOrDownloadArtifact(StringRef UniqueKey,
                                                  StringRef UrlPath);

/// Fetches any debuginfod artifact using the specified local cache directory,
/// server URLs, and request timeout (in milliseconds). If the artifact is
/// found, uses the UniqueKey for the local cache file.
Expected<std::string> getCachedOrDownloadArtifact(
    StringRef UniqueKey, StringRef UrlPath, StringRef CacheDirectoryPath,
    ArrayRef<StringRef> DebuginfodUrls, std::chrono::milliseconds Timeout);

} // end namespace llvm

#endif
