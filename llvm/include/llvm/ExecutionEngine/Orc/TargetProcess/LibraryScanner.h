//===- LibraryScanner.h - Scan Library -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for scanning dynamic library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
namespace orc {

class LibraryManager;

class LibraryPathCache {
  friend class PathResolver;

public:
  LibraryPathCache() = default;

  void clear();

  void markSeen(const std::string &canon_path) { m_seen.insert(canon_path); }

  bool hasSeen(StringRef canon_path, bool cache = true) {
    std::shared_lock lock(m_mutex);
    std::string s = canon_path.str();
    if (m_seen.count(s) > 0)
      return true;
    if (cache)
      markSeen(s);
    return false;
  }

private:
  mutable std::shared_mutex m_mutex;

  struct PathInfo {
    std::string canonicalPath;
    std::error_code errnoCode;
  };

  std::unordered_map<std::string, std::string> m_readlinkCache;
  std::unordered_map<std::string, PathInfo> m_realpathCache;
  std::unordered_map<std::string, mode_t> m_lstatCache;
  std::unordered_set<std::string> m_seen;
};

/// Resolves file system paths with optional caching of results.
///
/// Supports lstat, readlink, and realpath operations. Can resolve paths
/// relative to a base and handle symbolic links. Caches results to reduce
/// repeated system calls when enabled.
class PathResolver {
public:
  PathResolver(std::shared_ptr<LibraryPathCache> cache)
      : m_cache(std::move(cache)) {}

  std::optional<std::string> resolve(const std::string &path,
                                     std::error_code &ec) {
    return realpathCached(path, ec);
  }
  mode_t lstatCached(const std::string &path);
  std::optional<std::string> readlinkCached(const std::string &path);
  std::optional<std::string> realpathCached(StringRef path, std::error_code &ec,
                                            StringRef base = "",
                                            bool baseIsResolved = false,
                                            long symloopLevel = 40);

private:
  mutable std::shared_mutex m_mutex;
  std::shared_ptr<LibraryPathCache> m_cache;
};

class LibraryScanHelper;

class DylibPathResolver {
public:
  DylibPathResolver(LibraryScanHelper &m_helper) : m_helper(m_helper) {}

  /// Resolve a dynamic library path considering RPath, RunPath, and
  /// substitutions.
  std::optional<std::string> resolve(StringRef libStem,
                                     SmallVector<StringRef, 2> RPath = {},
                                     SmallVector<StringRef, 2> RunPath = {},
                                     StringRef libLoader = "",
                                     bool variateLibStem = true);

private:
  LibraryScanHelper &m_helper;

  std::optional<std::string> substOne(StringRef path, StringRef pattern,
                                      StringRef replacement);

  /// Apply all known loader substitutions to the path
  std::optional<std::string> substAll(StringRef path, StringRef loaderPath);

  std::optional<std::string> tryWithBasePaths(ArrayRef<StringRef> basePaths,
                                              StringRef stem,
                                              StringRef loaderPath);

  /// Try resolving the path using RPATH, searchPaths, and RUNPATH (in that
  /// order)
  std::optional<std::string> tryAllPaths(StringRef stem,
                                         ArrayRef<StringRef> RPath,
                                         ArrayRef<StringRef> RunPath,
                                         StringRef loaderPath);

  std::optional<std::string> tryWithExtensions(StringRef baseName,
                                               ArrayRef<StringRef> RPath,
                                               ArrayRef<StringRef> RunPath,
                                               StringRef loaderPath);

  std::optional<std::string> normalizeIfShared(StringRef path);
};

enum class PathKind { User, System };

enum class ScanState { NotScanned, Scanning, Scanned };

struct LibraryUnit {
  std::string basePath; // Canonical base directory path
  PathKind kind;        // User or System
  std::atomic<ScanState> state;

  LibraryUnit(std::string base, PathKind k)
      : basePath(std::move(base)), kind(k), state(ScanState::NotScanned) {}
};

/// Scans and tracks libraries for symbol resolution.
///
/// Maintains a list of library paths to scan, caches scanned units,
/// and resolves paths canonically for consistent tracking.
class LibraryScanHelper {
public:
  explicit LibraryScanHelper(const std::vector<std::string> &paths,
                             std::shared_ptr<LibraryPathCache> m_cache,
                             std::shared_ptr<PathResolver> m_resolver)
      : m_cache(std::move(m_cache)), m_resolver(std::move(m_resolver)) {
    for (const auto &p : paths)
      addBasePath(p);
  }

  void addBasePath(
      const std::string &path); // Add a canonical directory for scanning
  std::vector<std::shared_ptr<LibraryUnit>> getNextBatch(PathKind kind,
                                                         size_t batchSize);

  bool isTrackedBasePath(const std::string &path) const;
  std::vector<std::shared_ptr<LibraryUnit>> getAllUnits() const;

  PathResolver &getPathResolver() const { return *m_resolver; }

  LibraryPathCache &getCache() const { return *m_cache; }

  bool hasSeen(StringRef path) const { return m_cache->hasSeen(path); }

  std::optional<std::string> resolve(StringRef path,
                                     std::error_code &ec) const {
    return m_resolver->resolve(path.str(), ec);
  }

private:
  std::string resolveCanonical(const std::string &path,
                               std::error_code &ec) const;
  PathKind classifyKind(const std::string &path) const;

  mutable std::shared_mutex m_fileMutex;
  mutable std::shared_mutex m_mutex;
  std::shared_ptr<LibraryPathCache> m_cache;
  std::shared_ptr<PathResolver> m_resolver;

  std::unordered_map<std::string, std::shared_ptr<LibraryUnit>>
      m_units; // key: canonical path
  std::deque<std::string> m_unscannedUsr;
  std::deque<std::string> m_unscannedSys;
};

class LibraryScanner {
public:
  using shouldScanFn = std::function<bool(StringRef)>;

  LibraryScanner(
      LibraryScanHelper &H, LibraryManager &m_libMgr,
      shouldScanFn shouldScanCall = [](StringRef path) { return true; })
      : m_helper(H), m_libMgr(m_libMgr), m_libResolver(DylibPathResolver(H)),
        shouldScanCall(std::move(shouldScanCall)) {}

  void scanNext(PathKind kind, size_t batchSize = 1);

  struct LibraryDepsInfo {
    std::vector<std::string> storage;

    SmallVector<StringRef, 2> rpath;
    SmallVector<StringRef, 2> runPath;
    SmallVector<StringRef, 4> deps;
    bool isPIE = false;

    void addRPath(StringRef s) {
      storage.emplace_back(s);
      rpath.push_back(storage.back());
    }

    void addRunPath(StringRef s) {
      storage.emplace_back(s);
      runPath.push_back(storage.back());
    }

    void addDep(StringRef s) {
      storage.emplace_back(s);
      deps.push_back(storage.back());
    }
  };

private:
  LibraryScanHelper &m_helper;
  LibraryManager &m_libMgr;
  DylibPathResolver m_libResolver;
  shouldScanFn shouldScanCall;

  std::optional<std::string> shouldScan(StringRef filePath);
  Expected<LibraryDepsInfo> extractDeps(StringRef filePath);

  void handleLibrary(StringRef path, PathKind K, int level = 1);

  void scanBaseDir(std::shared_ptr<LibraryUnit> unit);
};

using LibraryDepsInfo = LibraryScanner::LibraryDepsInfo;

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H
