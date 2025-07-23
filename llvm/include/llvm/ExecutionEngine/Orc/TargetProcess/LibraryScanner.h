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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

#include <atomic>
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

  void clear() {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    m_seen.clear();
    m_realpathCache.clear();
#ifndef _WIN32
    m_readlinkCache.clear();
    m_lstatCache.clear();
#endif
  }

  void markSeen(const std::string &canon_path) {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    m_seen.insert(canon_path);
  }

  bool hasSeen(StringRef canon_path) const {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    return m_seen.contains(canon_path);
    // return m_seen.count(canon_path) > 0;
  }

  bool hasSeenOrMark(StringRef canon_path) {
    std::string s = canon_path.str();
    {
      std::shared_lock<std::shared_mutex> lock(m_mutex);
      if (m_seen.contains(s))
        return true;
    }
    {
      std::unique_lock<std::shared_mutex> lock(m_mutex);
      m_seen.insert(s);
    }
    return false;
  }

private:
  mutable std::shared_mutex m_mutex;

  struct PathInfo {
    std::string canonicalPath;
    std::error_code errnoCode;
  };

  void insert_realpath(StringRef path, const PathInfo &info) {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    m_realpathCache.insert({path, info});
  }

  std::optional<PathInfo> read_realpath(StringRef path) const {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    auto it = m_realpathCache.find(path);
    if (it != m_realpathCache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  StringSet<> m_seen;
  StringMap<PathInfo> m_realpathCache;

#ifndef _WIN32
  StringMap<std::string> m_readlinkCache;
  StringMap<mode_t> m_lstatCache;

  void insert_link(StringRef path, const std::string &s) {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    m_readlinkCache.insert({path, s});
  }

  std::optional<std::string> read_link(StringRef path) const {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    auto it = m_readlinkCache.find(path);
    if (it != m_readlinkCache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void insert_lstat(StringRef path, mode_t m) {
    std::unique_lock<std::shared_mutex> lock(m_mutex);
    m_lstatCache.insert({path, m});
  }

  std::optional<mode_t> read_lstat(StringRef path) const {
    std::shared_lock<std::shared_mutex> lock(m_mutex);
    auto it = m_lstatCache.find(path);
    if (it != m_lstatCache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

#endif
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

  std::optional<std::string> resolve(StringRef path, std::error_code &ec) {
    return realpathCached(path, ec);
  }
#ifndef _WIN32
  mode_t lstatCached(StringRef path);
  std::optional<std::string> readlinkCached(StringRef path);
#endif
  std::optional<std::string> realpathCached(StringRef path, std::error_code &ec,
                                            StringRef base = "",
                                            bool baseIsResolved = false,
                                            long symloopLevel = 40);

private:
  // mutable std::shared_mutex m_mutex;
  std::shared_ptr<LibraryPathCache> m_cache;
};

class DylibSubstitutor {
public:
  void configure(StringRef loaderPath);

  std::string substitute(StringRef input) const {
    for (const auto &[ph, value] : placeholders_) {
      if (input.starts_with(ph)) {
        return (Twine(value) + input.drop_front(ph.size())).str();
      }
    }
    return input.str();
  }

private:
  StringMap<std::string> placeholders_;
};

class DylibPathValidator {
public:
  DylibPathValidator(PathResolver &PR) : m_pathResolver(PR) {}

  static bool isSharedLibrary(StringRef path);

  std::optional<std::string> normalize(StringRef path) const {
    std::error_code ec;
    auto real = m_pathResolver.resolve(path, ec);
    if (!real || ec)
      return std::nullopt;

    return real;
  }

  std::optional<std::string> validate(StringRef path) const {
    auto realOpt = normalize(path);
    if (!realOpt)
      return std::nullopt;

    if (!isSharedLibrary(*realOpt))
      return std::nullopt;

    return realOpt;
  }

private:
  PathResolver &m_pathResolver;
};

class SearchPathResolver {
public:
  SearchPathResolver(ArrayRef<StringRef> searchPaths,
                     StringRef placeholderPrefix = "@rpath")
      : placeholderPrefix(placeholderPrefix) {
    for (auto &path : searchPaths)
      paths.emplace_back(path.str());
  }

  std::optional<std::string> resolve(StringRef stem,
                                     const DylibSubstitutor &subst,
                                     DylibPathValidator &validator) const;

private:
  std::vector<std::string> paths;
  std::string placeholderPrefix;
};

class DylibResolverImpl {
public:
  DylibResolverImpl(DylibSubstitutor substitutor, DylibPathValidator &validator,
                    std::vector<SearchPathResolver> resolvers)
      : substitutor(std::move(substitutor)), validator(validator),
        resolvers(std::move(resolvers)) {}

  std::optional<std::string> resolve(StringRef stem,
                                     bool variateLibStem = false) const;

private:
  std::optional<std::string> tryWithExtensions(StringRef libstem) const;

  DylibSubstitutor substitutor;
  DylibPathValidator &validator;
  std::vector<SearchPathResolver> resolvers;
};

class DylibResolver {
public:
  DylibResolver(DylibPathValidator &validator) : validator(validator) {}

  void configure(StringRef loaderPath, ArrayRef<StringRef> rpaths,
                 ArrayRef<StringRef> runpaths) {
    DylibSubstitutor substitutor;
    substitutor.configure(loaderPath);

    std::vector<SearchPathResolver> resolvers;
    if (!rpaths.empty())
      resolvers.emplace_back(rpaths, "@rpath");
    if (!runpaths.empty())
      resolvers.emplace_back(runpaths, "@rpath"); // still usually @rpath

    impl_ = std::make_unique<DylibResolverImpl>(
        std::move(substitutor), validator, std::move(resolvers));
  }

  std::optional<std::string> resolve(StringRef libStem,
                                     bool variateLibStem = false) const {
    if (!impl_)
      return std::nullopt;
    return impl_->resolve(libStem, variateLibStem);
  }

  static std::string resolvelinkerFlag(StringRef libStem,
                                       StringRef loaderPath) {
    // StringRef rpath("@rpath");
    // if (libStem.starts_with(rpath))
    //   return libStem.drop_front(rpath.size()).str();
    DylibSubstitutor substitutor;
    substitutor.configure(loaderPath);
    return substitutor.substitute(libStem);
  }

private:
  DylibPathValidator &validator;
  std::unique_ptr<DylibResolverImpl> impl_;
};

enum class PathType : uint8_t { User, System, Unknown };

enum class ScanState : uint8_t { NotScanned, Scanning, Scanned };

struct LibraryUnit {
  std::string basePath; // Canonical base directory path
  PathType kind;        // User or System
  std::atomic<ScanState> state;

  LibraryUnit(std::string base, PathType k)
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
    // LLVM_DEBUG(
    dbgs() << "LibraryScanHelper::LibraryScanHelper: base paths : "
           << paths.size() << "\n"; //);
    for (const auto &p : paths)
      addBasePath(p);
  }

  void
  addBasePath(const std::string &path,
              PathType Kind =
                  PathType::Unknown); // Add a canonical directory for scanning
  std::vector<std::shared_ptr<LibraryUnit>> getNextBatch(PathType kind,
                                                         size_t batchSize);

  bool leftToScan(PathType K) const;

  bool isTrackedBasePath(StringRef path) const;
  std::vector<std::shared_ptr<LibraryUnit>> getAllUnits() const;

  PathResolver &getPathResolver() const { return *m_resolver; }

  LibraryPathCache &getCache() const { return *m_cache; }

  bool hasSeenOrMark(StringRef path) const {
    return m_cache->hasSeenOrMark(path);
  }

  std::optional<std::string> resolve(StringRef path,
                                     std::error_code &ec) const {
    return m_resolver->resolve(path.str(), ec);
  }

private:
  std::string resolveCanonical(StringRef path, std::error_code &ec) const;
  PathType classifyKind(StringRef path) const;

  mutable std::shared_mutex m_mutex;
  std::shared_ptr<LibraryPathCache> m_cache;
  std::shared_ptr<PathResolver> m_resolver;

  StringMap<std::shared_ptr<LibraryUnit>> m_units; // key: canonical path
  std::deque<std::string> m_unscannedUsr;
  std::deque<std::string> m_unscannedSys;
};

class LibraryScanner {
public:
  using shouldScanFn = std::function<bool(StringRef)>;

  LibraryScanner(
      LibraryScanHelper &H, LibraryManager &m_libMgr,
      shouldScanFn shouldScanCall = [](StringRef path) { return true; })
      : m_helper(H), m_libMgr(m_libMgr),
        // m_libResolver(DylibPathResolver(H)),
        shouldScanCall(std::move(shouldScanCall)) {}

  void scanNext(PathType kind, size_t batchSize = 1);

  struct LibraryDepsInfo {
    llvm::BumpPtrAllocator Alloc;
    llvm::StringSaver Saver{Alloc};

    SmallVector<StringRef, 2> rpath;
    SmallVector<StringRef, 2> runPath;
    SmallVector<StringRef, 4> deps;
    bool isPIE = false;

    void addRPath(StringRef s) { rpath.push_back(Saver.save(s)); }

    void addRunPath(StringRef s) { runPath.push_back(Saver.save(s)); }

    void addDep(StringRef s) { deps.push_back(Saver.save(s)); }
  };

private:
  LibraryScanHelper &m_helper;
  LibraryManager &m_libMgr;
  // DylibPathResolver m_libResolver;
  shouldScanFn shouldScanCall;

  std::optional<std::string> shouldScan(StringRef filePath);
  Expected<LibraryDepsInfo> extractDeps(StringRef filePath);

  void handleLibrary(StringRef path, PathType K, int level = 1);

  void scanBaseDir(std::shared_ptr<LibraryUnit> unit);
};

using LibraryDepsInfo = LibraryScanner::LibraryDepsInfo;

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H
