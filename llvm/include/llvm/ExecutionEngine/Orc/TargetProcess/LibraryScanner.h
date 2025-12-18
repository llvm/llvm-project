//===- LibraryScanner.h - Scanner for Shared Libraries ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides functionality for scanning dynamic (shared) libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"

#include <atomic>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>

namespace llvm {
namespace orc {

class LibraryManager;

class LibraryPathCache {
  friend class PathResolver;

public:
  LibraryPathCache() = default;

  void clear(bool isRealPathCache = false) {
    std::unique_lock<std::shared_mutex> lock(Mtx);
    Seen.clear();
    if (isRealPathCache) {
      RealPathCache.clear();
#ifndef _WIN32
      ReadlinkCache.clear();
      LstatCache.clear();
#endif
    }
  }

  void markSeen(const std::string &CanonPath) {
    std::unique_lock<std::shared_mutex> lock(Mtx);
    Seen.insert(CanonPath);
  }

  bool hasSeen(StringRef CanonPath) const {
    std::shared_lock<std::shared_mutex> lock(Mtx);
    return Seen.contains(CanonPath);
  }

  bool hasSeenOrMark(StringRef CanonPath) {
    std::string s = CanonPath.str();
    {
      std::shared_lock<std::shared_mutex> lock(Mtx);
      if (Seen.contains(s))
        return true;
    }
    {
      std::unique_lock<std::shared_mutex> lock(Mtx);
      Seen.insert(s);
    }
    return false;
  }

private:
  mutable std::shared_mutex Mtx;

  struct PathInfo {
    std::string canonicalPath;
    std::error_code ErrnoCode;
  };

  void insert_realpath(StringRef Path, const PathInfo &Info) {
    std::unique_lock<std::shared_mutex> lock(Mtx);
    RealPathCache.insert({Path, Info});
  }

  std::optional<PathInfo> read_realpath(StringRef Path) const {
    std::shared_lock<std::shared_mutex> lock(Mtx);
    auto It = RealPathCache.find(Path);
    if (It != RealPathCache.end())
      return It->second;

    return std::nullopt;
  }

  StringSet<> Seen;
  StringMap<PathInfo> RealPathCache;

#ifndef _WIN32
  StringMap<std::string> ReadlinkCache;
  StringMap<mode_t> LstatCache;

  void insert_link(StringRef Path, const std::string &s) {
    std::unique_lock<std::shared_mutex> lock(Mtx);
    ReadlinkCache.insert({Path, s});
  }

  std::optional<std::string> read_link(StringRef Path) const {
    std::shared_lock<std::shared_mutex> lock(Mtx);
    auto It = ReadlinkCache.find(Path);
    if (It != ReadlinkCache.end())
      return It->second;

    return std::nullopt;
  }

  void insert_lstat(StringRef Path, mode_t m) {
    std::unique_lock<std::shared_mutex> lock(Mtx);
    LstatCache.insert({Path, m});
  }

  std::optional<mode_t> read_lstat(StringRef Path) const {
    std::shared_lock<std::shared_mutex> lock(Mtx);
    auto It = LstatCache.find(Path);
    if (It != LstatCache.end())
      return It->second;

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
private:
  std::shared_ptr<LibraryPathCache> LibPathCache;

public:
  PathResolver(std::shared_ptr<LibraryPathCache> cache)
      : LibPathCache(std::move(cache)) {}

  std::optional<std::string> resolve(StringRef Path, std::error_code &ec) {
    return realpathCached(Path, ec);
  }
#ifndef _WIN32
  mode_t lstatCached(StringRef Path);
  std::optional<std::string> readlinkCached(StringRef Path);
#endif
  std::optional<std::string> realpathCached(StringRef Path, std::error_code &ec,
                                            StringRef base = "",
                                            bool baseIsResolved = false,
                                            long symloopLevel = 40);
};

/// Performs placeholder substitution in dynamic library paths.
///
/// Configures known placeholders (like @loader_path) and replaces them
/// in input paths with their resolved values.
class DylibSubstitutor {
public:
  void configure(StringRef loaderPath);

  std::string substitute(StringRef input) const {
    for (const auto &[ph, value] : Placeholders) {
      if (input.starts_with_insensitive(ph))
        return (Twine(value) + input.drop_front(ph.size())).str();
    }
    return input.str();
  }

private:
  SmallVector<std::pair<std::string, std::string>> Placeholders;
};

/// Loads an object file and provides access to it.
///
/// Owns the underlying `ObjectFile` and ensures it is valid.
/// Any errors encountered during construction are stored and
/// returned when attempting to access the file.
class ObjectFileLoader {
public:
  /// Construct an object file loader from the given path.
  explicit ObjectFileLoader(StringRef Path) {
    auto ObjOrErr = loadObjectFileWithOwnership(Path);
    if (ObjOrErr)
      Obj = std::move(*ObjOrErr);
    else {
      consumeError(std::move(Err));
      Err = ObjOrErr.takeError();
    }
  }

  ObjectFileLoader(const ObjectFileLoader &) = delete;
  ObjectFileLoader &operator=(const ObjectFileLoader &) = delete;

  ObjectFileLoader(ObjectFileLoader &&) = default;
  ObjectFileLoader &operator=(ObjectFileLoader &&) = default;

  /// Get the loaded object file, or return an error if loading failed.
  Expected<object::ObjectFile &> getObjectFile() {
    if (Err) {
      // allow the error to be taken only once
      if (ErrorTaken)
        return createStringError(inconvertibleErrorCode(),
                                 "error already taken");

      ErrorTaken = true;
      return std::move(Err);
    }
    return *Obj.getBinary();
  }

  static bool isArchitectureCompatible(const object::ObjectFile &Obj);

private:
  object::OwningBinary<object::ObjectFile> Obj;
  Error Err = Error::success();
  bool ErrorTaken = false;

  static Expected<object::OwningBinary<object::ObjectFile>>
  loadObjectFileWithOwnership(StringRef FilePath);
};

class ObjFileCache {
public:
  void insert(StringRef Path, ObjectFileLoader &&Loader) {
    Cache.insert({Path, std::move(Loader)});
  }

  // Take ownership
  std::optional<ObjectFileLoader> take(StringRef Path) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    auto It = Cache.find(Path);
    if (It == Cache.end())
      return std::nullopt;

    ObjectFileLoader L = std::move(It->second);
    Cache.erase(It);
    return std::move(L);
  }

  bool contains(StringRef Path) const { return Cache.count(Path) != 0; }

private:
  mutable std::shared_mutex Mtx;
  StringMap<ObjectFileLoader> Cache;
};

/// Validates and normalizes dynamic library paths.
///
/// Uses a `PathResolver` to resolve paths to their canonical form and
/// checks whether they point to valid shared libraries.
class DylibPathValidator {
public:
  DylibPathValidator(PathResolver &PR, LibraryPathCache &LC,
                     ObjFileCache *ObjCache = nullptr)
      : LibPathResolver(PR), LibPathCache(LC), ObjCache(ObjCache) {}

  bool isSharedLibrary(StringRef Path) const;
  bool isSharedLibraryCached(StringRef Path) const {
    if (LibPathCache.hasSeen(Path))
      return true;
    return isSharedLibrary(Path);
  }

  std::optional<std::string> normalize(StringRef Path) const {
    std::error_code ec;
    auto real = LibPathResolver.resolve(Path, ec);
    if (!real || ec)
      return std::nullopt;

    return real;
  }

  /// Validate the given path as a shared library.
  std::optional<std::string> validate(StringRef Path) const {
    if (LibPathCache.hasSeen(Path))
      return Path.str();
    auto realOpt = normalize(Path);
    if (!realOpt)
      return std::nullopt;

    if (!isSharedLibraryCached(*realOpt))
      return std::nullopt;

    return realOpt;
  }

private:
  PathResolver &LibPathResolver;
  LibraryPathCache &LibPathCache;
  mutable ObjFileCache *ObjCache;
};

enum class SearchPathType {
  RPath,
  UsrOrSys,
  RunPath,
};

struct SearchPathConfig {
  ArrayRef<StringRef> Paths;
  SearchPathType type;
};

class SearchPathResolver {
public:
  SearchPathResolver(const SearchPathConfig &Cfg,
                     StringRef PlaceholderPrefix = "")
      : Kind(Cfg.type), PlaceholderPrefix(PlaceholderPrefix) {
    Paths.reserve(Cfg.Paths.size());
    for (auto &path : Cfg.Paths)
      Paths.emplace_back(path.str());
  }

  std::optional<std::string> resolve(StringRef libStem,
                                     const DylibSubstitutor &Subst,
                                     DylibPathValidator &Validator) const;
  SearchPathType searchPathType() const { return Kind; }

private:
  std::vector<std::string> Paths;
  SearchPathType Kind;
  std::string PlaceholderPrefix;
};

class DylibResolverImpl {
public:
  DylibResolverImpl(DylibSubstitutor Substitutor, DylibPathValidator &Validator,
                    std::vector<SearchPathResolver> Resolvers)
      : Substitutor(std::move(Substitutor)), Validator(Validator),
        Resolvers(std::move(Resolvers)) {}

  std::optional<std::string> resolve(StringRef Stem,
                                     bool VariateLibStem = false) const;

private:
  std::optional<std::string> tryWithExtensions(StringRef libstem) const;

  DylibSubstitutor Substitutor;
  DylibPathValidator &Validator;
  std::vector<SearchPathResolver> Resolvers;
};

class DylibResolver {
public:
  DylibResolver(DylibPathValidator &Validator) : Validator(Validator) {}

  void configure(StringRef loaderPath,
                 ArrayRef<SearchPathConfig> SearchPathCfg) {
    DylibSubstitutor Substitutor;
    Substitutor.configure(loaderPath);

    std::vector<SearchPathResolver> Resolvers;
    for (const auto &cfg : SearchPathCfg) {
      Resolvers.emplace_back(cfg,
                             cfg.type == SearchPathType::RPath ? "@rpath" : "");
    }

    impl_ = std::make_unique<DylibResolverImpl>(
        std::move(Substitutor), Validator, std::move(Resolvers));
  }

  std::optional<std::string> resolve(StringRef libStem,
                                     bool VariateLibStem = false) const {
    if (!impl_)
      return std::nullopt;
    return impl_->resolve(libStem, VariateLibStem);
  }

  static std::string resolvelinkerFlag(StringRef libStem,
                                       StringRef loaderPath) {
    DylibSubstitutor Substitutor;
    Substitutor.configure(loaderPath);
    return Substitutor.substitute(libStem);
  }

private:
  DylibPathValidator &Validator;
  std::unique_ptr<DylibResolverImpl> impl_;
};

enum class PathType : uint8_t { User, System, Unknown };

enum class ScanState : uint8_t { NotScanned, Scanning, Scanned };

struct LibrarySearchPath {
  std::string BasePath; // Canonical base directory path
  PathType Kind;        // User or System
  std::atomic<ScanState> State;

  LibrarySearchPath(std::string Base, PathType K)
      : BasePath(std::move(Base)), Kind(K), State(ScanState::NotScanned) {}
};

/// Scans and tracks libraries for symbol resolution.
///
/// Maintains a list of library paths to scan, caches scanned units,
/// and resolves paths canonically for consistent tracking.
class LibraryScanHelper {
public:
  explicit LibraryScanHelper(const std::vector<std::string> &SPaths,
                             std::shared_ptr<LibraryPathCache> LibPathCache,
                             std::shared_ptr<PathResolver> LibPathResolver)
      : LibPathCache(std::move(LibPathCache)),
        LibPathResolver(std::move(LibPathResolver)) {
    DEBUG_WITH_TYPE(
        "orc", dbgs() << "LibraryScanHelper::LibraryScanHelper: base paths : "
                      << SPaths.size() << "\n";);
    for (const auto &p : SPaths)
      addBasePath(p);
  }

  void
  addBasePath(const std::string &P,
              PathType Kind =
                  PathType::Unknown); // Add a canonical directory for scanning
  std::vector<const LibrarySearchPath *> getNextBatch(PathType Kind,
                                                      size_t batchSize);

  bool leftToScan(PathType K) const;
  void resetToScan();

  bool isTrackedBasePath(StringRef P) const;
  std::vector<const LibrarySearchPath *> getAllUnits() const;

  SmallVector<StringRef> getSearchPaths() const {
    SmallVector<StringRef> SearchPaths;
    for (const auto &[_, SP] : LibSearchPaths)
      SearchPaths.push_back(SP->BasePath);
    return SearchPaths;
  }

  PathResolver &getPathResolver() const { return *LibPathResolver; }

  LibraryPathCache &getCache() const { return *LibPathCache; }

  bool hasSeenOrMark(StringRef P) const {
    return LibPathCache->hasSeenOrMark(P);
  }

  std::optional<std::string> resolve(StringRef P, std::error_code &ec) const {
    return LibPathResolver->resolve(P.str(), ec);
  }

private:
  std::string resolveCanonical(StringRef P, std::error_code &ec) const;
  PathType classifyKind(StringRef P) const;

  mutable std::shared_mutex Mtx;
  std::shared_ptr<LibraryPathCache> LibPathCache;
  std::shared_ptr<PathResolver> LibPathResolver;

  StringMap<std::unique_ptr<LibrarySearchPath>>
      LibSearchPaths; // key: canonical path
  std::deque<StringRef> UnscannedUsr;
  std::deque<StringRef> UnscannedSys;
};

/// Scans libraries, resolves dependencies, and registers them.
class LibraryScanner {
public:
  using ShouldScanFn = std::function<bool(StringRef)>;

  LibraryScanner(
      LibraryScanHelper &H, LibraryManager &LibMgr,
      ShouldScanFn ShouldScanCall = [](StringRef path) { return true; })
      : ObjCache(ObjFileCache()), ScanHelper(H), LibMgr(LibMgr),
        Validator(ScanHelper.getPathResolver(), ScanHelper.getCache(),
                  &ObjCache),
        ShouldScanCall(std::move(ShouldScanCall)) {}

  void scanNext(PathType Kind, size_t batchSize = 1);

  /// Dependency info for a library.
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
  ObjFileCache ObjCache;
  LibraryScanHelper &ScanHelper;
  LibraryManager &LibMgr;
  DylibPathValidator Validator;
  ShouldScanFn ShouldScanCall;

  bool shouldScan(StringRef FilePath, bool IsResolvingDep = false);

  Expected<LibraryDepsInfo> extractDeps(StringRef FilePath);

  void handleLibrary(StringRef FilePath, PathType K, int level = 0);

  void scanBaseDir(LibrarySearchPath *U);
};

using LibraryDepsInfo = LibraryScanner::LibraryDepsInfo;

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYSCANNER_H
