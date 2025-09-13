//===- LibraryResolver.h - Automatic Dynamic Library Symbol Resolution -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for automatically searching symbols across
// dynamic libraries that have not yet been loaded.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DYNAMICLOADER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DYNAMICLOADER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolFilter.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryScanner.h"
#include "llvm/Support/Path.h"

#include <atomic>
#include <shared_mutex>
#include <unordered_map>

namespace llvm {
namespace orc {

/// Manages library metadata and state for symbol resolution.
///
/// Tracks libraries by load state and kind (user/system), and stores
/// associated Bloom filters and hash maps to speed up symbol lookups.
/// Thread-safe for concurrent access.
class LibraryManager {
public:
  enum class State : uint8_t { Unloaded = 0, Loaded = 1, Queried = 2 };

  class LibraryInfo {
  public:
    LibraryInfo(const LibraryInfo &) = delete;
    LibraryInfo &operator=(const LibraryInfo &) = delete;

    LibraryInfo(std::string filePath, State s, PathType k,
                std::optional<BloomFilter> filter = std::nullopt)
        : filePath(std::move(filePath)), state(s), kind(k),
          filter(std::move(filter)) {}

    StringRef getBasePath() const { return sys::path::parent_path(filePath); }
    StringRef getFileName() const { return sys::path::filename(filePath); }

    std::string getFullPath() const { return filePath; }

    bool setFilter(BloomFilter F) {
      std::lock_guard<std::shared_mutex> lock(mutex);
      if (filter)
        return false;
      filter.emplace(std::move(F));
      return true;
    }

    bool ensureFilterBuilt(const BloomFilterBuilder &FB,
                           ArrayRef<StringRef> Symbols) {
      std::lock_guard<std::shared_mutex> lock(mutex);
      if (filter)
        return false;
      filter.emplace(FB.build(Symbols));
      return true;
    }

    bool mayContain(StringRef symbol) const {
      assert(hasFilter());
      std::shared_lock<std::shared_mutex> lock(mutex);
      return filter->mayContain(symbol);
    }

    bool hasFilter() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      return filter.has_value();
    }

    State getState() const { return state.load(); }
    PathType getKind() const { return kind; }

    void setState(State s) { state.store(s); }

    bool operator==(const LibraryInfo &other) const {
      return filePath == other.filePath;
    }

  private:
    std::string filePath;
    std::atomic<State> state;
    PathType kind;
    std::optional<BloomFilter> filter;
    mutable std::shared_mutex mutex;
  };

  /// A read-only view of libraries filtered by state and kind.
  ///
  /// Lets you loop over only the libraries in a map that match a given State
  /// and PathType.
  class FilteredView {
  public:
    using Map = StringMap<std::shared_ptr<LibraryInfo>>;
    using Iterator = typename Map::const_iterator;
    class FilterIterator {
    public:
      FilterIterator(Iterator _it, Iterator _end, State s, PathType k)
          : it(_it), end(_end), state(s), kind(k) {
        advance();
      }

      bool operator!=(const FilterIterator &other) const {
        return it != other.it;
      }

      const std::shared_ptr<LibraryInfo> &operator*() const {
        return it->second;
      }

      FilterIterator &operator++() {
        ++it;
        advance();
        return *this;
      }

    private:
      void advance() {
        for (; it != end; ++it)
          if (it->second->getState() == state && it->second->getKind() == kind)
            break;
      }
      Iterator it;
      Iterator end;
      State state;
      PathType kind;
    };
    FilteredView(Iterator begin, Iterator end, State s, PathType k)
        : begin_(begin), end_(end), state_(s), kind_(k) {}

    FilterIterator begin() const {
      return FilterIterator(begin_, end_, state_, kind_);
    }

    FilterIterator end() const {
      return FilterIterator(end_, end_, state_, kind_);
    }

  private:
    Iterator begin_;
    Iterator end_;
    State state_;
    PathType kind_;
  };

private:
  StringMap<std::shared_ptr<LibraryInfo>> libraries;
  mutable std::shared_mutex mutex;

public:
  using LibraryVisitor = std::function<bool(const LibraryInfo &)>;

  LibraryManager() = default;
  ~LibraryManager() = default;

  bool addLibrary(std::string path, PathType kind,
                  std::optional<BloomFilter> filter = std::nullopt) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (libraries.count(path) > 0)
      return false;
    libraries.insert({std::move(path),
                      std::make_shared<LibraryInfo>(path, State::Unloaded, kind,
                                                    std::move(filter))});
    return true;
  }

  bool hasLibrary(StringRef path) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (libraries.count(path) > 0)
      return true;
    return false;
  }

  bool removeLibrary(StringRef path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    auto I = libraries.find(path);
    if (I == libraries.end())
      return false;
    libraries.erase(I);
    return true;
  }

  void markLoaded(StringRef path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end())
      it->second->setState(State::Loaded);
  }

  void markQueried(StringRef path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end())
      it->second->setState(State::Queried);
  }

  std::shared_ptr<LibraryInfo> getLibrary(StringRef path) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end())
      return it->second;
    return nullptr;
  }

  FilteredView getView(State s, PathType k) const {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return FilteredView(libraries.begin(), libraries.end(), s, k);
  }

  void forEachLibrary(const LibraryVisitor &visitor) const {
    std::unique_lock<std::shared_mutex> lock(mutex);
    for (const auto &[_, entry] : libraries) {
      if (!visitor(*entry))
        break;
    }
  }

  bool isLoaded(StringRef path) const {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path.str()); it != libraries.end())
      return it->second->getState() == State::Loaded;
    return false;
  }

  bool isQueried(StringRef path) const {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path.str()); it != libraries.end())
      return it->second->getState() == State::Queried;
    return false;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    libraries.clear();
  }
};

using LibraryInfo = LibraryManager::LibraryInfo;

struct SearchPlanEntry {
  LibraryManager::State state; // Loaded, Queried, Unloaded
  PathType type;               // User, System
};

struct SearchPolicy {
  std::vector<SearchPlanEntry> plan;

  static SearchPolicy defaultPlan() {
    return {{{LibraryManager::State::Loaded, PathType::User},
             {LibraryManager::State::Queried, PathType::User},
             {LibraryManager::State::Unloaded, PathType::User},
             {LibraryManager::State::Loaded, PathType::System},
             {LibraryManager::State::Queried, PathType::System},
             {LibraryManager::State::Unloaded, PathType::System}}};
  }
};

struct SymbolEnumeratorOptions {
  enum Filter : uint32_t {
    None = 0,
    IgnoreUndefined = 1 << 0,
    IgnoreWeak = 1 << 1,
    IgnoreIndirect = 1 << 2,
    IgnoreHidden = 1 << 3,
    IgnoreNonGlobal = 1 << 4
  };

  static SymbolEnumeratorOptions defaultOptions() {
    return {Filter::IgnoreUndefined | Filter::IgnoreWeak |
            Filter::IgnoreIndirect};
  }
  uint32_t FilterFlags = Filter::None;
};

struct SearchConfig {
  SearchPolicy policy;
  SymbolEnumeratorOptions options;

  SearchConfig()
      : policy(SearchPolicy::defaultPlan()), // default plan
        options(SymbolEnumeratorOptions::defaultOptions()) {}
};

/// Scans libraries and resolves Symbols across user and system paths.
///
/// Supports symbol enumeration and filtering via SymbolEnumerator, and tracks
/// symbol resolution results through SymbolQuery. Thread-safe and uses
/// LibraryScanHelper for efficient path resolution and caching.
class LibraryResolver {
  friend class LibraryResolutionDriver;

public:
  class SymbolEnumerator {
  public:
    enum class EnumerateResult { Continue, Stop, Error };

    using OnEachSymbolFn = std::function<EnumerateResult(StringRef Sym)>;

    static bool enumerateSymbols(StringRef Path, OnEachSymbolFn OnEach,
                                 const SymbolEnumeratorOptions &Opts);
  };

  /// Tracks a set of symbols and the libraries where they are resolved.
  ///
  /// SymbolQuery is used to keep track of which symbols have been resolved
  /// to which libraries. It supports concurrent read/write access using a
  /// shared mutex, allowing multiple readers or a single writer at a time.
  class SymbolQuery {
  public:
    /// Holds the result for a single symbol.
    struct Result {
      std::string Name;
      std::string ResolvedLibPath;
    };

  private:
    mutable std::shared_mutex mtx;
    StringMap<Result> Results;
    std::atomic<size_t> ResolvedCount = 0;

  public:
    explicit SymbolQuery(const std::vector<std::string> &Symbols) {
      for (const auto &s : Symbols) {
        if (!Results.contains(s))
          Results.insert({s, Result{s, ""}});
      }
    }

    SmallVector<StringRef> getUnresolvedSymbols() const {
      SmallVector<StringRef> Unresolved;
      std::shared_lock<std::shared_mutex> lock(mtx);
      for (const auto &[name, res] : Results) {
        if (res.ResolvedLibPath.empty())
          Unresolved.push_back(name);
      }
      return Unresolved;
    }

    void resolve(StringRef symbol, const std::string &libPath) {
      std::unique_lock<std::shared_mutex> lock(mtx);
      auto it = Results.find(symbol);
      if (it != Results.end() && it->second.ResolvedLibPath.empty()) {
        it->second.ResolvedLibPath = libPath;
        ResolvedCount.fetch_add(1, std::memory_order_relaxed);
      }
    }

    bool allResolved() const {
      return ResolvedCount.load(std::memory_order_relaxed) == Results.size();
    }

    bool hasUnresolved() const {
      return ResolvedCount.load(std::memory_order_relaxed) < Results.size();
    }

    std::optional<StringRef> getResolvedLib(StringRef symbol) const {
      std::shared_lock<std::shared_mutex> lock(mtx);
      auto it = Results.find(symbol);
      if (it != Results.end() && !it->second.ResolvedLibPath.empty())
        return StringRef(it->second.ResolvedLibPath);
      return std::nullopt;
    }

    bool isResolved(StringRef symbol) const {
      std::shared_lock<std::shared_mutex> lock(mtx);
      auto it = Results.find(symbol.str());
      return it != Results.end() && !it->second.ResolvedLibPath.empty();
    }

    std::vector<const Result *> getAllResults() const {
      std::shared_lock<std::shared_mutex> lock(mtx);
      std::vector<const Result *> out;
      out.reserve(Results.size());
      for (const auto &[_, res] : Results)
        out.push_back(&res);
      return out;
    }
  };

  struct Setup {
    std::vector<std::string> basePaths;
    std::shared_ptr<LibraryPathCache> cache;
    std::shared_ptr<PathResolver> resolver;

    size_t scanBatchSize = 1;

    LibraryScanner::shouldScanFn shouldScan = [](StringRef) { return true; };

    BloomFilterBuilder filterBuilder = BloomFilterBuilder();

    static Setup
    create(std::vector<std::string> basePaths,
           std::shared_ptr<LibraryPathCache> existingCache = nullptr,
           std::shared_ptr<PathResolver> existingResolver = nullptr,
           LibraryScanner::shouldScanFn customShouldScan = nullptr) {
      Setup setup;
      setup.basePaths = std::move(basePaths);

      setup.cache =
          existingCache ? existingCache : std::make_shared<LibraryPathCache>();

      setup.resolver = existingResolver
                           ? existingResolver
                           : std::make_shared<PathResolver>(setup.cache);

      if (customShouldScan)
        setup.shouldScan = std::move(customShouldScan);

      return setup;
    }
  };

  LibraryResolver() = delete;
  explicit LibraryResolver(const Setup &setup);
  ~LibraryResolver() = default;

  using OnSearchComplete = unique_function<void(SymbolQuery &)>;

  void dump() {
    int i = 0;
    m_libMgr.forEachLibrary([&](const LibraryInfo &lib) -> bool {
      dbgs() << ++i << ". Library Path : " << lib.getFullPath() << " -> \n\t\t:"
             << " ({Type : ("
             << (lib.getKind() == PathType::User ? "User" : "System")
             << ") }, { State : "
             << (lib.getState() == LibraryManager::State::Loaded ? "Loaded"
                                                                 : "Unloaded")
             << "})\n";
      return true;
    });
  }

  void searchSymbolsInLibraries(std::vector<std::string> &symbolNames,
                                OnSearchComplete callback,
                                const SearchConfig &config = SearchConfig());

private:
  bool scanLibrariesIfNeeded(PathType K);
  void resolveSymbolsInLibrary(LibraryInfo &library, SymbolQuery &query,
                               const SymbolEnumeratorOptions &Opts);
  bool
  symbolExistsInLibrary(const LibraryInfo &library, StringRef symbol,
                        std::vector<std::string> *matchedSymbols = nullptr);

  bool symbolExistsInLibrary(const LibraryInfo &lib, StringRef symbolName,
                             std::vector<std::string> *allSymbols,
                             const SymbolEnumeratorOptions &opts);

  std::shared_ptr<LibraryPathCache> m_cache;
  std::shared_ptr<PathResolver> m_PathResolver;
  LibraryScanHelper m_scanH;
  BloomFilterBuilder FB;
  LibraryManager m_libMgr;
  LibraryScanner::shouldScanFn m_shouldScan;
  size_t scanBatchSize;
};

using SymbolEnumerator = LibraryResolver::SymbolEnumerator;
using SymbolQuery = LibraryResolver::SymbolQuery;
using EnumerateResult = SymbolEnumerator::EnumerateResult;

class LibraryResolutionDriver {
public:
  static std::unique_ptr<LibraryResolutionDriver>
  create(const LibraryResolver::Setup &setup);

  void addScanPath(const std::string &path, PathType Kind);
  bool markLibraryLoaded(StringRef path);
  bool markLibraryUnLoaded(StringRef path);
  bool isLibraryLoaded(StringRef path) const {
    return Loader->m_libMgr.isLoaded(path);
  }
  void resolveSymbols(std::vector<std::string> Symbols,
                      LibraryResolver::OnSearchComplete OnCompletion,
                      const SearchConfig &config = SearchConfig());

  ~LibraryResolutionDriver() = default;

private:
  LibraryResolutionDriver(std::unique_ptr<LibraryResolver> loader)
      : Loader(std::move(loader)) {}

  std::unique_ptr<LibraryResolver> Loader;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DYNAMICLOADER_H
