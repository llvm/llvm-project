//===- LibraryResolver.h - Automatic Library Symbol Resolution -*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYRESOLVER_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYRESOLVER_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolFilter.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryScanner.h"
#include "llvm/Support/Path.h"

#include <atomic>
#include <shared_mutex>

namespace llvm {
namespace orc {

class LibraryManager;

enum class LibState : uint8_t { Unloaded = 0, Loaded = 1, Queried = 2 };

class LibraryInfo {
public:
  LibraryInfo(const LibraryInfo &) = delete;
  LibraryInfo &operator=(const LibraryInfo &) = delete;

  LibraryInfo(std::string FilePath, LibState S, PathType K,
              std::optional<BloomFilter> Filter = std::nullopt)
      : FilePath(std::move(FilePath)), S(S), K(K), Filter(std::move(Filter)) {}

  StringRef getBasePath() const { return sys::path::parent_path(FilePath); }
  StringRef getFileName() const { return sys::path::filename(FilePath); }

  std::string getFullPath() const { return FilePath; }

  void setFilter(BloomFilter F) {
    std::lock_guard<std::shared_mutex> Lock(Mtx);
    if (Filter)
      return;
    Filter.emplace(std::move(F));
  }

  void ensureFilterBuilt(const BloomFilterBuilder &FB,
                         ArrayRef<StringRef> Symbols) {
    std::lock_guard<std::shared_mutex> Lock(Mtx);
    if (Filter)
      return;
    Filter.emplace(FB.build(Symbols));
  }

  bool mayContain(StringRef Symbol) const {
    assert(hasFilter());
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    return Filter->mayContain(Symbol);
  }

  bool hasFilter() const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    return Filter.has_value();
  }

  LibState getState() const { return S.load(); }
  PathType getKind() const { return K; }

  void setState(LibState s) { S.store(s); }

  bool operator==(const LibraryInfo &other) const {
    return FilePath == other.FilePath;
  }

private:
  std::string FilePath;
  std::atomic<LibState> S;
  PathType K;
  std::optional<BloomFilter> Filter;
  mutable std::shared_mutex Mtx;
};

// The LibraryCursor iterates through a list of LibraryInfo pointers,
// returning only those libraries that match a specified LibState. LibraryIndex
// provides these lists based on PathType, and the cursor filters them as it
// iterates.
class LibraryCursor {
public:
  LibraryCursor(const std::vector<const LibraryInfo *> &L, LibState S)
      : Lists(L), S(S) {}

  const LibraryInfo *nextValidLib() {
    while (Pos < Lists.size()) {
      const LibraryInfo *Lib = Lists[Pos++];
      if (Lib->getState() == S)
        return Lib;
    }

    return nullptr;
  }

  bool hasMoreValidLib() const { return Pos < Lists.size(); }

private:
  const std::vector<const LibraryInfo *> &Lists;
  LibState S;
  size_t Pos = 0; // cursor position
};

// LibraryIndex keeps libraries grouped by PathType and lets you
// get a cursor to walk through libraries of a specific type/state.
class LibraryIndex {
  friend LibraryManager;

public:
  LibraryCursor getCursor(PathType K, LibState S) const {
    static std::vector<const LibraryInfo *> Empty;
    auto It = Lists.find(K);
    if (It == Lists.end())
      return LibraryCursor(Empty, S);

    return LibraryCursor(It->second, S);
  }

  bool hasLibLeftFor(PathType K, uint32_t Idx) const {
    auto It = Lists.find(K);
    if (It == Lists.end())
      return false;

    const auto &L = It->second;
    return L.size() > Idx;
  }

private:
  void addLibrary(const LibraryInfo *Lib) {
    Lists[Lib->getKind()].push_back(Lib);
  }

  void clear() { Lists.clear(); }

  DenseMap<PathType, std::vector<const LibraryInfo *>> Lists;
};

/// Manages library metadata and state for symbol resolution.
///
/// Tracks libraries by load state and kind (user/system), and stores
/// associated Bloom filters and hash maps to speed up symbol lookups.
/// Thread-safe for concurrent access.
class LibraryManager {
public:
  /// A read-only view of libraries filtered by state and kind.
  ///
  /// Lets you loop over only the libraries in a map that match a given State
  /// and PathType.
  class FilteredView {
  public:
    using Map = StringMap<std::unique_ptr<LibraryInfo>>;
    using Iterator = Map::const_iterator;
    class FilterIterator {
    public:
      FilterIterator(Iterator it_, Iterator end_, LibState S, PathType K)
          : it(it_), end(end_), S(S), K(K) {
        advance();
      }

      bool operator!=(const FilterIterator &other) const {
        return it != other.it;
      }

      const LibraryInfo &operator*() const { return *it->second; }

      FilterIterator &operator++() {
        ++it;
        advance();
        return *this;
      }

    private:
      void advance() {
        for (; it != end; ++it)
          if (it->second->getState() == S && it->second->getKind() == K)
            break;
      }
      Iterator it;
      Iterator end;
      LibState S;
      PathType K;
    };
    FilteredView(Iterator begin, Iterator end, LibState s, PathType k)
        : mapBegin(begin), mapEnd(end), state(s), kind(k) {}

    FilterIterator begin() const {
      return FilterIterator(mapBegin, mapEnd, state, kind);
    }

    FilterIterator end() const {
      return FilterIterator(mapEnd, mapEnd, state, kind);
    }

  private:
    Iterator mapBegin;
    Iterator mapEnd;
    LibState state;
    PathType kind;
  };

private:
  LibraryIndex Index;
  StringMap<std::unique_ptr<LibraryInfo>> Libraries;
  mutable std::shared_mutex Mtx;

public:
  using LibraryVisitor = std::function<bool(const LibraryInfo &)>;

  LibraryManager() = default;
  ~LibraryManager() = default;

  bool addLibrary(std::string Path, PathType Kind,
                  std::optional<BloomFilter> Filter = std::nullopt) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    if (Libraries.count(Path) > 0)
      return false;
    std::unique_ptr<LibraryInfo> Lib = std::make_unique<LibraryInfo>(
        Path, LibState::Unloaded, Kind, std::move(Filter));
    const LibraryInfo *Ptr = Lib.get();
    Libraries.insert({Path, std::move(Lib)});
    Index.addLibrary(Ptr);
    return true;
  }

  bool hasLibrary(StringRef Path) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    if (Libraries.count(Path) > 0)
      return true;
    return false;
  }

  void removeLibrary(StringRef Path) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    auto I = Libraries.find(Path);
    if (I == Libraries.end())
      return;
    Libraries.erase(I);
  }

  void markLoaded(StringRef Path) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path); It != Libraries.end())
      It->second->setState(LibState::Loaded);
  }

  void markUnloaded(StringRef Path) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path); It != Libraries.end())
      It->second->setState(LibState::Unloaded);
  }

  void markQueried(StringRef Path) {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path); It != Libraries.end())
      It->second->setState(LibState::Queried);
  }

  const LibraryInfo *getLibrary(StringRef Path) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path); It != Libraries.end())
      return It->second.get();
    return nullptr;
  }

  FilteredView getView(LibState S, PathType K) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    return FilteredView(Libraries.begin(), Libraries.end(), S, K);
  }

  using LibraryFilterFn = std::function<bool(const LibraryInfo &)>;
  void getLibraries(LibState S, PathType K,
                    std::vector<const LibraryInfo *> &Outs,
                    LibraryFilterFn Filter = nullptr) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    for (const auto &[_, Entry] : Libraries) {
      const auto &Info = *Entry;
      if (Info.getKind() != K || Info.getState() != S)
        continue;
      if (Filter && !Filter(Info))
        continue;
      Outs.push_back(&Info);
    }
  }

  LibraryCursor getCursor(PathType K, LibState S) const {
    return Index.getCursor(K, S);
  }

  void forEachLibrary(const LibraryVisitor &visitor) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    for (const auto &[_, entry] : Libraries) {
      if (!visitor(*entry))
        break;
    }
  }

  bool isLoaded(StringRef Path) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path.str()); It != Libraries.end())
      return It->second->getState() == LibState::Loaded;
    return false;
  }

  bool isQueried(StringRef Path) const {
    std::shared_lock<std::shared_mutex> Lock(Mtx);
    if (auto It = Libraries.find(Path.str()); It != Libraries.end())
      return It->second->getState() == LibState::Queried;
    return false;
  }

  void clear() {
    std::unique_lock<std::shared_mutex> Lock(Mtx);
    Libraries.clear();
  }
};

struct SearchPlanEntry {
  LibState State; // Loaded, Queried, Unloaded
  PathType Type;  // User, System
};

struct SearchPolicy {
  std::vector<SearchPlanEntry> Plan;

  static SearchPolicy defaultPlan() {
    return {{{LibState::Loaded, PathType::User},
             {LibState::Queried, PathType::User},
             {LibState::Unloaded, PathType::User},
             {LibState::Loaded, PathType::System},
             {LibState::Queried, PathType::System},
             {LibState::Unloaded, PathType::System}}};
  }
};

struct SymbolEnumeratorOptions {
  enum Filter : uint32_t {
    None = 0,
    IgnoreUndefined = 1 << 0,
    IgnoreWeak = 1 << 1,
    IgnoreIndirect = 1 << 2,
    IgnoreNonGlobal = 1 << 3,
    IgnoreHidden = 1 << 4,
    IgnoreNonExported = 1 << 5
  };

  static SymbolEnumeratorOptions defaultOptions() {
    return {Filter::IgnoreUndefined | Filter::IgnoreHidden |
            Filter::IgnoreNonGlobal};
  }
  uint32_t FilterFlags = Filter::None;
};

struct SearchConfig {
  SearchPolicy Policy;
  SymbolEnumeratorOptions Options;

  SearchConfig()
      : Policy(SearchPolicy::defaultPlan()), // default plan
        Options(SymbolEnumeratorOptions::defaultOptions()) {}
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

    static bool enumerateSymbols(object::ObjectFile *Obj, OnEachSymbolFn OnEach,
                                 const SymbolEnumeratorOptions &Opts);
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
    struct Entry {
      std::string Name;
      std::string ResolvedLibPath;
    };

  private:
    mutable std::shared_mutex Mtx;
    SmallVector<Entry, 24> Entries;
    std::atomic<size_t> ResolvedCount = 0;

  public:
    explicit SymbolQuery(ArrayRef<StringRef> Symbols) {
      for (const auto &S : Symbols) {
        if (!contains(S))
          Entries.push_back({S.str(), ""});
      }
    }

    bool contains(StringRef Name) const {
      return llvm::any_of(Entries,
                          [&](const Entry &E) { return E.Name == Name; });
    }

    using SymbolFilterFn = unique_function<bool(StringRef)>;
    void getUnresolvedSymbols(SmallVectorImpl<StringRef> &Unresolved,
                              SymbolFilterFn Allow) const {
      std::shared_lock<std::shared_mutex> Lock(Mtx);
      for (const auto &E : Entries) {
        if (E.ResolvedLibPath.empty() && Allow(E.Name))
          Unresolved.push_back(E.Name);
      }
    }

    void resolve(StringRef Sym, const std::string &LibPath) {
      std::unique_lock<std::shared_mutex> Lock(Mtx);
      for (auto &E : Entries) {
        if (E.Name == Sym && E.ResolvedLibPath.empty()) {
          E.ResolvedLibPath = LibPath;
          ResolvedCount.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
    }

    bool allResolved() const {
      return ResolvedCount.load(std::memory_order_relaxed) == Entries.size();
    }

    bool hasUnresolved() const {
      return ResolvedCount.load(std::memory_order_relaxed) < Entries.size();
    }

    std::optional<StringRef> getResolvedLib(StringRef Sym) const {
      std::shared_lock<std::shared_mutex> Lock(Mtx);
      for (const auto &E : Entries)
        if (E.Name == Sym && !E.ResolvedLibPath.empty())
          return E.ResolvedLibPath;
      return std::nullopt;
    }

    bool isResolved(StringRef Sym) const {
      std::shared_lock<std::shared_mutex> Lock(Mtx);
      for (const auto &E : Entries)
        if (E.Name == Sym && !E.ResolvedLibPath.empty())
          return true;
      return false;
    }

    std::vector<const Entry *> getAllResults() const {
      std::shared_lock<std::shared_mutex> Lock(Mtx);
      std::vector<const Entry *> Out;
      Out.reserve(Entries.size());
      for (const auto &E : Entries)
        Out.push_back(&E);
      return Out;
    }
  };

  struct Setup {
    std::vector<std::string> BasePaths;
    // std::shared_ptr<LibraryPathCache> Cache;
    // std::shared_ptr<PathResolver> PResolver;

    size_t ScanBatchSize = 0;

    LibraryScanner::ShouldScanFn ShouldScanCall = [](StringRef) {
      return true;
    };

    BloomFilterBuilder FilterBuilder = BloomFilterBuilder();

    static Setup
    create(std::vector<std::string> BasePaths,
           //  std::shared_ptr<LibraryPathCache> existingCache = nullptr,
           //  std::shared_ptr<PathResolver> existingResolver = nullptr,
           LibraryScanner::ShouldScanFn customShouldScan = nullptr) {
      Setup S;
      S.BasePaths = std::move(BasePaths);

      // S.Cache =
      //     existingCache ? existingCache :
      //     std::make_shared<LibraryPathCache>();

      // S.PResolver = existingResolver ? existingResolver
      //                                :
      //                                std::make_shared<PathResolver>(S.Cache);

      if (customShouldScan)
        S.ShouldScanCall = std::move(customShouldScan);

      return S;
    }
  };

  LibraryResolver() = delete;
  explicit LibraryResolver(const Setup &S);
  ~LibraryResolver() = default;

  using OnSearchComplete = unique_function<void(SymbolQuery &)>;

  void dump() {
    int i = 0;
    LibMgr.forEachLibrary([&](const LibraryInfo &Lib) -> bool {
      dbgs() << ++i << ". Library Path : " << Lib.getFullPath() << " -> \n\t\t:"
             << " ({Type : ("
             << (Lib.getKind() == PathType::User ? "User" : "System")
             << ") }, { State : "
             << (Lib.getState() == LibState::Loaded ? "Loaded" : "Unloaded")
             << "})\n";
      return true;
    });
  }

  void searchSymbolsInLibraries(ArrayRef<StringRef> SymList,
                                OnSearchComplete OnComplete,
                                const SearchConfig &Config = SearchConfig());

private:
  bool scanLibrariesIfNeeded(PathType K, size_t BatchSize = 0);
  bool scanForNewLibraries(PathType K, LibraryCursor &Cur);
  void resolveSymbolsInLibrary(LibraryInfo *Lib, SymbolQuery &Q,
                               const SymbolEnumeratorOptions &Opts);

  LibraryManager LibMgr;
  std::shared_ptr<LibraryPathCache> LibPathCache;
  std::shared_ptr<PathResolver> LibPathResolver;
  LibraryScanHelper ScanHelper;
  BloomFilterBuilder FB;
  LibraryScanner::ShouldScanFn ShouldScanCall;
  size_t scanBatchSize;
};

using SymbolEnumerator = LibraryResolver::SymbolEnumerator;
using SymbolQuery = LibraryResolver::SymbolQuery;
using EnumerateResult = SymbolEnumerator::EnumerateResult;

class LibraryResolutionDriver {
public:
  static std::unique_ptr<LibraryResolutionDriver>
  create(const LibraryResolver::Setup &S);

  void addScanPath(const std::string &Path, PathType Kind);
  void markLibraryLoaded(StringRef Path);
  void markLibraryUnLoaded(StringRef Path);
  bool isLibraryLoaded(StringRef Path) const {
    return LR->LibMgr.isLoaded(Path);
  }

  void resetAll() {
    LR->LibMgr.clear();
    LR->ScanHelper.resetToScan();
    LR->LibPathCache->clear();
  }

  void scanAll(size_t BatchSize = 0) {
    LR->scanLibrariesIfNeeded(PathType::User, BatchSize);
    LR->scanLibrariesIfNeeded(PathType::System, BatchSize);
  }

  void scan(PathType PK, size_t BatchSize = 0) {
    LR->scanLibrariesIfNeeded(PK, BatchSize);
  }

  void resolveSymbols(ArrayRef<StringRef> Symbols,
                      LibraryResolver::OnSearchComplete OnCompletion,
                      const SearchConfig &Config = SearchConfig());

  ~LibraryResolutionDriver() = default;

private:
  LibraryResolutionDriver(std::unique_ptr<LibraryResolver> L)
      : LR(std::move(L)) {}

  std::unique_ptr<LibraryResolver> LR;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_LIBRARYRESOLVER_H
