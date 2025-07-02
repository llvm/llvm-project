//===- DynamicLoader.h - Automatic Dynamic Library Symbol Resolution -*- C++
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

#include <shared_mutex>
#include <unordered_map>

namespace llvm {
namespace orc {

// Represents a collection of libraries, each optionally associated with a
// symbol filter and hash map.
// class LibraryCollection {
// public:
//   class LibraryInfo {
//   public:
//     LibraryInfo(const LibraryInfo &) = delete;
//     LibraryInfo &operator=(const LibraryInfo &) = delete;

//     LibraryInfo(std::string basePath,
//                 std::optional<BloomFilter> filter = std::nullopt
//         : basePath(std::move(basePath)), filter(std::move(filter)) {}

//     StringRef getBasePath() const { return basePath; }
//     StringRef getFileName() const { return fileName; }

//     std::string getFullPath() const {
//       llvm::SmallString<512> fullPath(basePath);
//       llvm::sys::path::append(fullPath, llvm::StringRef(fileName));
//       return std::string(fullPath.str());
//     }

//     bool setFilter(const BloomFilter &f);
//     bool buildFilter(const std::vector<std::string> &symbols);
//     bool ensureFilterBuilt(const std::vector<std::string> &symbols);
//     bool buildHashMap(const std::vector<std::string> &symbols);

//     bool mayContain(StringRef symbol) const;
//     bool hasFilter() const { return filter.has_value(); }
//     bool hasHashMap() const { return hashMap.has_value(); }

//   private:
//     std::string basePath;
//     std::string fileName;
//     std::optional<BloomFilter> filter;
//     std::mutex mutex;
//   };

//   using LibraryVisitor = unique_function<bool(const LibraryInfo &)>;

//   // Visit each library. Stops early if visitor returns false.
//   bool forEachLibrary(LibraryVisitor &&visitor) const;

//   // Visit each library. Removes those for which the visitor returns true.
//   bool removeIf(LibraryVisitor &&shouldRemove);

//   // Adds a new library with optional filter and hash map.
//   bool addLibrary(std::string basePath,
//                   std::optional<BloomFilter> filter = std::nullopt);

//   // Removes a library by base path.
//   bool removeLibrary(const std::string &basePath);

// private:
//   struct LibraryInfoHash {
//     size_t operator()(const LibraryInfo &lib) const {
//       return std::hash<size_t>()(lib.getBasePath().length()) ^
//              std::hash<std::string>()(std::string(lib.getFileName()));
//     }
//   };

//   std::unordered_set<LibraryInfo, LibraryInfoHash> libraries;
//   std::vector<const LibraryInfo *> Libs;
//   mutable std::mutex mutex;
// };

/// Manages library metadata and state for symbol resolution.
///
/// Tracks libraries by load state and kind (user/system), and stores associated
/// Bloom filters and hash maps to speed up symbol lookups. Thread-safe for
/// concurrent access.
class LibraryManager {
public:
  enum class State : uint8_t { Unloaded = 0, Loaded = 1, Queried = 2 };
  enum class Kind : uint8_t { User = 0, System = 1 };

  class LibraryInfo {
  public:
    LibraryInfo(const LibraryInfo &) = delete;
    LibraryInfo &operator=(const LibraryInfo &) = delete;

    LibraryInfo(std::string filePath, State s, Kind k,
                std::optional<BloomFilter> filter = std::nullopt)
        : filePath(std::move(filePath)), state(s), kind(k),
          filter(std::move(filter)) {}

    StringRef getBasePath() const { return sys::path::parent_path(filePath); }
    StringRef getFileName() const { return sys::path::filename(filePath); }

    std::string getFullPath() const { return filePath; }

    bool setFilter(BloomFilter F) {
      std::lock_guard lock(mutex);
      if (filter)
        return false;
      filter.emplace(std::move(F));
      return true;
    }

    bool ensureFilterBuilt(const BloomFilterBuilder &FB,
                           const std::vector<std::string> &symbols) {
      std::lock_guard lock(mutex);
      if (filter)
        return false;
      filter.emplace(FB.build(symbols));
      return true;
    }

    bool mayContain(StringRef symbol) const {
      assert(hasFilter());
      std::shared_lock lock(mutex);
      return filter->mayContain(symbol);
    }

    bool hasFilter() const {
      std::shared_lock lock(mutex);
      return filter.has_value();
    }

    State getState() const { return state.load(); }
    Kind getKind() const { return kind; }

    void setState(State s) { state.store(s); }

    bool operator==(const LibraryInfo &other) const {
      return filePath == other.filePath;
    }

  private:
    std::string filePath;
    std::atomic<State> state;
    Kind kind;
    std::optional<BloomFilter> filter;
    mutable std::shared_mutex mutex;
  };

  class FilteredView {
  public:
    using Map = std::unordered_map<std::string, std::shared_ptr<LibraryInfo>>;
    using Iterator = typename Map::const_iterator;
    class FilterIterator {
    public:
      FilterIterator(Iterator _it, Iterator _end, State s, Kind k)
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
        while (it != end) {
          const auto &lib = it->second;
          if (lib->getState() == state && lib->getKind() == kind)
            break;
          ++it;
        }
      }
      Iterator it, end;
      State state;
      Kind kind;
    };
    FilteredView(Iterator begin, Iterator end, State s, Kind k)
        : begin_(begin), end_(end), state_(s), kind_(k) {}

    FilterIterator begin() const {
      return FilterIterator(begin_, end_, state_, kind_);
    }

    FilterIterator end() const {
      return FilterIterator(end_, end_, state_, kind_);
    }

  private:
    Iterator begin_, end_;
    State state_;
    Kind kind_;
  };

private:
  std::unordered_map<std::string, std::shared_ptr<LibraryInfo>> libraries;
  mutable std::shared_mutex mutex;

public:
  using LibraryVisitor = std::function<bool(const LibraryInfo &)>;

  LibraryManager() = default;
  ~LibraryManager() = default;

  bool addLibrary(std::string path, Kind kind,
                  std::optional<BloomFilter> filter = std::nullopt) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    // SmallString<256> nativePath(path);
    // sys::path::native(nativePath);

    // auto P = nativePath.str();

    if (libraries.count(path) > 0)
      return false;
    libraries.emplace(std::move(path),
                      std::make_shared<LibraryInfo>(path, State::Unloaded, kind,
                                                    std::move(filter)));
    return true;
  }

  bool hasLibrary(StringRef path) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (libraries.count(path.str()) > 0)
      return true;
    return false;
  }

  bool removeLibrary(const std::string &path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    // auto P = sys::path::native(path);
    auto I = libraries.find(path);
    if (I == libraries.end())
      return false;
    libraries.erase(I);
    return true;
  }

  void markLoaded(const std::string &path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end()) {
      it->second->setState(State::Loaded);
    }
  }

  void markQueried(const std::string &path) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end()) {
      it->second->setState(State::Queried);
    }
  }

  std::shared_ptr<LibraryInfo> getLibrary(const std::string &path) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    if (auto it = libraries.find(path); it != libraries.end())
      return it->second;
    return nullptr;
  }

  FilteredView getView(State s, Kind k) const {
    std::shared_lock lock(mutex);
    return FilteredView(libraries.begin(), libraries.end(), s, k);
  }

  void forEachLibrary(const LibraryVisitor &visitor) const {
    std::unique_lock lock(mutex);
    for (const auto &[_, entry] : libraries) {
      if (!visitor(*entry))
        break;
    }
  }

  bool isLoaded(StringRef path) const {
    std::unique_lock lock(mutex);
    if (auto it = libraries.find(path.str()); it != libraries.end())
      return it->second->getState() == State::Loaded;
    return false;
  }

  bool isQueried(StringRef path) const {
    std::unique_lock lock(mutex);
    if (auto it = libraries.find(path.str()); it != libraries.end())
      return it->second->getState() == State::Queried;
    return false;
  }

  void clear() {
    std::unique_lock lock(mutex);
    libraries.clear();
  }
};

using LibraryInfo = LibraryManager::LibraryInfo;

/// Scans libraries and resolves symbols across user and system paths.
///
/// Supports symbol enumeration and filtering via SymbolEnumerator, and tracks
/// symbol resolution results through SymbolQuery. Thread-safe and uses
/// LibraryScanHelper for efficient path resolution and caching.
class DynamicLoader {
public:
  class SymbolEnumerator {
  public:
    enum class Result { Continue, Stop, Error };

    using OnEachSymbolFn = std::function<Result(const std::string &)>;

    enum class Filter : uint32_t {
      None = 0,
      IgnoreUndefined = 1 << 0,
      IgnoreWeak = 1 << 1,
      IgnoreIndirect = 1 << 2,

      Default = IgnoreUndefined
    };

    struct Options {
      uint32_t FilterFlags = static_cast<uint32_t>(Filter::Default);
    };

    static bool enumerateSymbols(llvm::StringRef Path, OnEachSymbolFn OnEach,
                                 const Options &Opts);
  };

  class SymbolQuery {
  public:
    struct Result {
      std::string Name;
      std::string ResolvedLibPath;
    };

  private:
    mutable std::shared_mutex mtx;
    std::unordered_map<std::string, Result> results;
    std::atomic<size_t> resolvedCount = 0;

  public:
    explicit SymbolQuery(const std::vector<std::string> &symbols) {
      for (const auto &s : symbols)
        results.emplace(s, Result{s, ""});
    }

    std::vector<std::string> getUnresolvedSymbols() const {
      std::vector<std::string> unresolved;
      std::shared_lock lock(mtx);
      for (const auto &[name, res] : results) {
        if (res.ResolvedLibPath.empty())
          unresolved.push_back(name);
      }
      return unresolved;
    }

    void resolve(const std::string &symbol, const std::string &libPath) {
      std::unique_lock lock(mtx);
      auto it = results.find(symbol);
      if (it != results.end() && it->second.ResolvedLibPath.empty()) {
        it->second.ResolvedLibPath = libPath;
        ++resolvedCount;
      }
    }

    bool allResolved() const {
      return resolvedCount.load(std::memory_order_relaxed) == results.size();
    }

    bool hasUnresolved() const {
      return resolvedCount.load(std::memory_order_relaxed) < results.size();
    }

    std::optional<std::string> getResolvedLib(const std::string &symbol) const {
      std::shared_lock lock(mtx);
      auto it = results.find(symbol);
      if (it != results.end() && !it->second.ResolvedLibPath.empty())
        return it->second.ResolvedLibPath;
      return std::nullopt;
    }

    bool isResolved(const std::string &symbol) const {
      std::shared_lock lock(mtx);
      auto it = results.find(symbol);
      return it != results.end() && !it->second.ResolvedLibPath.empty();
    }

    std::vector<Result> getAllResults() const {
      std::shared_lock lock(mtx);
      std::vector<Result> out;
      out.reserve(results.size());
      for (const auto &[_, res] : results)
        out.push_back(res);
      return out;
    }
  };

  struct Setup {
    std::vector<std::string> basePaths;
    std::shared_ptr<LibraryPathCache> cache;
    std::shared_ptr<PathResolver> resolver;
    // std::shared_ptr<DylibPathResolver> dylibResolver;

    bool includeSys = false;

    LibraryScanner::shouldScanFn shouldScan = [](StringRef) { return true; };

    BloomFilterBuilder filterBuilder = BloomFilterBuilder();

    static Setup create(
        std::vector<std::string> basePaths,
        std::shared_ptr<LibraryPathCache> existingCache = nullptr,
        std::shared_ptr<PathResolver> existingResolver = nullptr,
        //  std::shared_ptr<DylibPathResolver> existingDylibResolver = nullptr,
        LibraryScanner::shouldScanFn customShouldScan = nullptr) {
      Setup setup;
      setup.basePaths = std::move(basePaths);

      setup.cache =
          existingCache ? existingCache : std::make_shared<LibraryPathCache>();

      setup.resolver = existingResolver
                           ? existingResolver
                           : std::make_shared<PathResolver>(setup.cache);

      // setup.dylibResolver = std::move(existingDylibResolver);

      if (customShouldScan)
        setup.shouldScan = std::move(customShouldScan);

      return setup;
    }
  };

  DynamicLoader() = delete;
  explicit DynamicLoader(const Setup &setup);
  ~DynamicLoader() = default;

  using OnSearchComplete = unique_function<void(SymbolQuery &)>;

  void searchSymbolsInLibraries(std::vector<std::string> &symbolNames,
                                OnSearchComplete callback);

private:
  void scanLibrariesIfNeeded(LibraryManager::Kind K);
  void resolveSymbolsInLibrary(LibraryInfo &library, SymbolQuery &query);
  bool
  symbolExistsInLibrary(const LibraryInfo &library, llvm::StringRef symbol,
                        std::vector<std::string> *matchedSymbols = nullptr);

  bool symbolExistsInLibrary(const LibraryInfo &lib, llvm::StringRef symbolName,
                             std::vector<std::string> *allSymbols,
                             const SymbolEnumerator::Options &opts);

  std::shared_ptr<LibraryPathCache> m_cache;
  std::shared_ptr<PathResolver> m_PathResolver;
  LibraryScanHelper ScanH;
  BloomFilterBuilder FB;
  LibraryManager LibMgr;
  LibraryScanner::shouldScanFn m_shouldScan;
  // std::shared_ptr<DylibPathResolver> m_DylibPathResolver;
  bool includeSys;
};

using SymbolEnumerator = DynamicLoader::SymbolEnumerator;
using EnumerateResult = SymbolEnumerator::Result;

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESS_DYNAMICLOADER_H
