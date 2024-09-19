//===------------ AutoLoadDylibLookup.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_AUTOLOADDYNAMICLIBRARY_H
#define LLVM_EXECUTIONENGINE_ORC_AUTOLOADDYNAMICLIBRARY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/Orc/Shared/AutoLoadDylibUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace orc {
class DynamicLoader;

#if defined(LLVM_ON_UNIX)
const char *const kEnvDelim = ":";
#elif defined(_WIN32)
const char *const kEnvDelim = ";";
#else
#error "Unknown platform (environmental delimiter)"
#endif

class AutoLoadDynamicLibraryLookup {
public:
  /// Describes the library search paths.
  struct SearchPathInfo {
    /// The search path.
    std::string Path;

    /// True if the Path is on the LD_LIBRARY_PATH.
    bool IsUser;

    bool operator==(const SearchPathInfo &Other) const {
      return IsUser == Other.IsUser && Path == Other.Path;
    }
  };
  using SearchPathInfos = SmallVector<SearchPathInfo, 32>;

private:
  StringSet<> LoadedLibraries;

  /// System's include path, get initialized at construction time.
  SearchPathInfos SearchPaths;

  DynamicLoader *Dyld = nullptr;

  /// Concatenates current and system include paths to look up the filename.
  /// Considers RPATH and RUNPATH (see: https://en.wikipedia.org/wiki/Rpath).
  /// Returns the canonical file path or an empty string if not found.
  std::string lookupLibInPaths(StringRef libStem,
                               SmallVector<StringRef, 2> RPath = {},
                               SmallVector<StringRef, 2> RunPath = {},
                               StringRef libLoader = "") const;

  /// Concatenates current and system include paths, then looks up the filename.
  /// If not found, adds platform-specific extensions (e.g., .so, .dll, .dylib)
  /// and retries. Considers RPATH and RUNPATH (see:
  /// https://en.wikipedia.org/wiki/Rpath). Returns the canonical file path or
  /// an empty string if not found.
  std::string lookupLibMaybeAddExt(StringRef filename,
                                   SmallVector<StringRef, 2> RPath = {},
                                   SmallVector<StringRef, 2> RunPath = {},
                                   StringRef libLoader = "") const;

  /// On a success returns to full path to a shared object that holds the
  /// symbol pointed by func.
  static std::string getSymbolLocation(void *func);

public:
  AutoLoadDynamicLibraryLookup();
  ~AutoLoadDynamicLibraryLookup();
  AutoLoadDynamicLibraryLookup(const AutoLoadDynamicLibraryLookup &) = delete;
  AutoLoadDynamicLibraryLookup &
  operator=(const AutoLoadDynamicLibraryLookup &) = delete;

  const SearchPathInfos &getSearchPaths() const { return SearchPaths; }

  void addSearchPath(StringRef dir, bool isUser = true, bool prepend = false) {
    if (!dir.empty()) {
      for (auto &item : SearchPaths)
        if (dir.equals_insensitive(item.Path))
          return;
      auto pos = prepend ? SearchPaths.begin() : SearchPaths.end();
      SearchPaths.insert(pos, SearchPathInfo{dir.str(), isUser});
    }
  }

  /// Searches for a library in current and system include paths.
  /// Considers RPATH and RUNPATH (see: https://en.wikipedia.org/wiki/Rpath).
  /// Returns the canonical file path or an empty string if not found.
  std::string lookupLibrary(StringRef libStem,
                            SmallVector<StringRef, 2> RPath = {},
                            SmallVector<StringRef, 2> RunPath = {},
                            StringRef libLoader = "",
                            bool variateLibStem = true) const;

  /// Add loaded library.
  void addLoadedLib(StringRef lib);

  /// Returns true if the file was a dynamic library and it was already
  /// loaded.
  bool isLibraryLoaded(StringRef fullPath) const;

  /// Initializes the dynamic loader (dyld).
  /// Accepts a callback to decide if certain libraries, such as those
  /// overriding malloc, should be ignored.
  void initializeDynamicLoader(
      std::function<bool(StringRef)> shouldPermanentlyIgnore);

  /// Finds the first unloaded shared object containing the specified symbol.
  /// Returns the library name if found, or an empty string otherwise.
  std::string searchLibrariesForSymbol(StringRef mangledName,
                                       bool searchSystem = true) const;

  void dump(raw_ostream *S = nullptr) const;

  /// On a success returns to full path to a shared object that holds the
  /// symbol pointed by func.
  template <class T> static std::string getSymbolLocation(T func) {
    static_assert(std::is_pointer<T>::value, "Must be a function pointer!");
    return getSymbolLocation(reinterpret_cast<void *>(func));
  }

  static std::string normalizePath(StringRef path);

  /// Returns true if the file is a shared library.
  /// Also sets whether the file exists to help identify incompatible formats.
  static bool isSharedLibrary(StringRef libFullPath, bool *exists = nullptr);

  void BuildGlobalBloomFilter(BloomFilter &Filter) const;
};

enum class SplitMode {
  PruneNonExistant, ///< Don't add non-existant paths into output
  FailNonExistant,  ///< Fail on any non-existant paths
  AllowNonExistant  ///< Add all paths whether they exist or not
};

bool SplitPaths(StringRef PathStr, SmallVectorImpl<StringRef> &Paths,
                SplitMode Mode, StringRef Delim, bool Verbose = false);

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_AUTOLOADDYNAMICLIBRARY_H