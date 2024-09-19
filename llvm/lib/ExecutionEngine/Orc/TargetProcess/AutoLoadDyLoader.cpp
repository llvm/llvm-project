//===---------------- DynamicLibraryManagerSymbol.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/AutoLoadDylibLookup.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"

#include <algorithm>
#include <list>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef LLVM_ON_UNIX
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/stat.h>
#undef LC_LOAD_DYLIB
#undef LC_RPATH
#endif // __APPLE__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <libloaderapi.h> // For GetModuleFileNameA
#include <memoryapi.h>    // For VirtualQuery
#include <windows.h>
#endif

namespace {
#define PATH_MAX 1024
using BasePath = std::string;
using namespace llvm;

// // This is a GNU implementation of hash used in bloom filter!
static uint32_t GNUHash(StringRef S) {
  uint32_t H = 5381;
  for (uint8_t C : S)
    H = (H << 5) + H + C;
  return H;
}

/// An efficient representation of a full path to a library which does not
/// duplicate common path patterns reducing the overall memory footprint.
///
/// For example, `/home/.../lib/libA.so`, Path will contain a pointer
/// to  `/home/.../lib/`
/// will be stored and .second `libA.so`.
/// This approach reduces the duplicate paths as at one location there may be
/// plenty of libraries.
struct LibraryPath {
  const BasePath &Path;
  std::string LibName;
  orc::BloomFilter Filter;
  StringSet<> Symbols;
  // std::vector<const LibraryPath*> LibDeps;

  LibraryPath(const BasePath &Path, const std::string &LibName)
      : Path(Path), LibName(LibName) {}

  bool operator==(const LibraryPath &other) const {
    return (&Path == &other.Path || Path == other.Path) &&
           LibName == other.LibName;
  }

  const std::string GetFullName() const {
    SmallString<512> Vec(Path);
    sys::path::append(Vec, StringRef(LibName));
    return Vec.str().str();
  }

  void AddBloom(StringRef symbol) { Filter.AddSymbol(symbol); }

  StringRef AddSymbol(const std::string &symbol) {
    auto it = Symbols.insert(symbol);
    return it.first->getKey();
  }

  bool hasBloomFilter() const { return Filter.IsInitialized(); }

  bool isBloomFilterEmpty() const {
    assert(Filter.IsInitialized() && "Bloom filter not initialized!");
    return Filter.IsEmpty();
  }

  void InitializeBloomFilter(uint32_t newSymbolsCount) {
    assert(!Filter.IsInitialized() && "Cannot re-initialize non-empty filter!");
    Filter.Initialize(newSymbolsCount);
  }

  bool MayExistSymbol(StringRef symbol) const {
    // The library had no symbols and the bloom filter is empty.
    if (isBloomFilterEmpty())
      return false;

    return Filter.MayContain(symbol);
  }

  bool ExistSymbol(StringRef symbol) const {
    return Symbols.find(symbol) != Symbols.end();
  }
};

/// A helper class keeping track of loaded libraries. It implements a fast
/// search O(1) while keeping deterministic iterability in a memory efficient
/// way. The underlying set uses a custom hasher for better efficiency given the
/// specific problem where the library names (LibName) are relatively short
/// strings and the base paths (Path) are repetitive long strings.
class LibraryPaths {
  struct LibraryPathHashFn {
    size_t operator()(const LibraryPath &item) const {
      return std::hash<size_t>()(item.Path.length()) ^
             std::hash<std::string>()(item.LibName);
    }
  };

  std::vector<const LibraryPath *> Libs;
  std::unordered_set<LibraryPath, LibraryPathHashFn> LibsH;

public:
  bool HasRegisteredLib(const LibraryPath &Lib) const {
    return LibsH.count(Lib);
  }

  const LibraryPath *GetRegisteredLib(const LibraryPath &Lib) const {
    auto search = LibsH.find(Lib);
    if (search != LibsH.end())
      return &(*search);
    return nullptr;
  }

  const LibraryPath *RegisterLib(const LibraryPath &Lib) {
    auto it = LibsH.insert(Lib);
    assert(it.second && "Already registered!");
    Libs.push_back(&*it.first);
    return &*it.first;
  }

  void UnregisterLib(const LibraryPath &Lib) {
    auto found = LibsH.find(Lib);
    if (found == LibsH.end())
      return;

    Libs.erase(std::find(Libs.begin(), Libs.end(), &*found));
    LibsH.erase(found);
  }

  size_t size() const {
    assert(Libs.size() == LibsH.size());
    return Libs.size();
  }

  const std::vector<const LibraryPath *> &GetLibraries() const { return Libs; }
};

#ifndef _WIN32
// Cached version of system function lstat
static inline mode_t cached_lstat(const char *path) {
  static StringMap<mode_t> lstat_cache;

  // If already cached - retun cached result
  auto it = lstat_cache.find(path);
  if (it != lstat_cache.end())
    return it->second;

  // If result not in cache - call system function and cache result
  struct stat buf;
  mode_t st_mode = (lstat(path, &buf) == -1) ? 0 : buf.st_mode;
  lstat_cache.insert(std::pair<StringRef, mode_t>(path, st_mode));
  return st_mode;
}

// Cached version of system function readlink
static inline StringRef cached_readlink(const char *pathname) {
  static StringMap<std::string> readlink_cache;

  // If already cached - retun cached result
  auto it = readlink_cache.find(pathname);
  if (it != readlink_cache.end())
    return StringRef(it->second);

  // If result not in cache - call system function and cache result
  char buf[PATH_MAX];
  ssize_t len;
  if ((len = readlink(pathname, buf, sizeof(buf))) != -1) {
    buf[len] = '\0';
    std::string s(buf);
    readlink_cache.insert(std::pair<StringRef, std::string>(pathname, s));
    return readlink_cache[pathname];
  }
  return "";
}
#endif

// Cached version of system function realpath
std::string cached_realpath(StringRef path, StringRef base_path = "",
                            bool is_base_path_real = false,
                            long symlooplevel = 40) {
  if (path.empty()) {
    errno = ENOENT;
    return "";
  }

  if (!symlooplevel) {
    errno = ELOOP;
    return "";
  }

  // If already cached - retun cached result
  static StringMap<std::pair<std::string, int>> cache;
  bool relative_path = sys::path::is_relative(path);
  if (!relative_path) {
    auto it = cache.find(path);
    if (it != cache.end()) {
      errno = it->second.second;
      return it->second.first;
    }
  }

  // If result not in cache - call system function and cache result

  StringRef sep(sys::path::get_separator());
  SmallString<256> result(sep);
#ifndef _WIN32
  SmallVector<StringRef, 16> p;

  // Relative or absolute path
  if (relative_path) {
    if (is_base_path_real) {
      result.assign(base_path);
    } else {
      if (path[0] == '~' &&
          (path.size() == 1 || sys::path::is_separator(path[1]))) {
        static SmallString<128> home;
        if (home.str().empty())
          sys::path::home_directory(home);
        StringRef(home).split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else if (base_path.empty()) {
        static SmallString<256> current_path;
        if (current_path.str().empty())
          sys::fs::current_path(current_path);
        StringRef(current_path)
            .split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      } else {
        base_path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);
      }
    }
  }
  path.split(p, sep, /*MaxSplit*/ -1, /*KeepEmpty*/ false);

  // Handle path list items
  for (auto item : p) {
    if (item == ".")
      continue; // skip "." element in "abc/./def"
    if (item == "..") {
      // collapse "a/b/../c" to "a/c"
      size_t s = result.rfind(sep);
      if (s != StringRef::npos)
        result.resize(s);
      if (result.empty())
        result = sep;
      continue;
    }

    size_t old_size = result.size();
    sys::path::append(result, item);
    mode_t st_mode = cached_lstat(result.c_str());
    if (S_ISLNK(st_mode)) {
      StringRef symlink = cached_readlink(result.c_str());
      if (sys::path::is_relative(symlink)) {
        result.resize(old_size);
        result = cached_realpath(symlink, result, true, symlooplevel - 1);
      } else {
        result = cached_realpath(symlink, "", true, symlooplevel - 1);
      }
    } else if (st_mode == 0) {
      cache.insert(std::pair<StringRef, std::pair<std::string, int>>(
          path, std::pair<std::string, int>("", ENOENT)));
      errno = ENOENT;
      return "";
    }
  }
#else
  sys::fs::real_path(path, result);
#endif
  cache.insert(std::pair<StringRef, std::pair<std::string, int>>(
      path, std::pair<std::string, int>(result.str().str(), errno)));
  return result.str().str();
}

using namespace llvm;
using namespace object;

template <class ELFT>
static Expected<StringRef> getDynamicStrTab(const ELFFile<ELFT> *Elf) {
  auto DynamicEntriesOrError = Elf->dynamicEntries();
  if (!DynamicEntriesOrError)
    return DynamicEntriesOrError.takeError();

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_STRTAB) {
      auto MappedAddrOrError = Elf->toMappedAddr(Dyn.getPtr());
      if (!MappedAddrOrError)
        return MappedAddrOrError.takeError();
      return StringRef(reinterpret_cast<const char *>(*MappedAddrOrError));
    }
  }

  // If the dynamic segment is not present, we fall back on the sections.
  auto SectionsOrError = Elf->sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  for (const typename ELFT::Shdr &Sec : *SectionsOrError) {
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      return Elf->getStringTableForSymtab(Sec);
  }

  return createError("dynamic string table not found");
}

static StringRef GetGnuHashSection(object::ObjectFile *file) {
  for (auto S : file->sections()) {
    StringRef name = cantFail(S.getName());
    if (name == ".gnu.hash") {
      return cantFail(S.getContents());
    }
  }
  return "";
}

/// Bloom filter is a stochastic data structure which can tell us if a symbol
/// name does not exist in a library with 100% certainty. If it tells us it
/// exists this may not be true:
/// https://blogs.oracle.com/solaris/gnu-hash-elf-sections-v2
///
/// ELF has this optimization in the new linkers by default, It is stored in the
/// gnu.hash section of the object file.
///
///\returns true if the symbol may be in the library.
static bool MayExistInElfObjectFile(object::ObjectFile *soFile, uint32_t hash) {
  assert(soFile->isELF() && "Not ELF");

  // Compute the platform bitness -- either 64 or 32.
  const unsigned bits = 8 * soFile->getBytesInAddress();

  StringRef contents = GetGnuHashSection(soFile);
  if (contents.size() < 16)
    // We need to search if the library doesn't have .gnu.hash section!
    return true;
  const char *hashContent = contents.data();

  // See https://flapenguin.me/2017/05/10/elf-lookup-dt-gnu-hash/ for .gnu.hash
  // table layout.
  uint32_t maskWords = *reinterpret_cast<const uint32_t *>(hashContent + 8);
  uint32_t shift2 = *reinterpret_cast<const uint32_t *>(hashContent + 12);
  uint32_t hash2 = hash >> shift2;
  uint32_t n = (hash / bits) % maskWords;

  const char *bloomfilter = hashContent + 16;
  const char *hash_pos = bloomfilter + n * (bits / 8); // * (Bits / 8)
  uint64_t word = *reinterpret_cast<const uint64_t *>(hash_pos);
  uint64_t bitmask = ((1ULL << (hash % bits)) | (1ULL << (hash2 % bits)));
  return (bitmask & word) == bitmask;
}

} // namespace

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string GetExecutablePath() {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  return llvm::orc::AutoLoadDynamicLibraryLookup::getSymbolLocation(
      &GetExecutablePath);
}

namespace llvm {
namespace orc {
class DynamicLoader {
  struct BasePathHashFunction {
    size_t operator()(const BasePath &item) const {
      return std::hash<std::string>()(item);
    }
  };

  struct BasePathEqFunction {
    size_t operator()(const BasePath &l, const BasePath &r) const {
      return &l == &r || l == r;
    }
  };
  /// A memory efficient VectorSet. The class provides O(1) search
  /// complexity. It is tuned to compare BasePaths first by checking the
  /// address and then the representation which models the base path reuse.
  class BasePaths {
  public:
    std::unordered_set<BasePath, BasePathHashFunction, BasePathEqFunction>
        Paths;

  public:
    const BasePath &RegisterBasePath(const std::string &Path,
                                     bool *WasInserted = nullptr) {
      auto it = Paths.insert(Path);
      if (WasInserted)
        *WasInserted = it.second;

      return *it.first;
    }

    bool Contains(StringRef Path) { return Paths.count(Path.str()); }
  };

  bool FirstRun = true;
  bool FirstRunSysLib = true;
  bool UseBloomFilter = true;
  bool UseHashTable = true;

  const AutoLoadDynamicLibraryLookup &AutoLoadDylibMgr;

  /// The basename of `/home/.../lib/libA.so`,
  /// BasePaths will contain `/home/.../lib/`
  BasePaths BasePaths;

  LibraryPaths Libraries;
  LibraryPaths SysLibraries;
  /// Contains a set of libraries which we gave to the user via ResolveSymbol
  /// call and next time we should check if the user loaded them to avoid
  /// useless iterations.
  LibraryPaths QueriedLibraries;

  using PermanentlyIgnoreCallbackProto = std::function<bool(StringRef)>;
  const PermanentlyIgnoreCallbackProto ShouldPermanentlyIgnoreCallback;
  const StringRef ExecutableFormat;

  /// Scan for shared objects which are not yet loaded. They are a our symbol
  /// resolution candidate sources.
  /// NOTE: We only scan not loaded shared objects.
  /// \param[in] searchSystemLibraries - whether to decent to standard system
  ///            locations for shared objects.
  void ScanForLibraries(bool searchSystemLibraries = false);

  /// Builds a bloom filter lookup optimization.
  void BuildBloomFilter(LibraryPath *Lib, object::ObjectFile *BinObjFile,
                        unsigned IgnoreSymbolFlags = 0) const;

  /// Looks up symbols from a an object file, representing the library.
  ///\param[in] Lib - full path to the library.
  ///\param[in] mangledName - the mangled name to look for.
  ///\param[in] IgnoreSymbolFlags - The symbols to ignore upon a match.
  ///\returns true on success.
  bool ContainsSymbol(const LibraryPath *Lib, StringRef mangledName,
                      unsigned IgnoreSymbolFlags = 0) const;

  bool ShouldPermanentlyIgnore(StringRef FileName) const;
  void dumpDebugInfo() const;

public:
  DynamicLoader(const AutoLoadDynamicLibraryLookup &DLM,
                PermanentlyIgnoreCallbackProto shouldIgnore,
                StringRef execFormat)
      : AutoLoadDylibMgr(DLM), ShouldPermanentlyIgnoreCallback(shouldIgnore),
        ExecutableFormat(execFormat) {}

  ~DynamicLoader(){};

  std::string searchLibrariesForSymbol(StringRef mangledName,
                                       bool searchSystem);

  void BuildGlobalBloomFilter(BloomFilter &Filter) const;
};

std::string RPathToStr(SmallVector<StringRef, 2> V) {
  std::string result;
  for (auto item : V)
    result += item.str() + ",";
  if (!result.empty())
    result.pop_back();
  return result;
}

template <class ELFT>
void HandleDynTab(const ELFFile<ELFT> *Elf, StringRef FileName,
                  SmallVector<StringRef, 2> &RPath,
                  SmallVector<StringRef, 2> &RunPath,
                  std::vector<StringRef> &Deps, bool &isPIEExecutable) {
#define DEBUG_TYPE "Dyld:"
  const char *Data = "";
  if (Expected<StringRef> StrTabOrErr = getDynamicStrTab(Elf))
    Data = StrTabOrErr.get().data();

  isPIEExecutable = false;

  auto DynamicEntriesOrError = Elf->dynamicEntries();
  if (!DynamicEntriesOrError) {
    LLVM_DEBUG(dbgs() << "Dyld: failed to read dynamic entries in"
                      << "'" << FileName.str() << "'\n");
    return;
  }

  for (const typename ELFT::Dyn &Dyn : *DynamicEntriesOrError) {
    switch (Dyn.d_tag) {
    case ELF::DT_NEEDED:
      Deps.push_back(Data + Dyn.d_un.d_val);
      break;
    case ELF::DT_RPATH:
      SplitPaths(Data + Dyn.d_un.d_val, RPath, SplitMode::AllowNonExistant,
                 kEnvDelim, false);
      break;
    case ELF::DT_RUNPATH:
      SplitPaths(Data + Dyn.d_un.d_val, RunPath, SplitMode::AllowNonExistant,
                 kEnvDelim, false);
      break;
    case ELF::DT_FLAGS_1:
      // Check if this is not a pie executable.
      if (Dyn.d_un.d_val & ELF::DF_1_PIE)
        isPIEExecutable = true;
      break;
      // (Dyn.d_tag == ELF::DT_NULL) continue;
      // (Dyn.d_tag == ELF::DT_AUXILIARY || Dyn.d_tag == ELF::DT_FILTER)
    }
  }
#undef DEBUG_TYPE
}

void DynamicLoader::ScanForLibraries(bool searchSystemLibraries /* = false*/) {
#define DEBUG_TYPE "Dyld:ScanForLibraries:"
  const auto &searchPaths = AutoLoadDylibMgr.getSearchPaths();

  LLVM_DEBUG({
    dbgs() << "Dyld::ScanForLibraries: system="
           << (searchSystemLibraries ? "true" : "false") << "\n";
    for (const AutoLoadDynamicLibraryLookup::SearchPathInfo &Info : searchPaths)
      dbgs() << ">>>" << Info.Path << ", "
             << (Info.IsUser ? "user\n" : "system\n");
  });

  SmallSet<const BasePath *, 32> ScannedPaths;
  // FileName must be always full/absolute/resolved file name.
  std::function<void(StringRef, unsigned)> ProcessLibraryFile =
      [&](StringRef FileName, unsigned level) {
        LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries HandleLib:"
                          << FileName.str() << ", level=" << level << " -> ");

        StringRef FileRealPath = sys::path::parent_path(FileName);
        StringRef FileRealName = sys::path::filename(FileName);
        const BasePath &BaseP = BasePaths.RegisterBasePath(FileRealPath.str());
        LibraryPath LibPath(BaseP, FileRealName.str());

        if (SysLibraries.GetRegisteredLib(LibPath) ||
            Libraries.GetRegisteredLib(LibPath)) {
          LLVM_DEBUG(dbgs() << "Already handled!!!\n");
          return;
        }

        if (ShouldPermanentlyIgnore(FileName)) {
          LLVM_DEBUG(dbgs() << "PermanentlyIgnored!!!\n");
          return;
        }

        if (searchSystemLibraries)
          SysLibraries.RegisterLib(LibPath);
        else
          Libraries.RegisterLib(LibPath);

        SmallVector<StringRef, 2> RPath, RunPath;
        std::vector<StringRef> Deps;

        auto ObjFileOrErr = object::ObjectFile::createObjectFile(FileName);
        if (Error Err = ObjFileOrErr.takeError()) {
          LLVM_DEBUG({
            std::string Message;
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              Message += EIB.message() + "; ";
            });
            dbgs() << "Dyld::ScanForLibraries: Failed to read object file "
                   << FileName.str() << " Errors: " << Message << "\n";
          });
          return;
        }

        object::ObjectFile *BinObjF = ObjFileOrErr.get().getBinary();
        if (BinObjF->isELF()) {
          bool isPIEExecutable = false;

          if (const auto *ELF = dyn_cast<ELF32LEObjectFile>(BinObjF))
            HandleDynTab(&ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                         isPIEExecutable);
          else if (const auto *ELF = dyn_cast<ELF32BEObjectFile>(BinObjF))
            HandleDynTab(&ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                         isPIEExecutable);
          else if (const auto *ELF = dyn_cast<ELF64LEObjectFile>(BinObjF))
            HandleDynTab(&ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                         isPIEExecutable);
          else if (const auto *ELF = dyn_cast<ELF64BEObjectFile>(BinObjF))
            HandleDynTab(&ELF->getELFFile(), FileName, RPath, RunPath, Deps,
                         isPIEExecutable);

          if (level == 0 && isPIEExecutable) {
            if (searchSystemLibraries)
              SysLibraries.UnregisterLib(LibPath);
            else
              Libraries.UnregisterLib(LibPath);
            return;
          }
        } else if (BinObjF->isMachO()) {
          MachOObjectFile *Obj = dyn_cast<MachOObjectFile>(BinObjF);
          for (const auto &Command : Obj->load_commands()) {
            if (Command.C.cmd == MachO::LC_LOAD_DYLIB) {
              MachO::dylib_command dylibCmd =
                  Obj->getDylibIDLoadCommand(Command);
              Deps.push_back(StringRef(Command.Ptr + dylibCmd.dylib.name));
            } else if (Command.C.cmd == MachO::LC_RPATH) {
              MachO::rpath_command rpathCmd = Obj->getRpathCommand(Command);
              SplitPaths(Command.Ptr + rpathCmd.path, RPath,
                         SplitMode::AllowNonExistant, kEnvDelim, false);
            }
          }
        } else if (BinObjF->isCOFF()) {
          // TODO: Implement COFF support
        }

        LLVM_DEBUG({
          dbgs() << "Dyld::ScanForLibraries: Deps Info:\n";
          dbgs() << "   RPATH=" << RPathToStr(RPath) << "\n";
          dbgs() << "   RUNPATH=" << RPathToStr(RunPath) << "\n";
          for (size_t i = 0; i < Deps.size(); ++i)
            dbgs() << "   Deps[" << i << "]=" << Deps[i].str() << "\n";
        });

        // Heuristics for workaround performance problems:
        // (H1) If RPATH and RUNPATH == "" -> skip handling Deps
        if (RPath.empty() && RunPath.empty()) {
          LLVM_DEBUG(dbgs()
                     << "Dyld::ScanForLibraries: Skip all deps by Heuristic1: "
                     << FileName.str() << "\n");
          return;
        }

        // (H2) If RPATH subset of LD_LIBRARY_PATH &&
        // RUNPATH subset of LD_LIBRARY_PATH  -> skip handling Deps
        if (std::all_of(
                RPath.begin(), RPath.end(),
                [&](StringRef item) {
                  return std::any_of(
                      searchPaths.begin(), searchPaths.end(),
                      [&](const AutoLoadDynamicLibraryLookup::SearchPathInfo
                              &info) { return item == info.Path; });
                }) &&
            std::all_of(RunPath.begin(), RunPath.end(), [&](StringRef item) {
              return std::any_of(
                  searchPaths.begin(), searchPaths.end(),
                  [&](const AutoLoadDynamicLibraryLookup::SearchPathInfo
                          &info) { return item == info.Path; });
            })) {
          LLVM_DEBUG(dbgs()
                     << "Dyld::ScanForLibraries: Skip all deps by Heuristic2: "
                     << FileName.str() << "\n");
          return;
        }

        // Recursively handle dependencies
        for (StringRef dep : Deps) {
          std::string dep_full = AutoLoadDylibMgr.lookupLibrary(
              dep, RPath, RunPath, FileName, false);
          ProcessLibraryFile(dep_full, level + 1);
        }
      };

  for (const AutoLoadDynamicLibraryLookup::SearchPathInfo &Info : searchPaths) {
    if (Info.IsUser != searchSystemLibraries) {
      // Examples which we should handle.
      // File                      Real
      // /lib/1/1.so               /lib/1/1.so  // file
      // /lib/1/2.so->/lib/1/1.so  /lib/1/1.so  // file local link
      // /lib/1/3.so->/lib/3/1.so  /lib/3/1.so  // file external link
      // /lib/2->/lib/1                         // path link
      // /lib/2/1.so               /lib/1/1.so  // path link, file
      // /lib/2/2.so->/lib/1/1.so  /lib/1/1.so  // path link, file local link
      // /lib/2/3.so->/lib/3/1.so  /lib/3/1.so  // path link, file external link
      //
      // /lib/3/1.so
      // /lib/3/2.so->/system/lib/s.so
      // /lib/3/3.so
      // /system/lib/1.so
      //
      // libL.so NEEDED/RPATH libR.so    /lib/some-rpath/libR.so  //
      // needed/dependedt library in libL.so RPATH/RUNPATH or other (in)direct
      // dep
      //
      // Paths = /lib/1 : /lib/2 : /lib/3

      // BasePaths = ["/lib/1", "/lib/3", "/system/lib"]
      // *Libraries  = [<0,"1.so">, <1,"1.so">, <2,"s.so">, <1,"3.so">]

      LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries Iter:" << Info.Path
                        << " -> ");
      std::string RealPath = cached_realpath(Info.Path);

      StringRef DirPath(RealPath);
      LLVM_DEBUG(dbgs() << RealPath << "\n");

      if (!sys::fs::is_directory(DirPath) || DirPath.empty())
        continue;

      // Already searched?
      const BasePath &ScannedBPath = BasePaths.RegisterBasePath(RealPath);
      if (ScannedPaths.count(&ScannedBPath)) {
        LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries Already scanned: "
                          << RealPath << "\n");
        continue;
      }

      LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries: Iterator: " << DirPath
                        << "\n");
      std::error_code EC;
      for (sys::fs::directory_iterator DirIt(DirPath, EC), DirEnd;
           DirIt != DirEnd && !EC; DirIt.increment(EC)) {

        LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries: Iterator >>> "
                          << DirIt->path()
                          << ", type=" << (short)(DirIt->type()) << "\n");

        const sys::fs::file_type ft = DirIt->type();
        if (ft == sys::fs::file_type::regular_file) {
          ProcessLibraryFile(DirIt->path(), 0);
        } else if (ft == sys::fs::file_type::symlink_file) {
          std::string DepFileName_str = cached_realpath(DirIt->path());
          StringRef DepFileName = DepFileName_str;
          assert(!sys::fs::is_symlink_file(DepFileName));
          if (!sys::fs::is_directory(DepFileName))
            ProcessLibraryFile(DepFileName, 0);
        }
      }
      for (sys::fs::directory_iterator DirIt(DirPath, EC), DirEnd;
           DirIt != DirEnd && !EC; DirIt.increment(EC)) {
        LLVM_DEBUG(dbgs() << "Dyld::ScanForLibraries: File " << DirIt->path()
                          << ", type=" << (short)DirIt->type() << "\n");

        const sys::fs::file_type ft = DirIt->type();
        if (ft == sys::fs::file_type::regular_file ||
            (ft == sys::fs::file_type::symlink_file &&
             !sys::fs::is_symlink_file(DirIt->path()))) {
          ProcessLibraryFile(DirIt->path(), 0);
        }
      }

      // Register the DirPath as fully scanned.
      ScannedPaths.insert(&ScannedBPath);
    }
  }
#undef DEBUG_TYPE
}

void DynamicLoader::BuildBloomFilter(LibraryPath *Lib,
                                     object::ObjectFile *BinObjFile,
                                     unsigned IgnoreSymbolFlags /*= 0*/) const {
#define DEBUG_TYPE "Dyld::BuildBloomFilter:"
  assert(UseBloomFilter && "Bloom filter is disabled");
  assert(!Lib->hasBloomFilter() && "Bloom filter already built!");

  using namespace llvm;
  // using namespace object;

  LLVM_DEBUG(dbgs() << "Dyld::BuildBloomFilter: Building Bloom filter for: "
                    << Lib->GetFullName() << "\n");

  std::vector<StringRef> symbols;
  uint32_t SymbolsCount = 0;
  // Helper to process each symbol from a range
  auto ProcessSymbols = [&](auto range) {
    for (const object::SymbolRef &S : range) {
      uint32_t Flags = cantFail(S.getFlags());

      // Skip symbols based on the flags (e.g., undefined or ignored)
      if (Flags & IgnoreSymbolFlags || Flags & object::SymbolRef::SF_Undefined)
        continue;

      Expected<StringRef> SymName = S.getName();
      if (!SymName || SymName->empty()) {
        LLVM_DEBUG(dbgs() << "Dyld::BuildBloomFilter: Skipped empty or failed "
                             "to read symbol\n");
        continue;
      }

      symbols.push_back(*SymName);
      ++SymbolsCount;
    }
  };

  ProcessSymbols(BinObjFile->symbols());

  if (BinObjFile->isELF()) {
    // ELF file format has .dynstr section for the dynamic symbol table.
    const auto *ElfObj = cast<object::ELFObjectFileBase>(BinObjFile);
    ProcessSymbols(ElfObj->getDynamicSymbolIterators());
  } else if (BinObjFile->isCOFF()) { // On Windows, the symbols are present in
                                     // COFF format.
    object::COFFObjectFile *CoffObj = cast<object::COFFObjectFile>(BinObjFile);
    // In COFF, the symbols are not present in the SymbolTable section
    // of the Object file. They are present in the ExportDirectory section.
    for (const object::ExportDirectoryEntryRef &D :
         CoffObj->export_directories()) {
      // All the symbols are already flagged as exported.
      // We cannot really ignore symbols based on flags as we do on unix.
      StringRef Name;
      auto Err = D.getSymbolName(Name);

      if (Err) {
        std::string Message;
        handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
          Message += EIB.message() + "; ";
        });
        LLVM_DEBUG(dbgs() << "Dyld::BuildBloomFilter: Failed to read symbol "
                          << Message << "\n");
        continue;
      }
      if (Name.empty())
        continue;

      ++SymbolsCount;
      symbols.push_back(Name);
    }
  }

  // Initialize the Bloom filter with the count of symbols
  Lib->InitializeBloomFilter(SymbolsCount);

  if (!SymbolsCount) {
    LLVM_DEBUG(dbgs() << "Dyld::BuildBloomFilter: No symbols found.\n");
    return;
  }

  LLVM_DEBUG({
    dbgs() << "Dyld::BuildBloomFilter: Adding symbols to Bloom filter:\n";
    for (const auto &Sym : symbols)
      dbgs() << "- " << Sym << "\n";
  });

  // Add symbols to the Bloom filter
  for (const auto &S : symbols) {
    Lib->AddBloom(UseHashTable ? Lib->AddSymbol(S.str()) : S);
  }
#undef DEBUG_TYPE
}

bool DynamicLoader::ContainsSymbol(const LibraryPath *Lib,
                                   StringRef mangledName,
                                   unsigned IgnoreSymbolFlags /*= 0*/) const {
#define DEBUG_TYPE "Dyld::ContainsSymbol:"
  // Helper lambda to handle symbol search and logging
  auto logAndReturn = [](bool result, const std::string &msg) {
    LLVM_DEBUG(dbgs() << msg);
    return result;
  };

  const std::string library_filename = Lib->GetFullName();

  LLVM_DEBUG(dbgs() << "Dyld::ContainsSymbol: Find symbol: lib="
                    << library_filename << ", mangled=" << mangledName.str()
                    << "\n");

  auto ObjF = object::ObjectFile::createObjectFile(library_filename);
  if (Error Err = ObjF.takeError()) {
    std::string Message;
    handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
      Message += EIB.message() + "; ";
    });
    return logAndReturn(false,
                        "Dyld::ContainsSymbol: Failed to read object file " +
                            library_filename + " Errors: " + Message + "\n");
  }

  object::ObjectFile *BinObjFile = ObjF.get().getBinary();
  uint32_t hashedMangle = GNUHash(mangledName);

  // Check for the gnu.hash section if ELF and exit early if symbol doesn't
  // exist
  if (BinObjFile->isELF() &&
      !MayExistInElfObjectFile(BinObjFile, hashedMangle)) {
    return logAndReturn(false,
                        "Dyld::ContainsSymbol: ELF BloomFilter: Skip symbol <" +
                            mangledName.str() + ">.\n");
  }

  // Use Bloom filter if enabled
  if (UseBloomFilter) {
    // Use our bloom filters and create them if necessary.
    if (!Lib->hasBloomFilter()) {
      BuildBloomFilter(const_cast<LibraryPath *>(Lib), BinObjFile,
                       IgnoreSymbolFlags);
    }

    // If the symbol does not exist, exit early. In case it may exist, iterate.
    if (!Lib->MayExistSymbol(mangledName)) {
      return logAndReturn(false,
                          "Dyld::ContainsSymbol: BloomFilter: Skip symbol <" +
                              mangledName.str() + ">.\n");
    }
    LLVM_DEBUG(dbgs() << "Dyld::ContainsSymbol: BloomFilter: Symbol <"
                      << mangledName.str() << "> May exist. Search for it.\n");
  }

  // Use hash table if enabled
  if (UseHashTable) {
    bool result = Lib->ExistSymbol(mangledName);
    return logAndReturn(
        result, result ? "Dyld::ContainsSymbol: HashTable: Symbol Exist\n"
                       : "Dyld::ContainsSymbol: HashTable: Symbol Not exis\n");
  }

  // Symbol not found in hash table; iterate through all symbols
  LLVM_DEBUG(dbgs() << "Dyld::ContainsSymbol: Iterate all for <"
                    << mangledName.str() << ">");

  auto ForeachSymbol =
      [&library_filename](iterator_range<object::symbol_iterator> range,
                          unsigned IgnoreSymbolFlags,
                          StringRef mangledName) -> bool {
    for (const object::SymbolRef &S : range) {

      uint32_t Flags = cantFail(S.getFlags());
      if (Flags & IgnoreSymbolFlags)
        continue;

      // Note, we are at last resort and loading library based on a weak
      // symbol is allowed. Otherwise, the JIT will issue an unresolved
      // symbol error.
      //
      // There are other weak symbol kinds (marked as 'V') to denote
      // typeinfo and vtables. It is unclear whether we should load such
      // libraries or from which library we should resolve the symbol.
      // We seem to not have a way to differentiate it from the symbol API.

      Expected<StringRef> SymNameErr = S.getName();
      if (!SymNameErr || SymNameErr.get().empty())
        continue;

      if (SymNameErr.get() == mangledName) {
        return true;
      }
    }
    return false;
  };

  // If no hash symbol then iterate to detect symbol
  // We Iterate only if BloomFilter and/or SymbolHashTable are not supported.

  // Symbol may exist. Iterate.
  if (ForeachSymbol(BinObjFile->symbols(), IgnoreSymbolFlags, mangledName)) {
    return logAndReturn(true, "Dyld::ContainsSymbol: Symbol " +
                                  mangledName.str() + " found in " +
                                  library_filename + "\n");
  }

  // Check dynamic symbols for ELF files
  if (BinObjFile->isELF()) {
    // ELF file format has .dynstr section for the dynamic symbol table.
    const auto *ElfObj = cast<object::ELFObjectFileBase>(BinObjFile);
    bool result = ForeachSymbol(ElfObj->getDynamicSymbolIterators(),
                                IgnoreSymbolFlags, mangledName);
    return logAndReturn(
        result,
        result ? "Dyld::ContainsSymbol: Symbol found in ELF dynamic symbols.\n"
               : "Dyld::ContainsSymbol: Symbol not found in ELF dynamic "
                 "symbols.\n");
  }

  return logAndReturn(false, "Dyld::ContainsSymbol: Symbol not found.\n");
#undef DEBUG_TYPE
}

bool DynamicLoader::ShouldPermanentlyIgnore(StringRef FileName) const {
#define DEBUG_TYPE "Dyld:"
  assert(!ExecutableFormat.empty() && "Failed to find the object format!");

  if (!AutoLoadDynamicLibraryLookup::isSharedLibrary(FileName))
    return true;

  // No need to check linked libraries, as this function is only invoked
  // for symbols that cannot be found (neither by dlsym nor in the JIT).
  if (AutoLoadDylibMgr.isLibraryLoaded(FileName))
    return true;

  auto ObjF = object::ObjectFile::createObjectFile(FileName);
  if (!ObjF) {
    LLVM_DEBUG(dbgs() << "[DyLD] Failed to read object file " << FileName
                      << "\n");
    return true;
  }

  object::ObjectFile *file = ObjF.get().getBinary();

  LLVM_DEBUG(dbgs() << "Current executable format: " << ExecutableFormat
                    << ". Executable format of " << FileName << " : "
                    << file->getFileFormatName() << "\n");

  // Ignore libraries with different format than the executing one.
  if (ExecutableFormat != file->getFileFormatName())
    return true;

  if (isa<object::ELFObjectFileBase>(*file)) {
    for (auto S : file->sections()) {
      StringRef name = cantFail(S.getName());
      if (name == ".text") {
        // Check if the library has only debug symbols, usually when
        // stripped with objcopy --only-keep-debug. This check is done by
        // reading the manual of objcopy and inspection of stripped with
        // objcopy libraries.
        auto SecRef = static_cast<object::ELFSectionRef &>(S);
        if (SecRef.getType() == ELF::SHT_NOBITS)
          return true;

        return (SecRef.getFlags() & ELF::SHF_ALLOC) == 0;
      }
    }
    return true;
  }

  // FIXME: Handle osx using isStripped after upgrading to llvm9.

  return ShouldPermanentlyIgnoreCallback(FileName);
#undef DEBUG_TYPE
}

void DynamicLoader::dumpDebugInfo() const {
#define DEBUG_TYPE "Dyld:"
  LLVM_DEBUG({
    dbgs() << "---\n";
    size_t x = 0;
    for (auto const &item : BasePaths.Paths) {
      dbgs() << "Dyld: - BasePaths[" << x++ << "]:" << &item << ": " << item
             << "\n";
    }
    dbgs() << "---\n";
    x = 0;
    for (auto const &item : Libraries.GetLibraries()) {
      dbgs() << "Dyld: - Libraries[" << x++ << "]:" << &item << ": "
             << item->Path << ", " << item->LibName << "\n";
    }
    x = 0;
    for (auto const &item : SysLibraries.GetLibraries()) {
      dbgs() << "Dyld: - SysLibraries[" << x++ << "]:" << &item << ": "
             << item->Path << ", " << item->LibName << "\n";
    }
  });
#undef DEBUG_TYPE
}

std::string
DynamicLoader::searchLibrariesForSymbol(StringRef mangledName,
                                        bool searchSystem /* = true*/) {
#define DEBUG_TYPE "Dyld:searchLibrariesForSymbol:"
  assert(!sys::DynamicLibrary::SearchForAddressOfSymbol(mangledName.str()) &&
         "Library already loaded, please use dlsym!");
  assert(!mangledName.empty());

  using namespace sys::path;
  using namespace sys::fs;

  if (FirstRun) {
    LLVM_DEBUG(dbgs() << "Dyld::searchLibrariesForSymbol:" << mangledName.str()
                      << ", searchSystem=" << (searchSystem ? "true" : "false")
                      << ", FirstRun(user)... scanning\n");

    LLVM_DEBUG(
        dbgs()
        << "Dyld::searchLibrariesForSymbol: Before first ScanForLibraries\n");
    dumpDebugInfo();

    ScanForLibraries(/* SearchSystemLibraries= */ false);
    FirstRun = false;

    LLVM_DEBUG(
        dbgs()
        << "Dyld::searchLibrariesForSymbol: After first ScanForLibraries\n");
    dumpDebugInfo();
  }

  if (QueriedLibraries.size() > 0) {
    // Last call we were asked if a library contains a symbol. Usually, the
    // caller wants to load this library. Check if was loaded and remove it
    // from our lists of not-yet-loaded libs.

    LLVM_DEBUG({
      dbgs() << "Dyld::ResolveSymbol: QueriedLibraries:\n";
      size_t x = 0;
      for (auto item : QueriedLibraries.GetLibraries()) {
        dbgs() << "Dyld::ResolveSymbol - [" << x++ << "]:" << &item << ": "
               << item->GetFullName() << "\n";
      }
    });

    for (const LibraryPath *P : QueriedLibraries.GetLibraries()) {
      const std::string LibName = P->GetFullName();
      if (!AutoLoadDylibMgr.isLibraryLoaded(LibName))
        continue;

      Libraries.UnregisterLib(*P);
      SysLibraries.UnregisterLib(*P);
    }
    // TODO:  QueriedLibraries.clear ?
  }

  // Iterate over files under this path. We want to get each ".so" files
  for (const LibraryPath *P : Libraries.GetLibraries()) {
    if (ContainsSymbol(P, mangledName, /*ignore*/
                       object::SymbolRef::SF_Undefined)) {
      if (!QueriedLibraries.HasRegisteredLib(*P))
        QueriedLibraries.RegisterLib(*P);

      LLVM_DEBUG(
          dbgs() << "Dyld::ResolveSymbol: Search found match in [user lib]: "
                 << P->GetFullName() << "!\n");

      return P->GetFullName();
    }
  }

  if (!searchSystem)
    return "";

  LLVM_DEBUG(dbgs() << "Dyld::searchLibrariesForSymbol: SearchSystem!!!\n");

  // Lookup in non-system libraries failed. Expand the search to the system.
  if (FirstRunSysLib) {
    LLVM_DEBUG(dbgs() << "Dyld::searchLibrariesForSymbol:" << mangledName.str()
                      << ", searchSystem=" << (searchSystem ? "true" : "false")
                      << ", FirstRun(system)... scanning\n");

    LLVM_DEBUG(dbgs() << "Dyld::searchLibrariesForSymbol: Before first system "
                         "ScanForLibraries\n");
    dumpDebugInfo();

    ScanForLibraries(/* SearchSystemLibraries= */ true);
    FirstRunSysLib = false;

    LLVM_DEBUG(dbgs() << "Dyld::searchLibrariesForSymbol: After first system "
                         "ScanForLibraries\n");
    dumpDebugInfo();
  }

  for (const LibraryPath *P : SysLibraries.GetLibraries()) {
    if (ContainsSymbol(P, mangledName, /*ignore*/
                       object::SymbolRef::SF_Undefined |
                           object::SymbolRef::SF_Weak)) {
      if (!QueriedLibraries.HasRegisteredLib(*P))
        QueriedLibraries.RegisterLib(*P);

      LLVM_DEBUG(
          dbgs() << "Dyld::ResolveSymbol: Search found match in [system lib]: "
                 << P->GetFullName() << "!\n");

      return P->GetFullName();
    }
  }

  LLVM_DEBUG(dbgs() << "Dyld::ResolveSymbol: Search found no match!\n");

  return ""; // Search found no match.
#undef DEBUG_TYPE
}

void DynamicLoader::BuildGlobalBloomFilter(BloomFilter &Filter) const {
  assert(!Filter.IsInitialized());
  uint32_t GloabalBloomSize = 0;

  for (auto *L : Libraries.GetLibraries()) {
    GloabalBloomSize += L->Filter.getSymCount();
  }

  for (auto *L : SysLibraries.GetLibraries()) {
    GloabalBloomSize += L->Filter.getSymCount();
  }

  Filter.Initialize(GloabalBloomSize);
  for (auto *L : Libraries.GetLibraries()) {
    for (auto &S : L->Symbols) {
      Filter.AddSymbol(S.getKey());
    }
  }

  for (auto *L : SysLibraries.GetLibraries()) {
    for (auto &S : L->Symbols) {
      Filter.AddSymbol(S.getKey());
    }
  }
}

AutoLoadDynamicLibraryLookup::~AutoLoadDynamicLibraryLookup() {
  static_assert(sizeof(Dyld) > 0, "Incomplete type");
  delete Dyld;
}

void AutoLoadDynamicLibraryLookup::initializeDynamicLoader(
    std::function<bool(StringRef)> shouldPermanentlyIgnore) {
  assert(!Dyld && "Already initialized!");
  if (Dyld)
    return;
  std::string exeP = GetExecutablePath();
  auto ObjF = cantFail(object::ObjectFile::createObjectFile(exeP));

  Dyld = new DynamicLoader(*this, shouldPermanentlyIgnore,
                           ObjF.getBinary()->getFileFormatName());
}

void AutoLoadDynamicLibraryLookup::BuildGlobalBloomFilter(
    BloomFilter &Filter) const {
  Dyld->BuildGlobalBloomFilter(Filter);
}

std::string AutoLoadDynamicLibraryLookup::searchLibrariesForSymbol(
    StringRef mangledName, bool searchSystem /* = true*/) const {
  assert(Dyld && "Must call initialize dyld before!");
  return Dyld->searchLibrariesForSymbol(mangledName, searchSystem);
}

std::string AutoLoadDynamicLibraryLookup::getSymbolLocation(void *func) {
#if defined(__CYGWIN__) && defined(__GNUC__)
  return {};
#elif defined(_WIN32)
  MEMORY_BASIC_INFORMATION mbi;
  if (!VirtualQuery(func, &mbi, sizeof(mbi)))
    return {};

  HMODULE hMod = (HMODULE)mbi.AllocationBase;
  char moduleName[MAX_PATH];

  if (!GetModuleFileNameA(hMod, moduleName, sizeof(moduleName)))
    return {};

  return cached_realpath(moduleName);

#else
  // assume we have  defined HAVE_DLFCN_H and HAVE_DLADDR
  Dl_info info;
  if (dladdr((void *)func, &info) == 0) {
    // Not in a known shared library, let's give up
    return {};
  } else {
    std::string result = cached_realpath(info.dli_fname);
    if (!result.empty())
      return result;

      // Else absolute path. For all we know that's a binary.
      // Some people have dictionaries in binaries, this is how we find their
      // path: (see also https://stackoverflow.com/a/1024937/6182509)
#if defined(__APPLE__)
    char buf[PATH_MAX] = {0};
    uint32_t bufsize = sizeof(buf);
    if (_NSGetExecutablePath(buf, &bufsize) >= 0)
      return cached_realpath(buf);
    return cached_realpath(info.dli_fname);
#elif defined(LLVM_ON_UNIX)
    char buf[PATH_MAX] = {0};
    // Cross our fingers that /proc/self/exe exists.
    if (readlink("/proc/self/exe", buf, sizeof(buf)) > 0)
      return cached_realpath(buf);
    std::string pipeCmd = std::string("which \"") + info.dli_fname + "\"";
    FILE *pipe = popen(pipeCmd.c_str(), "r");
    if (!pipe)
      return cached_realpath(info.dli_fname);
    while (fgets(buf, sizeof(buf), pipe))
      result += buf;

    pclose(pipe);
    return cached_realpath(result);
#else
#error "Unsupported platform."
#endif
    return {};
  }
#endif
}

} // namespace orc
} // namespace llvm
