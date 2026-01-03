//===- LibraryResolver.cpp - Library Resolution of Unresolved Symbols ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Library resolution impl for unresolved symbols
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryResolver.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryScanner.h"

#include "llvm/ADT/StringSet.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/Error.h"

#include <mutex>

#define DEBUG_TYPE "orc-resolver"

namespace llvm::orc {

LibraryResolver::LibraryResolver(const LibraryResolver::Setup &S)
    : LibMgr(LibraryManager()),
      LibPathCache(std::make_shared<LibraryPathCache>()),
      LibPathResolver(std::make_shared<PathResolver>(LibPathCache)),
      ScanHelper(S.BasePaths, LibPathCache, LibPathResolver),
      FB(S.FilterBuilder),
      ShouldScanCall(S.ShouldScanCall ? S.ShouldScanCall
                                      : [](StringRef) -> bool { return true; }),
      scanBatchSize(S.ScanBatchSize) {

  if (!ScanHelper.hasSearchPath()) {
    LLVM_DEBUG(dbgs() << "Warning: No base paths provided for scanning.\n");
  }
}

std::unique_ptr<LibraryResolutionDriver>
LibraryResolutionDriver::create(const LibraryResolver::Setup &S) {
  auto LR = std::make_unique<LibraryResolver>(S);
  return std::unique_ptr<LibraryResolutionDriver>(
      new LibraryResolutionDriver(std::move(LR)));
}

void LibraryResolutionDriver::addScanPath(const std::string &Path, PathType K) {
  LR->ScanHelper.addBasePath(Path, K);
}

void LibraryResolutionDriver::markLibraryLoaded(StringRef Path) {
  LR->LibMgr.markLoaded(Path);
}

void LibraryResolutionDriver::markLibraryUnLoaded(StringRef Path) {
  LR->LibMgr.markUnloaded(Path);
}

void LibraryResolutionDriver::resolveSymbols(
    ArrayRef<StringRef> Symbols, LibraryResolver::OnSearchComplete OnCompletion,
    const SearchConfig &Config) {
  LR->searchSymbolsInLibraries(Symbols, std::move(OnCompletion), Config);
}

static bool shouldIgnoreSymbol(const object::SymbolRef &Sym,
                               uint32_t IgnoreFlags) {
  Expected<uint32_t> FlagsOrErr = Sym.getFlags();
  if (!FlagsOrErr) {
    consumeError(FlagsOrErr.takeError());
    return true;
  }

  uint32_t Flags = *FlagsOrErr;

  using Filter = SymbolEnumeratorOptions;
  if ((IgnoreFlags & Filter::IgnoreUndefined) &&
      (Flags & object::SymbolRef::SF_Undefined))
    return true;
  if ((IgnoreFlags & Filter::IgnoreNonExported) &&
      !(Flags & object::SymbolRef::SF_Exported))
    return true;
  if ((IgnoreFlags & Filter::IgnoreNonGlobal) &&
      !(Flags & object::SymbolRef::SF_Global))
    return true;
  if ((IgnoreFlags & Filter::IgnoreHidden) &&
      (Flags & object::SymbolRef::SF_Hidden))
    return true;
  if ((IgnoreFlags & Filter::IgnoreIndirect) &&
      (Flags & object::SymbolRef::SF_Indirect))
    return true;
  if ((IgnoreFlags & Filter::IgnoreWeak) &&
      (Flags & object::SymbolRef::SF_Weak))
    return true;

  return false;
}

bool SymbolEnumerator::enumerateSymbols(object::ObjectFile *Obj,
                                        OnEachSymbolFn OnEach,
                                        const SymbolEnumeratorOptions &Opts) {
  if (!Obj)
    return false;

  auto processSymbolRange =
      [&](object::ObjectFile::symbol_iterator_range Range) -> EnumerateResult {
    for (const auto &Sym : Range) {
      if (shouldIgnoreSymbol(Sym, Opts.FilterFlags))
        continue;

      auto NameOrErr = Sym.getName();
      if (!NameOrErr) {
        consumeError(NameOrErr.takeError());
        continue;
      }

      StringRef Name = *NameOrErr;
      if (Name.empty())
        continue;

      EnumerateResult Res = OnEach(Name);
      if (Res != EnumerateResult::Continue)
        return Res;
    }
    return EnumerateResult::Continue;
  };

  EnumerateResult Res = processSymbolRange(Obj->symbols());
  if (Res != EnumerateResult::Continue)
    return Res == EnumerateResult::Stop;

  if (Obj->isELF()) {
    const auto *ElfObj = cast<object::ELFObjectFileBase>(Obj);
    Res = processSymbolRange(ElfObj->getDynamicSymbolIterators());
    if (Res != EnumerateResult::Continue)
      return Res == EnumerateResult::Stop;
  } else if (Obj->isCOFF()) {
    const auto *CoffObj = cast<object::COFFObjectFile>(Obj);
    for (auto I = CoffObj->export_directory_begin(),
              E = CoffObj->export_directory_end();
         I != E; ++I) {
      StringRef Name;
      if (I->getSymbolName(Name))
        continue;
      if (Name.empty())
        continue;

      EnumerateResult Res = OnEach(Name);
      if (Res != EnumerateResult::Continue)
        return Res == EnumerateResult::Stop;
    }
  } else if (Obj->isMachO()) {
  }

  return true;
}

bool SymbolEnumerator::enumerateSymbols(StringRef Path, OnEachSymbolFn OnEach,
                                        const SymbolEnumeratorOptions &Opts) {
  ObjectFileLoader ObjLoader(Path);

  auto ObjOrErr = ObjLoader.getObjectFile();
  if (!ObjOrErr) {
    std::string ErrMsg;
    handleAllErrors(ObjOrErr.takeError(),
                    [&](const ErrorInfoBase &EIB) { ErrMsg = EIB.message(); });
    LLVM_DEBUG(dbgs() << "Failed loading object file: " << Path
                      << "\nError: " << ErrMsg << "\n");
    return false;
  }

  return SymbolEnumerator::enumerateSymbols(&ObjOrErr.get(), OnEach, Opts);
}

static StringRef GetGnuHashSection(llvm::object::ObjectFile *file) {
  for (auto S : file->sections()) {
    StringRef name = llvm::cantFail(S.getName());
    if (name == ".gnu.hash") {
      return llvm::cantFail(S.getContents());
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
static bool MayExistInElfObjectFile(llvm::object::ObjectFile *soFile,
                                    StringRef Sym) {
  assert(soFile->isELF() && "Not ELF");

  uint32_t hash = djbHash(Sym);
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

void LibraryResolver::resolveSymbolsInLibrary(
    LibraryInfo *Lib, SymbolQuery &Query, const SymbolEnumeratorOptions &Opts) {
  LLVM_DEBUG(dbgs() << "Checking unresolved symbols "
                    << " in library : " << Lib->getFileName() << "\n";);

  if (!Query.hasUnresolved()) {
    LLVM_DEBUG(dbgs() << "Skipping library: " << Lib->getFullPath()
                      << " — unresolved symbols exist.\n";);
    return;
  }

  bool HadAnySym = false;

  // Build candidate vector
  SmallVector<StringRef, 24> CandidateVec;

  Query.getUnresolvedSymbols(CandidateVec, [&](StringRef S) {
    return !Lib->hasFilter() || Lib->mayContain(S);
  });

  LLVM_DEBUG(dbgs() << "Total candidate symbols : " << CandidateVec.size()
                    << "\n";);
  if (CandidateVec.empty()) {
    LLVM_DEBUG(dbgs() << "No symbol Exist "
                         " in library: "
                      << Lib->getFullPath() << "\n";);
    return;
  }

  bool BuildingFilter = !Lib->hasFilter();

  ObjectFileLoader ObjLoader(Lib->getFullPath());
  auto ObjOrErr = ObjLoader.getObjectFile();
  if (!ObjOrErr) {
    std::string ErrMsg;
    handleAllErrors(ObjOrErr.takeError(),
                    [&](const ErrorInfoBase &EIB) { ErrMsg = EIB.message(); });
    LLVM_DEBUG(dbgs() << "Failed loading object file: " << Lib->getFullPath()
                      << "\nError: " << ErrMsg << "\n");
    return;
  }

  object::ObjectFile *Obj = &ObjOrErr.get();
  if (BuildingFilter && Obj->isELF()) {

    erase_if(CandidateVec,
             [&](StringRef C) { return !MayExistInElfObjectFile(Obj, C); });
    if (CandidateVec.empty())
      return;
  }

  SmallVector<StringRef, 256> SymbolVec;

  LLVM_DEBUG(dbgs() << "Enumerating symbols in library: " << Lib->getFullPath()
                    << "\n";);

  SymbolEnumerator::enumerateSymbols(
      Obj,
      [&](StringRef S) {
        // Collect symbols if we're building a filter
        if (BuildingFilter)
          SymbolVec.push_back(S);

        // auto It = std::lower_bound(CandidateVec.begin(),
        // CandidateVec.end(), S);
        auto It = std::find(CandidateVec.begin(), CandidateVec.end(), S);
        if (It != CandidateVec.end() && *It == S) {
          // Resolve and remove from CandidateVec
          LLVM_DEBUG(dbgs() << "Symbol '" << S << "' resolved in library: "
                            << Lib->getFullPath() << "\n";);
          Query.resolve(S, Lib->getFullPath());
          HadAnySym = true;
          *It = CandidateVec.back();
          CandidateVec.pop_back();

          // Stop — if nothing remains, stop enumeration
          if (!BuildingFilter && CandidateVec.empty()) {
            return EnumerateResult::Stop;
          }
          // Also stop if SymbolQuery has no more unresolved symbols
          if (!BuildingFilter && !Query.hasUnresolved())
            return EnumerateResult::Stop;
        }

        return EnumerateResult::Continue;
      },
      Opts);

  if (BuildingFilter) {
    LLVM_DEBUG(dbgs() << "Building filter for library: " << Lib->getFullPath()
                      << "\n";);
    if (SymbolVec.empty()) {
      LLVM_DEBUG(dbgs() << "  Skip : No symbols found in : "
                        << Lib->getFullPath() << "\n";);
      return;
    }

    Lib->ensureFilterBuilt(FB, SymbolVec);
    LLVM_DEBUG({
      dbgs() << "DiscoveredSymbols : " << SymbolVec.size() << "\n";
      for (const auto &S : SymbolVec)
        dbgs() << "DiscoveredSymbols : " << S << "\n";
    });
  }

  if (HadAnySym && Lib->getState() != LibState::Loaded)
    Lib->setState(LibState::Queried);
}

void LibraryResolver::searchSymbolsInLibraries(ArrayRef<StringRef> SymbolList,
                                               OnSearchComplete OnComplete,
                                               const SearchConfig &Config) {
  SymbolQuery Q(SymbolList);

  using LibraryType = PathType;
  auto tryResolveFrom = [&](LibState S, LibraryType K) {
    LLVM_DEBUG(dbgs() << "Trying resolve from state=" << static_cast<int>(S)
                      << " type=" << static_cast<int>(K) << "\n";);

    LibraryCursor Cur = LibMgr.getCursor(K, S);
    while (!Q.allResolved()) {
      const LibraryInfo *Lib = Cur.nextValidLib();
      // Cursor not valid?
      if (!Lib) {
        if (!scanForNewLibraries(K, Cur))
          break;  // nothing new was added
        continue; // Try to resolve next library
      }

      // can use Async here?
      resolveSymbolsInLibrary(const_cast<LibraryInfo *>(Lib), Q,
                              Config.Options);
      if (Q.allResolved())
        break;
    }
  };

  for (const auto &[St, Ty] : Config.Policy.Plan) {
    tryResolveFrom(St, Ty);
    if (Q.allResolved())
      break;
  }

  // done:
  LLVM_DEBUG({
    dbgs() << "Search complete.\n";
    for (const auto &r : Q.getAllResults())
      dbgs() << "Resolved Symbol:" << r->Name << " -> " << r->ResolvedLibPath
             << "\n";
  });

  OnComplete(Q);
}

bool LibraryResolver::scanForNewLibraries(PathType K, LibraryCursor &Cur) {
  while (ScanHelper.leftToScan(K)) {
    scanLibrariesIfNeeded(K, scanBatchSize);

    // Check if scanning added new libraries
    if (Cur.hasMoreValidLib())
      return true;
  }

  // No new libraries were added
  return false;
}

bool LibraryResolver::scanLibrariesIfNeeded(PathType PK, size_t BatchSize) {
  LLVM_DEBUG(dbgs() << "LibraryResolver::scanLibrariesIfNeeded: Scanning for "
                    << (PK == PathType::User ? "User" : "System")
                    << " libraries\n";);
  if (!ScanHelper.leftToScan(PK))
    return false;

  LibraryScanner Scanner(ScanHelper, LibMgr, ShouldScanCall);
  Scanner.scanNext(PK, BatchSize);
  return true;
}
} // end namespace llvm::orc
