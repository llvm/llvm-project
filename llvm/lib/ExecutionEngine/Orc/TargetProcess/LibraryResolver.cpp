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
#include "llvm/Support/Error.h"

#include <mutex>
#include <thread>

#define DEBUG_TYPE "orc-resolver"

namespace llvm::orc {

LibraryResolver::LibraryResolver(const LibraryResolver::Setup &S)
    : LibPathCache(S.Cache ? S.Cache : std::make_shared<LibraryPathCache>()),
      LibPathResolver(S.PResolver
                          ? S.PResolver
                          : std::make_shared<PathResolver>(LibPathCache)),
      ScanHelper(S.BasePaths, LibPathCache, LibPathResolver),
      FB(S.FilterBuilder), LibMgr(),
      ShouldScanCall(S.ShouldScanCall ? S.ShouldScanCall
                                      : [](StringRef) -> bool { return true; }),
      scanBatchSize(S.ScanBatchSize) {

  if (ScanHelper.getAllUnits().empty()) {
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
    std::vector<std::string> Syms,
    LibraryResolver::OnSearchComplete OnCompletion,
    const SearchConfig &Config) {
  LR->searchSymbolsInLibraries(Syms, std::move(OnCompletion), Config);
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

class SymbolSearchContext {
public:
  SymbolSearchContext(SymbolQuery &Q) : Q(Q) {}

  bool hasSearched(const LibraryInfo *Lib) const { return Searched.count(Lib); }

  void markSearched(const LibraryInfo *Lib) { Searched.insert(Lib); }

  inline bool allResolved() const { return Q.allResolved(); }

  SymbolQuery &query() { return Q; }

private:
  SymbolQuery &Q;
  DenseSet<const LibraryInfo *> Searched;
};

void LibraryResolver::resolveSymbolsInLibrary(
    LibraryInfo *Lib, SymbolQuery &UnresolvedSymbols,
    const SymbolEnumeratorOptions &Opts) {
  LLVM_DEBUG(dbgs() << "Checking unresolved symbols "
                    << " in library : " << Lib->getFileName() << "\n";);

  if (!UnresolvedSymbols.hasUnresolved()) {
    LLVM_DEBUG(dbgs() << "Skipping library: " << Lib->getFullPath()
                      << " — unresolved symbols exist.\n";);
    return;
  }

  bool HadAnySym = false;

  const auto &Unresolved = UnresolvedSymbols.getUnresolvedSymbols();
  LLVM_DEBUG(dbgs() << "Total unresolved symbols : " << Unresolved.size()
                    << "\n";);
  DenseSet<StringRef> CandidateSet;
  CandidateSet.reserve(Unresolved.size());
  for (const auto &Sym : Unresolved) {
    LLVM_DEBUG(dbgs() << "Checking symbol '" << Sym
                      << "' in filter: " << Lib->getFullPath() << "\n";);
    if (!Lib->hasFilter() || Lib->mayContain(Sym))
      CandidateSet.insert(Sym);
  }

  if (CandidateSet.empty()) {
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

  SmallVector<StringRef, 512> SymbolVec;

  LLVM_DEBUG(dbgs() << "Enumerating symbols in library: " << Lib->getFullPath()
                    << "\n";);
  SymbolEnumerator::enumerateSymbols(
      Obj,
      [&](StringRef S) {
        // If buildingFilter, collect for filter construction.
        if (BuildingFilter) {
          SymbolVec.push_back(S);
        }
        auto It = CandidateSet.find(S);
        if (It != CandidateSet.end()) {
          // Resolve symbol and remove from remaining set
          LLVM_DEBUG(dbgs() << "Symbol '" << S << "' resolved in library: "
                            << Lib->getFullPath() << "\n";);
          UnresolvedSymbols.resolve(*It, Lib->getFullPath());
          HadAnySym = true;

          // EARLY STOP — everything matched
          if (!BuildingFilter && !UnresolvedSymbols.hasUnresolved())
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

void LibraryResolver::searchSymbolsInLibraries(
    std::vector<std::string> &SymbolList, OnSearchComplete OnComplete,
    const SearchConfig &Config) {
  SymbolQuery Q(SymbolList);

  using LibraryType = PathType;
  auto tryResolveFrom = [&](LibState S, LibraryType K) {
    LLVM_DEBUG(dbgs() << "Trying resolve from state=" << static_cast<int>(S)
                      << " type=" << static_cast<int>(K) << "\n";);

    SymbolSearchContext Ctx(Q);
    LibraryCursor Cur = LibMgr.getCursor(K, S);
    while (!Ctx.allResolved()) {
      const LibraryInfo *Lib = Cur.nextValidLib();
      // Cursor not valid?
      if (!Lib) {
        bool NewLibAdded = false;

        // Scan as long as there is paths to scan
        while (ScanHelper.leftToScan(K)) {
          scanLibrariesIfNeeded(K, scanBatchSize);

          // NOW check any new libraries added
          if (Cur.hasMoreValidLib()) {
            NewLibAdded = true;
            break;
          }
        }

        if (!NewLibAdded)
          break; // No new libs found

        continue; // Try to resolve next library
      }

      // can use Async here?
      resolveSymbolsInLibrary(const_cast<LibraryInfo *>(Lib), Ctx.query(),
                              Config.Options);
      if (Ctx.allResolved())
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
