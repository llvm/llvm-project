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

bool LibraryResolutionDriver::markLibraryLoaded(StringRef Path) {
  auto Lib = LR->LibMgr.getLibrary(Path);
  if (!Lib)
    return false;

  Lib->setState(LibraryManager::LibState::Loaded);

  return true;
}

bool LibraryResolutionDriver::markLibraryUnLoaded(StringRef Path) {
  auto Lib = LR->LibMgr.getLibrary(Path);
  if (!Lib)
    return false;

  Lib->setState(LibraryManager::LibState::Unloaded);

  return true;
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

bool SymbolEnumerator::enumerateSymbols(StringRef Path, OnEachSymbolFn OnEach,
                                        const SymbolEnumeratorOptions &Opts) {
  if (Path.empty())
    return false;

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

  object::ObjectFile *Obj = &ObjOrErr.get();

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

      if (OnEach(Name) != EnumerateResult::Continue)
        return false;
    }
  } else if (Obj->isMachO()) {
  }

  return true;
}

class SymbolSearchContext {
public:
  SymbolSearchContext(SymbolQuery &Q) : Q(Q) {}

  bool hasSearched(LibraryInfo *Lib) const { return Searched.count(Lib); }

  void markSearched(LibraryInfo *Lib) { Searched.insert(Lib); }

  inline bool allResolved() const { return Q.allResolved(); }

  SymbolQuery &query() { return Q; }

private:
  SymbolQuery &Q;
  DenseSet<LibraryInfo *> Searched;
};

void LibraryResolver::resolveSymbolsInLibrary(
    LibraryInfo &Lib, SymbolQuery &UnresolvedSymbols,
    const SymbolEnumeratorOptions &Opts) {
  LLVM_DEBUG(dbgs() << "Checking unresolved symbols "
                    << " in library : " << Lib.getFileName() << "\n";);
  StringSet<> DiscoveredSymbols;

  if (!UnresolvedSymbols.hasUnresolved()) {
    LLVM_DEBUG(dbgs() << "Skipping library: " << Lib.getFullPath()
                      << " — unresolved symbols exist.\n";);
    return;
  }

  bool HasEnumerated = false;
  auto enumerateSymbolsIfNeeded = [&]() {
    if (HasEnumerated)
      return;

    HasEnumerated = true;

    LLVM_DEBUG(dbgs() << "Enumerating symbols in library: " << Lib.getFullPath()
                      << "\n";);
    SymbolEnumerator::enumerateSymbols(
        Lib.getFullPath(),
        [&](StringRef sym) {
          DiscoveredSymbols.insert(sym);
          return EnumerateResult::Continue;
        },
        Opts);

    if (DiscoveredSymbols.empty()) {
      LLVM_DEBUG(dbgs() << "  No symbols and remove library : "
                        << Lib.getFullPath() << "\n";);
      LibMgr.removeLibrary(Lib.getFullPath());
      return;
    }
  };

  if (!Lib.hasFilter()) {
    LLVM_DEBUG(dbgs() << "Building filter for library: " << Lib.getFullPath()
                      << "\n";);
    enumerateSymbolsIfNeeded();
    SmallVector<StringRef> SymbolVec;
    SymbolVec.reserve(DiscoveredSymbols.size());
    for (const auto &KV : DiscoveredSymbols)
      SymbolVec.push_back(KV.first());

    Lib.ensureFilterBuilt(FB, SymbolVec);
    LLVM_DEBUG({
      dbgs() << "DiscoveredSymbols : " << DiscoveredSymbols.size() << "\n";
      for (const auto &KV : DiscoveredSymbols)
        dbgs() << "DiscoveredSymbols : " << KV.first() << "\n";
    });
  }

  const auto &Unresolved = UnresolvedSymbols.getUnresolvedSymbols();
  bool HadAnySym = false;
  LLVM_DEBUG(dbgs() << "Total unresolved symbols : " << Unresolved.size()
                    << "\n";);
  for (const auto &Sym : Unresolved) {
    if (Lib.mayContain(Sym)) {
      LLVM_DEBUG(dbgs() << "Checking symbol '" << Sym
                        << "' in library: " << Lib.getFullPath() << "\n";);
      enumerateSymbolsIfNeeded();
      if (DiscoveredSymbols.count(Sym) > 0) {
        LLVM_DEBUG(dbgs() << "  Resolved symbol: " << Sym
                          << " in library: " << Lib.getFullPath() << "\n";);
        UnresolvedSymbols.resolve(Sym, Lib.getFullPath());
        HadAnySym = true;
      }
    }
  }

  using LibraryState = LibraryManager::LibState;
  if (HadAnySym && Lib.getState() != LibraryState::Loaded)
    Lib.setState(LibraryState::Queried);
}

void LibraryResolver::searchSymbolsInLibraries(
    std::vector<std::string> &SymbolList, OnSearchComplete OnComplete,
    const SearchConfig &Config) {
  SymbolQuery Q(SymbolList);

  using LibraryState = LibraryManager::LibState;
  using LibraryType = PathType;
  auto tryResolveFrom = [&](LibraryState S, LibraryType K) {
    LLVM_DEBUG(dbgs() << "Trying resolve from state=" << static_cast<int>(S)
                      << " type=" << static_cast<int>(K) << "\n";);

    SymbolSearchContext Ctx(Q);
    while (!Ctx.allResolved()) {

      for (auto &Lib : LibMgr.getView(S, K)) {
        if (Ctx.hasSearched(Lib.get()))
          continue;

        // can use Async here?
        resolveSymbolsInLibrary(*Lib, Ctx.query(), Config.Options);
        Ctx.markSearched(Lib.get());

        if (Ctx.allResolved())
          return;
      }

      if (Ctx.allResolved())
        return;

      if (!scanLibrariesIfNeeded(K, scanBatchSize))
        break; // no more new libs to scan
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

bool LibraryResolver::symbolExistsInLibrary(const LibraryInfo &Lib,
                                            StringRef SymName,
                                            std::vector<std::string> *AllSyms) {
  SymbolEnumeratorOptions Opts;
  return symbolExistsInLibrary(Lib, SymName, AllSyms, Opts);
}

bool LibraryResolver::symbolExistsInLibrary(
    const LibraryInfo &Lib, StringRef SymName,
    std::vector<std::string> *AllSyms, const SymbolEnumeratorOptions &Opts) {
  bool Found = false;

  SymbolEnumerator::enumerateSymbols(
      Lib.getFullPath(),
      [&](StringRef Sym) {
        if (AllSyms)
          AllSyms->emplace_back(Sym.str());

        if (Sym == SymName) {
          Found = true;
        }

        return EnumerateResult::Continue;
      },
      Opts);

  return Found;
}

} // end namespace llvm::orc
