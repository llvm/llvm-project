//===----- DynamicLoader.cpp - Defaults for host process -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/DynamicLoader.h"
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

#define DEBUG_TYPE "orc"

namespace llvm::orc {

DynamicLoader::DynamicLoader(const DynamicLoader::Setup &setup)
    : m_cache(setup.cache ? setup.cache : std::make_shared<LibraryPathCache>()),
      m_PathResolver(setup.resolver ? setup.resolver
                                    : std::make_shared<PathResolver>(m_cache)),
      // m_DylibPathResolver(setup.dylibResolver),
      m_scanH(setup.basePaths, m_cache, m_PathResolver),
      FB(setup.filterBuilder), m_libMgr(),
      m_shouldScan(setup.shouldScan ? setup.shouldScan
                                    : [](StringRef) { return true; }),
      includeSys(setup.includeSys) {

  if (m_scanH.getAllUnits().empty()) {
    errs() << "Warning: No base paths provided for scanning.\n";
  }
}

static bool shouldIgnoreSymbol(const object::SymbolRef &Sym,
                               uint32_t IgnoreFlags) {
  Expected<uint32_t> FlagsOrErr = Sym.getFlags();
  if (!FlagsOrErr) {
    consumeError(FlagsOrErr.takeError());
    return true;
  }

  uint32_t Flags = *FlagsOrErr;
  using Filter = SymbolEnumerator::Filter;
  if ((IgnoreFlags & static_cast<uint32_t>(Filter::IgnoreUndefined)) &&
      (Flags & object::SymbolRef::SF_Undefined))
    return true;
  if ((IgnoreFlags & static_cast<uint32_t>(Filter::IgnoreIndirect)) &&
      (Flags & object::SymbolRef::SF_Indirect))
    return true;
  if ((IgnoreFlags & static_cast<uint32_t>(Filter::IgnoreWeak)) &&
      (Flags & object::SymbolRef::SF_Weak))
    return true;

  return false;
}

bool SymbolEnumerator::enumerateSymbols(StringRef Path, OnEachSymbolFn OnEach,
                                        const Options &Opts) {
  if (Path.empty())
    return false;

  auto ObjOrErr = object::ObjectFile::createObjectFile(Path);
  if (!ObjOrErr) {
    handleAllErrors(ObjOrErr.takeError(), [&](const ErrorInfoBase &EIB) {
      errs() << "Error loading object: " << EIB.message() << "\n";
    });
    return false;
  }

  object::ObjectFile *Obj = ObjOrErr.get().getBinary();

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

void DynamicLoader::resolveSymbolsInLibrary(LibraryInfo &lib,
                                            SymbolQuery &unresolvedSymbols) {
  std::unordered_set<std::string> discoveredSymbols;
  bool hasEnumerated = false;

  auto enumerateSymbolsIfNeeded = [&]() {
    if (hasEnumerated)
      return;

    hasEnumerated = true;

    SymbolEnumerator::Options opts;
    opts.FilterFlags =
        static_cast<uint32_t>(SymbolEnumerator::Filter::IgnoreUndefined) |
        static_cast<uint32_t>(SymbolEnumerator::Filter::IgnoreWeak) |
        static_cast<uint32_t>(SymbolEnumerator::Filter::IgnoreIndirect);
    // LLVM_DEBUG(
    dbgs() << "Enumerating symbols in library: " << lib.getFullPath() << "\n";
    // );
    SymbolEnumerator::enumerateSymbols(
        lib.getFullPath(),
        [&](StringRef sym) {
          discoveredSymbols.insert(sym.str());
          return SymbolEnumerator::Result::Continue;
        },
        opts);
  };

  if (!unresolvedSymbols.hasUnresolved()) {
    // LLVM_DEBUG(
    dbgs() << "Skipping library: " << lib.getFullPath()
           << " — unresolved symbols exist.\n";
    // );
    return;
  }
  enumerateSymbolsIfNeeded();
  if (!lib.hasFilter()) {
    // LLVM_DEBUG(
    dbgs() << "Building filter for library: " << lib.getFullPath() << "\n"; //);
    lib.ensureFilterBuilt(FB,
                          {discoveredSymbols.begin(), discoveredSymbols.end()});
    dbgs() << "discoveredSymbols : " << discoveredSymbols.size() << "\n";
    for (const auto &sym : discoveredSymbols)
      dbgs() << "discoveredSymbols : " << sym << "\n";
  }

  const auto &unresolved = unresolvedSymbols.getUnresolvedSymbols();
  for (const auto &symbol : unresolved) {
    if (lib.mayContain(symbol)) {
      // LLVM_DEBUG(
      dbgs() << "Checking symbol '" << symbol
             << "' in library: " << lib.getFullPath() << "\n"; //);
      if (discoveredSymbols.count(symbol.str()) > 0) {
        // LLVM_DEBUG(
        dbgs() << "  Resolved symbol: " << symbol
               << " in library: " << lib.getFullPath() << "\n"; //);
        unresolvedSymbols.resolve(symbol, lib.getFullPath());
      }
    }
  }
}

void DynamicLoader::searchSymbolsInLibraries(
    std::vector<std::string> &symbolList, OnSearchComplete onComplete) {
  SymbolQuery query(symbolList);

  using LibraryState = LibraryManager::State;
  using LibraryType = LibraryManager::Kind;
  auto tryResolveFrom = [&](LibraryState S, LibraryType K) {
    if (query.allResolved())
      return;
    // LLVM_DEBUG(
    dbgs() << "Trying resolve from state=" << static_cast<int>(S)
           << " type=" << static_cast<int>(K) << "\n"; //);
    scanLibrariesIfNeeded(K);
    for (auto &lib : m_libMgr.getView(S, K)) {
      // can use Async here?
      resolveSymbolsInLibrary(*lib, query);
      if (query.allResolved())
        break;
    }
  };

  static constexpr LibraryState kStates[] = {
      LibraryState::Loaded, LibraryState::Queried, LibraryState::Unloaded};

  static constexpr LibraryType kTypes[] = {LibraryType::User,
                                           LibraryType::System};

  for (auto type : kTypes) {
    for (auto state : kStates) {
      tryResolveFrom(state, type);
      if (query.allResolved())
        goto done;
    }
  }

done:
  // LLVM_DEBUG({
  dbgs() << "Search complete.\n";
  for (const auto &r : query.getAllResults())
    dbgs() << "Resolved Symbol:" << r->Name << " -> " << r->ResolvedLibPath
           << "\n";
  //});

  // ProcessLib(query.getResolvedPath());
  onComplete(query);
}

void DynamicLoader::scanLibrariesIfNeeded(LibraryManager::Kind PK) {
  // LLVM_DEBUG(
  dbgs() << "DynamicLoader::scanLibrariesIfNeeded: Scanning for "
         << (PK == LibraryManager::Kind::User ? "User" : "System")
         << " libraries\n"; //);
  LibraryScanner Scanner(m_scanH, m_libMgr, m_shouldScan);
  Scanner.scanNext(PK == LibraryManager::Kind::User ? PathKind::User
                                                    : PathKind::System);
}

bool DynamicLoader::symbolExistsInLibrary(
    const LibraryInfo &lib, StringRef symbolName,
    std::vector<std::string> *allSymbols) {

  SymbolEnumerator::Options opts;
  return symbolExistsInLibrary(lib, symbolName, allSymbols, opts);
}

bool DynamicLoader::symbolExistsInLibrary(
    const LibraryInfo &lib, StringRef symbolName,
    std::vector<std::string> *allSymbols,
    const SymbolEnumerator::Options &opts) {

  bool found = false;

  SymbolEnumerator::enumerateSymbols(
      lib.getFullPath(),
      [&](StringRef sym) {
        if (allSymbols)
          allSymbols->emplace_back(sym.str());

        if (sym == symbolName) {
          found = true;
          // return SymbolEnumerator::Result::Stop;
        }

        return SymbolEnumerator::Result::Continue;
      },
      opts);

  return found;
}

} // end namespace llvm::orc
