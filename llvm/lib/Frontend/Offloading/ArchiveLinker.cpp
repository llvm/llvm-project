//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements shared functionality for linking static libraries
// (archives) in offloading tools.
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/ArchiveLinker.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <optional>
#include <string>

using namespace llvm;
using namespace llvm::object;

namespace llvm {
namespace offloading {

Expected<Symbol> Symbol::createFromObject(MemoryBufferRef File,
                                          const SymbolRef &Sym) {
  Symbol Result;
  Result.File = File;

  auto FlagsOrErr = Sym.getFlags();
  if (!FlagsOrErr)
    return FlagsOrErr.takeError();

  if (*FlagsOrErr & SymbolRef::SF_Undefined)
    Result.SymFlags |= Undefined;
  if (*FlagsOrErr & SymbolRef::SF_Weak)
    Result.SymFlags |= Weak;

  return Result;
}

static std::optional<std::string> findFile(StringRef Dir, StringRef Root,
                                           const Twine &Name) {
  SmallString<128> Path;
  if (Dir.starts_with("="))
    sys::path::append(Path, Root, Dir.substr(1), Name);
  else
    sys::path::append(Path, Dir, Name);

  if (sys::fs::exists(Path))
    return static_cast<std::string>(Path);
  return std::nullopt;
}

static std::optional<std::string>
findFromSearchPaths(StringRef Name, StringRef Root,
                    ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (std::optional<std::string> File = findFile(Dir, Root, Name))
      return File;
  return std::nullopt;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo` or `-l:libfoo.a`.
static std::optional<std::string>
searchLibrary(StringRef Input, StringRef Root,
              ArrayRef<StringRef> SearchPaths) {
  if (Input.starts_with(":"))
    return findFromSearchPaths(Input.drop_front(), Root, SearchPaths);
  SmallString<128> LibName;
  ("lib" + Input + ".a").toVector(LibName);
  return findFromSearchPaths(LibName, Root, SearchPaths);
}

static Expected<bool> getSymbolsFromBitcode(MemoryBufferRef Buffer,
                                            StringMap<Symbol> &SymTab,
                                            bool IsLazy) {
  Expected<IRSymtabFile> IRSymtabOrErr = readIRSymtab(Buffer);
  if (!IRSymtabOrErr)
    return IRSymtabOrErr.takeError();
  bool Extracted = !IsLazy;
  StringMap<Symbol> PendingSymbols;
  for (unsigned I = 0; I != IRSymtabOrErr->Mods.size(); ++I) {
    for (const auto &IRSym : IRSymtabOrErr->TheReader.module_symbols(I)) {
      if (IRSym.isFormatSpecific() || !IRSym.isGlobal())
        continue;

      StringMap<Symbol> &Target =
          (IsLazy && !SymTab.count(IRSym.getName())) ? PendingSymbols : SymTab;
      Symbol &OldSym = Target[IRSym.getName()];
      Symbol Sym = Symbol(Buffer, IRSym);
      if (OldSym.File.getBuffer().empty())
        OldSym = Sym;

      bool ResolvesReference =
          !Sym.isUndefined() &&
          (OldSym.isUndefined() || (OldSym.isWeak() && !Sym.isWeak())) &&
          !(OldSym.isWeak() && OldSym.isUndefined() && IsLazy);
      Extracted |= ResolvesReference;

      Sym.UsedInRegularObj = OldSym.UsedInRegularObj;
      if (ResolvesReference)
        OldSym = Sym;
    }
  }
  if (Extracted)
    for (const auto &[Name, Symbol] : PendingSymbols)
      SymTab[Name] = Symbol;
  return Extracted;
}

static Expected<bool> getSymbolsFromObject(ObjectFile &ObjFile,
                                           StringMap<Symbol> &SymTab,
                                           bool IsLazy) {
  bool Extracted = !IsLazy;
  StringMap<Symbol> PendingSymbols;
  for (SymbolRef ObjSym : ObjFile.symbols()) {
    auto NameOrErr = ObjSym.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    StringMap<Symbol> &Target =
        (IsLazy && !SymTab.count(*NameOrErr)) ? PendingSymbols : SymTab;
    Symbol &OldSym = Target[*NameOrErr];

    auto SymOrErr =
        Symbol::createFromObject(ObjFile.getMemoryBufferRef(), ObjSym);
    if (!SymOrErr)
      return SymOrErr.takeError();
    Symbol Sym = *SymOrErr;

    if (OldSym.File.getBuffer().empty())
      OldSym = Sym;

    bool ResolvesReference = OldSym.isUndefined() && !Sym.isUndefined() &&
                             (!OldSym.isWeak() || !IsLazy);
    Extracted |= ResolvesReference;

    if (ResolvesReference)
      OldSym = Sym;
    OldSym.UsedInRegularObj = true;
  }
  if (Extracted)
    for (const auto &[Name, Symbol] : PendingSymbols)
      SymTab[Name] = Symbol;
  return Extracted;
}

static Expected<bool> getSymbols(MemoryBufferRef Buffer,
                                 StringMap<Symbol> &SymTab, bool IsLazy) {
  switch (identify_magic(Buffer.getBuffer())) {
  case file_magic::bitcode: {
    return getSymbolsFromBitcode(Buffer, SymTab, IsLazy);
  }
  case file_magic::elf_relocatable: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Buffer);
    if (!ObjFile)
      return ObjFile.takeError();
    return getSymbolsFromObject(**ObjFile, SymTab, IsLazy);
  }
  default:
    return createStringError("Unsupported file type: '" +
                             Buffer.getBufferIdentifier() + "'");
  }
}

Expected<ResolvedInputs>
resolveArchiveMembers(const Inputs &In,
                      function_ref<bool(MemoryBufferRef)> IsFatBinary) {
  ResolvedInputs Result;
  SmallVector<std::pair<std::unique_ptr<MemoryBuffer>, bool>> InputFiles;

  // Process each input descriptor
  for (const InputDesc &Desc : In.Order) {
    std::optional<std::string> Filename;

    if (Desc.InputKind == InputDesc::Kind::Library) {
      Filename = searchLibrary(Desc.Value, In.Root, In.SearchPaths);
      if (!Filename)
        return createStringError("unable to find library -l%s",
                                 Desc.Value.str().c_str());
      if (sys::fs::is_directory(*Filename))
        return createStringError("'%s': Is a directory", Filename->c_str());
    } else {
      if (!sys::fs::exists(Desc.Value) || sys::fs::is_directory(Desc.Value))
        continue;
      Filename = Desc.Value.str();
    }

    if (!Filename)
      continue;

    auto BufferOrErr =
        errorOrToExpected(MemoryBuffer::getFileOrSTDIN(*Filename));
    if (!BufferOrErr)
      return createFileError(*Filename, BufferOrErr.takeError());

    MemoryBufferRef Buffer = (*BufferOrErr)->getMemBufferRef();
    switch (identify_magic(Buffer.getBuffer())) {
    case file_magic::bitcode:
    case file_magic::elf_relocatable:
      InputFiles.emplace_back(std::move(*BufferOrErr), /*IsLazy=*/false);
      break;
    case file_magic::archive: {
      Expected<std::unique_ptr<object::Archive>> LibFile =
          object::Archive::create(Buffer);
      if (!LibFile)
        return LibFile.takeError();
      Error Err = Error::success();
      for (auto Child : (*LibFile)->children(Err)) {
        auto ChildBufferOrErr = Child.getMemoryBufferRef();
        if (!ChildBufferOrErr)
          return ChildBufferOrErr.takeError();
        // Include archive name in buffer identifier for better diagnostics
        std::string BufferIdentifier =
            (*Filename + "(" + ChildBufferOrErr->getBufferIdentifier() + ")")
                .str();
        std::unique_ptr<MemoryBuffer> ChildBuffer =
            MemoryBuffer::getMemBufferCopy(ChildBufferOrErr->getBuffer(),
                                           BufferIdentifier);
        InputFiles.emplace_back(std::move(ChildBuffer), !Desc.WholeArchive);
      }
      if (Err)
        return Err;
      break;
    }
    default:
      return createStringError("Unsupported file type: '" + *Filename + "'");
    }
  }

  // Seed symbol table with forced undefined symbols
  for (StringRef Sym : In.ForcedUndefs)
    Result.SymTab[Sym] = Symbol(Symbol::Undefined);

  // Fixed-point loop to extract archive members
  bool Extracted = true;
  while (Extracted) {
    Extracted = false;
    for (auto &[Input, IsLazy] : InputFiles) {
      if (!Input)
        continue;

      // Check if this is a fat binary that should be passed through
      if (IsFatBinary && IsFatBinary(*Input)) {
        Result.Buffers.emplace_back(std::move(Input));
        continue;
      }

      // Archive members only extract if they define needed symbols
      Expected<bool> ExtractOrErr = getSymbols(*Input, Result.SymTab, IsLazy);
      if (!ExtractOrErr)
        return ExtractOrErr.takeError();

      Extracted |= *ExtractOrErr;
      if (!*ExtractOrErr)
        continue;

      Result.Buffers.emplace_back(std::move(Input));
    }
  }

  return Result;
}

} // namespace offloading
} // namespace llvm
