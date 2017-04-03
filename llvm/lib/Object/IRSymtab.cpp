//===- IRSymtab.cpp - implementation of IR symbol tables --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/IRSymtab.h"
#include "llvm/Analysis/ObjectUtils.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace irsymtab;

namespace {

/// Stores the temporary state that is required to build an IR symbol table.
struct Builder {
  SmallVector<char, 0> &Symtab;
  SmallVector<char, 0> &Strtab;
  Builder(SmallVector<char, 0> &Symtab, SmallVector<char, 0> &Strtab)
      : Symtab(Symtab), Strtab(Strtab) {}

  StringTableBuilder StrtabBuilder{StringTableBuilder::ELF};

  BumpPtrAllocator Alloc;
  StringSaver Saver{Alloc};

  DenseMap<const Comdat *, unsigned> ComdatMap;
  ModuleSymbolTable Msymtab;
  SmallPtrSet<GlobalValue *, 8> Used;
  Mangler Mang;
  Triple TT;

  std::vector<storage::Comdat> Comdats;
  std::vector<storage::Module> Mods;
  std::vector<storage::Symbol> Syms;
  std::vector<storage::Uncommon> Uncommons;

  std::string COFFLinkerOpts;
  raw_string_ostream COFFLinkerOptsOS{COFFLinkerOpts};

  void setStr(storage::Str &S, StringRef Value) {
    S.Offset = StrtabBuilder.add(Value);
  }
  template <typename T>
  void writeRange(storage::Range<T> &R, const std::vector<T> &Objs) {
    R.Offset = Symtab.size();
    R.Size = Objs.size();
    Symtab.insert(Symtab.end(), reinterpret_cast<const char *>(Objs.data()),
                  reinterpret_cast<const char *>(Objs.data() + Objs.size()));
  }

  Error addModule(Module *M);
  Error addSymbol(ModuleSymbolTable::Symbol Sym);

  Error build(ArrayRef<Module *> Mods);
};

Error Builder::addModule(Module *M) {
  collectUsedGlobalVariables(*M, Used, /*CompilerUsed*/ false);

  storage::Module Mod;
  Mod.Begin = Msymtab.symbols().size();
  Msymtab.addModule(M);
  Mod.End = Msymtab.symbols().size();
  Mods.push_back(Mod);

  if (TT.isOSBinFormatCOFF()) {
    if (auto E = M->materializeMetadata())
      return E;
    if (Metadata *Val = M->getModuleFlag("Linker Options")) {
      MDNode *LinkerOptions = cast<MDNode>(Val);
      for (const MDOperand &MDOptions : LinkerOptions->operands())
        for (const MDOperand &MDOption : cast<MDNode>(MDOptions)->operands())
          COFFLinkerOptsOS << " " << cast<MDString>(MDOption)->getString();
    }
  }

  return Error::success();
}

Error Builder::addSymbol(ModuleSymbolTable::Symbol Msym) {
  Syms.emplace_back();
  storage::Symbol &Sym = Syms.back();
  Sym = {};

  Sym.UncommonIndex = -1;
  storage::Uncommon *Unc = nullptr;
  auto Uncommon = [&]() -> storage::Uncommon & {
    if (Unc)
      return *Unc;
    Sym.UncommonIndex = Uncommons.size();
    Uncommons.emplace_back();
    Unc = &Uncommons.back();
    *Unc = {};
    setStr(Unc->COFFWeakExternFallbackName, "");
    return *Unc;
  };

  SmallString<64> Name;
  {
    raw_svector_ostream OS(Name);
    Msymtab.printSymbolName(OS, Msym);
  }
  setStr(Sym.Name, Saver.save(StringRef(Name)));

  auto Flags = Msymtab.getSymbolFlags(Msym);
  if (Flags & object::BasicSymbolRef::SF_Undefined)
    Sym.Flags |= 1 << storage::Symbol::FB_undefined;
  if (Flags & object::BasicSymbolRef::SF_Weak)
    Sym.Flags |= 1 << storage::Symbol::FB_weak;
  if (Flags & object::BasicSymbolRef::SF_Common)
    Sym.Flags |= 1 << storage::Symbol::FB_common;
  if (Flags & object::BasicSymbolRef::SF_Indirect)
    Sym.Flags |= 1 << storage::Symbol::FB_indirect;
  if (Flags & object::BasicSymbolRef::SF_Global)
    Sym.Flags |= 1 << storage::Symbol::FB_global;
  if (Flags & object::BasicSymbolRef::SF_FormatSpecific)
    Sym.Flags |= 1 << storage::Symbol::FB_format_specific;

  Sym.ComdatIndex = -1;
  auto *GV = Msym.dyn_cast<GlobalValue *>();
  if (!GV) {
    setStr(Sym.IRName, "");
    return Error::success();
  }

  setStr(Sym.IRName, GV->getName());

  if (Used.count(GV))
    Sym.Flags |= 1 << storage::Symbol::FB_used;
  if (GV->isThreadLocal())
    Sym.Flags |= 1 << storage::Symbol::FB_tls;
  if (GV->hasGlobalUnnamedAddr())
    Sym.Flags |= 1 << storage::Symbol::FB_unnamed_addr;
  if (canBeOmittedFromSymbolTable(GV))
    Sym.Flags |= 1 << storage::Symbol::FB_may_omit;
  Sym.Flags |= unsigned(GV->getVisibility()) << storage::Symbol::FB_visibility;

  if (Flags & object::BasicSymbolRef::SF_Common) {
    Uncommon().CommonSize = GV->getParent()->getDataLayout().getTypeAllocSize(
        GV->getType()->getElementType());
    Uncommon().CommonAlign = GV->getAlignment();
  }

  const GlobalObject *Base = GV->getBaseObject();
  if (!Base)
    return make_error<StringError>("Unable to determine comdat of alias!",
                                   inconvertibleErrorCode());
  if (const Comdat *C = Base->getComdat()) {
    auto P = ComdatMap.insert(std::make_pair(C, Comdats.size()));
    Sym.ComdatIndex = P.first->second;

    if (P.second) {
      storage::Comdat Comdat;
      setStr(Comdat.Name, C->getName());
      Comdats.push_back(Comdat);
    }
  }

  if (TT.isOSBinFormatCOFF()) {
    emitLinkerFlagsForGlobalCOFF(COFFLinkerOptsOS, GV, TT, Mang);

    if ((Flags & object::BasicSymbolRef::SF_Weak) &&
        (Flags & object::BasicSymbolRef::SF_Indirect)) {
      std::string FallbackName;
      raw_string_ostream OS(FallbackName);
      Msymtab.printSymbolName(
          OS, cast<GlobalValue>(
                  cast<GlobalAlias>(GV)->getAliasee()->stripPointerCasts()));
      OS.flush();
      setStr(Uncommon().COFFWeakExternFallbackName, Saver.save(FallbackName));
    }
  }

  return Error::success();
}

Error Builder::build(ArrayRef<Module *> IRMods) {
  storage::Header Hdr;

  assert(!IRMods.empty());
  setStr(Hdr.SourceFileName, IRMods[0]->getSourceFileName());
  TT = Triple(IRMods[0]->getTargetTriple());

  // This adds the symbols for each module to Msymtab.
  for (auto *M : IRMods)
    if (Error Err = addModule(M))
      return Err;

  for (ModuleSymbolTable::Symbol Msym : Msymtab.symbols())
    if (Error Err = addSymbol(Msym))
      return Err;

  COFFLinkerOptsOS.flush();
  setStr(Hdr.COFFLinkerOpts, COFFLinkerOpts);

  // We are about to fill in the header's range fields, so reserve space for it
  // and copy it in afterwards.
  Symtab.resize(sizeof(storage::Header));
  writeRange(Hdr.Modules, Mods);
  writeRange(Hdr.Comdats, Comdats);
  writeRange(Hdr.Symbols, Syms);
  writeRange(Hdr.Uncommons, Uncommons);

  *reinterpret_cast<storage::Header *>(Symtab.data()) = Hdr;

  raw_svector_ostream OS(Strtab);
  StrtabBuilder.finalizeInOrder();
  StrtabBuilder.write(OS);

  return Error::success();
}

} // anonymous namespace

Error irsymtab::build(ArrayRef<Module *> Mods, SmallVector<char, 0> &Symtab,
                      SmallVector<char, 0> &Strtab) {
  return Builder(Symtab, Strtab).build(Mods);
}
