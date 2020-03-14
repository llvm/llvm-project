//===----------- Mangling.cpp -- Name Mangling Utilities for ORC ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, const DataLayout &DL)
    : ES(ES), DL(DL) {}

SymbolStringPtr MangleAndInterner::operator()(StringRef Name) {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
  }
  return ES.intern(MangledName);
}

void IRSymbolMapper::add(ExecutionSession &ES, const ManglingOptions &MO,
                         ArrayRef<GlobalValue *> GVs,
                         SymbolFlagsMap &SymbolFlags,
                         SymbolNameToDefinitionMap *SymbolToDefinition) {
  if (GVs.empty())
    return;

  MangleAndInterner Mangle(ES, GVs[0]->getParent()->getDataLayout());
  for (auto *G : GVs) {
    assert(G && "GVs cannot contain null elements");
    if (!G->hasName() || G->isDeclaration() || G->hasLocalLinkage() ||
        G->hasAvailableExternallyLinkage() || G->hasAppendingLinkage())
      continue;

    if (G->isThreadLocal() && MO.EmulatedTLS) {
      auto *GV = cast<GlobalVariable>(G);

      auto Flags = JITSymbolFlags::fromGlobalValue(*GV);

      auto EmuTLSV = Mangle(("__emutls_v." + GV->getName()).str());
      SymbolFlags[EmuTLSV] = Flags;
      if (SymbolToDefinition)
        (*SymbolToDefinition)[EmuTLSV] = GV;

      // If this GV has a non-zero initializer we'll need to emit an
      // __emutls.t symbol too.
      if (GV->hasInitializer()) {
        const auto *InitVal = GV->getInitializer();

        // Skip zero-initializers.
        if (isa<ConstantAggregateZero>(InitVal))
          continue;
        const auto *InitIntValue = dyn_cast<ConstantInt>(InitVal);
        if (InitIntValue && InitIntValue->isZero())
          continue;

        auto EmuTLST = Mangle(("__emutls_t." + GV->getName()).str());
        SymbolFlags[EmuTLST] = Flags;
        if (SymbolToDefinition)
          (*SymbolToDefinition)[EmuTLST] = GV;
      }
      continue;
    }

    // Otherwise we just need a normal linker mangling.
    auto MangledName = Mangle(G->getName());
    SymbolFlags[MangledName] = JITSymbolFlags::fromGlobalValue(*G);
    if (SymbolToDefinition)
      (*SymbolToDefinition)[MangledName] = G;
  }
}

Expected<std::pair<SymbolFlagsMap, SymbolStringPtr>>
getObjectSymbolInfo(ExecutionSession &ES, MemoryBufferRef ObjBuffer) {
  auto Obj = object::ObjectFile::createObjectFile(ObjBuffer);

  if (!Obj)
    return Obj.takeError();

  bool IsMachO = isa<object::MachOObjectFile>(Obj->get());

  SymbolFlagsMap SymbolFlags;
  for (auto &Sym : (*Obj)->symbols()) {
    // Skip symbols not defined in this object file.
    if (Sym.getFlags() & object::BasicSymbolRef::SF_Undefined)
      continue;

    // Get the symbol name.
    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();

    bool IsLinkerPrivate = IsMachO && Name->startswith("l");

    // Skip symbols that are not global. Treat linker private symbols as global
    // hidden.
    if (!(Sym.getFlags() & object::BasicSymbolRef::SF_Global) &&
        !IsLinkerPrivate)
      continue;

    // Skip symbols that have type SF_File.
    if (auto SymType = Sym.getType()) {
      if (*SymType == object::SymbolRef::ST_File)
        continue;
    } else
      return SymType.takeError();

    auto InternedName = ES.intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();
    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  SymbolStringPtr InitSymbol;

  if (IsMachO) {
    auto &MachOObj = cast<object::MachOObjectFile>(*Obj->get());
    for (auto &Sec : MachOObj.sections()) {
      auto SecType = MachOObj.getSectionType(Sec);
      if ((SecType & MachO::SECTION_TYPE) == MachO::S_MOD_INIT_FUNC_POINTERS) {
        std::string InitSymString;
        raw_string_ostream(InitSymString)
            << "$." << ObjBuffer.getBufferIdentifier() << ".__inits";
        InitSymbol = ES.intern(InitSymString);
        SymbolFlags[InitSymbol] = JITSymbolFlags();
        break;
      }
    }
  }

  return std::make_pair(std::move(SymbolFlags), std::move(InitSymbol));
}

} // End namespace orc.
} // End namespace llvm.
