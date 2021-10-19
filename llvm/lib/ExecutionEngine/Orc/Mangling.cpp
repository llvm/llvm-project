//===----------- Mangling.cpp -- Name Mangling Utilities for ORC ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Object/ELFObjectFile.h"
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

static SymbolStringPtr addInitSymbol(SymbolFlagsMap &SymbolFlags,
                                     ExecutionSession &ES,
                                     StringRef ObjFileName) {
  SymbolStringPtr InitSymbol;
  size_t Counter = 0;

  do {
    std::string InitSymString;
    raw_string_ostream(InitSymString)
      << "$." << ObjFileName << ".__inits." << Counter++;
    InitSymbol = ES.intern(InitSymString);
  } while (SymbolFlags.count(InitSymbol));

  SymbolFlags[InitSymbol] = JITSymbolFlags::MaterializationSideEffectsOnly;
  return InitSymbol;
}

static Expected<std::pair<SymbolFlagsMap, SymbolStringPtr>>
getMachOObjectFileSymbolInfo(ExecutionSession &ES,
                             const object::MachOObjectFile &Obj) {
  SymbolFlagsMap SymbolFlags;

  for (auto &Sym : Obj.symbols()) {
    Expected<uint32_t> SymFlagsOrErr = Sym.getFlags();
    if (!SymFlagsOrErr)
      // TODO: Test this error.
      return SymFlagsOrErr.takeError();

    // Skip symbols not defined in this object file.
    if (*SymFlagsOrErr & object::BasicSymbolRef::SF_Undefined)
      continue;

    // Skip symbols that are not global.
    if (!(*SymFlagsOrErr & object::BasicSymbolRef::SF_Global))
      continue;

    // Skip symbols that have type SF_File.
    if (auto SymType = Sym.getType()) {
      if (*SymType == object::SymbolRef::ST_File)
        continue;
    } else
      return SymType.takeError();

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();
    auto InternedName = ES.intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();

    // Strip the 'exported' flag from MachO linker-private symbols.
    if (Name->startswith("l"))
      *SymFlags &= ~JITSymbolFlags::Exported;

    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  SymbolStringPtr InitSymbol;
  for (auto &Sec : Obj.sections()) {
    auto SecType = Obj.getSectionType(Sec);
    if ((SecType & MachO::SECTION_TYPE) == MachO::S_MOD_INIT_FUNC_POINTERS) {
      InitSymbol = addInitSymbol(SymbolFlags, ES, Obj.getFileName());
      break;
    }
    auto SegName = Obj.getSectionFinalSegmentName(Sec.getRawDataRefImpl());
    auto SecName = cantFail(Obj.getSectionName(Sec.getRawDataRefImpl()));
    if (MachOPlatform::isInitializerSection(SegName, SecName)) {
      InitSymbol = addInitSymbol(SymbolFlags, ES, Obj.getFileName());
      break;
    }
  }

  return std::make_pair(std::move(SymbolFlags), std::move(InitSymbol));
}

static Expected<std::pair<SymbolFlagsMap, SymbolStringPtr>>
getELFObjectFileSymbolInfo(ExecutionSession &ES,
                           const object::ELFObjectFileBase &Obj) {
  SymbolFlagsMap SymbolFlags;
  for (auto &Sym : Obj.symbols()) {
    Expected<uint32_t> SymFlagsOrErr = Sym.getFlags();
    if (!SymFlagsOrErr)
      // TODO: Test this error.
      return SymFlagsOrErr.takeError();

    // Skip symbols not defined in this object file.
    if (*SymFlagsOrErr & object::BasicSymbolRef::SF_Undefined)
      continue;

    // Skip symbols that are not global.
    if (!(*SymFlagsOrErr & object::BasicSymbolRef::SF_Global))
      continue;

    // Skip symbols that have type SF_File.
    if (auto SymType = Sym.getType()) {
      if (*SymType == object::SymbolRef::ST_File)
        continue;
    } else
      return SymType.takeError();

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();
    auto InternedName = ES.intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();

    // ELF STB_GNU_UNIQUE should map to Weak for ORC.
    if (Sym.getBinding() == ELF::STB_GNU_UNIQUE)
      *SymFlags |= JITSymbolFlags::Weak;

    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  SymbolStringPtr InitSymbol;
  for (auto &Sec : Obj.sections()) {
    if (auto SecName = Sec.getName()) {
      if (ELFNixPlatform::isInitializerSection(*SecName)) {
        InitSymbol = addInitSymbol(SymbolFlags, ES, Obj.getFileName());
        break;
      }
    }
  }

  return std::make_pair(std::move(SymbolFlags), InitSymbol);
}

Expected<std::pair<SymbolFlagsMap, SymbolStringPtr>>
getGenericObjectFileSymbolInfo(ExecutionSession &ES,
                               const object::ObjectFile &Obj) {
  SymbolFlagsMap SymbolFlags;
  for (auto &Sym : Obj.symbols()) {
    Expected<uint32_t> SymFlagsOrErr = Sym.getFlags();
    if (!SymFlagsOrErr)
      // TODO: Test this error.
      return SymFlagsOrErr.takeError();

    // Skip symbols not defined in this object file.
    if (*SymFlagsOrErr & object::BasicSymbolRef::SF_Undefined)
      continue;

    // Skip symbols that are not global.
    if (!(*SymFlagsOrErr & object::BasicSymbolRef::SF_Global))
      continue;

    // Skip symbols that have type SF_File.
    if (auto SymType = Sym.getType()) {
      if (*SymType == object::SymbolRef::ST_File)
        continue;
    } else
      return SymType.takeError();

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();
    auto InternedName = ES.intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();

    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  return std::make_pair(std::move(SymbolFlags), nullptr);
}

Expected<std::pair<SymbolFlagsMap, SymbolStringPtr>>
getObjectSymbolInfo(ExecutionSession &ES, MemoryBufferRef ObjBuffer) {
  auto Obj = object::ObjectFile::createObjectFile(ObjBuffer);

  if (!Obj)
    return Obj.takeError();

  if (auto *MachOObj = dyn_cast<object::MachOObjectFile>(Obj->get()))
    return getMachOObjectFileSymbolInfo(ES, *MachOObj);
  else if (auto *ELFObj = dyn_cast<object::ELFObjectFileBase>(Obj->get()))
    return getELFObjectFileSymbolInfo(ES, *ELFObj);

  return getGenericObjectFileSymbolInfo(ES, **Obj);
}

} // End namespace orc.
} // End namespace llvm.
