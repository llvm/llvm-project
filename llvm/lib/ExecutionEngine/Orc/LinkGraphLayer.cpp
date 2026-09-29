//===----- LinkGraphLayer.cpp - Add LinkGraphs to an ExecutionSession -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LinkGraphLayer.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/Orc/Shared/MachOObjectFormat.h"
#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace {

bool hasInitializerSection(LinkGraph &G) {
  bool IsMachO = G.getTargetTriple().isOSBinFormatMachO();
  bool IsElf = G.getTargetTriple().isOSBinFormatELF();
  if (!IsMachO && !IsElf)
    return false;

  for (auto &Sec : G.sections()) {
    if (IsMachO && isMachOInitializerSection(Sec.getName()))
      return true;
    if (IsElf && isELFInitializerSection(Sec.getName()))
      return true;
  }

  return false;
}

} // end anonymous namespace

namespace llvm::orc {

LinkGraphLayer::~LinkGraphLayer() = default;

MaterializationUnit::Interface LinkGraphLayer::getInterface(LinkGraph &G) {

  MaterializationUnit::Interface LGI;

  auto AddSymbol = [&](Symbol *Sym) {
    // Skip local symbols.
    if (Sym->getScope() == Scope::Local)
      return;
    assert(Sym->hasName() && "Anonymous non-local symbol?");

    LGI.SymbolFlags[Sym->getName()] = getJITSymbolFlagsForSymbol(*Sym);
  };

  for (auto *Sym : G.defined_symbols())
    AddSymbol(Sym);
  for (auto *Sym : G.absolute_symbols())
    AddSymbol(Sym);

  if (hasInitializerSection(G)) {
    std::string InitSymString;
    {
      raw_string_ostream(InitSymString)
          << "$." << G.getName() << ".__inits" << Counter++;
    }
    LGI.InitSymbol = ES.intern(InitSymString);
  }

  return LGI;
}

JITSymbolFlags LinkGraphLayer::getJITSymbolFlagsForSymbol(Symbol &Sym) {
  JITSymbolFlags Flags;

  if (Sym.getLinkage() == Linkage::Weak)
    Flags |= JITSymbolFlags::Weak;

  if (Sym.getScope() == Scope::Default)
    Flags |= JITSymbolFlags::Exported;
  else if (Sym.getScope() == Scope::SideEffectsOnly)
    Flags |= JITSymbolFlags::MaterializationSideEffectsOnly;

  if (Sym.isCallable())
    Flags |= JITSymbolFlags::Callable;

  return Flags;
}

StringRef LinkGraphMaterializationUnit::getName() const { return G->getName(); }

void LinkGraphMaterializationUnit::discard(const JITDylib &JD,
                                           const SymbolStringPtr &Name) {
  for (auto *Sym : G->defined_symbols())
    if (Sym->getName() == Name) {
      assert(Sym->getLinkage() == Linkage::Weak &&
             "Discarding non-weak definition");
      G->makeExternal(*Sym);
      break;
    }
}

} // namespace llvm::orc
