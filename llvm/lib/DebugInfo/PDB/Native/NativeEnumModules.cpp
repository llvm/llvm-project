//==- NativeEnumModules.cpp - Native Symbol Enumerator impl ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/NativeEnumModules.h"

#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/Native/NativeCompilandSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"

namespace llvm {
namespace pdb {

NativeEnumModules::NativeEnumModules(NativeSession &PDBSession,
                                     ArrayRef<ModuleInfoEx> Modules,
                                     uint32_t Index)
    : Session(PDBSession), Modules(Modules), Index(Index) {}

uint32_t NativeEnumModules::getChildCount() const {
  return static_cast<uint32_t>(Modules.size());
}

std::unique_ptr<PDBSymbol>
NativeEnumModules::getChildAtIndex(uint32_t Index) const {
  if (Index >= Modules.size())
    return nullptr;
  return std::unique_ptr<PDBSymbol>(new PDBSymbolCompiland(Session,
      std::unique_ptr<IPDBRawSymbol>(
          new NativeCompilandSymbol(Session, Modules[Index]))));
}

std::unique_ptr<PDBSymbol> NativeEnumModules::getNext() {
  if (Index >= Modules.size())
    return nullptr;
  return getChildAtIndex(Index++);
}

void NativeEnumModules::reset() { Index = 0; }

NativeEnumModules *NativeEnumModules::clone() const {
  return new NativeEnumModules(Session, Modules, Index);
}

}
}
