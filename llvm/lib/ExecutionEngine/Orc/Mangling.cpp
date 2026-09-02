//===----------- Mangling.cpp -- Name Mangling Utilities for ORC ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, StringRef ABIName)
    : ES(ES), Mode(fromTriple(ES.getTargetTriple(), ABIName)) {}

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, ManglingMode Mode)
    : ES(ES), Mode(Mode) {}

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, const DataLayout &DL)
    : ES(ES), Mode(fromDataLayout(DL)) {}

// TODO: The prefixing rules below, and the mangling-mode derivation in
// fromDataLayoutStr, duplicate logic that already lives in llvm::Mangler
// (getNameWithPrefix) and DataLayout (ManglingModeT and its "m:" spec
// parsing). They are re-implemented here only because those APIs require a
// full DataLayout, which this class is meant to work without. We should
// refactor to have one copy of this code, probably best defined in
// TargetParser, shared between all users.
SymbolStringPtr MangleAndInterner::operator()(StringRef Name) {
  if (Name.empty())
    return ES.intern(Name);

  if (Name.front() == '\1')
    return ES.intern(Name.substr(1));

  if (Name[0] == '?' && doNotMangleLeadingQuestionMark())
    return ES.intern(Name);

  if (Mode == ManglingMode::MachO || Mode == ManglingMode::WinCOFFX86)
    return ES.intern(("_" + Name).str());

  return ES.intern(Name);
}

MangleAndInterner::ManglingMode
MangleAndInterner::fromDataLayoutStr(StringRef DLStr) {
  for (StringRef Spec : split(DLStr, '-')) {
    if (!Spec.starts_with("m:"))
      continue;
    auto ModeStr = Spec.drop_front(2);
    assert(ModeStr.size() == 1 &&
           "invalid data layout string from Triple::computeDataLayout");
    switch (ModeStr[0]) {
    case 'e':
      return ManglingMode::ELF;
    case 'l':
      return ManglingMode::GOFF;
    case 'o':
      return ManglingMode::MachO;
    case 'm':
      return ManglingMode::Mips;
    case 'w':
      return ManglingMode::WinCOFF;
    case 'x':
      return ManglingMode::WinCOFFX86;
    case 'a':
      return ManglingMode::XCOFF;
    default:
      llvm_unreachable("Invalid mangling mode from Triple::computeDataLayout");
    }
  }
  return ManglingMode::None;
}

MangleAndInterner::ManglingMode
MangleAndInterner::fromTriple(const Triple &TT, StringRef ABIName) {
  return fromDataLayoutStr(TT.computeDataLayout(ABIName));
}

MangleAndInterner::ManglingMode
MangleAndInterner::fromDataLayout(const DataLayout &DL) {
  return fromDataLayoutStr(DL.getStringRepresentation());
}

bool MangleAndInterner::doNotMangleLeadingQuestionMark() const {
  return Mode == ManglingMode::WinCOFF || Mode == ManglingMode::WinCOFFX86;
}

void IRSymbolMapper::add(ExecutionSession &ES, const ManglingOptions &MO,
                         ArrayRef<GlobalValue *> GVs,
                         SymbolFlagsMap &SymbolFlags,
                         SymbolNameToDefinitionMap *SymbolToDefinition) {
  if (GVs.empty())
    return;

  MangleAndInterner Mangle(ES, GVs[0]->getDataLayout());
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

} // namespace llvm::orc
