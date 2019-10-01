//===- MCSymbolXCOFF.h -  ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLXCOFF_H
#define LLVM_MC_MCSYMBOLXCOFF_H

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class GlobalValue;

class MCSymbolXCOFF : public MCSymbol {
  // The IR symbol this MCSymbolXCOFF is based on. It is set on function
  // entry point symbols when they are the callee operand of a direct call
  // SDNode.
  const GlobalValue *GV = nullptr;

public:
  MCSymbolXCOFF(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindXCOFF, Name, isTemporary) {}

  void setGlobalValue(const GlobalValue *G) { GV = G; }
  const GlobalValue *getGlobalValue() const { return GV; }

  static bool classof(const MCSymbol *S) { return S->isXCOFF(); }
};

} // end namespace llvm

#endif // LLVM_MC_MCSYMBOLXCOFF_H
