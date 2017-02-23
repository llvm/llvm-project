//===- MCSymbolWasm.h -  ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLWASM_H
#define LLVM_MC_MCSYMBOLWASM_H

#include "llvm/MC/MCSymbol.h"

namespace llvm {
class MCSymbolWasm : public MCSymbol {
public:
  MCSymbolWasm(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindWasm, Name, isTemporary) {}

  static bool classof(const MCSymbol *S) { return S->isWasm(); }
};
}

#endif
