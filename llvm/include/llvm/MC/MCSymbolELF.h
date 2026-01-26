//===- MCSymbolELF.h -  -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLELF_H
#define LLVM_MC_MCSYMBOLELF_H

#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolTableEntry.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCSymbolELF : public MCSymbol {
  friend class MCAsmInfoELF;
  /// An expression describing how to calculate the size of a symbol. If a
  /// symbol has no size this field will be NULL.
  const MCExpr *SymbolSize = nullptr;

public:
  MCSymbolELF(const MCSymbolTableEntry *Name, bool isTemporary)
      : MCSymbol(Name, isTemporary) {}
  void setSize(const MCExpr *SS) { SymbolSize = SS; }

  const MCExpr *getSize() const { return SymbolSize; }

  LLVM_ABI void setVisibility(unsigned Visibility);
  LLVM_ABI unsigned getVisibility() const;

  LLVM_ABI void setOther(unsigned Other);
  LLVM_ABI unsigned getOther() const;

  LLVM_ABI void setType(unsigned Type) const;
  LLVM_ABI unsigned getType() const;

  LLVM_ABI void setBinding(unsigned Binding) const;
  LLVM_ABI unsigned getBinding() const;

  LLVM_ABI bool isBindingSet() const;

  LLVM_ABI void setIsWeakref() const;
  LLVM_ABI bool isWeakref() const;

  LLVM_ABI void setIsSignature() const;
  LLVM_ABI bool isSignature() const;

  LLVM_ABI void setMemtag(bool Tagged);
  LLVM_ABI bool isMemtag() const;

private:
  void setIsBindingSet() const;
};
}

#endif
