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
private:
  bool IsFunction = false;
  std::string ModuleName;
  SmallVector<unsigned, 1> Returns;
  SmallVector<unsigned, 4> Params;

  /// An expression describing how to calculate the size of a symbol. If a
  /// symbol has no size this field will be NULL.
  const MCExpr *SymbolSize = nullptr;

public:
  // Use a module name of "env" for now, for compatibility with existing tools.
  // This is temporary, and may change, as the ABI is not yet stable.
  MCSymbolWasm(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindWasm, Name, isTemporary),
        ModuleName("env") {}
  static bool classof(const MCSymbol *S) { return S->isWasm(); }

  const MCExpr *getSize() const { return SymbolSize; }
  void setSize(const MCExpr *SS) { SymbolSize = SS; }

  bool isFunction() const { return IsFunction; }
  void setIsFunction(bool isFunc) { IsFunction = isFunc; }

  const StringRef getModuleName() const { return ModuleName; }

  const SmallVector<unsigned, 1> &getReturns() const { return Returns; }
  void setReturns(SmallVectorImpl<unsigned> &&Rets) {
    Returns = std::move(Rets);
  }

  const SmallVector<unsigned, 4> &getParams() const { return Params; }
  void setParams(SmallVectorImpl<unsigned> &&Pars) {
    Params = std::move(Pars);
  }
};
}

#endif
