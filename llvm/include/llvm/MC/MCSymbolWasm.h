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

#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class MCSymbolWasm : public MCSymbol {
  wasm::WasmSymbolType Type = wasm::WASM_SYMBOL_TYPE_DATA;
  bool IsWeak = false;
  bool IsHidden = false;
  bool IsComdat = false;
  std::string ModuleName;
  SmallVector<wasm::ValType, 1> Returns;
  SmallVector<wasm::ValType, 4> Params;
  wasm::WasmGlobalType GlobalType;
  bool ParamsSet = false;
  bool ReturnsSet = false;
  bool GlobalTypeSet = false;

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

  bool isFunction() const { return Type == wasm::WASM_SYMBOL_TYPE_FUNCTION; }
  bool isData() const { return Type == wasm::WASM_SYMBOL_TYPE_DATA; }
  bool isGlobal() const { return Type == wasm::WASM_SYMBOL_TYPE_GLOBAL; }
  bool isSection() const { return Type == wasm::WASM_SYMBOL_TYPE_SECTION; }
  wasm::WasmSymbolType getType() const { return Type; }
  void setType(wasm::WasmSymbolType type) { Type = type; }

  bool isWeak() const { return IsWeak; }
  void setWeak(bool isWeak) { IsWeak = isWeak; }

  bool isHidden() const { return IsHidden; }
  void setHidden(bool isHidden) { IsHidden = isHidden; }

  bool isComdat() const { return IsComdat; }
  void setComdat(bool isComdat) { IsComdat = isComdat; }

  const StringRef getModuleName() const { return ModuleName; }
  void setModuleName(StringRef Name) { ModuleName = Name; }

  const SmallVector<wasm::ValType, 1> &getReturns() const {
    assert(ReturnsSet);
    return Returns;
  }

  void setReturns(SmallVectorImpl<wasm::ValType> &&Rets) {
    ReturnsSet = true;
    Returns = std::move(Rets);
  }

  const SmallVector<wasm::ValType, 4> &getParams() const {
    assert(ParamsSet);
    return Params;
  }

  void setParams(SmallVectorImpl<wasm::ValType> &&Pars) {
    ParamsSet = true;
    Params = std::move(Pars);
  }

  const wasm::WasmGlobalType &getGlobalType() const {
    assert(GlobalTypeSet);
    return GlobalType;
  }

  void setGlobalType(wasm::WasmGlobalType GT) {
    GlobalTypeSet = true;
    GlobalType = GT;
  }
};

}  // end namespace llvm

#endif // LLVM_MC_MCSYMBOLWASM_H
