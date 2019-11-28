//===- MCSymbolXCOFF.h -  ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_MC_MCSYMBOLXCOFF_H
#define LLVM_MC_MCSYMBOLXCOFF_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {

class MCSectionXCOFF;

class MCSymbolXCOFF : public MCSymbol {
public:
  MCSymbolXCOFF(const StringMapEntry<bool> *Name, bool isTemporary)
      : MCSymbol(SymbolKindXCOFF, Name, isTemporary) {}

  static bool classof(const MCSymbol *S) { return S->isXCOFF(); }

  void setStorageClass(XCOFF::StorageClass SC) {
    assert((!StorageClass.hasValue() || StorageClass.getValue() == SC) &&
           "Redefining StorageClass of XCOFF MCSymbol.");
    StorageClass = SC;
  };

  XCOFF::StorageClass getStorageClass() const {
    assert(StorageClass.hasValue() &&
           "StorageClass not set on XCOFF MCSymbol.");
    return StorageClass.getValue();
  }

  void setContainingCsect(MCSectionXCOFF *C) {
    assert((!ContainingCsect || ContainingCsect == C) &&
           "Trying to set a containing csect that doesn't match the one that"
           "this symbol is already mapped to.");
    ContainingCsect = C;
  }

  MCSectionXCOFF *getContainingCsect() const {
    assert(ContainingCsect &&
           "Trying to get containing csect but none was set.");
    return ContainingCsect;
  }

  bool hasContainingCsect() const { return ContainingCsect != nullptr; }

  StringRef getUnqualifiedName() const {
    const StringRef name = getName();
    if (name.back() == ']') {
      StringRef lhs, rhs;
      std::tie(lhs, rhs) = name.rsplit('[');
      assert(!rhs.empty() && "Invalid SMC format in XCOFF symbol.");
      return lhs;
    }
    return name;
  }

private:
  Optional<XCOFF::StorageClass> StorageClass;
  MCSectionXCOFF *ContainingCsect = nullptr;
};

} // end namespace llvm

#endif // LLVM_MC_MCSYMBOLXCOFF_H
