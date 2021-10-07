//===- llvm/GlobalIndirectSymbol.h - GlobalIndirectSymbol class -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalIndirectSymbol class, which
// is a base class for GlobalAlias and GlobalIFunc. It contains all common code
// for aliases and ifuncs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALINDIRECTSYMBOL_H
#define LLVM_IR_GLOBALINDIRECTSYMBOL_H

#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <cstddef>

namespace llvm {

class GlobalIndirectSymbol : public GlobalValue {
protected:
  GlobalIndirectSymbol(Type *Ty, ValueTy VTy, unsigned AddressSpace,
      LinkageTypes Linkage, const Twine &Name, Constant *Symbol);

public:
  GlobalIndirectSymbol(const GlobalIndirectSymbol &) = delete;
  GlobalIndirectSymbol &operator=(const GlobalIndirectSymbol &) = delete;

  // allocate space for exactly one operand
  void *operator new(size_t S) { return User::operator new(S, 1); }
  void operator delete(void *Ptr) { User::operator delete(Ptr); }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  void copyAttributesFrom(const GlobalValue *Src) {
    GlobalValue::copyAttributesFrom(Src);
  }

  /// These methods set and retrieve indirect symbol.
  void setIndirectSymbol(Constant *Symbol) {
    setOperand(0, Symbol);
  }
  const Constant *getIndirectSymbol() const {
    return getOperand(0);
  }
  Constant *getIndirectSymbol() {
    return const_cast<Constant *>(
          static_cast<const GlobalIndirectSymbol *>(this)->getIndirectSymbol());
  }

  const GlobalObject *getAliaseeObject() const;
  GlobalObject *getAliaseeObject() {
    return const_cast<GlobalObject *>(
        static_cast<const GlobalIndirectSymbol *>(this)->getAliaseeObject());
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalAliasVal ||
           V->getValueID() == Value::GlobalIFuncVal;
  }
};

template <>
struct OperandTraits<GlobalIndirectSymbol> :
  public FixedNumOperandTraits<GlobalIndirectSymbol, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GlobalIndirectSymbol, Constant)

} // end namespace llvm

#endif // LLVM_IR_GLOBALINDIRECTSYMBOL_H
