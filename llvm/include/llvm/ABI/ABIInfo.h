//===----- ABIInfo.h - ABI information access & encapsulation ----- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ABI information access & encapsulation
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_ABIINFO_H
#define LLVM_ABI_ABIINFO_H

#include "llvm/ABI/Types.h"
#include <cassert>

namespace llvm {
namespace abi {

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific C type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind { Direct, Indirect, Ignore, Expand, CoerceAndExpand, InAlloca };

private:
  Kind TheKind;
  const Type *CoercionType;

  bool InReg : 1;
  bool PaddingInReg : 1;

  unsigned IndirectAlign : 16;
  bool IndirectByVal : 1;

  ABIArgInfo(Kind K = Direct)
      : TheKind(K), CoercionType(nullptr), InReg(false), PaddingInReg(false),
        IndirectAlign(0), IndirectByVal(false) {}

public:
  static ABIArgInfo getDirect(const Type *T = nullptr) {
    ABIArgInfo AI(Direct);
    AI.CoercionType = T;
    return AI;
  }

  static ABIArgInfo getDirectInReg(const Type *T = nullptr) {
    ABIArgInfo AI = getDirect(T);
    AI.InReg = true;
    return AI;
  }

  static ABIArgInfo getIndirect(unsigned Align = 0, bool ByVal = true) {
    ABIArgInfo AI(Indirect);
    AI.IndirectAlign = Align;
    AI.IndirectByVal = ByVal;
    return AI;
  }

  static ABIArgInfo getIndirectInReg(unsigned Align = 0, bool ByVal = true) {
    ABIArgInfo AI = getIndirect(Align, ByVal);
    AI.InReg = true;
    return AI;
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  static ABIArgInfo getExpand() { return ABIArgInfo(Expand); }

  static ABIArgInfo getCoerceAndExpand(const Type *CoercionType) {
    ABIArgInfo AI(CoerceAndExpand);
    AI.CoercionType = CoercionType;
    return AI;
  }

  Kind getKind() const { return TheKind; }

  bool isDirect() const { return TheKind == Direct; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isIgnore() const { return TheKind == Ignore; }
  bool isExpand() const { return TheKind == Expand; }
  bool isCoerceAndExpand() const { return TheKind == CoerceAndExpand; }
  bool isInAlloca() const { return TheKind == InAlloca; }

  bool isInReg() const { return InReg; }
  bool hasPaddingInReg() const { return PaddingInReg; }

  unsigned getIndirectAlign() const {
    assert(isIndirect() && "Only indirect arguments have alignment");
    return IndirectAlign;
  }

  bool getIndirectByVal() const {
    assert(isIndirect() && "Only indirect arguments can be ByVal");
    return IndirectByVal;
  }

  const Type *getCoerceToType() const {
    assert((isDirect() || isCoerceAndExpand()) &&
           "Only Direct and CoerceAndExpand arguments can have coercion types");
    return CoercionType;
  }

  ABIArgInfo &setInReg(bool InReg = true) {
    this->InReg = InReg;
    return *this;
  }

  ABIArgInfo &setPaddingInReg(bool HasPadding = true) {
    this->PaddingInReg = HasPadding;
    return *this;
  }
};

/// Abstract base class for target-specific ABI information.
class ABIInfo {
public:
  virtual ~ABIInfo() = default;

  virtual ABIArgInfo classifyReturnType(const Type *RetTy) const = 0;
  virtual ABIArgInfo classifyArgumentType(const Type *ArgTy) const = 0;

  virtual bool isPassByRef(const Type *Ty) const { return false; }

  virtual unsigned getTypeAlignment(const Type *Ty) const = 0;

  virtual unsigned getTypeSize(const Type *Ty) const = 0;
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIINFO_H
