//===----- ABIFunctionInfo.h - ABI Function Information ----- C++ ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines ABIFunctionInfo and associated types used in representing the
// ABI-coerced types for function arguments and return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_ABIFUNCTIONINFO_H
#define LLVM_ABI_ABIFUNCTIONINFO_H

#include "llvm/ABI/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"

namespace llvm {
namespace abi {

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind {
    Direct,
    Extend,
    Indirect,
    Ignore,
  };

private:
  const Type *CoercionType;

  struct DirectAttrInfo {
    unsigned Offset;
    unsigned Align;
  };

  struct IndirectAttrInfo {
    unsigned Align;
    unsigned AddrSpace;
  };

  union {
    DirectAttrInfo DirectAttr;
    IndirectAttrInfo IndirectAttr;
  };

  Kind TheKind;
  bool SignExt : 1;
  bool ZeroExt : 1;
  bool IndirectByVal : 1;
  bool IndirectRealign : 1;

  ABIArgInfo(Kind K = Direct)
      : CoercionType(nullptr), TheKind(K), SignExt(false), ZeroExt(false),
        IndirectByVal(false), IndirectRealign(false) {}

public:
  static ABIArgInfo getDirect(const Type *T = nullptr, unsigned Offset = 0,
                              unsigned Align = 0) {
    ABIArgInfo AI(Direct);
    AI.CoercionType = T;
    AI.DirectAttr.Offset = Offset;
    AI.DirectAttr.Align = Align;
    return AI;
  }

  static ABIArgInfo getExtend(const Type *T) {
    assert(T && "Type cannot be null");
    assert(T->isInteger() && "Unexpected type - only integers can be extended");

    ABIArgInfo AI(Extend);
    AI.CoercionType = T;
    AI.DirectAttr.Offset = 0;
    AI.DirectAttr.Align = 0;

    const IntegerType *IntTy = dyn_cast<IntegerType>(T);
    if (IntTy->isSigned())
      AI.setSignExt();
    else
      AI.setZeroExt();

    return AI;
  }

  static ABIArgInfo getIndirect(unsigned Align = 0, bool ByVal = true,
                                unsigned AddrSpace = 0, bool Realign = false) {
    ABIArgInfo AI(Indirect);
    AI.IndirectAttr.Align = Align;
    AI.IndirectAttr.AddrSpace = AddrSpace;
    AI.IndirectByVal = ByVal;
    AI.IndirectRealign = Realign;
    return AI;
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  ABIArgInfo &setSignExt(bool SignExtend = true) {
    this->SignExt = SignExtend;
    if (SignExtend)
      this->ZeroExt = false;
    return *this;
  }

  ABIArgInfo &setZeroExt(bool ZeroExtend = true) {
    this->ZeroExt = ZeroExtend;
    if (ZeroExtend)
      this->SignExt = false;
    return *this;
  }

  Kind getKind() const { return TheKind; }
  bool isDirect() const { return TheKind == Direct; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isIgnore() const { return TheKind == Ignore; }
  bool isExtend() const { return TheKind == Extend; }

  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectAttr.Offset;
  }

  unsigned getDirectAlign() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectAttr.Align;
  }

  unsigned getIndirectAlign() const {
    assert(isIndirect() && "Invalid Kind!");
    return IndirectAttr.Align;
  }

  unsigned getIndirectAddrSpace() const {
    assert(isIndirect() && "Invalid Kind!");
    return IndirectAttr.AddrSpace;
  }

  bool getIndirectByVal() const {
    assert(isIndirect() && "Invalid Kind!");
    return IndirectByVal;
  }

  bool getIndirectRealign() const {
    assert(isIndirect() && "Invalid Kind!");
    return IndirectRealign;
  }

  bool isSignExt() const {
    assert(isExtend() && "Invalid Kind!");
    return SignExt;
  }

  bool isZeroExt() const {
    assert(isExtend() && "Invalid Kind!");
    return ZeroExt;
  }

  bool isNoExt() const {
    assert(isExtend() && "Invalid Kind!");
    return !SignExt && !ZeroExt;
  }

  const Type *getCoerceToType() const {
    assert((isDirect() || isExtend()) && "Invalid Kind!");
    return CoercionType;
  }
};

/// Calling convention and related attributes for a function call.
struct ABICallAttributes {
  CallingConv::ID CC = CallingConv::C;

  ABICallAttributes() = default;
  explicit ABICallAttributes(CallingConv::ID CC) : CC(CC) {}
};

/// Information about required vs optional arguments for variadic functions.
struct RequiredArgs {
private:
  unsigned NumRequired;
  static constexpr unsigned All = ~0U;

public:
  RequiredArgs() : NumRequired(All) {}
  explicit RequiredArgs(unsigned N) : NumRequired(N) {}

  static RequiredArgs forPrototypedFunction(unsigned NumArgs) {
    return RequiredArgs(NumArgs);
  }

  static RequiredArgs forVariadicFunction(unsigned NumRequired) {
    return RequiredArgs(NumRequired);
  }

  bool allowsOptionalArgs() const { return NumRequired != All; }

  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return NumRequired;
  }

  bool operator==(const RequiredArgs &Other) const {
    return NumRequired == Other.NumRequired;
  }
};

struct ABIFunctionInfoArgInfo {
  const Type *ABIType;
  ABIArgInfo ArgInfo;

  ABIFunctionInfoArgInfo()
      : ABIType(nullptr), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(const Type *T)
      : ABIType(T), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(const Type *T, ABIArgInfo A)
      : ABIType(T), ArgInfo(A) {}
};

class ABIFunctionInfo final
    : private TrailingObjects<ABIFunctionInfo, ABIFunctionInfoArgInfo> {
  using ArgInfo = ABIFunctionInfoArgInfo;

private:
  const Type *ReturnType;
  ABIArgInfo ReturnInfo;
  unsigned NumArgs;
  ABICallAttributes CallAttrs;
  RequiredArgs Required;

  ABIFunctionInfo(const Type *RetTy, unsigned NumArguments)
      : ReturnType(RetTy), ReturnInfo(ABIArgInfo::getDirect()),
        NumArgs(NumArguments) {}

  friend class TrailingObjects;

public:
  using const_arg_iterator = const ArgInfo *;
  using arg_iterator = ArgInfo *;

  void operator delete(void *p) { ::operator delete(p); }
  const_arg_iterator arg_begin() const { return getTrailingObjects(); }
  const_arg_iterator arg_end() const { return getTrailingObjects() + NumArgs; }
  arg_iterator arg_begin() { return getTrailingObjects(); }
  arg_iterator arg_end() { return getTrailingObjects() + NumArgs; }

  unsigned arg_size() const { return NumArgs; }

  static ABIFunctionInfo *
  create(CallingConv::ID CC, const Type *ReturnType,
         ArrayRef<const Type *> ArgTypes,
         const ABICallAttributes &CallAttrs = ABICallAttributes(),
         RequiredArgs Required = RequiredArgs());

  const Type *getReturnType() const { return ReturnType; }
  ABIArgInfo &getReturnInfo() { return ReturnInfo; }
  const ABIArgInfo &getReturnInfo() const { return ReturnInfo; }

  CallingConv::ID getCallingConvention() const { return CallAttrs.CC; }

  unsigned getNumRequiredArgs() const {
    return Required.allowsOptionalArgs() ? Required.getNumRequiredArgs()
                                         : arg_size();
  }

  ArrayRef<ArgInfo> arguments() const {
    return {getTrailingObjects(), NumArgs};
  }

  MutableArrayRef<ArgInfo> arguments() {
    return {getTrailingObjects(), NumArgs};
  }

  ArgInfo &getArgInfo(unsigned Index) {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }

  const ArgInfo &getArgInfo(unsigned Index) const {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }

  unsigned getNumArgs() const { return NumArgs; }
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIFUNCTIONINFO_H
