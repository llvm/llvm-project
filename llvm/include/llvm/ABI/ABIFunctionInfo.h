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
    IndirectAliased,
    Ignore,
    Expand,
    CoerceAndExpand,
    InAlloca
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
    unsigned AllocaFieldIndex;
  };
  union {
    const Type *PaddingType;
    const Type *UnpaddedCoerceAndExpandType;
  };
  Kind TheKind;
  bool InReg : 1;
  bool PaddingInReg : 1;
  bool SignExt : 1;
  bool ZeroExt : 1;
  bool IndirectByVal : 1;
  bool IndirectRealign : 1;
  bool SRetAfterThis : 1;
  bool CanBeFlattened : 1;
  bool InAllocaSRet : 1;
  bool InAllocaIndirect : 1;

  ABIArgInfo(Kind K = Direct)
      : CoercionType(nullptr), TheKind(K), InReg(false), PaddingInReg(false),
        SignExt(false), ZeroExt(false), IndirectByVal(false) {}

public:
  static ABIArgInfo getDirect(const Type *T = nullptr, unsigned Offset = 0,
                              const Type *Padding = nullptr,
                              bool CanBeFlattened = true, unsigned Align = 0) {
    ABIArgInfo AI(Direct);
    AI.CoercionType = T;
    AI.PaddingType = Padding;
    AI.DirectAttr.Offset = Offset;
    AI.DirectAttr.Align = Align;
    AI.CanBeFlattened = CanBeFlattened;
    return AI;
  }

  static ABIArgInfo getIndirectAliased(unsigned Align, unsigned AddrSpace = 0,
                                       bool Realign = false,
                                       const Type *Padding = nullptr) {
    ABIArgInfo AI(IndirectAliased);
    AI.IndirectAttr.Align = Align;
    AI.IndirectAttr.AddrSpace = AddrSpace;
    AI.IndirectRealign = Realign;
    AI.PaddingType = Padding;
    return AI;
  }
  static ABIArgInfo getDirectInReg(const Type *T = nullptr) {
    ABIArgInfo AI = getDirect(T);
    AI.InReg = true;
    return AI;
  }
  static ABIArgInfo getExtend(const Type *T) {
    assert(T && "Type cannot be null");
    assert(T->isInteger() && "Unexpected type - only integers can be extended");

    ABIArgInfo AI(Extend);
    AI.CoercionType = T;
    AI.DirectAttr.Offset = 0;
    AI.DirectAttr.Align = 0;
    AI.PaddingType = nullptr;

    const IntegerType *IntTy = dyn_cast<IntegerType>(T);
    if (IntTy->isSigned()) {
      AI.setSignExt();
    } else {
      AI.setZeroExt();
    }

    return AI;
  }

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

  static ABIArgInfo getIndirect(unsigned Align = 0, bool ByVal = true,
                                unsigned AddrSpace = 0, bool Realign = false,
                                const Type *Padding = nullptr) {
    ABIArgInfo AI(Indirect);
    AI.IndirectAttr.Align = Align;
    AI.IndirectAttr.AddrSpace = AddrSpace;
    AI.IndirectByVal = ByVal;
    AI.IndirectRealign = Realign;
    AI.SRetAfterThis = false;
    AI.PaddingType = Padding;
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
  bool isExtend() const { return TheKind == Extend; }
  bool isExpand() const { return TheKind == Expand; }
  bool isCoerceAndExpand() const { return TheKind == CoerceAndExpand; }
  bool isIndirectAliased() const { return TheKind == IndirectAliased; }
  bool isInAlloca() const { return TheKind == InAlloca; }
  bool isInReg() const { return InReg; }
  bool isSignExt() const { return SignExt; }
  bool hasPaddingInReg() const { return PaddingInReg; }

  const Type *getPaddingType() const {
    return canHavePaddingType() ? PaddingType : nullptr;
  }

  bool canHavePaddingType() const {
    return isDirect() || isExtend() || isIndirect() || isIndirectAliased() ||
           isExpand();
  }

  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectAttr.Offset;
  }
  unsigned getIndirectAlign() const {
    assert((isIndirect() || isIndirectAliased()) && "Invalid Kind!");
    return IndirectAttr.Align;
  }

  unsigned getIndirectAddrSpace() const {
    assert((isIndirect() || isIndirectAliased()) && "Invalid Kind!");
    return IndirectAttr.AddrSpace;
  }

  bool getIndirectByVal() const {
    assert(isIndirect() && "Invalid Kind!");
    return IndirectByVal;
  }
  bool getIndirectRealign() const {
    assert((isIndirect() || isIndirectAliased()) && "Invalid Kind!");
    return IndirectRealign;
  }

  bool isSRetAfterThis() const {
    assert(isIndirect() && "Invalid Kind!");
    return SRetAfterThis;
  }

  unsigned getInAllocaFieldIndex() const {
    assert(isInAlloca() && "Invalid kind!");
    return AllocaFieldIndex;
  }

  bool getInAllocaIndirect() const {
    assert(isInAlloca() && "Invalid kind!");
    return InAllocaIndirect;
  }

  bool getInAllocaSRet() const {
    assert(isInAlloca() && "Invalid kind!");
    return InAllocaSRet;
  }
  const Type *getUnpaddedCoerceAndExpandType() const {
    assert(isCoerceAndExpand());
    return UnpaddedCoerceAndExpandType;
  }
  bool isZeroExt() const {
    assert(isExtend() && "Invalid Kind!");
    return ZeroExt;
  }
  bool isNoExt() const {
    assert(isExtend() && "Invalid Kind!");
    return !SignExt && !ZeroExt;
  }

  unsigned getDirectAlign() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectAttr.Align;
  }

  bool getCanBeFlattened() const {
    assert(isDirect() && "Invalid kind!");
    return CanBeFlattened;
  }

  const Type *getCoerceToType() const {
    assert((isDirect() || isExtend() || isCoerceAndExpand()) &&
           "Invalid Kind!");
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

/// Function-level ABI attributes that affect argument/return passing
struct ABICallAttributes {
  CallingConv::ID CC = CallingConv::C;
  CallingConv::ID EffectiveCC = CallingConv::C;

  bool HasSRet = false;
  bool IsInstanceMethod = false;
  bool IsChainCall = false;
  bool IsDelegateCall = false;

  // Register usage controls
  bool HasRegParm = false;
  unsigned RegParm = 0;
  bool NoCallerSavedRegs = false;

  // Security extensions
  bool NoCfCheck = false;
  bool CmseNSCall = false;

  // Memory management
  bool ReturnsRetained = false;
  unsigned MaxVectorWidth = 0;

  ABICallAttributes() = default;
  ABICallAttributes(CallingConv::ID CC) : CC(CC), EffectiveCC(CC) {}
};

/// Information about required vs optional arguments for variadic functions
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
  bool isVariadic() const { return allowsOptionalArgs(); }

  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return NumRequired;
  }

  bool operator==(const RequiredArgs &Other) const {
    return NumRequired == Other.NumRequired;
  }
};

/// Argument information for ABIFunctionInfo
struct ABIFunctionInfoArgInfo {
  const Type *ABIType;
  ABIArgInfo ArgInfo;

  ABIFunctionInfoArgInfo()
      : ABIType(nullptr), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(Type *T)
      : ABIType(T), ArgInfo(ABIArgInfo::getDirect()) {}
  ABIFunctionInfoArgInfo(Type *T, ABIArgInfo A) : ABIType(T), ArgInfo(A) {}
};

class ABIFunctionInfo final
    : private TrailingObjects<ABIFunctionInfo, ABIFunctionInfoArgInfo> {
  typedef ABIFunctionInfoArgInfo ArgInfo;

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
  typedef const ArgInfo *const_arg_iterator;
  typedef ArgInfo *arg_iterator;
  void operator delete(void *p) {
    ::operator delete(p);
  }

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
  const ABICallAttributes &getCallAttributes() const { return CallAttrs; }
  RequiredArgs getRequiredArgs() const { return Required; }
  bool isVariadic() const { return Required.isVariadic(); }

  unsigned getNumRequiredArgs() const {
    return isVariadic() ? Required.getNumRequiredArgs() : arg_size();
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
