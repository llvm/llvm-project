//===----- FunctionInfo.h - ABI Function Information --------- C++ --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines FunctionInfo and associated types used in representing the
// ABI-coerced types for function arguments and return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_FUNCTIONINFO_H
#define LLVM_ABI_FUNCTIONINFO_H

#include "llvm/ABI/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TrailingObjects.h"
#include <optional>

namespace llvm {
namespace abi {

/// Helper class to encapsulate information about how a specific type should be
/// passed to or returned from a function.
class ArgInfo {
public:
  enum Kind {
    /// Pass the argument directly using the normal converted LLVM type, or by
    /// coercing to another specified type stored in 'CoerceToType'.
    Direct,
    /// Valid only for integer argument types. Same as 'direct' but also emit a
    /// zero/sign extension attribute.
    Extend,
    /// Pass the argument indirectly via a hidden pointer with the specified
    /// alignment and address space.
    Indirect,
    /// Ignore the argument (treat as void). Useful for void and empty structs.
    Ignore,
  };

private:
  const Type *CoercionType = nullptr;
  // Alignment is optional for direct arguments, but required for indirect
  // arguments. This invariant is enforced by the methods of this class.
  //
  // The field is not part of DirectAttrInfo/IndirectAttrInfo because it would
  // make the union non-trivial, disabling implicit copy/move constructors and
  // assignment operators for the entire class.
  MaybeAlign Alignment;

  struct DirectAttrInfo {
    unsigned Offset;
  };

  struct IndirectAttrInfo {
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

  ArgInfo(Kind K = Direct)
      : TheKind(K), SignExt(false), ZeroExt(false), IndirectByVal(false),
        IndirectRealign(false) {}

public:
  /// \param T The type to coerce to. If null, the argument's original type is
  ///          used directly.
  /// \param Offset Byte offset into the memory representation at which the
  ///               coerced type begins. Used when only part of a larger value
  ///               is passed directly (e.g. the high word of a multi-eightbyte
  ///               return value on x86-64).
  /// \param Align  Override for the argument's alignment. If absent, the
  ///               default alignment for \p T is used.
  static ArgInfo getDirect(const Type *T = nullptr, unsigned Offset = 0,
                           MaybeAlign Align = std::nullopt) {
    ArgInfo AI(Direct);
    AI.CoercionType = T;
    AI.Alignment = Align;
    AI.DirectAttr.Offset = Offset;
    return AI;
  }

  static ArgInfo getExtend(const Type *T) {
    assert(T && "Type cannot be null");
    assert(T->isInteger() && "Unexpected type - only integers can be extended");

    ArgInfo AI(Extend);
    AI.CoercionType = T;
    AI.Alignment = std::nullopt;
    AI.DirectAttr.Offset = 0;

    const IntegerType *IntTy = cast<IntegerType>(T);
    if (IntTy->isSigned())
      AI.setSignExt();
    else
      AI.setZeroExt();

    return AI;
  }

  /// Realign: the caller couldn't guarantee sufficient alignment - the callee
  /// must copy the argument to a properly aligned temporary before use.
  static ArgInfo getIndirect(Align Align, bool ByVal, unsigned AddrSpace = 0,
                             bool Realign = false) {
    ArgInfo AI(Indirect);
    AI.Alignment = Align;
    AI.IndirectAttr.AddrSpace = AddrSpace;
    AI.IndirectByVal = ByVal;
    AI.IndirectRealign = Realign;
    return AI;
  }

  static ArgInfo getIgnore() { return ArgInfo(Ignore); }

  ArgInfo &setSignExt(bool SignExtend = true) {
    this->SignExt = SignExtend;
    if (SignExtend)
      this->ZeroExt = false;
    return *this;
  }

  ArgInfo &setZeroExt(bool ZeroExtend = true) {
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

  MaybeAlign getDirectAlign() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return Alignment;
  }

  Align getIndirectAlign() const {
    assert(isIndirect() && "Invalid Kind!");
    assert(Alignment.has_value() &&
           "Indirect arguments must have an alignment");
    return *Alignment;
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

struct ArgEntry {
  const Type *ABIType;
  ArgInfo Info;

  ArgEntry(const Type *T) : ABIType(T), Info(ArgInfo::getDirect()) {}
  ArgEntry(const Type *T, ArgInfo A) : ABIType(T), Info(A) {}
};

class FunctionInfo final : private TrailingObjects<FunctionInfo, ArgEntry> {
private:
  const Type *ReturnType;
  ArgInfo ReturnInfo;
  unsigned NumArgs;
  CallingConv::ID CC = CallingConv::C;
  std::optional<unsigned> NumRequired;

  FunctionInfo(CallingConv::ID CC, const Type *RetTy, unsigned NumArguments,
               std::optional<unsigned> NumRequired)
      : ReturnType(RetTy), ReturnInfo(ArgInfo::getDirect()),
        NumArgs(NumArguments), CC(CC), NumRequired(NumRequired) {}

  friend class TrailingObjects;

public:
  using const_arg_iterator = const ArgEntry *;
  using arg_iterator = ArgEntry *;

  void operator delete(void *p) { ::operator delete(p); }
  const_arg_iterator arg_begin() const { return getTrailingObjects(); }
  const_arg_iterator arg_end() const { return getTrailingObjects() + NumArgs; }
  arg_iterator arg_begin() { return getTrailingObjects(); }
  arg_iterator arg_end() { return getTrailingObjects() + NumArgs; }

  unsigned arg_size() const { return NumArgs; }

  static FunctionInfo *
  create(CallingConv::ID CC, const Type *ReturnType,
         ArrayRef<const Type *> ArgTypes,
         std::optional<unsigned> NumRequired = std::nullopt);

  const Type *getReturnType() const { return ReturnType; }
  ArgInfo &getReturnInfo() { return ReturnInfo; }
  const ArgInfo &getReturnInfo() const { return ReturnInfo; }

  CallingConv::ID getCallingConvention() const { return CC; }

  bool isVariadic() const { return NumRequired.has_value(); }

  unsigned getNumRequiredArgs() const {
    return isVariadic() ? *NumRequired : arg_size();
  }

  ArrayRef<ArgEntry> arguments() const {
    return {getTrailingObjects(), NumArgs};
  }

  MutableArrayRef<ArgEntry> arguments() {
    return {getTrailingObjects(), NumArgs};
  }

  ArgEntry &getArgInfo(unsigned Index) {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }

  const ArgEntry &getArgInfo(unsigned Index) const {
    assert(Index < NumArgs && "Invalid argument index");
    return arguments()[Index];
  }
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_FUNCTIONINFO_H
