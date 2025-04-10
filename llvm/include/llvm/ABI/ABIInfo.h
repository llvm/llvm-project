//===- ABIInfo.h - ABI Information -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the interfaces used by frontends to get ABI information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_ABIINFO_H
#define LLVM_ABI_ABIINFO_H

#include "llvm/ABI/ABIType.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <vector>

namespace llvm {
class LLVMContext;

namespace abi {

/// Argument or return value ABI classification
enum class ABIArgKind {
  Direct,   // Just yeets the value straight into registers
  Indirect, // Passed indirectly via a hidden pointer
  Ignore,   // exists in the source but ghosted at compile time
  Expand,   // Expand into multiple arguments
  Coerce    // Coerce to different type
};

/// Information about how to handle an argument or return value
class ABIArgInfo {
  ABIArgKind Kind;
  const ABIType *CoerceToABIType; // Target ABI type for coercion
  Type *CoerceToLLVMType;         // Corresponding LLVM type
  unsigned DirectOffset;          // Offset for direct passing
  bool CanBeFlattened;            // Whether the type can be flattened
  bool InReg;                     // Whether to pass in registers

  /// For Expand kind - holds component types
  std::vector<ABIArgInfo> ExpandedArgs;

public:
  ABIArgInfo(ABIArgKind Kind, const ABIType *CoerceToABIType = nullptr,
             Type *CoerceToLLVMType = nullptr, unsigned DirectOffset = 0,
             bool CanBeFlattened = false, bool InReg = false)
      : Kind(Kind), CoerceToABIType(CoerceToABIType),
        CoerceToLLVMType(CoerceToLLVMType), DirectOffset(DirectOffset),
        CanBeFlattened(CanBeFlattened), InReg(InReg) {}

  // Factory methods
  static ABIArgInfo getDirect(const ABIType *ABITy = nullptr,
                              Type *LLVMTy = nullptr, unsigned Offset = 0,
                              bool CanBeFlattened = false, bool InReg = false) {
    return ABIArgInfo(ABIArgKind::Direct, ABITy, LLVMTy, Offset, CanBeFlattened,
                      InReg);
  }

  static ABIArgInfo getIndirect(const ABIType *ABITy = nullptr,
                                bool InReg = false) {
    return ABIArgInfo(ABIArgKind::Indirect, ABITy, nullptr, 0, false, InReg);
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(ABIArgKind::Ignore); }

  static ABIArgInfo getCoerce(const ABIType *ABITy, Type *LLVMTy = nullptr) {
    return ABIArgInfo(ABIArgKind::Coerce, ABITy, LLVMTy);
  }

  static ABIArgInfo getExpand() { return ABIArgInfo(ABIArgKind::Expand); }

  // Accessors
  ABIArgKind getKind() const { return Kind; }
  const ABIType *getCoerceToABIType() const { return CoerceToABIType; }
  Type *getCoerceToLLVMType() const { return CoerceToLLVMType; }
  unsigned getDirectOffset() const { return DirectOffset; }
  bool canBeFlattened() const { return CanBeFlattened; }
  bool getInReg() const { return InReg; }

  void setCoerceToLLVMType(Type *T) { CoerceToLLVMType = T; }

  // Predicates
  bool isDirect() const { return Kind == ABIArgKind::Direct; }
  bool isIndirect() const { return Kind == ABIArgKind::Indirect; }
  bool isIgnore() const { return Kind == ABIArgKind::Ignore; }
  bool isCoerce() const { return Kind == ABIArgKind::Coerce; }
  bool isExpand() const { return Kind == ABIArgKind::Expand; }

  // Expand kind management
  void setExpandedArgs(ArrayRef<ABIArgInfo> Args) {
    assert(Kind == ABIArgKind::Expand && "Not an expand kind");
    ExpandedArgs.assign(Args.begin(), Args.end());
  }

  ArrayRef<ABIArgInfo> getExpandedArgs() const {
    assert(Kind == ABIArgKind::Expand && "Not an expand kind");
    return ExpandedArgs;
  }
};

/// Base class for target-specific ABI information
class ABIInfo {
public:
  virtual ~ABIInfo() = default;

  /// Classify how a type should be passed as an argument
  virtual ABIArgInfo classifyArgumentType(const ABIType *Ty) = 0;

  /// Classify how a type should be returned
  virtual ABIArgInfo classifyReturnType(const ABIType *Ty) = 0;

  /// Convert an ABIType to the corresponding LLVM IR type
  virtual Type *getLLVMType(const ABIType *Ty, LLVMContext &Context) = 0;
};

/// Create target-specific ABI information based on the target triple
std::unique_ptr<ABIInfo> createABIInfo(StringRef TargetTriple);

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIINFO_H
