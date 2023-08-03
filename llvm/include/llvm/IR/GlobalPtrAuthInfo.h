//===- GlobalPtrAuthInfo.h - Analysis tools for ptrauth globals -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a set of utilities to analyze llvm.ptrauth globals, and
/// to decompose them into key, discriminator, and base pointer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALPTRAUTHINFO_H
#define LLVM_IR_GLOBALPTRAUTHINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/Error.h"

namespace llvm {

/// Helper class to access the information regarding an "llvm.ptrauth" global.
/// These globals are of the form:
///  @sg = constant { i8*, i32, i64, i64 }
///          { i8* bitcast (i32* @g to i8*),      ; base pointer
///            i32 2,                             ; key ID
///            i64 ptrtoint (i8** @pg to i64),    ; address discriminator
///            i64 42                             ; discriminator
///          }, section "llvm.ptrauth"
///
class GlobalPtrAuthInfo {
  const GlobalVariable *GV;

  const ConstantStruct *getInitializer() const {
    return cast<ConstantStruct>(GV->getInitializer());
  }

  GlobalPtrAuthInfo(const GlobalVariable *GV) : GV(GV) {}

public:
  /// A constant value for the address discriminator which has special
  /// significance to coroutine lowering.
  enum { AddrDiscriminator_UseCoroStorage = 1 };

  /// Try to analyze \p V as an authenticated global reference, and return its
  /// information if successful.
  static Optional<GlobalPtrAuthInfo> analyze(const Value *V);

  /// Try to analyze \p V as an authenticated global reference, and return its
  /// information if successful, or an error explaining the failure if not.
  static Expected<GlobalPtrAuthInfo> tryAnalyze(const Value *V);

  /// Access the information contained in the "llvm.ptrauth" globals.
  /// @{
  /// The "llvm.ptrauth" global itself.
  const GlobalVariable *getGV() const { return GV; }

  /// The pointer that is authenticated in this authenticated global reference.
  const Constant *getPointer() const {
    return cast<Constant>(getInitializer()->getOperand(0));
  }

  Constant *getPointer() {
    return cast<Constant>(getInitializer()->getOperand(0));
  }

  /// The Key ID, an i32 constant.
  const ConstantInt *getKey() const {
    return cast<ConstantInt>(getInitializer()->getOperand(1));
  }

  /// The address discriminator if any, or the null constant.
  /// If present, this must be a value equivalent to the storage location of
  /// the only user of the authenticated ptrauth global.
  const Constant *getAddrDiscriminator() const {
    return cast<Constant>(getInitializer()->getOperand(2));
  }

  /// Whether there is any non-null address discriminator.
  bool hasAddressDiversity() const {
    return !getAddrDiscriminator()->isNullValue();
  }

  /// Whether the address uses a special address discriminator.
  /// These discriminators can't be used in real pointer-auth values; they
  /// can only be used in "prototype" values that indicate how some real
  /// schema is supposed to be produced.
  bool hasSpecialAddressDiscriminator(uint64_t value) const {
    if (auto intValue = dyn_cast<ConstantInt>(getAddrDiscriminator()))
      return intValue->getValue() == value;
    return false;
  }

  /// The discriminator.
  const ConstantInt *getDiscriminator() const {
    return cast<ConstantInt>(getInitializer()->getOperand(3));
  }
  /// @}

  /// Check whether an authentication operation with key \p KeyV and (possibly
  /// blended) discriminator \p DiscriminatorV is compatible with this
  /// authenticated global reference.
  bool isCompatibleWith(const Value *Key, const Value *Discriminator,
                        const DataLayout &DL) const;

  /// Produce a "llvm.ptrauth" global that signs a value using the given
  /// schema.  The result will be casted to have the same type as the value.
  static llvm::Constant *create(Module &M, Constant *Pointer, ConstantInt *Key,
                                Constant *AddrDiscriminator,
                                ConstantInt *Discriminator);

  /// Produce a new "llvm.ptrauth" global for signing the given value using
  /// the same schema as is stored in this info.
  llvm::Constant *createWithSameSchema(Module &M, Constant *Pointer) const;
};

} // end namespace llvm

#endif
