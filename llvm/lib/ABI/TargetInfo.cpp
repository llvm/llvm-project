//===- TargetInfo.cpp - Target ABI information ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetInfo.h"

using namespace llvm::abi;

bool TargetInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Check for fundamental scalar types.
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer() || Ty->isVector())
    return false;

  // A matrix type is modeled as an array but lowers to a single flattened
  // vector and has scalar evaluation kind in classic CodeGen, so it is not an
  // aggregate for ABI purposes.
  if (const auto *AT = dyn_cast<ArrayType>(Ty))
    if (AT->isMatrixType())
      return false;

  // Everything else is treated as aggregate.
  return true;
}

bool TargetInfo::isPromotableInteger(const IntegerType *IT) const {
  // TODO: The threshold should be the target's int size rather than a
  // hardcoded 32.
  unsigned BitWidth = IT->getSizeInBits().getFixedValue();
  return BitWidth < 32;
}

ArgInfo TargetInfo::getNaturalAlignIndirect(const Type *Ty, bool ByVal) const {
  return ArgInfo::getIndirect(Ty->getAlignment(), ByVal);
}

RecordArgABI TargetInfo::getRecordArgABI(const RecordType *RT) const {
  if (RT && !RT->canPassInRegisters())
    return RAA_Indirect;
  return RAA_Default;
}

RecordArgABI TargetInfo::getRecordArgABI(const Type *Ty) const {
  // TODO: When Microsoft ABI is supported, CXX records may need different
  // handling here (see MicrosoftCXXABI::getRecordArgABI in Clang).
  const RecordType *RT = dyn_cast<RecordType>(Ty);
  if (!RT)
    return RAA_Default;
  return getRecordArgABI(RT);
}

const Type *TargetInfo::useFirstFieldIfTransparentUnion(const Type *Ty) const {
  if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    if (RT->isUnion() && RT->isTransparentUnion()) {
      auto Fields = RT->getFields();
      assert(!Fields.empty() && "transparent union cannot be empty");
      return Fields.front().FieldType;
    }
  }
  return Ty;
}

bool TargetInfo::maybeCommonClassifyReturnType(FunctionInfo &FI) const {
  const abi::Type *Ty = FI.getReturnType();

  // TODO: When Microsoft ABI is supported, CXX records may need different
  // handling here (see MicrosoftCXXABI::classifyReturnType in Clang).
  if (const auto *RT = llvm::dyn_cast<abi::RecordType>(Ty)) {
    if (!RT->canPassInRegisters()) {
      // A record that cannot pass in registers (e.g. a non-trivial copy/dtor)
      // is returned indirectly with ByVal=false. This is the RAA path and is
      // distinct from getIndirectReturnResult (plain aggregates), which uses
      // ByVal=true.
      FI.getReturnInfo() =
          ArgInfo::getIndirect(RT->getAlignment(), /*ByVal=*/false);
      return true;
    }
  }

  return false;
}
