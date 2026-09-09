//===- AMDGPU.cpp - AMDGPU ABI Implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace llvm {
namespace abi {

class AMDGPUTargetInfo : public TargetInfo {
private:
  TypeBuilder &TB;
  static const unsigned MaxNumRegsForArgsRet = 16;

  ArgInfo classifyReturnType(const Type *RetTy) const;
  ArgInfo classifyKernelArgumentType(const Type *Ty) const;
  ArgInfo classifyArgumentType(const Type *Ty, bool Variadic,
                               unsigned &NumRegsLeft) const;

  /// Target-independent fallback, mirroring classic CodeGen's DefaultABIInfo.
  ArgInfo classifyDefaultType(const Type *Ty, bool IsReturn) const;

  /// Estimate number of registers the type will use when passed in registers.
  uint64_t numRegsForType(const Type *Ty) const;

public:
  AMDGPUTargetInfo(TypeBuilder &TypeBuilder, const ABICompatInfo &Compat)
      : TargetInfo(Compat), TB(TypeBuilder) {}

  void computeInfo(FunctionInfo &FI) const override;
};

uint64_t AMDGPUTargetInfo::numRegsForType(const Type *Ty) const {
  uint64_t NumRegs = 0;

  if (const auto *VT = dyn_cast<VectorType>(Ty)) {
    // Compute from the number of elements. The reported size is based on the
    // in-memory size, which includes the padding 4th element for 3-vectors.
    const Type *EltTy = VT->getElementType();
    uint64_t EltSize = EltTy->getSizeInBits().getFixedValue();
    unsigned NumElts = VT->getNumElements().getFixedValue();

    // 16-bit element vectors should be passed as packed.
    if (EltSize == 16)
      return (NumElts + 1) / 2;

    uint64_t EltNumRegs = (EltSize + 31) / 32;
    return EltNumRegs * NumElts;
  }

  if (const auto *RT = dyn_cast<RecordType>(Ty)) {
    for (const FieldInfo &Field : RT->getFields())
      NumRegs += numRegsForType(Field.FieldType);
    return NumRegs;
  }

  return (Ty->getSizeInBits().getFixedValue() + 31) / 32;
}

ArgInfo AMDGPUTargetInfo::classifyDefaultType(const Type *Ty,
                                              bool IsReturn) const {
  if (IsReturn && Ty->isVoid())
    return ArgInfo::getIgnore();

  if (isAggregateTypeForABI(Ty)) {
    if (RecordArgABI RAA = getRecordArgABI(Ty); RAA != RAA_Default)
      return getNaturalAlignIndirect(Ty, /*ByVal=*/RAA == RAA_DirectInMemory);
    return getNaturalAlignIndirect(Ty, /*ByVal=*/!IsReturn);
  }

  if (const auto *IT = dyn_cast<IntegerType>(Ty))
    if (isPromotableInteger(IT))
      return ArgInfo::getExtend(Ty);

  return ArgInfo::getDirect();
}

ArgInfo AMDGPUTargetInfo::classifyReturnType(const Type *RetTy) const {
  if (RetTy->isVoid())
    return ArgInfo::getIgnore();

  if (isAggregateTypeForABI(RetTy)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // returned by value.
    if (getRecordArgABI(RetTy) == RAA_Default) {
      const auto *RT = dyn_cast<RecordType>(RetTy);

      // Ignore empty structs/unions.
      if (RT && RT->isEmpty())
        return ArgInfo::getIgnore();

      // Lower single-element structs to just return a regular value.
      if (const Type *SeltTy = isSingleElementStruct(RetTy))
        return ArgInfo::getDirect(SeltTy);

      if (RT && RT->hasFlexibleArrayMember())
        return classifyDefaultType(RetTy, /*IsReturn=*/true);

      // Pack aggregates <= 4 bytes into single VGPR or pair.
      uint64_t Size = RetTy->getSizeInBits().getFixedValue();
      if (Size <= 16)
        return ArgInfo::getDirect(TB.getIntegerType(16, Align(2), false));

      if (Size <= 32)
        return ArgInfo::getDirect(TB.getIntegerType(32, Align(4), false));

      if (Size <= 64) {
        const Type *I32Ty = TB.getIntegerType(32, Align(4), false);
        return ArgInfo::getDirect(TB.getArrayType(I32Ty, 2, /*SizeInBits=*/64));
      }

      if (numRegsForType(RetTy) <= MaxNumRegsForArgsRet)
        return ArgInfo::getDirect();
    }
  }

  // Otherwise just do the default thing.
  return classifyDefaultType(RetTy, /*IsReturn=*/true);
}

/// For kernels all parameters are really passed in a special buffer. It doesn't
/// make sense to pass anything byval, so everything must be direct.
ArgInfo AMDGPUTargetInfo::classifyKernelArgumentType(const Type *Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (const Type *SeltTy = isSingleElementStruct(Ty))
    Ty = SeltTy;

  // TODO: Classic coerces HIP scalar pointers from generic to global here; that
  // depends on LangOptions the ABI library cannot see, so it is skipped.
  if (isAggregateTypeForABI(Ty))
    return ArgInfo::getIndirect(Ty->getAlignment(), /*ByVal=*/false,
                                /*AddrSpace=*/AMDGPUAS::CONSTANT_ADDRESS);

  // TODO: Classic passes CanBeFlattened=false here to keep a struct intact;
  // ArgInfo cannot model that yet, so a multi-field record coerce may be
  // flattened.
  return ArgInfo::getDirect(Ty);
}

ArgInfo AMDGPUTargetInfo::classifyArgumentType(const Type *Ty, bool Variadic,
                                               unsigned &NumRegsLeft) const {
  assert(NumRegsLeft <= MaxNumRegsForArgsRet && "register estimate underflow");

  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO: Classic sets CanBeFlattened=false for variadics; not modeled here.
  if (Variadic)
    return ArgInfo::getDirect();

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (RecordArgABI RAA = getRecordArgABI(Ty); RAA != RAA_Default)
      return ArgInfo::getIndirect(Ty->getAlignment(),
                                  /*ByVal=*/RAA == RAA_DirectInMemory,
                                  /*AddrSpace=*/AMDGPUAS::PRIVATE_ADDRESS);

    // Ignore empty structs/unions.
    if (const auto *RT = dyn_cast<RecordType>(Ty); RT && RT->isEmpty())
      return ArgInfo::getIgnore();

    // Lower single-element structs to just pass a regular value.
    if (const Type *SeltTy = isSingleElementStruct(Ty))
      return ArgInfo::getDirect(SeltTy);

    if (const auto *RT = dyn_cast<RecordType>(Ty);
        RT && RT->hasFlexibleArrayMember())
      return classifyDefaultType(Ty, /*IsReturn=*/false);

    // Pack aggregates <= 8 bytes into single VGPR or pair.
    uint64_t Size = Ty->getSizeInBits().getFixedValue();
    if (Size <= 64) {
      unsigned NumRegs = (Size + 31) / 32;
      NumRegsLeft -= std::min(NumRegsLeft, NumRegs);

      if (Size <= 16)
        return ArgInfo::getDirect(TB.getIntegerType(16, Align(2), false));

      if (Size <= 32)
        return ArgInfo::getDirect(TB.getIntegerType(32, Align(4), false));

      const Type *I32Ty = TB.getIntegerType(32, Align(4), false);
      return ArgInfo::getDirect(TB.getArrayType(I32Ty, 2, /*SizeInBits=*/64));
    }

    if (NumRegsLeft > 0) {
      uint64_t NumRegs = numRegsForType(Ty);
      if (NumRegsLeft >= NumRegs) {
        NumRegsLeft -= NumRegs;
        return ArgInfo::getDirect();
      }
    }

    // Pass a struct argument by reference rather than by value.
    return ArgInfo::getIndirect(Ty->getAlignment(), /*ByVal=*/false,
                                /*AddrSpace=*/AMDGPUAS::PRIVATE_ADDRESS);
  }

  // Otherwise just do the default thing.
  ArgInfo AI = classifyDefaultType(Ty, /*IsReturn=*/false);
  if (!AI.isIndirect()) {
    uint64_t NumRegs = numRegsForType(Ty);
    NumRegsLeft -= std::min(NumRegs, uint64_t{NumRegsLeft});
  }

  return AI;
}

void AMDGPUTargetInfo::computeInfo(FunctionInfo &FI) const {
  CallingConv::ID CC = FI.getCallingConvention();

  if (!maybeCommonClassifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  unsigned ArgumentIndex = 0;
  const unsigned NumFixedArguments = FI.getNumRequiredArgs();

  unsigned NumRegsLeft = MaxNumRegsForArgsRet;
  for (ArgEntry &Arg : FI.arguments()) {
    if (CC == CallingConv::AMDGPU_KERNEL) {
      Arg.Info = classifyKernelArgumentType(Arg.ABIType);
    } else {
      bool FixedArgument = ArgumentIndex++ < NumFixedArguments;
      Arg.Info = classifyArgumentType(Arg.ABIType, !FixedArgument, NumRegsLeft);
    }
  }
}

std::unique_ptr<TargetInfo> createAMDGPUTargetInfo(TypeBuilder &TB) {
  return std::make_unique<AMDGPUTargetInfo>(TB, ABICompatInfo());
}

} // namespace abi
} // namespace llvm
