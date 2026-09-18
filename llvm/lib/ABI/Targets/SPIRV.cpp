//===- SPIRV.cpp - SPIR-V ABI Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Casting.h"

namespace llvm::abi {

class SPIRVTargetInfo : public TargetInfo {
private:
  ABICompatInfo CompatInfo;

  // When set, SPIR_KERNEL aggregate arguments are passed indirectly (byval)
  // so the callee gets its own copy. The consumer sets this for device
  // compilations, matching classic CodeGen's isTargetDevice() gate.
  bool KernelPassAggregatesIndirect;

  ArgInfo classifyReturnType(const Type *RetTy) const {
    if (RetTy->isVoid())
      return ArgInfo::getIgnore();

    if (isAggregateTypeForABI(RetTy))
      return getNaturalAlignIndirect(RetTy, /*ByVal=*/false);

    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy))
      if (IntTy->isBitInt() && IntTy->getSizeInBits().getFixedValue() > 128)
        return getNaturalAlignIndirect(RetTy, /*ByVal=*/false);

    if (const auto *IntTy = dyn_cast<IntegerType>(RetTy))
      if (isPromotableInteger(IntTy))
        return ArgInfo::getExtend(RetTy);

    return ArgInfo::getDirect();
  }

  ArgInfo classifyArgumentType(const Type *ArgTy) const {
    ArgTy = useFirstFieldIfTransparentUnion(ArgTy);

    if (isAggregateTypeForABI(ArgTy))
      return getNaturalAlignIndirect(ArgTy, /*ByVal=*/true);

    if (const auto *IntTy = dyn_cast<IntegerType>(ArgTy)) {
      if (IntTy->isBitInt() && IntTy->getSizeInBits().getFixedValue() > 128)
        return getNaturalAlignIndirect(ArgTy, /*ByVal=*/true);

      if (isPromotableInteger(IntTy))
        return ArgInfo::getExtend(ArgTy);
    }

    return ArgInfo::getDirect();
  }

  ArgInfo classifyKernelArgumentType(const Type *ArgTy) const {
    // Aggregate kernel arguments are forced byval so the callee gets its own
    // copy, which is required for the object to be valid on the device, like
    // Clang's SPIR-V CodeGen.
    if (KernelPassAggregatesIndirect && isAggregateTypeForABI(ArgTy))
      return getNaturalAlignIndirect(ArgTy, /*ByVal=*/true);

    return classifyArgumentType(ArgTy);
  }

public:
  SPIRVTargetInfo(TypeBuilder &Builder, const ABICompatInfo &Info,
                  bool KernelPassAggregatesIndirect)
      : TargetInfo(Builder), CompatInfo(Info),
        KernelPassAggregatesIndirect(KernelPassAggregatesIndirect) {}

  const ABICompatInfo &getABICompatInfo() const override { return CompatInfo; }

  void computeInfo(FunctionInfo &FI) const override {
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    bool IsKernel = FI.getCallingConvention() == CallingConv::SPIR_KERNEL;
    for (auto &I : FI.arguments())
      I.Info = IsKernel ? classifyKernelArgumentType(I.ABIType)
                        : classifyArgumentType(I.ABIType);
  }
};

std::unique_ptr<TargetInfo>
createSPIRVTargetInfo(TypeBuilder &TB, const ABICompatInfo &Compat,
                      bool KernelPassAggregatesIndirect) {
  return std::make_unique<SPIRVTargetInfo>(TB, Compat,
                                           KernelPassAggregatesIndirect);
}

} // namespace llvm::abi
