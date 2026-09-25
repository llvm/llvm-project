//===- SPIRV.cpp - SPIR-V ABI Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/DefaultTargetInfo.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/IR/CallingConv.h"

namespace llvm::abi {

// Mirrors Clang's SPIRVABIInfo: the default classification applies, except
// for SPIR_KERNEL arguments.
class SPIRVTargetInfo : public DefaultTargetInfo {
private:
  ABICompatInfo CompatInfo;

  // When set, SPIR_KERNEL aggregate arguments are passed indirectly (byval)
  // so the callee gets its own copy. The consumer sets this for device
  // compilations, matching classic CodeGen's isTargetDevice() gate.
  bool KernelPassAggregatesIndirect;

  ArgInfo classifyKernelArgumentType(const Type *ArgTy) const {
    // TODO: Clang also coerces default address space pointer arguments to
    // CrossWorkGroup pointers, which needs the target's language address space
    // mapping.

    // Force copying aggregate type in kernel arguments by value. This is
    // required for the object copied to be valid on the device. TODO:
    // hardcoding to 0 should be revisited if HIPSPV / byval starts making use
    // of the AS of an indirect arg.
    if (KernelPassAggregatesIndirect && isAggregateTypeForABI(ArgTy))
      return getNaturalAlignIndirect(ArgTy, /*AddrSpace=*/0, /*ByVal=*/true);

    return classifyArgumentType(ArgTy);
  }

public:
  SPIRVTargetInfo(TypeBuilder &Builder, const ABICompatInfo &Info,
                  bool KernelPassAggregatesIndirect)
      : DefaultTargetInfo(Builder), CompatInfo(Info),
        KernelPassAggregatesIndirect(KernelPassAggregatesIndirect) {}

  const ABICompatInfo &getABICompatInfo() const override { return CompatInfo; }

  void computeInfo(FunctionInfo &FI) const override {
    // The logic is the same as in DefaultTargetInfo, except for kernel
    // arguments.
    if (!maybeCommonClassifyReturnType(FI))
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
