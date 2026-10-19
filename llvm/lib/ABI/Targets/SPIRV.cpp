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
#include "llvm/Support/Casting.h"

namespace llvm::abi {

// The LLVM address space of the SPIR-V CrossWorkgroup storage class.
static constexpr unsigned CrossWorkgroupAddrSpace = 1;

// The default classification applies, except for SPIR_KERNEL arguments, which
// SPIR-V restricts in which address spaces they may point to and requires to be
// copied when they are aggregates.
class SPIRVTargetInfo final : public DefaultTargetInfo {
private:
  ABICompatInfo CompatInfo;
  unsigned DefaultAddrSpace;
  bool CoerceDefaultPtrKernelArgs;
  bool KernelPassAggregatesIndirect;

  ArgInfo classifyKernelArgumentType(const Type *ArgTy) const {
    // Kernel arguments may not be default address space pointers, so pass them
    // as CrossWorkgroup pointers. An atomic pointer of the same size is passed
    // as the pointer itself.
    if (CoerceDefaultPtrKernelArgs) {
      const Type *Ty = ArgTy;
      if (const auto *AT = dyn_cast<AtomicType>(Ty);
          AT && AT->getSizeInBits() == AT->getValueType()->getSizeInBits())
        Ty = AT->getValueType();
      if (const auto *PtrTy = dyn_cast<PointerType>(Ty);
          PtrTy && PtrTy->getAddrSpace() == DefaultAddrSpace) {
        const Type *Coerced = TB.getPointerType(
            PtrTy->getSizeInBits().getFixedValue(), PtrTy->getAlignment(),
            CrossWorkgroupAddrSpace, /*HasAddressSpaceQualifier=*/true);
        return ArgInfo::getDirect(Coerced, /*Offset=*/0, /*Align=*/std::nullopt,
                                  /*CanBeFlattened=*/false);
      }
    }

    // Force copying aggregate type in kernel arguments by value. This is
    // required for the object copied to be valid on the device. TODO:
    // hardcoding to 0 should be revisited if HIPSPV / byval starts making use
    // of the AS of an indirect arg.
    if (KernelPassAggregatesIndirect && isAggregateTypeForABI(ArgTy))
      return getNaturalAlignIndirect(ArgTy, /*AddrSpace=*/0, /*ByVal=*/true);

    return classifyArgumentType(ArgTy);
  }

public:
  SPIRVTargetInfo(TypeBuilder &Builder, unsigned DefaultAddrSpace,
                  bool CoerceDefaultPtrKernelArgs,
                  bool KernelPassAggregatesIndirect)
      : DefaultTargetInfo(Builder), DefaultAddrSpace(DefaultAddrSpace),
        CoerceDefaultPtrKernelArgs(CoerceDefaultPtrKernelArgs),
        KernelPassAggregatesIndirect(KernelPassAggregatesIndirect) {}

  const ABICompatInfo &getABICompatInfo() const override { return CompatInfo; }

  bool hasInt128Type() const override { return false; }

  // A record that cannot be copied is constructed in place, so its sret
  // pointer follows the default address space like a "this" pointer.
  unsigned getSRetAddrSpace() const override { return DefaultAddrSpace; }

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
createSPIRVTargetInfo(TypeBuilder &TB, unsigned DefaultAddrSpace,
                      bool CoerceDefaultPtrKernelArgs,
                      bool KernelPassAggregatesIndirect) {
  return std::make_unique<SPIRVTargetInfo>(TB, DefaultAddrSpace,
                                           CoerceDefaultPtrKernelArgs,
                                           KernelPassAggregatesIndirect);
}

} // namespace llvm::abi
