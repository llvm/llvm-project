//===- AMDGPU.cpp - AMDGPU ABI Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by amdgcn and the AMDGCN-flavoured SPIR-V that lowers to it. The two
// differ only in the values carried by AMDGPUABIOptions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

namespace llvm::abi {

class AMDGPUTargetInfo : public TargetInfo {
private:
  /// Registers available for arguments and return values, in 32-bit units.
  static constexpr unsigned MaxNumRegsForArgsRet = 16;

  TypeBuilder &TB;
  AMDGPUABIOptions Opts;
  const IntegerType *Int16Ty;
  const IntegerType *Int32Ty;
  const ArrayType *Int32PairTy;

  uint64_t numRegsForType(const Type *Ty) const {
    if (const auto *VT = dyn_cast<VectorType>(Ty)) {
      // Compute from the number of elements. The reported size is based on the
      // in-memory size, which includes the padding 4th element for 3-vectors.
      uint64_t EltSize =
          VT->getElementType()->getTypeAllocSizeInBits().getFixedValue();
      uint64_t NumElts = VT->getNumElements().getFixedValue();

      // 16-bit element vectors should be passed as packed.
      if (EltSize == 16)
        return (NumElts + 1) / 2;

      return divideCeil(EltSize, 32) * NumElts;
    }

    if (const auto *RT = dyn_cast<RecordType>(Ty)) {
      assert(!RT->hasFlexibleArrayMember());

      // Bases are deliberately not counted, matching RecordDecl::fields() in
      // the classic classifier.
      uint64_t NumRegs = 0;
      for (const FieldInfo &Field : RT->getFields())
        NumRegs += numRegsForType(Field.FieldType);

      return NumRegs;
    }

    return divideCeil(Ty->getTypeAllocSizeInBits().getFixedValue(), 32);
  }

  /// Coerce a scalar pointer argument from the generic address space to the
  /// one kernel arguments must use.
  const Type *coerceKernelArgumentType(const Type *Ty) const {
    const auto *PtrTy = dyn_cast<PointerType>(Ty);
    if (PtrTy && PtrTy->getAddrSpace() == Opts.GenericAddrSpace)
      return TB.getPointerType(PtrTy->getSizeInBits().getFixedValue(),
                               PtrTy->getAlignment(), Opts.KernelArgAddrSpace);
    return Ty;
  }

  /// The non-aggregate tail shared by the two Clang DefaultABIInfo rules.
  ArgInfo defaultClassifyScalar(const Type *Ty) const {
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt() &&
          IntTy->getSizeInBits().getFixedValue() > getMaxDirectBitIntWidth())
        return getNaturalAlignIndirect(Ty, /*ByVal=*/true,
                                       Opts.AllocaAddrSpace);

      if (isPromotableInteger(IntTy))
        return ArgInfo::getExtend(Ty);
    }

    return ArgInfo::getDirect();
  }

  /// The Clang DefaultABIInfo rules, the fallback for anything AMDGPU does not
  /// pack into registers itself.
  ArgInfo defaultClassifyArgumentType(const Type *Ty) const {
    Ty = useFirstFieldIfTransparentUnion(Ty);

    if (isAggregateTypeForABI(Ty))
      return getNaturalAlignIndirect(Ty, getRecordArgABI(Ty) != RAA_Indirect,
                                     Opts.AllocaAddrSpace);

    return defaultClassifyScalar(Ty);
  }

  ArgInfo defaultClassifyReturnType(const Type *RetTy) const {
    if (RetTy->isVoid())
      return ArgInfo::getIgnore();

    if (isAggregateTypeForABI(RetTy))
      return getNaturalAlignIndirect(RetTy, /*ByVal=*/true,
                                     Opts.AllocaAddrSpace);

    return defaultClassifyScalar(RetTy);
  }

  ArgInfo classifyReturnType(const Type *RetTy) const {
    const auto *RT = dyn_cast<RecordType>(RetTy);

    if (RetTy->isVoid() || !isAggregateTypeForABI(RetTy) || getRecordArgABI(RT))
      return defaultClassifyReturnType(RetTy);

    // Ignore empty structs/unions.
    if (RT && RT->isEmpty())
      return ArgInfo::getIgnore();

    // Lower single-element structs to just return a regular value.
    if (const Type *SeltTy = isSingleElementStruct(RetTy))
      return ArgInfo::getDirect(SeltTy);

    if (RT && RT->hasFlexibleArrayMember())
      return defaultClassifyReturnType(RetTy);

    // Pack aggregates <= 4 bytes into single VGPR or pair.
    uint64_t Size = RetTy->getTypeAllocSizeInBits().getFixedValue();
    if (Size <= 16)
      return ArgInfo::getDirect(Int16Ty);

    if (Size <= 32)
      return ArgInfo::getDirect(Int32Ty);

    if (Size <= 64)
      return ArgInfo::getDirect(Int32PairTy);

    if (numRegsForType(RetTy) <= MaxNumRegsForArgsRet)
      return ArgInfo::getDirect();

    return defaultClassifyReturnType(RetTy);
  }

  ArgInfo classifyArgumentType(const Type *Ty, bool Variadic,
                               unsigned &NumRegsLeft) const {
    assert(NumRegsLeft <= MaxNumRegsForArgsRet &&
           "register estimate underflow");

    Ty = useFirstFieldIfTransparentUnion(Ty);

    if (Variadic)
      return ArgInfo::getDirect(/*T=*/nullptr, /*Offset=*/0,
                                /*Align=*/std::nullopt,
                                /*CanBeFlattened=*/false);

    if (!isAggregateTypeForABI(Ty)) {
      ArgInfo Info = defaultClassifyScalar(Ty);
      if (!Info.isIndirect())
        NumRegsLeft -= std::min<uint64_t>(numRegsForType(Ty), NumRegsLeft);

      return Info;
    }

    const auto *RT = dyn_cast<RecordType>(Ty);

    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (RecordArgABI RAA = getRecordArgABI(RT))
      return getNaturalAlignIndirect(Ty, RAA == RAA_DirectInMemory,
                                     Opts.AllocaAddrSpace);

    // Ignore empty structs/unions.
    if (RT && RT->isEmpty())
      return ArgInfo::getIgnore();

    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using getExpand(),
    // though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty))
      return ArgInfo::getDirect(SeltTy);

    if (RT && RT->hasFlexibleArrayMember())
      return defaultClassifyArgumentType(Ty);

    // Pack aggregates <= 8 bytes into single VGPR or pair.
    uint64_t Size = Ty->getTypeAllocSizeInBits().getFixedValue();
    if (Size <= 64) {
      NumRegsLeft -= std::min<uint64_t>(NumRegsLeft, divideCeil(Size, 32));

      if (Size <= 16)
        return ArgInfo::getDirect(Int16Ty);

      if (Size <= 32)
        return ArgInfo::getDirect(Int32Ty);

      // XXX: Should this be i64 instead, and should the limit increase?
      return ArgInfo::getDirect(Int32PairTy);
    }

    if (NumRegsLeft > 0) {
      uint64_t NumRegs = numRegsForType(Ty);
      if (NumRegsLeft >= NumRegs) {
        NumRegsLeft -= NumRegs;
        return ArgInfo::getDirect();
      }
    }

    // Use pass-by-reference instead of pass-by-value for struct arguments in
    // function ABI.
    return ArgInfo::getIndirectAliased(Ty->getAlignment(),
                                       Opts.PrivateAddrSpace);
  }

  /// For kernels all parameters are really passed in a special buffer. It
  /// doesn't make sense to pass anything byval, so everything must be direct.
  ArgInfo classifyKernelArgumentType(const Type *Ty) const {
    Ty = useFirstFieldIfTransparentUnion(Ty);

    // TODO: Can we omit empty structs?

    if (const Type *SeltTy = isSingleElementStruct(Ty))
      Ty = SeltTy;

    // FIXME: This doesn't apply the optimization of coercing pointers in
    // structs to global address space when using byref. This would require
    // implementing a new kind of coercion of the in-memory type when for
    // indirect arguments.
    if (isAggregateTypeForABI(Ty))
      return ArgInfo::getIndirectAliased(Ty->getAlignment(),
                                         Opts.ConstantAddrSpace);

    const Type *CoercedTy = Ty;
    if (Opts.CoerceKernelPointerArgs)
      CoercedTy = coerceKernelArgumentType(Ty);

    // If we set CanBeFlattened to true, CodeGen will expand the struct to its
    // individual elements, which confuses the Clover OpenCL backend; therefore
    // we have to set it to false here.
    return ArgInfo::getDirect(CoercedTy, /*Offset=*/0, /*Align=*/std::nullopt,
                              /*CanBeFlattened=*/false);
  }

  uint64_t getMaxDirectBitIntWidth() const { return Opts.HasInt128 ? 128 : 64; }

public:
  AMDGPUTargetInfo(TypeBuilder &TB, const AMDGPUABIOptions &Opts)
      : TB(TB), Opts(Opts),
        Int16Ty(TB.getIntegerType(16, Align(2), /*Signed=*/false)),
        Int32Ty(TB.getIntegerType(32, Align(4), /*Signed=*/false)),
        Int32PairTy(TB.getArrayType(Int32Ty, 2, 64)) {}

  void computeInfo(FunctionInfo &FI) const override {
    // A record that cannot be copied is constructed in place, so the sret
    // pointer uses the generic address space rather than the alloca one.
    if (!maybeCommonClassifyReturnType(FI, Opts.GenericAddrSpace))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    const bool IsKernel = FI.getCallingConvention() == Opts.KernelCC;
    const unsigned NumRequiredArgs = FI.getNumRequiredArgs();
    unsigned NumRegsLeft = MaxNumRegsForArgsRet;

    for (auto [Index, Arg] : enumerate(FI.arguments())) {
      Arg.Info =
          IsKernel ? classifyKernelArgumentType(Arg.ABIType)
                   : classifyArgumentType(Arg.ABIType, Index >= NumRequiredArgs,
                                          NumRegsLeft);
    }
  }
};

std::unique_ptr<TargetInfo>
createAMDGPUTargetInfo(TypeBuilder &TB, const AMDGPUABIOptions &Opts) {
  return std::make_unique<AMDGPUTargetInfo>(TB, Opts);
}

} // namespace llvm::abi
