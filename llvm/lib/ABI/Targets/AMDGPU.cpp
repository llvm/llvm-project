//===- AMDGPU.cpp - AMDGPU ABI Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared by amdgcn and the AMDGCN-flavoured SPIR-V that lowers to it.
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

namespace llvm::abi {

static const PointerType *getGlobalsPointerType(TypeBuilder &TB,
                                                const DataLayout &DL) {
  const unsigned AS = DL.getDefaultGlobalsAddressSpace();
  return TB.getPointerType(DL.getPointerSizeInBits(AS),
                           DL.getPointerABIAlignment(AS), AS);
}

class AMDGPUTargetInfo : public TargetInfo {
private:
  /// Registers available for arguments and return values, in 32-bit units.
  static constexpr unsigned MaxNumRegsForArgsRet = 16;

  static constexpr uint64_t MaxDirectBitIntWidth = 128;

  const DataLayout &DL;
  AMDGPUABIOptions Opts;
  const IntegerType *Int16Ty;
  const IntegerType *Int32Ty;
  const ArrayType *Int32PairTy;
  const PointerType *KernelArgPtrTy;

  uint64_t numRegsForType(const Type *Ty) const {
    if (const auto *VT = dyn_cast<VectorType>(Ty)) {
      // Compute from the number of elements. The reported size is based on the
      // in-memory size, which includes the padding 4th element for 3-vectors.
      uint64_t EltSize =
          VT->getElementType()->getTypeAllocSizeInBits().getFixedValue();
      uint64_t NumElts = VT->getNumElements().getFixedValue();

      // 16-bit element vectors should be passed as packed.
      if (EltSize == 16)
        return divideCeil(NumElts, 2);

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

  /// Returns the element type if \p Ty is a struct wrapping exactly one
  /// non-empty element with no padding beyond it, else nullptr. Single-element
  /// arrays are looked through.
  // TODO: X86 has its own copy of this. Hoist both into TargetInfo.
  const Type *isSingleElementStruct(const Type *Ty) const {
    const auto *RT = dyn_cast<RecordType>(Ty);
    if (!RT)
      return nullptr;

    if (RT->hasFlexibleArrayMember())
      return nullptr;

    const Type *Found = nullptr;

    for (const FieldInfo &Base : RT->getBaseClasses()) {
      const auto *BaseRT = dyn_cast<RecordType>(Base.FieldType);
      if (!BaseRT || BaseRT->isEmpty())
        continue;

      if (Found)
        return nullptr;

      Found = isSingleElementStruct(Base.FieldType);
      if (!Found)
        return nullptr;
    }

    for (const FieldInfo &Field : RT->getFields()) {
      if (Field.isEmpty())
        continue;

      if (Found)
        return nullptr;

      const Type *FieldTy = Field.FieldType;

      // Treat single element arrays as the element.
      while (const auto *AT = dyn_cast<ArrayType>(FieldTy)) {
        if (AT->getNumElements() != 1)
          break;
        FieldTy = AT->getElementType();
      }

      if (!isAggregateTypeForABI(FieldTy)) {
        Found = FieldTy;
      } else {
        Found = isSingleElementStruct(FieldTy);
        if (!Found)
          return nullptr;
      }
    }

    // Padding beyond the element disqualifies the struct. Compare in-memory
    // sizes, not raw bit widths, so an element with trailing padding of its own
    // still matches the record wrapping it.
    if (Found && Found->getTypeAllocSize() != Ty->getTypeAllocSize())
      return nullptr;

    return Found;
  }

  /// Coerce a scalar pointer argument from the generic address space to the
  /// one kernel arguments must use.
  const Type *coerceKernelArgumentType(const Type *Ty) const {
    if (!Opts.CoerceKernelPointerArgs)
      return Ty;

    // Classic CodeGen coerces the lowered type, which has no atomic wrapper.
    const Type *Unwrapped = Ty;
    if (const auto *AT = dyn_cast<AtomicType>(Ty))
      Unwrapped = AT->getValueType();

    const auto *PtrTy = dyn_cast<PointerType>(Unwrapped);
    if (PtrTy && PtrTy->getAddrSpace() == Opts.GenericAddrSpace)
      return KernelArgPtrTy;
    return Ty;
  }

  /// Pack an aggregate of \p Size bits into a VGPR or a VGPR pair.
  const Type *getRegisterCoerceType(uint64_t Size) const {
    if (Size <= 16)
      return Int16Ty;
    if (Size <= 32)
      return Int32Ty;
    return Int32PairTy;
  }

  /// The non-aggregate tail shared by the two Clang DefaultABIInfo rules.
  ArgInfo defaultClassifyScalar(const Type *Ty) const {
    if (const auto *IntTy = dyn_cast<IntegerType>(Ty)) {
      if (IntTy->isBitInt() &&
          IntTy->getSizeInBits().getFixedValue() > MaxDirectBitIntWidth)
        return getNaturalAlignIndirect(Ty, /*ByVal=*/true,
                                       DL.getAllocaAddrSpace());

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
                                     DL.getAllocaAddrSpace());

    return defaultClassifyScalar(Ty);
  }

  ArgInfo defaultClassifyReturnType(const Type *RetTy) const {
    if (RetTy->isVoid())
      return ArgInfo::getIgnore();

    if (isAggregateTypeForABI(RetTy))
      return getNaturalAlignIndirect(RetTy, /*ByVal=*/true,
                                     DL.getAllocaAddrSpace());

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
    if (Size <= 64)
      return ArgInfo::getDirect(getRegisterCoerceType(Size));

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
                                     DL.getAllocaAddrSpace());

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

      // XXX: Should the 64-bit case be i64 instead, and should the limit
      // increase?
      return ArgInfo::getDirect(getRegisterCoerceType(Size));
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
                                       DL.getAllocaAddrSpace());
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

    // If we set CanBeFlattened to true, CodeGen will expand the struct to its
    // individual elements, which confuses the Clover OpenCL backend; therefore
    // we have to set it to false here.
    return ArgInfo::getDirect(coerceKernelArgumentType(Ty), /*Offset=*/0,
                              /*Align=*/std::nullopt,
                              /*CanBeFlattened=*/false);
  }

protected:
  /// A record that cannot be copied is constructed in place, so the sret
  /// pointer uses the generic address space rather than the alloca one.
  unsigned getSRetAddrSpace(const RecordType *RT) const override {
    return Opts.GenericAddrSpace;
  }

public:
  AMDGPUTargetInfo(TypeBuilder &TB, const DataLayout &DL,
                   const AMDGPUABIOptions &Opts)
      : TargetInfo(TB), DL(DL), Opts(Opts),
        Int16Ty(TB.getIntegerType(16, Align(2), /*Signed=*/false)),
        Int32Ty(TB.getIntegerType(32, Align(4), /*Signed=*/false)),
        Int32PairTy(TB.getArrayType(Int32Ty, 2, 64)),
        KernelArgPtrTy(getGlobalsPointerType(TB, DL)) {}

  void computeInfo(FunctionInfo &FI) const override {
    if (!maybeCommonClassifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

    const CallingConv::ID CC = FI.getCallingConvention();
    const bool IsKernel =
        CC == CallingConv::AMDGPU_KERNEL || CC == CallingConv::SPIR_KERNEL;
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
createAMDGPUTargetInfo(TypeBuilder &TB, const DataLayout &DL,
                       const AMDGPUABIOptions &Opts) {
  return std::make_unique<AMDGPUTargetInfo>(TB, DL, Opts);
}

} // namespace llvm::abi
