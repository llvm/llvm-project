//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains NVVM-specific IR verification logic. These checks are
/// always compiled and linked as part of LLVMCore.
///
//===----------------------------------------------------------------------===//

#include "VerifierInternal.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace llvm;

#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      VS.CheckFailed(__VA_ARGS__);                                             \
      return;                                                                  \
    }                                                                          \
  } while (false)

namespace {

struct SPVectorInfo {
  unsigned ElemSize;
  unsigned NumElements;
  unsigned NumRegisters;
};

} // namespace

static std::optional<SPVectorInfo> getSPVectorInfo(Type *Ty) {
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  if (!VT)
    return std::nullopt;
  auto *ElemTy = dyn_cast<IntegerType>(VT->getElementType());
  if (!ElemTy)
    return std::nullopt;
  unsigned ElemSize = ElemTy->getBitWidth();
  if (ElemSize != 8 && ElemSize != 16)
    return std::nullopt;
  unsigned NumElements = VT->getNumElements();
  return SPVectorInfo{ElemSize, NumElements,
                      divideCeil(NumElements, 32 / ElemSize)};
}

static std::optional<unsigned> getSPMetadataRegisters(Type *Ty) {
  if (Ty->isIntegerTy(32))
    return 1;
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  if (!VT || !VT->getElementType()->isIntegerTy(32))
    return std::nullopt;
  return VT->getNumElements();
}

static void verifySPCompress(VerifierSupport &VS, CallBase &Call) {
  auto *ResultTy = dyn_cast<StructType>(Call.getType());
  Check(ResultTy && ResultTy->getNumElements() == 2,
        "invalid llvm.nvvm.spcompress result type", &Call);

  auto MDataRegs = getSPMetadataRegisters(ResultTy->getElementType(0));
  auto CData = getSPVectorInfo(ResultTy->getElementType(1));
  auto Data = getSPVectorInfo(Call.getArgOperand(0)->getType());
  Check(MDataRegs && CData && Data && CData->ElemSize == Data->ElemSize,
        "invalid llvm.nvvm.spcompress operand or result type", &Call);

  unsigned IdxSize = cast<ConstantInt>(Call.getArgOperand(2))->getZExtValue();
  unsigned NumTgt = cast<ConstantInt>(Call.getArgOperand(3))->getZExtValue();
  Check(NumTgt == 4 && Data->NumElements % NumTgt == 0 &&
            CData->NumElements == 2 * (Data->NumElements / NumTgt) &&
            Data->NumRegisters % 2 == 0,
        "invalid llvm.nvvm.spcompress layout", &Call);

  unsigned RepeatFactor = Data->NumRegisters / 2;
  auto Layout =
      nvvm::getSPCompressLayout(Data->ElemSize, IdxSize, RepeatFactor);
  Check(Layout && *MDataRegs == Layout->MetadataSize &&
            CData->NumRegisters == Layout->CompressedDataSize &&
            Data->NumRegisters == Layout->DataSize,
        "invalid llvm.nvvm.spcompress layout", &Call);
}

static void verifySPDecompress(VerifierSupport &VS, CallBase &Call) {
  auto Data = getSPVectorInfo(Call.getType());
  auto MDataRegs = getSPMetadataRegisters(Call.getArgOperand(0)->getType());
  auto CData = getSPVectorInfo(Call.getArgOperand(1)->getType());
  Check(Data && MDataRegs && CData && Data->ElemSize == CData->ElemSize,
        "invalid llvm.nvvm.spdecompress operand or result type", &Call);

  unsigned IdxSize = cast<ConstantInt>(Call.getArgOperand(2))->getZExtValue();
  unsigned NumTgt = cast<ConstantInt>(Call.getArgOperand(3))->getZExtValue();
  Check(NumTgt != 0 && Data->NumElements % NumTgt == 0,
        "invalid llvm.nvvm.spdecompress layout", &Call);

  unsigned RepeatFactor = Data->NumElements / NumTgt;
  Check(RepeatFactor != 0 && CData->NumElements % RepeatFactor == 0,
        "invalid llvm.nvvm.spdecompress layout", &Call);

  unsigned NumSrc = CData->NumElements / RepeatFactor;
  auto Layout = nvvm::getSPDecompressLayout(NumSrc, NumTgt, Data->ElemSize,
                                            IdxSize, RepeatFactor);
  Check(Layout && *MDataRegs == Layout->MetadataSize &&
            CData->NumRegisters == Layout->CompressedDataSize &&
            Data->NumRegisters == Layout->DataSize,
        "invalid llvm.nvvm.spdecompress layout", &Call);
}

void llvm::verifyNVVMIntrinsicCall(VerifierSupport &VS, Intrinsic::ID ID,
                                   CallBase &Call) {
  switch (ID) {
  default:
    return;
  case Intrinsic::nvvm_spcompress:
    verifySPCompress(VS, Call);
    return;
  case Intrinsic::nvvm_spdecompress:
    verifySPDecompress(VS, Call);
    return;
  }
}

#undef Check
