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

// Register layout of the sparse intrinsic operands, used by the IR verifier.
struct SPOperandLayout {
  unsigned MetadataSize;
  unsigned CompressedDataSize;
  unsigned DataSize;
};

} // namespace

// PTX limits the combined vector size of the mdata, cdata, and data operands
// of spcompress and spdecompress to 253 32-bit registers.
constexpr unsigned MaxSPOperandRegisters = 253;

static bool isValidSPElemSize(unsigned ElemSize) {
  return ElemSize == 8 || ElemSize == 16;
}

static bool isValidSPIdxSize(unsigned IdxSize) {
  return IdxSize == 2 || IdxSize == 4;
}

static bool isValidSPRepeatFactor(unsigned RepeatFactor) {
  return isPowerOf2_32(RepeatFactor) && RepeatFactor <= 64;
}

static bool isValidSPDecompressFactor(unsigned NumSrc, unsigned NumTgt) {
  switch (NumSrc) {
  case 1:
    return NumTgt == 2 || NumTgt == 4 || NumTgt == 8 || NumTgt == 16;
  case 2:
    return NumTgt == 4 || NumTgt == 8 || NumTgt == 16;
  case 4:
    return NumTgt == 8 || NumTgt == 16;
  default:
    return false;
  }
}

static std::optional<SPOperandLayout>
getSPCompressLayout(unsigned ElemSize, unsigned IdxSize,
                    unsigned RepeatFactor) {
  if (!isValidSPElemSize(ElemSize) || !isValidSPIdxSize(IdxSize) ||
      !isValidSPRepeatFactor(RepeatFactor))
    return std::nullopt;

  SPOperandLayout Layout = {divideCeil(RepeatFactor * IdxSize, ElemSize),
                            RepeatFactor, RepeatFactor * 2};
  if (Layout.MetadataSize + Layout.CompressedDataSize + Layout.DataSize >
      MaxSPOperandRegisters)
    return std::nullopt;
  return Layout;
}

static std::optional<SPOperandLayout>
getSPDecompressLayout(unsigned NumSrc, unsigned NumTgt, unsigned ElemSize,
                      unsigned IdxSize, unsigned RepeatFactor) {
  if (!isValidSPDecompressFactor(NumSrc, NumTgt) ||
      !isValidSPElemSize(ElemSize) || !isValidSPIdxSize(IdxSize) ||
      !isValidSPRepeatFactor(RepeatFactor) || NumSrc * ElemSize > 32 ||
      (IdxSize == 2 && NumTgt > 4))
    return std::nullopt;

  unsigned DataBits = NumTgt * ElemSize * RepeatFactor;
  if (DataBits < 32 || DataBits > 4096)
    return std::nullopt;

  SPOperandLayout Layout = {divideCeil(NumSrc * IdxSize * RepeatFactor, 32),
                            divideCeil(NumSrc * ElemSize * RepeatFactor, 32),
                            divideCeil(DataBits, 32)};
  if (Layout.MetadataSize + Layout.CompressedDataSize + Layout.DataSize >
      MaxSPOperandRegisters)
    return std::nullopt;
  return Layout;
}

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
  auto Layout = getSPCompressLayout(Data->ElemSize, IdxSize, RepeatFactor);
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
  auto Layout = getSPDecompressLayout(NumSrc, NumTgt, Data->ElemSize, IdxSize,
                                      RepeatFactor);
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
