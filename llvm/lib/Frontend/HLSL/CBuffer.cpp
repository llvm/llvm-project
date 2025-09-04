//===- CBuffer.cpp - HLSL constant buffer handling ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/CBuffer.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::hlsl;

static bool isPadding(Type *Ty) {
  // TODO: We should use an explicit padding type rather than array of i8
  if (const auto *ATy = dyn_cast<ArrayType>(Ty))
    if (const auto *ITy = dyn_cast<IntegerType>(ATy->getElementType()))
      return ITy->getBitWidth() == 8;
  return false;
}

static SmallVector<size_t> getMemberOffsets(const DataLayout &DL,
                                            GlobalVariable *Handle) {
  SmallVector<size_t> Offsets;

  auto *HandleTy = cast<TargetExtType>(Handle->getValueType());
  assert(HandleTy->getName().ends_with(".CBuffer") && "Not a cbuffer type");
  assert(HandleTy->getNumTypeParameters() == 1 && "Expected layout type");
  auto *LayoutTy = cast<StructType>(HandleTy->getTypeParameter(0));

  const StructLayout *SL = DL.getStructLayout(LayoutTy);
  for (int I = 0, E = LayoutTy->getNumElements(); I < E; ++I)
    if (!isPadding(LayoutTy->getElementType(I)))
      Offsets.push_back(SL->getElementOffset(I));

  return Offsets;
}

std::optional<CBufferMetadata> CBufferMetadata::get(Module &M) {
  NamedMDNode *CBufMD = M.getNamedMetadata("hlsl.cbs");
  if (!CBufMD)
    return std::nullopt;

  std::optional<CBufferMetadata> Result({CBufMD});

  for (const MDNode *MD : CBufMD->operands()) {
    assert(MD->getNumOperands() && "Invalid cbuffer metadata");

    auto *Handle = cast<GlobalVariable>(
        cast<ValueAsMetadata>(MD->getOperand(0))->getValue());
    CBufferMapping &Mapping = Result->Mappings.emplace_back(Handle);

    SmallVector<size_t> MemberOffsets =
        getMemberOffsets(M.getDataLayout(), Handle);

    for (int I = 1, E = MD->getNumOperands(); I < E; ++I) {
      Metadata *OpMD = MD->getOperand(I);
      // Some members may be null if they've been optimized out.
      if (!OpMD)
        continue;
      auto *V = cast<GlobalVariable>(cast<ValueAsMetadata>(OpMD)->getValue());
      Mapping.Members.emplace_back(V, MemberOffsets[I - 1]);
    }
  }

  return Result;
}

void CBufferMetadata::eraseFromModule() {
  // Remove the cbs named metadata
  MD->eraseFromParent();
}

APInt hlsl::translateCBufArrayOffset(const DataLayout &DL, APInt Offset,
                                     ArrayType *Ty) {
  int64_t TypeSize = DL.getTypeSizeInBits(Ty->getElementType()) / 8;
  int64_t RoundUp = alignTo(TypeSize, Align(CBufferRowSizeInBytes));
  return Offset.udiv(TypeSize) * RoundUp;
}
