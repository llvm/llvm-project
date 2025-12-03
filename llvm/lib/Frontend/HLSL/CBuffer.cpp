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

static SmallVector<size_t>
getMemberOffsets(const DataLayout &DL, GlobalVariable *Handle,
                 llvm::function_ref<bool(Type *)> IsPadding) {
  SmallVector<size_t> Offsets;

  auto *HandleTy = cast<TargetExtType>(Handle->getValueType());
  assert((HandleTy->getName().ends_with(".CBuffer") ||
          HandleTy->getName() == "spirv.VulkanBuffer") &&
         "Not a cbuffer type");
  assert(HandleTy->getNumTypeParameters() == 1 && "Expected layout type");
  auto *LayoutTy = cast<StructType>(HandleTy->getTypeParameter(0));

  const StructLayout *SL = DL.getStructLayout(LayoutTy);
  for (int I = 0, E = LayoutTy->getNumElements(); I < E; ++I)
    if (!IsPadding(LayoutTy->getElementType(I)))
      Offsets.push_back(SL->getElementOffset(I));

  return Offsets;
}

std::optional<CBufferMetadata>
CBufferMetadata::get(Module &M, llvm::function_ref<bool(Type *)> IsPadding) {
  NamedMDNode *CBufMD = M.getNamedMetadata("hlsl.cbs");
  if (!CBufMD)
    return std::nullopt;

  std::optional<CBufferMetadata> Result({CBufMD});

  for (const MDNode *MD : CBufMD->operands()) {
    assert(MD->getNumOperands() && "Invalid cbuffer metadata");

    // For an unused cbuffer, the handle may have been optimized out
    Metadata *OpMD = MD->getOperand(0);
    if (!OpMD)
      continue;

    auto *Handle =
        cast<GlobalVariable>(cast<ValueAsMetadata>(OpMD)->getValue());
    CBufferMapping &Mapping = Result->Mappings.emplace_back(Handle);

    SmallVector<size_t> MemberOffsets =
        getMemberOffsets(M.getDataLayout(), Handle, IsPadding);

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
