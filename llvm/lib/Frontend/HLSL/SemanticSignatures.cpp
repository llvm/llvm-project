//===- SemanticSignatures.cpp - HLSL Semantic Signature helpers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements a library for working with HLSL shader input and
/// output semantic signatures and their DirectX metadata representation.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace llvm::hlsl;

Expected<SemanticSignatureElement>
SemanticSignatureElement::fromMetadata(const MDNode *Node) {
  return SemanticSignatureElement{};
}

MDNode *SemanticSignatureElement::toMetadata(LLVMContext &Ctx) const {
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  auto GetI32 = [&](uint32_t Val) -> Metadata * {
    return ConstantAsMetadata::get(ConstantInt::get(I32Ty, Val));
  };
  auto GetI8 = [&](uint8_t Val) -> Metadata * {
    return ConstantAsMetadata::get(ConstantInt::get(I8Ty, Val));
  };

  SmallVector<Metadata *> IndexOps;
  for (uint32_t Index : SemanticIndices)
    IndexOps.push_back(GetI32(Index));

  return MDNode::get(
      Ctx, {GetI32(SigId), MDString::get(Ctx, SemanticName),
            GetI32(static_cast<uint32_t>(CompType)),
            GetI32(static_cast<uint32_t>(SemanticKind)),
            MDNode::get(Ctx, IndexOps),
            GetI32(static_cast<uint32_t>(InterpMode)), GetI32(Rows),
            GetI8(Cols), GetI32(StartRow), GetI8(StartCol), GetI8(UsageMask),
            GetI8(DynIndexMask), GetI32(GSStream)});
}
