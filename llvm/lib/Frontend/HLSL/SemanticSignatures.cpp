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
#include "llvm/IR/Metadata.h"

using namespace llvm;
using namespace llvm::hlsl;

Expected<SemanticSignatureElement>
SemanticSignatureElement::fromMetadata(const MDNode *Node) {
  return SemanticSignatureElement{};
}

MDNode *SemanticSignatureElement::toMetadata(LLVMContext &Ctx) const {
  return MDNode::get(Ctx, {});
}
