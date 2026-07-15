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
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::hlsl;

namespace {
// The fixed number of operands in a signature element node
constexpr unsigned NumElementOperands = 13;

// Inclusive upper bounds of the operand enums
constexpr uint32_t MaxCompType =
    static_cast<uint32_t>(dxil::ElementType::PackedU8x32);
constexpr uint32_t MaxSemanticKind =
    static_cast<uint32_t>(dxbc::PSV::SemanticKind::Invalid);
constexpr uint32_t MaxInterpMode =
    static_cast<uint32_t>(dxbc::PSV::InterpolationMode::Invalid);

Error makeError(const Twine &Msg) {
  return createStringError(inconvertibleErrorCode(), Msg);
}

Expected<uint64_t> extractInt(const MDNode *Node, unsigned OpId) {
  auto *CI = mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(OpId));
  if (!CI)
    return makeError("expected integer operand " + Twine(OpId));
  return CI->getZExtValue();
}
} // namespace

Expected<SemanticSignatureElement>
SemanticSignatureElement::fromMetadata(const MDNode *Node) {
  if (!Node)
    return makeError("signature element node is null");
  if (Node->getNumOperands() != NumElementOperands)
    return makeError("signature element node has wrong number of operands");

  SemanticSignatureElement Elem;

  Expected<uint64_t> SigId = extractInt(Node, 0);
  if (!SigId)
    return SigId.takeError();
  Elem.SigId = *SigId;

  auto *Name = dyn_cast<MDString>(Node->getOperand(1));
  if (!Name)
    return makeError("expected semantic name string");
  Elem.SemanticName = Name->getString();

  Expected<uint64_t> CompType = extractInt(Node, 2);
  if (!CompType)
    return CompType.takeError();
  if (*CompType > MaxCompType)
    return makeError("invalid component type");
  Elem.CompType = static_cast<dxil::ElementType>(*CompType);

  Expected<uint64_t> SemanticKind = extractInt(Node, 3);
  if (!SemanticKind)
    return SemanticKind.takeError();
  if (*SemanticKind > MaxSemanticKind)
    return makeError("invalid semantic kind");
  Elem.SemanticKind = static_cast<dxbc::PSV::SemanticKind>(*SemanticKind);

  auto *Indices = dyn_cast<MDNode>(Node->getOperand(4));
  if (!Indices)
    return makeError("expected semantic indices node");
  for (unsigned I = 0, E = Indices->getNumOperands(); I != E; ++I) {
    Expected<uint64_t> Index = extractInt(Indices, I);
    if (!Index)
      return Index.takeError();
    Elem.SemanticIndices.push_back(*Index);
  }

  Expected<uint64_t> InterpMode = extractInt(Node, 5);
  if (!InterpMode)
    return InterpMode.takeError();
  if (*InterpMode > MaxInterpMode)
    return makeError("invalid interpolation mode");
  Elem.InterpMode = static_cast<dxbc::PSV::InterpolationMode>(*InterpMode);

  Expected<uint64_t> Rows = extractInt(Node, 6);
  if (!Rows)
    return Rows.takeError();
  Elem.Rows = *Rows;

  Expected<uint64_t> Cols = extractInt(Node, 7);
  if (!Cols)
    return Cols.takeError();
  if (*Cols < 1 || *Cols > 4)
    return makeError("number of components per row must be within 1-4");
  Elem.Cols = *Cols;

  Expected<uint64_t> StartRow = extractInt(Node, 8);
  if (!StartRow)
    return StartRow.takeError();
  Elem.StartRow = *StartRow;

  Expected<uint64_t> StartCol = extractInt(Node, 9);
  if (!StartCol)
    return StartCol.takeError();
  if (*StartCol > 3 && *StartCol != UnallocatedCol)
    return makeError("start column must be within 0-3 or unallocated");
  Elem.StartCol = *StartCol;

  // The row/col sentinels are always set together
  if ((Elem.StartRow == UnallocatedRow) != (Elem.StartCol == UnallocatedCol))
    return makeError("start row and column sentinels must be set together");

  Expected<uint64_t> UsageMask = extractInt(Node, 10);
  if (!UsageMask)
    return UsageMask.takeError();
  if (*UsageMask > 0xF)
    return makeError("usage mask must be a 4-bit value");
  Elem.UsageMask = *UsageMask;

  Expected<uint64_t> DynIndexMask = extractInt(Node, 11);
  if (!DynIndexMask)
    return DynIndexMask.takeError();
  if (*DynIndexMask > 0xF)
    return makeError("dynamic index mask must be a 4-bit value");
  Elem.DynIndexMask = *DynIndexMask;

  Expected<uint64_t> GSStream = extractInt(Node, 12);
  if (!GSStream)
    return GSStream.takeError();
  if (*GSStream > 3)
    return makeError("geometry shader stream index must be within 0-3");
  Elem.GSStream = *GSStream;

  if (Elem.SemanticIndices.size() != Elem.Rows)
    return makeError("number of semantic indices must equal the number of rows");

  return Elem;
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
