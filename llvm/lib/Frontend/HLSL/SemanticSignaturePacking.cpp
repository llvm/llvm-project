//===- SemanticSignaturePacking.cpp - HLSL signature packing helpers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements helpers for packing HLSL semantic signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignaturePacking.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include <array>
#include <cassert>
#include <optional>

using namespace llvm;
using namespace llvm::hlsl;

char SignaturePackingError::ID;

namespace {

struct SignatureRow {
  uint8_t OccupiedColumns = 0;
};

} // namespace

static uint8_t getStartColumn(uint8_t ColumnMask) {
  assert(ColumnMask != 0 && "expected at least one occupied column");
  return countr_zero(ColumnMask);
}

// Returns the columns that Element would occupy if it was placed at StartRow,
// or nullopt if it cannot be placed there.
static std::optional<uint8_t>
canPlaceAt(ArrayRef<SignatureRow> Rows, unsigned StartRow,
           const SemanticSignatureElement &Element) {
  if (StartRow > Rows.size() || Element.Rows > Rows.size() - StartRow)
    return std::nullopt;

  uint8_t OccupiedColumns = 0;
  for (unsigned ElementRow = 0; ElementRow != Element.Rows; ++ElementRow)
    OccupiedColumns |= Rows[StartRow + ElementRow].OccupiedColumns;

  for (unsigned StartCol = 0; StartCol + Element.Cols <= MaxSignatureCols;
       ++StartCol) {
    const uint8_t ColumnMask = static_cast<uint8_t>(
        ((1U << Element.Cols) - 1U) << static_cast<unsigned>(StartCol));
    if (!(OccupiedColumns & ColumnMask))
      return ColumnMask;
  }
  return std::nullopt;
}

static void placeAt(MutableArrayRef<SignatureRow> Rows, unsigned StartRow,
                    uint8_t ColumnMask, SemanticSignatureElement &Element) {
  for (unsigned ElementRow = 0; ElementRow != Element.Rows; ++ElementRow) {
    SignatureRow &Row = Rows[StartRow + ElementRow];
    assert(!(Row.OccupiedColumns & ColumnMask) &&
           "cannot overlap signature elements");
    Row.OccupiedColumns |= ColumnMask;
  }

  Element.StartRow = StartRow;
  Element.StartCol = getStartColumn(ColumnMask);
}

void SignaturePackingError::log(raw_ostream &OS) const {
  switch (Kind) {
  case SignatureOverflow:
    OS << "signature elements do not fit in 32 rows";
    break;
  case ClipCullOverflow:
    OS << "clip/cull elements do not fit in two rows";
    break;
  }
  OS << " (element " << ElementIndex << ")";
}

Error llvm::hlsl::packSignatureStacked(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  unsigned NextRow = 0;
  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert((Interpretation == SemanticInterpretation::Arbitrary ||
            Interpretation == SemanticInterpretation::SV ||
            Interpretation == SemanticInterpretation::SGV) &&
           "unexpected semantic interpretation for stacked packing");

    if (Element.Rows > MaxSignatureRows - NextRow)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));

    Element.StartRow = NextRow;
    Element.StartCol = 0;
    NextRow += Element.Rows;
  }

  return Error::success();
}

Error llvm::hlsl::packSignatureIndexed(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert(Interpretation == SemanticInterpretation::Target &&
           "unexpected semantic interpretation for indexed packing");
    assert(Element.Rows == 1 && Element.SemanticIndices.size() == 1 &&
           "target elements must occupy one semantic row");

    const uint32_t Row = Element.SemanticIndices.front();
    if (Row >= MaxSignatureRows)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));

    Element.StartRow = Row;
    Element.StartCol = 0;
  }

  return Error::success();
}

Error llvm::hlsl::packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy,
    bool UseNative16BitTypes) {
  (void)UseNative16BitTypes;

  std::array<SignatureRow, MaxSignatureRows> Rows = {};
  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert((Interpretation == SemanticInterpretation::Arbitrary ||
            Interpretation == SemanticInterpretation::SV ||
            Interpretation == SemanticInterpretation::SGV ||
            Interpretation == SemanticInterpretation::ClipCull ||
            Interpretation == SemanticInterpretation::TessFactor) &&
           "unexpected semantic interpretation for prefix-stable packing");

    bool Placed = false;
    for (unsigned StartRow = 0; StartRow != Rows.size(); ++StartRow) {
      std::optional<uint8_t> ColumnMask = canPlaceAt(Rows, StartRow, Element);
      if (!ColumnMask)
        continue;

      placeAt(Rows, StartRow, *ColumnMask, Element);
      Placed = true;
      break;
    }

    if (!Placed)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));
  }

  return Error::success();
}
