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
#include "llvm/ADT/bit.h"
#include <array>
#include <cassert>
#include <optional>

using namespace llvm;
using namespace llvm::hlsl;

static constexpr StringLiteral SignatureOverflowMessage =
    "signature elements do not fit in 32 rows";

namespace {

struct SignatureRow {
  uint8_t OccupiedColumns = 0;
};

} // namespace

static uint8_t getStartColumn(uint8_t ColumnMask) {
  assert(ColumnMask != 0 && "expected at least one occupied column");
  return countr_zero(ColumnMask);
}

static std::optional<uint8_t> canPlaceElement(ArrayRef<SignatureRow> Rows,
                                              unsigned StartRow,
                                              unsigned RowCount,
                                              unsigned ColumnCount) {
  uint8_t OccupiedColumns = 0;
  for (unsigned ElementRow = 0; ElementRow != RowCount; ++ElementRow)
    OccupiedColumns |= Rows[StartRow + ElementRow].OccupiedColumns;

  for (unsigned StartCol = 0; StartCol + ColumnCount <= MaxSignatureCols;
       ++StartCol) {
    const uint8_t ColumnMask =
        static_cast<uint8_t>(((1U << ColumnCount) - 1U) << StartCol);
    if (!(OccupiedColumns & ColumnMask))
      return ColumnMask;
  }
  return std::nullopt;
}

static void placeElement(MutableArrayRef<SignatureRow> Rows, unsigned StartRow,
                         unsigned RowCount, uint8_t ColumnMask) {
  for (unsigned ElementRow = 0; ElementRow != RowCount; ++ElementRow)
    Rows[StartRow + ElementRow].OccupiedColumns |= ColumnMask;
}

Error llvm::hlsl::packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements, Triple::EnvironmentType,
    SemanticSignatureKind, bool UseNative16BitTypes) {
  (void)UseNative16BitTypes;

  std::array<SignatureRow, MaxSignatureRows> Rows{};
  for (SemanticSignatureElement &Element : Elements) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && Element.Rows <= MaxSignatureRows &&
           "signature element must have between 1 and 32 rows");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    bool Placed = false;
    for (unsigned StartRow = 0; StartRow + Element.Rows <= MaxSignatureRows;
         ++StartRow) {
      std::optional<uint8_t> ColumnMask =
          canPlaceElement(Rows, StartRow, Element.Rows, Element.Cols);
      if (!ColumnMask)
        continue;

      placeElement(Rows, StartRow, Element.Rows, *ColumnMask);
      Element.StartRow = StartRow;
      Element.StartCol = getStartColumn(*ColumnMask);
      Placed = true;
      break;
    }

    if (!Placed)
      return createStringError(SignatureOverflowMessage);
  }

  return Error::success();
}
