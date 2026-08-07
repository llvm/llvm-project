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

// Categories are ordered by the component order they must be packed in, so an
// element may only be co-packed to the right of a lesser or equal category.
enum class SemanticCategory {
  NotPacked,
  Arbitrary,
  SystemValue,
  SystemGeneratedValue,
};

struct SignatureRow {
  uint8_t OccupiedColumns = 0;
  unsigned ComponentWidth = 0;
  dxbc::PSV::InterpolationMode InterpMode =
      dxbc::PSV::InterpolationMode::Undefined;
  SemanticCategory RightmostCategory = SemanticCategory::Arbitrary;
};

// Everything the packing rules need to know about the element that is being
// placed. It applies to every row that the element covers.
struct ElementPlacement {
  unsigned Rows;
  unsigned Cols;
  unsigned ComponentWidth;
  dxbc::PSV::InterpolationMode InterpMode;
  SemanticCategory Category;
};

} // namespace

static uint8_t getStartColumn(uint8_t ColumnMask) {
  assert(ColumnMask != 0 && "expected at least one occupied column");
  return countr_zero(ColumnMask);
}

static SemanticCategory
getSemanticCategory(dxbc::PSV::SemanticKind SemanticKind) {
  switch (SemanticKind) {
  case dxbc::PSV::SemanticKind::Arbitrary:
    return SemanticCategory::Arbitrary;
  case dxbc::PSV::SemanticKind::Position:
  case dxbc::PSV::SemanticKind::RenderTargetArrayIndex:
    return SemanticCategory::SystemValue;
  case dxbc::PSV::SemanticKind::PrimitiveID:
  case dxbc::PSV::SemanticKind::IsFrontFace:
    return SemanticCategory::SystemGeneratedValue;
  case dxbc::PSV::SemanticKind::SampleIndex:
    return SemanticCategory::NotPacked;
  default:
    return SemanticCategory::Arbitrary;
  }
}

static unsigned getComponentWidth(dxil::ElementType ComponentType,
                                  bool UseNative16BitTypes) {
  assert(ComponentType != dxil::ElementType::I64 &&
         ComponentType != dxil::ElementType::U64 &&
         ComponentType != dxil::ElementType::F64 &&
         ComponentType != dxil::ElementType::SNormF64 &&
         ComponentType != dxil::ElementType::UNormF64 &&
         "64-bit types cannot be used in a signature");

  switch (ComponentType) {
  case dxil::ElementType::F16:
  case dxil::ElementType::I16:
  case dxil::ElementType::U16:
  case dxil::ElementType::SNormF16:
  case dxil::ElementType::UNormF16:
    // Without native 16-bit types these are min-precision types that occupy a
    // whole 32-bit component.
    return UseNative16BitTypes ? 16 : 32;
  default:
    // A boolean is loaded and stored as a 32-bit value.
    return 32;
  }
}

// Returns whether Placement may be co-packed into a Row that it covers.
static bool canCoPack(const SignatureRow &Row,
                      const ElementPlacement &Placement) {
  if (Row.OccupiedColumns && Row.ComponentWidth != Placement.ComponentWidth)
    return false;
  if (Row.InterpMode != dxbc::PSV::InterpolationMode::Undefined &&
      Row.InterpMode != Placement.InterpMode)
    return false;
  if (Row.OccupiedColumns && Placement.Category < Row.RightmostCategory)
    return false;
  return true;
}

// Returns the columns that Placement would occupy if it was placed at StartRow,
// or nullopt if it cannot be placed there.
static std::optional<uint8_t> canPlaceAt(ArrayRef<SignatureRow> Rows,
                                         unsigned StartRow,
                                         const ElementPlacement &Placement) {
  uint8_t OccupiedColumns = 0;
  for (unsigned ElementRow = 0; ElementRow != Placement.Rows; ++ElementRow) {
    const SignatureRow &Row = Rows[StartRow + ElementRow];
    if (!canCoPack(Row, Placement))
      return std::nullopt;
    OccupiedColumns |= Row.OccupiedColumns;
  }

  for (unsigned StartCol = 0; StartCol + Placement.Cols <= MaxSignatureCols;
       ++StartCol) {
    const uint8_t ColumnMask =
        static_cast<uint8_t>(((1U << Placement.Cols) - 1U) << StartCol);
    if (!(OccupiedColumns & ColumnMask))
      return ColumnMask;
  }
  return std::nullopt;
}

static void placeAt(MutableArrayRef<SignatureRow> Rows, unsigned StartRow,
                    const ElementPlacement &Placement, uint8_t ColumnMask) {
  for (unsigned ElementRow = 0; ElementRow != Placement.Rows; ++ElementRow) {
    SignatureRow &Row = Rows[StartRow + ElementRow];
    const uint8_t PreviousOccupiedColumns = Row.OccupiedColumns;
    if (!PreviousOccupiedColumns)
      Row.ComponentWidth = Placement.ComponentWidth;
    Row.OccupiedColumns |= ColumnMask;
    if (Row.InterpMode == dxbc::PSV::InterpolationMode::Undefined)
      Row.InterpMode = Placement.InterpMode;
    // Non-overlapping masks compare according to their rightmost set bit.
    if (!PreviousOccupiedColumns || ColumnMask > PreviousOccupiedColumns)
      Row.RightmostCategory = Placement.Category;
  }
}

// Packs Element into the first rows that it fits into.
static Error packElement(SemanticSignatureElement &Element,
                         MutableArrayRef<SignatureRow> Rows,
                         const ElementPlacement &Placement) {
  for (unsigned StartRow = 0; StartRow + Placement.Rows <= Rows.size();
       ++StartRow) {
    std::optional<uint8_t> ColumnMask = canPlaceAt(Rows, StartRow, Placement);
    if (!ColumnMask)
      continue;
    placeAt(Rows, StartRow, Placement, *ColumnMask);
    Element.StartRow = StartRow;
    Element.StartCol = getStartColumn(*ColumnMask);
    return Error::success();
  }
  return createStringError(SignatureOverflowMessage);
}

Error llvm::hlsl::packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements, Triple::EnvironmentType,
    SemanticSignatureKind, bool UseNative16BitTypes) {
  std::array<SignatureRow, MaxSignatureRows> Rows;
  for (SemanticSignatureElement &Element : Elements) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && Element.Rows <= MaxSignatureRows &&
           "signature element must have between 1 and 32 rows");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");

    const unsigned ComponentWidth =
        getComponentWidth(Element.CompType, UseNative16BitTypes);
    const SemanticCategory Category = getSemanticCategory(Element.SemanticKind);
    if (Category == SemanticCategory::NotPacked)
      continue;
    const ElementPlacement Placement = {Element.Rows, Element.Cols,
                                        ComponentWidth, Element.InterpMode,
                                        Category};
    if (Error E = packElement(Element, Rows, Placement))
      return E;
  }

  return Error::success();
}
