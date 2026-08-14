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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <optional>

using namespace llvm;
using namespace llvm::hlsl;

char SignaturePackingError::ID;

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

namespace {

// The range of rows covered by a dynamically indexable element. Only an
// element that covers multiple rows is dynamically indexable, so a single row
// element has an empty range.
struct IndexedRowRange {
  uint8_t Begin = 0;
  uint8_t End = 0;

  static IndexedRowRange of(unsigned StartRow, unsigned RowCount) {
    if (RowCount < 2)
      return {};
    return {static_cast<uint8_t>(StartRow),
            static_cast<uint8_t>(StartRow + RowCount)};
  }

  bool isEmpty() const { return Begin == End; }

  // An empty range is contained by every range.
  bool contains(IndexedRowRange Other) const {
    return Other.isEmpty() || (Begin <= Other.Begin && Other.End <= End);
  }

  IndexedRowRange unionWith(IndexedRowRange Other) const {
    if (isEmpty())
      return Other;
    if (Other.isEmpty())
      return *this;
    return {std::min(Begin, Other.Begin), std::max(End, Other.End)};
  }

  bool operator==(IndexedRowRange Other) const {
    return Begin == Other.Begin && End == Other.End;
  }
};

// Categories are ordered by the component order they must be packed in, so an
// element may only be co-packed to the right of a lesser or equal category.
enum class SemanticCategory {
  NotPacked,
  Arbitrary,
  SystemValue,
  SystemGeneratedValue,
  CullClip,
  TessFactor,
};

struct SignatureRow {
  uint8_t OccupiedColumns = 0;
  IndexedRowRange IndexedRange;
  bool IndexedRangeFixed = false;
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

// Clip/cull elements are first packed into an independent two-row grid. Each
// row used in that grid maps to a whole reserved row in the signature.
struct ClipCullState {
  std::array<SignatureRow, MaxClipCullRows> Rows;
  std::array<unsigned, MaxClipCullRows> SignatureRows = {UnallocatedRow,
                                                         UnallocatedRow};
  unsigned RowsUsed = 0;
};

} // namespace

static uint8_t getStartColumn(uint8_t ColumnMask) {
  assert(ColumnMask != 0 && "expected at least one occupied column");
  return countr_zero(ColumnMask);
}

static SemanticCategory
getSemanticCategory(dxbc::PSV::SemanticKind SemanticKind,
                    Triple::EnvironmentType ShaderStage,
                    SemanticSignatureKind SignatureKind, unsigned Rows) {
  const bool IsInput = SignatureKind == SemanticSignatureKind::Input;
  const bool IsOutput = SignatureKind == SemanticSignatureKind::Output;
  const bool IsPatchConstOrPrim =
      SignatureKind == SemanticSignatureKind::PatchConstOrPrim;

  switch (SemanticKind) {
  case dxbc::PSV::SemanticKind::Arbitrary:
    switch (ShaderStage) {
    case Triple::EnvironmentType::Vertex:
      return IsOutput ? SemanticCategory::Arbitrary
                      : SemanticCategory::NotPacked;
    case Triple::EnvironmentType::Hull:
    case Triple::EnvironmentType::Domain:
      return SemanticCategory::Arbitrary;
    case Triple::EnvironmentType::Geometry:
      return IsPatchConstOrPrim ? SemanticCategory::NotPacked
                                : SemanticCategory::Arbitrary;
    case Triple::EnvironmentType::Pixel:
      return IsInput ? SemanticCategory::Arbitrary
                     : SemanticCategory::NotPacked;
    case Triple::EnvironmentType::Mesh:
      return IsInput ? SemanticCategory::NotPacked
                     : SemanticCategory::Arbitrary;
    default:
      return SemanticCategory::NotPacked;
    }

  case dxbc::PSV::SemanticKind::Position:
    if (ShaderStage == Triple::EnvironmentType::Vertex)
      return IsInput ? SemanticCategory::Arbitrary
                     : SemanticCategory::SystemValue;
    return ShaderStage == Triple::EnvironmentType::Pixel && IsInput
               ? SemanticCategory::SystemValue
               : SemanticCategory::NotPacked;

  case dxbc::PSV::SemanticKind::VertexID:
    return ShaderStage == Triple::EnvironmentType::Vertex && IsInput
               ? SemanticCategory::SystemValue
               : SemanticCategory::NotPacked;

  case dxbc::PSV::SemanticKind::RenderTargetArrayIndex:
    return SemanticCategory::SystemValue;

  case dxbc::PSV::SemanticKind::PrimitiveID:
  case dxbc::PSV::SemanticKind::IsFrontFace:
    return SemanticCategory::SystemGeneratedValue;

  case dxbc::PSV::SemanticKind::ClipDistance:
  case dxbc::PSV::SemanticKind::CullDistance:
    switch (ShaderStage) {
    case Triple::EnvironmentType::Vertex:
      return IsInput ? SemanticCategory::Arbitrary : SemanticCategory::CullClip;
    case Triple::EnvironmentType::Hull:
    case Triple::EnvironmentType::Domain:
      return IsPatchConstOrPrim ? SemanticCategory::Arbitrary
                                : SemanticCategory::CullClip;
    case Triple::EnvironmentType::Geometry:
      return IsPatchConstOrPrim ? SemanticCategory::NotPacked
                                : SemanticCategory::CullClip;
    case Triple::EnvironmentType::Pixel:
      return IsInput ? SemanticCategory::CullClip : SemanticCategory::NotPacked;
    case Triple::EnvironmentType::Mesh:
      return IsOutput ? SemanticCategory::CullClip
                      : SemanticCategory::NotPacked;
    default:
      return SemanticCategory::NotPacked;
    }

  case dxbc::PSV::SemanticKind::TessFactor:
  case dxbc::PSV::SemanticKind::InsideTessFactor:
    // Only a tess factor that covers multiple rows is dynamically indexable
    // and needs to be reserved in the last column.
    return Rows > 1 ? SemanticCategory::TessFactor
                    : SemanticCategory::SystemValue;

  // Semantics that are not packed into a signature register, either because
  // they are accessed through a dedicated intrinsic, or because their location
  // is not assigned by this algorithm.
  case dxbc::PSV::SemanticKind::SampleIndex:
  case dxbc::PSV::SemanticKind::DispatchThreadID:
  case dxbc::PSV::SemanticKind::GroupID:
  case dxbc::PSV::SemanticKind::GroupIndex:
  case dxbc::PSV::SemanticKind::GroupThreadID:
    return SemanticCategory::NotPacked;

  // As each semantic is added to the frontend, it should be added to this
  // categorization
  default:
    return SemanticCategory::NotPacked;
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

// Returns whether Placement may be co-packed into a Row that it covers, where
// IndexedRange is the range of rows that it is dynamically indexed over.
static bool canCoPack(const SignatureRow &Row,
                      const ElementPlacement &Placement,
                      IndexedRowRange IndexedRange) {
  const bool IsSystemValue =
      Placement.Category == SemanticCategory::SystemValue ||
      Placement.Category == SemanticCategory::SystemGeneratedValue;

  // A system value is never dynamically indexable, so it cannot be placed in a
  // row that is.
  if (IsSystemValue && !Row.IndexedRange.isEmpty())
    return false;

  // A row whose indexed range is fixed only accepts elements that are indexed
  // within that range.
  if (Row.IndexedRangeFixed && !Row.IndexedRange.contains(IndexedRange))
    return false;

  // A tess factor fixes the indexed range of the rows it is reserved in, so it
  // may only extend the range that those rows already have.
  if (Placement.Category == SemanticCategory::TessFactor &&
      !IndexedRange.contains(Row.IndexedRange))
    return false;

  if (Row.OccupiedColumns && Row.ComponentWidth != Placement.ComponentWidth)
    return false;

  if (Row.InterpMode != dxbc::PSV::InterpolationMode::Undefined &&
      Row.InterpMode != Placement.InterpMode)
    return false;

  // Elements are packed in category order, so an element may not be placed to
  // the left of a greater category. An indexed tess factor is an exception as
  // it is reserved in the last column.
  if (Row.OccupiedColumns && Placement.Category < Row.RightmostCategory &&
      !(Placement.Category == SemanticCategory::Arbitrary &&
        Row.RightmostCategory == SemanticCategory::TessFactor))
    return false;

  return true;
}

// Returns the columns that Placement would occupy if it was placed at StartRow,
// or nullopt if it cannot be placed there.
static std::optional<uint8_t> canPlaceAt(ArrayRef<SignatureRow> Rows,
                                         unsigned StartRow,
                                         const ElementPlacement &Placement) {
  const IndexedRowRange IndexedRange =
      IndexedRowRange::of(StartRow, Placement.Rows);
  uint8_t OccupiedColumns = 0;
  for (unsigned ElementRow = 0; ElementRow != Placement.Rows; ++ElementRow) {
    const SignatureRow &Row = Rows[StartRow + ElementRow];
    if (!canCoPack(Row, Placement, IndexedRange))
      return std::nullopt;
    OccupiedColumns |= Row.OccupiedColumns;
  }

  // An indexed tess factor is reserved in the last column so that other
  // elements can still be co-packed into the rows that it covers.
  if (Placement.Category == SemanticCategory::TessFactor) {
    constexpr uint8_t LastColumn = 1U << (MaxSignatureCols - 1);
    if (Placement.Cols != 1 || (OccupiedColumns & LastColumn))
      return std::nullopt;
    return LastColumn;
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
  const IndexedRowRange IndexedRange =
      IndexedRowRange::of(StartRow, Placement.Rows);
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

    Row.IndexedRange = Row.IndexedRange.unionWith(IndexedRange);
    if (Placement.Category == SemanticCategory::SystemValue ||
        Placement.Category == SemanticCategory::SystemGeneratedValue ||
        Placement.Category == SemanticCategory::TessFactor) {
      assert(Row.IndexedRange == IndexedRange && "incompatible index range");
      Row.IndexedRangeFixed = true;
    }
  }
}

// Packs Element into the first rows that it fits into. Returns whether it was
// placed.
static bool packElement(SemanticSignatureElement &Element,
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
    return true;
  }
  return false;
}

// A clip/cull grid row is backed by a whole reserved signature row.
static ElementPlacement
getClipCullReservation(const ElementPlacement &Placement, unsigned RowCount) {
  ElementPlacement Reservation = Placement;
  Reservation.Rows = RowCount;
  Reservation.Cols = MaxSignatureCols;
  return Reservation;
}

static bool reserveClipCullRows(MutableArrayRef<SignatureRow> Rows,
                                unsigned StartRow,
                                const ElementPlacement &Reservation) {
  std::optional<uint8_t> ColumnMask = canPlaceAt(Rows, StartRow, Reservation);
  if (!ColumnMask)
    return false;
  placeAt(Rows, StartRow, Reservation, *ColumnMask);
  return true;
}

static std::optional<unsigned>
reserveNextClipCullRows(MutableArrayRef<SignatureRow> Rows,
                        const ElementPlacement &Reservation) {
  for (unsigned StartRow = 0; StartRow + Reservation.Rows <= Rows.size();
       ++StartRow)
    if (reserveClipCullRows(Rows, StartRow, Reservation))
      return StartRow;
  return std::nullopt;
}

// Reserves the whole signature rows that back the clip/cull grid rows that the
// element is packed into. Existing rows cannot be moved without breaking
// prefix stability, so an indexed element requires them to be adjacent.
static std::optional<SignaturePackingError::ErrorKind>
reserveClipCullSignatureRows(MutableArrayRef<SignatureRow> SignatureRows,
                             ClipCullState &State,
                             const ElementPlacement &Placement,
                             unsigned NewRowsUsed) {
  if (Placement.Rows == 1) {
    const ElementPlacement Reservation = getClipCullReservation(Placement, 1);
    for (unsigned Row = State.RowsUsed; Row < NewRowsUsed; ++Row) {
      std::optional<unsigned> StartRow =
          reserveNextClipCullRows(SignatureRows, Reservation);
      if (!StartRow)
        return SignaturePackingError::SignatureOverflow;
      State.SignatureRows[Row] = *StartRow;
    }
    return std::nullopt;
  }

  if (State.RowsUsed == 0) {
    std::optional<unsigned> StartRow = reserveNextClipCullRows(
        SignatureRows, getClipCullReservation(Placement, MaxClipCullRows));
    if (!StartRow)
      return SignaturePackingError::SignatureOverflow;
    State.SignatureRows[0] = *StartRow;
    State.SignatureRows[1] = *StartRow + 1;
    return std::nullopt;
  }

  if (State.RowsUsed == 1) {
    const unsigned StartRow = State.SignatureRows[0] + 1;
    if (StartRow >= SignatureRows.size() ||
        !reserveClipCullRows(SignatureRows, StartRow,
                             getClipCullReservation(Placement, 1)))
      return SignaturePackingError::SignatureOverflow;
    State.SignatureRows[1] = StartRow;
    return std::nullopt;
  }

  if (State.SignatureRows[0] + 1 != State.SignatureRows[1])
    return SignaturePackingError::ClipCullOverflow;
  return std::nullopt;
}

static std::optional<SignaturePackingError::ErrorKind>
packClipCullElement(SemanticSignatureElement &Element,
                    MutableArrayRef<SignatureRow> SignatureRows,
                    ClipCullState &State, const ElementPlacement &Placement) {
  std::optional<uint8_t> ColumnMask;
  unsigned ClipCullStartRow = 0;
  while (ClipCullStartRow + Placement.Rows <= MaxClipCullRows) {
    ColumnMask = canPlaceAt(State.Rows, ClipCullStartRow, Placement);
    if (ColumnMask)
      break;
    ClipCullStartRow += 1;
  }
  if (!ColumnMask)
    return SignaturePackingError::ClipCullOverflow;

  const unsigned NewRowsUsed = ClipCullStartRow + Placement.Rows;
  if (std::optional<SignaturePackingError::ErrorKind> Kind =
          reserveClipCullSignatureRows(SignatureRows, State, Placement,
                                       NewRowsUsed))
    return Kind;

  placeAt(State.Rows, ClipCullStartRow, Placement, *ColumnMask);
  State.RowsUsed = std::max(State.RowsUsed, NewRowsUsed);
  Element.StartRow = State.SignatureRows[ClipCullStartRow];
  Element.StartCol = getStartColumn(*ColumnMask);
  return std::nullopt;
}

Error llvm::hlsl::packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, SemanticSignatureKind SignatureKind,
    bool UseNative16BitTypes) {
  // Only a geometry shader output signature packs its streams independently.
  const unsigned StreamCount =
      ShaderStage == Triple::EnvironmentType::Geometry &&
              SignatureKind == SemanticSignatureKind::Output
          ? MaxGeometryStreams
          : 1;

  SmallVector<std::array<SignatureRow, MaxSignatureRows>, 1> Rows(StreamCount);
  SmallVector<ClipCullState, 1> ClipCullStates(StreamCount);

  for (const auto &[Index, Element] : enumerate(Elements)) {
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && Element.Rows <= MaxSignatureRows &&
           "signature element must have between 1 and 32 rows");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");
    assert(Element.GSStream < StreamCount &&
           "signature element has an unexpected geometry stream");

    const unsigned ComponentWidth =
        getComponentWidth(Element.CompType, UseNative16BitTypes);
    const SemanticCategory Category = getSemanticCategory(
        Element.SemanticKind, ShaderStage, SignatureKind, Element.Rows);
    const ElementPlacement Placement = {Element.Rows, Element.Cols,
                                        ComponentWidth, Element.InterpMode,
                                        Category};

    const unsigned StreamIndex = StreamCount == 1 ? 0 : Element.GSStream;
    MutableArrayRef<SignatureRow> StreamRows = Rows[StreamIndex];

    switch (Category) {
    case SemanticCategory::NotPacked:
      continue;
    case SemanticCategory::CullClip:
      if (std::optional<SignaturePackingError::ErrorKind> Kind =
              packClipCullElement(Element, StreamRows,
                                  ClipCullStates[StreamIndex], Placement))
        return make_error<SignaturePackingError>(*Kind,
                                                 static_cast<unsigned>(Index));
      continue;
    case SemanticCategory::TessFactor:
    case SemanticCategory::Arbitrary:
    case SemanticCategory::SystemValue:
    case SemanticCategory::SystemGeneratedValue:
      if (!packElement(Element, StreamRows, Placement))
        return make_error<SignaturePackingError>(
            SignaturePackingError::SignatureOverflow,
            static_cast<unsigned>(Index));
      continue;
    }
  }

  return Error::success();
}
