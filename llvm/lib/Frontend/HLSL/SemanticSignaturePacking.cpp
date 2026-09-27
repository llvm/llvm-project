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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

using namespace llvm;
using namespace llvm::hlsl;

char SignaturePackingError::ID;

namespace {

// The range of rows covered by a dynamically indexable element. Only an
// element that covers multiple rows is dynamically indexable, so a single-row
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

static_assert(SemanticInterpretation::Arbitrary < SemanticInterpretation::SV &&
                  SemanticInterpretation::SV < SemanticInterpretation::SGV &&
                  SemanticInterpretation::SGV <
                      SemanticInterpretation::ClipCull &&
                  SemanticInterpretation::ClipCull <
                      SemanticInterpretation::TessFactor,
              "semantic interpretations must be in component packing order");

struct SignatureRow {
  uint8_t OccupiedColumns = 0;
  IndexedRowRange IndexedRange;
  bool IndexedRangeFixed = false;
  unsigned ComponentWidth = 0;
  dxbc::PSV::InterpolationMode InterpMode =
      dxbc::PSV::InterpolationMode::Undefined;
  SemanticInterpretation RightmostInterpretation =
      SemanticInterpretation::Arbitrary;
};

using SignatureRows = std::array<SignatureRow, MaxSignatureRows>;

struct ElementPlacement {
  unsigned Rows;
  unsigned Cols;
  unsigned ComponentWidth;
  dxbc::PSV::InterpolationMode InterpMode;
  SemanticInterpretation Interpretation;
};

struct ElementLocation {
  uint32_t Row = UnallocatedRow;
  uint8_t Col = UnallocatedCol;
};

struct OptimizedClipCullElement {
  unsigned Index;
  ElementPlacement Placement;
  ElementLocation Location;
};

// Clip/cull elements are first packed into an independent two-row grid. Each
// row used in that grid maps to a whole reserved row in the signature.
struct ClipCullState {
  std::array<SignatureRow, MaxClipCullRows> Rows;
  std::array<unsigned, MaxClipCullRows> SignatureRows = {UnallocatedRow,
                                                         UnallocatedRow};
  unsigned RowsUsed = 0;
};

enum class PackingGroup : unsigned {
  FullRegister,
  IndexedTessFactor,
  Arbitrary,
  SystemValue,
  ClipCull,
  SystemGenerated,
  NotAllocated,
};

} // namespace

static uint8_t getStartColumn(uint8_t ColumnMask) {
  assert(ColumnMask != 0 && "expected at least one occupied column");
  return countr_zero(ColumnMask);
}

static PackingGroup
getOptimizedPackingGroup(const SemanticSignatureElement &Element,
                         Triple::EnvironmentType ShaderStage, IOType IOTy) {
  const SemanticInterpretation Interpretation =
      getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
  assert((Interpretation != SemanticInterpretation::Invalid &&
          Interpretation != SemanticInterpretation::Target) &&
         "unexpected semantic interpretation for optimized packing, "
         "should have been diagnosed by Sema");

  if (Element.Cols == MaxSignatureCols &&
      (Interpretation == SemanticInterpretation::Arbitrary ||
       Interpretation == SemanticInterpretation::SV))
    return PackingGroup::FullRegister;

  if (Interpretation == SemanticInterpretation::TessFactor && Element.Rows > 1)
    return PackingGroup::IndexedTessFactor;

  switch (Interpretation) {
  case SemanticInterpretation::Arbitrary:
    return PackingGroup::Arbitrary;
  case SemanticInterpretation::SV:
  case SemanticInterpretation::TessFactor:
    return PackingGroup::SystemValue;
  case SemanticInterpretation::ClipCull:
    return PackingGroup::ClipCull;
  case SemanticInterpretation::SGV:
    return PackingGroup::SystemGenerated;
  case SemanticInterpretation::NotAllocated:
    return PackingGroup::NotAllocated;
  case SemanticInterpretation::Invalid:
  case SemanticInterpretation::Target:
    break;
  }
  llvm_unreachable("unexpected semantic interpretation for optimized packing");
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

static ElementPlacement
getElementPlacement(const SemanticSignatureElement &Element,
                    SemanticInterpretation Interpretation,
                    bool UseNative16BitTypes) {
  // Only indexed tessellation factors need the reserved last column.
  if (Interpretation == SemanticInterpretation::TessFactor && Element.Rows == 1)
    Interpretation = SemanticInterpretation::SV;
  return {Element.Rows, Element.Cols,
          getComponentWidth(Element.CompType, UseNative16BitTypes),
          Element.InterpMode, Interpretation};
}

static SemanticInterpretation
getComponentOrder(SemanticInterpretation Interpretation) {
  // Clip/cull values have system-value component ordering, but may be indexed.
  return Interpretation == SemanticInterpretation::ClipCull
             ? SemanticInterpretation::SV
             : Interpretation;
}

// Returns whether Placement may be co-packed into a Row that it covers, where
// IndexedRange is the range of rows that it is dynamically indexed over.
static bool canCoPack(const SignatureRow &Row,
                      const ElementPlacement &Placement,
                      IndexedRowRange IndexedRange) {
  const bool IsSystemValue =
      Placement.Interpretation == SemanticInterpretation::SV ||
      Placement.Interpretation == SemanticInterpretation::SGV;

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
  if (Placement.Interpretation == SemanticInterpretation::TessFactor &&
      !IndexedRange.contains(Row.IndexedRange))
    return false;

  if (Row.OccupiedColumns && Row.ComponentWidth != Placement.ComponentWidth)
    return false;
  if (Row.InterpMode != dxbc::PSV::InterpolationMode::Undefined &&
      Row.InterpMode != Placement.InterpMode)
    return false;
  // Do not append an earlier semantic category after a later one in a row.
  // Indexed tess factors are reserved in the last column, so arbitrary values
  // may still fill the columns to their left without violating that ordering.
  if (Row.OccupiedColumns &&
      getComponentOrder(Placement.Interpretation) <
          getComponentOrder(Row.RightmostInterpretation) &&
      !(Placement.Interpretation == SemanticInterpretation::Arbitrary &&
        Row.RightmostInterpretation == SemanticInterpretation::TessFactor))
    return false;
  return true;
}

// Returns the columns that Placement would occupy if it was placed at StartRow,
// or nullopt if it cannot be placed there.
static std::optional<uint8_t> canPlaceAt(ArrayRef<SignatureRow> Rows,
                                         unsigned StartRow,
                                         const ElementPlacement &Placement) {
  if (StartRow >= Rows.size() || Placement.Rows > Rows.size() - StartRow)
    return std::nullopt;

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
  if (Placement.Interpretation == SemanticInterpretation::TessFactor) {
    constexpr uint8_t LastColumn = 1U << (MaxSignatureCols - 1);
    if (Placement.Cols != 1 || (OccupiedColumns & LastColumn))
      return std::nullopt;
    return LastColumn;
  }

  for (unsigned StartCol = 0; StartCol + Placement.Cols <= MaxSignatureCols;
       ++StartCol) {
    const uint8_t ColumnMask = static_cast<uint8_t>(
        ((1U << Placement.Cols) - 1U) << static_cast<unsigned>(StartCol));
    if (!(OccupiedColumns & ColumnMask))
      return ColumnMask;
  }
  return std::nullopt;
}

static void placeRowsAt(MutableArrayRef<SignatureRow> Rows, unsigned StartRow,
                        const ElementPlacement &Placement, uint8_t ColumnMask) {
  const IndexedRowRange IndexedRange =
      IndexedRowRange::of(StartRow, Placement.Rows);
  for (unsigned ElementRow = 0; ElementRow != Placement.Rows; ++ElementRow) {
    SignatureRow &Row = Rows[StartRow + ElementRow];
    assert(!(Row.OccupiedColumns & ColumnMask) &&
           "cannot overlap signature elements");
    const uint8_t PreviousOccupiedColumns = Row.OccupiedColumns;
    if (!PreviousOccupiedColumns)
      Row.ComponentWidth = Placement.ComponentWidth;
    Row.OccupiedColumns |= ColumnMask;
    if (Row.InterpMode == dxbc::PSV::InterpolationMode::Undefined)
      Row.InterpMode = Placement.InterpMode;
    // Non-overlapping masks compare according to their rightmost set bit.
    if (!PreviousOccupiedColumns || ColumnMask > PreviousOccupiedColumns)
      Row.RightmostInterpretation = Placement.Interpretation;

    Row.IndexedRange = Row.IndexedRange.unionWith(IndexedRange);
    if (Placement.Interpretation == SemanticInterpretation::SV ||
        Placement.Interpretation == SemanticInterpretation::SGV ||
        Placement.Interpretation == SemanticInterpretation::TessFactor) {
      assert(Row.IndexedRange == IndexedRange && "incompatible index range");
      Row.IndexedRangeFixed = true;
    }
  }
}

static void placeAt(MutableArrayRef<SignatureRow> Rows, unsigned StartRow,
                    const ElementPlacement &Placement, uint8_t ColumnMask,
                    ElementLocation &Location) {
  placeRowsAt(Rows, StartRow, Placement, ColumnMask);
  Location.Row = StartRow;
  Location.Col = getStartColumn(ColumnMask);
}

static bool prefixPackElement(ElementLocation &Location,
                              MutableArrayRef<SignatureRow> Rows,
                              const ElementPlacement &Placement) {
  for (unsigned StartRow = 0; StartRow != Rows.size(); ++StartRow) {
    std::optional<uint8_t> ColumnMask = canPlaceAt(Rows, StartRow, Placement);
    if (!ColumnMask)
      continue;
    placeAt(Rows, StartRow, Placement, *ColumnMask, Location);
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
  placeRowsAt(Rows, StartRow, Reservation, *ColumnMask);
  return true;
}

static std::optional<unsigned>
reserveNextClipCullRows(MutableArrayRef<SignatureRow> Rows,
                        const ElementPlacement &Reservation) {
  for (unsigned StartRow = 0; StartRow != Rows.size(); ++StartRow)
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
    if (StartRow >= SignatureRows.size())
      return SignaturePackingError::SignatureOverflow;
    if (!reserveClipCullRows(SignatureRows, StartRow,
                             getClipCullReservation(Placement, 1)))
      return SignaturePackingError::ClipCullNotAdjacent;
    State.SignatureRows[1] = StartRow;
    return std::nullopt;
  }

  if (State.SignatureRows[0] + 1 != State.SignatureRows[1])
    return SignaturePackingError::ClipCullNotAdjacent;
  return std::nullopt;
}

static std::optional<SignaturePackingError::ErrorKind>
packClipCullElement(ElementLocation &Location,
                    MutableArrayRef<SignatureRow> SignatureRows,
                    ClipCullState &State, const ElementPlacement &Placement) {
  std::optional<uint8_t> ColumnMask;
  unsigned ClipCullStartRow = 0;
  while (ClipCullStartRow + Placement.Rows <= MaxClipCullRows) {
    ColumnMask = canPlaceAt(State.Rows, ClipCullStartRow, Placement);
    if (ColumnMask)
      break;
    ++ClipCullStartRow;
  }
  if (!ColumnMask)
    return SignaturePackingError::ClipCullOverflow;

  const unsigned NewRowsUsed = ClipCullStartRow + Placement.Rows;
  if (std::optional<SignaturePackingError::ErrorKind> Kind =
          reserveClipCullSignatureRows(SignatureRows, State, Placement,
                                       NewRowsUsed))
    return Kind;

  placeRowsAt(State.Rows, ClipCullStartRow, Placement, *ColumnMask);
  State.RowsUsed = std::max(State.RowsUsed, NewRowsUsed);
  Location.Row = State.SignatureRows[ClipCullStartRow];
  Location.Col = getStartColumn(*ColumnMask);
  return std::nullopt;
}

// Work only on scratch rows and locations. The caller commits the entire
// clip/cull phase after every stream succeeds.
static Error
packOptimizedClipCullStream(MutableArrayRef<OptimizedClipCullElement> Elements,
                            MutableArrayRef<SignatureRow> Rows) {
  if (Elements.empty())
    return Error::success();

  std::array<SignatureRow, MaxClipCullRows> LocalRows;
  bool HasIndexed = false;
  for (auto &Element : Elements) {
    if (!prefixPackElement(Element.Location, LocalRows, Element.Placement))
      return make_error<SignaturePackingError>(
          SignaturePackingError::ClipCullOverflow, Element.Index);
    HasIndexed |= Element.Placement.Rows > 1;
  }

  if (!HasIndexed) {
    // Keep single-row elements grouped into at most two rows, rather than
    // scattering individual distances across unrelated signature gaps.
    std::array<ElementLocation, MaxClipCullRows> Destinations;
    for (unsigned Row = 0; Row != MaxClipCullRows; ++Row) {
      auto First = llvm::find_if(Elements, [Row](const auto &Element) {
        return Element.Location.Row == Row;
      });
      if (First == Elements.end())
        continue;
      ElementPlacement Bundle = First->Placement;
      Bundle.Cols = popcount(LocalRows[Row].OccupiedColumns);
      if (!prefixPackElement(Destinations[Row], Rows, Bundle))
        return make_error<SignaturePackingError>(
            SignaturePackingError::SignatureOverflow, First->Index);
      // Later elements can establish an initially undefined interpolation mode.
      Rows[Destinations[Row].Row].InterpMode = LocalRows[Row].InterpMode;
    }
    for (auto &Element : Elements) {
      ElementLocation Destination = Destinations[Element.Location.Row];
      Element.Location = {
          Destination.Row,
          static_cast<uint8_t>(Destination.Col + Element.Location.Col)};
    }
    return Error::success();
  }

  // Try the whole indexed group in each adjacent pair. Preserve absolute row
  // coordinates for existing indexed ranges, even those extending beyond it.
  for (unsigned Row = 0; Row + MaxClipCullRows <= Rows.size(); ++Row) {
    SmallVector<SignatureRow, MaxSignatureRows> CandidateRows(Rows.begin(),
                                                              Rows.end());
    bool Fits = true;
    for (auto &Element : Elements) {
      Element.Location = {};
      for (unsigned Start = Row;
           Start + Element.Placement.Rows <= Row + MaxClipCullRows; ++Start) {
        if (auto Mask = canPlaceAt(CandidateRows, Start, Element.Placement)) {
          placeAt(CandidateRows, Start, Element.Placement, *Mask,
                  Element.Location);
          break;
        }
      }
      if (Element.Location.Row == UnallocatedRow) {
        Fits = false;
        break;
      }
    }
    if (Fits) {
      llvm::copy(ArrayRef(CandidateRows).slice(Row, MaxClipCullRows),
                 Rows.begin() + Row);
      return Error::success();
    }
  }
  // No element alone necessarily caused this failure; identify the group.
  return make_error<SignaturePackingError>(
      SignaturePackingError::SignatureOverflow, Elements.front().Index);
}

static Expected<unsigned>
packOptimizedClipCull(MutableArrayRef<SemanticSignatureElement> Elements,
                      ArrayRef<unsigned> Order,
                      MutableArrayRef<SignatureRows> Rows,
                      bool UseNative16BitTypes) {
  if (Order.empty())
    return 0;
  SmallVector<SmallVector<OptimizedClipCullElement>, MaxGeometryStreams>
      Streams(Rows.size());
  for (unsigned Index : Order) {
    const auto &Element = Elements[Index];
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");
    if (Element.GSStream >= Rows.size())
      return make_error<SignaturePackingError>(
          SignaturePackingError::InvalidGeometryStream, Index);
    Streams[Element.GSStream].push_back(
        {Index,
         getElementPlacement(Element, SemanticInterpretation::ClipCull,
                             UseNative16BitTypes),
         {}});
  }

  SmallVector<SignatureRows, MaxGeometryStreams> CandidateRows(Rows.begin(),
                                                               Rows.end());
  for (unsigned Stream = 0; Stream != Streams.size(); ++Stream)
    if (Error Err =
            packOptimizedClipCullStream(Streams[Stream], CandidateRows[Stream]))
      return std::move(Err);

  // Publish only after every stream succeeds. No rollback or partial-prefix
  // recovery is needed, and preceding non-clip/cull allocations remain intact.
  llvm::copy(CandidateRows, Rows.begin());
  unsigned NumRows = 0;
  for (const auto &Stream : Streams)
    for (const auto &Element : Stream) {
      Elements[Element.Index].StartRow = Element.Location.Row;
      Elements[Element.Index].StartCol = Element.Location.Col;
      NumRows =
          std::max(NumRows, Element.Location.Row + Element.Placement.Rows);
    }
  return NumRows;
}

void SignaturePackingError::log(raw_ostream &OS) const {
  switch (Kind) {
  case SignatureOverflow:
    OS << "signature elements do not fit in " << MaxSignatureRows << " rows";
    break;
  case SemanticIndexOutOfRange:
    OS << "semantic index must be less than " << MaxSignatureRows;
    break;
  case ClipCullOverflow:
    OS << "clip/cull elements do not fit in " << MaxClipCullRows << " rows";
    break;
  case ClipCullNotAdjacent:
    OS << "indexed clip/cull elements require adjacent signature rows";
    break;
  case InvalidGeometryStream:
    OS << "signature element has an invalid geometry stream: expected an index "
          "less than "
       << MaxGeometryStreams << " for geometry outputs, or zero otherwise";
    break;
  }
  OS << " (element " << ElementIndex << ")";
}

Expected<unsigned> llvm::hlsl::packSignatureStacked(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  assert(ShaderStage == Triple::Vertex && IOTy == IOType::In &&
         "stacked packing is only valid for a vertex shader input signature");

  unsigned NextRow = 0;
  for (auto &&[Index, Element] : enumerate(Elements)) {
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
           "unexpected semantic interpretation for stacked packing, should "
           "have been diagnosed by Sema");

    if (Element.Rows > MaxSignatureRows - NextRow)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));

    Element.StartRow = NextRow;
    Element.StartCol = 0;
    NextRow += Element.Rows;
  }

  return NextRow;
}

template <typename IndexRange>
static Expected<unsigned> packSignatureInOrder(
    MutableArrayRef<SemanticSignatureElement> Elements, const IndexRange &Order,
    Triple::EnvironmentType ShaderStage, IOType IOTy, bool UseNative16BitTypes,
    MutableArrayRef<SignatureRows> Rows) {
  SmallVector<ClipCullState, 1> ClipCullStates(Rows.size());
  unsigned NumRows = 0;
  for (unsigned Index : Order) {
    const SemanticSignatureElement &Element = Elements[Index];
    assert(Element.StartRow == UnallocatedRow &&
           Element.StartCol == UnallocatedCol && "already allocated?");
    assert(Element.Rows > 0 && "signature element must have at least one row");
    assert(Element.Cols > 0 && Element.Cols <= MaxSignatureCols &&
           "signature element must have between 1 and 4 columns");
    if (Element.GSStream >= Rows.size())
      return make_error<SignaturePackingError>(
          SignaturePackingError::InvalidGeometryStream,
          static_cast<unsigned>(Index));

    SemanticInterpretation Interpretation =
        getInterpretationKind(Element.SemanticKind, ShaderStage, IOTy);
    if (Interpretation == SemanticInterpretation::NotAllocated)
      continue;

    assert((Interpretation == SemanticInterpretation::Arbitrary ||
            Interpretation == SemanticInterpretation::SV ||
            Interpretation == SemanticInterpretation::SGV ||
            Interpretation == SemanticInterpretation::ClipCull ||
            Interpretation == SemanticInterpretation::TessFactor) &&
           "unexpected semantic interpretation for prefix-stable packing, "
           "should have been diagnosed by Sema");

    const ElementPlacement Placement =
        getElementPlacement(Element, Interpretation, UseNative16BitTypes);

    const unsigned StreamIndex = Element.GSStream;
    MutableArrayRef<SignatureRow> StreamRows = Rows[StreamIndex];
    ElementLocation Location;

    if (Interpretation == SemanticInterpretation::ClipCull) {
      if (std::optional<SignaturePackingError::ErrorKind> Kind =
              packClipCullElement(Location, StreamRows,
                                  ClipCullStates[StreamIndex], Placement))
        return make_error<SignaturePackingError>(*Kind, Index);
    } else if (!prefixPackElement(Location, StreamRows, Placement)) {
      return make_error<SignaturePackingError>(
          SignaturePackingError::SignatureOverflow,
          static_cast<unsigned>(Index));
    }

    Elements[Index].StartRow = Location.Row;
    Elements[Index].StartCol = Location.Col;
    NumRows = std::max(NumRows, Location.Row + Element.Rows);
  }

  return NumRows;
}

Expected<unsigned> llvm::hlsl::packSignaturePrefixStable(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy,
    bool UseNative16BitTypes) {
  assert(!(ShaderStage == Triple::Vertex && IOTy == IOType::In) &&
         !(ShaderStage == Triple::Pixel && IOTy == IOType::Out) &&
         "prefix-stable packing is not valid for vertex inputs or pixel "
         "outputs");
  const unsigned StreamCount =
      ShaderStage == Triple::Geometry && IOTy == IOType::Out
          ? MaxGeometryStreams
          : 1;
  SmallVector<SignatureRows, 1> Rows(StreamCount);
  return packSignatureInOrder(Elements, llvm::seq<unsigned>(0, Elements.size()),
                              ShaderStage, IOTy, UseNative16BitTypes, Rows);
}

Expected<unsigned> llvm::hlsl::packSignatureIndexed(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy) {
  assert(ShaderStage == Triple::Pixel && IOTy == IOType::Out &&
         "indexed packing is only valid for a pixel shader output signature");

  static_assert(MaxSignatureRows <= std::numeric_limits<uint32_t>::digits,
                "row allocation mask is too small");
  [[maybe_unused]] uint32_t AllocatedRows = 0;
  unsigned NumRows = 0;
  for (auto &&[Index, Element] : enumerate(Elements)) {
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
           "unexpected semantic interpretation for indexed packing, should "
           "have been diagnosed by Sema");
    assert(Element.Rows == 1 && Element.SemanticIndices.size() == 1 &&
           "target elements must occupy one semantic row");

    const uint32_t Row = Element.SemanticIndices.front();
    if (Row >= MaxSignatureRows)
      return make_error<SignaturePackingError>(
          SignaturePackingError::SemanticIndexOutOfRange,
          static_cast<unsigned>(Index));

    const uint32_t RowMask = uint32_t{1} << Row;
    assert(!(AllocatedRows & RowMask) &&
           "target semantic indices must be unique, verified in SemaHLSL");
    AllocatedRows |= RowMask;

    Element.StartRow = Row;
    Element.StartCol = 0;
    NumRows = std::max(NumRows, Row + 1);
  }

  return NumRows;
}

Expected<unsigned> llvm::hlsl::packSignatureOptimized(
    MutableArrayRef<SemanticSignatureElement> Elements,
    Triple::EnvironmentType ShaderStage, IOType IOTy,
    bool UseNative16BitTypes) {
  assert(!(ShaderStage == Triple::Vertex && IOTy == IOType::In) &&
         !(ShaderStage == Triple::Pixel && IOTy == IOType::Out) &&
         "optimized packing is not valid for vertex inputs or pixel outputs");

  struct SortKey {
    PackingGroup Group;
    dxbc::PSV::InterpolationMode InterpMode;
    uint32_t Rows;
    uint8_t Cols;
    uint32_t SigId;
    unsigned OriginalIndex;
  };
  SmallVector<SortKey> SortedKeys;
  SortedKeys.reserve(Elements.size());
  for (auto [Index, Element] : enumerate(Elements))
    SortedKeys.push_back({getOptimizedPackingGroup(Element, ShaderStage, IOTy),
                          Element.InterpMode, Element.Rows, Element.Cols,
                          Element.SigId, static_cast<unsigned>(Index)});

  llvm::sort(SortedKeys, [](const SortKey &Left, const SortKey &Right) {
    if (Left.Group != Right.Group)
      return Left.Group < Right.Group;
    if (Left.InterpMode != Right.InterpMode)
      return Left.InterpMode < Right.InterpMode;
    if (Left.Rows != Right.Rows)
      return Left.Rows > Right.Rows;
    if (Left.Cols != Right.Cols)
      return Left.Cols > Right.Cols;
    return Left.SigId < Right.SigId;
  });

  const unsigned StreamCount =
      ShaderStage == Triple::Geometry && IOTy == IOType::Out
          ? MaxGeometryStreams
          : 1;
  SmallVector<SignatureRows, 1> Rows(StreamCount);
  auto ClipBegin = llvm::partition_point(SortedKeys, [](const SortKey &Key) {
    return Key.Group < PackingGroup::ClipCull;
  });
  auto ClipEnd = llvm::partition_point(
      make_range(ClipBegin, SortedKeys.end()),
      [](const SortKey &Key) { return Key.Group == PackingGroup::ClipCull; });

  auto Pack = [&](auto Begin, auto End) {
    auto Order = map_range(make_range(Begin, End), [](const SortKey &Key) {
      return Key.OriginalIndex;
    });
    return packSignatureInOrder(Elements, Order, ShaderStage, IOTy,
                                UseNative16BitTypes, Rows);
  };

  Expected<unsigned> Before = Pack(SortedKeys.begin(), ClipBegin);
  if (!Before)
    return Before.takeError();
  SmallVector<unsigned> ClipCullOrder;
  for (const SortKey &Key : make_range(ClipBegin, ClipEnd))
    ClipCullOrder.push_back(Key.OriginalIndex);
  Expected<unsigned> ClipCull =
      packOptimizedClipCull(Elements, ClipCullOrder, Rows, UseNative16BitTypes);
  if (!ClipCull)
    return ClipCull.takeError();
  Expected<unsigned> After = Pack(ClipEnd, SortedKeys.end());
  if (!After)
    return After.takeError();
  return std::max({*Before, *ClipCull, *After});
}
