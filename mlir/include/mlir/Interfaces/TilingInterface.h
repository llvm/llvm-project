//===- TilingInterface.h - Interface for tiling operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the TilingInterface defined in
// `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_TILINGINTERFACE_H_
#define MLIR_INTERFACES_TILINGINTERFACE_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

/// Container for result values of tiling.
/// - `tiledOps` contains operations created by the tiling implementation that
///   are returned to the caller for further transformations.
/// - `tiledValues` contains the tiled value corresponding to the result of the
///   untiled operation.
/// - `generatedSlices` contains the list of slices that are generated during
///   tiling. These slices can be used for fusing producers.
struct TilingResult {
  SmallVector<Operation *> tiledOps;
  SmallVector<Value> tiledValues;
  SmallVector<Operation *> generatedSlices;
};

/// Tiling can be thought of as splitting a dimension into 2 and
/// materializing the outer dimension as a loop:
///
/// op[original] -> op[original / x, x] -> loop[original] { op[x] }
///
/// For parallel dimensions, the split can only happen in one way, with both
/// dimensions being parallel. For reduction dimensions however, there is a
/// choice in how we split the reduction dimension. This enum exposes this
/// choice.
enum class ReductionTilingStrategy {
  // [reduction] -> [reduction1, reduction2]
  // -> loop[reduction1] { [reduction2] }
  FullReduction,
  // [reduction] -> [reduction1, parallel2]
  // -> loop[reduction1] { [parallel2] }; merge[reduction1]
  PartialReductionOuterReduction,
  // [reduction] -> [parallel1, reduction2]
  // -> loop[parallel1] { [reduction2] }; merge[parallel1]
  PartialReductionOuterParallel
};

/// Container for the result of merge operation of tiling.
/// - `mergeOps` contains operations created during the merge.
/// - `replacements` contains the values that represents the result of the
/// merge. These are used as replacements for the original tiled operation.
struct MergeResult {
  SmallVector<Operation *> mergeOps;
  SmallVector<Value> replacements;
};

/// Per-dimension alignment of a loop tile size to a `linalg.pack` /
/// `linalg.unpack` inner tile size, supplied by the caller (which performed the
/// tiling and knows both the tile sizes and the inner tiles) so that
/// pack/unpack TilingInterface implementations need not re-derive it from the
/// materialized IR. An absent entry (or `Unknown`) means "no information": the
/// implementation must fall back to its prior behavior for that dimension.
///   - `Multiple`: the loop tile size is an integer multiple of the pack/unpack
///   inner tile.
///   - `Equal`:    the loop tile size equals the pack/unpack inner tile size.
enum class InnerTileAlignment : int64_t { Unknown = 0, Multiple, Equal };

/// Returns true iff `value` is a valid `InnerTileAlignment` enumerator.
inline bool isValidInnerTileAlignment(int64_t value) {
  switch (static_cast<InnerTileAlignment>(value)) {
  case InnerTileAlignment::Unknown:
  case InnerTileAlignment::Multiple:
  case InnerTileAlignment::Equal:
    return true;
  }
  return false;
}

/// Verifies that every entry of a raw `inner_tile_alignments` integer array is
/// a valid `InnerTileAlignment`, emitting the standard op error on `op`
/// otherwise.
LogicalResult verifyInnerTileAlignments(Operation *op,
                                        ArrayRef<int64_t> alignments);

/// Maps a validated `inner_tile_alignments` integer array onto the
/// per-dimension `InnerTileAlignment` hints consumed by the tiling driver.
SmallVector<InnerTileAlignment>
convertInnerTileAlignments(ArrayRef<int64_t> alignments);

/// Returns the keyword spelling of an `InnerTileAlignment` (`Unknown`,
/// `Multiple` or `Equal`) used by the `inner_tile_alignments` assembly syntax.
StringRef stringifyInnerTileAlignment(InnerTileAlignment alignment);

/// Returns the `InnerTileAlignment` for a keyword spelling, or `std::nullopt`
/// if `keyword` is not one of `Unknown`, `Multiple` or `Equal`.
std::optional<InnerTileAlignment>
symbolizeInnerTileAlignment(StringRef keyword);

/// Custom directive parser/printer for an `inner_tile_alignments` attribute,
/// rendering the `DenseI64ArrayAttr` as a keyword list, e.g.
/// `[Equal, Multiple, Unknown]` (see `InnerTileAlignment`). Shared by the
/// transform ops that carry the hint.
ParseResult parseInnerTileAlignmentArray(OpAsmParser &parser,
                                         DenseI64ArrayAttr &alignments);
void printInnerTileAlignmentArray(OpAsmPrinter &printer, Operation *op,
                                  DenseI64ArrayAttr alignments);

} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Interfaces/TilingInterface.h.inc"

#endif // MLIR_INTERFACES_TILINGINTERFACE_H_
