//===- StridedMetadataRangeAnalysis.cpp - Integer range analysis --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer range inference
// which is used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "strided-metadata-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

/// Get the entry state for a value. For any value that is not a ranked memref,
/// this function sets the metadata to a top state with no offsets, sizes, or
/// strides. For `memref` types, this function will use the metadata in the type
/// to try to deduce as much informaiton as possible.
static StridedMetadataRange getEntryStateImpl(Value v, int32_t indexBitwidth) {
  // TODO: generalize this method with a type interface.
  auto mTy = dyn_cast<BaseMemRefType>(v.getType());

  // If not a memref or it's un-ranked, don't infer any metadata.
  if (!mTy || !mTy.hasRank())
    return StridedMetadataRange::getMaxRanges(indexBitwidth, 0, 0, 0);

  // Get the top state.
  auto metadata =
      StridedMetadataRange::getMaxRanges(indexBitwidth, mTy.getRank());

  // Compute the offset and strides.
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(cast<MemRefType>(mTy).getStridesAndOffset(strides, offset)))
    return metadata;

  // Refine the metadata if we know it from the type.
  if (!ShapedType::isDynamic(offset)) {
    metadata.getOffsets()[0] =
        ConstantIntRanges::constant(APInt(indexBitwidth, offset));
  }
  for (auto &&[size, range] :
       llvm::zip_equal(mTy.getShape(), metadata.getSizes())) {
    if (ShapedType::isDynamic(size))
      continue;
    range = ConstantIntRanges::constant(APInt(indexBitwidth, size));
  }
  for (auto &&[stride, range] :
       llvm::zip_equal(strides, metadata.getStrides())) {
    if (ShapedType::isDynamic(stride))
      continue;
    range = ConstantIntRanges::constant(APInt(indexBitwidth, stride));
  }

  return metadata;
}

StridedMetadataRangeAnalysis::StridedMetadataRangeAnalysis(
    DataFlowSolver &solver, int32_t indexBitwidth)
    : SparseForwardDataFlowAnalysis(solver), indexBitwidth(indexBitwidth) {
  assert(indexBitwidth > 0 && "invalid bitwidth");
}

void StridedMetadataRangeAnalysis::setToEntryState(
    StridedMetadataRangeLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(getEntryStateImpl(
                                  lattice->getAnchor(), indexBitwidth)));
}

LogicalResult StridedMetadataRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const StridedMetadataRangeLattice *> operands,
    ArrayRef<StridedMetadataRangeLattice *> results) {
  auto inferrable = dyn_cast<InferStridedMetadataOpInterface>(op);

  // Bail if we cannot reason about the op.
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LDBG() << "Inferring metadata for: "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());

  // Helper function to retrieve int range values.
  auto getIntRange = [&](Value value) -> IntegerValueRange {
    auto lattice = getOrCreateFor<IntegerValueRangeLattice>(
        getProgramPointAfter(op), value);
    return lattice ? lattice->getValue() : IntegerValueRange();
  };

  // Convert the arguments lattices to a vector.
  SmallVector<StridedMetadataRange> argRanges = llvm::map_to_vector(
      operands, [](const StridedMetadataRangeLattice *lattice) {
        return lattice->getValue();
      });

  // Callback to set metadata on a result.
  auto joinCallback = [&](Value v, const StridedMetadataRange &md) {
    auto result = cast<OpResult>(v);
    assert(llvm::is_contained(op->getResults(), result));
    LDBG() << "- Inferred metadata: " << md;
    StridedMetadataRangeLattice *lattice = results[result.getResultNumber()];
    ChangeResult changed = lattice->join(md);
    LDBG() << "- Joined metadata: " << lattice->getValue();
    propagateIfChanged(lattice, changed);
  };

  // Infer the metadata.
  inferrable.inferStridedMetadataRanges(argRanges, getIntRange, joinCallback,
                                        indexBitwidth);
  return success();
}
