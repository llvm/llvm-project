//===- NNZ.cpp - NNZ-statistics for direct sparse2sparse conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains method definitions for `SparseTensorNNZ`.
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/Storage.h"

using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
SparseTensorNNZ::SparseTensorNNZ(const std::vector<uint64_t> &lvlSizes,
                                 const std::vector<DimLevelType> &lvlTypes)
    : lvlSizes(lvlSizes), lvlTypes(lvlTypes), nnz(getLvlRank()) {
  assert(lvlSizes.size() == lvlTypes.size() && "Rank mismatch");
  bool alreadyCompressed = false;
  (void)alreadyCompressed;
  uint64_t sz = 1; // the product of all `lvlSizes` strictly less than `l`.
  for (uint64_t l = 0, lvlrank = getLvlRank(); l < lvlrank; ++l) {
    const DimLevelType dlt = lvlTypes[l];
    if (isCompressedDLT(dlt)) {
      if (alreadyCompressed)
        MLIR_SPARSETENSOR_FATAL(
            "Multiple compressed levels not currently supported");
      alreadyCompressed = true;
      nnz[l].resize(sz, 0); // Both allocate and zero-initialize.
    } else if (isDenseDLT(dlt)) {
      if (alreadyCompressed)
        MLIR_SPARSETENSOR_FATAL(
            "Dense after compressed not currently supported");
    } else if (isSingletonDLT(dlt)) {
      // Singleton after Compressed causes no problems for allocating
      // `nnz` nor for the yieldPos loop.  This remains true even
      // when adding support for multiple compressed dimensions or
      // for dense-after-compressed.
    } else {
      MLIR_SPARSETENSOR_FATAL("unsupported level type: %d\n",
                              static_cast<uint8_t>(dlt));
    }
    sz = detail::checkedMul(sz, lvlSizes[l]);
  }
}

void SparseTensorNNZ::forallIndices(uint64_t stopLvl,
                                    SparseTensorNNZ::NNZConsumer yield) const {
  assert(stopLvl < getLvlRank() && "Level out of bounds");
  assert(isCompressedDLT(lvlTypes[stopLvl]) &&
         "Cannot look up non-compressed levels");
  forallIndices(yield, stopLvl, 0, 0);
}

void SparseTensorNNZ::add(const std::vector<uint64_t> &lvlInd) {
  uint64_t parentPos = 0;
  for (uint64_t l = 0, lvlrank = getLvlRank(); l < lvlrank; ++l) {
    if (isCompressedDLT(lvlTypes[l]))
      nnz[l][parentPos]++;
    parentPos = parentPos * lvlSizes[l] + lvlInd[l];
  }
}

void SparseTensorNNZ::forallIndices(SparseTensorNNZ::NNZConsumer yield,
                                    uint64_t stopLvl, uint64_t parentPos,
                                    uint64_t l) const {
  assert(l <= stopLvl);
  if (l == stopLvl) {
    assert(parentPos < nnz[l].size() && "Cursor is out of range");
    yield(nnz[l][parentPos]);
  } else {
    const uint64_t sz = lvlSizes[l];
    const uint64_t pstart = parentPos * sz;
    for (uint64_t i = 0; i < sz; ++i)
      forallIndices(yield, stopLvl, pstart + i, l + 1);
  }
}
