//===- MapRef.cpp - A dim2lvl/lvl2dim map reference wrapper ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/SparseTensor/MapRef.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"

mlir::sparse_tensor::MapRef::MapRef(uint64_t d, uint64_t l, const uint64_t *d2l,
                                    const uint64_t *l2d)
    : dimRank(d), lvlRank(l), dim2lvl(d2l), lvl2dim(l2d),
      isPermutation(isPermutationMap()) {
  if (isPermutation) {
    for (uint64_t l = 0; l < lvlRank; l++)
      assert(lvl2dim[dim2lvl[l]] == l);
  }
}

bool mlir::sparse_tensor::MapRef::isPermutationMap() const {
  if (dimRank != lvlRank)
    return false;
  std::vector<bool> seen(dimRank, false);
  for (uint64_t l = 0; l < lvlRank; l++) {
    const uint64_t d = dim2lvl[l];
    if (d >= dimRank || seen[d])
      return false;
    seen[d] = true;
  }
  return true;
}

bool mlir::sparse_tensor::MapRef::isFloor(uint64_t l, uint64_t &i,
                                          uint64_t &c) const {
  if (isEncodedFloor(dim2lvl[l])) {
    i = decodeIndex(dim2lvl[l]);
    c = decodeConst(dim2lvl[l]);
    return true;
  }
  return false;
}

bool mlir::sparse_tensor::MapRef::isMod(uint64_t l, uint64_t &i,
                                        uint64_t &c) const {
  if (isEncodedMod(dim2lvl[l])) {
    i = decodeIndex(dim2lvl[l]);
    c = decodeConst(dim2lvl[l]);
    return true;
  }
  return false;
}

bool mlir::sparse_tensor::MapRef::isMul(uint64_t d, uint64_t &i, uint64_t &c,
                                        uint64_t &ii) const {
  if (isEncodedMul(lvl2dim[d])) {
    i = decodeIndex(lvl2dim[d]);
    c = decodeMulc(lvl2dim[d]);
    ii = decodeMuli(lvl2dim[d]);
    return true;
  }
  return false;
}
