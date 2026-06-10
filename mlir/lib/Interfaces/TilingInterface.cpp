//===- TilingInterface.cpp - Tiling interface -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the interface in `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/SmallVectorExtras.h"

using namespace mlir;

LogicalResult mlir::verifyInnerTileAlignments(Operation *op,
                                              ArrayRef<int64_t> alignments) {
  for (int64_t a : alignments)
    if (!isValidInnerTileAlignment(a))
      return op->emitOpError()
             << "expected inner_tile_alignments entries to be one of 0 "
                "(Unknown), 1 (Multiple) or 2 (Equal), but got "
             << a;
  return success();
}

SmallVector<InnerTileAlignment>
mlir::convertInnerTileAlignments(ArrayRef<int64_t> alignments) {
  return llvm::map_to_vector(alignments, [](int64_t v) {
    assert(isValidInnerTileAlignment(v) &&
           "invalid InnerTileAlignment; should be rejected by the verifier");
    return static_cast<InnerTileAlignment>(v);
  });
}

#include "mlir/Interfaces/TilingInterface.cpp.inc"
