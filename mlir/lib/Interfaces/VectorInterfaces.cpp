//===- VectorInterfaces.cpp - Unrollable vector operations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;

VectorType mlir::vector::detail::transferMaskType(VectorType vecType,
                                                  AffineMap map) {
  auto i1Type = IntegerType::get(map.getContext(), 1);
  SmallVector<int64_t, 8> shape;
  for (int64_t i = 0; i < vecType.getRank(); ++i) {
    // Only result dims have a corresponding dim in the mask.
    if (map.getResult(i).template isa<AffineDimExpr>()) {
      shape.push_back(vecType.getDimSize(i));
    }
  }
  return shape.empty() ? VectorType() : VectorType::get(shape, i1Type);
}

//===----------------------------------------------------------------------===//
// VectorUnroll Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the VectorUnroll interfaces.
#include "mlir/Interfaces/VectorInterfaces.cpp.inc"
