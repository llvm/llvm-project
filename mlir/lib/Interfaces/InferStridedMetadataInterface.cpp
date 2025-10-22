//===- InferStridedMetadataInterface.cpp - Strided md inference interface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferStridedMetadataInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <optional>

using namespace mlir;

#include "mlir/Interfaces/InferStridedMetadataInterface.cpp.inc"

void StridedMetadataRange::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "strided_metadata<None>";
    return;
  }
  os << "strided_metadata<offset = [";
  llvm::interleaveComma(*offsets, os, [&](const ConstantIntRanges &range) {
    os << "{" << range << "}";
  });
  os << "], sizes = [";
  llvm::interleaveComma(sizes, os, [&](const ConstantIntRanges &range) {
    os << "{" << range << "}";
  });
  os << "], strides = [";
  llvm::interleaveComma(strides, os, [&](const ConstantIntRanges &range) {
    os << "{" << range << "}";
  });
  os << "]>";
}
