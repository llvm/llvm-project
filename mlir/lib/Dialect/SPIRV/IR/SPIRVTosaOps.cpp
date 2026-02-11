//===- SPIRVTosaOps.cpp - MLIR SPIR-V Tosa operations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tosa operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "llvm/Support/InterleavedRange.h"

namespace mlir::spirv {

//===----------------------------------------------------------------------===//
// SPIRV Tosa Custom formatters
//===----------------------------------------------------------------------===//

ParseResult parseSPIRV_I32_1DArmTensor(OpAsmParser &parser,
                                       DenseIntElementsAttr &attr) {
  SmallVector<int32_t, 6> elements;
  auto f = [&]() {
    int32_t value;
    ParseResult r = parser.parseInteger(value);
    elements.push_back(value);
    return r;
  };
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Square, f,
          "parsing values in integer list attribute")) {
    return failure();
  }

  auto i32Type = IntegerType::get(parser.getContext(), 32);
  auto type = TensorArmType::get(
      ArrayRef{static_cast<int64_t>(elements.size())}, i32Type);
  attr = DenseIntElementsAttr::get(type, elements);
  return success();
}

void printSPIRV_I32_1DArmTensor(OpAsmPrinter &printer, Operation *,
                                DenseIntElementsAttr attr) {
  printer << llvm::interleaved_array(
      llvm::map_range(attr.getValues<APInt>(),
                      [](const APInt &a) { return a.getSExtValue(); }));
}

} // namespace mlir::spirv
