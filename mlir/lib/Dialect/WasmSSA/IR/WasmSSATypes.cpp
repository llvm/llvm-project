//===- WasmSSAOps.cpp - WasmSSA dialect operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir::wasmssa {
#include "mlir/Dialect/WasmSSA/IR/WasmSSATypeConstraints.cpp.inc"
} // namespace mlir::wasmssa

using namespace mlir;
using namespace mlir::wasmssa;

Type LimitType::parse(::mlir::AsmParser &parser) {
  auto res = parser.parseLSquare();
  uint32_t minLimit{0};
  std::optional<uint32_t> maxLimit{std::nullopt};
  res = parser.parseInteger(minLimit);
  res = parser.parseColon();
  uint32_t maxValue{0};
  auto maxParseRes = parser.parseOptionalInteger(maxValue);
  if (maxParseRes.has_value() && (*maxParseRes).succeeded())
    maxLimit = maxValue;

  res = parser.parseRSquare();
  return LimitType::get(parser.getContext(), minLimit, maxLimit);
}

void LimitType::print(AsmPrinter &printer) const {
  printer << '[' << getMin() << ':';
  std::optional<uint32_t> maxLim = getMax();
  if (maxLim)
    printer << *maxLim;
  printer << ']';
}
