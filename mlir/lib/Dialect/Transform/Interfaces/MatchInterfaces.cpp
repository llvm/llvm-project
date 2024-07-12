//===- MatchInterfaces.cpp - Transform Dialect Interfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Printing and parsing for match ops.
//===----------------------------------------------------------------------===//

/// Keyword syntax for positional specification inversion.
constexpr const static llvm::StringLiteral kDimExceptKeyword = "except";

/// Keyword syntax for full inclusion in positional specification.
constexpr const static llvm::StringLiteral kDimAllKeyword = "all";

ParseResult transform::parseTransformMatchDims(OpAsmParser &parser,
                                               DenseI64ArrayAttr &rawDimList,
                                               UnitAttr &isInverted,
                                               UnitAttr &isAll) {
  Builder &builder = parser.getBuilder();
  if (parser.parseOptionalKeyword(kDimAllKeyword).succeeded()) {
    rawDimList = builder.getDenseI64ArrayAttr({});
    isInverted = nullptr;
    isAll = builder.getUnitAttr();
    return success();
  }

  isAll = nullptr;
  isInverted = nullptr;
  if (parser.parseOptionalKeyword(kDimExceptKeyword).succeeded()) {
    isInverted = builder.getUnitAttr();
  }

  if (isInverted) {
    if (parser.parseLParen().failed())
      return failure();
  }

  SmallVector<int64_t> values;
  ParseResult listResult = parser.parseCommaSeparatedList(
      [&]() { return parser.parseInteger(values.emplace_back()); });
  if (listResult.failed())
    return failure();

  rawDimList = builder.getDenseI64ArrayAttr(values);

  if (isInverted) {
    if (parser.parseRParen().failed())
      return failure();
  }
  return success();
}

void transform::printTransformMatchDims(OpAsmPrinter &printer, Operation *op,
                                        DenseI64ArrayAttr rawDimList,
                                        UnitAttr isInverted, UnitAttr isAll) {
  if (isAll) {
    printer << kDimAllKeyword;
    return;
  }
  if (isInverted) {
    printer << kDimExceptKeyword << "(";
  }
  llvm::interleaveComma(rawDimList.asArrayRef(), printer.getStream(),
                        [&](int64_t value) { printer << value; });
  if (isInverted) {
    printer << ")";
  }
}

LogicalResult transform::verifyTransformMatchDimsOp(Operation *op,
                                                    ArrayRef<int64_t> raw,
                                                    bool inverted, bool all) {
  if (all) {
    if (inverted) {
      return op->emitOpError()
             << "cannot request both 'all' and 'inverted' values in the list";
    }
    if (!raw.empty()) {
      return op->emitOpError()
             << "cannot both request 'all' and specific values in the list";
    }
  }
  if (!all && raw.empty()) {
    return op->emitOpError() << "must request specific values in the list if "
                                "'all' is not specified";
  }
  SmallVector<int64_t> rawVector = llvm::to_vector(raw);
  auto *it = llvm::unique(rawVector);
  if (it != rawVector.end())
    return op->emitOpError() << "expected the listed values to be unique";

  return success();
}

DiagnosedSilenceableFailure transform::expandTargetSpecification(
    Location loc, bool isAll, bool isInverted, ArrayRef<int64_t> rawList,
    int64_t maxNumber, SmallVectorImpl<int64_t> &result) {
  assert(maxNumber > 0 && "expected size to be positive");
  assert(!(isAll && isInverted) && "cannot invert all");
  if (isAll) {
    result = llvm::to_vector(llvm::seq<int64_t>(0, maxNumber));
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<int64_t> expanded;
  llvm::SmallDenseSet<int64_t> visited;
  expanded.reserve(rawList.size());
  SmallVectorImpl<int64_t> &target = isInverted ? expanded : result;
  for (int64_t raw : rawList) {
    int64_t updated = raw < 0 ? maxNumber + raw : raw;
    if (updated >= maxNumber) {
      return emitSilenceableFailure(loc)
             << "position overflow " << updated << " (updated from " << raw
             << ") for maximum " << maxNumber;
    }
    if (updated < 0) {
      return emitSilenceableFailure(loc) << "position underflow " << updated
                                         << " (updated from " << raw << ")";
    }
    if (!visited.insert(updated).second) {
      return emitSilenceableFailure(loc) << "repeated position " << updated
                                         << " (updated from " << raw << ")";
    }
    target.push_back(updated);
  }

  if (!isInverted)
    return DiagnosedSilenceableFailure::success();

  result.reserve(result.size() + (maxNumber - expanded.size()));
  for (int64_t candidate : llvm::seq<int64_t>(0, maxNumber)) {
    if (llvm::is_contained(expanded, candidate))
      continue;
    result.push_back(candidate);
  }

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.cpp.inc"
