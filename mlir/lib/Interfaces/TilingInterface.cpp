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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSwitch.h"

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

StringRef mlir::stringifyInnerTileAlignment(InnerTileAlignment alignment) {
  switch (alignment) {
  case InnerTileAlignment::Unknown:
    return "Unknown";
  case InnerTileAlignment::Multiple:
    return "Multiple";
  case InnerTileAlignment::Equal:
    return "Equal";
  }
  llvm_unreachable("unknown InnerTileAlignment");
}

std::optional<InnerTileAlignment>
mlir::symbolizeInnerTileAlignment(StringRef keyword) {
  return llvm::StringSwitch<std::optional<InnerTileAlignment>>(keyword)
      .Case("Unknown", InnerTileAlignment::Unknown)
      .Case("Multiple", InnerTileAlignment::Multiple)
      .Case("Equal", InnerTileAlignment::Equal)
      .Default(std::nullopt);
}

ParseResult mlir::parseInnerTileAlignmentArray(OpAsmParser &parser,
                                               DenseI64ArrayAttr &alignments) {
  SmallVector<int64_t> values;
  auto parseEntry = [&]() -> ParseResult {
    StringRef keyword;
    llvm::SMLoc loc = parser.getCurrentLocation();
    if (parser.parseKeyword(&keyword))
      return failure();
    std::optional<InnerTileAlignment> alignment =
        symbolizeInnerTileAlignment(keyword);
    if (!alignment)
      return parser.emitError(loc)
             << "expected one of 'Unknown', 'Multiple' or 'Equal', but got '"
             << keyword << "'";
    values.push_back(static_cast<int64_t>(*alignment));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseEntry))
    return failure();
  alignments = DenseI64ArrayAttr::get(parser.getContext(), values);
  return success();
}

void mlir::printInnerTileAlignmentArray(OpAsmPrinter &printer, Operation *,
                                        DenseI64ArrayAttr alignments) {
  printer << "[";
  llvm::interleaveComma(alignments.asArrayRef(), printer, [&](int64_t value) {
    printer << stringifyInnerTileAlignment(
        static_cast<InnerTileAlignment>(value));
  });
  printer << "]";
}

#include "mlir/Interfaces/TilingInterface.cpp.inc"
