//===- LvlTypeParser.h - `DimLevelType` parser ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LvlTypeParser.h"

using namespace mlir;
using namespace mlir::sparse_tensor;
using namespace mlir::sparse_tensor::ir_detail;

//===----------------------------------------------------------------------===//
// TODO(wrengr): rephrase these to do the trick for gobbling up any trailing
// semicolon
//
// NOTE: There's no way for `FAILURE_IF_FAILED` to simultaneously support
// both `OptionalParseResult` and `InFlightDiagnostic` return types.
// We can get the compiler to accept the code if we returned "`{}`",
// however for `OptionalParseResult` that would become the nullopt result,
// whereas for `InFlightDiagnostic` it would become a result that can
// be implicitly converted to success.  By using "`failure()`" we ensure
// that `OptionalParseResult` behaves as intended, however that means the
// macro cannot be used for `InFlightDiagnostic` since there's no implicit
// conversion.
#define FAILURE_IF_FAILED(STMT)                                                \
  if (failed(STMT)) {                                                          \
    return failure();                                                          \
  }

// Although `ERROR_IF` is phrased to return `InFlightDiagnostic`, that type
// can be implicitly converted to all four of `LogicalResult, `FailureOr`,
// `ParseResult`, and `OptionalParseResult`.  (However, beware that the
// conversion to `OptionalParseResult` doesn't properly delegate to
// `InFlightDiagnostic::operator ParseResult`.)
//
// NOTE: this macro assumes `AsmParser parser` and `SMLoc loc` are in scope.
#define ERROR_IF(COND, MSG)                                                    \
  if (COND) {                                                                  \
    return parser.emitError(loc, MSG);                                         \
  }

//===----------------------------------------------------------------------===//
// `LvlTypeParser` implementation.
//===----------------------------------------------------------------------===//

std::optional<DimLevelType> LvlTypeParser::lookup(StringRef str) const {
  // NOTE: `StringMap::lookup` will return a default-constructed value if
  // the key isn't found; which for enums means zero, and therefore makes
  // it impossible to distinguish between actual zero-DimLevelType vs
  // not-found.  Whereas `StringMap::at` asserts that the key is found,
  // which we don't want either.
  const auto it = map.find(str);
  return it == map.end() ? std::nullopt : std::make_optional(it->second);
}

std::optional<DimLevelType> LvlTypeParser::lookup(StringAttr str) const {
  return str ? lookup(str.getValue()) : std::nullopt;
}

FailureOr<DimLevelType> LvlTypeParser::parseLvlType(AsmParser &parser) const {
  DimLevelType out;
  FAILURE_IF_FAILED(parseLvlType(parser, out))
  return out;
}

ParseResult LvlTypeParser::parseLvlType(AsmParser &parser,
                                        DimLevelType &out) const {
  const auto loc = parser.getCurrentLocation();
  StringRef strVal;
  FAILURE_IF_FAILED(parser.parseOptionalKeyword(&strVal));
  const auto lvlType = lookup(strVal);
  ERROR_IF(!lvlType, "unknown level-type '" + strVal + "'")
  out = *lvlType;
  return success();
}

//===----------------------------------------------------------------------===//
