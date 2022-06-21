//===------------ ParserUtils.h - Parse text to SPIR-V ops ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities used for parsing types and ops for SPIR-V
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_PARSERUTILS_H_
#define MLIR_DIALECT_SPIRV_IR_PARSERUTILS_H_

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {

/// Parses the next keyword in `parser` as an enumerant of the given
/// `EnumClass`.
template <typename EnumClass, typename ParserType>
static ParseResult
parseEnumKeywordAttr(EnumClass &value, ParserType &parser,
                     StringRef attrName = spirv::attributeName<EnumClass>()) {
  StringRef keyword;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();
  if (Optional<EnumClass> attr = spirv::symbolizeEnum<EnumClass>(keyword)) {
    value = *attr;
    return success();
  }
  return parser.emitError(loc, "invalid ")
         << attrName << " attribute specification: " << keyword;
}

} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_IR_PARSERUTILS_H_
