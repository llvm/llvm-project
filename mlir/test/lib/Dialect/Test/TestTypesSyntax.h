//===- TestTypesSyntax.h - Test types in `Syntax` ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_DIALECT_TEST_TESTTYPESSYNTAX_H
#define MLIR_TEST_DIALECT_TEST_TESTTYPESSYNTAX_H

#include <optional>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

// The types in `Syntax` use the types in `Core`.
#include "TestTypesCore.h"

namespace test {
/// A custom type for a test type parameter.
struct CustomParam {
  int value;

  bool operator==(const CustomParam &other) const {
    return other.value == value;
  }
};

inline llvm::hash_code hash_value(const test::CustomParam &param) {
  return llvm::hash_value(param.value);
}
} // namespace test

namespace mlir {
template <>
struct FieldParser<test::CustomParam> {
  static FailureOr<test::CustomParam> parse(AsmParser &parser) {
    auto value = FieldParser<int>::parse(parser);
    if (failed(value))
      return failure();
    return test::CustomParam{*value};
  }
};

inline mlir::AsmPrinter &operator<<(mlir::AsmPrinter &printer,
                                    test::CustomParam param) {
  return printer << param.value;
}

/// Overload the attribute parameter parser for optional integers.
template <>
struct FieldParser<std::optional<int>> {
  static FailureOr<std::optional<int>> parse(AsmParser &parser) {
    std::optional<int> value;
    value.emplace();
    OptionalParseResult result = parser.parseOptionalInteger(*value);
    if (result.has_value()) {
      if (succeeded(*result))
        return value;
      return failure();
    }
    value.reset();
    return value;
  }
};
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TestTypesSyntax.h.inc"

#endif // MLIR_TEST_DIALECT_TEST_TESTTYPESSYNTAX_H
