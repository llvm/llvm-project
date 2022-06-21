//===- DialectImplementation.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities classes for implementing dialect attributes and
// types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECTIMPLEMENTATION_H
#define MLIR_IR_DIALECTIMPLEMENTATION_H

#include "mlir/IR/OpImplementation.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// DialectAsmPrinter
//===----------------------------------------------------------------------===//

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom printAttribute/printType() method on a
/// dialect.
class DialectAsmPrinter : public AsmPrinter {
public:
  using AsmPrinter::AsmPrinter;
  ~DialectAsmPrinter() override;
};

//===----------------------------------------------------------------------===//
// DialectAsmParser
//===----------------------------------------------------------------------===//

/// The DialectAsmParser has methods for interacting with the asm parser when
/// parsing attributes and types.
class DialectAsmParser : public AsmParser {
public:
  using AsmParser::AsmParser;
  ~DialectAsmParser() override;

  /// Returns the full specification of the symbol being parsed. This allows for
  /// using a separate parser if necessary.
  virtual StringRef getFullSymbolSpec() const = 0;
};

//===----------------------------------------------------------------------===//
// Parse Fields
//===----------------------------------------------------------------------===//

/// Provide a template class that can be specialized by users to dispatch to
/// parsers. Auto-generated parsers generate calls to `FieldParser<T>::parse`,
/// where `T` is the parameter storage type, to parse custom types.
template <typename T, typename = T>
struct FieldParser;

/// Parse an attribute.
template <typename AttributeT>
struct FieldParser<
    AttributeT, std::enable_if_t<std::is_base_of<Attribute, AttributeT>::value,
                                 AttributeT>> {
  static FailureOr<AttributeT> parse(AsmParser &parser) {
    AttributeT value;
    if (parser.parseCustomAttributeWithFallback(value))
      return failure();
    return value;
  }
};

/// Parse an attribute.
template <typename TypeT>
struct FieldParser<
    TypeT, std::enable_if_t<std::is_base_of<Type, TypeT>::value, TypeT>> {
  static FailureOr<TypeT> parse(AsmParser &parser) {
    TypeT value;
    if (parser.parseCustomTypeWithFallback(value))
      return failure();
    return value;
  }
};

/// Parse any integer.
template <typename IntT>
struct FieldParser<IntT,
                   std::enable_if_t<std::is_integral<IntT>::value, IntT>> {
  static FailureOr<IntT> parse(AsmParser &parser) {
    IntT value;
    if (parser.parseInteger(value))
      return failure();
    return value;
  }
};

/// Parse a string.
template <>
struct FieldParser<std::string> {
  static FailureOr<std::string> parse(AsmParser &parser) {
    std::string value;
    if (parser.parseString(&value))
      return failure();
    return value;
  }
};

/// Parse any container that supports back insertion as a list.
template <typename ContainerT>
struct FieldParser<
    ContainerT, std::enable_if_t<std::is_member_function_pointer<
                                     decltype(&ContainerT::push_back)>::value,
                                 ContainerT>> {
  using ElementT = typename ContainerT::value_type;
  static FailureOr<ContainerT> parse(AsmParser &parser) {
    ContainerT elements;
    auto elementParser = [&]() {
      auto element = FieldParser<ElementT>::parse(parser);
      if (failed(element))
        return failure();
      elements.push_back(*element);
      return success();
    };
    if (parser.parseCommaSeparatedList(elementParser))
      return failure();
    return elements;
  }
};

/// Parse an affine map.
template <>
struct FieldParser<AffineMap> {
  static FailureOr<AffineMap> parse(AsmParser &parser) {
    AffineMap map;
    if (failed(parser.parseAffineMap(map)))
      return failure();
    return map;
  }
};

} // namespace mlir

#endif // MLIR_IR_DIALECTIMPLEMENTATION_H
