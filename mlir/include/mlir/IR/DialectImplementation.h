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
#include <type_traits>

namespace {

// reference https://stackoverflow.com/a/16000226
template <typename T, typename = void>
struct HasStaticDialectName : std::false_type {};

template <typename T>
struct HasStaticDialectName<
    T, typename std::enable_if<
           std::is_same<::llvm::StringLiteral,
                        std::decay_t<decltype(T::dialectName)>>::value,
           void>::type> : std::true_type {};

} // namespace

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
///
/// A parser is key-value compositional only if it consumes exactly one value
/// and leaves the comma separating the next key unconsumed. For example, an
/// undelimited array parser for `values = 1, 2, next = 9` cannot distinguish
/// its element commas from the comma before `next` and may try to parse `next`
/// as another element. Marking it non-compositional lets a keyed property list
/// use a self-delimiting attribute such as `array<i64: 1, 2>` instead.
/// Specializations with this behavior, or that may succeed without consuming a
/// token, should define `isKeyValueCompositional` as false.
template <typename T, typename = T>
struct FieldParser;

/// Parse an attribute.
template <typename AttributeT>
struct FieldParser<
    AttributeT, std::enable_if_t<std::is_base_of<Attribute, AttributeT>::value,
                                 AttributeT>> {
  static FailureOr<AttributeT> parse(AsmParser &parser) {
    if constexpr (HasStaticDialectName<AttributeT>::value) {
      parser.getContext()->getOrLoadDialect(AttributeT::dialectName);
    }
    AttributeT value;
    if (parser.parseCustomAttributeWithFallback(value))
      return failure();
    return value;
  }
};

/// Parse a type.
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
struct FieldParser<IntT, std::enable_if_t<(std::is_integral<IntT>::value ||
                                           std::is_same_v<IntT, llvm::APInt>),
                                          IntT>> {
  static FailureOr<IntT> parse(AsmParser &parser) {
    IntT value{};
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

/// Parse an Optional attribute.
template <typename AttributeT>
struct FieldParser<
    std::optional<AttributeT>,
    std::enable_if_t<std::is_base_of<Attribute, AttributeT>::value,
                     std::optional<AttributeT>>> {
  static constexpr bool isKeyValueCompositional = false;

  static FailureOr<std::optional<AttributeT>> parse(AsmParser &parser) {
    if constexpr (HasStaticDialectName<AttributeT>::value) {
      parser.getContext()->getOrLoadDialect(AttributeT::dialectName);
    }
    AttributeT attr;
    OptionalParseResult result = parser.parseOptionalAttribute(attr);
    if (result.has_value()) {
      if (succeeded(*result))
        return {std::optional<AttributeT>(attr)};
      return failure();
    }
    return {std::nullopt};
  }
};

/// Parse an Optional integer.
template <typename IntT>
struct FieldParser<
    std::optional<IntT>,
    std::enable_if_t<std::is_integral<IntT>::value, std::optional<IntT>>> {
  static constexpr bool isKeyValueCompositional = false;

  static FailureOr<std::optional<IntT>> parse(AsmParser &parser) {
    IntT value;
    OptionalParseResult result = parser.parseOptionalInteger(value);
    if (result.has_value()) {
      if (succeeded(*result))
        return {std::optional<IntT>(value)};
      return failure();
    }
    return {std::nullopt};
  }
};

namespace detail {
template <typename T>
using has_push_back_t = decltype(std::declval<T>().push_back(
    std::declval<typename T::value_type &&>()));

template <typename StorageType, typename = void>
struct HasFieldParser : std::false_type {};

template <typename StorageType>
struct HasFieldParser<StorageType,
                      std::void_t<decltype(sizeof(FieldParser<StorageType>)),
                                  decltype(FieldParser<StorageType>::parse(
                                      std::declval<OpAsmParser &>()))>>
    : std::true_type {};

template <typename ContainerT, typename = void>
struct HasFieldParserContainer : std::false_type {};

template <typename ContainerT>
struct HasFieldParserContainer<ContainerT,
                               std::void_t<has_push_back_t<ContainerT>>>
    : HasFieldParser<typename ContainerT::value_type> {};

template <typename Parser, typename = void>
struct IsKeyValueCompositional : std::true_type {};

template <typename Parser>
struct IsKeyValueCompositional<
    Parser, std::void_t<decltype(Parser::isKeyValueCompositional)>>
    : std::bool_constant<Parser::isKeyValueCompositional> {};

/// Whether the selected FieldParser consumes exactly one value in a keyed
/// property list. Parser specializations may set isKeyValueCompositional to
/// false if they can succeed without consuming a token or consume an
/// undelimited comma-separated list.
template <typename StorageType>
struct HasKeyValueFieldParser
    : std::conjunction<HasFieldParser<StorageType>,
                       IsKeyValueCompositional<FieldParser<StorageType>>> {};
} // namespace detail

/// Parse any container that supports back insertion as a list.
template <typename ContainerT>
struct FieldParser<
    ContainerT,
    std::enable_if_t<detail::HasFieldParserContainer<ContainerT>::value,
                     ContainerT>> {
  static constexpr bool isKeyValueCompositional = false;

  using ElementT = typename ContainerT::value_type;
  static FailureOr<ContainerT> parse(AsmParser &parser) {
    ContainerT elements;
    auto elementParser = [&]() {
      auto element = FieldParser<ElementT>::parse(parser);
      if (failed(element))
        return failure();
      elements.push_back(std::move(*element));
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

namespace detail {
/// Parse a property with its FieldParser when one is available, otherwise
/// fall back to the property's attribute conversion.
template <typename StorageType, typename ConvertFromAttribute>
ParseResult
parsePropertyWithFallback(OpAsmParser &parser, StorageType &storage,
                          ConvertFromAttribute convertFromAttribute) {
  if constexpr (HasKeyValueFieldParser<StorageType>::value) {
    auto value = FieldParser<StorageType>::parse(parser);
    if (failed(value))
      return failure();
    storage = std::move(*value);
    return success();
  } else {
    Attribute attr;
    if (parser.parseAttribute(attr))
      return failure();
    return convertFromAttribute(storage, attr);
  }
}
} // namespace detail

} // namespace mlir

#endif // MLIR_IR_DIALECTIMPLEMENTATION_H
