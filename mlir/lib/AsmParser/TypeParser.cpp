//===- TypeParser.cpp - MLIR Type Parser Implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the MLIR Types.
//
//===----------------------------------------------------------------------===//

#include "Parser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::detail;

/// Optionally parse a type.
OptionalParseResult Parser::parseOptionalType(Type &type) {
  // There are many different starting tokens for a type, check them here.
  switch (getToken().getKind()) {
  case Token::l_paren:
  case Token::kw_memref:
  case Token::kw_tensor:
  case Token::kw_complex:
  case Token::kw_tuple:
  case Token::kw_vector:
  case Token::inttype:
  case Token::kw_f8E5M2:
  case Token::kw_f8E4M3FN:
  case Token::kw_f8E5M2FNUZ:
  case Token::kw_f8E4M3FNUZ:
  case Token::kw_f8E4M3B11FNUZ:
  case Token::kw_bf16:
  case Token::kw_f16:
  case Token::kw_tf32:
  case Token::kw_f32:
  case Token::kw_f64:
  case Token::kw_f80:
  case Token::kw_f128:
  case Token::kw_index:
  case Token::kw_none:
  case Token::exclamation_identifier:
    return failure(!(type = parseType()));

  default:
    return std::nullopt;
  }
}

/// Parse an arbitrary type.
///
///   type ::= function-type
///          | non-function-type
///
Type Parser::parseType() {
  if (getToken().is(Token::l_paren))
    return parseFunctionType();
  return parseNonFunctionType();
}

/// Parse a function result type.
///
///   function-result-type ::= type-list-parens
///                          | non-function-type
///
ParseResult Parser::parseFunctionResultTypes(SmallVectorImpl<Type> &elements) {
  if (getToken().is(Token::l_paren))
    return parseTypeListParens(elements);

  Type t = parseNonFunctionType();
  if (!t)
    return failure();
  elements.push_back(t);
  return success();
}

/// Parse a list of types without an enclosing parenthesis.  The list must have
/// at least one member.
///
///   type-list-no-parens ::=  type (`,` type)*
///
ParseResult Parser::parseTypeListNoParens(SmallVectorImpl<Type> &elements) {
  auto parseElt = [&]() -> ParseResult {
    auto elt = parseType();
    elements.push_back(elt);
    return elt ? success() : failure();
  };

  return parseCommaSeparatedList(parseElt);
}

/// Parse a parenthesized list of types.
///
///   type-list-parens ::= `(` `)`
///                      | `(` type-list-no-parens `)`
///
ParseResult Parser::parseTypeListParens(SmallVectorImpl<Type> &elements) {
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();

  // Handle empty lists.
  if (getToken().is(Token::r_paren))
    return consumeToken(), success();

  if (parseTypeListNoParens(elements) ||
      parseToken(Token::r_paren, "expected ')'"))
    return failure();
  return success();
}

/// Parse a complex type.
///
///   complex-type ::= `complex` `<` type `>`
///
Type Parser::parseComplexType() {
  consumeToken(Token::kw_complex);

  // Parse the '<'.
  if (parseToken(Token::less, "expected '<' in complex type"))
    return nullptr;

  SMLoc elementTypeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType ||
      parseToken(Token::greater, "expected '>' in complex type"))
    return nullptr;
  if (!isa<FloatType>(elementType) && !isa<IntegerType>(elementType))
    return emitError(elementTypeLoc, "invalid element type for complex"),
           nullptr;

  return ComplexType::get(elementType);
}

/// Parse a function type.
///
///   function-type ::= type-list-parens `->` function-result-type
///
Type Parser::parseFunctionType() {
  assert(getToken().is(Token::l_paren));

  SmallVector<Type, 4> arguments, results;
  if (parseTypeListParens(arguments) ||
      parseToken(Token::arrow, "expected '->' in function type") ||
      parseFunctionResultTypes(results))
    return nullptr;

  return builder.getFunctionType(arguments, results);
}

/// Parse a memref type.
///
///   memref-type ::= ranked-memref-type | unranked-memref-type
///
///   ranked-memref-type ::= `memref` `<` dimension-list-ranked type
///                          (`,` layout-specification)? (`,` memory-space)? `>`
///
///   unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
///
///   stride-list ::= `[` (dimension (`,` dimension)*)? `]`
///   strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
///   layout-specification ::= semi-affine-map | strided-layout | attribute
///   memory-space ::= integer-literal | attribute
///
Type Parser::parseMemRefType() {
  SMLoc loc = getToken().getLoc();
  consumeToken(Token::kw_memref);

  if (parseToken(Token::less, "expected '<' in memref type"))
    return nullptr;

  bool isUnranked;
  SmallVector<int64_t, 4> dimensions;

  if (consumeIf(Token::star)) {
    // This is an unranked memref type.
    isUnranked = true;
    if (parseXInDimensionList())
      return nullptr;

  } else {
    isUnranked = false;
    if (parseDimensionListRanked(dimensions))
      return nullptr;
  }

  // Parse the element type.
  auto typeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType)
    return nullptr;

  // Check that memref is formed from allowed types.
  if (!BaseMemRefType::isValidElementType(elementType))
    return emitError(typeLoc, "invalid memref element type"), nullptr;

  MemRefLayoutAttrInterface layout;
  Attribute memorySpace;

  auto parseElt = [&]() -> ParseResult {
    // Either it is MemRefLayoutAttrInterface or memory space attribute.
    Attribute attr = parseAttribute();
    if (!attr)
      return failure();

    if (isa<MemRefLayoutAttrInterface>(attr)) {
      layout = cast<MemRefLayoutAttrInterface>(attr);
    } else if (memorySpace) {
      return emitError("multiple memory spaces specified in memref type");
    } else {
      memorySpace = attr;
      return success();
    }

    if (isUnranked)
      return emitError("cannot have affine map for unranked memref type");
    if (memorySpace)
      return emitError("expected memory space to be last in memref type");

    return success();
  };

  // Parse a list of mappings and address space if present.
  if (!consumeIf(Token::greater)) {
    // Parse comma separated list of affine maps, followed by memory space.
    if (parseToken(Token::comma, "expected ',' or '>' in memref type") ||
        parseCommaSeparatedListUntil(Token::greater, parseElt,
                                     /*allowEmptyList=*/false)) {
      return nullptr;
    }
  }

  if (isUnranked)
    return getChecked<UnrankedMemRefType>(loc, elementType, memorySpace);

  return getChecked<MemRefType>(loc, dimensions, elementType, layout,
                                memorySpace);
}

/// Parse any type except the function type.
///
///   non-function-type ::= integer-type
///                       | index-type
///                       | float-type
///                       | extended-type
///                       | vector-type
///                       | tensor-type
///                       | memref-type
///                       | complex-type
///                       | tuple-type
///                       | none-type
///
///   index-type ::= `index`
///   float-type ::= `f16` | `bf16` | `f32` | `f64` | `f80` | `f128`
///   none-type ::= `none`
///
Type Parser::parseNonFunctionType() {
  switch (getToken().getKind()) {
  default:
    return (emitWrongTokenError("expected non-function type"), nullptr);
  case Token::kw_memref:
    return parseMemRefType();
  case Token::kw_tensor:
    return parseTensorType();
  case Token::kw_complex:
    return parseComplexType();
  case Token::kw_tuple:
    return parseTupleType();
  case Token::kw_vector:
    return parseVectorType();
  // integer-type
  case Token::inttype: {
    auto width = getToken().getIntTypeBitwidth();
    if (!width.has_value())
      return (emitError("invalid integer width"), nullptr);
    if (*width > IntegerType::kMaxWidth) {
      emitError(getToken().getLoc(), "integer bitwidth is limited to ")
          << IntegerType::kMaxWidth << " bits";
      return nullptr;
    }

    IntegerType::SignednessSemantics signSemantics = IntegerType::Signless;
    if (std::optional<bool> signedness = getToken().getIntTypeSignedness())
      signSemantics = *signedness ? IntegerType::Signed : IntegerType::Unsigned;

    consumeToken(Token::inttype);
    return IntegerType::get(getContext(), *width, signSemantics);
  }

  // float-type
  case Token::kw_f8E5M2:
    consumeToken(Token::kw_f8E5M2);
    return builder.getFloat8E5M2Type();
  case Token::kw_f8E4M3FN:
    consumeToken(Token::kw_f8E4M3FN);
    return builder.getFloat8E4M3FNType();
  case Token::kw_f8E5M2FNUZ:
    consumeToken(Token::kw_f8E5M2FNUZ);
    return builder.getFloat8E5M2FNUZType();
  case Token::kw_f8E4M3FNUZ:
    consumeToken(Token::kw_f8E4M3FNUZ);
    return builder.getFloat8E4M3FNUZType();
  case Token::kw_f8E4M3B11FNUZ:
    consumeToken(Token::kw_f8E4M3B11FNUZ);
    return builder.getFloat8E4M3B11FNUZType();
  case Token::kw_bf16:
    consumeToken(Token::kw_bf16);
    return builder.getBF16Type();
  case Token::kw_f16:
    consumeToken(Token::kw_f16);
    return builder.getF16Type();
  case Token::kw_tf32:
    consumeToken(Token::kw_tf32);
    return builder.getTF32Type();
  case Token::kw_f32:
    consumeToken(Token::kw_f32);
    return builder.getF32Type();
  case Token::kw_f64:
    consumeToken(Token::kw_f64);
    return builder.getF64Type();
  case Token::kw_f80:
    consumeToken(Token::kw_f80);
    return builder.getF80Type();
  case Token::kw_f128:
    consumeToken(Token::kw_f128);
    return builder.getF128Type();

  // index-type
  case Token::kw_index:
    consumeToken(Token::kw_index);
    return builder.getIndexType();

  // none-type
  case Token::kw_none:
    consumeToken(Token::kw_none);
    return builder.getNoneType();

  // extended type
  case Token::exclamation_identifier:
    return parseExtendedType();

  // Handle completion of a dialect type.
  case Token::code_complete:
    if (getToken().isCodeCompletionFor(Token::exclamation_identifier))
      return parseExtendedType();
    return codeCompleteType();
  }
}

/// Parse a tensor type.
///
///   tensor-type ::= `tensor` `<` dimension-list type `>`
///   dimension-list ::= dimension-list-ranked | `*x`
///
Type Parser::parseTensorType() {
  consumeToken(Token::kw_tensor);

  if (parseToken(Token::less, "expected '<' in tensor type"))
    return nullptr;

  bool isUnranked;
  SmallVector<int64_t, 4> dimensions;

  if (consumeIf(Token::star)) {
    // This is an unranked tensor type.
    isUnranked = true;

    if (parseXInDimensionList())
      return nullptr;

  } else {
    isUnranked = false;
    if (parseDimensionListRanked(dimensions))
      return nullptr;
  }

  // Parse the element type.
  auto elementTypeLoc = getToken().getLoc();
  auto elementType = parseType();

  // Parse an optional encoding attribute.
  Attribute encoding;
  if (consumeIf(Token::comma)) {
    auto parseResult = parseOptionalAttribute(encoding);
    if (parseResult.has_value()) {
      if (failed(parseResult.value()))
        return nullptr;
      if (auto v = dyn_cast_or_null<VerifiableTensorEncoding>(encoding)) {
        if (failed(v.verifyEncoding(dimensions, elementType,
                                    [&] { return emitError(); })))
          return nullptr;
      }
    }
  }

  if (!elementType || parseToken(Token::greater, "expected '>' in tensor type"))
    return nullptr;
  if (!TensorType::isValidElementType(elementType))
    return emitError(elementTypeLoc, "invalid tensor element type"), nullptr;

  if (isUnranked) {
    if (encoding)
      return emitError("cannot apply encoding to unranked tensor"), nullptr;
    return UnrankedTensorType::get(elementType);
  }
  return RankedTensorType::get(dimensions, elementType, encoding);
}

/// Parse a tuple type.
///
///   tuple-type ::= `tuple` `<` (type (`,` type)*)? `>`
///
Type Parser::parseTupleType() {
  consumeToken(Token::kw_tuple);

  // Parse the '<'.
  if (parseToken(Token::less, "expected '<' in tuple type"))
    return nullptr;

  // Check for an empty tuple by directly parsing '>'.
  if (consumeIf(Token::greater))
    return TupleType::get(getContext());

  // Parse the element types and the '>'.
  SmallVector<Type, 4> types;
  if (parseTypeListNoParens(types) ||
      parseToken(Token::greater, "expected '>' in tuple type"))
    return nullptr;

  return TupleType::get(getContext(), types);
}

/// Parse a vector type.
///
/// vector-type ::= `vector` `<` vector-dim-list vector-element-type `>`
/// vector-dim-list := (static-dim-list `x`)? (`[` static-dim-list `]` `x`)?
/// static-dim-list ::= decimal-literal (`x` decimal-literal)*
///
VectorType Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (parseToken(Token::less, "expected '<' in vector type"))
    return nullptr;

  SmallVector<int64_t, 4> dimensions;
  SmallVector<bool, 4> scalableDims;
  if (parseVectorDimensionList(dimensions, scalableDims))
    return nullptr;
  if (any_of(dimensions, [](int64_t i) { return i <= 0; }))
    return emitError(getToken().getLoc(),
                     "vector types must have positive constant sizes"),
           nullptr;

  // Parse the element type.
  auto typeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType || parseToken(Token::greater, "expected '>' in vector type"))
    return nullptr;

  if (!VectorType::isValidElementType(elementType))
    return emitError(typeLoc, "vector elements must be int/index/float type"),
           nullptr;

  return VectorType::get(dimensions, elementType, scalableDims);
}

/// Parse a dimension list in a vector type. This populates the dimension list.
/// For i-th dimension, `scalableDims[i]` contains either:
///   * `false` for a non-scalable dimension (e.g. `4`),
///   * `true` for a scalable dimension (e.g. `[4]`).
///
/// vector-dim-list := (static-dim-list `x`)?
/// static-dim-list ::= static-dim (`x` static-dim)*
/// static-dim ::= (decimal-literal | `[` decimal-literal `]`)
///
ParseResult
Parser::parseVectorDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                 SmallVectorImpl<bool> &scalableDims) {
  // If there is a set of fixed-length dimensions, consume it
  while (getToken().is(Token::integer) || getToken().is(Token::l_square)) {
    int64_t value;
    bool scalable = consumeIf(Token::l_square);
    if (parseIntegerInDimensionList(value))
      return failure();
    dimensions.push_back(value);
    if (scalable) {
      if (!consumeIf(Token::r_square))
        return emitWrongTokenError("missing ']' closing scalable dimension");
    }
    scalableDims.push_back(scalable);
    // Make sure we have an 'x' or something like 'xbf32'.
    if (parseXInDimensionList())
      return failure();
  }

  return success();
}

/// Parse a dimension list of a tensor or memref type.  This populates the
/// dimension list, using ShapedType::kDynamic for the `?` dimensions if
/// `allowDynamic` is set and errors out on `?` otherwise. Parsing the trailing
/// `x` is configurable.
///
///   dimension-list ::= eps | dimension (`x` dimension)*
///   dimension-list-with-trailing-x ::= (dimension `x`)*
///   dimension ::= `?` | decimal-literal
///
/// When `allowDynamic` is not set, this is used to parse:
///
///   static-dimension-list ::= eps | decimal-literal (`x` decimal-literal)*
///   static-dimension-list-with-trailing-x ::= (dimension `x`)*
ParseResult
Parser::parseDimensionListRanked(SmallVectorImpl<int64_t> &dimensions,
                                 bool allowDynamic, bool withTrailingX) {
  auto parseDim = [&]() -> LogicalResult {
    auto loc = getToken().getLoc();
    if (consumeIf(Token::question)) {
      if (!allowDynamic)
        return emitError(loc, "expected static shape");
      dimensions.push_back(ShapedType::kDynamic);
    } else {
      int64_t value;
      if (failed(parseIntegerInDimensionList(value)))
        return failure();
      dimensions.push_back(value);
    }
    return success();
  };

  if (withTrailingX) {
    while (getToken().isAny(Token::integer, Token::question)) {
      if (failed(parseDim()) || failed(parseXInDimensionList()))
        return failure();
    }
    return success();
  }

  if (getToken().isAny(Token::integer, Token::question)) {
    if (failed(parseDim()))
      return failure();
    while (getToken().is(Token::bare_identifier) &&
           getTokenSpelling()[0] == 'x') {
      if (failed(parseXInDimensionList()) || failed(parseDim()))
        return failure();
    }
  }
  return success();
}

ParseResult Parser::parseIntegerInDimensionList(int64_t &value) {
  // Hexadecimal integer literals (starting with `0x`) are not allowed in
  // aggregate type declarations.  Therefore, `0xf32` should be processed as
  // a sequence of separate elements `0`, `x`, `f32`.
  if (getTokenSpelling().size() > 1 && getTokenSpelling()[1] == 'x') {
    // We can get here only if the token is an integer literal.  Hexadecimal
    // integer literals can only start with `0x` (`1x` wouldn't lex as a
    // literal, just `1` would, at which point we don't get into this
    // branch).
    assert(getTokenSpelling()[0] == '0' && "invalid integer literal");
    value = 0;
    state.lex.resetPointer(getTokenSpelling().data() + 1);
    consumeToken();
  } else {
    // Make sure this integer value is in bound and valid.
    std::optional<uint64_t> dimension = getToken().getUInt64IntegerValue();
    if (!dimension ||
        *dimension > (uint64_t)std::numeric_limits<int64_t>::max())
      return emitError("invalid dimension");
    value = (int64_t)*dimension;
    consumeToken(Token::integer);
  }
  return success();
}

/// Parse an 'x' token in a dimension list, handling the case where the x is
/// juxtaposed with an element type, as in "xf32", leaving the "f32" as the next
/// token.
ParseResult Parser::parseXInDimensionList() {
  if (getToken().isNot(Token::bare_identifier) || getTokenSpelling()[0] != 'x')
    return emitWrongTokenError("expected 'x' in dimension list");

  // If we had a prefix of 'x', lex the next token immediately after the 'x'.
  if (getTokenSpelling().size() != 1)
    state.lex.resetPointer(getTokenSpelling().data() + 1);

  // Consume the 'x'.
  consumeToken(Token::bare_identifier);

  return success();
}
