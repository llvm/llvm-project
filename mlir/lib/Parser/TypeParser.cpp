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
#include "mlir/IR/BuiltinTypes.h"

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
  case Token::kw_bf16:
  case Token::kw_f16:
  case Token::kw_f32:
  case Token::kw_f64:
  case Token::kw_index:
  case Token::kw_none:
  case Token::exclamation_identifier:
    return failure(!(type = parseType()));

  default:
    return llvm::None;
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

  llvm::SMLoc elementTypeLoc = getToken().getLoc();
  auto elementType = parseType();
  if (!elementType ||
      parseToken(Token::greater, "expected '>' in complex type"))
    return nullptr;
  if (!elementType.isa<FloatType>() && !elementType.isa<IntegerType>())
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

/// Parse the offset and strides from a strided layout specification.
///
///   strided-layout ::= `offset:` dimension `,` `strides: ` stride-list
///
ParseResult Parser::parseStridedLayout(int64_t &offset,
                                       SmallVectorImpl<int64_t> &strides) {
  // Parse offset.
  consumeToken(Token::kw_offset);
  if (!consumeIf(Token::colon))
    return emitError("expected colon after `offset` keyword");
  auto maybeOffset = getToken().getUnsignedIntegerValue();
  bool question = getToken().is(Token::question);
  if (!maybeOffset && !question)
    return emitError("invalid offset");
  offset = maybeOffset ? static_cast<int64_t>(maybeOffset.getValue())
                       : MemRefType::getDynamicStrideOrOffset();
  consumeToken();

  if (!consumeIf(Token::comma))
    return emitError("expected comma after offset value");

  // Parse stride list.
  if (!consumeIf(Token::kw_strides))
    return emitError("expected `strides` keyword after offset specification");
  if (!consumeIf(Token::colon))
    return emitError("expected colon after `strides` keyword");
  if (failed(parseStrideList(strides)))
    return emitError("invalid braces-enclosed stride list");
  if (llvm::any_of(strides, [](int64_t st) { return st == 0; }))
    return emitError("invalid memref stride");

  return success();
}

/// Parse a memref type.
///
///   memref-type ::= ranked-memref-type | unranked-memref-type
///
///   ranked-memref-type ::= `memref` `<` dimension-list-ranked type
///                          (`,` semi-affine-map-composition)? (`,`
///                          memory-space)? `>`
///
///   unranked-memref-type ::= `memref` `<*x` type (`,` memory-space)? `>`
///
///   semi-affine-map-composition ::= (semi-affine-map `,` )* semi-affine-map
///   memory-space ::= integer-literal /* | TODO: address-space-id */
///
Type Parser::parseMemRefType() {
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
  if (!elementType.isIntOrIndexOrFloat() &&
      !elementType.isa<VectorType, ComplexType>())
    return emitError(typeLoc, "invalid memref element type"), nullptr;

  // Parse semi-affine-map-composition.
  SmallVector<AffineMap, 2> affineMapComposition;
  Optional<unsigned> memorySpace;
  unsigned numDims = dimensions.size();

  auto parseElt = [&]() -> ParseResult {
    // Check for the memory space.
    if (getToken().is(Token::integer)) {
      if (memorySpace)
        return emitError("multiple memory spaces specified in memref type");
      memorySpace = getToken().getUnsignedIntegerValue();
      if (!memorySpace.hasValue())
        return emitError("invalid memory space in memref type");
      consumeToken(Token::integer);
      return success();
    }
    if (isUnranked)
      return emitError("cannot have affine map for unranked memref type");
    if (memorySpace)
      return emitError("expected memory space to be last in memref type");

    AffineMap map;
    llvm::SMLoc mapLoc = getToken().getLoc();
    if (getToken().is(Token::kw_offset)) {
      int64_t offset;
      SmallVector<int64_t, 4> strides;
      if (failed(parseStridedLayout(offset, strides)))
        return failure();
      // Construct strided affine map.
      map = makeStridedLinearLayoutMap(strides, offset, state.context);
    } else {
      // Parse an affine map attribute.
      auto affineMap = parseAttribute();
      if (!affineMap)
        return failure();
      auto affineMapAttr = affineMap.dyn_cast<AffineMapAttr>();
      if (!affineMapAttr)
        return emitError("expected affine map in memref type");
      map = affineMapAttr.getValue();
    }

    if (map.getNumDims() != numDims) {
      size_t i = affineMapComposition.size();
      return emitError(mapLoc, "memref affine map dimension mismatch between ")
             << (i == 0 ? Twine("memref rank") : "affine map " + Twine(i))
             << " and affine map" << i + 1 << ": " << numDims
             << " != " << map.getNumDims();
    }
    numDims = map.getNumResults();
    affineMapComposition.push_back(map);
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
    return UnrankedMemRefType::get(elementType, memorySpace.getValueOr(0));

  return MemRefType::get(dimensions, elementType, affineMapComposition,
                         memorySpace.getValueOr(0));
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
///   float-type ::= `f16` | `bf16` | `f32` | `f64`
///   none-type ::= `none`
///
Type Parser::parseNonFunctionType() {
  switch (getToken().getKind()) {
  default:
    return (emitError("expected non-function type"), nullptr);
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
    if (!width.hasValue())
      return (emitError("invalid integer width"), nullptr);
    if (width.getValue() > IntegerType::kMaxWidth) {
      emitError(getToken().getLoc(), "integer bitwidth is limited to ")
          << IntegerType::kMaxWidth << " bits";
      return nullptr;
    }

    IntegerType::SignednessSemantics signSemantics = IntegerType::Signless;
    if (Optional<bool> signedness = getToken().getIntTypeSignedness())
      signSemantics = *signedness ? IntegerType::Signed : IntegerType::Unsigned;

    consumeToken(Token::inttype);
    return IntegerType::get(width.getValue(), signSemantics, getContext());
  }

  // float-type
  case Token::kw_bf16:
    consumeToken(Token::kw_bf16);
    return builder.getBF16Type();
  case Token::kw_f16:
    consumeToken(Token::kw_f16);
    return builder.getF16Type();
  case Token::kw_f32:
    consumeToken(Token::kw_f32);
    return builder.getF32Type();
  case Token::kw_f64:
    consumeToken(Token::kw_f64);
    return builder.getF64Type();

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
  if (!elementType || parseToken(Token::greater, "expected '>' in tensor type"))
    return nullptr;
  if (!TensorType::isValidElementType(elementType))
    return emitError(elementTypeLoc, "invalid tensor element type"), nullptr;

  if (isUnranked)
    return UnrankedTensorType::get(elementType);
  return RankedTensorType::get(dimensions, elementType);
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

  return TupleType::get(types, getContext());
}

/// Parse a vector type.
///
///   vector-type ::= `vector` `<` non-empty-static-dimension-list type `>`
///   non-empty-static-dimension-list ::= decimal-literal `x`
///                                       static-dimension-list
///   static-dimension-list ::= (decimal-literal `x`)*
///
VectorType Parser::parseVectorType() {
  consumeToken(Token::kw_vector);

  if (parseToken(Token::less, "expected '<' in vector type"))
    return nullptr;

  SmallVector<int64_t, 4> dimensions;
  if (parseDimensionListRanked(dimensions, /*allowDynamic=*/false))
    return nullptr;
  if (dimensions.empty())
    return (emitError("expected dimension size in vector type"), nullptr);
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
    return emitError(typeLoc, "vector elements must be int or float type"),
           nullptr;

  return VectorType::get(dimensions, elementType);
}

/// Parse a dimension list of a tensor or memref type.  This populates the
/// dimension list, using -1 for the `?` dimensions if `allowDynamic` is set and
/// errors out on `?` otherwise.
///
///   dimension-list-ranked ::= (dimension `x`)*
///   dimension ::= `?` | decimal-literal
///
/// When `allowDynamic` is not set, this is used to parse:
///
///   static-dimension-list ::= (decimal-literal `x`)*
ParseResult
Parser::parseDimensionListRanked(SmallVectorImpl<int64_t> &dimensions,
                                 bool allowDynamic) {
  while (getToken().isAny(Token::integer, Token::question)) {
    if (consumeIf(Token::question)) {
      if (!allowDynamic)
        return emitError("expected static shape");
      dimensions.push_back(-1);
    } else {
      // Hexadecimal integer literals (starting with `0x`) are not allowed in
      // aggregate type declarations.  Therefore, `0xf32` should be processed as
      // a sequence of separate elements `0`, `x`, `f32`.
      if (getTokenSpelling().size() > 1 && getTokenSpelling()[1] == 'x') {
        // We can get here only if the token is an integer literal.  Hexadecimal
        // integer literals can only start with `0x` (`1x` wouldn't lex as a
        // literal, just `1` would, at which point we don't get into this
        // branch).
        assert(getTokenSpelling()[0] == '0' && "invalid integer literal");
        dimensions.push_back(0);
        state.lex.resetPointer(getTokenSpelling().data() + 1);
        consumeToken();
      } else {
        // Make sure this integer value is in bound and valid.
        auto dimension = getToken().getUnsignedIntegerValue();
        if (!dimension.hasValue())
          return emitError("invalid dimension");
        dimensions.push_back((int64_t)dimension.getValue());
        consumeToken(Token::integer);
      }
    }

    // Make sure we have an 'x' or something like 'xbf32'.
    if (parseXInDimensionList())
      return failure();
  }

  return success();
}

/// Parse an 'x' token in a dimension list, handling the case where the x is
/// juxtaposed with an element type, as in "xf32", leaving the "f32" as the next
/// token.
ParseResult Parser::parseXInDimensionList() {
  if (getToken().isNot(Token::bare_identifier) || getTokenSpelling()[0] != 'x')
    return emitError("expected 'x' in dimension list");

  // If we had a prefix of 'x', lex the next token immediately after the 'x'.
  if (getTokenSpelling().size() != 1)
    state.lex.resetPointer(getTokenSpelling().data() + 1);

  // Consume the 'x'.
  consumeToken(Token::bare_identifier);

  return success();
}

// Parse a comma-separated list of dimensions, possibly empty:
//   stride-list ::= `[` (dimension (`,` dimension)*)? `]`
ParseResult Parser::parseStrideList(SmallVectorImpl<int64_t> &dimensions) {
  if (!consumeIf(Token::l_square))
    return failure();
  // Empty list early exit.
  if (consumeIf(Token::r_square))
    return success();
  while (true) {
    if (consumeIf(Token::question)) {
      dimensions.push_back(MemRefType::getDynamicStrideOrOffset());
    } else {
      // This must be an integer value.
      int64_t val;
      if (getToken().getSpelling().getAsInteger(10, val))
        return emitError("invalid integer value: ") << getToken().getSpelling();
      // Make sure it is not the one value for `?`.
      if (ShapedType::isDynamic(val))
        return emitError("invalid integer value: ")
               << getToken().getSpelling()
               << ", use `?` to specify a dynamic dimension";
      dimensions.push_back(val);
      consumeToken(Token::integer);
    }
    if (!consumeIf(Token::comma))
      break;
  }
  if (!consumeIf(Token::r_square))
    return failure();
  return success();
}
