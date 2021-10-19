//===- FunctionImplementation.cpp - Utilities for function-like ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

ParseResult mlir::function_like_impl::parseFunctionArgumentList(
    OpAsmParser &parser, bool allowAttributes, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic) {
  if (parser.parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's and
  // sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::OperandType argument;
    Type argumentType;
    if (succeeded(parser.parseOptionalRegionArgument(argument)) &&
        !argument.name.empty()) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return parser.emitError(loc, "expected type instead of SSA identifier");
      argNames.push_back(argument);

      if (parser.parseColonType(argumentType))
        return failure();
    } else if (allowVariadic && succeeded(parser.parseOptionalEllipsis())) {
      isVariadic = true;
      return success();
    } else if (!argNames.empty()) {
      // Reject this if the preceding argument had a name.
      return parser.emitError(loc, "expected SSA identifier");
    } else if (parser.parseType(argumentType)) {
      return failure();
    }

    // Add the argument type.
    argTypes.push_back(argumentType);

    // Parse any argument attributes.
    NamedAttrList attrs;
    if (parser.parseOptionalAttrDict(attrs))
      return failure();
    if (!allowAttributes && !attrs.empty())
      return parser.emitError(loc, "expected arguments without attributes");
    argAttrs.push_back(attrs);

    // Parse a location if specified.  TODO: Don't drop it on the floor.
    Optional<Location> explicitLoc;
    if (!argument.name.empty() &&
        parser.parseOptionalLocationSpecifier(explicitLoc))
      return failure();

    return success();
  };

  // Parse the function arguments.
  isVariadic = false;
  if (failed(parser.parseOptionalRParen())) {
    do {
      unsigned numTypedArguments = argTypes.size();
      if (parseArgument())
        return failure();

      llvm::SMLoc loc = parser.getCurrentLocation();
      if (argTypes.size() == numTypedArguments &&
          succeeded(parser.parseOptionalComma()))
        return parser.emitError(
            loc, "variadic arguments must be in the end of the argument list");
    } while (succeeded(parser.parseOptionalComma()));
    parser.parseRParen();
  }

  return success();
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///                          | non-function-type
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= type attribute-dict?
///
static ParseResult
parseFunctionResultList(OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<NamedAttrList> &resultAttrs) {
  if (failed(parser.parseOptionalLParen())) {
    // We already know that there is no `(`, so parse a type.
    // Because there is no `(`, it cannot be a function type.
    Type ty;
    if (parser.parseType(ty))
      return failure();
    resultTypes.push_back(ty);
    resultAttrs.emplace_back();
    return success();
  }

  // Special case for an empty set of parens.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  // Parse individual function results.
  do {
    resultTypes.emplace_back();
    resultAttrs.emplace_back();
    if (parser.parseType(resultTypes.back()) ||
        parser.parseOptionalAttrDict(resultAttrs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// Parses a function signature using `parser`. The `allowVariadic` argument
/// indicates whether functions with variadic arguments are supported. The
/// trailing arguments are populated by this function with names, types and
/// attributes of the arguments and those of the results.
ParseResult mlir::function_like_impl::parseFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs) {
  bool allowArgAttrs = true;
  if (parseFunctionArgumentList(parser, allowArgAttrs, allowVariadic, argNames,
                                argTypes, argAttrs, isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return success();
}

/// Implementation of `addArgAndResultAttrs` that is attribute list type
/// agnostic.
template <typename AttrListT, typename AttrArrayBuildFnT>
static void addArgAndResultAttrsImpl(Builder &builder, OperationState &result,
                                     ArrayRef<AttrListT> argAttrs,
                                     ArrayRef<AttrListT> resultAttrs,
                                     AttrArrayBuildFnT &&buildAttrArrayFn) {
  auto nonEmptyAttrsFn = [](const AttrListT &attrs) { return !attrs.empty(); };

  // Add the attributes to the function arguments.
  if (!argAttrs.empty() && llvm::any_of(argAttrs, nonEmptyAttrsFn)) {
    ArrayAttr attrDicts = builder.getArrayAttr(buildAttrArrayFn(argAttrs));
    result.addAttribute(function_like_impl::getArgDictAttrName(), attrDicts);
  }
  // Add the attributes to the function results.
  if (!resultAttrs.empty() && llvm::any_of(resultAttrs, nonEmptyAttrsFn)) {
    ArrayAttr attrDicts = builder.getArrayAttr(buildAttrArrayFn(resultAttrs));
    result.addAttribute(function_like_impl::getResultDictAttrName(), attrDicts);
  }
}

void mlir::function_like_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result, ArrayRef<DictionaryAttr> argAttrs,
    ArrayRef<DictionaryAttr> resultAttrs) {
  auto buildFn = [](ArrayRef<DictionaryAttr> attrs) {
    return ArrayRef<Attribute>(attrs.data(), attrs.size());
  };
  addArgAndResultAttrsImpl(builder, result, argAttrs, resultAttrs, buildFn);
}
void mlir::function_like_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result, ArrayRef<NamedAttrList> argAttrs,
    ArrayRef<NamedAttrList> resultAttrs) {
  MLIRContext *context = builder.getContext();
  auto buildFn = [=](ArrayRef<NamedAttrList> attrs) {
    return llvm::to_vector<8>(
        llvm::map_range(attrs, [=](const NamedAttrList &attrList) -> Attribute {
          return attrList.getDictionary(context);
        }));
  };
  addArgAndResultAttrsImpl(builder, result, argAttrs, resultAttrs, buildFn);
}

/// Parser implementation for function-like operations.  Uses `funcTypeBuilder`
/// to construct the custom function type given lists of input and output types.
ParseResult mlir::function_like_impl::parseFunctionLikeOp(
    OpAsmParser &parser, OperationState &result, bool allowVariadic,
    FuncTypeBuilder funcTypeBuilder) {
  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility.
  impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  llvm::SMLoc signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (parseFunctionSignature(parser, allowVariadic, entryArgs, argTypes,
                             argAttrs, isVariadic, resultTypes, resultAttrs))
    return failure();

  std::string errorMessage;
  Type type = funcTypeBuilder(builder, argTypes, resultTypes,
                              VariadicFlag(isVariadic), errorMessage);
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;
  }
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        getTypeAttrName()}) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult = parser.parseOptionalRegion(
      *body, entryArgs, entryArgs.empty() ? ArrayRef<Type>() : argTypes,
      /*enableNameShadowing=*/false);
  if (parseResult.hasValue()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

/// Print a function result list. The provided `attrs` must either be null, or
/// contain a set of DictionaryAttrs of the same arity as `types`.
static void printFunctionResultList(OpAsmPrinter &p, ArrayRef<Type> types,
                                    ArrayAttr attrs) {
  assert(!types.empty() && "Should not be called for empty result list.");
  assert((!attrs || attrs.size() == types.size()) &&
         "Invalid number of attributes.");

  auto &os = p.getStream();
  bool needsParens = types.size() > 1 || types[0].isa<FunctionType>() ||
                     (attrs && !attrs[0].cast<DictionaryAttr>().empty());
  if (needsParens)
    os << '(';
  llvm::interleaveComma(llvm::seq<size_t>(0, types.size()), os, [&](size_t i) {
    p.printType(types[i]);
    if (attrs)
      p.printOptionalAttrDict(attrs[i].cast<DictionaryAttr>().getValue());
  });
  if (needsParens)
    os << ')';
}

/// Print the signature of the function-like operation `op`.  Assumes `op` has
/// the FunctionLike trait and passed the verification.
void mlir::function_like_impl::printFunctionSignature(
    OpAsmPrinter &p, Operation *op, ArrayRef<Type> argTypes, bool isVariadic,
    ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  ArrayAttr argAttrs = op->getAttrOfType<ArrayAttr>(getArgDictAttrName());
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    if (!isExternal) {
      ArrayRef<NamedAttribute> attrs;
      if (argAttrs)
        attrs = argAttrs[i].cast<DictionaryAttr>().getValue();
      p.printRegionArgument(body.getArgument(i), attrs);
    } else {
      p.printType(argTypes[i]);
      if (argAttrs)
        p.printOptionalAttrDict(argAttrs[i].cast<DictionaryAttr>().getValue());
    }
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  if (!resultTypes.empty()) {
    p.getStream() << " -> ";
    auto resultAttrs = op->getAttrOfType<ArrayAttr>(getResultDictAttrName());
    printFunctionResultList(p, resultTypes, resultAttrs);
  }
}

/// Prints the list of function prefixed with the "attributes" keyword. The
/// attributes with names listed in "elided" as well as those used by the
/// function-like operation internally are not printed. Nothing is printed
/// if all attributes are elided. Assumes `op` has the `FunctionLike` trait and
/// passed the verification.
void mlir::function_like_impl::printFunctionAttributes(
    OpAsmPrinter &p, Operation *op, unsigned numInputs, unsigned numResults,
    ArrayRef<StringRef> elided) {
  // Print out function attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      ::mlir::SymbolTable::getSymbolAttrName(), getTypeAttrName(),
      getArgDictAttrName(), getResultDictAttrName()};
  ignoredAttrs.append(elided.begin(), elided.end());

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);
}

/// Printer implementation for function-like operations.  Accepts lists of
/// argument and result types to use while printing.
void mlir::function_like_impl::printFunctionLikeOp(OpAsmPrinter &p,
                                                   Operation *op,
                                                   ArrayRef<Type> argTypes,
                                                   bool isVariadic,
                                                   ArrayRef<Type> resultTypes) {
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);

  printFunctionSignature(p, op, argTypes, isVariadic, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          {visibilityAttrName});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}
