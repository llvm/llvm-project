//===- FunctionImplementation.cpp - Utilities for function-like ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

static ParseResult
parseFunctionArgumentList(OpAsmParser &parser, bool allowVariadic,
                          SmallVectorImpl<OpAsmParser::Argument> &arguments,
                          bool &isVariadic) {

  // Parse the function arguments.  The argument list either has to consistently
  // have ssa-id's followed by types, or just be a type list.  It isn't ok to
  // sometimes have SSA ID's and sometimes not.
  isVariadic = false;

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        // Ellipsis must be at end of the list.
        if (isVariadic)
          return parser.emitError(
              parser.getCurrentLocation(),
              "variadic arguments must be in the end of the argument list");

        // Handle ellipsis as a special case.
        if (allowVariadic && succeeded(parser.parseOptionalEllipsis())) {
          // This is a variadic designator.
          isVariadic = true;
          return success(); // Stop parsing arguments.
        }
        // Parse argument name if present.
        OpAsmParser::Argument argument;
        auto argPresent = parser.parseOptionalArgument(
            argument, /*allowType=*/true, /*allowAttrs=*/true);
        if (argPresent.has_value()) {
          if (failed(argPresent.value()))
            return failure(); // Present but malformed.

          // Reject this if the preceding argument was missing a name.
          if (!arguments.empty() && arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected type instead of SSA identifier");

        } else {
          argument.ssaName.location = parser.getCurrentLocation();
          // Otherwise we just have a type list without SSA names.  Reject
          // this if the preceding argument had a name.
          if (!arguments.empty() && !arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected SSA identifier");

          NamedAttrList attrs;
          if (parser.parseType(argument.type) ||
              parser.parseOptionalAttrDict(attrs) ||
              parser.parseOptionalLocationSpecifier(argument.sourceLoc))
            return failure();
          argument.attrs = attrs.getDictionary(parser.getContext());
        }
        arguments.push_back(argument);
        return success();
      });
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
                        SmallVectorImpl<DictionaryAttr> &resultAttrs) {
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
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        resultTypes.emplace_back();
        resultAttrs.emplace_back();
        NamedAttrList attrs;
        if (parser.parseType(resultTypes.back()) ||
            parser.parseOptionalAttrDict(attrs))
          return failure();
        resultAttrs.back() = attrs.getDictionary(parser.getContext());
        return success();
      }))
    return failure();

  return parser.parseRParen();
}

ParseResult mlir::function_interface_impl::parseFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &arguments, bool &isVariadic,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<DictionaryAttr> &resultAttrs) {
  if (parseFunctionArgumentList(parser, allowVariadic, arguments, isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return success();
}

void mlir::function_interface_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result, ArrayRef<DictionaryAttr> argAttrs,
    ArrayRef<DictionaryAttr> resultAttrs) {
  auto nonEmptyAttrsFn = [](DictionaryAttr attrs) {
    return attrs && !attrs.empty();
  };
  // Convert the specified array of dictionary attrs (which may have null
  // entries) to an ArrayAttr of dictionaries.
  auto getArrayAttr = [&](ArrayRef<DictionaryAttr> dictAttrs) {
    SmallVector<Attribute> attrs;
    for (auto &dict : dictAttrs)
      attrs.push_back(dict ? dict : builder.getDictionaryAttr({}));
    return builder.getArrayAttr(attrs);
  };

  // Add the attributes to the function arguments.
  if (llvm::any_of(argAttrs, nonEmptyAttrsFn))
    result.addAttribute(function_interface_impl::getArgDictAttrName(),
                        getArrayAttr(argAttrs));

  // Add the attributes to the function results.
  if (llvm::any_of(resultAttrs, nonEmptyAttrsFn))
    result.addAttribute(function_interface_impl::getResultDictAttrName(),
                        getArrayAttr(resultAttrs));
}

void mlir::function_interface_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result,
    ArrayRef<OpAsmParser::Argument> args,
    ArrayRef<DictionaryAttr> resultAttrs) {
  SmallVector<DictionaryAttr> argAttrs;
  for (const auto &arg : args)
    argAttrs.push_back(arg.attrs);
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);
}

ParseResult mlir::function_interface_impl::parseFunctionOp(
    OpAsmParser &parser, OperationState &result, bool allowVariadic,
    FuncTypeBuilder funcTypeBuilder) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility.
  (void)impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (parseFunctionSignature(parser, allowVariadic, entryArgs, isVariadic,
                             resultTypes, resultAttrs))
    return failure();

  std::string errorMessage;
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
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
  SMLoc attributeDictLocation = parser.getCurrentLocation();
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
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs);

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs,
                                 /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
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

void mlir::function_interface_impl::printFunctionSignature(
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

void mlir::function_interface_impl::printFunctionAttributes(
    OpAsmPrinter &p, Operation *op, unsigned numInputs, unsigned numResults,
    ArrayRef<StringRef> elided) {
  // Print out function attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      ::mlir::SymbolTable::getSymbolAttrName(), getTypeAttrName(),
      getArgDictAttrName(), getResultDictAttrName()};
  ignoredAttrs.append(elided.begin(), elided.end());

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);
}

void mlir::function_interface_impl::printFunctionOp(OpAsmPrinter &p,
                                                    FunctionOpInterface op,
                                                    bool isVariadic) {
  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, isVariadic, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          {visibilityAttrName});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}
