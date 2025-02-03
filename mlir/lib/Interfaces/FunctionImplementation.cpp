//===- FunctionImplementation.cpp - Utilities for function-like ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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

ParseResult function_interface_impl::parseFunctionSignatureWithArguments(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &arguments, bool &isVariadic,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<DictionaryAttr> &resultAttrs) {
  if (parseFunctionArgumentList(parser, allowVariadic, arguments, isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return call_interface_impl::parseFunctionResultList(parser, resultTypes,
                                                        resultAttrs);
  return success();
}

ParseResult function_interface_impl::parseFunctionOp(
    OpAsmParser &parser, OperationState &result, bool allowVariadic,
    StringAttr typeAttrName, FuncTypeBuilder funcTypeBuilder,
    StringAttr argAttrsName, StringAttr resAttrsName) {
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
  if (parseFunctionSignatureWithArguments(parser, allowVariadic, entryArgs,
                                          isVariadic, resultTypes, resultAttrs))
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
  result.addAttribute(typeAttrName, TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        typeAttrName.getValue()}) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, argAttrsName, resAttrsName);

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

void function_interface_impl::printFunctionAttributes(
    OpAsmPrinter &p, Operation *op, ArrayRef<StringRef> elided) {
  // Print out function attributes, if present.
  SmallVector<StringRef, 8> ignoredAttrs = {SymbolTable::getSymbolAttrName()};
  ignoredAttrs.append(elided.begin(), elided.end());

  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);
}

void function_interface_impl::printFunctionOp(
    OpAsmPrinter &p, FunctionOpInterface op, bool isVariadic,
    StringRef typeAttrName, StringAttr argAttrsName, StringAttr resAttrsName) {
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
  printFunctionAttributes(
      p, op, {visibilityAttrName, typeAttrName, argAttrsName, resAttrsName});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}
