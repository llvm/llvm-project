//===- CallImplementation.pp - Call and Callable Op utilities -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

static ParseResult
parseTypeAndAttrList(OpAsmParser &parser, SmallVectorImpl<Type> &types,
                     SmallVectorImpl<DictionaryAttr> &attrs) {
  // Parse individual function results.
  return parser.parseCommaSeparatedList([&]() -> ParseResult {
    types.emplace_back();
    attrs.emplace_back();
    NamedAttrList attrList;
    if (parser.parseType(types.back()) ||
        parser.parseOptionalAttrDict(attrList))
      return failure();
    attrs.back() = attrList.getDictionary(parser.getContext());
    return success();
  });
}

ParseResult call_interface_impl::parseFunctionResultList(
    OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
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
  if (parseTypeAndAttrList(parser, resultTypes, resultAttrs))
    return failure();
  return parser.parseRParen();
}

ParseResult call_interface_impl::parseFunctionSignature(
    OpAsmParser &parser, SmallVectorImpl<Type> &argTypes,
    SmallVectorImpl<DictionaryAttr> &argAttrs,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<DictionaryAttr> &resultAttrs, bool mustParseEmptyResult) {
  // Parse arguments.
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parseTypeAndAttrList(parser, argTypes, argAttrs))
      return failure();
    if (parser.parseRParen())
      return failure();
  }
  // Parse results.
  if (succeeded(parser.parseOptionalArrow()))
    return call_interface_impl::parseFunctionResultList(parser, resultTypes,
                                                        resultAttrs);
  if (mustParseEmptyResult)
    return failure();
  return success();
}

/// Print a function result list. The provided `attrs` must either be null, or
/// contain a set of DictionaryAttrs of the same arity as `types`.
static void printFunctionResultList(OpAsmPrinter &p, TypeRange types,
                                    ArrayAttr attrs) {
  assert(!types.empty() && "Should not be called for empty result list.");
  assert((!attrs || attrs.size() == types.size()) &&
         "Invalid number of attributes.");

  auto &os = p.getStream();
  bool needsParens = types.size() > 1 || llvm::isa<FunctionType>(types[0]) ||
                     (attrs && !llvm::cast<DictionaryAttr>(attrs[0]).empty());
  if (needsParens)
    os << '(';
  llvm::interleaveComma(llvm::seq<size_t>(0, types.size()), os, [&](size_t i) {
    p.printType(types[i]);
    if (attrs)
      p.printOptionalAttrDict(llvm::cast<DictionaryAttr>(attrs[i]).getValue());
  });
  if (needsParens)
    os << ')';
}

void call_interface_impl::printFunctionSignature(
    OpAsmPrinter &p, ArgumentAttributesOpInterface op, TypeRange argTypes,
    bool isVariadic, TypeRange resultTypes, Region *body,
    bool printEmptyResult) {
  bool isExternal = !body || body->empty();
  ArrayAttr argAttrs = op.getArgAttrsAttr();
  ArrayAttr resultAttrs = op.getResAttrsAttr();
  if (!isExternal && !isVariadic && !argAttrs && !resultAttrs &&
      printEmptyResult) {
    p.printFunctionalType(argTypes, resultTypes);
    return;
  }

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    if (!isExternal) {
      ArrayRef<NamedAttribute> attrs;
      if (argAttrs)
        attrs = llvm::cast<DictionaryAttr>(argAttrs[i]).getValue();
      p.printRegionArgument(body->getArgument(i), attrs);
    } else {
      p.printType(argTypes[i]);
      if (argAttrs)
        p.printOptionalAttrDict(
            llvm::cast<DictionaryAttr>(argAttrs[i]).getValue());
    }
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  if (!resultTypes.empty()) {
    p << " -> ";
    printFunctionResultList(p, resultTypes, resultAttrs);
  } else if (printEmptyResult) {
    p << " -> ()";
  }
}

void call_interface_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result, ArrayRef<DictionaryAttr> argAttrs,
    ArrayRef<DictionaryAttr> resultAttrs, StringAttr argAttrsName,
    StringAttr resAttrsName) {
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
    result.addAttribute(argAttrsName, getArrayAttr(argAttrs));

  // Add the attributes to the function results.
  if (llvm::any_of(resultAttrs, nonEmptyAttrsFn))
    result.addAttribute(resAttrsName, getArrayAttr(resultAttrs));
}

void call_interface_impl::addArgAndResultAttrs(
    Builder &builder, OperationState &result,
    ArrayRef<OpAsmParser::Argument> args, ArrayRef<DictionaryAttr> resultAttrs,
    StringAttr argAttrsName, StringAttr resAttrsName) {
  SmallVector<DictionaryAttr> argAttrs;
  for (const auto &arg : args)
    argAttrs.push_back(arg.attrs);
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs, argAttrsName,
                       resAttrsName);
}
