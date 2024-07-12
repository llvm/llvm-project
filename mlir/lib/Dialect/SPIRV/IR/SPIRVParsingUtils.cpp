//===- SPIRVParsingUtilities.cpp - MLIR SPIR-V Dialect Parsing Utils-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements common SPIR-V dialect parsing functions.
//
//===----------------------------------------------------------------------===//

#include "SPIRVParsingUtils.h"

#include "llvm/ADT/StringExtras.h"

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

ParseResult parseVariableDecorations(OpAsmParser &parser,
                                     OperationState &state) {
  auto builtInName = llvm::convertToSnakeFromCamelCase(
      stringifyDecoration(spirv::Decoration::BuiltIn));
  if (succeeded(parser.parseOptionalKeyword("bind"))) {
    Attribute set, binding;
    // Parse optional descriptor binding
    auto descriptorSetName = llvm::convertToSnakeFromCamelCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingName = llvm::convertToSnakeFromCamelCase(
        stringifyDecoration(spirv::Decoration::Binding));
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseLParen() ||
        parser.parseAttribute(set, i32Type, descriptorSetName,
                              state.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(binding, i32Type, bindingName,
                              state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  } else if (succeeded(parser.parseOptionalKeyword(builtInName))) {
    StringAttr builtIn;
    if (parser.parseLParen() ||
        parser.parseAttribute(builtIn, builtInName, state.attributes) ||
        parser.parseRParen()) {
      return failure();
    }
  }

  // Parse other attributes
  if (parser.parseOptionalAttrDict(state.attributes))
    return failure();

  return success();
}

} // namespace mlir::spirv
