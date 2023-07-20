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

using namespace mlir::spirv::AttrNames;

namespace mlir::spirv {

ParseResult parseMemoryAccessAttributes(OpAsmParser &parser,
                                        OperationState &state,
                                        StringRef attrName) {
  // Parse an optional list of attributes staring with '['
  if (parser.parseOptionalLSquare()) {
    // Nothing to do
    return success();
  }

  spirv::MemoryAccess memoryAccessAttr;
  if (spirv::parseEnumStrAttr<spirv::MemoryAccessAttr>(memoryAccessAttr, parser,
                                                       state, attrName))
    return failure();

  if (spirv::bitEnumContainsAll(memoryAccessAttr,
                                spirv::MemoryAccess::Aligned)) {
    // Parse integer attribute for alignment.
    Attribute alignmentAttr;
    Type i32Type = parser.getBuilder().getIntegerType(32);
    if (parser.parseComma() ||
        parser.parseAttribute(alignmentAttr, i32Type,
                              AttrNames::kAlignmentAttrName,
                              state.attributes)) {
      return failure();
    }
  }
  return parser.parseRSquare();
}

} // namespace mlir::spirv
