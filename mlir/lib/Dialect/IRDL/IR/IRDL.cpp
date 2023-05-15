//===- IRDL.cpp - IRDL dialect ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::irdl;

//===----------------------------------------------------------------------===//
// IRDL dialect.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.cpp.inc"

#include "mlir/Dialect/IRDL/IR/IRDLDialect.cpp.inc"

void IRDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing/Printing
//===----------------------------------------------------------------------===//

/// Parse a region, and add a single block if the region is empty.
/// If no region is parsed, create a new region with a single empty block.
static ParseResult parseSingleBlockRegion(OpAsmParser &p, Region &region) {
  auto regionParseRes = p.parseOptionalRegion(region);
  if (regionParseRes.has_value() && failed(regionParseRes.value()))
    return failure();

  // If the region is empty, add a single empty block.
  if (region.empty())
    region.push_back(new Block());

  return success();
}

static void printSingleBlockRegion(OpAsmPrinter &p, Operation *op,
                                   Region &region) {
  if (!region.getBlocks().front().empty())
    p.printRegion(region);
}

LogicalResult DialectOp::verify() {
  if (!Dialect::isValidNamespace(getName()))
    return emitOpError("invalid dialect name");
  return success();
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.cpp.inc"
