//===- GENXDialect.cpp - GENX IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the GENX IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The GENX dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace GENX;

#include "mlir/Dialect/LLVMIR/GENXOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/GENXOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// GENXDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

// TODO: This should be the llvm.GENX dialect once this is supported.
void GENXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all GENX operations are registered.
  allowUnknownOperations();
}

LogicalResult GENXDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // TODO: fill this in.
  return success();
}

static mlir::ParseResult parseMemoryFenceFlags(OpAsmParser &parser,
                                               IntegerAttr &flagsAttr) {
  MemoryFenceFlagAttr memoryFenceFlagAttr;
  int flags = 0;
  do {
    if (parser.parseCustomAttributeWithFallback(memoryFenceFlagAttr))
      return failure();
    flags |= static_cast<int>(memoryFenceFlagAttr.getValue());
  } while (succeeded(parser.parseOptionalComma()));
  flagsAttr =
      IntegerAttr::get(IntegerType::get(parser.getContext(), 32), flags);
  return success();
}

static void printMemoryFenceFlags(OpAsmPrinter &p, FenceOp op,
                                  IntegerAttr flags) {
  bool firstFlag = true;
  auto printFlag = [&](int flag) {
    assert(flag == 1 | flag == 2 | flag == 4 &&
           "Expecting valid memory fence flag");
    if (!firstFlag)
      p << ",";
    p.printStrippedAttrOrType(MemoryFenceFlagAttr::get(
        flags.getContext(), static_cast<MemoryFenceFlag>(flag)));
    firstFlag = false;
  };
  if (flags.getInt() & 1)
    printFlag(1);
  if (flags.getInt() & 2)
    printFlag(2);
  if (flags.getInt() & 4)
    printFlag(4);
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"
