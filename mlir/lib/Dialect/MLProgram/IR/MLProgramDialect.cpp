//===- MLProgramDialect.cpp - MLProgram dialect implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ml_program;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgramOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/MLProgram/IR/MLProgramTypes.cpp.inc"

namespace {

struct MLProgramInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    // We have no specific opinion on whether ops defined in this dialect should
    // be inlined.
    return true;
  }
};

struct MLProgramOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<ExternAttr>(attr)) {
      os << "extern";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void ml_program::MLProgramDialect::initialize() {
#define GET_ATTRDEF_LIST
  addAttributes<
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.cpp.inc"
      >();

#define GET_TYPEDEF_LIST
  addTypes<
#include "mlir/Dialect/MLProgram/IR/MLProgramTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MLProgram/IR/MLProgramOps.cpp.inc"
      >();

  addInterfaces<MLProgramInlinerInterface, MLProgramOpAsmDialectInterface>();
}
