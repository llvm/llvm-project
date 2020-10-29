//===- TosaOps.cpp - MLIR Dialect for TOSA ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TOSA Specification:
// https://developer.mlplatform.org/w/tosa/
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaStructs.cc.inc"

// Tosa dialect interfaces

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cc.inc"

// Dialect Function Inliner Interface

struct TosaInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *region,
                       BlockAndValueMapping &map) const {
    return true;
  }

  /* This ensures the callable region of the operator can be inlined.
     Without this, the regions will NOT inline. */
  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &map) const {
    return (isa<tosa::IfOp>(dest->getParentOp()) ||
            isa<tosa::WhileOp>(dest->getParentOp()));
  }
};

// TOSA control flow support

Region &tosa::WhileOp::getLoopBody() { return body(); }

bool tosa::WhileOp::isDefinedOutsideOfLoop(Value value) {
  // WIP MLIR enhancements with exposed API
  return false;
}

LogicalResult WhileOp::moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return success();

  Operation *tosa_while_op = this->getOperation();
  for (auto op : ops)
    op->moveBefore(tosa_while_op);

  return success();
}

struct TosaDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const final {
    return isa<tosa::WhileOp>(region->getParentOp());
  }
};

// Tosa Dialect

TosaDialect::TosaDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TosaDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cc.inc"
      >();
  addInterfaces<TosaInlinerInterface, TosaDialectFoldInterface>();

  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

void ConstOp::build(OpBuilder &builder, OperationState &result, Type type,
                    Attribute value) {
  result.addTypes(type);
  result.addAttribute("value", value);
  return;
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cc.inc"

Operation *TosaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (value.isa<OpaqueElementsAttr>() ||
      (value.isa<ElementsAttr>() && value.getType() != type))
    return builder.create<tosa::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}
