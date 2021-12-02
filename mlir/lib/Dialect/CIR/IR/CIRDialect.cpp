//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/CIR/IR/CIRAttrs.h"
#include "mlir/Dialect/CIR/IR/CIRTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::cir;

#include "mlir/Dialect/CIR/IR/CIROpsEnums.cpp.inc"

#include "mlir/Dialect/CIR/IR/CIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto opType = getType();
  auto val = getValue();
  auto valueType = val.getType();

  if (val.isa<NullAttr>()) {
    if (opType.isa<::mlir::cir::PointerType>())
      return success();
    return emitOpError("nullptr expects pointer type");
  }

  // ODS already generates checks to make sure the result type is valid. We just
  // need to additionally check that the value's attribute type is consistent
  // with the result type.
  if (val.isa<BoolAttr>()) {
    if (!opType.isa<mlir::cir::BoolType>())
      return emitOpError("result type (")
             << opType << ") must be '!cir.bool' for '" << val << "'";
    return success();
  }

  if (opType.isa<IntegerType, FloatType>()) {
    if (valueType != opType)
      return emitOpError("result type (")
             << opType << ") does not match value type (" << valueType << ")";
    return success();
  }

  return emitOpError("cannot have value of type ") << valueType;
}

static ParseResult parseConstantValue(OpAsmParser &parser,
                                      mlir::Attribute &valueAttr) {
  NamedAttrList attr;
  if (parser.parseAttribute(valueAttr, "value", attr))
    return ::mlir::failure();

  return success();
}

static void printConstantValue(OpAsmPrinter &p, cir::ConstantOp op,
                               Attribute value) {
  p.printAttribute(value);
}

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(getOperation()->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType)
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type boolType = ::mlir::cir::BoolType::get(builder.getContext());

  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, boolType, result.operands))
    return failure();

  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition() << " ";
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = this->getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

Block *IfOp::thenBlock() { return &getThenRegion().back(); }
Block *IfOp::elseBlock() {
  Region &r = getElseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}

/// Default callback for IfOp builders. Inserts nothing for now.
void mlir::cir::buildTerminatedBody(OpBuilder &builder, Location loc) {}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  // bool condition;
  // if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
  //   assert(0 && "not implemented");
  // condition = condAttr.getValue().isOneValue();
  // Add the successor regions using the condition.
  // regions.push_back(RegionSuccessor(condition ? &thenRegion() :
  // elseRegion));
  // return;
  // }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  // If the else region does not exist, it is not a viable successor.
  if (elseRegion)
    regions.push_back(RegionSuccessor(elseRegion));
  return;
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!withElseRegion)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

LogicalResult IfOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

ParseResult ScopeOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create one region within 'scope'.
  result.regions.reserve(1);
  Region *scopeRegion = result.addRegion();

  // Parse the scope region.
  if (parser.parseRegion(*scopeRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  ScopeOp::ensureTerminator(*scopeRegion, parser.getBuilder(), result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void ScopeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printRegion(getScopeRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  p.printOptionalAttrDict(getOperation()->getAttrs());
}

Block *ScopeOp::scopeBlock() { return &getScopeRegion().back(); }

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ScopeOp::getSuccessorRegions(mlir::RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The only region always branch back to the parent operation.
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor());
    return;
  }

  // If the condition isn't constant, both regions may be executed.
  regions.push_back(RegionSuccessor(&getScopeRegion()));
}

void ScopeOp::build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTypes,
                    function_ref<void(OpBuilder &, Location)> scopeBuilder) {
  assert(scopeBuilder && "the builder callback for 'then' must be present");
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *scopeRegion = result.addRegion();
  builder.createBlock(scopeRegion);
  scopeBuilder(builder, result.location);
}

LogicalResult ScopeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  if (!llvm::isa<IfOp, ScopeOp>(getOperation()->getParentOp()))
    return emitOpError()
           << "expects 'cir.if' or 'cir.scope' as the parent operation'";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/CIR/IR/CIROps.cpp.inc"
