//===- Ops.cpp - Loop MLIR Operations -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

using namespace mlir;
using namespace mlir::loop;

//===----------------------------------------------------------------------===//
// LoopOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

struct LoopSideEffectsInterface : public SideEffectsDialectInterface {
  using SideEffectsDialectInterface::SideEffectsDialectInterface;

  SideEffecting isSideEffecting(Operation *op) const override {
    if (isa<IfOp>(op) || isa<ForOp>(op)) {
      return Recursive;
    }
    return SideEffectsDialectInterface::isSideEffecting(op);
  };
};

} // namespace

//===----------------------------------------------------------------------===//
// LoopOpsDialect
//===----------------------------------------------------------------------===//

LoopOpsDialect::LoopOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
      >();
  addInterfaces<LoopSideEffectsInterface>();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(Builder *builder, OperationState &result, Value lb, Value ub,
                  Value step) {
  result.addOperands({lb, ub, step});
  Region *bodyRegion = result.addRegion();
  ForOp::ensureTerminator(*bodyRegion, *builder, result.location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

static LogicalResult verify(ForOp op) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(op.step().getDefiningOp()))
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (body->getNumArguments() != 1 || !body->getArgument(0).getType().isIndex())
    return op.emitOpError("expected body to have a single index argument for "
                          "the induction variable");
  return success();
}

static void print(OpAsmPrinter &p, ForOp op) {
  p << op.getOperationName() << " " << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(op.getAttrs());
}

static ParseResult parseForOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser.parseRegionArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable, indexType))
    return failure();

  ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Region &ForOp::getLoopBody() { return region(); }

bool ForOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult ForOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto op : ops)
    op->moveBefore(this->getOperation());
  return success();
}

ForOp mlir::loop::getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<ForOp>(containingInst);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void IfOp::build(Builder *builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  result.addOperands(cond);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  IfOp::ensureTerminator(*thenRegion, *builder, result.location);
  if (withElseRegion)
    IfOp::ensureTerminator(*elseRegion, *builder, result.location);
}

static LogicalResult verify(IfOp op) {
  // Verify that the entry of each child region does not have arguments.
  for (auto &region : op.getOperation()->getRegions()) {
    if (region.empty())
      continue;

    for (auto &b : region)
      if (b.getNumArguments() != 0)
        return op.emitOpError(
            "requires that child entry blocks have no arguments");
  }
  return success();
}

static ParseResult parseIfOp(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
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

static void print(OpAsmPrinter &p, IfOp op) {
  p << IfOp::getOperationName() << " " << op.condition();
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }

  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(Builder *builder, OperationState &result, ValueRange lbs,
                       ValueRange ubs, ValueRange steps) {
  result.addOperands(lbs);
  result.addOperands(ubs);
  result.addOperands(steps);
  Region *bodyRegion = result.addRegion();
  ParallelOp::ensureTerminator(*bodyRegion, *builder, result.location);
  for (size_t i = 0; i < steps.size(); ++i)
    bodyRegion->front().addArgument(builder->getIndexType());
}

static LogicalResult verify(ParallelOp op) {
  // Check that there is at least one value in lowerBound, upperBound and step.
  // It is sufficient to test only step, because it is ensured already that the
  // number of elements in lowerBound, upperBound and step are the same.
  Operation::operand_range stepValues = op.step();
  if (stepValues.empty())
    return op.emitOpError(
        "needs at least one tuple element for lowerBound, upperBound and step");

  // Check whether all constant step values are positive.
  for (Value stepValue : stepValues)
    if (auto cst = dyn_cast_or_null<ConstantIndexOp>(stepValue.getDefiningOp()))
      if (cst.getValue() <= 0)
        return op.emitOpError("constant step operand must be positive");

  // Check that the body defines the same number of block arguments as the
  // number of tuple elements in step.
  Block *body = op.getBody();
  if (body->getNumArguments() != stepValues.size())
    return op.emitOpError(
        "expects the same number of induction variables as bound and step "
        "values");
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return op.emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the number of results is the same as the number of ReduceOps.
  SmallVector<ReduceOp, 4> reductions(body->getOps<ReduceOp>());
  if (op.results().size() != reductions.size())
    return op.emitOpError(
        "expects number of results to be the same as number of reductions");

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(op.results(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.operand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce to be the same as result type: "
             << resultType;
  }
  return success();
}

static ParseResult parseParallelOp(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::OperandType, 4> ivs;
  if (parser.parseRegionArgumentList(ivs, /*requiredOperandCount=*/-1,
                                     OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::OperandType, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::OperandType, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step value.
  SmallVector<OpAsmParser::OperandType, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type, 4> types(ivs.size(), builder.getIndexType());
  if (parser.parseRegion(*body, ivs, types))
    return failure();

  // Parse attributes and optional results (in case there is a reduce).
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseOptionalColonTypeList(result.types))
    return failure();

  // Add a terminator if none was parsed.
  ForOp::ensureTerminator(*body, builder, result.location);

  return success();
}

static void print(OpAsmPrinter &p, ParallelOp op) {
  p << op.getOperationName() << " (";
  p.printOperands(op.getBody()->getArguments());
  p << ") = (" << op.lowerBound() << ") to (" << op.upperBound() << ") step ("
    << op.step() << ")";
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op.getAttrs());
  if (!op.results().empty())
    p << " : " << op.getResultTypes();
}

ParallelOp mlir::loop::getParallelForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ParallelOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingInst = ivArg.getOwner()->getParentOp();
  return dyn_cast<ParallelOp>(containingInst);
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReduceOp op) {
  // The region of a ReduceOp has two arguments of the same type as its operand.
  auto type = op.operand().getType();
  Block &block = op.reductionOperator().front();
  if (block.empty())
    return op.emitOpError("the block inside reduce should not be empty");
  if (block.getNumArguments() != 2 ||
      llvm::any_of(block.getArguments(), [&](const BlockArgument &arg) {
        return arg.getType() != type;
      }))
    return op.emitOpError()
           << "expects two arguments to reduce block of type " << type;

  // Check that the block is terminated by a ReduceReturnOp.
  if (!isa<ReduceReturnOp>(block.getTerminator()))
    return op.emitOpError("the block inside reduce should be terminated with a "
                          "'loop.reduce.return' op");

  return success();
}

static ParseResult parseReduceOp(OpAsmParser &parser, OperationState &result) {
  // Parse an opening `(` followed by the reduced value followed by `)`
  OpAsmParser::OperandType operand;
  if (parser.parseLParen() || parser.parseOperand(operand) ||
      parser.parseRParen())
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // And the type of the operand (and also what reduce computes on).
  Type resultType;
  if (parser.parseColonType(resultType) ||
      parser.resolveOperand(operand, resultType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ReduceOp op) {
  p << op.getOperationName() << "(" << op.operand() << ") ";
  p.printRegion(op.reductionOperator());
  p << " : " << op.operand().getType();
}

//===----------------------------------------------------------------------===//
// ReduceReturnOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReduceReturnOp op) {
  // The type of the return value should be the same type as the type of the
  // operand of the enclosing ReduceOp.
  auto reduceOp = cast<ReduceOp>(op.getParentOp());
  Type reduceType = reduceOp.operand().getType();
  if (reduceType != op.result().getType())
    return op.emitOpError() << "needs to have type " << reduceType
                            << " (the type of the enclosing ReduceOp)";
  return success();
}

static ParseResult parseReduceReturnOp(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::OperandType operand;
  Type resultType;
  if (parser.parseOperand(operand) || parser.parseColonType(resultType) ||
      parser.resolveOperand(operand, resultType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ReduceReturnOp op) {
  p << op.getOperationName() << " " << op.result() << " : "
    << op.result().getType();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
