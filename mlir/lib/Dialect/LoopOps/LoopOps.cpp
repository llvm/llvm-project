//===- Ops.cpp - Loop MLIR Operations -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

using namespace mlir;
using namespace mlir::loop;

//===----------------------------------------------------------------------===//
// LoopOpsDialect
//===----------------------------------------------------------------------===//

LoopOpsDialect::LoopOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(Builder *builder, OperationState &result, Value lb, Value ub,
                  Value step, ValueRange iterArgs) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  ForOp::ensureTerminator(*bodyRegion, *builder, result.location);
  bodyRegion->front().addArgument(builder->getIndexType());
  for (Value v : iterArgs)
    bodyRegion->front().addArgument(v.getType());
}

static LogicalResult verify(ForOp op) {
  if (auto cst = dyn_cast_or_null<ConstantIndexOp>(op.step().getDefiningOp()))
    if (cst.getValue() <= 0)
      return op.emitOpError("constant step operand must be positive");

  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = op.getBody();
  if (!body->getArgument(0).getType().isIndex())
    return op.emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = op.getNumResults();
  if (opNumResults == 0)
    return success();
  // If ForOp defines values, check that the number and types of
  // the defined values match ForOp initial iter operands and backedge
  // basic block arguments.
  if (op.getNumIterOperands() != opNumResults)
    return op.emitOpError(
        "mismatch in number of loop-carried values and defined values");
  if (op.getNumRegionIterArgs() != opNumResults)
    return op.emitOpError(
        "mismatch in number of basic block args and defined values");
  auto iterOperands = op.getIterOperands();
  auto iterArgs = op.getRegionIterArgs();
  auto opResults = op.getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return op.emitOpError() << "types mismatch between " << i
                              << "th iter region arg and defined value";

    i++;
  }
  return success();
}

static void print(OpAsmPrinter &p, ForOp op) {
  bool printBlockTerminators = false;
  p << op.getOperationName() << " " << op.getInductionVar() << " = "
    << op.lowerBound() << " to " << op.upperBound() << " step " << op.step();

  if (op.hasIterOperands()) {
    p << " iter_args(";
    auto regionArgs = op.getRegionIterArgs();
    auto operands = op.getIterOperands();

    mlir::interleaveComma(llvm::zip(regionArgs, operands), p, [&](auto it) {
      p << std::get<0>(it) << " = " << std::get<1>(it);
    });
    p << ")";
    p << " -> (" << op.getResultTypes() << ")";
    printBlockTerminators = true;
  }
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);
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

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::OperandType, 4> regionArgs, operands;
  SmallVector<Type, 4> argTypes;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  }
  // Induction variable.
  argTypes.push_back(indexType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, argTypes))
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
  if (op.getNumResults() != 0 && op.elseRegion().empty())
    return op.emitOpError("must have an else block if defining values");

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
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
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
  bool printBlockTerminators = false;

  p << IfOp::getOperationName() << " " << op.condition();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p.printRegion(op.thenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = op.elseRegion();
  if (!elseRegion.empty()) {
    p << " else";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict(op.getAttrs());
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(Builder *builder, OperationState &result, ValueRange lbs,
                       ValueRange ubs, ValueRange steps, ValueRange initVals) {
  result.addOperands(lbs);
  result.addOperands(ubs);
  result.addOperands(steps);
  result.addOperands(initVals);
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder->getI32VectorAttr({static_cast<int32_t>(lbs.size()),
                                 static_cast<int32_t>(ubs.size()),
                                 static_cast<int32_t>(steps.size()),
                                 static_cast<int32_t>(initVals.size())}));
  Region *bodyRegion = result.addRegion();
  ParallelOp::ensureTerminator(*bodyRegion, *builder, result.location);
  for (size_t i = 0, e = steps.size(); i < e; ++i)
    bodyRegion->front().addArgument(builder->getIndexType());
  for (Value init : initVals)
    result.addTypes(init.getType());
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
    return op.emitOpError()
           << "expects the same number of induction variables: "
           << body->getNumArguments()
           << " as bound and step values: " << stepValues.size();
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return op.emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the number of results is the same as the number of ReduceOps.
  SmallVector<ReduceOp, 4> reductions(body->getOps<ReduceOp>());
  auto resultsSize = op.results().size();
  auto reductionsSize = reductions.size();
  auto initValsSize = op.initVals().size();
  if (resultsSize != reductionsSize)
    return op.emitOpError()
           << "expects number of results: " << resultsSize
           << " to be the same as number of reductions: " << reductionsSize;
  if (resultsSize != initValsSize)
    return op.emitOpError()
           << "expects number of results: " << resultsSize
           << " to be the same as number of initial values: " << initValsSize;

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(op.results(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.operand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
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

  // Parse step values.
  SmallVector<OpAsmParser::OperandType, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Parse init values.
  SmallVector<OpAsmParser::OperandType, 4> initVals;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseOperandList(initVals, /*requiredOperandCount=*/-1,
                                OpAsmParser::Delimiter::Paren))
      return failure();
  }

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  SmallVector<Type, 4> types(ivs.size(), builder.getIndexType());
  if (parser.parseRegion(*body, ivs, types))
    return failure();

  // Set `operand_segment_sizes` attribute.
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({static_cast<int32_t>(lower.size()),
                                static_cast<int32_t>(upper.size()),
                                static_cast<int32_t>(steps.size()),
                                static_cast<int32_t>(initVals.size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!initVals.empty())
    parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                           result.operands);
  // Add a terminator if none was parsed.
  ForOp::ensureTerminator(*body, builder, result.location);

  return success();
}

static void print(OpAsmPrinter &p, ParallelOp op) {
  p << op.getOperationName() << " (" << op.getBody()->getArguments() << ") = ("
    << op.lowerBound() << ") to (" << op.upperBound() << ") step (" << op.step()
    << ")";
  if (!op.initVals().empty())
    p << " init (" << op.initVals() << ")";
  p.printOptionalArrowTypeList(op.getResultTypes());
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      op.getAttrs(), /*elidedAttrs=*/ParallelOp::getOperandSegmentSizeAttr());
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

void ReduceOp::build(Builder *builder, OperationState &result, Value operand) {
  auto type = operand.getType();
  result.addOperands(operand);
  Region *bodyRegion = result.addRegion();

  Block *b = new Block();
  b->addArguments(ArrayRef<Type>{type, type});
  bodyRegion->getBlocks().insert(bodyRegion->end(), b);
}

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

  Type resultType;
  // Parse the type of the operand (and also what reduce computes on).
  if (parser.parseColonType(resultType) ||
      parser.resolveOperand(operand, resultType, result.operands))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

static void print(OpAsmPrinter &p, ReduceOp op) {
  p << op.getOperationName() << "(" << op.operand() << ") ";
  p << " : " << op.operand().getType();
  p.printRegion(op.reductionOperator());
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

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//
static LogicalResult verify(YieldOp op) {
  auto parentOp = op.getParentOp();
  auto results = parentOp->getResults();
  auto operands = op.getOperands();

  if (isa<IfOp>(parentOp) || isa<ForOp>(parentOp)) {
    if (parentOp->getNumResults() != op.getNumOperands())
      return op.emitOpError() << "parent of yield must have same number of "
                                 "results as the yield operands";
    for (auto e : llvm::zip(results, operands)) {
      if (std::get<0>(e).getType() != std::get<1>(e).getType())
        return op.emitOpError()
               << "types mismatch between yield op and its parent";
    }
  } else if (isa<ParallelOp>(parentOp)) {
    if (op.getNumOperands() != 0)
      return op.emitOpError()
             << "yield inside loop.parallel is not allowed to have operands";
  } else {
    return op.emitOpError()
           << "yield only terminates If, For or Parallel regions";
  }

  return success();
}

static ParseResult parseYieldOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> types;
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse variadic operands list, their types, and resolve operands to SSA
  // values.
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalColonTypeList(types) ||
      parser.resolveOperands(operands, types, loc, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, YieldOp op) {
  p << op.getOperationName();
  if (op.getNumOperands() != 0)
    p << ' ' << op.getOperands() << " : " << op.getOperandTypes();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/LoopOps/LoopOps.cpp.inc"
