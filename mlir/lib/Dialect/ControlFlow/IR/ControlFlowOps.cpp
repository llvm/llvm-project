//===- ControlFlowOps.cpp - ControlFlow Operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::cf;

//===----------------------------------------------------------------------===//
// ControlFlowDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with control flow
/// operations.
struct ControlFlowInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  ~ControlFlowInlinerInterface() override = default;

  /// All control flow operations can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  /// ControlFlow terminator operations don't really need any special handing.
  void handleTerminator(Operation *op, Block *newDest) const final {}
};
} // namespace

//===----------------------------------------------------------------------===//
// ControlFlowDialect
//===----------------------------------------------------------------------===//

void ControlFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.cpp.inc"
      >();
  addInterfaces<ControlFlowInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  // Erase assertion if argument is constant true.
  if (matchPattern(op.getArg(), m_One())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

/// Given a successor, try to collapse it to a new destination if it only
/// contains a passthrough unconditional branch. If the successor is
/// collapsable, `successor` and `successorOperands` are updated to reference
/// the new destination and values. `argStorage` is used as storage if operands
/// to the collapsed successor need to be remapped. It must outlive uses of
/// successorOperands.
static LogicalResult collapseBranch(Block *&successor,
                                    ValueRange &successorOperands,
                                    SmallVectorImpl<Value> &argStorage) {
  // Check that the successor only contains a unconditional branch.
  if (std::next(successor->begin()) != successor->end())
    return failure();
  // Check that the terminator is an unconditional branch.
  BranchOp successorBranch = dyn_cast<BranchOp>(successor->getTerminator());
  if (!successorBranch)
    return failure();
  // Check that the arguments are only used within the terminator.
  for (BlockArgument arg : successor->getArguments()) {
    for (Operation *user : arg.getUsers())
      if (user != successorBranch)
        return failure();
  }
  // Don't try to collapse branches to infinite loops.
  Block *successorDest = successorBranch.getDest();
  if (successorDest == successor)
    return failure();

  // Update the operands to the successor. If the branch parent has no
  // arguments, we can use the branch operands directly.
  OperandRange operands = successorBranch.getOperands();
  if (successor->args_empty()) {
    successor = successorDest;
    successorOperands = operands;
    return success();
  }

  // Otherwise, we need to remap any argument operands.
  for (Value operand : operands) {
    BlockArgument argOperand = operand.dyn_cast<BlockArgument>();
    if (argOperand && argOperand.getOwner() == successor)
      argStorage.push_back(successorOperands[argOperand.getArgNumber()]);
    else
      argStorage.push_back(operand);
  }
  successor = successorDest;
  successorOperands = argStorage;
  return success();
}

/// Simplify a branch to a block that has a single predecessor. This effectively
/// merges the two blocks.
static LogicalResult
simplifyBrToBlockWithSinglePred(BranchOp op, PatternRewriter &rewriter) {
  // Check that the successor block has a single predecessor.
  Block *succ = op.getDest();
  Block *opParent = op->getBlock();
  if (succ == opParent || !llvm::hasSingleElement(succ->getPredecessors()))
    return failure();

  // Merge the successor into the current block and erase the branch.
  rewriter.mergeBlocks(succ, opParent, op.getOperands());
  rewriter.eraseOp(op);
  return success();
}

///   br ^bb1
/// ^bb1
///   br ^bbN(...)
///
///  -> br ^bbN(...)
///
static LogicalResult simplifyPassThroughBr(BranchOp op,
                                           PatternRewriter &rewriter) {
  Block *dest = op.getDest();
  ValueRange destOperands = op.getOperands();
  SmallVector<Value, 4> destOperandStorage;

  // Try to collapse the successor if it points somewhere other than this
  // block.
  if (dest == op->getBlock() ||
      failed(collapseBranch(dest, destOperands, destOperandStorage)))
    return failure();

  // Create a new branch with the collapsed successor.
  rewriter.replaceOpWithNewOp<BranchOp>(op, dest, destOperands);
  return success();
}

LogicalResult BranchOp::canonicalize(BranchOp op, PatternRewriter &rewriter) {
  return success(succeeded(simplifyBrToBlockWithSinglePred(op, rewriter)) ||
                 succeeded(simplifyPassThroughBr(op, rewriter)));
}

void BranchOp::setDest(Block *block) { return setSuccessor(block); }

void BranchOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }

Optional<MutableOperandRange>
BranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getDestOperandsMutable();
}

Block *BranchOp::getSuccessorForOperands(ArrayRef<Attribute>) {
  return getDest();
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

namespace {
/// cf.cond_br true, ^bb1, ^bb2
///  -> br ^bb1
/// cf.cond_br false, ^bb1, ^bb2
///  -> br ^bb2
///
struct SimplifyConstCondBranchPred : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    if (matchPattern(condbr.getCondition(), m_NonZero())) {
      // True branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueOperands());
      return success();
    }
    if (matchPattern(condbr.getCondition(), m_Zero())) {
      // False branch taken.
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseOperands());
      return success();
    }
    return failure();
  }
};

///   cf.cond_br %cond, ^bb1, ^bb2
/// ^bb1
///   br ^bbN(...)
/// ^bb2
///   br ^bbK(...)
///
///  -> cf.cond_br %cond, ^bbN(...), ^bbK(...)
///
struct SimplifyPassThroughCondBranch : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    Block *trueDest = condbr.getTrueDest(), *falseDest = condbr.getFalseDest();
    ValueRange trueDestOperands = condbr.getTrueOperands();
    ValueRange falseDestOperands = condbr.getFalseOperands();
    SmallVector<Value, 4> trueDestOperandStorage, falseDestOperandStorage;

    // Try to collapse one of the current successors.
    LogicalResult collapsedTrue =
        collapseBranch(trueDest, trueDestOperands, trueDestOperandStorage);
    LogicalResult collapsedFalse =
        collapseBranch(falseDest, falseDestOperands, falseDestOperandStorage);
    if (failed(collapsedTrue) && failed(collapsedFalse))
      return failure();

    // Create a new branch with the collapsed successors.
    rewriter.replaceOpWithNewOp<CondBranchOp>(condbr, condbr.getCondition(),
                                              trueDest, trueDestOperands,
                                              falseDest, falseDestOperands);
    return success();
  }
};

/// cf.cond_br %cond, ^bb1(A, ..., N), ^bb1(A, ..., N)
///  -> br ^bb1(A, ..., N)
///
/// cf.cond_br %cond, ^bb1(A), ^bb1(B)
///  -> %select = arith.select %cond, A, B
///     br ^bb1(%select)
///
struct SimplifyCondBranchIdenticalSuccessors
    : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that the true and false destinations are the same and have the same
    // operands.
    Block *trueDest = condbr.getTrueDest();
    if (trueDest != condbr.getFalseDest())
      return failure();

    // If all of the operands match, no selects need to be generated.
    OperandRange trueOperands = condbr.getTrueOperands();
    OperandRange falseOperands = condbr.getFalseOperands();
    if (trueOperands == falseOperands) {
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, trueOperands);
      return success();
    }

    // Otherwise, if the current block is the only predecessor insert selects
    // for any mismatched branch operands.
    if (trueDest->getUniquePredecessor() != condbr->getBlock())
      return failure();

    // Generate a select for any operands that differ between the two.
    SmallVector<Value, 8> mergedOperands;
    mergedOperands.reserve(trueOperands.size());
    Value condition = condbr.getCondition();
    for (auto it : llvm::zip(trueOperands, falseOperands)) {
      if (std::get<0>(it) == std::get<1>(it))
        mergedOperands.push_back(std::get<0>(it));
      else
        mergedOperands.push_back(rewriter.create<arith::SelectOp>(
            condbr.getLoc(), condition, std::get<0>(it), std::get<1>(it)));
    }

    rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, mergedOperands);
    return success();
  }
};

///   ...
///   cf.cond_br %cond, ^bb1(...), ^bb2(...)
/// ...
/// ^bb1: // has single predecessor
///   ...
///   cf.cond_br %cond, ^bb3(...), ^bb4(...)
///
/// ->
///
///   ...
///   cf.cond_br %cond, ^bb1(...), ^bb2(...)
/// ...
/// ^bb1: // has single predecessor
///   ...
///   br ^bb3(...)
///
struct SimplifyCondBranchFromCondBranchOnSameCondition
    : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that we have a single distinct predecessor.
    Block *currentBlock = condbr->getBlock();
    Block *predecessor = currentBlock->getSinglePredecessor();
    if (!predecessor)
      return failure();

    // Check that the predecessor terminates with a conditional branch to this
    // block and that it branches on the same condition.
    auto predBranch = dyn_cast<CondBranchOp>(predecessor->getTerminator());
    if (!predBranch || condbr.getCondition() != predBranch.getCondition())
      return failure();

    // Fold this branch to an unconditional branch.
    if (currentBlock == predBranch.getTrueDest())
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getTrueDest(),
                                            condbr.getTrueDestOperands());
    else
      rewriter.replaceOpWithNewOp<BranchOp>(condbr, condbr.getFalseDest(),
                                            condbr.getFalseDestOperands());
    return success();
  }
};

///   cf.cond_br %arg0, ^trueB, ^falseB
///
/// ^trueB:
///   "test.consumer1"(%arg0) : (i1) -> ()
///    ...
///
/// ^falseB:
///   "test.consumer2"(%arg0) : (i1) -> ()
///   ...
///
/// ->
///
///   cf.cond_br %arg0, ^trueB, ^falseB
/// ^trueB:
///   "test.consumer1"(%true) : (i1) -> ()
///   ...
///
/// ^falseB:
///   "test.consumer2"(%false) : (i1) -> ()
///   ...
struct CondBranchTruthPropagation : public OpRewritePattern<CondBranchOp> {
  using OpRewritePattern<CondBranchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CondBranchOp condbr,
                                PatternRewriter &rewriter) const override {
    // Check that we have a single distinct predecessor.
    bool replaced = false;
    Type ty = rewriter.getI1Type();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    // TODO These checks can be expanded to encompas any use with only
    // either the true of false edge as a predecessor. For now, we fall
    // back to checking the single predecessor is given by the true/fasle
    // destination, thereby ensuring that only that edge can reach the
    // op.
    if (condbr.getTrueDest()->getSinglePredecessor()) {
      for (OpOperand &use :
           llvm::make_early_inc_range(condbr.getCondition().getUses())) {
        if (use.getOwner()->getBlock() == condbr.getTrueDest()) {
          replaced = true;

          if (!constantTrue)
            constantTrue = rewriter.create<arith::ConstantOp>(
                condbr.getLoc(), ty, rewriter.getBoolAttr(true));

          rewriter.updateRootInPlace(use.getOwner(),
                                     [&] { use.set(constantTrue); });
        }
      }
    }
    if (condbr.getFalseDest()->getSinglePredecessor()) {
      for (OpOperand &use :
           llvm::make_early_inc_range(condbr.getCondition().getUses())) {
        if (use.getOwner()->getBlock() == condbr.getFalseDest()) {
          replaced = true;

          if (!constantFalse)
            constantFalse = rewriter.create<arith::ConstantOp>(
                condbr.getLoc(), ty, rewriter.getBoolAttr(false));

          rewriter.updateRootInPlace(use.getOwner(),
                                     [&] { use.set(constantFalse); });
        }
      }
    }
    return success(replaced);
  }
};
} // namespace

void CondBranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<SimplifyConstCondBranchPred, SimplifyPassThroughCondBranch,
              SimplifyCondBranchIdenticalSuccessors,
              SimplifyCondBranchFromCondBranchOnSameCondition,
              CondBranchTruthPropagation>(context);
}

Optional<MutableOperandRange>
CondBranchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? getTrueDestOperandsMutable()
                            : getFalseDestOperandsMutable();
}

Block *CondBranchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
    return condAttr.getValue().isOneValue() ? getTrueDest() : getFalseDest();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     DenseIntElementsAttr caseValues,
                     BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands) {
  build(builder, result, value, defaultOperands, caseOperands, caseValues,
        defaultDestination, caseDestinations);
}

void SwitchOp::build(OpBuilder &builder, OperationState &result, Value value,
                     Block *defaultDestination, ValueRange defaultOperands,
                     ArrayRef<APInt> caseValues, BlockRange caseDestinations,
                     ArrayRef<ValueRange> caseOperands) {
  DenseIntElementsAttr caseValuesAttr;
  if (!caseValues.empty()) {
    ShapedType caseValueType = VectorType::get(
        static_cast<int64_t>(caseValues.size()), value.getType());
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseValues);
  }
  build(builder, result, value, defaultDestination, defaultOperands,
        caseValuesAttr, caseDestinations, caseOperands);
}

/// <cases> ::= `default` `:` bb-id (`(` ssa-use-and-type-list `)`)?
///             ( `,` integer `:` bb-id (`(` ssa-use-and-type-list `)`)? )*
static ParseResult parseSwitchOpCases(
    OpAsmParser &parser, Type &flagType, Block *&defaultDestination,
    SmallVectorImpl<OpAsmParser::OperandType> &defaultOperands,
    SmallVectorImpl<Type> &defaultOperandTypes,
    DenseIntElementsAttr &caseValues,
    SmallVectorImpl<Block *> &caseDestinations,
    SmallVectorImpl<SmallVector<OpAsmParser::OperandType>> &caseOperands,
    SmallVectorImpl<SmallVector<Type>> &caseOperandTypes) {
  if (parser.parseKeyword("default") || parser.parseColon() ||
      parser.parseSuccessor(defaultDestination))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRegionArgumentList(defaultOperands) ||
        parser.parseColonTypeList(defaultOperandTypes) || parser.parseRParen())
      return failure();
  }

  SmallVector<APInt> values;
  unsigned bitWidth = flagType.getIntOrFloatBitWidth();
  while (succeeded(parser.parseOptionalComma())) {
    int64_t value = 0;
    if (failed(parser.parseInteger(value)))
      return failure();
    values.push_back(APInt(bitWidth, value));

    Block *destination;
    SmallVector<OpAsmParser::OperandType> operands;
    SmallVector<Type> operandTypes;
    if (failed(parser.parseColon()) ||
        failed(parser.parseSuccessor(destination)))
      return failure();
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseRegionArgumentList(operands)) ||
          failed(parser.parseColonTypeList(operandTypes)) ||
          failed(parser.parseRParen()))
        return failure();
    }
    caseDestinations.push_back(destination);
    caseOperands.emplace_back(operands);
    caseOperandTypes.emplace_back(operandTypes);
  }

  if (!values.empty()) {
    ShapedType caseValueType =
        VectorType::get(static_cast<int64_t>(values.size()), flagType);
    caseValues = DenseIntElementsAttr::get(caseValueType, values);
  }
  return success();
}

static void printSwitchOpCases(
    OpAsmPrinter &p, SwitchOp op, Type flagType, Block *defaultDestination,
    OperandRange defaultOperands, TypeRange defaultOperandTypes,
    DenseIntElementsAttr caseValues, SuccessorRange caseDestinations,
    OperandRangeRange caseOperands, const TypeRangeRange &caseOperandTypes) {
  p << "  default: ";
  p.printSuccessorAndUseList(defaultDestination, defaultOperands);

  if (!caseValues)
    return;

  for (const auto &it : llvm::enumerate(caseValues.getValues<APInt>())) {
    p << ',';
    p.printNewline();
    p << "  ";
    p << it.value().getLimitedValue();
    p << ": ";
    p.printSuccessorAndUseList(caseDestinations[it.index()],
                               caseOperands[it.index()]);
  }
  p.printNewline();
}

LogicalResult SwitchOp::verify() {
  auto caseValues = getCaseValues();
  auto caseDestinations = getCaseDestinations();

  if (!caseValues && caseDestinations.empty())
    return success();

  Type flagType = getFlag().getType();
  Type caseValueType = caseValues->getType().getElementType();
  if (caseValueType != flagType)
    return emitOpError() << "'flag' type (" << flagType
                         << ") should match case value type (" << caseValueType
                         << ")";

  if (caseValues &&
      caseValues->size() != static_cast<int64_t>(caseDestinations.size()))
    return emitOpError() << "number of case values (" << caseValues->size()
                         << ") should match number of "
                            "case destinations ("
                         << caseDestinations.size() << ")";
  return success();
}

Optional<MutableOperandRange>
SwitchOp::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == 0 ? getDefaultOperandsMutable()
                    : getCaseOperandsMutable(index - 1);
}

Block *SwitchOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
  Optional<DenseIntElementsAttr> caseValues = getCaseValues();

  if (!caseValues)
    return getDefaultDestination();

  SuccessorRange caseDests = getCaseDestinations();
  if (auto value = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    for (const auto &it : llvm::enumerate(caseValues->getValues<APInt>()))
      if (it.value() == value.getValue())
        return caseDests[it.index()];
    return getDefaultDestination();
  }
  return nullptr;
}

/// switch %flag : i32, [
///   default:  ^bb1
/// ]
///  -> br ^bb1
static LogicalResult simplifySwitchWithOnlyDefault(SwitchOp op,
                                                   PatternRewriter &rewriter) {
  if (!op.getCaseDestinations().empty())
    return failure();

  rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDefaultDestination(),
                                        op.getDefaultOperands());
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb1,
///   43: ^bb2
/// ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   43: ^bb2
/// ]
static LogicalResult
dropSwitchCasesThatMatchDefault(SwitchOp op, PatternRewriter &rewriter) {
  SmallVector<Block *> newCaseDestinations;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<APInt> newCaseValues;
  bool requiresChange = false;
  auto caseValues = op.getCaseValues();
  auto caseDests = op.getCaseDestinations();

  for (const auto &it : llvm::enumerate(caseValues->getValues<APInt>())) {
    if (caseDests[it.index()] == op.getDefaultDestination() &&
        op.getCaseOperands(it.index()) == op.getDefaultOperands()) {
      requiresChange = true;
      continue;
    }
    newCaseDestinations.push_back(caseDests[it.index()]);
    newCaseOperands.push_back(op.getCaseOperands(it.index()));
    newCaseValues.push_back(it.value());
  }

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(
      op, op.getFlag(), op.getDefaultDestination(), op.getDefaultOperands(),
      newCaseValues, newCaseDestinations, newCaseOperands);
  return success();
}

/// Helper for folding a switch with a constant value.
/// switch %c_42 : i32, [
///   default: ^bb1 ,
///   42: ^bb2,
///   43: ^bb3
/// ]
/// -> br ^bb2
static void foldSwitch(SwitchOp op, PatternRewriter &rewriter,
                       const APInt &caseValue) {
  auto caseValues = op.getCaseValues();
  for (const auto &it : llvm::enumerate(caseValues->getValues<APInt>())) {
    if (it.value() == caseValue) {
      rewriter.replaceOpWithNewOp<BranchOp>(
          op, op.getCaseDestinations()[it.index()],
          op.getCaseOperands(it.index()));
      return;
    }
  }
  rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDefaultDestination(),
                                        op.getDefaultOperands());
}

/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb2,
///   43: ^bb3
/// ]
/// -> br ^bb2
static LogicalResult simplifyConstSwitchValue(SwitchOp op,
                                              PatternRewriter &rewriter) {
  APInt caseValue;
  if (!matchPattern(op.getFlag(), m_ConstantInt(&caseValue)))
    return failure();

  foldSwitch(op, rewriter, caseValue);
  return success();
}

/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb3
/// ->
/// switch %c_42 : i32, [
///   default: ^bb1,
///   42: ^bb3,
/// ]
static LogicalResult simplifyPassThroughSwitch(SwitchOp op,
                                               PatternRewriter &rewriter) {
  SmallVector<Block *> newCaseDests;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<SmallVector<Value>> argStorage;
  auto caseValues = op.getCaseValues();
  auto caseDests = op.getCaseDestinations();
  bool requiresChange = false;
  for (int64_t i = 0, size = caseValues->size(); i < size; ++i) {
    Block *caseDest = caseDests[i];
    ValueRange caseOperands = op.getCaseOperands(i);
    argStorage.emplace_back();
    if (succeeded(collapseBranch(caseDest, caseOperands, argStorage.back())))
      requiresChange = true;

    newCaseDests.push_back(caseDest);
    newCaseOperands.push_back(caseOperands);
  }

  Block *defaultDest = op.getDefaultDestination();
  ValueRange defaultOperands = op.getDefaultOperands();
  argStorage.emplace_back();

  if (succeeded(
          collapseBranch(defaultDest, defaultOperands, argStorage.back())))
    requiresChange = true;

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(op, op.getFlag(), defaultDest,
                                        defaultOperands, caseValues.getValue(),
                                        newCaseDests, newCaseOperands);
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   switch %flag : i32, [
///     default: ^bb3,
///     42: ^bb4
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb4
///
///  and
///
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   switch %flag : i32, [
///     default: ^bb3,
///     43: ^bb4
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb2:
///   br ^bb3
static LogicalResult
simplifySwitchFromSwitchOnSameCondition(SwitchOp op,
                                        PatternRewriter &rewriter) {
  // Check that we have a single distinct predecessor.
  Block *currentBlock = op->getBlock();
  Block *predecessor = currentBlock->getSinglePredecessor();
  if (!predecessor)
    return failure();

  // Check that the predecessor terminates with a switch branch to this block
  // and that it branches on the same condition and that this branch isn't the
  // default destination.
  auto predSwitch = dyn_cast<SwitchOp>(predecessor->getTerminator());
  if (!predSwitch || op.getFlag() != predSwitch.getFlag() ||
      predSwitch.getDefaultDestination() == currentBlock)
    return failure();

  // Fold this switch to an unconditional branch.
  SuccessorRange predDests = predSwitch.getCaseDestinations();
  auto it = llvm::find(predDests, currentBlock);
  if (it != predDests.end()) {
    Optional<DenseIntElementsAttr> predCaseValues = predSwitch.getCaseValues();
    foldSwitch(op, rewriter,
               predCaseValues->getValues<APInt>()[it - predDests.begin()]);
  } else {
    rewriter.replaceOpWithNewOp<BranchOp>(op, op.getDefaultDestination(),
                                          op.getDefaultOperands());
  }
  return success();
}

/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2
/// ]
/// ^bb1:
///   switch %flag : i32, [
///     default: ^bb3,
///     42: ^bb4,
///     43: ^bb5
///   ]
/// ->
/// switch %flag : i32, [
///   default: ^bb1,
///   42: ^bb2,
/// ]
/// ^bb1:
///   switch %flag : i32, [
///     default: ^bb3,
///     43: ^bb5
///   ]
static LogicalResult
simplifySwitchFromDefaultSwitchOnSameCondition(SwitchOp op,
                                               PatternRewriter &rewriter) {
  // Check that we have a single distinct predecessor.
  Block *currentBlock = op->getBlock();
  Block *predecessor = currentBlock->getSinglePredecessor();
  if (!predecessor)
    return failure();

  // Check that the predecessor terminates with a switch branch to this block
  // and that it branches on the same condition and that this branch is the
  // default destination.
  auto predSwitch = dyn_cast<SwitchOp>(predecessor->getTerminator());
  if (!predSwitch || op.getFlag() != predSwitch.getFlag() ||
      predSwitch.getDefaultDestination() != currentBlock)
    return failure();

  // Delete case values that are not possible here.
  DenseSet<APInt> caseValuesToRemove;
  auto predDests = predSwitch.getCaseDestinations();
  auto predCaseValues = predSwitch.getCaseValues();
  for (int64_t i = 0, size = predCaseValues->size(); i < size; ++i)
    if (currentBlock != predDests[i])
      caseValuesToRemove.insert(predCaseValues->getValues<APInt>()[i]);

  SmallVector<Block *> newCaseDestinations;
  SmallVector<ValueRange> newCaseOperands;
  SmallVector<APInt> newCaseValues;
  bool requiresChange = false;

  auto caseValues = op.getCaseValues();
  auto caseDests = op.getCaseDestinations();
  for (const auto &it : llvm::enumerate(caseValues->getValues<APInt>())) {
    if (caseValuesToRemove.contains(it.value())) {
      requiresChange = true;
      continue;
    }
    newCaseDestinations.push_back(caseDests[it.index()]);
    newCaseOperands.push_back(op.getCaseOperands(it.index()));
    newCaseValues.push_back(it.value());
  }

  if (!requiresChange)
    return failure();

  rewriter.replaceOpWithNewOp<SwitchOp>(
      op, op.getFlag(), op.getDefaultDestination(), op.getDefaultOperands(),
      newCaseValues, newCaseDestinations, newCaseOperands);
  return success();
}

void SwitchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(&simplifySwitchWithOnlyDefault)
      .add(&dropSwitchCasesThatMatchDefault)
      .add(&simplifyConstSwitchValue)
      .add(&simplifyPassThroughSwitch)
      .add(&simplifySwitchFromSwitchOnSameCondition)
      .add(&simplifySwitchFromDefaultSwitchOnSameCondition);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.cpp.inc"
