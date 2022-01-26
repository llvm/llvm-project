//===- Ops.cpp - Standard MLIR Operations ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
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

#include "mlir/Dialect/StandardOps/IR/OpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// StandardOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining with standard
/// operations.
struct StdInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = dyn_cast<ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    OpBuilder builder(op);
    builder.create<BranchOp>(op->getLoc(), newDest, returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// StandardOpsDialect
//===----------------------------------------------------------------------===//

void StandardOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/StandardOps/IR/Ops.cpp.inc"
      >();
  addInterfaces<StdInlinerInterface>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *StandardOpsDialect::materializeConstant(OpBuilder &builder,
                                                   Attribute value, Type type,
                                                   Location loc) {
  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, value);
  return builder.create<ConstantOp>(loc, type, value);
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
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

/// Fold indirect calls that have a constant function as the callee operand.
LogicalResult CallIndirectOp::canonicalize(CallIndirectOp indirectCall,
                                           PatternRewriter &rewriter) {
  // Check that the callee is a constant callee.
  SymbolRefAttr calledFn;
  if (!matchPattern(indirectCall.getCallee(), m_Constant(&calledFn)))
    return failure();

  // Replace with a direct call.
  rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn,
                                      indirectCall.getResultTypes(),
                                      indirectCall.getArgOperands());
  return success();
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  if (type.isa<UnrankedTensorType>())
    return UnrankedTensorType::get(i1Type);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return VectorType::get(vectorType.getShape(), i1Type,
                           vectorType.getNumScalableDims());
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

namespace {
/// cond_br true, ^bb1, ^bb2
///  -> br ^bb1
/// cond_br false, ^bb1, ^bb2
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

///   cond_br %cond, ^bb1, ^bb2
/// ^bb1
///   br ^bbN(...)
/// ^bb2
///   br ^bbK(...)
///
///  -> cond_br %cond, ^bbN(...), ^bbK(...)
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

/// cond_br %cond, ^bb1(A, ..., N), ^bb1(A, ..., N)
///  -> br ^bb1(A, ..., N)
///
/// cond_br %cond, ^bb1(A), ^bb1(B)
///  -> %select = select %cond, A, B
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
        mergedOperands.push_back(rewriter.create<SelectOp>(
            condbr.getLoc(), condition, std::get<0>(it), std::get<1>(it)));
    }

    rewriter.replaceOpWithNewOp<BranchOp>(condbr, trueDest, mergedOperands);
    return success();
  }
};

///   ...
///   cond_br %cond, ^bb1(...), ^bb2(...)
/// ...
/// ^bb1: // has single predecessor
///   ...
///   cond_br %cond, ^bb3(...), ^bb4(...)
///
/// ->
///
///   ...
///   cond_br %cond, ^bb1(...), ^bb2(...)
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

///   cond_br %arg0, ^trueB, ^falseB
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
///   cond_br %arg0, ^trueB, ^falseB
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
// ConstantOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, ConstantOp &op) {
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});

  if (op->getAttrs().size() > 1)
    p << ' ';
  p << op.getValue();

  // If the value is a symbol reference, print a trailing type.
  if (op.getValue().isa<SymbolRefAttr>())
    p << " : " << op.getType();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // If the attribute is a symbol reference, then we expect a trailing type.
  Type type;
  if (!valueAttr.isa<SymbolRefAttr>())
    type = valueAttr.getType();
  else if (parser.parseColonType(type))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(type, result.types);
}

/// The constant op requires an attribute, and furthermore requires that it
/// matches the return type.
static LogicalResult verify(ConstantOp &op) {
  auto value = op.getValue();
  if (!value)
    return op.emitOpError("requires a 'value' attribute");

  Type type = op.getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return op.emitOpError() << "requires attribute's type (" << value.getType()
                            << ") to match op's return type (" << type << ")";

  if (type.isa<FunctionType>()) {
    auto fnAttr = value.dyn_cast<FlatSymbolRefAttr>();
    if (!fnAttr)
      return op.emitOpError("requires 'value' to be a function reference");

    // Try to find the referenced function.
    auto fn =
        op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(fnAttr.getValue());
    if (!fn)
      return op.emitOpError()
             << "reference to undefined function '" << fnAttr.getValue() << "'";

    // Check that the referenced function has the correct type.
    if (fn.getType() != type)
      return op.emitOpError("reference to function with mismatched type");

    return success();
  }

  if (type.isa<NoneType>() && value.isa<UnitAttr>())
    return success();

  return op.emitOpError("unsupported 'value' attribute: ") << value;
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  Type type = getType();
  if (type.isa<FunctionType>()) {
    setNameFn(getResult(), "f");
  } else {
    setNameFn(getResult(), "cst");
  }
}

/// Returns true if a constant operation can be built with the given value and
/// result type.
bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  // SymbolRefAttr can only be used with a function type.
  if (value.isa<SymbolRefAttr>())
    return type.isa<FunctionType>();
  // Otherwise, this must be a UnitAttr.
  return value.isa<UnitAttr>() && type.isa<NoneType>();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReturnOp op) {
  auto function = cast<FuncOp>(op->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError("has ")
           << op.getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (op.getOperand(i).getType() != results[i])
      return op.emitError()
             << "type of return operand " << i << " ("
             << op.getOperand(i).getType()
             << ") doesn't match function result type (" << results[i] << ")"
             << " in function @" << function.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Transforms a select of a boolean to arithmetic operations
//
//  select %arg, %x, %y : i1
//
//  becomes
//
//  and(%arg, %x) or and(!%arg, %y)
struct SelectI1Simplify : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isInteger(1))
      return failure();

    Value falseConstant =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), true, 1);
    Value notCondition = rewriter.create<arith::XOrIOp>(
        op.getLoc(), op.getCondition(), falseConstant);

    Value trueVal = rewriter.create<arith::AndIOp>(
        op.getLoc(), op.getCondition(), op.getTrueValue());
    Value falseVal = rewriter.create<arith::AndIOp>(op.getLoc(), notCondition,
                                                    op.getFalseValue());
    rewriter.replaceOpWithNewOp<arith::OrIOp>(op, trueVal, falseVal);
    return success();
  }
};

//  select %arg, %c1, %c0 => extui %arg
struct SelectToExtUI : public OpRewritePattern<SelectOp> {
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &rewriter) const override {
    // Cannot extui i1 to i1, or i1 to f32
    if (!op.getType().isa<IntegerType>() || op.getType().isInteger(1))
      return failure();

    // select %x, c1, %c0 => extui %arg
    if (matchPattern(op.getTrueValue(), m_One()))
      if (matchPattern(op.getFalseValue(), m_Zero())) {
        rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(),
                                                    op.getCondition());
        return success();
      }

    // select %x, c0, %c1 => extui (xor %arg, true)
    if (matchPattern(op.getTrueValue(), m_Zero()))
      if (matchPattern(op.getFalseValue(), m_One())) {
        rewriter.replaceOpWithNewOp<arith::ExtUIOp>(
            op, op.getType(),
            rewriter.create<arith::XOrIOp>(
                op.getLoc(), op.getCondition(),
                rewriter.create<arith::ConstantIntOp>(
                    op.getLoc(), 1, op.getCondition().getType())));
        return success();
      }

    return failure();
  }
};

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<SelectI1Simplify, SelectToExtUI>(context);
}

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
  auto trueVal = getTrueValue();
  auto falseVal = getFalseValue();
  if (trueVal == falseVal)
    return trueVal;

  auto condition = getCondition();

  // select true, %0, %1 => %0
  if (matchPattern(condition, m_One()))
    return trueVal;

  // select false, %0, %1 => %1
  if (matchPattern(condition, m_Zero()))
    return falseVal;

  // select %x, true, false => %x
  if (getType().isInteger(1))
    if (matchPattern(getTrueValue(), m_One()))
      if (matchPattern(getFalseValue(), m_Zero()))
        return condition;

  if (auto cmp = dyn_cast_or_null<arith::CmpIOp>(condition.getDefiningOp())) {
    auto pred = cmp.getPredicate();
    if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne) {
      auto cmpLhs = cmp.getLhs();
      auto cmpRhs = cmp.getRhs();

      // %0 = arith.cmpi eq, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg1

      // %0 = arith.cmpi ne, %arg0, %arg1
      // %1 = select %0, %arg0, %arg1 => %arg0

      if ((cmpLhs == trueVal && cmpRhs == falseVal) ||
          (cmpRhs == trueVal && cmpLhs == falseVal))
        return pred == arith::CmpIPredicate::ne ? trueVal : falseVal;
    }
  }
  return nullptr;
}

static void print(OpAsmPrinter &p, SelectOp op) {
  p << " " << op.getOperands();
  p.printOptionalAttrDict(op->getAttrs());
  p << " : ";
  if (ShapedType condType = op.getCondition().getType().dyn_cast<ShapedType>())
    p << condType << ", ";
  p << op.getType();
}

static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &result) {
  Type conditionType, resultType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  // Check for the explicit condition type if this is a masked tensor or vector.
  if (succeeded(parser.parseOptionalComma())) {
    conditionType = resultType;
    if (parser.parseType(resultType))
      return failure();
  } else {
    conditionType = parser.getBuilder().getI1Type();
  }

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {conditionType, resultType, resultType},
                                parser.getNameLoc(), result.operands);
}

static LogicalResult verify(SelectOp op) {
  Type conditionType = op.getCondition().getType();
  if (conditionType.isSignlessInteger(1))
    return success();

  // If the result type is a vector or tensor, the type can be a mask with the
  // same elements.
  Type resultType = op.getType();
  if (!resultType.isa<TensorType, VectorType>())
    return op.emitOpError()
           << "expected condition to be a signless i1, but got "
           << conditionType;
  Type shapedConditionType = getI1SameShape(resultType);
  if (conditionType != shapedConditionType)
    return op.emitOpError()
           << "expected condition type to have the same shape "
              "as the result type, expected "
           << shapedConditionType << ", but got " << conditionType;
  return success();
}

//===----------------------------------------------------------------------===//
// SplatOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(SplatOp op) {
  // TODO: we could replace this by a trait.
  if (op.getOperand().getType() !=
      op.getType().cast<ShapedType>().getElementType())
    return op.emitError("operand should be of elemental type of result type");

  return success();
}

// Constant folding hook for SplatOp.
OpFoldResult SplatOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1 && "splat takes one operand");

  auto constOperand = operands.front();
  if (!constOperand || !constOperand.isa<IntegerAttr, FloatAttr>())
    return {};

  auto shapedType = getType().cast<ShapedType>();
  assert(shapedType.getElementType() == constOperand.getType() &&
         "incorrect input attribute type for folding");

  // SplatElementsAttr::get treats single value for second arg as being a splat.
  return SplatElementsAttr::get(shapedType, {constOperand});
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

static LogicalResult verify(SwitchOp op) {
  auto caseValues = op.getCaseValues();
  auto caseDestinations = op.getCaseDestinations();

  if (!caseValues && caseDestinations.empty())
    return success();

  Type flagType = op.getFlag().getType();
  Type caseValueType = caseValues->getType().getElementType();
  if (caseValueType != flagType)
    return op.emitOpError()
           << "'flag' type (" << flagType << ") should match case value type ("
           << caseValueType << ")";

  if (caseValues &&
      caseValues->size() != static_cast<int64_t>(caseDestinations.size()))
    return op.emitOpError() << "number of case values (" << caseValues->size()
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
#include "mlir/Dialect/StandardOps/IR/Ops.cpp.inc"
