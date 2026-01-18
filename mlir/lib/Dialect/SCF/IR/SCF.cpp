//===- SCF.cpp - Structured Control Flow Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ParallelCombiningOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include <optional>

using namespace mlir;
using namespace mlir::scf;

#include "mlir/Dialect/SCF/IR/SCFOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SCFDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct SCFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in scf dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto retValOp = dyn_cast<scf::YieldOp>(op);
    if (!retValOp)
      return;

    for (auto retValue : llvm::zip(valuesToRepl, retValOp.getOperands())) {
      std::get<0>(retValue).replaceAllUsesWith(std::get<1>(retValue));
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SCFDialect
//===----------------------------------------------------------------------===//

void SCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SCF/IR/SCFOps.cpp.inc"
      >();
  addInterfaces<SCFInlinerInterface>();
  declarePromisedInterface<ConvertToEmitCPatternInterface, SCFDialect>();
  declarePromisedInterfaces<bufferization::BufferDeallocationOpInterface,
                            InParallelOp, ReduceReturnOp>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, ConditionOp,
                            ExecuteRegionOp, ForOp, IfOp, IndexSwitchOp,
                            ForallOp, InParallelOp, WhileOp, YieldOp>();
  declarePromisedInterface<ValueBoundsOpInterface, ForOp>();
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir::scf::buildTerminatedBody(OpBuilder &builder, Location loc) {
  scf::YieldOp::create(builder, loc);
}

/// Verifies that the first block of the given `region` is terminated by a
/// TerminatorTy. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
}

std::optional<llvm::APSInt> mlir::scf::computeUbMinusLb(Value lb, Value ub,
                                                        bool isSigned) {
  llvm::APSInt diff;
  auto addOp = ub.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;
  if ((isSigned && !addOp.hasNoSignedWrap()) ||
      (!isSigned && !addOp.hasNoUnsignedWrap()))
    return std::nullopt;

  if (addOp.getLhs() != lb ||
      !matchPattern(addOp.getRhs(), m_ConstantInt(&diff)))
    return std::nullopt;
  return diff;
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

///
/// (ssa-id `=`)? `execute_region` `->` function-result-type `{`
///    block+
/// `}`
///
/// Example:
///   scf.execute_region -> i32 {
///     %idx = load %rI[%i] : memref<128xi32>
///     return %idx : i32
///   }
///
ParseResult ExecuteRegionOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("no_inline")))
    result.addAttribute("no_inline", parser.getBuilder().getUnitAttr());

  // Introduce the body region and parse it.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ExecuteRegionOp::print(OpAsmPrinter &p) {
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  if (getNoInline())
    p << "no_inline ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"no_inline"});
}

LogicalResult ExecuteRegionOp::verify() {
  if (getRegion().empty())
    return emitOpError("region needs to have at least one block");
  if (getRegion().front().getNumArguments() > 0)
    return emitOpError("region cannot have any arguments");
  return success();
}

// Inline an ExecuteRegionOp if its parent can contain multiple blocks.
// TODO generalize the conditions for operations which can be inlined into.
// func @func_execute_region_elim() {
//     "test.foo"() : () -> ()
//     %v = scf.execute_region -> i64 {
//       %c = "test.cmp"() : () -> i1
//       cf.cond_br %c, ^bb2, ^bb3
//     ^bb2:
//       %x = "test.val1"() : () -> i64
//       cf.br ^bb4(%x : i64)
//     ^bb3:
//       %y = "test.val2"() : () -> i64
//       cf.br ^bb4(%y : i64)
//     ^bb4(%z : i64):
//       scf.yield %z : i64
//     }
//     "test.bar"(%v) : (i64) -> ()
//   return
// }
//
//  becomes
//
// func @func_execute_region_elim() {
//    "test.foo"() : () -> ()
//    %c = "test.cmp"() : () -> i1
//    cf.cond_br %c, ^bb1, ^bb2
//  ^bb1:  // pred: ^bb0
//    %x = "test.val1"() : () -> i64
//    cf.br ^bb3(%x : i64)
//  ^bb2:  // pred: ^bb0
//    %y = "test.val2"() : () -> i64
//    cf.br ^bb3(%y : i64)
//  ^bb3(%z: i64):  // 2 preds: ^bb1, ^bb2
//    "test.bar"(%z) : (i64) -> ()
//    return
//  }
//
struct MultiBlockExecuteInliner : public OpRewritePattern<ExecuteRegionOp> {
  using OpRewritePattern<ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNoInline())
      return failure();
    if (!isa<FunctionOpInterface, ExecuteRegionOp>(op->getParentOp()))
      return failure();

    Block *prevBlock = op->getBlock();
    Block *postBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);

    cf::BranchOp::create(rewriter, op.getLoc(), &op.getRegion().front());

    for (Block &blk : op.getRegion()) {
      if (YieldOp yieldOp = dyn_cast<YieldOp>(blk.getTerminator())) {
        rewriter.setInsertionPoint(yieldOp);
        cf::BranchOp::create(rewriter, yieldOp.getLoc(), postBlock,
                             yieldOp.getResults());
        rewriter.eraseOp(yieldOp);
      }
    }

    rewriter.inlineRegionBefore(op.getRegion(), postBlock);
    SmallVector<Value> blockArgs;

    for (auto res : op.getResults())
      blockArgs.push_back(postBlock->addArgument(res.getType(), res.getLoc()));

    rewriter.replaceOp(op, blockArgs);
    return success();
  }
};

void ExecuteRegionOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<MultiBlockExecuteInliner>(context);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, ExecuteRegionOp::getOperationName());
  // Inline ops with a single block that are not marked as "no_inline".
  populateRegionBranchOpInterfaceInliningPattern(
      results, ExecuteRegionOp::getOperationName(),
      mlir::detail::defaultReplBuilderFn, [](Operation *op) {
        return failure(cast<ExecuteRegionOp>(op).getNoInline());
      });
}

void ExecuteRegionOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor::parent());
}

ValueRange ExecuteRegionOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getOperation()->getResults())
                              : ValueRange();
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(RegionSuccessor point) {
  assert(
      (point.isParent() || point.getSuccessor() == &getParentOp().getAfter()) &&
      "condition op can only exit the loop or branch to the after"
      "region");
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

void ConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands, *this);

  WhileOp whileOp = getParentOp();

  // Condition can either lead to the after region or back to the parent op
  // depending on whether the condition is true or not.
  auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&whileOp.getAfter());
  if (!boolAttr || !boolAttr.getValue())
    regions.push_back(RegionSuccessor::parent());
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, ValueRange initArgs,
                  BodyBuilderFn bodyBuilder, bool unsignedCmp) {
  OpBuilder::InsertionGuard guard(builder);

  if (unsignedCmp)
    result.addAttribute(getUnsignedCmpAttrName(result.name),
                        builder.getUnitAttr());
  result.addOperands({lb, ub, step});
  result.addOperands(initArgs);
  for (Value v : initArgs)
    result.addTypes(v.getType());
  Type t = lb.getType();
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(t, result.location);
  for (Value v : initArgs)
    bodyBlock->addArgument(v.getType(), v.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (initArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0),
                bodyBlock->getArguments().drop_front());
  }
}

LogicalResult ForOp::verify() {
  // Check that the number of init args and op results is the same.
  if (getInitArgs().size() != getNumResults())
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");

  return success();
}

LogicalResult ForOp::verifyRegions() {
  // Check that the body defines as single block argument for the induction
  // variable.
  if (getInductionVar().getType() != getLowerBound().getType())
    return emitOpError(
        "expected induction variable to be same type as bounds and step");

  if (getNumRegionIterArgs() != getNumResults())
    return emitOpError(
        "mismatch in number of basic block args and defined values");

  auto initArgs = getInitArgs();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(initArgs, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    ++i;
  }
  return success();
}

std::optional<SmallVector<Value>> ForOp::getLoopInductionVars() {
  return SmallVector<Value>{getInductionVar()};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopLowerBounds() {
  return SmallVector<OpFoldResult>{OpFoldResult(getLowerBound())};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopSteps() {
  return SmallVector<OpFoldResult>{OpFoldResult(getStep())};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopUpperBounds() {
  return SmallVector<OpFoldResult>{OpFoldResult(getUpperBound())};
}

bool ForOp::isValidInductionVarType(Type type) {
  return type.isIndex() || type.isSignlessInteger();
}

LogicalResult ForOp::setLoopLowerBounds(ArrayRef<OpFoldResult> bounds) {
  if (bounds.size() != 1)
    return failure();
  if (auto val = dyn_cast<Value>(bounds[0])) {
    setLowerBound(val);
    return success();
  }
  return failure();
}

LogicalResult ForOp::setLoopUpperBounds(ArrayRef<OpFoldResult> bounds) {
  if (bounds.size() != 1)
    return failure();
  if (auto val = dyn_cast<Value>(bounds[0])) {
    setUpperBound(val);
    return success();
  }
  return failure();
}

LogicalResult ForOp::setLoopSteps(ArrayRef<OpFoldResult> steps) {
  if (steps.size() != 1)
    return failure();
  if (auto val = dyn_cast<Value>(steps[0])) {
    setStep(val);
    return success();
  }
  return failure();
}

std::optional<ResultRange> ForOp::getLoopResults() { return getResults(); }

/// Promotes the loop body of a forOp to its containing block if the forOp
/// it can be determined that the loop has a single iteration.
LogicalResult ForOp::promoteIfSingleIteration(RewriterBase &rewriter) {
  std::optional<APInt> tripCount = getStaticTripCount();
  LDBG() << "promoteIfSingleIteration tripCount is " << tripCount
         << " for loop "
         << OpWithFlags(getOperation(), OpPrintingFlags().skipRegions());
  if (!tripCount.has_value() || tripCount->getSExtValue() > 1)
    return failure();

  if (*tripCount == 0) {
    rewriter.replaceAllUsesWith(getResults(), getInitArgs());
    rewriter.eraseOp(*this);
    return success();
  }

  // Replace all results with the yielded values.
  auto yieldOp = cast<scf::YieldOp>(getBody()->getTerminator());
  rewriter.replaceAllUsesWith(getResults(), getYieldedValues());

  // Replace block arguments with lower bound (replacement for IV) and
  // iter_args.
  SmallVector<Value> bbArgReplacements;
  bbArgReplacements.push_back(getLowerBound());
  llvm::append_range(bbArgReplacements, getInitArgs());

  // Move the loop body operations to the loop's containing block.
  rewriter.inlineBlockBefore(getBody(), getOperation()->getBlock(),
                             getOperation()->getIterator(), bbArgReplacements);

  // Erase the old terminator and the loop.
  rewriter.eraseOp(yieldOp);
  rewriter.eraseOp(*this);

  return success();
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void ForOp::print(OpAsmPrinter &p) {
  if (getUnsignedCmp())
    p << " unsigned";

  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();

  printInitializationList(p, getRegionIterArgs(), getInitArgs(), " iter_args");
  if (!getInitArgs().empty())
    p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  if (Type t = getInductionVar().getType(); !t.isIndex())
    p << " : " << t << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!getInitArgs().empty());
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/getUnsignedCmpAttrName().strref());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type type;

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  if (succeeded(parser.parseOptionalKeyword("unsigned")))
    result.addAttribute(getUnsignedCmpAttrName(result.name),
                        builder.getUnitAttr());

  // Parse the induction variable followed by '='.
  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  // Parse optional type, else assume Index.
  if (parser.parseOptionalColon())
    type = builder.getIndexType();
  else if (parser.parseType(type))
    return failure();

  // Set block argument types, so that they are known when parsing the region.
  regionArgs.front().type = type;
  for (auto [iterArg, type] :
       llvm::zip_equal(llvm::drop_begin(regionArgs), result.types))
    iterArg.type = type;

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  ForOp::ensureTerminator(*body, builder, result.location);

  // Resolve input operands. This should be done after parsing the region to
  // catch invalid IR where operands were defined inside of the region.
  if (parser.resolveOperand(lb, type, result.operands) ||
      parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();
  if (hasIterArgs) {
    for (auto argOperandType : llvm::zip_equal(llvm::drop_begin(regionArgs),
                                               operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

SmallVector<Region *> ForOp::getLoopRegions() { return {&getRegion()}; }

Block::BlockArgListType ForOp::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(getNumInductionVars());
}

MutableArrayRef<OpOperand> ForOp::getInitsMutable() {
  return getInitArgsMutable();
}

FailureOr<LoopLikeOpInterface>
ForOp::replaceWithAdditionalYields(RewriterBase &rewriter,
                                   ValueRange newInitOperands,
                                   bool replaceInitOperandUsesInLoop,
                                   const NewYieldValuesFn &newYieldValuesFn) {
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(getOperation());
  auto inits = llvm::to_vector(getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  scf::ForOp newLoop = scf::ForOp::create(
      rewriter, getLoc(), getLowerBound(), getUpperBound(), getStep(), inits,
      [](OpBuilder &, Location, Value, ValueRange) {}, getUnsignedCmp());
  newLoop->setAttrs(getPrunedAttributeList(getOperation(), {}));

  // Generate the new yield values and append them to the scf.yield operation.
  auto yieldOp = cast<scf::YieldOp>(getBody()->getTerminator());
  ArrayRef<BlockArgument> newIterArgs =
      newLoop.getBody()->getArguments().take_back(newInitOperands.size());
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> newYieldedValues =
        newYieldValuesFn(rewriter, getLoc(), newIterArgs);
    assert(newInitOperands.size() == newYieldedValues.size() &&
           "expected as many new yield values as new iter operands");
    rewriter.modifyOpInPlace(yieldOp, [&]() {
      yieldOp.getResultsMutable().append(newYieldedValues);
    });
  }

  // Move the loop body to the new op.
  rewriter.mergeBlocks(getBody(), newLoop.getBody(),
                       newLoop.getBody()->getArguments().take_front(
                           getBody()->getNumArguments()));

  if (replaceInitOperandUsesInLoop) {
    // Replace all uses of `newInitOperands` with the corresponding basic block
    // arguments.
    for (auto it : llvm::zip(newInitOperands, newIterArgs)) {
      rewriter.replaceUsesWithIf(std::get<0>(it), std::get<1>(it),
                                 [&](OpOperand &use) {
                                   Operation *user = use.getOwner();
                                   return newLoop->isProperAncestor(user);
                                 });
    }
  }

  // Replace the old loop.
  rewriter.replaceOp(getOperation(),
                     newLoop->getResults().take_front(getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

ForOp mlir::scf::getForInductionVarOwner(Value val) {
  auto ivArg = llvm::dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<ForOp>(containingOp);
}

OperandRange ForOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  return getInitArgs();
}

void ForOp::getSuccessorRegions(RegionBranchPoint point,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  if (std::optional<APInt> tripCount = getStaticTripCount()) {
    // The loop has a known static trip count.
    if (point.isParent()) {
      if (*tripCount == 0) {
        // The loop has zero iterations. It branches directly back to the
        // parent.
        regions.push_back(RegionSuccessor::parent());
      } else {
        // The loop has at least one iteration. It branches into the body.
        regions.push_back(RegionSuccessor(&getRegion()));
      }
      return;
    } else if (*tripCount == 1) {
      // The loop has exactly 1 iteration. Therefore, it branches from the
      // region to the parent. (No further iteration.)
      regions.push_back(RegionSuccessor::parent());
      return;
    }
  }

  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor::parent());
}

ValueRange ForOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getResults())
                              : ValueRange(getRegionIterArgs());
}

SmallVector<Region *> ForallOp::getLoopRegions() { return {&getRegion()}; }

/// Promotes the loop body of a forallOp to its containing block if it can be
/// determined that the loop has a single iteration.
LogicalResult scf::ForallOp::promoteIfSingleIteration(RewriterBase &rewriter) {
  for (auto [lb, ub, step] :
       llvm::zip(getMixedLowerBound(), getMixedUpperBound(), getMixedStep())) {
    auto tripCount =
        constantTripCount(lb, ub, step, /*isSigned=*/true, computeUbMinusLb);
    if (!tripCount.has_value() || *tripCount != 1)
      return failure();
  }

  promote(rewriter, *this);
  return success();
}

Block::BlockArgListType ForallOp::getRegionIterArgs() {
  return getBody()->getArguments().drop_front(getRank());
}

MutableArrayRef<OpOperand> ForallOp::getInitsMutable() {
  return getOutputsMutable();
}

/// Promotes the loop body of a scf::ForallOp to its containing block.
void mlir::scf::promote(RewriterBase &rewriter, scf::ForallOp forallOp) {
  OpBuilder::InsertionGuard g(rewriter);
  scf::InParallelOp terminator = forallOp.getTerminator();

  // Replace block arguments with lower bounds (replacements for IVs) and
  // outputs.
  SmallVector<Value> bbArgReplacements = forallOp.getLowerBound(rewriter);
  bbArgReplacements.append(forallOp.getOutputs().begin(),
                           forallOp.getOutputs().end());

  // Move the loop body operations to the loop's containing block.
  rewriter.inlineBlockBefore(forallOp.getBody(), forallOp->getBlock(),
                             forallOp->getIterator(), bbArgReplacements);

  // Replace the terminator with tensor.insert_slice ops.
  rewriter.setInsertionPointAfter(forallOp);
  SmallVector<Value> results;
  results.reserve(forallOp.getResults().size());
  for (auto &yieldingOp : terminator.getYieldingOps()) {
    auto parallelInsertSliceOp =
        dyn_cast<tensor::ParallelInsertSliceOp>(yieldingOp);
    if (!parallelInsertSliceOp)
      continue;

    Value dst = parallelInsertSliceOp.getDest();
    Value src = parallelInsertSliceOp.getSource();
    if (llvm::isa<TensorType>(src.getType())) {
      results.push_back(tensor::InsertSliceOp::create(
          rewriter, forallOp.getLoc(), dst.getType(), src, dst,
          parallelInsertSliceOp.getOffsets(), parallelInsertSliceOp.getSizes(),
          parallelInsertSliceOp.getStrides(),
          parallelInsertSliceOp.getStaticOffsets(),
          parallelInsertSliceOp.getStaticSizes(),
          parallelInsertSliceOp.getStaticStrides()));
    } else {
      llvm_unreachable("unsupported terminator");
    }
  }
  rewriter.replaceAllUsesWith(forallOp.getResults(), results);

  // Erase the old terminator and the loop.
  rewriter.eraseOp(terminator);
  rewriter.eraseOp(forallOp);
}

LoopNest mlir::scf::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps, ValueRange iterArgs,
    function_ref<ValueVector(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilder) {
  assert(lbs.size() == ubs.size() &&
         "expected the same number of lower and upper bounds");
  assert(lbs.size() == steps.size() &&
         "expected the same number of lower bounds and steps");

  // If there are no bounds, call the body-building function and return early.
  if (lbs.empty()) {
    ValueVector results =
        bodyBuilder ? bodyBuilder(builder, loc, ValueRange(), iterArgs)
                    : ValueVector();
    assert(results.size() == iterArgs.size() &&
           "loop nest body must return as many values as loop has iteration "
           "arguments");
    return LoopNest{{}, std::move(results)};
  }

  // First, create the loop structure iteratively using the body-builder
  // callback of `ForOp::build`. Do not create `YieldOp`s yet.
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<scf::ForOp, 4> loops;
  SmallVector<Value, 4> ivs;
  loops.reserve(lbs.size());
  ivs.reserve(lbs.size());
  ValueRange currentIterArgs = iterArgs;
  Location currentLoc = loc;
  for (unsigned i = 0, e = lbs.size(); i < e; ++i) {
    auto loop = scf::ForOp::create(
        builder, currentLoc, lbs[i], ubs[i], steps[i], currentIterArgs,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange args) {
          ivs.push_back(iv);
          // It is safe to store ValueRange args because it points to block
          // arguments of a loop operation that we also own.
          currentIterArgs = args;
          currentLoc = nestedLoc;
        });
    // Set the builder to point to the body of the newly created loop. We don't
    // do this in the callback because the builder is reset when the callback
    // returns.
    builder.setInsertionPointToStart(loop.getBody());
    loops.push_back(loop);
  }

  // For all loops but the innermost, yield the results of the nested loop.
  for (unsigned i = 0, e = loops.size() - 1; i < e; ++i) {
    builder.setInsertionPointToEnd(loops[i].getBody());
    scf::YieldOp::create(builder, loc, loops[i + 1].getResults());
  }

  // In the body of the innermost loop, call the body building function if any
  // and yield its results.
  builder.setInsertionPointToStart(loops.back().getBody());
  ValueVector results = bodyBuilder
                            ? bodyBuilder(builder, currentLoc, ivs,
                                          loops.back().getRegionIterArgs())
                            : ValueVector();
  assert(results.size() == iterArgs.size() &&
         "loop nest body must return as many values as loop has iteration "
         "arguments");
  builder.setInsertionPointToEnd(loops.back().getBody());
  scf::YieldOp::create(builder, loc, results);

  // Return the loops.
  ValueVector nestResults;
  llvm::append_range(nestResults, loops.front().getResults());
  return LoopNest{std::move(loops), std::move(nestResults)};
}

LoopNest mlir::scf::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  // Delegate to the main function by wrapping the body builder.
  return buildLoopNest(builder, loc, lbs, ubs, steps, {},
                       [&bodyBuilder](OpBuilder &nestedBuilder,
                                      Location nestedLoc, ValueRange ivs,
                                      ValueRange) -> ValueVector {
                         if (bodyBuilder)
                           bodyBuilder(nestedBuilder, nestedLoc, ivs);
                         return {};
                       });
}

SmallVector<Value>
mlir::scf::replaceAndCastForOpIterArg(RewriterBase &rewriter, scf::ForOp forOp,
                                      OpOperand &operand, Value replacement,
                                      const ValueTypeCastFnTy &castFn) {
  assert(operand.getOwner() == forOp);
  Type oldType = operand.get().getType(), newType = replacement.getType();

  // 1. Create new iter operands, exactly 1 is replaced.
  assert(operand.getOperandNumber() >= forOp.getNumControlOperands() &&
         "expected an iter OpOperand");
  assert(operand.get().getType() != replacement.getType() &&
         "Expected a different type");
  SmallVector<Value> newIterOperands;
  for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
    if (opOperand.getOperandNumber() == operand.getOperandNumber()) {
      newIterOperands.push_back(replacement);
      continue;
    }
    newIterOperands.push_back(opOperand.get());
  }

  // 2. Create the new forOp shell.
  scf::ForOp newForOp = scf::ForOp::create(
      rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands, /*bodyBuilder=*/nullptr,
      forOp.getUnsignedCmp());
  newForOp->setAttrs(forOp->getAttrs());
  Block &newBlock = newForOp.getRegion().front();
  SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                             newBlock.getArguments().end());

  // 3. Inject an incoming cast op at the beginning of the block for the bbArg
  // corresponding to the `replacement` value.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&newBlock);
  BlockArgument newRegionIterArg = newForOp.getTiedLoopRegionIterArg(
      &newForOp->getOpOperand(operand.getOperandNumber()));
  Value castIn = castFn(rewriter, newForOp.getLoc(), oldType, newRegionIterArg);
  newBlockTransferArgs[newRegionIterArg.getArgNumber()] = castIn;

  // 4. Steal the old block ops, mapping to the newBlockTransferArgs.
  Block &oldBlock = forOp.getRegion().front();
  rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

  // 5. Inject an outgoing cast op at the end of the block and yield it instead.
  auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
  rewriter.setInsertionPoint(clonedYieldOp);
  unsigned yieldIdx =
      newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
  Value castOut = castFn(rewriter, newForOp.getLoc(), newType,
                         clonedYieldOp.getOperand(yieldIdx));
  SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
  newYieldOperands[yieldIdx] = castOut;
  scf::YieldOp::create(rewriter, newForOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(clonedYieldOp);

  // 6. Inject an outgoing cast op after the forOp.
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> newResults = newForOp.getResults();
  newResults[yieldIdx] =
      castFn(rewriter, newForOp.getLoc(), oldType, newResults[yieldIdx]);

  return newResults;
}

namespace {
/// Fold scf.for iter_arg/result pairs that go through incoming/ougoing
/// a tensor.cast op pair so as to pull the tensor.cast inside the scf.for:
///
/// ```
///   %0 = tensor.cast %t0 : tensor<32x1024xf32> to tensor<?x?xf32>
///   %1 = scf.for %i = %c0 to %c1024 step %c32 iter_args(%iter_t0 = %0)
///      -> (tensor<?x?xf32>) {
///     %2 = call @do(%iter_t0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
///     scf.yield %2 : tensor<?x?xf32>
///   }
///   use_of(%1)
/// ```
///
/// folds into:
///
/// ```
///   %0 = scf.for %arg2 = %c0 to %c1024 step %c32 iter_args(%arg3 = %arg0)
///       -> (tensor<32x1024xf32>) {
///     %2 = tensor.cast %arg3 : tensor<32x1024xf32> to tensor<?x?xf32>
///     %3 = call @do(%2) : (tensor<?x?xf32>) -> tensor<?x?xf32>
///     %4 = tensor.cast %3 : tensor<?x?xf32> to tensor<32x1024xf32>
///     scf.yield %4 : tensor<32x1024xf32>
///   }
///   %1 = tensor.cast %0 : tensor<32x1024xf32> to tensor<?x?xf32>
///   use_of(%1)
/// ```
struct ForOpTensorCastFolder : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    for (auto it : llvm::zip(op.getInitArgsMutable(), op.getResults())) {
      OpOperand &iterOpOperand = std::get<0>(it);
      auto incomingCast = iterOpOperand.get().getDefiningOp<tensor::CastOp>();
      if (!incomingCast ||
          incomingCast.getSource().getType() == incomingCast.getType())
        continue;
      // If the dest type of the cast does not preserve static information in
      // the source type.
      if (!tensor::preservesStaticInformation(
              incomingCast.getDest().getType(),
              incomingCast.getSource().getType()))
        continue;
      if (!std::get<1>(it).hasOneUse())
        continue;

      // Create a new ForOp with that iter operand replaced.
      rewriter.replaceOp(
          op, replaceAndCastForOpIterArg(
                  rewriter, op, iterOpOperand, incomingCast.getSource(),
                  [](OpBuilder &b, Location loc, Type type, Value source) {
                    return tensor::CastOp::create(b, loc, type, source);
                  }));
      return success();
    }
    return failure();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ForOpTensorCastFolder>(context);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, ForOp::getOperationName());
  populateRegionBranchOpInterfaceInliningPattern(
      results, ForOp::getOperationName(),
      /*replBuilderFn=*/[](OpBuilder &builder, Location loc, Value value) {
        // scf.for has only one non-successor input value: the loop induction
        // variable. In case of a single acyclic path through the op, the IV can
        // be safely replaced with the lower bound.
        auto blockArg = cast<BlockArgument>(value);
        assert(blockArg.getArgNumber() == 0 && "expected induction variable");
        auto forOp = cast<ForOp>(blockArg.getOwner()->getParentOp());
        return forOp.getLowerBound();
      });
}

std::optional<APInt> ForOp::getConstantStep() {
  IntegerAttr step;
  if (matchPattern(getStep(), m_Constant(&step)))
    return step.getValue();
  return {};
}

std::optional<MutableArrayRef<OpOperand>> ForOp::getYieldedValuesMutable() {
  return cast<scf::YieldOp>(getBody()->getTerminator()).getResultsMutable();
}

Speculation::Speculatability ForOp::getSpeculatability() {
  // `scf.for (I = Start; I < End; I += 1)` terminates for all values of Start
  // and End.
  if (auto constantStep = getConstantStep())
    if (*constantStep == 1)
      return Speculation::RecursivelySpeculatable;

  // For Step != 1, the loop may not terminate.  We can add more smarts here if
  // needed.
  return Speculation::NotSpeculatable;
}

std::optional<APInt> ForOp::getStaticTripCount() {
  return constantTripCount(getLowerBound(), getUpperBound(), getStep(),
                           /*isSigned=*/!getUnsignedCmp(), computeUbMinusLb);
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

LogicalResult ForallOp::verify() {
  unsigned numLoops = getRank();
  // Check number of outputs.
  if (getNumResults() != getOutputs().size())
    return emitOpError("produces ")
           << getNumResults() << " results, but has only "
           << getOutputs().size() << " outputs";

  // Check that the body defines block arguments for thread indices and outputs.
  auto *body = getBody();
  if (body->getNumArguments() != numLoops + getOutputs().size())
    return emitOpError("region expects ") << numLoops << " arguments";
  for (int64_t i = 0; i < numLoops; ++i)
    if (!body->getArgument(i).getType().isIndex())
      return emitOpError("expects ")
             << i << "-th block argument to be an index";
  for (unsigned i = 0; i < getOutputs().size(); ++i)
    if (body->getArgument(i + numLoops).getType() != getOutputs()[i].getType())
      return emitOpError("type mismatch between ")
             << i << "-th output and corresponding block argument";
  if (getMapping().has_value() && !getMapping()->empty()) {
    if (getDeviceMappingAttrs().size() != numLoops)
      return emitOpError() << "mapping attribute size must match op rank";
    if (failed(getDeviceMaskingAttr()))
      return emitOpError() << getMappingAttrName()
                           << " supports at most one device masking attribute";
  }

  // Verify mixed static/dynamic control variables.
  Operation *op = getOperation();
  if (failed(verifyListOfOperandsOrIntegers(op, "lower bound", numLoops,
                                            getStaticLowerBound(),
                                            getDynamicLowerBound())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(op, "upper bound", numLoops,
                                            getStaticUpperBound(),
                                            getDynamicUpperBound())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(op, "step", numLoops,
                                            getStaticStep(), getDynamicStep())))
    return failure();

  return success();
}

void ForallOp::print(OpAsmPrinter &p) {
  Operation *op = getOperation();
  p << " (" << getInductionVars();
  if (isNormalized()) {
    p << ") in ";
    printDynamicIndexList(p, op, getDynamicUpperBound(), getStaticUpperBound(),
                          /*valueTypes=*/{}, /*scalables=*/{},
                          OpAsmParser::Delimiter::Paren);
  } else {
    p << ") = ";
    printDynamicIndexList(p, op, getDynamicLowerBound(), getStaticLowerBound(),
                          /*valueTypes=*/{}, /*scalables=*/{},
                          OpAsmParser::Delimiter::Paren);
    p << " to ";
    printDynamicIndexList(p, op, getDynamicUpperBound(), getStaticUpperBound(),
                          /*valueTypes=*/{}, /*scalables=*/{},
                          OpAsmParser::Delimiter::Paren);
    p << " step ";
    printDynamicIndexList(p, op, getDynamicStep(), getStaticStep(),
                          /*valueTypes=*/{}, /*scalables=*/{},
                          OpAsmParser::Delimiter::Paren);
  }
  printInitializationList(p, getRegionOutArgs(), getOutputs(), " shared_outs");
  p << " ";
  if (!getRegionOutArgs().empty())
    p << "-> (" << getResultTypes() << ") ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/getNumResults() > 0);
  p.printOptionalAttrDict(op->getAttrs(), {getOperandSegmentSizesAttrName(),
                                           getStaticLowerBoundAttrName(),
                                           getStaticUpperBoundAttrName(),
                                           getStaticStepAttrName()});
}

ParseResult ForallOp::parse(OpAsmParser &parser, OperationState &result) {
  OpBuilder b(parser.getContext());
  auto indexType = b.getIndexType();

  // Parse an opening `(` followed by thread index variables followed by `)`
  // TODO: when we can refer to such "induction variable"-like handles from the
  // declarative assembly format, we can implement the parser as a custom hook.
  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  DenseI64ArrayAttr staticLbs, staticUbs, staticSteps;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicLbs, dynamicUbs,
      dynamicSteps;
  if (succeeded(parser.parseOptionalKeyword("in"))) {
    // Parse upper bounds.
    if (parseDynamicIndexList(parser, dynamicUbs, staticUbs,
                              /*valueTypes=*/nullptr,
                              OpAsmParser::Delimiter::Paren) ||
        parser.resolveOperands(dynamicUbs, indexType, result.operands))
      return failure();

    unsigned numLoops = ivs.size();
    staticLbs = b.getDenseI64ArrayAttr(SmallVector<int64_t>(numLoops, 0));
    staticSteps = b.getDenseI64ArrayAttr(SmallVector<int64_t>(numLoops, 1));
  } else {
    // Parse lower bounds.
    if (parser.parseEqual() ||
        parseDynamicIndexList(parser, dynamicLbs, staticLbs,
                              /*valueTypes=*/nullptr,
                              OpAsmParser::Delimiter::Paren) ||

        parser.resolveOperands(dynamicLbs, indexType, result.operands))
      return failure();

    // Parse upper bounds.
    if (parser.parseKeyword("to") ||
        parseDynamicIndexList(parser, dynamicUbs, staticUbs,
                              /*valueTypes=*/nullptr,
                              OpAsmParser::Delimiter::Paren) ||
        parser.resolveOperands(dynamicUbs, indexType, result.operands))
      return failure();

    // Parse step values.
    if (parser.parseKeyword("step") ||
        parseDynamicIndexList(parser, dynamicSteps, staticSteps,
                              /*valueTypes=*/nullptr,
                              OpAsmParser::Delimiter::Paren) ||
        parser.resolveOperands(dynamicSteps, indexType, result.operands))
      return failure();
  }

  // Parse out operands and results.
  SmallVector<OpAsmParser::Argument, 4> regionOutArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> outOperands;
  SMLoc outOperandsLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("shared_outs"))) {
    if (outOperands.size() != result.types.size())
      return parser.emitError(outOperandsLoc,
                              "mismatch between out operands and types");
    if (parser.parseAssignmentList(regionOutArgs, outOperands) ||
        parser.parseOptionalArrowTypeList(result.types) ||
        parser.resolveOperands(outOperands, result.types, outOperandsLoc,
                               result.operands))
      return failure();
  }

  // Parse region.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  for (auto &iv : ivs) {
    iv.type = b.getIndexType();
    regionArgs.push_back(iv);
  }
  for (const auto &it : llvm::enumerate(regionOutArgs)) {
    auto &out = it.value();
    out.type = result.types[it.index()];
    regionArgs.push_back(out);
  }
  if (parser.parseRegion(*region, regionArgs))
    return failure();

  // Ensure terminator and move region.
  ForallOp::ensureTerminator(*region, b, result.location);
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addAttribute("staticLowerBound", staticLbs);
  result.addAttribute("staticUpperBound", staticUbs);
  result.addAttribute("staticStep", staticSteps);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(dynamicLbs.size()),
                           static_cast<int32_t>(dynamicUbs.size()),
                           static_cast<int32_t>(dynamicSteps.size()),
                           static_cast<int32_t>(outOperands.size())}));
  return success();
}

// Builder that takes loop bounds.
void ForallOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs,
    ArrayRef<OpFoldResult> steps, ValueRange outputs,
    std::optional<ArrayAttr> mapping,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  SmallVector<int64_t> staticLbs, staticUbs, staticSteps;
  SmallVector<Value> dynamicLbs, dynamicUbs, dynamicSteps;
  dispatchIndexOpFoldResults(lbs, dynamicLbs, staticLbs);
  dispatchIndexOpFoldResults(ubs, dynamicUbs, staticUbs);
  dispatchIndexOpFoldResults(steps, dynamicSteps, staticSteps);

  result.addOperands(dynamicLbs);
  result.addOperands(dynamicUbs);
  result.addOperands(dynamicSteps);
  result.addOperands(outputs);
  result.addTypes(TypeRange(outputs));

  result.addAttribute(getStaticLowerBoundAttrName(result.name),
                      b.getDenseI64ArrayAttr(staticLbs));
  result.addAttribute(getStaticUpperBoundAttrName(result.name),
                      b.getDenseI64ArrayAttr(staticUbs));
  result.addAttribute(getStaticStepAttrName(result.name),
                      b.getDenseI64ArrayAttr(staticSteps));
  result.addAttribute(
      "operandSegmentSizes",
      b.getDenseI32ArrayAttr({static_cast<int32_t>(dynamicLbs.size()),
                              static_cast<int32_t>(dynamicUbs.size()),
                              static_cast<int32_t>(dynamicSteps.size()),
                              static_cast<int32_t>(outputs.size())}));
  if (mapping.has_value()) {
    result.addAttribute(ForallOp::getMappingAttrName(result.name),
                        mapping.value());
  }

  Region *bodyRegion = result.addRegion();
  OpBuilder::InsertionGuard g(b);
  b.createBlock(bodyRegion);
  Block &bodyBlock = bodyRegion->front();

  // Add block arguments for indices and outputs.
  bodyBlock.addArguments(
      SmallVector<Type>(lbs.size(), b.getIndexType()),
      SmallVector<Location>(staticLbs.size(), result.location));
  bodyBlock.addArguments(
      TypeRange(outputs),
      SmallVector<Location>(outputs.size(), result.location));

  b.setInsertionPointToStart(&bodyBlock);
  if (!bodyBuilderFn) {
    ForallOp::ensureTerminator(*bodyRegion, b, result.location);
    return;
  }
  bodyBuilderFn(b, result.location, bodyBlock.getArguments());
}

// Builder that takes loop bounds.
void ForallOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    ArrayRef<OpFoldResult> ubs, ValueRange outputs,
    std::optional<ArrayAttr> mapping,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  unsigned numLoops = ubs.size();
  SmallVector<OpFoldResult> lbs(numLoops, b.getIndexAttr(0));
  SmallVector<OpFoldResult> steps(numLoops, b.getIndexAttr(1));
  build(b, result, lbs, ubs, steps, outputs, mapping, bodyBuilderFn);
}

// Checks if the lbs are zeros and steps are ones.
bool ForallOp::isNormalized() {
  auto allEqual = [](ArrayRef<OpFoldResult> results, int64_t val) {
    return llvm::all_of(results, [&](OpFoldResult ofr) {
      auto intValue = getConstantIntValue(ofr);
      return intValue.has_value() && intValue == val;
    });
  };
  return allEqual(getMixedLowerBound(), 0) && allEqual(getMixedStep(), 1);
}

InParallelOp ForallOp::getTerminator() {
  return cast<InParallelOp>(getBody()->getTerminator());
}

SmallVector<Operation *> ForallOp::getCombiningOps(BlockArgument bbArg) {
  SmallVector<Operation *> storeOps;
  for (Operation *user : bbArg.getUsers()) {
    if (auto parallelOp = dyn_cast<ParallelCombiningOpInterface>(user)) {
      storeOps.push_back(parallelOp);
    }
  }
  return storeOps;
}

SmallVector<DeviceMappingAttrInterface> ForallOp::getDeviceMappingAttrs() {
  SmallVector<DeviceMappingAttrInterface> res;
  if (!getMapping())
    return res;
  for (auto attr : getMapping()->getValue()) {
    auto m = dyn_cast<DeviceMappingAttrInterface>(attr);
    if (m)
      res.push_back(m);
  }
  return res;
}

FailureOr<DeviceMaskingAttrInterface> ForallOp::getDeviceMaskingAttr() {
  DeviceMaskingAttrInterface res;
  if (!getMapping())
    return res;
  for (auto attr : getMapping()->getValue()) {
    auto m = dyn_cast<DeviceMaskingAttrInterface>(attr);
    if (m && res)
      return failure();
    if (m)
      res = m;
  }
  return res;
}

bool ForallOp::usesLinearMapping() {
  SmallVector<DeviceMappingAttrInterface> ifaces = getDeviceMappingAttrs();
  if (ifaces.empty())
    return false;
  return ifaces.front().isLinearMapping();
}

std::optional<SmallVector<Value>> ForallOp::getLoopInductionVars() {
  return SmallVector<Value>{getBody()->getArguments().take_front(getRank())};
}

// Get lower bounds as OpFoldResult.
std::optional<SmallVector<OpFoldResult>> ForallOp::getLoopLowerBounds() {
  Builder b(getOperation()->getContext());
  return getMixedValues(getStaticLowerBound(), getDynamicLowerBound(), b);
}

// Get upper bounds as OpFoldResult.
std::optional<SmallVector<OpFoldResult>> ForallOp::getLoopUpperBounds() {
  Builder b(getOperation()->getContext());
  return getMixedValues(getStaticUpperBound(), getDynamicUpperBound(), b);
}

// Get steps as OpFoldResult.
std::optional<SmallVector<OpFoldResult>> ForallOp::getLoopSteps() {
  Builder b(getOperation()->getContext());
  return getMixedValues(getStaticStep(), getDynamicStep(), b);
}

ForallOp mlir::scf::getForallOpThreadIndexOwner(Value val) {
  auto tidxArg = llvm::dyn_cast<BlockArgument>(val);
  if (!tidxArg)
    return ForallOp();
  assert(tidxArg.getOwner() && "unlinked block argument");
  auto *containingOp = tidxArg.getOwner()->getParentOp();
  return dyn_cast<ForallOp>(containingOp);
}

namespace {
/// Fold tensor.dim(forall shared_outs(... = %t)) to tensor.dim(%t).
struct DimOfForallOp : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::DimOp dimOp,
                                PatternRewriter &rewriter) const final {
    auto forallOp = dimOp.getSource().getDefiningOp<ForallOp>();
    if (!forallOp)
      return failure();
    Value sharedOut =
        forallOp.getTiedOpOperand(llvm::cast<OpResult>(dimOp.getSource()))
            ->get();
    rewriter.modifyOpInPlace(
        dimOp, [&]() { dimOp.getSourceMutable().assign(sharedOut); });
    return success();
  }
};

class ForallOpControlOperandsFolder : public OpRewritePattern<ForallOp> {
public:
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedLowerBound(op.getMixedLowerBound());
    SmallVector<OpFoldResult> mixedUpperBound(op.getMixedUpperBound());
    SmallVector<OpFoldResult> mixedStep(op.getMixedStep());
    if (failed(foldDynamicIndexList(mixedLowerBound)) &&
        failed(foldDynamicIndexList(mixedUpperBound)) &&
        failed(foldDynamicIndexList(mixedStep)))
      return failure();

    rewriter.modifyOpInPlace(op, [&]() {
      SmallVector<Value> dynamicLowerBound, dynamicUpperBound, dynamicStep;
      SmallVector<int64_t> staticLowerBound, staticUpperBound, staticStep;
      dispatchIndexOpFoldResults(mixedLowerBound, dynamicLowerBound,
                                 staticLowerBound);
      op.getDynamicLowerBoundMutable().assign(dynamicLowerBound);
      op.setStaticLowerBound(staticLowerBound);

      dispatchIndexOpFoldResults(mixedUpperBound, dynamicUpperBound,
                                 staticUpperBound);
      op.getDynamicUpperBoundMutable().assign(dynamicUpperBound);
      op.setStaticUpperBound(staticUpperBound);

      dispatchIndexOpFoldResults(mixedStep, dynamicStep, staticStep);
      op.getDynamicStepMutable().assign(dynamicStep);
      op.setStaticStep(staticStep);

      op->setAttr(ForallOp::getOperandSegmentSizeAttr(),
                  rewriter.getDenseI32ArrayAttr(
                      {static_cast<int32_t>(dynamicLowerBound.size()),
                       static_cast<int32_t>(dynamicUpperBound.size()),
                       static_cast<int32_t>(dynamicStep.size()),
                       static_cast<int32_t>(op.getNumResults())}));
    });
    return success();
  }
};

/// The following canonicalization pattern folds the iter arguments of
/// scf.forall op if :-
/// 1. The corresponding result has zero uses.
/// 2. The iter argument is NOT being modified within the loop body.
/// uses.
///
/// Example of first case :-
///  INPUT:
///   %res:3 = scf.forall ... shared_outs(%arg0 = %a, %arg1 = %b, %arg2 = %c)
///            {
///                ...
///                <SOME USE OF %arg0>
///                <SOME USE OF %arg1>
///                <SOME USE OF %arg2>
///                ...
///                scf.forall.in_parallel {
///                    <STORE OP WITH DESTINATION %arg1>
///                    <STORE OP WITH DESTINATION %arg0>
///                    <STORE OP WITH DESTINATION %arg2>
///                }
///             }
///   return %res#1
///
///  OUTPUT:
///   %res:3 = scf.forall ... shared_outs(%new_arg0 = %b)
///            {
///                ...
///                <SOME USE OF %a>
///                <SOME USE OF %new_arg0>
///                <SOME USE OF %c>
///                ...
///                scf.forall.in_parallel {
///                    <STORE OP WITH DESTINATION %new_arg0>
///                }
///             }
///   return %res
///
/// NOTE: 1. All uses of the folded shared_outs (iter argument) within the
///          scf.forall is replaced by their corresponding operands.
///       2. Even if there are <STORE OP WITH DESTINATION *> ops within the body
///          of the scf.forall besides within scf.forall.in_parallel terminator,
///          this canonicalization remains valid. For more details, please refer
///          to :
///          https://github.com/llvm/llvm-project/pull/90189#discussion_r1589011124
///       3. TODO(avarma): Generalize it for other store ops. Currently it
///          handles tensor.parallel_insert_slice ops only.
///
/// Example of second case :-
///  INPUT:
///   %res:2 = scf.forall ... shared_outs(%arg0 = %a, %arg1 = %b)
///            {
///                ...
///                <SOME USE OF %arg0>
///                <SOME USE OF %arg1>
///                ...
///                scf.forall.in_parallel {
///                    <STORE OP WITH DESTINATION %arg1>
///                }
///             }
///   return %res#0, %res#1
///
///  OUTPUT:
///   %res = scf.forall ... shared_outs(%new_arg0 = %b)
///            {
///                ...
///                <SOME USE OF %a>
///                <SOME USE OF %new_arg0>
///                ...
///                scf.forall.in_parallel {
///                    <STORE OP WITH DESTINATION %new_arg0>
///                }
///             }
///   return %a, %res
struct ForallOpIterArgsFolder : public OpRewritePattern<ForallOp> {
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp forallOp,
                                PatternRewriter &rewriter) const final {
    // Step 1: For a given i-th result of scf.forall, check the following :-
    //         a. If it has any use.
    //         b. If the corresponding iter argument is being modified within
    //            the loop, i.e. has at least one store op with the iter arg as
    //            its destination operand. For this we use
    //            ForallOp::getCombiningOps(iter_arg).
    //
    //         Based on the check we maintain the following :-
    //         a. op results, block arguments, outputs to delete
    //         b. new outputs (i.e., outputs to retain)
    SmallVector<Value> resultsToDelete;
    SmallVector<Value> outsToDelete;
    SmallVector<BlockArgument> blockArgsToDelete;
    SmallVector<Value> newOuts;
    BitVector resultIndicesToDelete(forallOp.getNumResults(), false);
    BitVector blockIndicesToDelete(forallOp.getBody()->getNumArguments(),
                                   false);
    for (OpResult result : forallOp.getResults()) {
      OpOperand *opOperand = forallOp.getTiedOpOperand(result);
      BlockArgument blockArg = forallOp.getTiedBlockArgument(opOperand);
      if (result.use_empty() || forallOp.getCombiningOps(blockArg).empty()) {
        resultsToDelete.push_back(result);
        outsToDelete.push_back(opOperand->get());
        blockArgsToDelete.push_back(blockArg);
        resultIndicesToDelete[result.getResultNumber()] = true;
        blockIndicesToDelete[blockArg.getArgNumber()] = true;
      } else {
        newOuts.push_back(opOperand->get());
      }
    }

    // Return early if all results of scf.forall have at least one use and being
    // modified within the loop.
    if (resultsToDelete.empty())
      return failure();

    // Step 2: Erase combining ops and replace uses of deleted results and
    //         block arguments with the corresponding outputs.
    for (auto blockArg : blockArgsToDelete) {
      SmallVector<Operation *> combiningOps =
          forallOp.getCombiningOps(blockArg);
      for (Operation *combiningOp : combiningOps)
        rewriter.eraseOp(combiningOp);
    }
    for (auto [blockArg, result, out] :
         llvm::zip_equal(blockArgsToDelete, resultsToDelete, outsToDelete)) {
      rewriter.replaceAllUsesWith(blockArg, out);
      rewriter.replaceAllUsesWith(result, out);
    }
    // TODO: There is no rewriter API for erasing block arguments.
    rewriter.modifyOpInPlace(forallOp, [&]() {
      forallOp.getBody()->eraseArguments(blockIndicesToDelete);
    });

    // Step 3. Create a new scf.forall op with only the shared_outs/results
    //         that should be retained.
    auto newForallOp = cast<scf::ForallOp>(
        rewriter.eraseOpResults(forallOp, resultIndicesToDelete));
    newForallOp.getOutputsMutable().assign(newOuts);

    return success();
  }
};

struct ForallOpSingleOrZeroIterationDimsFolder
    : public OpRewritePattern<ForallOp> {
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    // Do not fold dimensions if they are mapped to processing units.
    if (op.getMapping().has_value() && !op.getMapping()->empty())
      return failure();
    Location loc = op.getLoc();

    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<OpFoldResult> newMixedLowerBounds, newMixedUpperBounds,
        newMixedSteps;
    IRMapping mapping;
    for (auto [lb, ub, step, iv] :
         llvm::zip(op.getMixedLowerBound(), op.getMixedUpperBound(),
                   op.getMixedStep(), op.getInductionVars())) {
      auto numIterations =
          constantTripCount(lb, ub, step, /*isSigned=*/true, computeUbMinusLb);
      if (numIterations.has_value()) {
        // Remove the loop if it performs zero iterations.
        if (*numIterations == 0) {
          rewriter.replaceOp(op, op.getOutputs());
          return success();
        }
        // Replace the loop induction variable by the lower bound if the loop
        // performs a single iteration. Otherwise, copy the loop bounds.
        if (*numIterations == 1) {
          mapping.map(iv, getValueOrCreateConstantIndexOp(rewriter, loc, lb));
          continue;
        }
      }
      newMixedLowerBounds.push_back(lb);
      newMixedUpperBounds.push_back(ub);
      newMixedSteps.push_back(step);
    }

    // All of the loop dimensions perform a single iteration. Inline loop body.
    if (newMixedLowerBounds.empty()) {
      promote(rewriter, op);
      return success();
    }

    // Exit if none of the loop dimensions perform a single iteration.
    if (newMixedLowerBounds.size() == static_cast<unsigned>(op.getRank())) {
      return rewriter.notifyMatchFailure(
          op, "no dimensions have 0 or 1 iterations");
    }

    // Replace the loop by a lower-dimensional loop.
    ForallOp newOp;
    newOp = ForallOp::create(rewriter, loc, newMixedLowerBounds,
                             newMixedUpperBounds, newMixedSteps,
                             op.getOutputs(), std::nullopt, nullptr);
    newOp.getBodyRegion().getBlocks().clear();
    // The new loop needs to keep all attributes from the old one, except for
    // "operandSegmentSizes" and static loop bound attributes which capture
    // the outdated information of the old iteration domain.
    SmallVector<StringAttr> elidedAttrs{newOp.getOperandSegmentSizesAttrName(),
                                        newOp.getStaticLowerBoundAttrName(),
                                        newOp.getStaticUpperBoundAttrName(),
                                        newOp.getStaticStepAttrName()};
    for (const auto &namedAttr : op->getAttrs()) {
      if (llvm::is_contained(elidedAttrs, namedAttr.getName()))
        continue;
      rewriter.modifyOpInPlace(newOp, [&]() {
        newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
      });
    }
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Replace all induction vars with a single trip count with their lower bound.
struct ForallOpReplaceConstantInductionVar : public OpRewritePattern<ForallOp> {
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool changed = false;
    for (auto [lb, ub, step, iv] :
         llvm::zip(op.getMixedLowerBound(), op.getMixedUpperBound(),
                   op.getMixedStep(), op.getInductionVars())) {
      if (iv.hasNUses(0))
        continue;
      auto numIterations =
          constantTripCount(lb, ub, step, /*isSigned=*/true, computeUbMinusLb);
      if (!numIterations.has_value() || numIterations.value() != 1) {
        continue;
      }
      rewriter.replaceAllUsesWith(
          iv, getValueOrCreateConstantIndexOp(rewriter, loc, lb));
      changed = true;
    }
    return success(changed);
  }
};

struct FoldTensorCastOfOutputIntoForallOp
    : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  struct TypeCast {
    Type srcType;
    Type dstType;
  };

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const final {
    llvm::SmallMapVector<unsigned, TypeCast, 2> tensorCastProducers;
    llvm::SmallVector<Value> newOutputTensors = forallOp.getOutputs();
    for (auto en : llvm::enumerate(newOutputTensors)) {
      auto castOp = en.value().getDefiningOp<tensor::CastOp>();
      if (!castOp)
        continue;

      // Only casts that that preserve static information, i.e. will make the
      // loop result type "more" static than before, will be folded.
      if (!tensor::preservesStaticInformation(castOp.getDest().getType(),
                                              castOp.getSource().getType())) {
        continue;
      }

      tensorCastProducers[en.index()] =
          TypeCast{castOp.getSource().getType(), castOp.getType()};
      newOutputTensors[en.index()] = castOp.getSource();
    }

    if (tensorCastProducers.empty())
      return failure();

    // Create new loop.
    Location loc = forallOp.getLoc();
    auto newForallOp = ForallOp::create(
        rewriter, loc, forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(),
        newOutputTensors, forallOp.getMapping(),
        [&](OpBuilder nestedBuilder, Location nestedLoc, ValueRange bbArgs) {
          auto castBlockArgs =
              llvm::to_vector(bbArgs.take_back(forallOp->getNumResults()));
          for (auto [index, cast] : tensorCastProducers) {
            Value &oldTypeBBArg = castBlockArgs[index];
            oldTypeBBArg = tensor::CastOp::create(nestedBuilder, nestedLoc,
                                                  cast.dstType, oldTypeBBArg);
          }

          // Move old body into new parallel loop.
          SmallVector<Value> ivsBlockArgs =
              llvm::to_vector(bbArgs.take_front(forallOp.getRank()));
          ivsBlockArgs.append(castBlockArgs);
          rewriter.mergeBlocks(forallOp.getBody(),
                               bbArgs.front().getParentBlock(), ivsBlockArgs);
        });

    // After `mergeBlocks` happened, the destinations in the terminator were
    // mapped to the tensor.cast old-typed results of the output bbArgs. The
    // destination have to be updated to point to the output bbArgs directly.
    auto terminator = newForallOp.getTerminator();
    for (auto [yieldingOp, outputBlockArg] : llvm::zip(
             terminator.getYieldingOps(), newForallOp.getRegionIterArgs())) {
      if (auto parallelCombingingOp =
              dyn_cast<ParallelCombiningOpInterface>(yieldingOp)) {
        parallelCombingingOp.getUpdatedDestinations().assign(outputBlockArg);
      }
    }

    // Cast results back to the original types.
    rewriter.setInsertionPointAfter(newForallOp);
    SmallVector<Value> castResults = newForallOp.getResults();
    for (auto &item : tensorCastProducers) {
      Value &oldTypeResult = castResults[item.first];
      oldTypeResult = tensor::CastOp::create(rewriter, loc, item.second.dstType,
                                             oldTypeResult);
    }
    rewriter.replaceOp(forallOp, castResults);
    return success();
  }
};

} // namespace

void ForallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<DimOfForallOp, FoldTensorCastOfOutputIntoForallOp,
              ForallOpControlOperandsFolder, ForallOpIterArgsFolder,
              ForallOpSingleOrZeroIterationDimsFolder,
              ForallOpReplaceConstantInductionVar>(context);
}

void ForallOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  // There are two region branch points:
  // 1. "parent": entering the forall op for the first time.
  // 2. scf.in_parallel terminator
  if (point.isParent()) {
    // When first entering the forall op, the control flow typically branches
    // into the forall body. (In parallel for multiple threads.)
    regions.push_back(RegionSuccessor(&getRegion()));
    // However, when there are 0 threads, the control flow may branch back to
    // the parent immediately.
    regions.push_back(RegionSuccessor::parent());
  } else {
    // In accordance with the semantics of forall, its body is executed in
    // parallel by multiple threads. We should not expect to branch back into
    // the forall body after the region's execution is complete.
    regions.push_back(RegionSuccessor::parent());
  }
}

//===----------------------------------------------------------------------===//
// InParallelOp
//===----------------------------------------------------------------------===//

// Build a InParallelOp with mixed static and dynamic entries.
void InParallelOp::build(OpBuilder &b, OperationState &result) {
  OpBuilder::InsertionGuard g(b);
  Region *bodyRegion = result.addRegion();
  b.createBlock(bodyRegion);
}

LogicalResult InParallelOp::verify() {
  scf::ForallOp forallOp =
      dyn_cast<scf::ForallOp>(getOperation()->getParentOp());
  if (!forallOp)
    return this->emitOpError("expected forall op parent");

  for (Operation &op : getRegion().front().getOperations()) {
    auto parallelCombiningOp = dyn_cast<ParallelCombiningOpInterface>(&op);
    if (!parallelCombiningOp) {
      return this->emitOpError("expected only ParallelCombiningOpInterface")
             << " ops";
    }

    // Verify that inserts are into out block arguments.
    MutableOperandRange dests = parallelCombiningOp.getUpdatedDestinations();
    ArrayRef<BlockArgument> regionOutArgs = forallOp.getRegionOutArgs();
    for (OpOperand &dest : dests) {
      if (!llvm::is_contained(regionOutArgs, dest.get()))
        return op.emitOpError("may only insert into an output block argument");
    }
  }

  return success();
}

void InParallelOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult InParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::Argument, 8> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, regionOperands))
    return failure();

  if (region->empty())
    OpBuilder(builder.getContext()).createBlock(region.get());
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

OpResult InParallelOp::getParentResult(int64_t idx) {
  return getOperation()->getParentOp()->getResult(idx);
}

SmallVector<BlockArgument> InParallelOp::getDests() {
  SmallVector<BlockArgument> updatedDests;
  for (Operation &yieldingOp : getYieldingOps()) {
    auto parallelCombiningOp =
        dyn_cast<ParallelCombiningOpInterface>(&yieldingOp);
    if (!parallelCombiningOp)
      continue;
    for (OpOperand &updatedOperand :
         parallelCombiningOp.getUpdatedDestinations())
      updatedDests.push_back(cast<BlockArgument>(updatedOperand.get()));
  }
  return updatedDests;
}

llvm::iterator_range<Block::iterator> InParallelOp::getYieldingOps() {
  return getRegion().front().getOperations();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

bool mlir::scf::insideMutuallyExclusiveBranches(Operation *a, Operation *b) {
  assert(a && "expected non-empty operation");
  assert(b && "expected non-empty operation");

  IfOp ifOp = a->getParentOfType<IfOp>();
  while (ifOp) {
    // Check if b is inside ifOp. (We already know that a is.)
    if (ifOp->isProperAncestor(b))
      // b is contained in ifOp. a and b are in mutually exclusive branches if
      // they are in different blocks of ifOp.
      return static_cast<bool>(ifOp.thenBlock()->findAncestorOpInBlock(*a)) !=
             static_cast<bool>(ifOp.thenBlock()->findAncestorOpInBlock(*b));
    // Check next enclosing IfOp.
    ifOp = ifOp->getParentOfType<IfOp>();
  }

  // Could not find a common IfOp among a's and b's ancestors.
  return false;
}

LogicalResult
IfOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                       IfOp::Adaptor adaptor,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getRegions().empty())
    return failure();
  Region *r = &adaptor.getThenRegion();
  if (r->empty())
    return failure();
  Block &b = r->front();
  if (b.empty())
    return failure();
  auto yieldOp = llvm::dyn_cast<YieldOp>(b.back());
  if (!yieldOp)
    return failure();
  TypeRange types = yieldOp.getOperandTypes();
  llvm::append_range(inferredReturnTypes, types);
  return success();
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond) {
  return build(builder, result, resultTypes, cond, /*addThenBlock=*/false,
               /*addElseBlock=*/false);
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond, bool addThenBlock,
                 bool addElseBlock) {
  assert((!addElseBlock || addThenBlock) &&
         "must not create else block w/o then block");
  result.addTypes(resultTypes);
  result.addOperands(cond);

  // Add regions and blocks.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  if (addThenBlock)
    builder.createBlock(thenRegion);
  Region *elseRegion = result.addRegion();
  if (addElseBlock)
    builder.createBlock(elseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  build(builder, result, TypeRange{}, cond, withElseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond, bool withElseRegion) {
  result.addTypes(resultTypes);
  result.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  if (resultTypes.empty())
    IfOp::ensureTerminator(*thenRegion, builder, result.location);

  // Build else region.
  Region *elseRegion = result.addRegion();
  if (withElseRegion) {
    builder.createBlock(elseRegion);
    if (resultTypes.empty())
      IfOp::ensureTerminator(*elseRegion, builder, result.location);
  }
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");
  result.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  // Build else region.
  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }

  // Infer result types.
  SmallVector<Type> inferredReturnTypes;
  MLIRContext *ctx = builder.getContext();
  auto attrDict = DictionaryAttr::get(ctx, result.attributes);
  if (succeeded(inferReturnTypes(ctx, std::nullopt, result.operands, attrDict,
                                 /*properties=*/nullptr, result.regions,
                                 inferredReturnTypes))) {
    result.addTypes(inferredReturnTypes);
  }
}

LogicalResult IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");
  return success();
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
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

void IfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation or one
  // of the recursive parent operations (early exit case).
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor::parent());
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    regions.push_back(RegionSuccessor::parent());
  else
    regions.push_back(RegionSuccessor(elseRegion));
}

ValueRange IfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getOperation()->getResults())
                              : ValueRange();
}

void IfOp::getEntrySuccessorRegions(ArrayRef<Attribute> operands,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  FoldAdaptor adaptor(operands, *this);
  auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCondition());
  if (!boolAttr || boolAttr.getValue())
    regions.emplace_back(&getThenRegion());

  // If the else region is empty, execution continues after the parent op.
  if (!boolAttr || !boolAttr.getValue()) {
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    else
      regions.emplace_back(RegionSuccessor::parent());
  }
}

LogicalResult IfOp::fold(FoldAdaptor adaptor,
                         SmallVectorImpl<OpFoldResult> &results) {
  // if (!c) then A() else B() -> if c then B() else A()
  if (getElseRegion().empty())
    return failure();

  arith::XOrIOp xorStmt = getCondition().getDefiningOp<arith::XOrIOp>();
  if (!xorStmt)
    return failure();

  if (!matchPattern(xorStmt.getRhs(), m_One()))
    return failure();

  getConditionMutable().assign(xorStmt.getLhs());
  Block *thenBlock = &getThenRegion().front();
  // It would be nicer to use iplist::swap, but that has no implemented
  // callbacks See: https://llvm.org/doxygen/ilist_8h_source.html#l00224
  getThenRegion().getBlocks().splice(getThenRegion().getBlocks().begin(),
                                     getElseRegion().getBlocks());
  getElseRegion().getBlocks().splice(getElseRegion().getBlocks().begin(),
                                     getThenRegion().getBlocks(), thenBlock);
  return success();
}

void IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  if (auto cond = llvm::dyn_cast_or_null<BoolAttr>(operands[0])) {
    // If the condition is known, then one region is known to be executed once
    // and the other zero times.
    invocationBounds.emplace_back(0, cond.getValue() ? 1 : 0);
    invocationBounds.emplace_back(0, cond.getValue() ? 0 : 1);
  } else {
    // Non-constant condition. Each region may be executed 0 or 1 times.
    invocationBounds.assign(2, {0, 1});
  }
}

namespace {
/// Hoist any yielded results whose operands are defined outside
/// the if, to a select instruction.
struct ConvertTrivialIfToSelect : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0)
      return failure();

    auto cond = op.getCondition();
    auto thenYieldArgs = op.thenYield().getOperands();
    auto elseYieldArgs = op.elseYield().getOperands();

    SmallVector<Type> nonHoistable;
    for (auto [trueVal, falseVal] : llvm::zip(thenYieldArgs, elseYieldArgs)) {
      if (&op.getThenRegion() == trueVal.getParentRegion() ||
          &op.getElseRegion() == falseVal.getParentRegion())
        nonHoistable.push_back(trueVal.getType());
    }
    // Early exit if there aren't any yielded values we can
    // hoist outside the if.
    if (nonHoistable.size() == op->getNumResults())
      return failure();

    IfOp replacement = IfOp::create(rewriter, op.getLoc(), nonHoistable, cond,
                                    /*withElseRegion=*/false);
    if (replacement.thenBlock())
      rewriter.eraseBlock(replacement.thenBlock());
    replacement.getThenRegion().takeBody(op.getThenRegion());
    replacement.getElseRegion().takeBody(op.getElseRegion());

    SmallVector<Value> results(op->getNumResults());
    assert(thenYieldArgs.size() == results.size());
    assert(elseYieldArgs.size() == results.size());

    SmallVector<Value> trueYields;
    SmallVector<Value> falseYields;
    rewriter.setInsertionPoint(replacement);
    for (const auto &it :
         llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
      Value trueVal = std::get<0>(it.value());
      Value falseVal = std::get<1>(it.value());
      if (&replacement.getThenRegion() == trueVal.getParentRegion() ||
          &replacement.getElseRegion() == falseVal.getParentRegion()) {
        results[it.index()] = replacement.getResult(trueYields.size());
        trueYields.push_back(trueVal);
        falseYields.push_back(falseVal);
      } else if (trueVal == falseVal)
        results[it.index()] = trueVal;
      else
        results[it.index()] = arith::SelectOp::create(rewriter, op.getLoc(),
                                                      cond, trueVal, falseVal);
    }

    rewriter.setInsertionPointToEnd(replacement.thenBlock());
    rewriter.replaceOpWithNewOp<YieldOp>(replacement.thenYield(), trueYields);

    rewriter.setInsertionPointToEnd(replacement.elseBlock());
    rewriter.replaceOpWithNewOp<YieldOp>(replacement.elseYield(), falseYields);

    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Allow the true region of an if to assume the condition is true
/// and vice versa. For example:
///
///   scf.if %cmp {
///      print(%cmp)
///   }
///
///  becomes
///
///   scf.if %cmp {
///      print(true)
///   }
///
struct ConditionPropagation : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  /// Kind of parent region in the ancestor cache.
  enum class Parent { Then, Else, None };

  /// Returns the kind of region ("then", "else", or "none") of the
  /// IfOp that the given region is transitively nested in. Updates
  /// the cache accordingly.
  static Parent getParentType(Region *toCheck, IfOp op,
                              DenseMap<Region *, Parent> &cache,
                              Region *endRegion) {
    SmallVector<Region *> seen;
    while (toCheck != endRegion) {
      auto found = cache.find(toCheck);
      if (found != cache.end())
        return found->second;
      seen.push_back(toCheck);
      if (&op.getThenRegion() == toCheck) {
        for (Region *region : seen)
          cache[region] = Parent::Then;
        return Parent::Then;
      }
      if (&op.getElseRegion() == toCheck) {
        for (Region *region : seen)
          cache[region] = Parent::Else;
        return Parent::Else;
      }
      toCheck = toCheck->getParentRegion();
    }

    for (Region *region : seen)
      cache[region] = Parent::None;
    return Parent::None;
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Early exit if the condition is constant since replacing a constant
    // in the body with another constant isn't a simplification.
    if (matchPattern(op.getCondition(), m_Constant()))
      return failure();

    bool changed = false;
    mlir::Type i1Ty = rewriter.getI1Type();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;
    Value constantFalse = nullptr;

    DenseMap<Region *, Parent> cache;
    for (OpOperand &use :
         llvm::make_early_inc_range(op.getCondition().getUses())) {
      switch (getParentType(use.getOwner()->getParentRegion(), op, cache,
                            op.getCondition().getParentRegion())) {
      case Parent::Then: {
        changed = true;

        if (!constantTrue)
          constantTrue = arith::ConstantOp::create(
              rewriter, op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantTrue); });
        break;
      }
      case Parent::Else: {
        changed = true;

        if (!constantFalse)
          constantFalse = arith::ConstantOp::create(
              rewriter, op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 0));

        rewriter.modifyOpInPlace(use.getOwner(),
                                 [&]() { use.set(constantFalse); });
        break;
      }
      case Parent::None:
        break;
      }
    }

    return success(changed);
  }
};

/// Remove any statements from an if that are equivalent to the condition
/// or its negation. For example:
///
///    %res:2 = scf.if %cmp {
///       yield something(), true
///    } else {
///       yield something2(), false
///    }
///    print(%res#1)
///
///  becomes
///    %res = scf.if %cmp {
///       yield something()
///    } else {
///       yield something2()
///    }
///    print(%cmp)
///
/// Additionally if both branches yield the same value, replace all uses
/// of the result with the yielded value.
///
///    %res:2 = scf.if %cmp {
///       yield something(), %arg1
///    } else {
///       yield something2(), %arg1
///    }
///    print(%res#1)
///
///  becomes
///    %res = scf.if %cmp {
///       yield something()
///    } else {
///       yield something2()
///    }
///    print(%arg1)
///
struct ReplaceIfYieldWithConditionOrValue : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there are no results that could be replaced.
    if (op.getNumResults() == 0)
      return failure();

    auto trueYield =
        cast<scf::YieldOp>(op.getThenRegion().back().getTerminator());
    auto falseYield =
        cast<scf::YieldOp>(op.getElseRegion().back().getTerminator());

    rewriter.setInsertionPoint(op->getBlock(),
                               op.getOperation()->getIterator());
    bool changed = false;
    Type i1Ty = rewriter.getI1Type();
    for (auto [trueResult, falseResult, opResult] :
         llvm::zip(trueYield.getResults(), falseYield.getResults(),
                   op.getResults())) {
      if (trueResult == falseResult) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(trueResult);
          changed = true;
        }
        continue;
      }

      BoolAttr trueYield, falseYield;
      if (!matchPattern(trueResult, m_Constant(&trueYield)) ||
          !matchPattern(falseResult, m_Constant(&falseYield)))
        continue;

      bool trueVal = trueYield.getValue();
      bool falseVal = falseYield.getValue();
      if (!trueVal && falseVal) {
        if (!opResult.use_empty()) {
          Dialect *constDialect = trueResult.getDefiningOp()->getDialect();
          Value notCond = arith::XOrIOp::create(
              rewriter, op.getLoc(), op.getCondition(),
              constDialect
                  ->materializeConstant(rewriter,
                                        rewriter.getIntegerAttr(i1Ty, 1), i1Ty,
                                        op.getLoc())
                  ->getResult(0));
          opResult.replaceAllUsesWith(notCond);
          changed = true;
        }
      }
      if (trueVal && !falseVal) {
        if (!opResult.use_empty()) {
          opResult.replaceAllUsesWith(op.getCondition());
          changed = true;
        }
      }
    }
    return success(changed);
  }
};

/// Merge any consecutive scf.if's with the same condition.
///
///    scf.if %cond {
///       firstCodeTrue();...
///    } else {
///       firstCodeFalse();...
///    }
///    %res = scf.if %cond {
///       secondCodeTrue();...
///    } else {
///       secondCodeFalse();...
///    }
///
///  becomes
///    %res = scf.if %cmp {
///       firstCodeTrue();...
///       secondCodeTrue();...
///    } else {
///       firstCodeFalse();...
///       secondCodeFalse();...
///    }
struct CombineIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp nextIf,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<IfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    // Determine the logical then/else blocks when prevIf's
    // condition is used. Null means the block does not exist
    // in that case (e.g. empty else). If neither of these
    // are set, the two conditions cannot be compared.
    Block *nextThen = nullptr;
    Block *nextElse = nullptr;
    if (nextIf.getCondition() == prevIf.getCondition()) {
      nextThen = nextIf.thenBlock();
      if (!nextIf.getElseRegion().empty())
        nextElse = nextIf.elseBlock();
    }
    if (arith::XOrIOp notv =
            nextIf.getCondition().getDefiningOp<arith::XOrIOp>()) {
      if (notv.getLhs() == prevIf.getCondition() &&
          matchPattern(notv.getRhs(), m_One())) {
        nextElse = nextIf.thenBlock();
        if (!nextIf.getElseRegion().empty())
          nextThen = nextIf.elseBlock();
      }
    }
    if (arith::XOrIOp notv =
            prevIf.getCondition().getDefiningOp<arith::XOrIOp>()) {
      if (notv.getLhs() == nextIf.getCondition() &&
          matchPattern(notv.getRhs(), m_One())) {
        nextElse = nextIf.thenBlock();
        if (!nextIf.getElseRegion().empty())
          nextThen = nextIf.elseBlock();
      }
    }

    if (!nextThen && !nextElse)
      return failure();

    SmallVector<Value> prevElseYielded;
    if (!prevIf.getElseRegion().empty())
      prevElseYielded = prevIf.elseYield().getOperands();
    // Replace all uses of return values of op within nextIf with the
    // corresponding yields
    for (auto it : llvm::zip(prevIf.getResults(),
                             prevIf.thenYield().getOperands(), prevElseYielded))
      for (OpOperand &use :
           llvm::make_early_inc_range(std::get<0>(it).getUses())) {
        if (nextThen && nextThen->getParent()->isAncestor(
                            use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeOpModification(use.getOwner());
        } else if (nextElse && nextElse->getParent()->isAncestor(
                                   use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<2>(it));
          rewriter.finalizeOpModification(use.getOwner());
        }
      }

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    IfOp combinedIf = IfOp::create(rewriter, nextIf.getLoc(), mergedTypes,
                                   prevIf.getCondition(), /*hasElse=*/false);
    rewriter.eraseBlock(&combinedIf.getThenRegion().back());

    rewriter.inlineRegionBefore(prevIf.getThenRegion(),
                                combinedIf.getThenRegion(),
                                combinedIf.getThenRegion().begin());

    if (nextThen) {
      YieldOp thenYield = combinedIf.thenYield();
      YieldOp thenYield2 = cast<YieldOp>(nextThen->getTerminator());
      rewriter.mergeBlocks(nextThen, combinedIf.thenBlock());
      rewriter.setInsertionPointToEnd(combinedIf.thenBlock());

      SmallVector<Value> mergedYields(thenYield.getOperands());
      llvm::append_range(mergedYields, thenYield2.getOperands());
      YieldOp::create(rewriter, thenYield2.getLoc(), mergedYields);
      rewriter.eraseOp(thenYield);
      rewriter.eraseOp(thenYield2);
    }

    rewriter.inlineRegionBefore(prevIf.getElseRegion(),
                                combinedIf.getElseRegion(),
                                combinedIf.getElseRegion().begin());

    if (nextElse) {
      if (combinedIf.getElseRegion().empty()) {
        rewriter.inlineRegionBefore(*nextElse->getParent(),
                                    combinedIf.getElseRegion(),
                                    combinedIf.getElseRegion().begin());
      } else {
        YieldOp elseYield = combinedIf.elseYield();
        YieldOp elseYield2 = cast<YieldOp>(nextElse->getTerminator());
        rewriter.mergeBlocks(nextElse, combinedIf.elseBlock());

        rewriter.setInsertionPointToEnd(combinedIf.elseBlock());

        SmallVector<Value> mergedElseYields(elseYield.getOperands());
        llvm::append_range(mergedElseYields, elseYield2.getOperands());

        YieldOp::create(rewriter, elseYield2.getLoc(), mergedElseYields);
        rewriter.eraseOp(elseYield);
        rewriter.eraseOp(elseYield2);
      }
    }

    SmallVector<Value> prevValues;
    SmallVector<Value> nextValues;
    for (const auto &pair : llvm::enumerate(combinedIf.getResults())) {
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    }
    rewriter.replaceOp(prevIf, prevValues);
    rewriter.replaceOp(nextIf, nextValues);
    return success();
  }
};

/// Pattern to remove an empty else branch.
struct RemoveEmptyElseBranch : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Cannot remove else region when there are operation results.
    if (ifOp.getNumResults())
      return failure();
    Block *elseBlock = ifOp.elseBlock();
    if (!elseBlock || !llvm::hasSingleElement(*elseBlock))
      return failure();
    auto newIfOp = rewriter.cloneWithoutRegions(ifOp);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.eraseOp(ifOp);
    return success();
  }
};

/// Convert nested `if`s into `arith.andi` + single `if`.
///
///    scf.if %arg0 {
///      scf.if %arg1 {
///        ...
///        scf.yield
///      }
///      scf.yield
///    }
///  becomes
///
///    %0 = arith.andi %arg0, %arg1
///    scf.if %0 {
///      ...
///      scf.yield
///    }
struct CombineNestedIfs : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    auto nestedOps = op.thenBlock()->without_terminator();
    // Nested `if` must be the only op in block.
    if (!llvm::hasSingleElement(nestedOps))
      return failure();

    // If there is an else block, it can only yield
    if (op.elseBlock() && !llvm::hasSingleElement(*op.elseBlock()))
      return failure();

    auto nestedIf = dyn_cast<IfOp>(*nestedOps.begin());
    if (!nestedIf)
      return failure();

    if (nestedIf.elseBlock() && !llvm::hasSingleElement(*nestedIf.elseBlock()))
      return failure();

    SmallVector<Value> thenYield(op.thenYield().getOperands());
    SmallVector<Value> elseYield;
    if (op.elseBlock())
      llvm::append_range(elseYield, op.elseYield().getOperands());

    // A list of indices for which we should upgrade the value yielded
    // in the else to a select.
    SmallVector<unsigned> elseYieldsToUpgradeToSelect;

    // If the outer scf.if yields a value produced by the inner scf.if,
    // only permit combining if the value yielded when the condition
    // is false in the outer scf.if is the same value yielded when the
    // inner scf.if condition is false.
    // Note that the array access to elseYield will not go out of bounds
    // since it must have the same length as thenYield, since they both
    // come from the same scf.if.
    for (const auto &tup : llvm::enumerate(thenYield)) {
      if (tup.value().getDefiningOp() == nestedIf) {
        auto nestedIdx = llvm::cast<OpResult>(tup.value()).getResultNumber();
        if (nestedIf.elseYield().getOperand(nestedIdx) !=
            elseYield[tup.index()]) {
          return failure();
        }
        // If the correctness test passes, we will yield
        // corresponding value from the inner scf.if
        thenYield[tup.index()] = nestedIf.thenYield().getOperand(nestedIdx);
        continue;
      }

      // Otherwise, we need to ensure the else block of the combined
      // condition still returns the same value when the outer condition is
      // true and the inner condition is false. This can be accomplished if
      // the then value is defined outside the outer scf.if and we replace the
      // value with a select that considers just the outer condition. Since
      // the else region contains just the yield, its yielded value is
      // defined outside the scf.if, by definition.

      // If the then value is defined within the scf.if, bail.
      if (tup.value().getParentRegion() == &op.getThenRegion()) {
        return failure();
      }
      elseYieldsToUpgradeToSelect.push_back(tup.index());
    }

    Location loc = op.getLoc();
    Value newCondition = arith::AndIOp::create(rewriter, loc, op.getCondition(),
                                               nestedIf.getCondition());
    auto newIf = IfOp::create(rewriter, loc, op.getResultTypes(), newCondition);
    Block *newIfBlock = rewriter.createBlock(&newIf.getThenRegion());

    SmallVector<Value> results;
    llvm::append_range(results, newIf.getResults());
    rewriter.setInsertionPoint(newIf);

    for (auto idx : elseYieldsToUpgradeToSelect)
      results[idx] =
          arith::SelectOp::create(rewriter, op.getLoc(), op.getCondition(),
                                  thenYield[idx], elseYield[idx]);

    rewriter.mergeBlocks(nestedIf.thenBlock(), newIfBlock);
    rewriter.setInsertionPointToEnd(newIf.thenBlock());
    rewriter.replaceOpWithNewOp<YieldOp>(newIf.thenYield(), thenYield);
    if (!elseYield.empty()) {
      rewriter.createBlock(&newIf.getElseRegion());
      rewriter.setInsertionPointToEnd(newIf.elseBlock());
      YieldOp::create(rewriter, loc, elseYield);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void IfOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                       MLIRContext *context) {
  results.add<CombineIfs, CombineNestedIfs, ConditionPropagation,
              ConvertTrivialIfToSelect, RemoveEmptyElseBranch,
              ReplaceIfYieldWithConditionOrValue>(context);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, IfOp::getOperationName());
  populateRegionBranchOpInterfaceInliningPattern(results,
                                                 IfOp::getOperationName());
}

Block *IfOp::thenBlock() { return &getThenRegion().back(); }
YieldOp IfOp::thenYield() { return cast<YieldOp>(&thenBlock()->back()); }
Block *IfOp::elseBlock() {
  Region &r = getElseRegion();
  if (r.empty())
    return nullptr;
  return &r.back();
}
YieldOp IfOp::elseYield() { return cast<YieldOp>(&elseBlock()->back()); }

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ValueRange lowerBounds,
    ValueRange upperBounds, ValueRange steps, ValueRange initVals,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  result.addOperands(lowerBounds);
  result.addOperands(upperBounds);
  result.addOperands(steps);
  result.addOperands(initVals);
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lowerBounds.size()),
                                    static_cast<int32_t>(upperBounds.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(initVals.size())}));
  result.addTypes(initVals.getTypes());

  OpBuilder::InsertionGuard guard(builder);
  unsigned numIVs = steps.size();
  SmallVector<Type, 8> argTypes(numIVs, builder.getIndexType());
  SmallVector<Location, 8> argLocs(numIVs, result.location);
  Region *bodyRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion, {}, argTypes, argLocs);

  if (bodyBuilderFn) {
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilderFn(builder, result.location,
                  bodyBlock->getArguments().take_front(numIVs),
                  bodyBlock->getArguments().drop_front(numIVs));
  }
  // Add terminator only if there are no reductions.
  if (initVals.empty())
    ParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void ParallelOp::build(
    OpBuilder &builder, OperationState &result, ValueRange lowerBounds,
    ValueRange upperBounds, ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
  // Only pass a non-null wrapper if bodyBuilderFn is non-null itself. Make sure
  // we don't capture a reference to a temporary by constructing the lambda at
  // function level.
  auto wrappedBuilderFn = [&bodyBuilderFn](OpBuilder &nestedBuilder,
                                           Location nestedLoc, ValueRange ivs,
                                           ValueRange) {
    bodyBuilderFn(nestedBuilder, nestedLoc, ivs);
  };
  function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> wrapper;
  if (bodyBuilderFn)
    wrapper = wrappedBuilderFn;

  build(builder, result, lowerBounds, upperBounds, steps, ValueRange(),
        wrapper);
}

LogicalResult ParallelOp::verify() {
  // Check that there is at least one value in lowerBound, upperBound and step.
  // It is sufficient to test only step, because it is ensured already that the
  // number of elements in lowerBound, upperBound and step are the same.
  Operation::operand_range stepValues = getStep();
  if (stepValues.empty())
    return emitOpError(
        "needs at least one tuple element for lowerBound, upperBound and step");

  // Check whether all constant step values are positive.
  for (Value stepValue : stepValues)
    if (auto cst = getConstantIntValue(stepValue))
      if (*cst <= 0)
        return emitOpError("constant step operand must be positive");

  // Check that the body defines the same number of block arguments as the
  // number of tuple elements in step.
  Block *body = getBody();
  if (body->getNumArguments() != stepValues.size())
    return emitOpError() << "expects the same number of induction variables: "
                         << body->getNumArguments()
                         << " as bound and step values: " << stepValues.size();
  for (auto arg : body->getArguments())
    if (!arg.getType().isIndex())
      return emitOpError(
          "expects arguments for the induction variable to be of index type");

  // Check that the terminator is an scf.reduce op.
  auto reduceOp = verifyAndGetTerminator<scf::ReduceOp>(
      *this, getRegion(), "expects body to terminate with 'scf.reduce'");
  if (!reduceOp)
    return failure();

  // Check that the number of results is the same as the number of reductions.
  auto resultsSize = getResults().size();
  auto reductionsSize = reduceOp.getReductions().size();
  auto initValsSize = getInitVals().size();
  if (resultsSize != reductionsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of reductions: "
                         << reductionsSize;
  if (resultsSize != initValsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of initial values: "
                         << initValsSize;
  if (reduceOp.getNumOperands() != initValsSize)
    // Delegate error reporting to ReduceOp
    return success();

  // Check that the types of the results and reductions are the same.
  for (int64_t i = 0; i < static_cast<int64_t>(reductionsSize); ++i) {
    auto resultType = getOperation()->getResult(i).getType();
    auto reductionOperandType = reduceOp.getOperands()[i].getType();
    if (resultType != reductionOperandType)
      return reduceOp.emitOpError()
             << "expects type of " << i
             << "-th reduction operand: " << reductionOperandType
             << " to be the same as the " << i
             << "-th result type: " << resultType;
  }
  return success();
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by induction variables followed by `)`
  SmallVector<OpAsmParser::Argument, 4> ivs;
  if (parser.parseArgumentList(ivs, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse loop bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> lower;
  if (parser.parseEqual() ||
      parser.parseOperandList(lower, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(lower, builder.getIndexType(), result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> upper;
  if (parser.parseKeyword("to") ||
      parser.parseOperandList(upper, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(upper, builder.getIndexType(), result.operands))
    return failure();

  // Parse step values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> steps;
  if (parser.parseKeyword("step") ||
      parser.parseOperandList(steps, ivs.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(steps, builder.getIndexType(), result.operands))
    return failure();

  // Parse init values.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> initVals;
  if (succeeded(parser.parseOptionalKeyword("init"))) {
    if (parser.parseOperandList(initVals, OpAsmParser::Delimiter::Paren))
      return failure();
  }

  // Parse optional results in case there is a reduce.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Now parse the body.
  Region *body = result.addRegion();
  for (auto &iv : ivs)
    iv.type = builder.getIndexType();
  if (parser.parseRegion(*body, ivs))
    return failure();

  // Set `operandSegmentSizes` attribute.
  result.addAttribute(
      ParallelOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(lower.size()),
                                    static_cast<int32_t>(upper.size()),
                                    static_cast<int32_t>(steps.size()),
                                    static_cast<int32_t>(initVals.size())}));

  // Parse attributes.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(initVals, result.types, parser.getNameLoc(),
                             result.operands))
    return failure();

  // Add a terminator if none was parsed.
  ParallelOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  p << " (" << getBody()->getArguments() << ") = (" << getLowerBound()
    << ") to (" << getUpperBound() << ") step (" << getStep() << ")";
  if (!getInitVals().empty())
    p << " init (" << getInitVals() << ")";
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/ParallelOp::getOperandSegmentSizeAttr());
}

SmallVector<Region *> ParallelOp::getLoopRegions() { return {&getRegion()}; }

std::optional<SmallVector<Value>> ParallelOp::getLoopInductionVars() {
  return SmallVector<Value>{getBody()->getArguments()};
}

std::optional<SmallVector<OpFoldResult>> ParallelOp::getLoopLowerBounds() {
  return getLowerBound();
}

std::optional<SmallVector<OpFoldResult>> ParallelOp::getLoopUpperBounds() {
  return getUpperBound();
}

std::optional<SmallVector<OpFoldResult>> ParallelOp::getLoopSteps() {
  return getStep();
}

ParallelOp mlir::scf::getParallelForInductionVarOwner(Value val) {
  auto ivArg = llvm::dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return ParallelOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<ParallelOp>(containingOp);
}

namespace {
// Collapse loop dimensions that perform a single iteration.
struct ParallelOpSingleOrZeroIterationDimsFolder
    : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<Value> newLowerBounds, newUpperBounds, newSteps;
    IRMapping mapping;
    for (auto [lb, ub, step, iv] :
         llvm::zip(op.getLowerBound(), op.getUpperBound(), op.getStep(),
                   op.getInductionVars())) {
      auto numIterations =
          constantTripCount(lb, ub, step, /*isSigned=*/true, computeUbMinusLb);
      if (numIterations.has_value()) {
        // Remove the loop if it performs zero iterations.
        if (*numIterations == 0) {
          rewriter.replaceOp(op, op.getInitVals());
          return success();
        }
        // Replace the loop induction variable by the lower bound if the loop
        // performs a single iteration. Otherwise, copy the loop bounds.
        if (*numIterations == 1) {
          mapping.map(iv, getValueOrCreateConstantIndexOp(rewriter, loc, lb));
          continue;
        }
      }
      newLowerBounds.push_back(lb);
      newUpperBounds.push_back(ub);
      newSteps.push_back(step);
    }
    // Exit if none of the loop dimensions perform a single iteration.
    if (newLowerBounds.size() == op.getLowerBound().size())
      return failure();

    if (newLowerBounds.empty()) {
      // All of the loop dimensions perform a single iteration. Inline
      // loop body and nested ReduceOp's
      SmallVector<Value> results;
      results.reserve(op.getInitVals().size());
      for (auto &bodyOp : op.getBody()->without_terminator())
        rewriter.clone(bodyOp, mapping);
      auto reduceOp = cast<ReduceOp>(op.getBody()->getTerminator());
      for (int64_t i = 0, e = reduceOp.getReductions().size(); i < e; ++i) {
        Block &reduceBlock = reduceOp.getReductions()[i].front();
        auto initValIndex = results.size();
        mapping.map(reduceBlock.getArgument(0), op.getInitVals()[initValIndex]);
        mapping.map(reduceBlock.getArgument(1),
                    mapping.lookupOrDefault(reduceOp.getOperands()[i]));
        for (auto &reduceBodyOp : reduceBlock.without_terminator())
          rewriter.clone(reduceBodyOp, mapping);

        auto result = mapping.lookupOrDefault(
            cast<ReduceReturnOp>(reduceBlock.getTerminator()).getResult());
        results.push_back(result);
      }

      rewriter.replaceOp(op, results);
      return success();
    }
    // Replace the parallel loop by lower-dimensional parallel loop.
    auto newOp =
        ParallelOp::create(rewriter, op.getLoc(), newLowerBounds,
                           newUpperBounds, newSteps, op.getInitVals(), nullptr);
    // Erase the empty block that was inserted by the builder.
    rewriter.eraseBlock(newOp.getBody());
    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct MergeNestedParallelLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = *op.getBody();
    if (!llvm::hasSingleElement(outerBody.without_terminator()))
      return failure();

    auto innerOp = dyn_cast<ParallelOp>(outerBody.front());
    if (!innerOp)
      return failure();

    for (auto val : outerBody.getArguments())
      if (llvm::is_contained(innerOp.getLowerBound(), val) ||
          llvm::is_contained(innerOp.getUpperBound(), val) ||
          llvm::is_contained(innerOp.getStep(), val))
        return failure();

    // Reductions are not supported yet.
    if (!op.getInitVals().empty() || !innerOp.getInitVals().empty())
      return failure();

    auto bodyBuilder = [&](OpBuilder &builder, Location /*loc*/,
                           ValueRange iterVals, ValueRange) {
      Block &innerBody = *innerOp.getBody();
      assert(iterVals.size() ==
             (outerBody.getNumArguments() + innerBody.getNumArguments()));
      IRMapping mapping;
      mapping.map(outerBody.getArguments(),
                  iterVals.take_front(outerBody.getNumArguments()));
      mapping.map(innerBody.getArguments(),
                  iterVals.take_back(innerBody.getNumArguments()));
      for (Operation &op : innerBody.without_terminator())
        builder.clone(op, mapping);
    };

    auto concatValues = [](const auto &first, const auto &second) {
      SmallVector<Value> ret;
      ret.reserve(first.size() + second.size());
      ret.assign(first.begin(), first.end());
      ret.append(second.begin(), second.end());
      return ret;
    };

    auto newLowerBounds =
        concatValues(op.getLowerBound(), innerOp.getLowerBound());
    auto newUpperBounds =
        concatValues(op.getUpperBound(), innerOp.getUpperBound());
    auto newSteps = concatValues(op.getStep(), innerOp.getStep());

    rewriter.replaceOpWithNewOp<ParallelOp>(op, newLowerBounds, newUpperBounds,
                                            newSteps, ValueRange(),
                                            bodyBuilder);
    return success();
  }
};

} // namespace

void ParallelOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results
      .add<ParallelOpSingleOrZeroIterationDimsFolder, MergeNestedParallelLoops>(
          context);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ParallelOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // Both the operation itself and the region may be branching into the body or
  // back into the operation itself. It is possible for loop not to enter the
  // body.
  regions.push_back(RegionSuccessor(&getRegion()));
  regions.push_back(RegionSuccessor::parent());
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::build(OpBuilder &builder, OperationState &result) {}

void ReduceOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange operands) {
  result.addOperands(operands);
  for (Value v : operands) {
    OpBuilder::InsertionGuard guard(builder);
    Region *bodyRegion = result.addRegion();
    builder.createBlock(bodyRegion, {},
                        ArrayRef<Type>{v.getType(), v.getType()},
                        {result.location, result.location});
  }
}

LogicalResult ReduceOp::verifyRegions() {
  if (getReductions().size() != getOperands().size())
    return emitOpError() << "expects number of reduction regions: "
                         << getReductions().size()
                         << " to be the same as number of reduction operands: "
                         << getOperands().size();
  // The region of a ReduceOp has two arguments of the same type as its
  // corresponding operand.
  for (int64_t i = 0, e = getReductions().size(); i < e; ++i) {
    auto type = getOperands()[i].getType();
    Block &block = getReductions()[i].front();
    if (block.empty())
      return emitOpError() << i << "-th reduction has an empty body";
    if (block.getNumArguments() != 2 ||
        llvm::any_of(block.getArguments(), [&](const BlockArgument &arg) {
          return arg.getType() != type;
        }))
      return emitOpError() << "expected two block arguments with type " << type
                           << " in the " << i << "-th reduction region";

    // Check that the block is terminated by a ReduceReturnOp.
    if (!isa<ReduceReturnOp>(block.getTerminator()))
      return emitOpError("reduction bodies must be terminated with an "
                         "'scf.reduce.return' op");
  }

  return success();
}

MutableOperandRange
ReduceOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // No operands are forwarded to the next iteration.
  return MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0);
}

//===----------------------------------------------------------------------===//
// ReduceReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceReturnOp::verify() {
  // The type of the return value should be the same type as the types of the
  // block arguments of the reduction body.
  Block *reductionBody = getOperation()->getBlock();
  // Should already be verified by an op trait.
  assert(isa<ReduceOp>(reductionBody->getParentOp()) && "expected scf.reduce");
  Type expectedResultType = reductionBody->getArgument(0).getType();
  if (expectedResultType != getResult().getType())
    return emitOpError() << "must have type " << expectedResultType
                         << " (the type of the reduction inputs)";
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void WhileOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, TypeRange resultTypes,
                    ValueRange inits, BodyBuilderFn beforeBuilder,
                    BodyBuilderFn afterBuilder) {
  odsState.addOperands(inits);
  odsState.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(odsBuilder);

  // Build before region.
  SmallVector<Location, 4> beforeArgLocs;
  beforeArgLocs.reserve(inits.size());
  for (Value operand : inits) {
    beforeArgLocs.push_back(operand.getLoc());
  }

  Region *beforeRegion = odsState.addRegion();
  Block *beforeBlock = odsBuilder.createBlock(beforeRegion, /*insertPt=*/{},
                                              inits.getTypes(), beforeArgLocs);
  if (beforeBuilder)
    beforeBuilder(odsBuilder, odsState.location, beforeBlock->getArguments());

  // Build after region.
  SmallVector<Location, 4> afterArgLocs(resultTypes.size(), odsState.location);

  Region *afterRegion = odsState.addRegion();
  Block *afterBlock = odsBuilder.createBlock(afterRegion, /*insertPt=*/{},
                                             resultTypes, afterArgLocs);

  if (afterBuilder)
    afterBuilder(odsBuilder, odsState.location, afterBlock->getArguments());
}

ConditionOp WhileOp::getConditionOp() {
  return cast<ConditionOp>(getBeforeBody()->getTerminator());
}

YieldOp WhileOp::getYieldOp() {
  return cast<YieldOp>(getAfterBody()->getTerminator());
}

std::optional<MutableArrayRef<OpOperand>> WhileOp::getYieldedValuesMutable() {
  return getYieldOp().getResultsMutable();
}

Block::BlockArgListType WhileOp::getBeforeArguments() {
  return getBeforeBody()->getArguments();
}

Block::BlockArgListType WhileOp::getAfterArguments() {
  return getAfterBody()->getArguments();
}

Block::BlockArgListType WhileOp::getRegionIterArgs() {
  return getBeforeArguments();
}

OperandRange WhileOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  assert(successor.getSuccessor() == &getBefore() &&
         "WhileOp is expected to branch only to the first region");
  return getInits();
}

void WhileOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getBefore());
    return;
  }

  assert(llvm::is_contained(
             {&getAfter(), &getBefore()},
             point.getTerminatorPredecessorOrNull()->getParentRegion()) &&
         "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (point.getTerminatorPredecessorOrNull()->getParentRegion() ==
      &getAfter()) {
    regions.emplace_back(&getBefore());
    return;
  }

  regions.push_back(RegionSuccessor::parent());
  regions.emplace_back(&getAfter());
}

ValueRange WhileOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.isParent())
    return getOperation()->getResults();
  if (successor == &getBefore())
    return getBefore().getArguments();
  if (successor == &getAfter())
    return getAfter().getArguments();
  llvm_unreachable("invalid region successor");
}

SmallVector<Region *> WhileOp::getLoopRegions() {
  return {&getBefore(), &getAfter()};
}

/// Parses a `while` op.
///
/// op ::= `scf.while` assignments `:` function-type region `do` region
///         `attributes` attribute-dict
/// initializer ::= /* empty */ | `(` assignment-list `)`
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
ParseResult scf::WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Region *before = result.addRegion();
  Region *after = result.addRegion();

  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  FunctionType functionType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  if (functionType.getNumInputs() != operands.size()) {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands " << "(expected "
           << operands.size() << " got " << functionType.getNumInputs() << ")";
  }

  // Resolve input operands.
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  // Propagate the types into the region arguments.
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = functionType.getInput(i);

  return failure(parser.parseRegion(*before, regionArgs) ||
                 parser.parseKeyword("do") || parser.parseRegion(*after) ||
                 parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

/// Prints a `while` op.
void scf::WhileOp::print(OpAsmPrinter &p) {
  printInitializationList(p, getBeforeArguments(), getInits(), " ");
  p << " : ";
  p.printFunctionalType(getInits().getTypes(), getResults().getTypes());
  p << ' ';
  p.printRegion(getBefore(), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getAfter());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
}

/// Verifies that two ranges of types match, i.e. have the same number of
/// entries and that types are pairwise equals. Reports errors on the given
/// operation in case of mismatch.
template <typename OpTy>
static LogicalResult verifyTypeRangesMatch(OpTy op, TypeRange left,
                                           TypeRange right, StringRef message) {
  if (left.size() != right.size())
    return op.emitOpError("expects the same number of ") << message;

  for (unsigned i = 0, e = left.size(); i < e; ++i) {
    if (left[i] != right[i]) {
      InFlightDiagnostic diag = op.emitOpError("expects the same types for ")
                                << message;
      diag.attachNote() << "for argument " << i << ", found " << left[i]
                        << " and " << right[i];
      return diag;
    }
  }

  return success();
}

LogicalResult scf::WhileOp::verify() {
  auto beforeTerminator = verifyAndGetTerminator<scf::ConditionOp>(
      *this, getBefore(),
      "expects the 'before' region to terminate with 'scf.condition'");
  if (!beforeTerminator)
    return failure();

  auto afterTerminator = verifyAndGetTerminator<scf::YieldOp>(
      *this, getAfter(),
      "expects the 'after' region to terminate with 'scf.yield'");
  return success(afterTerminator != nullptr);
}

namespace {
/// Move a scf.if op that is directly before the scf.condition op in the while
/// before region, and whose condition matches the condition of the
/// scf.condition op, down into the while after region.
///
/// scf.while (..) : (...) -> ... {
///  %additional_used_values = ...
///  %cond = ...
///  ...
///  %res = scf.if %cond -> (...) {
///    use(%additional_used_values)
///    ... // then block
///    scf.yield %then_value
///  } else {
///    scf.yield %else_value
///  }
///  scf.condition(%cond) %res, ...
/// } do {
/// ^bb0(%res_arg, ...):
///    use(%res_arg)
///    ...
///
/// becomes
/// scf.while (..) : (...) -> ... {
///  %additional_used_values = ...
///  %cond = ...
///  ...
///  scf.condition(%cond) %else_value, ..., %additional_used_values
/// } do {
/// ^bb0(%res_arg ..., %additional_args): :
///    use(%additional_args)
///    ... // if then block
///    use(%then_value)
///    ...
struct WhileMoveIfDown : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto conditionOp = op.getConditionOp();

    // Only support ifOp right before the condition at the moment. Relaxing this
    // would require to:
    // - check that the body does not have side-effects conflicting with
    //    operations between the if and the condition.
    // - check that results of the if operation are only used as arguments to
    //    the condition.
    auto ifOp = dyn_cast_or_null<scf::IfOp>(conditionOp->getPrevNode());

    // Check that the ifOp is directly before the conditionOp and that it
    // matches the condition of the conditionOp. Also ensure that the ifOp has
    // no else block with content, as that would complicate the transformation.
    // TODO: support else blocks with content.
    if (!ifOp || ifOp.getCondition() != conditionOp.getCondition() ||
        (ifOp.elseBlock() && !ifOp.elseBlock()->without_terminator().empty()))
      return failure();

    assert((ifOp->use_empty() || (llvm::all_equal(ifOp->getUsers()) &&
                                  *ifOp->user_begin() == conditionOp)) &&
           "ifOp has unexpected uses");

    Location loc = op.getLoc();

    // Replace uses of ifOp results in the conditionOp with the yielded values
    // from the ifOp branches.
    for (auto [idx, arg] : llvm::enumerate(conditionOp.getArgs())) {
      auto it = llvm::find(ifOp->getResults(), arg);
      if (it != ifOp->getResults().end()) {
        size_t ifOpIdx = it.getIndex();
        Value thenValue = ifOp.thenYield()->getOperand(ifOpIdx);
        Value elseValue = ifOp.elseYield()->getOperand(ifOpIdx);

        rewriter.replaceAllUsesWith(ifOp->getResults()[ifOpIdx], elseValue);
        rewriter.replaceAllUsesWith(op.getAfterArguments()[idx], thenValue);
      }
    }

    // Collect additional used values from before region.
    SetVector<Value> additionalUsedValuesSet;
    visitUsedValuesDefinedAbove(ifOp.getThenRegion(), [&](OpOperand *operand) {
      if (&op.getBefore() == operand->get().getParentRegion())
        additionalUsedValuesSet.insert(operand->get());
    });

    // Create new whileOp with additional used values as results.
    auto additionalUsedValues = additionalUsedValuesSet.getArrayRef();
    auto additionalValueTypes = llvm::map_to_vector(
        additionalUsedValues, [](Value val) { return val.getType(); });
    size_t additionalValueSize = additionalUsedValues.size();
    SmallVector<Type> newResultTypes(op.getResultTypes());
    newResultTypes.append(additionalValueTypes);

    auto newWhileOp =
        scf::WhileOp::create(rewriter, loc, newResultTypes, op.getInits());

    rewriter.modifyOpInPlace(newWhileOp, [&] {
      newWhileOp.getBefore().takeBody(op.getBefore());
      newWhileOp.getAfter().takeBody(op.getAfter());
      newWhileOp.getAfter().addArguments(
          additionalValueTypes,
          SmallVector<Location>(additionalValueSize, loc));
    });

    rewriter.modifyOpInPlace(conditionOp, [&] {
      conditionOp.getArgsMutable().append(additionalUsedValues);
    });

    // Replace uses of additional used values inside the ifOp then region with
    // the whileOp after region arguments.
    rewriter.replaceUsesWithIf(
        additionalUsedValues,
        newWhileOp.getAfterArguments().take_back(additionalValueSize),
        [&](OpOperand &use) {
          return ifOp.getThenRegion().isAncestor(
              use.getOwner()->getParentRegion());
        });

    // Inline ifOp then region into new whileOp after region.
    rewriter.eraseOp(ifOp.thenYield());
    rewriter.inlineBlockBefore(ifOp.thenBlock(), newWhileOp.getAfterBody(),
                               newWhileOp.getAfterBody()->begin());
    rewriter.eraseOp(ifOp);
    rewriter.replaceOp(op,
                       newWhileOp->getResults().drop_back(additionalValueSize));
    return success();
  }
};

/// Replace uses of the condition within the do block with true, since otherwise
/// the block would not be evaluated.
///
/// scf.while (..) : (i1, ...) -> ... {
///  %condition = call @evaluate_condition() : () -> i1
///  scf.condition(%condition) %condition : i1, ...
/// } do {
/// ^bb0(%arg0: i1, ...):
///    use(%arg0)
///    ...
///
/// becomes
/// scf.while (..) : (i1, ...) -> ... {
///  %condition = call @evaluate_condition() : () -> i1
///  scf.condition(%condition) %condition : i1, ...
/// } do {
/// ^bb0(%arg0: i1, ...):
///    use(%true)
///    ...
struct WhileConditionTruth : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = op.getConditionOp();

    // These variables serve to prevent creating duplicate constants
    // and hold constant true or false values.
    Value constantTrue = nullptr;

    bool replaced = false;
    for (auto yieldedAndBlockArgs :
         llvm::zip(term.getArgs(), op.getAfterArguments())) {
      if (std::get<0>(yieldedAndBlockArgs) == term.getCondition()) {
        if (!std::get<1>(yieldedAndBlockArgs).use_empty()) {
          if (!constantTrue)
            constantTrue = arith::ConstantOp::create(
                rewriter, op.getLoc(), term.getCondition().getType(),
                rewriter.getBoolAttr(true));

          rewriter.replaceAllUsesWith(std::get<1>(yieldedAndBlockArgs),
                                      constantTrue);
          replaced = true;
        }
      }
    }
    return success(replaced);
  }
};

/// Replace operations equivalent to the condition in the do block with true,
/// since otherwise the block would not be evaluated.
///
/// scf.while (..) : (i32, ...) -> ... {
///  %z = ... : i32
///  %condition = cmpi pred %z, %a
///  scf.condition(%condition) %z : i32, ...
/// } do {
/// ^bb0(%arg0: i32, ...):
///    %condition2 = cmpi pred %arg0, %a
///    use(%condition2)
///    ...
///
/// becomes
/// scf.while (..) : (i32, ...) -> ... {
///  %z = ... : i32
///  %condition = cmpi pred %z, %a
///  scf.condition(%condition) %z : i32, ...
/// } do {
/// ^bb0(%arg0: i32, ...):
///    use(%true)
///    ...
struct WhileCmpCond : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    using namespace scf;
    auto cond = op.getConditionOp();
    auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    bool changed = false;
    for (auto tup : llvm::zip(cond.getArgs(), op.getAfterArguments())) {
      for (size_t opIdx = 0; opIdx < 2; opIdx++) {
        if (std::get<0>(tup) != cmp.getOperand(opIdx))
          continue;
        for (OpOperand &u :
             llvm::make_early_inc_range(std::get<1>(tup).getUses())) {
          auto cmp2 = dyn_cast<arith::CmpIOp>(u.getOwner());
          if (!cmp2)
            continue;
          // For a binary operator 1-opIdx gets the other side.
          if (cmp2.getOperand(1 - opIdx) != cmp.getOperand(1 - opIdx))
            continue;
          bool samePredicate;
          if (cmp2.getPredicate() == cmp.getPredicate())
            samePredicate = true;
          else if (cmp2.getPredicate() ==
                   arith::invertPredicate(cmp.getPredicate()))
            samePredicate = false;
          else
            continue;

          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(cmp2, samePredicate,
                                                            1);
          changed = true;
        }
      }
    }
    return success(changed);
  }
};

/// If both ranges contain same values return mappping indices from args2 to
/// args1. Otherwise return std::nullopt.
static std::optional<SmallVector<unsigned>> getArgsMapping(ValueRange args1,
                                                           ValueRange args2) {
  if (args1.size() != args2.size())
    return std::nullopt;

  SmallVector<unsigned> ret(args1.size());
  for (auto &&[i, arg1] : llvm::enumerate(args1)) {
    auto it = llvm::find(args2, arg1);
    if (it == args2.end())
      return std::nullopt;

    ret[std::distance(args2.begin(), it)] = static_cast<unsigned>(i);
  }

  return ret;
}

static bool hasDuplicates(ValueRange args) {
  llvm::SmallDenseSet<Value> set;
  for (Value arg : args) {
    if (!set.insert(arg).second)
      return true;
  }
  return false;
}

/// If `before` block args are directly forwarded to `scf.condition`, rearrange
/// `scf.condition` args into same order as block args. Update `after` block
/// args and op result values accordingly.
/// Needed to simplify `scf.while` -> `scf.for` uplifting.
struct WhileOpAlignBeforeArgs : public OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto *oldBefore = loop.getBeforeBody();
    ConditionOp oldTerm = loop.getConditionOp();
    ValueRange beforeArgs = oldBefore->getArguments();
    ValueRange termArgs = oldTerm.getArgs();
    if (beforeArgs == termArgs)
      return failure();

    if (hasDuplicates(termArgs))
      return failure();

    auto mapping = getArgsMapping(beforeArgs, termArgs);
    if (!mapping)
      return failure();

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(oldTerm);
      rewriter.replaceOpWithNewOp<ConditionOp>(oldTerm, oldTerm.getCondition(),
                                               beforeArgs);
    }

    auto *oldAfter = loop.getAfterBody();

    SmallVector<Type> newResultTypes(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping))
      newResultTypes[j] = loop.getResult(i).getType();

    auto newLoop = WhileOp::create(
        rewriter, loop.getLoc(), newResultTypes, loop.getInits(),
        /*beforeBuilder=*/nullptr, /*afterBuilder=*/nullptr);
    auto *newBefore = newLoop.getBeforeBody();
    auto *newAfter = newLoop.getAfterBody();

    SmallVector<Value> newResults(beforeArgs.size());
    SmallVector<Value> newAfterArgs(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping)) {
      newResults[i] = newLoop.getResult(j);
      newAfterArgs[i] = newAfter->getArgument(j);
    }

    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfterArgs);

    rewriter.replaceOp(loop, newResults);
    return success();
  }
};
} // namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<WhileConditionTruth, WhileCmpCond, WhileOpAlignBeforeArgs,
              WhileMoveIfDown>(context);
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, WhileOp::getOperationName());
  populateRegionBranchOpInterfaceInliningPattern(results,
                                                 WhileOp::getOperationName());
}

//===----------------------------------------------------------------------===//
// IndexSwitchOp
//===----------------------------------------------------------------------===//

/// Parse the case regions and values.
static ParseResult
parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (p.parseInteger(value) || p.parseRegion(region, /*arguments=*/{}))
      return failure();
    caseValues.push_back(value);
  }
  cases = p.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

/// Print the case regions and values.
static void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
}

LogicalResult scf::IndexSwitchOp::verify() {
  if (getCases().size() != getCaseRegions().size()) {
    return emitOpError("has ")
           << getCaseRegions().size() << " case regions but "
           << getCases().size() << " case values";
  }

  DenseSet<int64_t> valueSet;
  for (int64_t value : getCases())
    if (!valueSet.insert(value).second)
      return emitOpError("has duplicate case value: ") << value;
  auto verifyRegion = [&](Region &region, const Twine &name) -> LogicalResult {
    auto yield = dyn_cast<YieldOp>(region.front().back());
    if (!yield)
      return emitOpError("expected region to end with scf.yield, but got ")
             << region.front().back().getName();

    if (yield.getNumOperands() != getNumResults()) {
      return (emitOpError("expected each region to return ")
              << getNumResults() << " values, but " << name << " returns "
              << yield.getNumOperands())
                 .attachNote(yield.getLoc())
             << "see yield operation here";
    }
    for (auto [idx, result, operand] :
         llvm::enumerate(getResultTypes(), yield.getOperands())) {
      if (!operand)
        return yield.emitOpError() << "operand " << idx << " is null\n";
      if (result == operand.getType())
        continue;
      return (emitOpError("expected result #")
              << idx << " of each region to be " << result)
                 .attachNote(yield.getLoc())
             << name << " returns " << operand.getType() << " here";
    }
    return success();
  };

  if (failed(verifyRegion(getDefaultRegion(), "default region")))
    return failure();
  for (auto [idx, caseRegion] : llvm::enumerate(getCaseRegions()))
    if (failed(verifyRegion(caseRegion, "case region #" + Twine(idx))))
      return failure();

  return success();
}

unsigned scf::IndexSwitchOp::getNumCases() { return getCases().size(); }

Block &scf::IndexSwitchOp::getDefaultBlock() {
  return getDefaultRegion().front();
}

Block &scf::IndexSwitchOp::getCaseBlock(unsigned idx) {
  assert(idx < getNumCases() && "case index out-of-bounds");
  return getCaseRegions()[idx].front();
}

void IndexSwitchOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) {
  // All regions branch back to the parent op.
  if (!point.isParent()) {
    successors.push_back(RegionSuccessor::parent());
    return;
  }

  llvm::append_range(successors, getRegions());
}

ValueRange IndexSwitchOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getOperation()->getResults())
                              : ValueRange();
}

void IndexSwitchOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &successors) {
  FoldAdaptor adaptor(operands, *this);

  // If a constant was not provided, all regions are possible successors.
  auto arg = dyn_cast_or_null<IntegerAttr>(adaptor.getArg());
  if (!arg) {
    llvm::append_range(successors, getRegions());
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the
  // default region is the only successor.
  for (auto [caseValue, caseRegion] : llvm::zip(getCases(), getCaseRegions())) {
    if (caseValue == arg.getInt()) {
      successors.emplace_back(&caseRegion);
      return;
    }
  }
  successors.emplace_back(&getDefaultRegion());
}

void IndexSwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = llvm::dyn_cast_or_null<IntegerAttr>(operands.front());
  if (!operandValue) {
    // All regions are invoked at most once.
    bounds.append(getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
    return;
  }

  unsigned liveIndex = getNumRegions() - 1;
  const auto *it = llvm::find(getCases(), operandValue.getInt());
  if (it != getCases().end())
    liveIndex = std::distance(getCases().begin(), it);
  for (unsigned i = 0, e = getNumRegions(); i < e; ++i)
    bounds.emplace_back(/*lb=*/0, /*ub=*/i == liveIndex);
}

void IndexSwitchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      results, IndexSwitchOp::getOperationName());
  populateRegionBranchOpInterfaceInliningPattern(
      results, IndexSwitchOp::getOperationName());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/IR/SCFOps.cpp.inc"
