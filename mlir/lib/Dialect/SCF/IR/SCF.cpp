//===- SCF.cpp - Structured Control Flow Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

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
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  // Operations in scf dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
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
}

/// Default callback for IfOp builders. Inserts a yield without arguments.
void mlir::scf::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<scf::YieldOp>(loc);
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.mergeBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

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
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult ExecuteRegionOp::verify() {
  if (getRegion().empty())
    return emitOpError("region needs to have at least one block");
  if (getRegion().front().getNumArguments() > 0)
    return emitOpError("region cannot have any arguments");
  return success();
}

// Inline an ExecuteRegionOp if it only contains one block.
//     "test.foo"() : () -> ()
//      %v = scf.execute_region -> i64 {
//        %x = "test.val"() : () -> i64
//        scf.yield %x : i64
//      }
//      "test.bar"(%v) : (i64) -> ()
//
//  becomes
//
//     "test.foo"() : () -> ()
//     %x = "test.val"() : () -> i64
//     "test.bar"(%x) : (i64) -> ()
//
struct SingleBlockExecuteInliner : public OpRewritePattern<ExecuteRegionOp> {
  using OpRewritePattern<ExecuteRegionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::hasSingleElement(op.getRegion()))
      return failure();
    replaceOpWithRegion(rewriter, op, op.getRegion());
    return success();
  }
};

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
    if (!isa<FunctionOpInterface, ExecuteRegionOp>(op->getParentOp()))
      return failure();

    Block *prevBlock = op->getBlock();
    Block *postBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);

    rewriter.create<cf::BranchOp>(op.getLoc(), &op.getRegion().front());

    for (Block &blk : op.getRegion()) {
      if (YieldOp yieldOp = dyn_cast<YieldOp>(blk.getTerminator())) {
        rewriter.setInsertionPoint(yieldOp);
        rewriter.create<cf::BranchOp>(yieldOp.getLoc(), postBlock,
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
  results.add<SingleBlockExecuteInliner, MultiBlockExecuteInliner>(context);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ExecuteRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (!index) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

MutableOperandRange
ConditionOp::getMutableSuccessorOperands(Optional<unsigned> index) {
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

void ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                  Value ub, Value step, ValueRange iterArgs,
                  BodyBuilderFn bodyBuilder) {
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType(), result.location);
  for (Value v : iterArgs)
    bodyBlock.addArgument(v.getType(), v.getLoc());

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  if (iterArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                bodyBlock.getArguments().drop_front());
  }
}

LogicalResult ForOp::verify() {
  IntegerAttr step;
  if (matchPattern(getStep(), m_Constant(&step)) && step.getInt() <= 0)
    return emitOpError("constant step operand must be positive");

  auto opNumResults = getNumResults();
  if (opNumResults == 0)
    return success();
  // If ForOp defines values, check that the number and types of
  // the defined values match ForOp initial iter operands and backedge
  // basic block arguments.
  if (getNumIterOperands() != opNumResults)
    return emitOpError(
        "mismatch in number of loop-carried values and defined values");
  return success();
}

LogicalResult ForOp::verifyRegions() {
  // Check that the body defines as single block argument for the induction
  // variable.
  auto *body = getBody();
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the induction variable");

  auto opNumResults = getNumResults();
  if (opNumResults == 0)
    return success();

  if (getNumRegionIterArgs() != opNumResults)
    return emitOpError(
        "mismatch in number of basic block args and defined values");

  auto iterOperands = getIterOperands();
  auto iterArgs = getRegionIterArgs();
  auto opResults = getResults();
  unsigned i = 0;
  for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i
                           << "th iter region arg and defined value";

    i++;
  }
  return success();
}

Optional<Value> ForOp::getSingleInductionVar() { return getInductionVar(); }

Optional<OpFoldResult> ForOp::getSingleLowerBound() {
  return OpFoldResult(getLowerBound());
}

Optional<OpFoldResult> ForOp::getSingleStep() {
  return OpFoldResult(getStep());
}

Optional<OpFoldResult> ForOp::getSingleUpperBound() {
  return OpFoldResult(getUpperBound());
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
  p << " " << getInductionVar() << " = " << getLowerBound() << " to "
    << getUpperBound() << " step " << getStep();

  printInitializationList(p, getRegionIterArgs(), getIterOperands(),
                          " iter_args");
  if (!getIterOperands().empty())
    p << " -> (" << getIterOperands().getTypes() << ')';
  p << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/hasIterOperands());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type indexType = builder.getIndexType();

  OpAsmParser::Argument inductionVariable;
  inductionVariable.type = indexType;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual() ||
      // Parse loop bounds.
      parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();

    // Resolve input operands.
    for (auto argOperandType :
         llvm::zip(llvm::drop_begin(regionArgs), operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands))
        return failure();
    }
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

Region &ForOp::getLoopBody() { return getRegion(); }

ForOp mlir::scf::getForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast_or_null<ForOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those excluding the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange ForOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && *index == 0 && "invalid region index");

  // The initial operands map to the loop arguments after the induction
  // variable.
  return getInitArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ForOp::getSuccessorRegions(Optional<unsigned> index,
                                ArrayRef<Attribute> operands,
                                SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ForOp, branch into the body using the iterator
  // arguments.
  if (!index) {
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    return;
  }

  // Otherwise, the loop may branch back to itself or the parent operation.
  assert(*index == 0 && "expected loop region");
  regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
  regions.push_back(RegionSuccessor(getResults()));
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
    return LoopNest();
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
    auto loop = builder.create<scf::ForOp>(
        currentLoc, lbs[i], ubs[i], steps[i], currentIterArgs,
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
    builder.create<scf::YieldOp>(loc, loops[i + 1].getResults());
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
  builder.create<scf::YieldOp>(loc, results);

  // Return the loops.
  LoopNest res;
  res.loops.assign(loops.begin(), loops.end());
  return res;
}

LoopNest mlir::scf::buildLoopNest(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  // Delegate to the main function by wrapping the body builder.
  return buildLoopNest(builder, loc, lbs, ubs, steps, llvm::None,
                       [&bodyBuilder](OpBuilder &nestedBuilder,
                                      Location nestedLoc, ValueRange ivs,
                                      ValueRange) -> ValueVector {
                         if (bodyBuilder)
                           bodyBuilder(nestedBuilder, nestedLoc, ivs);
                         return {};
                       });
}

namespace {
// Fold away ForOp iter arguments when:
// 1) The op yields the iter arguments.
// 2) The iter arguments have no use and the corresponding outer region
// iterators (inputs) are yielded.
// 3) The iter arguments have no use and the corresponding (operation) results
// have no use.
//
// These arguments must be defined outside of
// the ForOp region and can just be forwarded after simplifying the op inits,
// yields and returns.
//
// The implementation uses `mergeBlockBefore` to steal the content of the
// original ForOp and avoid cloning.
struct ForOpIterArgsFolder : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    bool canonicalize = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    // An internal flat vector of block transfer
    // arguments `newBlockTransferArgs` keeps the 1-1 mapping of original to
    // transformed block argument mappings. This plays the role of a
    // BlockAndValueMapping for the particular use case of calling into
    // `mergeBlockBefore`.
    SmallVector<bool, 4> keepMask;
    keepMask.reserve(yieldOp.getNumOperands());
    SmallVector<Value, 4> newBlockTransferArgs, newIterArgs, newYieldValues,
        newResultValues;
    newBlockTransferArgs.reserve(1 + forOp.getNumIterOperands());
    newBlockTransferArgs.push_back(Value()); // iv placeholder with null value
    newIterArgs.reserve(forOp.getNumIterOperands());
    newYieldValues.reserve(yieldOp.getNumOperands());
    newResultValues.reserve(forOp.getNumResults());
    for (auto it : llvm::zip(forOp.getIterOperands(),   // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {
      // Forwarded is `true` when:
      // 1) The region `iter` argument is yielded.
      // 2) The region `iter` argument has no use, and the corresponding iter
      // operand (input) is yielded.
      // 3) The region `iter` argument has no use, and the corresponding op
      // result has no use.
      bool forwarded = ((std::get<1>(it) == std::get<3>(it)) ||
                        (std::get<1>(it).use_empty() &&
                         (std::get<0>(it) == std::get<3>(it) ||
                          std::get<2>(it).use_empty())));
      keepMask.push_back(!forwarded);
      canonicalize |= forwarded;
      if (forwarded) {
        newBlockTransferArgs.push_back(std::get<0>(it));
        newResultValues.push_back(std::get<0>(it));
        continue;
      }
      newIterArgs.push_back(std::get<0>(it));
      newYieldValues.push_back(std::get<3>(it));
      newBlockTransferArgs.push_back(Value()); // placeholder with null value
      newResultValues.push_back(Value());      // placeholder with null value
    }

    if (!canonicalize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block &newBlock = newForOp.getRegion().front();

    // Replace the null placeholders with newly constructed values.
    newBlockTransferArgs[0] = newBlock.getArgument(0); // iv
    for (unsigned idx = 0, collapsedIdx = 0, e = newResultValues.size();
         idx != e; ++idx) {
      Value &blockTransferArg = newBlockTransferArgs[1 + idx];
      Value &newResultVal = newResultValues[idx];
      assert((blockTransferArg && newResultVal) ||
             (!blockTransferArg && !newResultVal));
      if (!blockTransferArg) {
        blockTransferArg = newForOp.getRegionIterArgs()[collapsedIdx];
        newResultVal = newForOp.getResult(collapsedIdx++);
      }
    }

    Block &oldBlock = forOp.getRegion().front();
    assert(oldBlock.getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");

    // No results case: the scf::ForOp builder already created a zero
    // result terminator. Merge before this terminator and just get rid of the
    // original terminator that has been merged in.
    if (newIterArgs.empty()) {
      auto newYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
      rewriter.mergeBlockBefore(&oldBlock, newYieldOp, newBlockTransferArgs);
      rewriter.eraseOp(newBlock.getTerminator()->getPrevNode());
      rewriter.replaceOp(forOp, newResultValues);
      return success();
    }

    // No terminator case: merge and rewrite the merged terminator.
    auto cloneFilteredTerminator = [&](scf::YieldOp mergedTerminator) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(mergedTerminator);
      SmallVector<Value, 4> filteredOperands;
      filteredOperands.reserve(newResultValues.size());
      for (unsigned idx = 0, e = keepMask.size(); idx < e; ++idx)
        if (keepMask[idx])
          filteredOperands.push_back(mergedTerminator.getOperand(idx));
      rewriter.create<scf::YieldOp>(mergedTerminator.getLoc(),
                                    filteredOperands);
    };

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto mergedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    cloneFilteredTerminator(mergedYieldOp);
    rewriter.eraseOp(mergedYieldOp);
    rewriter.replaceOp(forOp, newResultValues);
    return success();
  }
};

/// Util function that tries to compute a constant diff between u and l.
/// Returns llvm::None when the difference between two AffineValueMap is
/// dynamic.
static Optional<int64_t> computeConstDiff(Value l, Value u) {
  IntegerAttr clb, cub;
  if (matchPattern(l, m_Constant(&clb)) && matchPattern(u, m_Constant(&cub))) {
    llvm::APInt lbValue = clb.getValue();
    llvm::APInt ubValue = cub.getValue();
    return (ubValue - lbValue).getSExtValue();
  }

  // Else a simple pattern match for x + c or c + x
  llvm::APInt diff;
  if (matchPattern(
          u, m_Op<arith::AddIOp>(matchers::m_Val(l), m_ConstantInt(&diff))) ||
      matchPattern(
          u, m_Op<arith::AddIOp>(m_ConstantInt(&diff), matchers::m_Val(l))))
    return diff.getSExtValue();
  return llvm::None;
}

/// Rewriting pattern that erases loops that are known not to iterate, replaces
/// single-iteration loops with their bodies, and removes empty loops that
/// iterate at least once and only return values defined outside of the loop.
struct SimplifyTrivialLoops : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    // If the upper bound is the same as the lower bound, the loop does not
    // iterate, just remove it.
    if (op.getLowerBound() == op.getUpperBound()) {
      rewriter.replaceOp(op, op.getIterOperands());
      return success();
    }

    Optional<int64_t> diff =
        computeConstDiff(op.getLowerBound(), op.getUpperBound());
    if (!diff)
      return failure();

    // If the loop is known to have 0 iterations, remove it.
    if (*diff <= 0) {
      rewriter.replaceOp(op, op.getIterOperands());
      return success();
    }

    llvm::Optional<llvm::APInt> maybeStepValue = op.getConstantStep();
    if (!maybeStepValue)
      return failure();

    // If the loop is known to have 1 iteration, inline its body and remove the
    // loop.
    llvm::APInt stepValue = *maybeStepValue;
    if (stepValue.sge(*diff)) {
      SmallVector<Value, 4> blockArgs;
      blockArgs.reserve(op.getNumIterOperands() + 1);
      blockArgs.push_back(op.getLowerBound());
      llvm::append_range(blockArgs, op.getIterOperands());
      replaceOpWithRegion(rewriter, op, op.getLoopBody(), blockArgs);
      return success();
    }

    // Now we are left with loops that have more than 1 iterations.
    Block &block = op.getRegion().front();
    if (!llvm::hasSingleElement(block))
      return failure();
    // If the loop is empty, iterates at least once, and only returns values
    // defined outside of the loop, remove it and replace it with yield values.
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    auto yieldOperands = yieldOp.getOperands();
    if (llvm::any_of(yieldOperands,
                     [&](Value v) { return !op.isDefinedOutsideOfLoop(v); }))
      return failure();
    rewriter.replaceOp(op, yieldOperands);
    return success();
  }
};

/// Perform a replacement of one iter OpOperand of an scf.for to the
/// `replacement` value which is expected to be the source of a tensor.cast.
/// tensor.cast ops are inserted inside the block to account for the type cast.
static ForOp replaceTensorCastForOpIterArg(PatternRewriter &rewriter,
                                           OpOperand &operand,
                                           Value replacement) {
  Type oldType = operand.get().getType(), newType = replacement.getType();
  assert(oldType.isa<RankedTensorType>() && newType.isa<RankedTensorType>() &&
         "expected ranked tensor types");

  // 1. Create new iter operands, exactly 1 is replaced.
  ForOp forOp = cast<ForOp>(operand.getOwner());
  assert(operand.getOperandNumber() >= forOp.getNumControlOperands() &&
         "expected an iter OpOperand");
  if (operand.get().getType() == replacement.getType())
    return forOp;
  SmallVector<Value> newIterOperands;
  for (OpOperand &opOperand : forOp.getIterOpOperands()) {
    if (opOperand.getOperandNumber() == operand.getOperandNumber()) {
      newIterOperands.push_back(replacement);
      continue;
    }
    newIterOperands.push_back(opOperand.get());
  }

  // 2. Create the new forOp shell.
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newIterOperands);
  newForOp->setAttrs(forOp->getAttrs());
  Block &newBlock = newForOp.getRegion().front();
  SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                             newBlock.getArguments().end());

  // 3. Inject an incoming cast op at the beginning of the block for the bbArg
  // corresponding to the `replacement` value.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(&newBlock, newBlock.begin());
  BlockArgument newRegionIterArg = newForOp.getRegionIterArgForOpOperand(
      newForOp->getOpOperand(operand.getOperandNumber()));
  Value castIn = rewriter.create<tensor::CastOp>(newForOp.getLoc(), oldType,
                                                 newRegionIterArg);
  newBlockTransferArgs[newRegionIterArg.getArgNumber()] = castIn;

  // 4. Steal the old block ops, mapping to the newBlockTransferArgs.
  Block &oldBlock = forOp.getRegion().front();
  rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

  // 5. Inject an outgoing cast op at the end of the block and yield it instead.
  auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
  rewriter.setInsertionPoint(clonedYieldOp);
  unsigned yieldIdx =
      newRegionIterArg.getArgNumber() - forOp.getNumInductionVars();
  Value castOut = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), newType, clonedYieldOp.getOperand(yieldIdx));
  SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
  newYieldOperands[yieldIdx] = castOut;
  rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  rewriter.eraseOp(clonedYieldOp);

  // 6. Inject an outgoing cast op after the forOp.
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> newResults = newForOp.getResults();
  newResults[yieldIdx] = rewriter.create<tensor::CastOp>(
      newForOp.getLoc(), oldType, newResults[yieldIdx]);

  return newForOp;
}

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
///   %2 = tensor.cast %1 : tensor<?x?xf32> to tensor<32x1024xf32>
///   use_of(%2)
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
///   use_of(%0)
/// ```
struct ForOpTensorCastFolder : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    for (auto it : llvm::zip(op.getIterOpOperands(), op.getResults())) {
      OpOperand &iterOpOperand = std::get<0>(it);
      auto incomingCast = iterOpOperand.get().getDefiningOp<tensor::CastOp>();
      if (!incomingCast)
        continue;
      if (!std::get<1>(it).hasOneUse())
        continue;
      auto outgoingCastOp =
          dyn_cast<tensor::CastOp>(*std::get<1>(it).user_begin());
      if (!outgoingCastOp)
        continue;

      // Must be a tensor.cast op pair with matching types.
      if (outgoingCastOp.getResult().getType() !=
          incomingCast.getSource().getType())
        continue;

      // Create a new ForOp with that iter operand replaced.
      auto newForOp = replaceTensorCastForOpIterArg(rewriter, iterOpOperand,
                                                    incomingCast.getSource());

      // Insert outgoing cast and use it to replace the corresponding result.
      rewriter.setInsertionPointAfter(newForOp);
      SmallVector<Value> replacements = newForOp.getResults();
      unsigned returnIdx =
          iterOpOperand.getOperandNumber() - op.getNumControlOperands();
      replacements[returnIdx] = rewriter.create<tensor::CastOp>(
          op.getLoc(), incomingCast.getDest().getType(),
          replacements[returnIdx]);
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

/// Canonicalize the iter_args of an scf::ForOp that involve a
/// `bufferization.to_tensor` and for which only the last loop iteration is
/// actually visible outside of the loop. The canonicalization looks for a
/// pattern such as:
/// ```
///    %t0 = ... : tensor_type
///    %0 = scf.for ... iter_args(%bb0 : %t0) -> (tensor_type) {
///      ...
///      // %m is either buffer_cast(%bb00) or defined above the loop
///      %m... : memref_type
///      ... // uses of %m with potential inplace updates
///      %new_tensor = bufferization.to_tensor %m : memref_type
///      ...
///      scf.yield %new_tensor : tensor_type
///    }
/// ```
///
/// `%bb0` may have either 0 or 1 use. If it has 1 use it must be exactly a
/// `%m = buffer_cast %bb0` op that feeds into the yielded
/// `bufferization.to_tensor` op.
///
/// If no aliasing write to the memref `%m`, from which `%new_tensor`is loaded,
/// occurs between `bufferization.to_tensor and yield then the value %0
/// visible outside of the loop is the last `bufferization.to_tensor`
/// produced in the loop.
///
/// For now, we approximate the absence of aliasing by only supporting the case
/// when the bufferization.to_tensor is the operation immediately preceding
/// the yield.
//
/// The canonicalization rewrites the pattern as:
/// ```
///    // %m is either a buffer_cast or defined above
///    %m... : memref_type
///    scf.for ... iter_args(%bb0 : %t0) -> (tensor_type) {
///      ... // uses of %m with potential inplace updates
///      scf.yield %bb0: tensor_type
///    }
///    %0 = bufferization.to_tensor %m : memref_type
/// ```
///
/// A later bbArg canonicalization will further rewrite as:
/// ```
///    // %m is either a buffer_cast or defined above
///    %m... : memref_type
///    scf.for ... { // no iter_args
///      ... // uses of %m with potential inplace updates
///    }
///    %0 = bufferization.to_tensor %m : memref_type
/// ```
struct LastTensorLoadCanonicalization : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    assert(std::next(forOp.getRegion().begin()) == forOp.getRegion().end() &&
           "unexpected multiple blocks");

    Location loc = forOp.getLoc();
    DenseMap<Value, Value> replacements;
    for (BlockArgument bbArg : forOp.getRegionIterArgs()) {
      unsigned idx = bbArg.getArgNumber() - /*numIv=*/1;
      auto yieldOp =
          cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
      Value yieldVal = yieldOp->getOperand(idx);
      auto tensorLoadOp = yieldVal.getDefiningOp<bufferization::ToTensorOp>();
      bool isTensor = bbArg.getType().isa<TensorType>();

      bufferization::ToMemrefOp tensorToMemref;
      // Either bbArg has no use or it has a single buffer_cast use.
      if (bbArg.hasOneUse())
        tensorToMemref =
            dyn_cast<bufferization::ToMemrefOp>(*bbArg.getUsers().begin());
      if (!isTensor || !tensorLoadOp || (!bbArg.use_empty() && !tensorToMemref))
        continue;
      // If tensorToMemref is present, it must feed into the `ToTensorOp`.
      if (tensorToMemref && tensorLoadOp.getMemref() != tensorToMemref)
        continue;
      // TODO: Any aliasing write of tensorLoadOp.memref() nested under `forOp`
      // must be before `ToTensorOp` in the block so that the lastWrite
      // property is not subject to additional side-effects.
      // For now, we only support the case when ToTensorOp appears
      // immediately before the terminator.
      if (tensorLoadOp->getNextNode() != yieldOp)
        continue;

      // Clone the optional tensorToMemref before forOp.
      if (tensorToMemref) {
        rewriter.setInsertionPoint(forOp);
        rewriter.replaceOpWithNewOp<bufferization::ToMemrefOp>(
            tensorToMemref, tensorToMemref.getMemref().getType(),
            tensorToMemref.getTensor());
      }

      // Clone the tensorLoad after forOp.
      rewriter.setInsertionPointAfter(forOp);
      Value newTensorLoad = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorLoadOp.getMemref());
      Value forOpResult = forOp.getResult(bbArg.getArgNumber() - /*iv=*/1);
      replacements.insert(std::make_pair(forOpResult, newTensorLoad));

      // Make the terminator just yield the bbArg, the old tensorLoadOp + the
      // old bbArg (that is now directly yielded) will canonicalize away.
      rewriter.startRootUpdate(yieldOp);
      yieldOp.setOperand(idx, bbArg);
      rewriter.finalizeRootUpdate(yieldOp);
    }
    if (replacements.empty())
      return failure();

    // We want to replace a subset of the results of `forOp`. rewriter.replaceOp
    // replaces the whole op and erase it unconditionally. This is wrong for
    // `forOp` as it generally contains ops with side effects.
    // Instead, use `rewriter.replaceOpWithIf`.
    SmallVector<Value> newResults;
    newResults.reserve(forOp.getNumResults());
    for (Value v : forOp.getResults()) {
      auto it = replacements.find(v);
      newResults.push_back((it != replacements.end()) ? it->second : v);
    }
    unsigned idx = 0;
    rewriter.replaceOpWithIf(forOp, newResults, [&](OpOperand &op) {
      return op.get() != newResults[idx++];
    });
    return success();
  }
};
} // namespace

void ForOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<ForOpIterArgsFolder, SimplifyTrivialLoops,
              LastTensorLoadCanonicalization, ForOpTensorCastFolder>(context);
}

Optional<APInt> ForOp::getConstantStep() {
  IntegerAttr step;
  if (matchPattern(getStep(), m_Constant(&step)))
    return step.getValue();
  return {};
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

//===----------------------------------------------------------------------===//
// ForeachThreadOp
//===----------------------------------------------------------------------===//

LogicalResult ForeachThreadOp::verify() {
  // Call terminator's verify to produce most informative error messages.
  if (failed(getTerminator().verify()))
    return failure();

  // Check number of outputs.
  if (getNumResults() != getOutputs().size())
    return emitOpError("produces ")
           << getNumResults() << " results, but has only "
           << getOutputs().size() << " outputs";

  // Check that the body defines block arguments for thread indices and outputs.
  auto *body = getBody();
  if (body->getNumArguments() != getRank() + getOutputs().size())
    return emitOpError("region expects ") << getRank() << " arguments";
  for (int64_t i = 0; i < getRank(); ++i)
    if (!body->getArgument(i).getType().isIndex())
      return emitOpError("expects ")
             << i << "-th block argument to be an index";
  for (unsigned i = 0; i < getOutputs().size(); ++i)
    if (body->getArgument(i + getRank()).getType() != getOutputs()[i].getType())
      return emitOpError("type mismatch between ")
             << i << "-th output and corresponding block argument";
  if (getMapping().has_value() && !getMapping()->empty()) {
    if (static_cast<int64_t>(getMapping()->size()) != getRank())
      return emitOpError() << "mapping attribute size must match op rank";
    for (auto map : getMapping()->getValue()) {
      if (!isa<DeviceMappingAttrInterface>(map))
        return emitOpError()
               << getMappingAttrName() << " is not device mapping attribute";
    }
  }

  return success();
}

void ForeachThreadOp::print(OpAsmPrinter &p) {
  p << " (";
  llvm::interleaveComma(getThreadIndices(), p);
  p << ") in (";
  llvm::interleaveComma(getNumThreads(), p);
  p << ")";
  printInitializationList(p, getRegionOutArgs(), getOutputs(), " shared_outs");
  p << " ";
  if (!getRegionOutArgs().empty())
    p << "-> (" << getResultTypes() << ") ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/getNumResults() > 0);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          {"operand_segment_sizes"});
}

ParseResult ForeachThreadOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse an opening `(` followed by thread index variables followed by `)`
  // TODO: when we can refer to such "induction variable"-like handles from the
  // declarative assembly format, we can implement the parser as a custom hook.
  SmallVector<OpAsmParser::Argument, 4> threadIndices;
  if (parser.parseArgumentList(threadIndices, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse `in` threadNums.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> threadNums;
  if (parser.parseKeyword("in") ||
      parser.parseOperandList(threadNums, threadIndices.size(),
                              OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(threadNums, builder.getIndexType(),
                             result.operands))
    return failure();

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
  for (auto &idx : threadIndices) {
    idx.type = builder.getIndexType();
    regionArgs.push_back(idx);
  }
  for (const auto &it : llvm::enumerate(regionOutArgs)) {
    auto &out = it.value();
    out.type = result.types[it.index()];
    regionArgs.push_back(out);
  }
  if (parser.parseRegion(*region, regionArgs))
    return failure();

  // Ensure terminator and move region.
  OpBuilder b(builder.getContext());
  ForeachThreadOp::ensureTerminator(*region, b, result.location);
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(threadNums.size()),
                           static_cast<int32_t>(outOperands.size())}));
  return success();
}

// Bodyless builder, outputs must be specified.
void ForeachThreadOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &result, ValueRange outputs,
                            ValueRange numThreads,
                            Optional<ArrayAttr> mapping) {
  result.addOperands(numThreads);
  result.addOperands(outputs);
  if (mapping.has_value()) {
    result.addAttribute(ForeachThreadOp::getMappingAttrName(result.name),
                        mapping.value());
  }

  result.addAttribute(
      "operand_segment_sizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(numThreads.size()),
                                    static_cast<int32_t>(outputs.size())}));
  result.addTypes(TypeRange(outputs));

  Region *bodyRegion = result.addRegion();
  OpBuilder::InsertionGuard g(builder);
  // createBlock sets the IP inside the block.
  // Generally we would guard against that but the default ensureTerminator impl
  // expects it ..
  builder.createBlock(bodyRegion);
  Block &bodyBlock = bodyRegion->front();
  // Add block arguments for indices and outputs.
  bodyBlock.addArguments(
      SmallVector<Type>(numThreads.size(), builder.getIndexType()),
      SmallVector<Location>(numThreads.size(), result.location));
  bodyBlock.addArguments(
      TypeRange(outputs),
      SmallVector<Location>(outputs.size(), result.location));
  ForeachThreadOp::ensureTerminator(*bodyRegion, builder, result.location);
}

// Builder that takes a bodyBuilder lambda.
void ForeachThreadOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, ValueRange outputs,
    ValueRange numThreads, ArrayRef<Attribute> mapping,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  result.addOperands(numThreads);
  result.addOperands(outputs);
  result.addAttribute(ForeachThreadOp::getMappingAttrName(result.name),
                      builder.getArrayAttr(mapping));
  result.addAttribute(
      "operand_segment_sizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(numThreads.size()),
                                    static_cast<int32_t>(outputs.size())}));
  result.addTypes(TypeRange(outputs));

  Region *bodyRegion = result.addRegion();
  OpBuilder::InsertionGuard g(builder);
  builder.createBlock(bodyRegion);
  Block &bodyBlock = bodyRegion->front();
  // Add block arguments for indices and outputs.
  bodyBlock.addArguments(
      SmallVector<Type>(numThreads.size(), builder.getIndexType()),
      SmallVector<Location>(numThreads.size(), result.location));
  bodyBlock.addArguments(
      TypeRange(outputs),
      SmallVector<Location>(outputs.size(), result.location));

  builder.setInsertionPointToStart(&bodyBlock);
  bodyBuilder(builder, result.location, bodyBlock.getArguments());
#ifndef NDEBUG
  auto terminator =
      llvm::dyn_cast<PerformConcurrentlyOp>(bodyBlock.getTerminator());
  assert(terminator &&
         "expected bodyBuilder to create PerformConcurrentlyOp terminator");
#endif // NDEBUG
}

// The ensureTerminator method generated by SingleBlockImplicitTerminator is
// unaware of the fact that our terminator also needs a region to be
// well-formed. We override it here to ensure that we do the right thing.
void ForeachThreadOp::ensureTerminator(Region &region, OpBuilder &builder,
                                       Location loc) {
  OpTrait::SingleBlockImplicitTerminator<PerformConcurrentlyOp>::Impl<
      ForeachThreadOp>::ensureTerminator(region, builder, loc);
  auto terminator =
      llvm::dyn_cast<PerformConcurrentlyOp>(region.front().getTerminator());
  if (terminator.getRegion().empty())
    builder.createBlock(&terminator.getRegion());
}

PerformConcurrentlyOp ForeachThreadOp::getTerminator() {
  return cast<PerformConcurrentlyOp>(getBody()->getTerminator());
}

/// Helper to sort `values` according to matching `keys`.
SmallVector<Value> ForeachThreadOp::getValuesSortedByKey(
    ArrayRef<Attribute> keys, ValueRange values,
    llvm::function_ref<bool(Attribute, Attribute)> compare) {
  if (keys.empty())
    return values;
  assert(keys.size() == values.size() && "unexpected mismatching sizes");
  auto indices = llvm::to_vector(llvm::seq<int64_t>(0, values.size()));
  std::sort(indices.begin(), indices.end(),
            [&](int64_t i, int64_t j) { return compare(keys[i], keys[j]); });
  SmallVector<Value> res;
  res.reserve(values.size());
  for (int64_t i = 0, e = indices.size(); i < e; ++i)
    res.push_back(values[indices[i]]);
  return res;
}

ForeachThreadOp mlir::scf::getForeachThreadOpThreadIndexOwner(Value val) {
  auto tidxArg = val.dyn_cast<BlockArgument>();
  if (!tidxArg)
    return ForeachThreadOp();
  assert(tidxArg.getOwner() && "unlinked block argument");
  auto *containingOp = tidxArg.getOwner()->getParentOp();
  return dyn_cast<ForeachThreadOp>(containingOp);
}

//===----------------------------------------------------------------------===//
// PerformConcurrentlyOp
//===----------------------------------------------------------------------===//

// Build a PerformConcurrentlyOp with mixed static and dynamic entries.
void PerformConcurrentlyOp::build(OpBuilder &b, OperationState &result) {
  OpBuilder::InsertionGuard g(b);
  Region *bodyRegion = result.addRegion();
  b.createBlock(bodyRegion);
}

LogicalResult PerformConcurrentlyOp::verify() {
  scf::ForeachThreadOp foreachThreadOp =
      dyn_cast<scf::ForeachThreadOp>(getOperation()->getParentOp());
  if (!foreachThreadOp)
    return this->emitOpError("expected foreach_thread op parent");

  // TODO: PerformConcurrentlyOpInterface.
  for (Operation &op : getRegion().front().getOperations()) {
    if (!isa<tensor::ParallelInsertSliceOp>(op)) {
      return this->emitOpError("expected only ")
             << tensor::ParallelInsertSliceOp::getOperationName() << " ops";
    }

    // Verify that inserts are into out block arguments.
    Value dest = cast<tensor::ParallelInsertSliceOp>(op).getDest();
    ArrayRef<BlockArgument> regionOutArgs = foreachThreadOp.getRegionOutArgs();
    if (!llvm::is_contained(regionOutArgs, dest))
      return op.emitOpError("may only insert into an output block argument");
  }
  return success();
}

void PerformConcurrentlyOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult PerformConcurrentlyOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
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

OpResult PerformConcurrentlyOp::getParentResult(int64_t idx) {
  return getOperation()->getParentOp()->getResult(idx);
}

SmallVector<BlockArgument> PerformConcurrentlyOp::getDests() {
  return llvm::to_vector<4>(
      llvm::map_range(getYieldingOps(), [](Operation &op) {
        // Add new ops here as needed.
        auto insertSliceOp = cast<tensor::ParallelInsertSliceOp>(&op);
        return insertSliceOp.getDest().cast<BlockArgument>();
      }));
}

llvm::iterator_range<Block::iterator> PerformConcurrentlyOp::getYieldingOps() {
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

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 bool withElseRegion) {
  build(builder, result, /*resultTypes=*/llvm::None, cond, withElseRegion);
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond, bool withElseRegion) {
  auto addTerminator = [&](OpBuilder &nested, Location loc) {
    if (resultTypes.empty())
      IfOp::ensureTerminator(*nested.getInsertionBlock()->getParent(), nested,
                             loc);
  };

  build(builder, result, resultTypes, cond, addTerminator,
        withElseRegion ? addTerminator
                       : function_ref<void(OpBuilder &, Location)>());
}

void IfOp::build(OpBuilder &builder, OperationState &result,
                 TypeRange resultTypes, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");

  result.addOperands(cond);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  Region *elseRegion = result.addRegion();
  if (!elseBuilder)
    return;

  builder.createBlock(elseRegion);
  elseBuilder(builder, result.location);
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  build(builder, result, TypeRange(), cond, thenBuilder, elseBuilder);
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

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void IfOp::getSuccessorRegions(Optional<unsigned> index,
                               ArrayRef<Attribute> operands,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (index) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // Don't consider the else region if it is empty.
  Region *elseRegion = &this->getElseRegion();
  if (elseRegion->empty())
    elseRegion = nullptr;

  // Otherwise, the successor is dependent on the condition.
  bool condition;
  if (auto condAttr = operands.front().dyn_cast_or_null<IntegerAttr>()) {
    condition = condAttr.getValue().isOneValue();
  } else {
    // If the condition isn't constant, both regions may be executed.
    regions.push_back(RegionSuccessor(&getThenRegion()));
    // If the else region does not exist, it is not a viable successor.
    if (elseRegion)
      regions.push_back(RegionSuccessor(elseRegion));
    return;
  }

  // Add the successor regions using the condition.
  regions.push_back(RegionSuccessor(condition ? &getThenRegion() : elseRegion));
}

LogicalResult IfOp::fold(ArrayRef<Attribute> operands,
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
  if (auto cond = operands[0].dyn_cast_or_null<BoolAttr>()) {
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
// Pattern to remove unused IfOp results.
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    SmallVector<Value, 4> usedOperands;
    llvm::transform(usedResults, std::back_inserter(usedOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.updateRootInPlace(yieldOp,
                               [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Compute the list of used results.
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    // Replace the operation if only a subset of its results have uses.
    if (usedResults.size() == op.getNumResults())
      return failure();

    // Compute the result types of the replacement operation.
    SmallVector<Type, 4> newTypes;
    llvm::transform(usedResults, std::back_inserter(newTypes),
                    [](OpResult result) { return result.getType(); });

    // Create a replacement operation with empty then and else regions.
    auto emptyBuilder = [](OpBuilder &, Location) {};
    auto newOp = rewriter.create<IfOp>(op.getLoc(), newTypes, op.getCondition(),
                                       emptyBuilder, emptyBuilder);

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct RemoveStaticCondition : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    BoolAttr condition;
    if (!matchPattern(op.getCondition(), m_Constant(&condition)))
      return failure();

    if (condition.getValue())
      replaceOpWithRegion(rewriter, op, op.getThenRegion());
    else if (!op.getElseRegion().empty())
      replaceOpWithRegion(rewriter, op, op.getElseRegion());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

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
    for (const auto &it :
         llvm::enumerate(llvm::zip(thenYieldArgs, elseYieldArgs))) {
      Value trueVal = std::get<0>(it.value());
      Value falseVal = std::get<1>(it.value());
      if (&op.getThenRegion() == trueVal.getParentRegion() ||
          &op.getElseRegion() == falseVal.getParentRegion())
        nonHoistable.push_back(trueVal.getType());
    }
    // Early exit if there aren't any yielded values we can
    // hoist outside the if.
    if (nonHoistable.size() == op->getNumResults())
      return failure();

    IfOp replacement = rewriter.create<IfOp>(op.getLoc(), nonHoistable, cond);
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
        results[it.index()] = rewriter.create<arith::SelectOp>(
            op.getLoc(), cond, trueVal, falseVal);
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

    for (OpOperand &use :
         llvm::make_early_inc_range(op.getCondition().getUses())) {
      if (op.getThenRegion().isAncestor(use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantTrue)
          constantTrue = rewriter.create<arith::ConstantOp>(
              op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 1));

        rewriter.updateRootInPlace(use.getOwner(),
                                   [&]() { use.set(constantTrue); });
      } else if (op.getElseRegion().isAncestor(
                     use.getOwner()->getParentRegion())) {
        changed = true;

        if (!constantFalse)
          constantFalse = rewriter.create<arith::ConstantOp>(
              op.getLoc(), i1Ty, rewriter.getIntegerAttr(i1Ty, 0));

        rewriter.updateRootInPlace(use.getOwner(),
                                   [&]() { use.set(constantFalse); });
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
          Value notCond = rewriter.create<arith::XOrIOp>(
              op.getLoc(), op.getCondition(),
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
          rewriter.startRootUpdate(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeRootUpdate(use.getOwner());
        } else if (nextElse && nextElse->getParent()->isAncestor(
                                   use.getOwner()->getParentRegion())) {
          rewriter.startRootUpdate(use.getOwner());
          use.set(std::get<2>(it));
          rewriter.finalizeRootUpdate(use.getOwner());
        }
      }

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    IfOp combinedIf = rewriter.create<IfOp>(
        nextIf.getLoc(), mergedTypes, prevIf.getCondition(), /*hasElse=*/false);
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
      rewriter.create<YieldOp>(thenYield2.getLoc(), mergedYields);
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

        rewriter.create<YieldOp>(elseYield2.getLoc(), mergedElseYields);
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
        auto nestedIdx = tup.value().cast<OpResult>().getResultNumber();
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
    Value newCondition = rewriter.create<arith::AndIOp>(
        loc, op.getCondition(), nestedIf.getCondition());
    auto newIf = rewriter.create<IfOp>(loc, op.getResultTypes(), newCondition);

    SmallVector<Value> results;
    llvm::append_range(results, newIf.getResults());
    rewriter.setInsertionPoint(newIf);

    for (auto idx : elseYieldsToUpgradeToSelect)
      results[idx] = rewriter.create<arith::SelectOp>(
          op.getLoc(), op.getCondition(), thenYield[idx], elseYield[idx]);

    Block *newIfBlock = newIf.thenBlock();
    if (newIfBlock)
      rewriter.eraseOp(newIfBlock->getTerminator());
    else
      newIfBlock = rewriter.createBlock(&newIf.getThenRegion());
    rewriter.mergeBlocks(nestedIf.thenBlock(), newIfBlock);
    rewriter.setInsertionPointToEnd(newIf.thenBlock());
    rewriter.replaceOpWithNewOp<YieldOp>(newIf.thenYield(), thenYield);
    if (!elseYield.empty()) {
      rewriter.createBlock(&newIf.getElseRegion());
      rewriter.setInsertionPointToEnd(newIf.elseBlock());
      rewriter.create<YieldOp>(loc, elseYield);
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
              RemoveStaticCondition, RemoveUnusedResults,
              ReplaceIfYieldWithConditionOrValue>(context);
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
    if (auto cst = stepValue.getDefiningOp<arith::ConstantIndexOp>())
      if (cst.value() <= 0)
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

  // Check that the yield has no results
  Operation *yield = body->getTerminator();
  if (yield->getNumOperands() != 0)
    return yield->emitOpError() << "not allowed to have operands inside '"
                                << ParallelOp::getOperationName() << "'";

  // Check that the number of results is the same as the number of ReduceOps.
  SmallVector<ReduceOp, 4> reductions(body->getOps<ReduceOp>());
  auto resultsSize = getResults().size();
  auto reductionsSize = reductions.size();
  auto initValsSize = getInitVals().size();
  if (resultsSize != reductionsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of reductions: "
                         << reductionsSize;
  if (resultsSize != initValsSize)
    return emitOpError() << "expects number of results: " << resultsSize
                         << " to be the same as number of initial values: "
                         << initValsSize;

  // Check that the types of the results and reductions are the same.
  for (auto resultAndReduce : llvm::zip(getResults(), reductions)) {
    auto resultType = std::get<0>(resultAndReduce).getType();
    auto reduceOp = std::get<1>(resultAndReduce);
    auto reduceType = reduceOp.getOperand().getType();
    if (resultType != reduceType)
      return reduceOp.emitOpError()
             << "expects type of reduce: " << reduceType
             << " to be the same as result type: " << resultType;
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

  // Set `operand_segment_sizes` attribute.
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
  ForOp::ensureTerminator(*body, builder, result.location);
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

Region &ParallelOp::getLoopBody() { return getRegion(); }

ParallelOp mlir::scf::getParallelForInductionVarOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return ParallelOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<ParallelOp>(containingOp);
}

namespace {
// Collapse loop dimensions that perform a single iteration.
struct CollapseSingleIterationLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    BlockAndValueMapping mapping;
    // Compute new loop bounds that omit all single-iteration loop dimensions.
    SmallVector<Value, 2> newLowerBounds;
    SmallVector<Value, 2> newUpperBounds;
    SmallVector<Value, 2> newSteps;
    newLowerBounds.reserve(op.getLowerBound().size());
    newUpperBounds.reserve(op.getUpperBound().size());
    newSteps.reserve(op.getStep().size());
    for (auto [lowerBound, upperBound, step, iv] :
         llvm::zip(op.getLowerBound(), op.getUpperBound(), op.getStep(),
                   op.getInductionVars())) {
      // Collect the statically known loop bounds.
      auto lowerBoundConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
      auto upperBoundConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
      auto stepConstant =
          dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
      // Replace the loop induction variable by the lower bound if the loop
      // performs a single iteration. Otherwise, copy the loop bounds.
      if (lowerBoundConstant && upperBoundConstant && stepConstant &&
          (upperBoundConstant.value() - lowerBoundConstant.value()) > 0 &&
          (upperBoundConstant.value() - lowerBoundConstant.value()) <=
              stepConstant.value()) {
        mapping.map(iv, lowerBound);
      } else {
        newLowerBounds.push_back(lowerBound);
        newUpperBounds.push_back(upperBound);
        newSteps.push_back(step);
      }
    }
    // Exit if none of the loop dimensions perform a single iteration.
    if (newLowerBounds.size() == op.getLowerBound().size())
      return failure();

    if (newLowerBounds.empty()) {
      // All of the loop dimensions perform a single iteration. Inline
      // loop body and nested ReduceOp's
      SmallVector<Value> results;
      results.reserve(op.getInitVals().size());
      for (auto &bodyOp : op.getLoopBody().front().without_terminator()) {
        auto reduce = dyn_cast<ReduceOp>(bodyOp);
        if (!reduce) {
          rewriter.clone(bodyOp, mapping);
          continue;
        }
        Block &reduceBlock = reduce.getReductionOperator().front();
        auto initValIndex = results.size();
        mapping.map(reduceBlock.getArgument(0), op.getInitVals()[initValIndex]);
        mapping.map(reduceBlock.getArgument(1),
                    mapping.lookupOrDefault(reduce.getOperand()));
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
        rewriter.create<ParallelOp>(op.getLoc(), newLowerBounds, newUpperBounds,
                                    newSteps, op.getInitVals(), nullptr);
    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

/// Removes parallel loops in which at least one lower/upper bound pair consists
/// of the same values - such loops have an empty iteration domain.
struct RemoveEmptyParallelLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    for (auto dim : llvm::zip(op.getLowerBound(), op.getUpperBound())) {
      if (std::get<0>(dim) == std::get<1>(dim)) {
        rewriter.replaceOp(op, op.getInitVals());
        return success();
      }
    }
    return failure();
  }
};

struct MergeNestedParallelLoops : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = op.getLoopBody().front();
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
      Block &innerBody = innerOp.getLoopBody().front();
      assert(iterVals.size() ==
             (outerBody.getNumArguments() + innerBody.getNumArguments()));
      BlockAndValueMapping mapping;
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
                                            newSteps, llvm::None, bodyBuilder);
    return success();
  }
};

} // namespace

void ParallelOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<CollapseSingleIterationLoops, RemoveEmptyParallelLoops,
              MergeNestedParallelLoops>(context);
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::build(
    OpBuilder &builder, OperationState &result, Value operand,
    function_ref<void(OpBuilder &, Location, Value, Value)> bodyBuilderFn) {
  auto type = operand.getType();
  result.addOperands(operand);

  OpBuilder::InsertionGuard guard(builder);
  Region *bodyRegion = result.addRegion();
  Block *body = builder.createBlock(bodyRegion, {}, ArrayRef<Type>{type, type},
                                    {result.location, result.location});
  if (bodyBuilderFn)
    bodyBuilderFn(builder, result.location, body->getArgument(0),
                  body->getArgument(1));
}

LogicalResult ReduceOp::verifyRegions() {
  // The region of a ReduceOp has two arguments of the same type as its operand.
  auto type = getOperand().getType();
  Block &block = getReductionOperator().front();
  if (block.empty())
    return emitOpError("the block inside reduce should not be empty");
  if (block.getNumArguments() != 2 ||
      llvm::any_of(block.getArguments(), [&](const BlockArgument &arg) {
        return arg.getType() != type;
      }))
    return emitOpError() << "expects two arguments to reduce block of type "
                         << type;

  // Check that the block is terminated by a ReduceReturnOp.
  if (!isa<ReduceReturnOp>(block.getTerminator()))
    return emitOpError("the block inside reduce should be terminated with a "
                       "'scf.reduce.return' op");

  return success();
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse an opening `(` followed by the reduced value followed by `)`
  OpAsmParser::UnresolvedOperand operand;
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

void ReduceOp::print(OpAsmPrinter &p) {
  p << "(" << getOperand() << ") ";
  p << " : " << getOperand().getType() << ' ';
  p.printRegion(getReductionOperator());
}

//===----------------------------------------------------------------------===//
// ReduceReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceReturnOp::verify() {
  // The type of the return value should be the same type as the type of the
  // operand of the enclosing ReduceOp.
  auto reduceOp = cast<ReduceOp>((*this)->getParentOp());
  Type reduceType = reduceOp.getOperand().getType();
  if (reduceType != getResult().getType())
    return emitOpError() << "needs to have type " << reduceType
                         << " (the type of the enclosing ReduceOp)";
  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void WhileOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, TypeRange resultTypes,
                    ValueRange operands, BodyBuilderFn beforeBuilder,
                    BodyBuilderFn afterBuilder) {
  assert(beforeBuilder && "the builder callback for 'before' must be present");
  assert(afterBuilder && "the builder callback for 'after' must be present");

  odsState.addOperands(operands);
  odsState.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(odsBuilder);

  SmallVector<Location, 4> blockArgLocs;
  for (Value operand : operands) {
    blockArgLocs.push_back(operand.getLoc());
  }

  Region *beforeRegion = odsState.addRegion();
  Block *beforeBlock = odsBuilder.createBlock(beforeRegion, /*insertPt=*/{},
                                              resultTypes, blockArgLocs);
  beforeBuilder(odsBuilder, odsState.location, beforeBlock->getArguments());

  Region *afterRegion = odsState.addRegion();
  Block *afterBlock = odsBuilder.createBlock(afterRegion, /*insertPt=*/{},
                                             resultTypes, blockArgLocs);
  afterBuilder(odsBuilder, odsState.location, afterBlock->getArguments());
}

OperandRange WhileOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && *index == 0 &&
         "WhileOp is expected to branch only to the first region");

  return getInits();
}

ConditionOp WhileOp::getConditionOp() {
  return cast<ConditionOp>(getBefore().front().getTerminator());
}

YieldOp WhileOp::getYieldOp() {
  return cast<YieldOp>(getAfter().front().getTerminator());
}

Block::BlockArgListType WhileOp::getBeforeArguments() {
  return getBefore().front().getArguments();
}

Block::BlockArgListType WhileOp::getAfterArguments() {
  return getAfter().front().getArguments();
}

void WhileOp::getSuccessorRegions(Optional<unsigned> index,
                                  ArrayRef<Attribute> operands,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (!index) {
    regions.emplace_back(&getBefore(), getBefore().getArguments());
    return;
  }

  assert(*index < 2 && "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (*index == 1) {
    regions.emplace_back(&getBefore(), getBefore().getArguments());
    return;
  }

  // Try to narrow the successor to the condition region.
  assert(!operands.empty() && "expected at least one operand");
  auto cond = operands[0].dyn_cast_or_null<BoolAttr>();
  if (!cond || !cond.getValue())
    regions.emplace_back(getResults());
  if (!cond || cond.getValue())
    regions.emplace_back(&getAfter(), getAfter().getArguments());
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
           << "expected as many input types as operands "
           << "(expected " << operands.size() << " got "
           << functionType.getNumInputs() << ")";
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
  printInitializationList(p, getBefore().front().getArguments(), getInits(),
                          " ");
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

/// Verifies that the first block of the given `region` is terminated by a
/// YieldOp. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(scf::WhileOp op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = region.front().getTerminator();
  if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
    return yield;

  auto diag = op.emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
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
            constantTrue = rewriter.create<arith::ConstantOp>(
                op.getLoc(), term.getCondition().getType(),
                rewriter.getBoolAttr(true));

          std::get<1>(yieldedAndBlockArgs).replaceAllUsesWith(constantTrue);
          replaced = true;
        }
      }
    }
    return success(replaced);
  }
};

/// Remove loop invariant arguments from `before` block of scf.while.
/// A before block argument is considered loop invariant if :-
///   1. i-th yield operand is equal to the i-th while operand.
///   2. i-th yield operand is k-th after block argument which is (k+1)-th
///      condition operand AND this (k+1)-th condition operand is equal to i-th
///      iter argument/while operand.
/// For the arguments which are removed, their uses inside scf.while
/// are replaced with their corresponding initial value.
///
/// Eg:
///    INPUT :-
///    %res = scf.while <...> iter_args(%arg0_before = %a, %arg1_before = %b,
///                                     ..., %argN_before = %N)
///           {
///                ...
///                scf.condition(%cond) %arg1_before, %arg0_before,
///                                     %arg2_before, %arg0_before, ...
///           } do {
///             ^bb0(%arg1_after, %arg0_after_1, %arg2_after, %arg0_after_2,
///                  ..., %argK_after):
///                ...
///                scf.yield %arg0_after_2, %b, %arg1_after, ..., %argN
///           }
///
///    OUTPUT :-
///    %res = scf.while <...> iter_args(%arg2_before = %c, ..., %argN_before =
///                                     %N)
///           {
///                ...
///                scf.condition(%cond) %b, %a, %arg2_before, %a, ...
///           } do {
///             ^bb0(%arg1_after, %arg0_after_1, %arg2_after, %arg0_after_2,
///                  ..., %argK_after):
///                ...
///                scf.yield %arg1_after, ..., %argN
///           }
///
///    EXPLANATION:
///      We iterate over each yield operand.
///        1. 0-th yield operand %arg0_after_2 is 4-th condition operand
///           %arg0_before, which in turn is the 0-th iter argument. So we
///           remove 0-th before block argument and yield operand, and replace
///           all uses of the 0-th before block argument with its initial value
///           %a.
///        2. 1-th yield operand %b is equal to the 1-th iter arg's initial
///           value. So we remove this operand and the corresponding before
///           block argument and replace all uses of 1-th before block argument
///           with %b.
struct RemoveLoopInvariantArgsFromBeforeBlock
    : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    Block &afterBlock = op.getAfter().front();
    Block::BlockArgListType beforeBlockArgs = op.getBeforeArguments();
    ConditionOp condOp = op.getConditionOp();
    OperandRange condOpArgs = condOp.getArgs();
    Operation *yieldOp = afterBlock.getTerminator();
    ValueRange yieldOpArgs = yieldOp->getOperands();

    bool canSimplify = false;
    for (const auto &it :
         llvm::enumerate(llvm::zip(op.getOperands(), yieldOpArgs))) {
      auto index = static_cast<unsigned>(it.index());
      auto [initVal, yieldOpArg] = it.value();
      // If i-th yield operand is equal to the i-th operand of the scf.while,
      // the i-th before block argument is a loop invariant.
      if (yieldOpArg == initVal) {
        canSimplify = true;
        break;
      }
      // If the i-th yield operand is k-th after block argument, then we check
      // if the (k+1)-th condition op operand is equal to either the i-th before
      // block argument or the initial value of i-th before block argument. If
      // the comparison results `true`, i-th before block argument is a loop
      // invariant.
      auto yieldOpBlockArg = yieldOpArg.dyn_cast<BlockArgument>();
      if (yieldOpBlockArg && yieldOpBlockArg.getOwner() == &afterBlock) {
        Value condOpArg = condOpArgs[yieldOpBlockArg.getArgNumber()];
        if (condOpArg == beforeBlockArgs[index] || condOpArg == initVal) {
          canSimplify = true;
          break;
        }
      }
    }

    if (!canSimplify)
      return failure();

    SmallVector<Value> newInitArgs, newYieldOpArgs;
    DenseMap<unsigned, Value> beforeBlockInitValMap;
    SmallVector<Location> newBeforeBlockArgLocs;
    for (const auto &it :
         llvm::enumerate(llvm::zip(op.getOperands(), yieldOpArgs))) {
      auto index = static_cast<unsigned>(it.index());
      auto [initVal, yieldOpArg] = it.value();

      // If i-th yield operand is equal to the i-th operand of the scf.while,
      // the i-th before block argument is a loop invariant.
      if (yieldOpArg == initVal) {
        beforeBlockInitValMap.insert({index, initVal});
        continue;
      } else {
        // If the i-th yield operand is k-th after block argument, then we check
        // if the (k+1)-th condition op operand is equal to either the i-th
        // before block argument or the initial value of i-th before block
        // argument. If the comparison results `true`, i-th before block
        // argument is a loop invariant.
        auto yieldOpBlockArg = yieldOpArg.dyn_cast<BlockArgument>();
        if (yieldOpBlockArg && yieldOpBlockArg.getOwner() == &afterBlock) {
          Value condOpArg = condOpArgs[yieldOpBlockArg.getArgNumber()];
          if (condOpArg == beforeBlockArgs[index] || condOpArg == initVal) {
            beforeBlockInitValMap.insert({index, initVal});
            continue;
          }
        }
      }
      newInitArgs.emplace_back(initVal);
      newYieldOpArgs.emplace_back(yieldOpArg);
      newBeforeBlockArgLocs.emplace_back(beforeBlockArgs[index].getLoc());
    }

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, newYieldOpArgs);
    }

    auto newWhile =
        rewriter.create<WhileOp>(op.getLoc(), op.getResultTypes(), newInitArgs);

    Block &newBeforeBlock = *rewriter.createBlock(
        &newWhile.getBefore(), /*insertPt*/ {},
        ValueRange(newYieldOpArgs).getTypes(), newBeforeBlockArgLocs);

    Block &beforeBlock = op.getBefore().front();
    SmallVector<Value> newBeforeBlockArgs(beforeBlock.getNumArguments());
    // For each i-th before block argument we find it's replacement value as :-
    //   1. If i-th before block argument is a loop invariant, we fetch it's
    //      initial value from `beforeBlockInitValMap` by querying for key `i`.
    //   2. Else we fetch j-th new before block argument as the replacement
    //      value of i-th before block argument.
    for (unsigned i = 0, j = 0, n = beforeBlock.getNumArguments(); i < n; i++) {
      // If the index 'i' argument was a loop invariant we fetch it's initial
      // value from `beforeBlockInitValMap`.
      if (beforeBlockInitValMap.count(i) != 0)
        newBeforeBlockArgs[i] = beforeBlockInitValMap[i];
      else
        newBeforeBlockArgs[i] = newBeforeBlock.getArgument(j++);
    }

    rewriter.mergeBlocks(&beforeBlock, &newBeforeBlock, newBeforeBlockArgs);
    rewriter.inlineRegionBefore(op.getAfter(), newWhile.getAfter(),
                                newWhile.getAfter().begin());

    rewriter.replaceOp(op, newWhile.getResults());
    return success();
  }
};

/// Remove loop invariant value from result (condition op) of scf.while.
/// A value is considered loop invariant if the final value yielded by
/// scf.condition is defined outside of the `before` block. We remove the
/// corresponding argument in `after` block and replace the use with the value.
/// We also replace the use of the corresponding result of scf.while with the
/// value.
///
/// Eg:
///    INPUT :-
///    %res_input:K = scf.while <...> iter_args(%arg0_before = , ...,
///                                             %argN_before = %N) {
///                ...
///                scf.condition(%cond) %arg0_before, %a, %b, %arg1_before, ...
///           } do {
///             ^bb0(%arg0_after, %arg1_after, %arg2_after, ..., %argK_after):
///                ...
///                some_func(%arg1_after)
///                ...
///                scf.yield %arg0_after, %arg2_after, ..., %argN_after
///           }
///
///    OUTPUT :-
///    %res_output:M = scf.while <...> iter_args(%arg0 = , ..., %argN = %N) {
///                ...
///                scf.condition(%cond) %arg0, %arg1, ..., %argM
///           } do {
///             ^bb0(%arg0, %arg3, ..., %argM):
///                ...
///                some_func(%a)
///                ...
///                scf.yield %arg0, %b, ..., %argN
///           }
///
///     EXPLANATION:
///       1. The 1-th and 2-th operand of scf.condition are defined outside the
///          before block of scf.while, so they get removed.
///       2. %res_input#1's uses are replaced by %a and %res_input#2's uses are
///          replaced by %b.
///       3. The corresponding after block argument %arg1_after's uses are
///          replaced by %a and %arg2_after's uses are replaced by %b.
struct RemoveLoopInvariantValueYielded : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    Block &beforeBlock = op.getBefore().front();
    ConditionOp condOp = op.getConditionOp();
    OperandRange condOpArgs = condOp.getArgs();

    bool canSimplify = false;
    for (Value condOpArg : condOpArgs) {
      // Those values not defined within `before` block will be considered as
      // loop invariant values. We map the corresponding `index` with their
      // value.
      if (condOpArg.getParentBlock() != &beforeBlock) {
        canSimplify = true;
        break;
      }
    }

    if (!canSimplify)
      return failure();

    Block::BlockArgListType afterBlockArgs = op.getAfterArguments();

    SmallVector<Value> newCondOpArgs;
    SmallVector<Type> newAfterBlockType;
    DenseMap<unsigned, Value> condOpInitValMap;
    SmallVector<Location> newAfterBlockArgLocs;
    for (const auto &it : llvm::enumerate(condOpArgs)) {
      auto index = static_cast<unsigned>(it.index());
      Value condOpArg = it.value();
      // Those values not defined within `before` block will be considered as
      // loop invariant values. We map the corresponding `index` with their
      // value.
      if (condOpArg.getParentBlock() != &beforeBlock) {
        condOpInitValMap.insert({index, condOpArg});
      } else {
        newCondOpArgs.emplace_back(condOpArg);
        newAfterBlockType.emplace_back(condOpArg.getType());
        newAfterBlockArgLocs.emplace_back(afterBlockArgs[index].getLoc());
      }
    }

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(condOp);
      rewriter.replaceOpWithNewOp<ConditionOp>(condOp, condOp.getCondition(),
                                               newCondOpArgs);
    }

    auto newWhile = rewriter.create<WhileOp>(op.getLoc(), newAfterBlockType,
                                             op.getOperands());

    Block &newAfterBlock =
        *rewriter.createBlock(&newWhile.getAfter(), /*insertPt*/ {},
                              newAfterBlockType, newAfterBlockArgLocs);

    Block &afterBlock = op.getAfter().front();
    // Since a new scf.condition op was created, we need to fetch the new
    // `after` block arguments which will be used while replacing operations of
    // previous scf.while's `after` blocks. We'd also be fetching new result
    // values too.
    SmallVector<Value> newAfterBlockArgs(afterBlock.getNumArguments());
    SmallVector<Value> newWhileResults(afterBlock.getNumArguments());
    for (unsigned i = 0, j = 0, n = afterBlock.getNumArguments(); i < n; i++) {
      Value afterBlockArg, result;
      // If index 'i' argument was loop invariant we fetch it's value from the
      // `condOpInitMap` map.
      if (condOpInitValMap.count(i) != 0) {
        afterBlockArg = condOpInitValMap[i];
        result = afterBlockArg;
      } else {
        afterBlockArg = newAfterBlock.getArgument(j);
        result = newWhile.getResult(j);
        j++;
      }
      newAfterBlockArgs[i] = afterBlockArg;
      newWhileResults[i] = result;
    }

    rewriter.mergeBlocks(&afterBlock, &newAfterBlock, newAfterBlockArgs);
    rewriter.inlineRegionBefore(op.getBefore(), newWhile.getBefore(),
                                newWhile.getBefore().begin());

    rewriter.replaceOp(op, newWhileResults);
    return success();
  }
};

/// Remove WhileOp results that are also unused in 'after' block.
///
///  %0:2 = scf.while () : () -> (i32, i64) {
///    %condition = "test.condition"() : () -> i1
///    %v1 = "test.get_some_value"() : () -> i32
///    %v2 = "test.get_some_value"() : () -> i64
///    scf.condition(%condition) %v1, %v2 : i32, i64
///  } do {
///  ^bb0(%arg0: i32, %arg1: i64):
///    "test.use"(%arg0) : (i32) -> ()
///    scf.yield
///  }
///  return %0#0 : i32
///
/// becomes
///  %0 = scf.while () : () -> (i32) {
///    %condition = "test.condition"() : () -> i1
///    %v1 = "test.get_some_value"() : () -> i32
///    %v2 = "test.get_some_value"() : () -> i64
///    scf.condition(%condition) %v1 : i32
///  } do {
///  ^bb0(%arg0: i32):
///    "test.use"(%arg0) : (i32) -> ()
///    scf.yield
///  }
///  return %0 : i32
struct WhileUnusedResult : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = op.getConditionOp();
    auto afterArgs = op.getAfterArguments();
    auto termArgs = term.getArgs();

    // Collect results mapping, new terminator args and new result types.
    SmallVector<unsigned> newResultsIndices;
    SmallVector<Type> newResultTypes;
    SmallVector<Value> newTermArgs;
    SmallVector<Location> newArgLocs;
    bool needUpdate = false;
    for (const auto &it :
         llvm::enumerate(llvm::zip(op.getResults(), afterArgs, termArgs))) {
      auto i = static_cast<unsigned>(it.index());
      Value result = std::get<0>(it.value());
      Value afterArg = std::get<1>(it.value());
      Value termArg = std::get<2>(it.value());
      if (result.use_empty() && afterArg.use_empty()) {
        needUpdate = true;
      } else {
        newResultsIndices.emplace_back(i);
        newTermArgs.emplace_back(termArg);
        newResultTypes.emplace_back(result.getType());
        newArgLocs.emplace_back(result.getLoc());
      }
    }

    if (!needUpdate)
      return failure();

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                               newTermArgs);
    }

    auto newWhile =
        rewriter.create<WhileOp>(op.getLoc(), newResultTypes, op.getInits());

    Block &newAfterBlock = *rewriter.createBlock(
        &newWhile.getAfter(), /*insertPt*/ {}, newResultTypes, newArgLocs);

    // Build new results list and new after block args (unused entries will be
    // null).
    SmallVector<Value> newResults(op.getNumResults());
    SmallVector<Value> newAfterBlockArgs(op.getNumResults());
    for (const auto &it : llvm::enumerate(newResultsIndices)) {
      newResults[it.value()] = newWhile.getResult(it.index());
      newAfterBlockArgs[it.value()] = newAfterBlock.getArgument(it.index());
    }

    rewriter.inlineRegionBefore(op.getBefore(), newWhile.getBefore(),
                                newWhile.getBefore().begin());

    Block &afterBlock = op.getAfter().front();
    rewriter.mergeBlocks(&afterBlock, &newAfterBlock, newAfterBlockArgs);

    rewriter.replaceOp(op, newResults);
    return success();
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
    for (auto tup :
         llvm::zip(cond.getArgs(), op.getAfter().front().getArguments())) {
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

struct WhileUnusedArg : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {

    if (!llvm::any_of(op.getBeforeArguments(),
                      [](Value arg) { return arg.use_empty(); }))
      return failure();

    YieldOp yield = op.getYieldOp();

    // Collect results mapping, new terminator args and new result types.
    SmallVector<Value> newYields;
    SmallVector<Value> newInits;
    llvm::BitVector argsToErase(op.getBeforeArguments().size());
    for (const auto &it : llvm::enumerate(llvm::zip(
             op.getBeforeArguments(), yield.getOperands(), op.getInits()))) {
      Value beforeArg = std::get<0>(it.value());
      Value yieldValue = std::get<1>(it.value());
      Value initValue = std::get<2>(it.value());
      if (beforeArg.use_empty()) {
        argsToErase.set(it.index());
      } else {
        newYields.emplace_back(yieldValue);
        newInits.emplace_back(initValue);
      }
    }

    if (argsToErase.none())
      return failure();

    rewriter.startRootUpdate(op);
    op.getBefore().front().eraseArguments(argsToErase);
    rewriter.finalizeRootUpdate(op);

    WhileOp replacement =
        rewriter.create<WhileOp>(op.getLoc(), op.getResultTypes(), newInits);
    replacement.getBefore().takeBody(op.getBefore());
    replacement.getAfter().takeBody(op.getAfter());
    rewriter.replaceOp(op, replacement.getResults());

    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<YieldOp>(yield, newYields);
    return success();
  }
};
} // namespace

void WhileOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<RemoveLoopInvariantArgsFromBeforeBlock,
              RemoveLoopInvariantValueYielded, WhileConditionTruth,
              WhileCmpCond, WhileUnusedResult>(context);
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
    auto yield = cast<YieldOp>(region.front().getTerminator());
    if (yield.getNumOperands() != getNumResults()) {
      return (emitOpError("expected each region to return ")
              << getNumResults() << " values, but " << name << " returns "
              << yield.getNumOperands())
                 .attachNote(yield.getLoc())
             << "see yield operation here";
    }
    for (auto [idx, result, operand] :
         llvm::zip(llvm::seq<unsigned>(0, getNumResults()), getResultTypes(),
                   yield.getOperandTypes())) {
      if (result == operand)
        continue;
      return (emitOpError("expected result #")
              << idx << " of each region to be " << result)
                 .attachNote(yield.getLoc())
             << name << " returns " << operand << " here";
    }
    return success();
  };

  if (failed(verifyRegion(getDefaultRegion(), "default region")))
    return failure();
  for (auto &[idx, caseRegion] : llvm::enumerate(getCaseRegions()))
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
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &successors) {
  // All regions branch back to the parent op.
  if (index) {
    successors.emplace_back(getResults());
    return;
  }

  // If a constant was not provided, all regions are possible successors.
  auto operandValue = operands.front().dyn_cast_or_null<IntegerAttr>();
  if (!operandValue) {
    for (Region &caseRegion : getCaseRegions())
      successors.emplace_back(&caseRegion);
    successors.emplace_back(&getDefaultRegion());
    return;
  }

  // Otherwise, try to find a case with a matching value. If not, the default
  // region is the only successor.
  for (auto [caseValue, caseRegion] : llvm::zip(getCases(), getCaseRegions())) {
    if (caseValue == operandValue.getInt()) {
      successors.emplace_back(&caseRegion);
      return;
    }
  }
  successors.emplace_back(&getDefaultRegion());
}

void IndexSwitchOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands, SmallVectorImpl<InvocationBounds> &bounds) {
  auto operandValue = operands.front().dyn_cast_or_null<IntegerAttr>();
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

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/SCF/IR/SCFOps.cpp.inc"
