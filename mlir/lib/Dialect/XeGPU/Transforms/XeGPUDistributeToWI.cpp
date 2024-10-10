//===- XeGPUDistributeToWI.cpp - XeGPU ditribute SIMD to WI -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "xegpu-distribute-to-wi"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUDISTRIBUTETOWI
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-distribute-to-wi"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

namespace {
bool divisible(APInt lhs, APInt rhs) { return !lhs.urem(rhs); }

struct XeGPUDistributeToWIPass final
    : public xegpu::impl::XeGPUDistributeToWIBase<XeGPUDistributeToWIPass> {
  void runOnOperation() override;
};

struct FuncDistributor final : public OpRewritePattern<func::FuncOp> {
  FuncDistributor(MLIRContext *ctx, const unsigned subgroup_size = 16)
      : OpRewritePattern(ctx), subgroupSize(subgroup_size) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override;

private:
  unsigned subgroupSize;
};

struct DescriptorHoister final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

struct WarpOpStoreNd final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

struct WarpOpLoadNd final
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override;
};

// TODO: reuse from VectorDistribute.cpp
// Clones `op` into a new operation that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(RewriterBase &rewriter,
                                              Location loc, Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return rewriter.create(res);
}

/// Helper to create a new WarpExecuteOnLane0Op with different signature.
static vector::WarpExecuteOnLane0Op moveRegionToNewWarpOpAndReplaceReturns(
    RewriterBase &rewriter, vector::WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);
  auto newWarpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      warpOp.getLoc(), newReturnTypes, warpOp.getLaneid(), warpOp.getWarpSize(),
      warpOp.getArgs(), warpOp.getBody()->getArgumentTypes());

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  Block &newOpFirstBlock = newOpBody.front();
  rewriter.inlineRegionBefore(opBody, newOpBody, newOpBody.begin());
  rewriter.eraseBlock(&newOpFirstBlock);
  assert(newWarpOp.getWarpRegion().hasOneBlock() &&
         "expected WarpOp with single block");

  auto yield =
      cast<vector::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());

  rewriter.modifyOpInPlace(
      yield, [&]() { yield.getOperandsMutable().assign(newYieldedValues); });
  return newWarpOp;
}

vector::WarpExecuteOnLane0Op moveRegionToNewWarpOpAndReplaceInputs(
    RewriterBase &rewriter, vector::WarpExecuteOnLane0Op warpOp,
    ValueRange newArgumentValues, TypeRange newArgumentTypes) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);
  auto newWarpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      warpOp.getLoc(), warpOp.getResultTypes(), warpOp.getLaneid(),
      warpOp.getWarpSize(), newArgumentValues, newArgumentTypes);

  DBGS() << "New empty warp op: " << newWarpOp << "\n";

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  Block &newOpFirstBlock = newOpBody.front();
  rewriter.inlineRegionBefore(opBody, newOpBody, newOpBody.begin());
  rewriter.eraseBlock(&newOpFirstBlock);
  assert(newWarpOp.getWarpRegion().hasOneBlock() &&
         "expected WarpOp with single block");

  rewriter.modifyOpInPlace(newWarpOp, [&]() {
    newWarpOp.getBody()->eraseArguments(0,
                                        newWarpOp.getBody()->getNumArguments());
    newWarpOp.getBody()->addArguments(
        newArgumentTypes,
        SmallVector<Location>(newArgumentTypes.size(), newWarpOp.getLoc()));
    newWarpOp.getArgsMutable();
  });

  DBGS() << "New warp op with rewritten block arguments:\n"
         << newWarpOp << "\n";

  return newWarpOp;
}

vector::WarpExecuteOnLane0Op moveRegionToNewWarpOpAndRewriteInputs(
    RewriterBase &rewriter, vector::WarpExecuteOnLane0Op warpOp,
    ValueRange newInputValues, TypeRange newInputTypes,
    llvm::SmallVector<size_t> &indices) {
  SmallVector<Type> types(warpOp.getOperandTypes().begin(),
                          warpOp.getOperandTypes().end());
  llvm::SmallSetVector<Value, 32> args(warpOp.getOperands().begin(),
                                       warpOp.getOperands().end());
  for (auto newArg : llvm::zip(newInputValues, newInputTypes)) {
    if (args.insert(std::get<0>(newArg))) {
      types.push_back(std::get<1>(newArg));
      indices.push_back(args.size() - 1);
    } else {
      // If the value already exit the region don't create a new output.
      for (auto [idx, yieldOperand] : llvm::enumerate(args.getArrayRef())) {
        if (yieldOperand == std::get<0>(newArg)) {
          indices.push_back(idx);
          break;
        }
      }
    }
  }
  args.insert(newInputValues.begin(), newInputValues.end());
  DBGS() << "Creating a new warp op\nArgument types:\n" << types << "\n";
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndReplaceInputs(rewriter, warpOp,
                                            args.getArrayRef(), types);
  rewriter.replaceOp(warpOp, newWarpOp);
  return newWarpOp;
}
/// Helper to create a new WarpExecuteOnLane0Op region with extra outputs.
/// `indices` return the index of each new output.
vector::WarpExecuteOnLane0Op moveRegionToNewWarpOpAndAppendReturns(
    RewriterBase &rewriter, vector::WarpExecuteOnLane0Op warpOp,
    ValueRange newYieldedValues, TypeRange newReturnTypes,
    llvm::SmallVector<size_t> &indices) {
  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  llvm::SmallSetVector<Value, 32> yieldValues(yield.getOperands().begin(),
                                              yield.getOperands().end());
  for (auto newRet : llvm::zip(newYieldedValues, newReturnTypes)) {
    if (yieldValues.insert(std::get<0>(newRet))) {
      types.push_back(std::get<1>(newRet));
      indices.push_back(yieldValues.size() - 1);
    } else {
      // If the value already exit the region don't create a new output.
      for (auto [idx, yieldOperand] :
           llvm::enumerate(yieldValues.getArrayRef())) {
        if (yieldOperand == std::get<0>(newRet)) {
          indices.push_back(idx);
          break;
        }
      }
    }
  }
  yieldValues.insert(newYieldedValues.begin(), newYieldedValues.end());
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndReplaceReturns(rewriter, warpOp,
                                             yieldValues.getArrayRef(), types);
  rewriter.replaceOp(warpOp,
                     newWarpOp.getResults().take_front(warpOp.getNumResults()));
  return newWarpOp;
}

// TOUTILS
/// Return a value yielded by `warpOp` which statifies the filter lamdba
/// condition and is not dead.
static OpOperand *getWarpResult(vector::WarpExecuteOnLane0Op warpOp,
                                const std::function<bool(Operation *)> &fn) {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}
static std::pair<BlockArgument *, Operation *>
getWarpArgument(vector::WarpExecuteOnLane0Op warpOp,
                const std::function<bool(Operation *)> &fn) {
  for (BlockArgument &arg : warpOp.getBody()->getArguments()) {
    for (Operation *user : arg.getUsers()) {
      if (user && fn(user))
        return {&arg, user};
    }
  }
  return {nullptr, nullptr};
}
struct WarpOpElementwise
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(warpOp, [](Operation *op) {
      return OpTrait::hasElementwiseMappableTraits(op);
    });
    if (!yieldOperand)
      return failure();

    Operation *elementWise = yieldOperand->get().getDefiningOp();
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    Location loc = warpOp.getLoc();
    for (OpOperand &operand : elementWise->getOpOperands()) {
      Type targetType;
      if (auto vecType = dyn_cast<VectorType>(distributedVal.getType())) {
        // If the result type is a vector, the operands must also be vectors.
        auto operandType = cast<VectorType>(operand.get().getType());
        targetType =
            VectorType::get(vecType.getShape(), operandType.getElementType());
      } else {
        auto operandType = operand.get().getType();
        assert(!isa<VectorType>(operandType) &&
               "unexpected yield of vector from op with scalar result type");
        targetType = operandType;
      }
      retTypes.push_back(targetType);
      yieldValues.push_back(operand.get());
    }
    SmallVector<size_t> newRetIndices;
    vector::WarpExecuteOnLane0Op newWarpOp =
        moveRegionToNewWarpOpAndAppendReturns(rewriter, warpOp, yieldValues,
                                              retTypes, newRetIndices);
    rewriter.setInsertionPointAfter(newWarpOp);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : llvm::seq(unsigned(0), elementWise->getNumOperands())) {
      newOperands[i] = newWarpOp.getResult(newRetIndices[i]);
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, loc, elementWise, newOperands,
        {newWarpOp.getResult(operandIndex).getType()});
    rewriter.replaceAllUsesWith(newWarpOp.getResult(operandIndex),
                                newOp->getResult(0));
    return success();
  }
};
struct WarpOpConstant : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand =
        getWarpResult(warpOp, llvm::IsaPred<arith::ConstantOp>);
    if (!yieldOperand)
      return failure();
    auto constantOp = yieldOperand->get().getDefiningOp<arith::ConstantOp>();
    auto dense = dyn_cast<SplatElementsAttr>(constantOp.getValue());
    if (!dense)
      return failure();
    // Notify the rewriter that the warp op is changing (see the comment on
    // the WarpOpTransferRead pattern).
    rewriter.startOpModification(warpOp);
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Attribute scalarAttr = dense.getSplatValue<Attribute>();
    auto newAttr = DenseElementsAttr::get(
        cast<ShapedType>(warpOp.getResult(operandIndex).getType()), scalarAttr);
    Location loc = warpOp.getLoc();
    rewriter.setInsertionPointAfter(warpOp);
    Value distConstant = rewriter.create<arith::ConstantOp>(loc, newAttr);
    rewriter.replaceAllUsesWith(warpOp.getResult(operandIndex), distConstant);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};
struct WarpOpForwardOperand
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValues;
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Value valForwarded;
    unsigned resultIndex;
    for (OpOperand &operand : yield->getOpOperands()) {
      Value result = warpOp.getResult(operand.getOperandNumber());
      if (result.use_empty())
        continue;

      // Assume all the values coming from above are uniform.
      if (!warpOp.getBodyRegion().isAncestor(operand.get().getParentRegion())) {
        if (result.getType() != operand.get().getType())
          continue;
        valForwarded = operand.get();
        resultIndex = operand.getOperandNumber();
        break;
      }
      auto arg = dyn_cast<BlockArgument>(operand.get());
      if (!arg || arg.getOwner()->getParentOp() != warpOp.getOperation())
        continue;
      Value warpOperand = warpOp.getArgs()[arg.getArgNumber()];
      if (result.getType() != warpOperand.getType())
        continue;
      valForwarded = warpOperand;
      resultIndex = operand.getOperandNumber();
      break;
    }
    if (!valForwarded)
      return failure();
    // Notify the rewriter that the warp op is changing (see the comment on
    // the WarpOpTransferRead pattern).
    rewriter.startOpModification(warpOp);
    rewriter.replaceAllUsesWith(warpOp.getResult(resultIndex), valForwarded);
    rewriter.finalizeOpModification(warpOp);
    return success();
  }
};
struct WarpOpDeadResult
    : public OpRewritePattern<vector::WarpExecuteOnLane0Op> {
  using OpRewritePattern<vector::WarpExecuteOnLane0Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(warpOp->getNumResults());
    SmallVector<Value> newYieldValues;
    newYieldValues.reserve(warpOp->getNumResults());
    DenseMap<Value, int64_t> dedupYieldOperandPositionMap;
    DenseMap<OpResult, int64_t> dedupResultPositionMap;
    auto yield = cast<vector::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());

    // Some values may be yielded multiple times and correspond to multiple
    // results. Deduplicating occurs by taking each result with its matching
    // yielded value, and:
    //   1. recording the unique first position at which the value is yielded.
    //   2. recording for the result, the first position at which the dedup'ed
    //      value is yielded.
    //   3. skipping from the new result types / new yielded values any result
    //      that has no use or whose yielded value has already been seen.
    for (OpResult result : warpOp.getResults()) {
      Value yieldOperand = yield.getOperand(result.getResultNumber());
      auto it = dedupYieldOperandPositionMap.insert(
          std::make_pair(yieldOperand, newResultTypes.size()));
      dedupResultPositionMap.insert(std::make_pair(result, it.first->second));
      if (result.use_empty() || !it.second)
        continue;
      newResultTypes.push_back(result.getType());
      newYieldValues.push_back(yieldOperand);
    }
    // No modification, exit early.
    if (yield.getNumOperands() == newYieldValues.size())
      return failure();
    // Move the body of the old warpOp to a new warpOp.
    vector::WarpExecuteOnLane0Op newWarpOp =
        moveRegionToNewWarpOpAndReplaceReturns(rewriter, warpOp, newYieldValues,
                                               newResultTypes);

    // Simplify the new warp op after dropping dead results.
    newWarpOp.getBody()->walk([&](Operation *op) {
      if (isOpTriviallyDead(op))
        rewriter.eraseOp(op);
      else {
        auto load = dyn_cast<xegpu::LoadNdOp>(op);
        if (load) {
          DBGS() << "Found a non-dead op: " << load << "\n";
          DBGS() << "use_empty() = " << load->use_empty() << "\n";
          DBGS() << "is op trivially dead: " << isOpTriviallyDead(load) << "\n";
          SmallVector<Operation *, 1> effectingOps(1, load);
          while (!effectingOps.empty()) {
            Operation *op = effectingOps.pop_back_val();

            // If the operation has recursive effects, push all of the nested
            // operations on to the stack to consider.
            bool hasRecursiveEffects =
                op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
            if (hasRecursiveEffects) {
              for (Region &region : op->getRegions()) {
                for (auto &block : region) {
                  for (auto &nestedOp : block)
                    effectingOps.push_back(&nestedOp);
                }
              }
            }
            if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
              // Check to see if this op either has no effects, or only
              // allocates/reads memory.
              SmallVector<MemoryEffects::EffectInstance, 1> effects;
              effectInterface.getEffects(effects);
              DBGS() << "The load has memory effects\n";
              // Gather all results of this op that are allocated.
              SmallPtrSet<Value, 4> allocResults;
              for (const MemoryEffects::EffectInstance &it : effects)
                if (isa<MemoryEffects::Allocate>(it.getEffect()) &&
                    it.getValue() && it.getValue().getDefiningOp() == op)
                  allocResults.insert(it.getValue());

              DBGS() << "alloc results size: " << allocResults.size() << "\n";
              if (!llvm::all_of(effects, [&allocResults](
                                             const MemoryEffects::EffectInstance
                                                 &it) {
                    // We can drop effects if the value is an allocation and
                    // is a result of the operation.
                    if (allocResults.contains(it.getValue())) {
                      DBGS() << "we can drop effects since the value is an "
                                "allocation and is a result of the operation\n";
                      return false;
                    }
                    // Otherwise, the effect must be a read.
                    DBGS() << "The effect must be a read, it is: "
                           << isa<MemoryEffects::Read>(it.getEffect()) << "\n";
                    return isa<MemoryEffects::Read>(it.getEffect());
                  })) {
                DBGS() << "not all conditions met\n";
              }
            }
            if (hasRecursiveEffects)
              continue;

            // If there were no effect interfaces, we treat this op as
            // conservatively having effects.
            // return false;
            DBGS() << "Treating op as conservatively having effects\n";
            break;
          }
        }
      }
    });

    // Replace results of the old warpOp by the new, deduplicated results.
    SmallVector<Value> newValues;
    newValues.reserve(warpOp->getNumResults());
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        newValues.push_back(Value());
      else
        newValues.push_back(
            newWarpOp.getResult(dedupResultPositionMap.lookup(result)));
    }
    rewriter.replaceOp(warpOp, newValues);
    return success();
  }
};

} // namespace

LogicalResult
WarpOpStoreNd::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                               PatternRewriter &rewriter) const {
  auto yield = cast<vector::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  Operation *lastNode = yield->getPrevNode();
  auto storeOp = dyn_cast_or_null<xegpu::StoreNdOp>(lastNode);
  if (!storeOp)
    return failure();

  // 1. Compute distributed type.
  auto origType = storeOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        storeOp, "the source tensor descriptor lacks sg_map attribute");

  auto origShape = origType.getShape();
  if (origShape.size() != 2)
    return rewriter.notifyMatchFailure(storeOp, "unsupported shape");

  llvm::SmallVector<int64_t, 2> distributedShape;
  auto layout = sgMap.getWiLayout();
  auto inputVectorShape = storeOp.getValueType().getShape();

  for (const auto [l, o] : llvm::zip(layout, inputVectorShape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          storeOp,
          "the tensor dimentions are not divisible by the distribution factor");
    distributedShape.push_back(o / l);
  }

  DBGS() << "Matched store_nd: " << storeOp << "\n";

  auto storeValue = storeOp.getValue();
  auto newVectorType = mlir::VectorType::get(
      distributedShape, storeOp.getValueType().getElementType(),
      storeOp.getValueType().getScalableDims());

  SmallVector<size_t> newRetIndices;
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp,
          ValueRange{storeOp.getTensorDesc(), storeOp.getValue()},
          TypeRange{storeOp.getTensorDescType(), newVectorType}, newRetIndices);

  DBGS() << "Return indices:\n";
  llvm::interleaveComma(newRetIndices, DBGS());
  DBGS() << "End of indices:\n";

  rewriter.setInsertionPointAfter(newWarpOp);
  auto newStoreOp =
      cast<xegpu::StoreNdOp>(rewriter.clone(*storeOp.getOperation()));
  rewriter.eraseOp(storeOp);
  newStoreOp.getTensorDescMutable().assign(
      newWarpOp.getResult(newRetIndices[0]));
  newStoreOp.getValueMutable().assign(newWarpOp.getResult(newRetIndices[1]));

  DBGS() << "IR after store distribution:\n" << newWarpOp << "\n";
  DBGS() << "IR after store distribution:\n"
         << newWarpOp.getOperation()->getParentOfType<func::FuncOp>() << "\n";
  return success();
}

LogicalResult
DescriptorHoister::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                   PatternRewriter &rewriter) const {
  // This is unnecessary, greedily finding create_nd_desc op is enough
  auto [blockArg, descPtr] =
      getWarpArgument(warpOp, llvm::IsaPred<xegpu::CreateNdDescOp>);
  if (!blockArg)
    return failure();
  assert(descPtr && "Descriptor ptr must be non-null");
  auto desc = dyn_cast<xegpu::CreateNdDescOp>(descPtr);
  assert(desc && "Descriptor must be non-null");

  // If the descriptor points to the block argument, a hoisted descriptor
  // should take function argument as its operand
  auto funcOp = warpOp->getParentOfType<func::FuncOp>();
  BlockArgument argument = dyn_cast<BlockArgument>(desc.getSource());
  assert(argument && "desc source must be a block argument");
  unsigned warpBlockArgNum = argument.getArgNumber();

  DBGS() << "Operand: " << warpOp.getArgs()[argument.getArgNumber()] << "\n";
  auto funcArg =
      warpOp.getArgs()[argument.getArgNumber()].dyn_cast<BlockArgument>();
  assert(funcArg && "func arg must not be null");
  DBGS() << "Source for the argument at pos " << argument.getArgNumber()
         << " : " << warpOp.getArgs()[argument.getArgNumber()] << "\n";
  rewriter.setInsertionPoint(warpOp);

  auto srcTypedVal = dyn_cast<TypedValue<MemRefType>>(
      funcOp.getFunctionBody().getArgument(funcArg.getArgNumber()));

  // auto descOffsets = desc.getStaticOffsets();
  auto descOffsets = desc.getMixedOffsets();
  if (!descOffsets.size())
    rewriter.notifyMatchFailure(desc,
                                "non-static offsets currently not supported");
  auto laneid = warpOp.getLaneid();
  auto shiftX = rewriter.create<arith::AddIOp>(
      funcOp.getLoc(),

      getValueOrCreateConstantIndexOp(rewriter, funcOp.getLoc(),
                                      descOffsets[0]),
      laneid);
  auto shiftY = rewriter.create<arith::AddIOp>(
      funcOp.getLoc(),
      getValueOrCreateConstantIndexOp(rewriter, funcOp.getLoc(),
                                      descOffsets[1]),
      laneid);
  xegpu::SGMapAttr sgMap = desc.getType().getSGMapAttr();
  auto layout = sgMap.getWiLayout();
  auto dataSizes = sgMap.getWiData();
  auto viewShape = desc.getTensorDescShape();
  SmallVector<int64_t, 2> distributedShape;

  for (const auto [l, s] : llvm::zip(layout, viewShape)) {
    if (!divisible(APInt(64, s), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          desc,
          "the tensor dimentions are not divisible by the distribution factor");
    distributedShape.push_back(s / l);
  }

  auto srcType = srcTypedVal.getType();

  SmallVector<OpFoldResult> mixedOffsets = getAsOpFoldResult({shiftX, shiftY});

  // use the base memref strides
  SmallVector<OpFoldResult> overwriteStrides =
      getAsIndexOpFoldResult(rewriter.getContext(), SmallVector<int64_t>{1, 1});
  SmallVector<OpFoldResult> overwriteSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), distributedShape);

  auto subview = rewriter.create<memref::SubViewOp>(
      funcOp.getLoc(), srcTypedVal, mixedOffsets, overwriteSizes,
      overwriteStrides);

  auto distributedDescType = xegpu::TensorDescType::get(
      distributedShape, desc.getElementType(), /*array_lenght=*/1,
      /*boundary_check=*/true,
      static_cast<xegpu::MemorySpace>(desc.getSourceMemorySpace()),
      desc.getType().getSgMap());
  auto newDescOp = rewriter.create<xegpu::CreateNdDescOp>(
      desc.getLoc(), distributedDescType, subview.getResult(), mixedOffsets);

  rewriter.startOpModification(warpOp);
  rewriter.replaceAllUsesWith(desc, newDescOp);
  rewriter.eraseOp(desc);
  warpOp.getArgsMutable().erase(argument.getArgNumber());
  warpOp.getBody()->eraseArgument(argument.getArgNumber());
  DBGS() << "After replacing uses:\n" << funcOp << "\n";
  rewriter.finalizeOpModification(warpOp);
  DBGS() << "after inserts:\n" << funcOp << "---\n";
  return success();
}

LogicalResult WarpOpLoadNd::matchAndRewrite(vector::WarpExecuteOnLane0Op warpOp,
                                            PatternRewriter &rewriter) const {
  OpOperand *operand = getWarpResult(warpOp, [](Operation *op) {
    // Don't duplicate transfer_read ops when distributing.
    return isa<xegpu::LoadNdOp>(op) && op->hasOneUse();
  });

  if (!operand)
    return rewriter.notifyMatchFailure(warpOp,
                                       "warp result is not a xegpu::LoadNd op");
  auto loadOp = operand->get().getDefiningOp<xegpu::LoadNdOp>();
  DBGS() << "Found a suitable load for distribution: " << loadOp << "\n";

  if (!warpOp.isDefinedOutsideOfRegion(loadOp.getTensorDesc()))
    return rewriter.notifyMatchFailure(
        loadOp, "source must be defined outside of the region");

  xegpu::TensorDescType origType = loadOp.getTensorDescType();
  xegpu::SGMapAttr sgMap = origType.getSGMapAttr();
  if (!sgMap)
    return rewriter.notifyMatchFailure(
        loadOp, "the source tensor descriptor lacks sg_map attribute");

  auto origShape = origType.getShape();
  if (origShape.size() != 2)
    return rewriter.notifyMatchFailure(loadOp, "unsupported shape");

  llvm::SmallVector<int64_t, 2> distributedResultShape;
  auto layout = sgMap.getWiLayout();
  auto outputVectorShape = loadOp.getType().getShape();

  for (const auto [l, o, v] : llvm::zip(layout, origShape, outputVectorShape)) {
    if (!divisible(APInt(64, o), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          loadOp,
          "the tensor dimentions are not divisible by the distribution factor");
    if (!divisible(APInt(64, v), APInt(64, l)))
      return rewriter.notifyMatchFailure(
          loadOp, "the output vector dimentions are not divisible by the "
                  "distribution factor");
    distributedResultShape.push_back(v / l);
  }

  if (loadOp.getPacked())
    distributedResultShape.push_back(outputVectorShape.back());

  auto distributedVectorType = mlir::VectorType::get(
      distributedResultShape, loadOp.getType().getElementType(),
      loadOp.getType().getScalableDims());

  unsigned operandIdx = operand->getOperandNumber();

  SmallVector<size_t> newRetIndices;
  vector::WarpExecuteOnLane0Op newWarpOp =
      moveRegionToNewWarpOpAndAppendReturns(
          rewriter, warpOp, loadOp.getTensorDesc(), loadOp.getTensorDescType(),
          newRetIndices);

  DBGS() << "New warp op:\n" << newWarpOp << "\n";
  DBGS() << "Return indices:\n";
  llvm::interleaveComma(newRetIndices, DBGS());
  DBGS() << "\nEnd of indices:\n";

  rewriter.setInsertionPointAfter(newWarpOp);
  // auto newLoadOp =
  //     cast<xegpu::LoadNdOp>(rewriter.clone(*loadOp.getOperation()));
  auto newLoadOp = rewriter.create<xegpu::LoadNdOp>(
      loadOp.getLoc(), distributedVectorType, loadOp.getTensorDesc(),
      loadOp.getPackedAttr(), loadOp.getTransposeAttr(), loadOp.getL1HintAttr(),
      loadOp.getL2HintAttr(), loadOp.getL3HintAttr());

  DBGS() << "IR after load distribution:\n" << newWarpOp << "\n";
  DBGS() << "IR after load distribution:\n"
         << newWarpOp.getOperation()->getParentOfType<func::FuncOp>() << "\n";

  // this makes the load feeding into yield dead
  Value distributedVal = newWarpOp.getResult(operandIdx);
  rewriter.replaceAllUsesWith(distributedVal, newLoadOp);
  return success();
}

LogicalResult
FuncDistributor::matchAndRewrite(func::FuncOp funcOp,
                                 PatternRewriter &rewriter) const {
  if (funcOp.getName().starts_with("distributed_"))
    return failure();
  DBGS() << "Handling function name: " << funcOp.getName() << "\n";
  if (funcOp.getResultTypes().size() != 0)
    return failure();
  DBGS() << "Original function: " << funcOp << "\n";
  DBGS() << "End of original function:\n";
  std::string clonedFuncOpName = "distributed_" + funcOp.getName().str();
  auto clonedFuncOp = rewriter.create<func::FuncOp>(
      funcOp->getLoc(), clonedFuncOpName, funcOp.getFunctionType());
  SymbolTable::setSymbolVisibility(clonedFuncOp,
                                   SymbolTable::getSymbolVisibility(funcOp));

  assert(clonedFuncOp.getBlocks().size() == 0);
  Block *entry = clonedFuncOp.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);
  Location loc = clonedFuncOp.getLoc();

  auto laneId =
      rewriter.create<gpu::LaneIdOp>(loc, rewriter.getIndexAttr(subgroupSize));
  auto warpOp = rewriter.create<vector::WarpExecuteOnLane0Op>(
      loc, TypeRange(), laneId.getResult(), subgroupSize,
      clonedFuncOp.getArguments(), clonedFuncOp.getArgumentTypes());
  warpOp.getWarpRegion().takeBody(funcOp.getFunctionBody());
  Block &warpBlock = funcOp.getFunctionBody().emplaceBlock();
  rewriter.eraseOp(&warpOp.getWarpRegion().getBlocks().back().back());
  rewriter.setInsertionPointToEnd(&warpOp.getWarpRegion().getBlocks().back());
  rewriter.create<vector::YieldOp>(loc);

  rewriter.setInsertionPointToEnd(entry);
  auto ret = rewriter.create<func::ReturnOp>(loc);

  rewriter.replaceOp(funcOp, clonedFuncOp);
  DBGS() << "After replacing:\n" << clonedFuncOp << "\n";
  DBGS() << "End of replaced function:\n";
  return success();
}

void xegpu::populateXeGPUDistributeToWIPatterns(RewritePatternSet &patterns) {
  patterns.add<FuncDistributor>(patterns.getContext());
  patterns.add<DescriptorHoister>(patterns.getContext());
  patterns.add<WarpOpStoreNd>(patterns.getContext());
  patterns.add<WarpOpLoadNd>(patterns.getContext());

  patterns.add<WarpOpDeadResult>(patterns.getContext());
  patterns.add<WarpOpForwardOperand>(patterns.getContext());
  patterns.add<WarpOpElementwise>(patterns.getContext());
  patterns.add<WarpOpConstant>(patterns.getContext());
}

void XeGPUDistributeToWIPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  xegpu::populateXeGPUDistributeToWIPatterns(patterns);
  if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
          .failed()) {
    return signalPassFailure();
  }
}
