//===- AllReduceLowering.cpp - Implementation of all-reduce lowering ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements in-dialect lowering of the all-reduce op to a block of
// simpler instructions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

struct GpuAllReduceRewriter {
  using AccumulatorFactory = std::function<Value(Value, Value)>;

  GpuAllReduceRewriter(gpu::GPUFuncOp funcOp, gpu::AllReduceOp reduceOp,
                       PatternRewriter &rewriter)
      : funcOp(funcOp), reduceOp(reduceOp), rewriter(rewriter),
        loc(reduceOp.getLoc()), valueType(reduceOp.getValue().getType()),
        indexType(IndexType::get(reduceOp.getContext())),
        int32Type(IntegerType::get(reduceOp.getContext(), /*width=*/32)) {}

  /// Creates an all_reduce across the workgroup.
  ///
  /// First reduce the elements within a subgroup. The first invocation of each
  /// subgroup writes the intermediate result to workgroup memory. After
  /// synchronizing the workgroup, the first subgroup reduces the values from
  /// workgroup memory. The result is broadcasted to all invocations through
  /// workgroup memory.
  ///
  ///     %subgroup_reduce = `createSubgroupReduce(%operand)`
  ///     cf.cond_br %is_first_lane, ^then1, ^continue1
  ///   ^then1:
  ///     store %subgroup_reduce, %workgroup_buffer[%subgroup_id]
  ///     cf.br ^continue1
  ///   ^continue1:
  ///     gpu.barrier
  ///     %is_valid_subgroup = arith.cmpi "slt" %invocation_idx, %num_subgroups
  ///     cf.cond_br %is_valid_subgroup, ^then2, ^continue2
  ///   ^then2:
  ///     %partial_reduce = load %workgroup_buffer[%invocation_idx]
  ///     %all_reduce = `createSubgroupReduce(%partial_reduce)`
  ///     store %all_reduce, %workgroup_buffer[%zero]
  ///     llvm.br ^continue2
  ///   ^continue2:
  ///     gpu.barrier
  ///     %result = load %workgroup_buffer[%zero]
  ///     return %result
  ///
  void rewrite() {
    rewriter.setInsertionPoint(reduceOp);

    // Compute linear invocation index and workgroup size.
    Value dimX = getDimOp<gpu::BlockDimOp>(gpu::Dimension::x);
    Value dimY = getDimOp<gpu::BlockDimOp>(gpu::Dimension::y);
    Value dimZ = getDimOp<gpu::BlockDimOp>(gpu::Dimension::z);
    Value tidX = getDimOp<gpu::ThreadIdOp>(gpu::Dimension::x);
    Value tidY = getDimOp<gpu::ThreadIdOp>(gpu::Dimension::y);
    Value tidZ = getDimOp<gpu::ThreadIdOp>(gpu::Dimension::z);
    Value tmp1 = create<arith::MulIOp>(int32Type, tidZ, dimY);
    Value tmp2 = create<arith::AddIOp>(int32Type, tmp1, tidY);
    Value tmp3 = create<arith::MulIOp>(int32Type, tmp2, dimX);
    Value tmp4 = create<arith::MulIOp>(int32Type, dimX, dimY);
    Value invocationIdx = create<arith::AddIOp>(int32Type, tmp3, tidX);
    Value workgroupSize = create<arith::MulIOp>(int32Type, tmp4, dimZ);

    // Compute lane id (invocation id withing the subgroup).
    Value subgroupMask =
        create<arith::ConstantIntOp>(int32Type, kSubgroupSize - 1);
    Value laneId = create<arith::AndIOp>(invocationIdx, subgroupMask);
    Value isFirstLane =
        create<arith::CmpIOp>(arith::CmpIPredicate::eq, laneId,
                              create<arith::ConstantIntOp>(int32Type, 0));

    Value numThreadsWithSmallerSubgroupId =
        create<arith::SubIOp>(invocationIdx, laneId);
    // The number of active invocations starting from the current subgroup.
    // The consumers do not require the value to be clamped to the size of the
    // subgroup.
    Value activeWidth =
        create<arith::SubIOp>(workgroupSize, numThreadsWithSmallerSubgroupId);

    // Create factory for op which accumulates to values.
    AccumulatorFactory accumFactory = getFactory();
    assert(accumFactory && "failed to create accumulator factory");

    // Reduce elements within each subgroup to produce the intermediate results.
    Value subgroupReduce = createSubgroupReduce(
        activeWidth, laneId, reduceOp.getValue(), accumFactory);

    // Add workgroup buffer to parent function for intermediate result.
    Value buffer = createWorkgroupBuffer();

    // Write the intermediate results to workgroup memory, using the first lane
    // of each subgroup.
    createPredicatedBlock(isFirstLane, [&] {
      Value subgroupId = getDivideBySubgroupSize(invocationIdx);
      Value index = create<arith::IndexCastOp>(indexType, subgroupId);
      create<memref::StoreOp>(subgroupReduce, buffer, index);
    });
    create<gpu::BarrierOp>();

    // Compute number of active subgroups.
    Value biasedBlockSize =
        create<arith::AddIOp>(int32Type, workgroupSize, subgroupMask);
    Value numSubgroups = getDivideBySubgroupSize(biasedBlockSize);
    Value isValidSubgroup = create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                  invocationIdx, numSubgroups);

    // Use the first numSubgroups invocations to reduce the intermediate results
    // from workgroup memory. The final result is written to workgroup memory
    // again.
    Value zero = create<arith::ConstantIndexOp>(0);
    createPredicatedBlock(isValidSubgroup, [&] {
      Value index = create<arith::IndexCastOp>(indexType, invocationIdx);
      Value value = create<memref::LoadOp>(valueType, buffer, index);
      Value result =
          createSubgroupReduce(numSubgroups, laneId, value, accumFactory);
      create<memref::StoreOp>(result, buffer, zero);
    });

    // Synchronize workgroup and load result from workgroup memory.
    create<gpu::BarrierOp>();
    Value result = create<memref::LoadOp>(valueType, buffer, zero);

    rewriter.replaceOp(reduceOp, result);
  }

private:
  // Shortcut to create an op from rewriter using loc as the first argument.
  template <typename T, typename... Args>
  T create(Args... args) {
    return T::create(rewriter, loc, std::forward<Args>(args)...);
  }

  // Creates dimension op of type T, with the result casted to int32.
  template <typename T>
  Value getDimOp(gpu::Dimension dimension) {
    Value dim = create<T>(indexType, dimension);
    return create<arith::IndexCastOp>(int32Type, dim);
  }

  /// Adds type to funcOp's workgroup attributions.
  Value createWorkgroupBuffer() {
    // TODO: Pick a proper location for the attribution.
    auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
        funcOp->getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto bufferType = MemRefType::get({kSubgroupSize}, valueType, AffineMap{},
                                      workgroupMemoryAddressSpace);
    return funcOp.addWorkgroupAttribution(bufferType, rewriter.getUnknownLoc());
  }

  /// Returns an accumulator factory using either the op attribute or the body
  /// region.
  AccumulatorFactory getFactory() {
    auto &body = reduceOp.getBody();
    if (!body.empty())
      return getFactory(body);
    auto opAttr = reduceOp.getOp();
    if (opAttr)
      return getFactory(*opAttr);
    return AccumulatorFactory();
  }

  /// Returns an accumulator factory that clones the body. The body's entry
  /// block is expected to have 2 arguments. The gpu.yield return the
  /// accumulated value of the same type.
  AccumulatorFactory getFactory(Region &body) {
    return [&body, this](Value lhs, Value rhs) -> Value {
      Block *block = rewriter.getInsertionBlock();
      Block *split = rewriter.splitBlock(block, rewriter.getInsertionPoint());

      // Insert accumulator body between split block.
      IRMapping mapping;
      mapping.map(body.getArgument(0), lhs);
      mapping.map(body.getArgument(1), rhs);
      rewriter.cloneRegionBefore(body, *split->getParent(),
                                 split->getIterator(), mapping);

      // Add branch before inserted body, into body.
      block = block->getNextNode();
      create<cf::BranchOp>(block, ValueRange());

      // Replace all gpu.yield ops with branch out of body.
      for (; block != split; block = block->getNextNode()) {
        Operation *terminator = block->getTerminator();
        if (!isa<gpu::YieldOp>(terminator))
          continue;
        rewriter.setInsertionPointToEnd(block);
        rewriter.replaceOpWithNewOp<cf::BranchOp>(
            terminator, split, ValueRange(terminator->getOperand(0)));
      }

      // Return accumulator result.
      rewriter.setInsertionPointToStart(split);
      return split->addArgument(lhs.getType(), lhs.getLoc());
    };
  }

  /// Returns an accumulator factory that creates an op specified by opName.
  AccumulatorFactory getFactory(gpu::AllReduceOperation opName) {
    return [opName, this](Value lhs, Value rhs) {
      return vector::makeArithReduction(rewriter, loc,
                                        convertReductionKind(opName), lhs, rhs);
    };
  }

  /// Creates an if-block skeleton and calls the two factories to generate the
  /// ops in the `then` and `else` block..
  ///
  ///     llvm.cond_br %condition, ^then, ^continue
  ///   ^then:
  ///     %then_operands = `thenOpsFactory()`
  ///     llvm.br ^continue(%then_operands)
  ///   ^else:
  ///     %else_operands = `elseOpsFactory()`
  ///     llvm.br ^continue(%else_operands)
  ///   ^continue(%block_operands):
  ///
  template <typename ThenOpsFactory, typename ElseOpsFactory>
  void createIf(Value condition, ThenOpsFactory &&thenOpsFactory,
                ElseOpsFactory &&elseOpsFactory) {
    Block *currentBlock = rewriter.getInsertionBlock();
    auto currentPoint = rewriter.getInsertionPoint();

    Block *thenBlock = rewriter.splitBlock(currentBlock, currentPoint);
    Block *elseBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());
    Block *continueBlock = rewriter.splitBlock(elseBlock, elseBlock->begin());

    rewriter.setInsertionPointToEnd(currentBlock);
    create<cf::CondBranchOp>(condition, thenBlock,
                             /*trueOperands=*/ArrayRef<Value>(), elseBlock,
                             /*falseOperands=*/ArrayRef<Value>());

    rewriter.setInsertionPointToStart(thenBlock);
    auto thenOperands = thenOpsFactory();
    create<cf::BranchOp>(continueBlock, thenOperands);

    rewriter.setInsertionPointToStart(elseBlock);
    auto elseOperands = elseOpsFactory();
    create<cf::BranchOp>(continueBlock, elseOperands);

    assert(thenOperands.size() == elseOperands.size());
    rewriter.setInsertionPointToStart(continueBlock);
    for (auto operand : thenOperands)
      continueBlock->addArgument(operand.getType(), operand.getLoc());
  }

  /// Shortcut for createIf with empty else block and no block operands.
  template <typename Factory>
  void createPredicatedBlock(Value condition, Factory &&predicatedOpsFactory) {
    static_assert(std::is_same<decltype(predicatedOpsFactory()), void>::value,
                  "predicatedOpsFactory should not return any value");
    createIf(
        condition,
        [&] {
          predicatedOpsFactory();
          return ArrayRef<Value>();
        },
        [&] { return ArrayRef<Value>(); });
  }

  /// Creates a reduction across the first activeWidth lanes of a subgroup, or
  /// the entire subgroup if activeWidth is larger than the subgroup width.
  /// The first lane returns the result, all others return values are undefined.
  Value createSubgroupReduce(Value activeWidth, Value laneId, Value operand,
                             AccumulatorFactory &accumFactory) {
    Value subgroupSize = create<arith::ConstantIntOp>(int32Type, kSubgroupSize);
    Value isPartialSubgroup = create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                    activeWidth, subgroupSize);
    std::array<Type, 2> shuffleType = {valueType, rewriter.getI1Type()};

    createIf(
        isPartialSubgroup,
        // Generate reduction over a (potentially) partial subgroup.
        [&] {
          Value value = operand;
          // Repeatedly shuffle value from 'laneId ^ i' and accumulate if source
          // lane is within the active range. The accumulated value is available
          // in the first lane.
          for (int i = 1; i < kSubgroupSize; i <<= 1) {
            Value offset = create<arith::ConstantIntOp>(int32Type, i);
            auto shuffleOp = create<gpu::ShuffleOp>(
                shuffleType, value, offset, activeWidth, gpu::ShuffleMode::XOR);
            // Skip the accumulation if the shuffle op read from a lane outside
            // of the active range.
            createIf(
                shuffleOp.getResult(1),
                [&] {
                  return SmallVector<Value, 1>{
                      accumFactory(value, shuffleOp.getResult(0))};
                },
                [&] { return llvm::ArrayRef(value); });
            value = rewriter.getInsertionBlock()->getArgument(0);
          }
          return SmallVector<Value, 1>{value};
        },
        // Generate a reduction over the entire subgroup. This is a
        // specialization of the above reduction with unconditional
        // accumulation.
        [&] {
          Value value = operand;
          for (int i = 1; i < kSubgroupSize; i <<= 1) {
            Value offset = create<arith::ConstantIntOp>(int32Type, i);
            auto shuffleOp =
                create<gpu::ShuffleOp>(shuffleType, value, offset, subgroupSize,
                                       gpu::ShuffleMode::XOR);
            value = accumFactory(value, shuffleOp.getResult(0));
          }
          return SmallVector<Value, 1>{value};
        });
    return rewriter.getInsertionBlock()->getArgument(0);
  }

  /// Returns value divided by the subgroup size (i.e. 32).
  Value getDivideBySubgroupSize(Value value) {
    Value subgroupSize = create<arith::ConstantIntOp>(int32Type, kSubgroupSize);
    return create<arith::DivSIOp>(int32Type, value, subgroupSize);
  }

  gpu::GPUFuncOp funcOp;
  gpu::AllReduceOp reduceOp;
  PatternRewriter &rewriter;

  Location loc;
  Type valueType;
  Type indexType;
  IntegerType int32Type;

  static constexpr int kSubgroupSize = 32;
};

struct GpuAllReduceRewrite : public RewritePattern {
  explicit GpuAllReduceRewrite(MLIRContext *context)
      : RewritePattern(gpu::GPUFuncOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = cast<gpu::GPUFuncOp>(op);

    SmallVector<gpu::AllReduceOp> reduceOps;
    auto callback = [&](gpu::AllReduceOp reduceOp) -> WalkResult {
      if (!reduceOp.getUniform())
        return WalkResult::interrupt();

      reduceOps.emplace_back(reduceOp);
      return WalkResult::advance();
    };

    if (funcOp.walk(callback).wasInterrupted() || reduceOps.empty())
      return rewriter.notifyMatchFailure(
          op, "Non uniform reductions are not supported yet.");

    for (gpu::AllReduceOp reduceOp : reduceOps)
      GpuAllReduceRewriter(funcOp, reduceOp, rewriter).rewrite();

    return success();
  }
};
} // namespace

void mlir::populateGpuAllReducePatterns(RewritePatternSet &patterns) {
  patterns.add<GpuAllReduceRewrite>(patterns.getContext());
}
