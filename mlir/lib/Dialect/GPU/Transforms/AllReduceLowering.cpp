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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct GpuAllReduceRewriter {
  using AccumulatorFactory = std::function<Value(Value, Value)>;

  GpuAllReduceRewriter(gpu::GPUFuncOp funcOp_, gpu::AllReduceOp reduceOp_,
                       PatternRewriter &rewriter_)
      : funcOp(funcOp_), reduceOp(reduceOp_), rewriter(rewriter_),
        loc(reduceOp.getLoc()), valueType(reduceOp.value().getType()),
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
  ///     cond_br %is_first_lane, ^then1, ^continue1
  ///   ^then1:
  ///     store %subgroup_reduce, %workgroup_buffer[%subgroup_id]
  ///     br ^continue1
  ///   ^continue1:
  ///     gpu.barrier
  ///     %is_valid_subgroup = arith.cmpi "slt" %invocation_idx, %num_subgroups
  ///     cond_br %is_valid_subgroup, ^then2, ^continue2
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
    Value dimX = getDimOp<gpu::BlockDimOp>("x");
    Value dimY = getDimOp<gpu::BlockDimOp>("y");
    Value dimZ = getDimOp<gpu::BlockDimOp>("z");
    Value tidX = getDimOp<gpu::ThreadIdOp>("x");
    Value tidY = getDimOp<gpu::ThreadIdOp>("y");
    Value tidZ = getDimOp<gpu::ThreadIdOp>("z");
    Value tmp1 = create<arith::MulIOp>(int32Type, tidZ, dimY);
    Value tmp2 = create<arith::AddIOp>(int32Type, tmp1, tidY);
    Value tmp3 = create<arith::MulIOp>(int32Type, tmp2, dimX);
    Value tmp4 = create<arith::MulIOp>(int32Type, dimX, dimY);
    Value invocationIdx = create<arith::AddIOp>(int32Type, tmp3, tidX);
    Value workgroupSize = create<arith::MulIOp>(int32Type, tmp4, dimZ);

    // Compute lane id (invocation id withing the subgroup).
    Value subgroupMask =
        create<arith::ConstantIntOp>(kSubgroupSize - 1, int32Type);
    Value laneId = create<arith::AndIOp>(invocationIdx, subgroupMask);
    Value isFirstLane =
        create<arith::CmpIOp>(arith::CmpIPredicate::eq, laneId,
                              create<arith::ConstantIntOp>(0, int32Type));

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
    Value subgroupReduce = createSubgroupReduce(activeWidth, laneId,
                                                reduceOp.value(), accumFactory);

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
    return rewriter.create<T>(loc, std::forward<Args>(args)...);
  }

  // Creates dimension op of type T, with the result casted to int32.
  template <typename T>
  Value getDimOp(StringRef dimension) {
    Value dim = create<T>(indexType, rewriter.getStringAttr(dimension));
    return create<arith::IndexCastOp>(int32Type, dim);
  }

  /// Adds type to funcOp's workgroup attributions.
  Value createWorkgroupBuffer() {
    int workgroupMemoryAddressSpace =
        gpu::GPUDialect::getWorkgroupAddressSpace();
    auto bufferType =
        MemRefType::get({kSubgroupSize}, valueType, ArrayRef<AffineMap>{},
                        workgroupMemoryAddressSpace);
    return funcOp.addWorkgroupAttribution(bufferType);
  }

  /// Returns an accumulator factory using either the op attribute or the body
  /// region.
  AccumulatorFactory getFactory() {
    auto &body = reduceOp.body();
    if (!body.empty())
      return getFactory(body);
    auto opAttr = reduceOp.op();
    if (opAttr)
      return getFactory(*opAttr);
    return AccumulatorFactory();
  }

  /// Returns an accumulator factory that clones the body. The body's entry
  /// block is expected to have 2 arguments. The gpu.yield return the
  /// accumulated value of the same type.
  AccumulatorFactory getFactory(Region &body) {
    return AccumulatorFactory([&](Value lhs, Value rhs) {
      Block *block = rewriter.getInsertionBlock();
      Block *split = rewriter.splitBlock(block, rewriter.getInsertionPoint());

      // Insert accumulator body between split block.
      BlockAndValueMapping mapping;
      mapping.map(body.getArgument(0), lhs);
      mapping.map(body.getArgument(1), rhs);
      rewriter.cloneRegionBefore(body, *split->getParent(),
                                 split->getIterator(), mapping);

      // Add branch before inserted body, into body.
      block = block->getNextNode();
      create<BranchOp>(block, ValueRange());

      // Replace all gpu.yield ops with branch out of body.
      for (; block != split; block = block->getNextNode()) {
        Operation *terminator = block->getTerminator();
        if (!isa<gpu::YieldOp>(terminator))
          continue;
        rewriter.setInsertionPointToEnd(block);
        rewriter.replaceOpWithNewOp<BranchOp>(
            terminator, split, ValueRange(terminator->getOperand(0)));
      }

      // Return accumulator result.
      rewriter.setInsertionPointToStart(split);
      return split->addArgument(lhs.getType());
    });
  }

  /// Returns an accumulator factory that creates an op specified by opName.
  AccumulatorFactory getFactory(StringRef opName) {
    bool isFloatingPoint = valueType.isa<FloatType>();
    if (opName == "add")
      return isFloatingPoint ? getFactory<arith::AddFOp>()
                             : getFactory<arith::AddIOp>();
    if (opName == "mul")
      return isFloatingPoint ? getFactory<arith::MulFOp>()
                             : getFactory<arith::MulIOp>();
    if (opName == "and") {
      return getFactory<arith::AndIOp>();
    }
    if (opName == "or") {
      return getFactory<arith::OrIOp>();
    }
    if (opName == "xor") {
      return getFactory<arith::XOrIOp>();
    }
    if (opName == "max") {
      return isFloatingPoint
                 ? getCmpFactory<arith::CmpFOp, arith::CmpFPredicate,
                                 arith::CmpFPredicate::UGT>()
                 : getCmpFactory<arith::CmpIOp, arith::CmpIPredicate,
                                 arith::CmpIPredicate::ugt>();
    }
    if (opName == "min") {
      return isFloatingPoint
                 ? getCmpFactory<arith::CmpFOp, arith::CmpFPredicate,
                                 arith::CmpFPredicate::ULT>()
                 : getCmpFactory<arith::CmpIOp, arith::CmpIPredicate,
                                 arith::CmpIPredicate::ult>();
    }
    return AccumulatorFactory();
  }

  /// Returns an accumulator factory that creates an op of type T.
  template <typename T>
  AccumulatorFactory getFactory() {
    return [&](Value lhs, Value rhs) {
      return create<T>(lhs.getType(), lhs, rhs);
    };
  }

  /// Returns an accumulator for comparison such as min, max. T is the type
  /// of the compare op.
  template <typename T, typename PredicateEnum, PredicateEnum predicate>
  AccumulatorFactory getCmpFactory() const {
    return [&](Value lhs, Value rhs) {
      Value cmp = rewriter.create<T>(loc, predicate, lhs, rhs);
      return rewriter.create<SelectOp>(loc, cmp, lhs, rhs);
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
    create<CondBranchOp>(condition, thenBlock,
                         /*trueOperands=*/ArrayRef<Value>(), elseBlock,
                         /*falseOperands=*/ArrayRef<Value>());

    rewriter.setInsertionPointToStart(thenBlock);
    auto thenOperands = thenOpsFactory();
    create<BranchOp>(continueBlock, thenOperands);

    rewriter.setInsertionPointToStart(elseBlock);
    auto elseOperands = elseOpsFactory();
    create<BranchOp>(continueBlock, elseOperands);

    assert(thenOperands.size() == elseOperands.size());
    rewriter.setInsertionPointToStart(continueBlock);
    for (auto operand : thenOperands)
      continueBlock->addArgument(operand.getType());
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
    Value subgroupSize = create<arith::ConstantIntOp>(kSubgroupSize, int32Type);
    Value isPartialSubgroup = create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                    activeWidth, subgroupSize);
    std::array<Type, 2> shuffleType = {valueType, rewriter.getI1Type()};
    auto xorAttr = rewriter.getStringAttr("xor");

    createIf(
        isPartialSubgroup,
        // Generate reduction over a (potentially) partial subgroup.
        [&] {
          Value value = operand;
          // Repeatedly shuffle value from 'laneId ^ i' and accumulate if source
          // lane is within the active range. The accumulated value is available
          // in the first lane.
          for (int i = 1; i < kSubgroupSize; i <<= 1) {
            Value offset = create<arith::ConstantIntOp>(i, int32Type);
            auto shuffleOp = create<gpu::ShuffleOp>(shuffleType, value, offset,
                                                    activeWidth, xorAttr);
            // Skip the accumulation if the shuffle op read from a lane outside
            // of the active range.
            createIf(
                shuffleOp.getResult(1),
                [&] {
                  return SmallVector<Value, 1>{
                      accumFactory(value, shuffleOp.getResult(0))};
                },
                [&] { return llvm::makeArrayRef(value); });
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
            Value offset = create<arith::ConstantIntOp>(i, int32Type);
            auto shuffleOp = create<gpu::ShuffleOp>(shuffleType, value, offset,
                                                    subgroupSize, xorAttr);
            value = accumFactory(value, shuffleOp.getResult(0));
          }
          return SmallVector<Value, 1>{value};
        });
    return rewriter.getInsertionBlock()->getArgument(0);
  }

  /// Returns value divided by the subgroup size (i.e. 32).
  Value getDivideBySubgroupSize(Value value) {
    Value subgroupSize = create<arith::ConstantIntOp>(kSubgroupSize, int32Type);
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

struct GpuAllReduceConversion : public RewritePattern {
  explicit GpuAllReduceConversion(MLIRContext *context)
      : RewritePattern(gpu::GPUFuncOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = cast<gpu::GPUFuncOp>(op);
    auto callback = [&](gpu::AllReduceOp reduceOp) {
      GpuAllReduceRewriter(funcOp, reduceOp, rewriter).rewrite();
      // Performing a rewrite invalidates the walk iterator. Report interrupt
      // so that we can start a new walk until all all_reduce ops are replaced.
      return WalkResult::interrupt();
    };
    while (funcOp.walk(callback).wasInterrupted()) {
    }
    return success();
  }
};
} // namespace

void mlir::populateGpuAllReducePatterns(RewritePatternSet &patterns) {
  patterns.add<GpuAllReduceConversion>(patterns.getContext());
}
