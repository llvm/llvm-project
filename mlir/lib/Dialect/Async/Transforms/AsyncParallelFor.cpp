//===- AsyncParallelFor.cpp - Implementation of Async Parallel For --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements scf.parallel to src.for + async.execute conversion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-parallel-for"

namespace {

// Rewrite scf.parallel operation into multiple concurrent async.execute
// operations over non overlapping subranges of the original loop.
//
// Example:
//
//   scf.for (%i, %j) = (%lbi, %lbj) to (%ubi, %ubj) step (%si, %sj) {
//     "do_some_compute"(%i, %j): () -> ()
//   }
//
// Converted to:
//
//   %c0 = constant 0 : index
//   %c1 = constant 1 : index
//
//   // Compute blocks sizes for each induction variable.
//   %num_blocks_i = ... : index
//   %num_blocks_j = ... : index
//   %block_size_i = ... : index
//   %block_size_j = ... : index
//
//   // Create an async group to track async execute ops.
//   %group = async.create_group
//
//   scf.for %bi = %c0 to %num_blocks_i step %c1 {
//     %block_start_i = ... : index
//     %block_end_i   = ... : index
//
//     scf.for %bj = %c0 to %num_blocks_j step %c1 {
//       %block_start_j = ... : index
//       %block_end_j   = ... : index
//
//       // Execute the body of original parallel operation for the current
//       // block.
//       %token = async.execute {
//         scf.for %i = %block_start_i to %block_end_i step %si {
//           scf.for %j = %block_start_j to %block_end_j step %sj {
//             "do_some_compute"(%i, %j): () -> ()
//           }
//         }
//       }
//
//       // Add produced async token to the group.
//       async.add_to_group %token, %group
//     }
//   }
//
//   // Await completion of all async.execute operations.
//   async.await_all %group
//
// In this example outer loop launches inner block level loops as separate async
// execute operations which will be executed concurrently.
//
// At the end it waits for the completiom of all async execute operations.
//
struct AsyncParallelForRewrite : public OpRewritePattern<scf::ParallelOp> {
public:
  AsyncParallelForRewrite(MLIRContext *ctx, int numConcurrentAsyncExecute)
      : OpRewritePattern(ctx),
        numConcurrentAsyncExecute(numConcurrentAsyncExecute) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override;

private:
  int numConcurrentAsyncExecute;
};

struct AsyncParallelForPass
    : public AsyncParallelForBase<AsyncParallelForPass> {
  AsyncParallelForPass() = default;
  AsyncParallelForPass(int numWorkerThreads) {
    assert(numWorkerThreads >= 1);
    numConcurrentAsyncExecute = numWorkerThreads;
  }
  void runOnFunction() override;
};

} // namespace

LogicalResult
AsyncParallelForRewrite::matchAndRewrite(scf::ParallelOp op,
                                         PatternRewriter &rewriter) const {
  // We do not currently support rewrite for parallel op with reductions.
  if (op.getNumReductions() != 0)
    return failure();

  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();

  // Index constants used below.
  auto indexTy = IndexType::get(ctx);
  auto zero = IntegerAttr::get(indexTy, 0);
  auto one = IntegerAttr::get(indexTy, 1);
  auto c0 = rewriter.create<ConstantOp>(loc, indexTy, zero);
  auto c1 = rewriter.create<ConstantOp>(loc, indexTy, one);

  // Shorthand for signed integer ceil division operation.
  auto divup = [&](Value x, Value y) -> Value {
    return rewriter.create<SignedCeilDivIOp>(loc, x, y);
  };

  // Compute trip count for each loop induction variable:
  //   tripCount = divUp(upperBound - lowerBound, step);
  SmallVector<Value, 4> tripCounts(op.getNumLoops());
  for (size_t i = 0; i < op.getNumLoops(); ++i) {
    auto lb = op.lowerBound()[i];
    auto ub = op.upperBound()[i];
    auto step = op.step()[i];
    auto range = rewriter.create<SubIOp>(loc, ub, lb);
    tripCounts[i] = divup(range, step);
  }

  // The target number of concurrent async.execute ops.
  auto numExecuteOps = rewriter.create<ConstantOp>(
      loc, indexTy, IntegerAttr::get(indexTy, numConcurrentAsyncExecute));

  // Blocks sizes configuration for each induction variable.

  // We try to use maximum available concurrency in outer dimensions first
  // (assuming that parallel induction variables are corresponding to some
  // multidimensional access, e.g. in (%d0, %d1, ..., %dn) = (<from>) to (<to>)
  // we will try to parallelize iteration along the %d0. If %d0 is too small,
  // we'll parallelize iteration over %d1, and so on.
  SmallVector<Value, 4> targetNumBlocks(op.getNumLoops());
  SmallVector<Value, 4> blockSize(op.getNumLoops());
  SmallVector<Value, 4> numBlocks(op.getNumLoops());

  // Compute block size and number of blocks along the first induction variable.
  targetNumBlocks[0] = numExecuteOps;
  blockSize[0] = divup(tripCounts[0], targetNumBlocks[0]);
  numBlocks[0] = divup(tripCounts[0], blockSize[0]);

  // Assign remaining available concurrency to other induction variables.
  for (size_t i = 1; i < op.getNumLoops(); ++i) {
    targetNumBlocks[i] = divup(targetNumBlocks[i - 1], numBlocks[i - 1]);
    blockSize[i] = divup(tripCounts[i], targetNumBlocks[i]);
    numBlocks[i] = divup(tripCounts[i], blockSize[i]);
  }

  // Create an async.group to wait on all async tokens from async execute ops.
  auto group = rewriter.create<CreateGroupOp>(loc, GroupType::get(ctx));

  // Build a scf.for loop nest from the parallel operation.

  // Lower/upper bounds for nest block level computations.
  SmallVector<Value, 4> blockLowerBounds(op.getNumLoops());
  SmallVector<Value, 4> blockUpperBounds(op.getNumLoops());
  SmallVector<Value, 4> blockInductionVars(op.getNumLoops());

  using LoopBodyBuilder =
      std::function<void(OpBuilder &, Location, Value, ValueRange)>;
  using LoopBuilder = std::function<LoopBodyBuilder(size_t loopIdx)>;

  // Builds inner loop nest inside async.execute operation that does all the
  // work concurrently.
  LoopBuilder workLoopBuilder = [&](size_t loopIdx) -> LoopBodyBuilder {
    return [&, loopIdx](OpBuilder &b, Location loc, Value iv, ValueRange args) {
      blockInductionVars[loopIdx] = iv;

      // Continue building async loop nest.
      if (loopIdx < op.getNumLoops() - 1) {
        b.create<scf::ForOp>(
            loc, blockLowerBounds[loopIdx + 1], blockUpperBounds[loopIdx + 1],
            op.step()[loopIdx + 1], ValueRange(), workLoopBuilder(loopIdx + 1));
        b.create<scf::YieldOp>(loc);
        return;
      }

      // Copy the body of the parallel op with new loop bounds.
      BlockAndValueMapping mapping;
      mapping.map(op.getInductionVars(), blockInductionVars);

      for (auto &bodyOp : op.getLoopBody().getOps())
        b.clone(bodyOp, mapping);
    };
  };

  // Builds a loop nest that does async execute op dispatching.
  LoopBuilder asyncLoopBuilder = [&](size_t loopIdx) -> LoopBodyBuilder {
    return [&, loopIdx](OpBuilder &b, Location loc, Value iv, ValueRange args) {
      auto lb = op.lowerBound()[loopIdx];
      auto ub = op.upperBound()[loopIdx];
      auto step = op.step()[loopIdx];

      // Compute lower bound for the current block:
      //   blockLowerBound = iv * blockSize * step + lowerBound
      auto s0 = b.create<MulIOp>(loc, iv, blockSize[loopIdx]);
      auto s1 = b.create<MulIOp>(loc, s0, step);
      auto s2 = b.create<AddIOp>(loc, s1, lb);
      blockLowerBounds[loopIdx] = s2;

      // Compute upper bound for the current block:
      //   blockUpperBound = min(upperBound,
      //                         blockLowerBound + blockSize * step)
      auto e0 = b.create<MulIOp>(loc, blockSize[loopIdx], step);
      auto e1 = b.create<AddIOp>(loc, e0, s2);
      auto e2 = b.create<CmpIOp>(loc, CmpIPredicate::slt, e1, ub);
      auto e3 = b.create<SelectOp>(loc, e2, e1, ub);
      blockUpperBounds[loopIdx] = e3;

      // Continue building async dispatch loop nest.
      if (loopIdx < op.getNumLoops() - 1) {
        b.create<scf::ForOp>(loc, c0, numBlocks[loopIdx + 1], c1, ValueRange(),
                             asyncLoopBuilder(loopIdx + 1));
        b.create<scf::YieldOp>(loc);
        return;
      }

      // Build the inner loop nest that will do the actual work inside the
      // `async.execute` body region.
      auto executeBodyBuilder = [&](OpBuilder &executeBuilder,
                                    Location executeLoc,
                                    ValueRange executeArgs) {
        executeBuilder.create<scf::ForOp>(executeLoc, blockLowerBounds[0],
                                          blockUpperBounds[0], op.step()[0],
                                          ValueRange(), workLoopBuilder(0));
        executeBuilder.create<async::YieldOp>(executeLoc, ValueRange());
      };

      auto execute = b.create<ExecuteOp>(
          loc, /*resultTypes=*/TypeRange(), /*dependencies=*/ValueRange(),
          /*operands=*/ValueRange(), executeBodyBuilder);
      auto rankType = IndexType::get(ctx);
      b.create<AddToGroupOp>(loc, rankType, execute.token(), group.result());
      b.create<scf::YieldOp>(loc);
    };
  };

  // Start building a loop nest from the first induction variable.
  rewriter.create<scf::ForOp>(loc, c0, numBlocks[0], c1, ValueRange(),
                              asyncLoopBuilder(0));

  // Wait for the completion of all subtasks.
  rewriter.create<AwaitAllOp>(loc, group.result());

  // Erase the original parallel operation.
  rewriter.eraseOp(op);

  return success();
}

void AsyncParallelForPass::runOnFunction() {
  MLIRContext *ctx = &getContext();

  OwningRewritePatternList patterns(ctx);
  patterns.insert<AsyncParallelForRewrite>(ctx, numConcurrentAsyncExecute);

  if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAsyncParallelForPass() {
  return std::make_unique<AsyncParallelForPass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createAsyncParallelForPass(int numWorkerThreads) {
  return std::make_unique<AsyncParallelForPass>(numWorkerThreads);
}
