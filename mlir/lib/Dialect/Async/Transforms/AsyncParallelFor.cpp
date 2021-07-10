//===- AsyncParallelFor.cpp - Implementation of Async Parallel For --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements scf.parallel to scf.for + async.execute conversion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::async;

#define DEBUG_TYPE "async-parallel-for"

namespace {

// Rewrite scf.parallel operation into multiple concurrent async.execute
// operations over non overlapping subranges of the original loop.
//
// Example:
//
//   scf.parallel (%i, %j) = (%lbi, %lbj) to (%ubi, %ubj) step (%si, %sj) {
//     "do_some_compute"(%i, %j): () -> ()
//   }
//
// Converted to:
//
//   // Parallel compute function that executes the parallel body region for
//   // a subset of the parallel iteration space defined by the one-dimensional
//   // compute block index.
//   func parallel_compute_function(%block_index : index, %block_size : index,
//                                  <parallel operation properties>, ...) {
//     // Compute multi-dimensional loop bounds for %block_index.
//     %block_lbi, %block_lbj = ...
//     %block_ubi, %block_ubj = ...
//
//     // Clone parallel operation body into the scf.for loop nest.
//     scf.for %i = %blockLbi to %blockUbi {
//       scf.for %j = block_lbj to %block_ubj {
//         "do_some_compute"(%i, %j): () -> ()
//       }
//     }
//   }
//
// And a dispatch function depending on the `asyncDispatch` option.
//
// When async dispatch is on: (pseudocode)
//
//   %block_size = ... compute parallel compute block size
//   %block_count = ... compute the number of compute blocks
//
//   func @async_dispatch(%block_start : index, %block_end : index, ...) {
//     // Keep splitting block range until we reached a range of size 1.
//     while (%block_end - %block_start > 1) {
//       %mid_index = block_start + (block_end - block_start) / 2;
//       async.execute { call @async_dispatch(%mid_index, %block_end); }
//       %block_end = %mid_index
//     }
//
//     // Call parallel compute function for a single block.
//     call @parallel_compute_fn(%block_start, %block_size, ...);
//   }
//
//   // Launch async dispatch for [0, block_count) range.
//   call @async_dispatch(%c0, %block_count);
//
// When async dispatch is off:
//
//   %block_size = ... compute parallel compute block size
//   %block_count = ... compute the number of compute blocks
//
//   scf.for %block_index = %c0 to %block_count {
//      call @parallel_compute_fn(%block_index, %block_size, ...)
//   }
//
struct AsyncParallelForPass
    : public AsyncParallelForBase<AsyncParallelForPass> {
  AsyncParallelForPass() = default;

  AsyncParallelForPass(bool asyncDispatch, int32_t numWorkerThreads,
                       int32_t targetBlockSize) {
    this->asyncDispatch = asyncDispatch;
    this->numWorkerThreads = numWorkerThreads;
    this->targetBlockSize = targetBlockSize;
  }

  void runOnOperation() override;
};

struct AsyncParallelForRewrite : public OpRewritePattern<scf::ParallelOp> {
public:
  AsyncParallelForRewrite(MLIRContext *ctx, bool asyncDispatch,
                          int32_t numWorkerThreads, int32_t targetBlockSize)
      : OpRewritePattern(ctx), asyncDispatch(asyncDispatch),
        numWorkerThreads(numWorkerThreads), targetBlockSize(targetBlockSize) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override;

private:
  bool asyncDispatch;
  int32_t numWorkerThreads;
  int32_t targetBlockSize;
};

struct ParallelComputeFunctionType {
  FunctionType type;
  llvm::SmallVector<Value> captures;
};

struct ParallelComputeFunction {
  FuncOp func;
  llvm::SmallVector<Value> captures;
};

} // namespace

// Converts one-dimensional iteration index in the [0, tripCount) interval
// into multidimensional iteration coordinate.
static SmallVector<Value> delinearize(ImplicitLocOpBuilder &b, Value index,
                                      ArrayRef<Value> tripCounts) {
  SmallVector<Value> coords(tripCounts.size());
  assert(!tripCounts.empty() && "tripCounts must be not empty");

  for (ssize_t i = tripCounts.size() - 1; i >= 0; --i) {
    coords[i] = b.create<SignedRemIOp>(index, tripCounts[i]);
    index = b.create<SignedDivIOp>(index, tripCounts[i]);
  }

  return coords;
}

// Returns a function type and implicit captures for a parallel compute
// function. We'll need a list of implicit captures to setup block and value
// mapping when we'll clone the body of the parallel operation.
static ParallelComputeFunctionType
getParallelComputeFunctionType(scf::ParallelOp op, PatternRewriter &rewriter) {
  // Values implicitly captured by the parallel operation.
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op.region(), op.region(), captures);

  llvm::SmallVector<Type> inputs;
  inputs.reserve(2 + 4 * op.getNumLoops() + captures.size());

  Type indexTy = rewriter.getIndexType();

  // One-dimensional iteration space defined by the block index and size.
  inputs.push_back(indexTy); // blockIndex
  inputs.push_back(indexTy); // blockSize

  // Multi-dimensional parallel iteration space defined by the loop trip counts.
  for (unsigned i = 0; i < op.getNumLoops(); ++i)
    inputs.push_back(indexTy); // loop tripCount

  // Parallel operation lower bound, upper bound and step.
  for (unsigned i = 0; i < op.getNumLoops(); ++i) {
    inputs.push_back(indexTy); // lower bound
    inputs.push_back(indexTy); // upper bound
    inputs.push_back(indexTy); // step
  }

  // Types of the implicit captures.
  for (Value capture : captures)
    inputs.push_back(capture.getType());

  // Convert captures to vector for later convenience.
  SmallVector<Value> capturesVector(captures.begin(), captures.end());
  return {rewriter.getFunctionType(inputs, TypeRange()), capturesVector};
}

// Create a parallel compute fuction from the parallel operation.
static ParallelComputeFunction
createParallelComputeFunction(scf::ParallelOp op, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  ModuleOp module = op->getParentOfType<ModuleOp>();

  ParallelComputeFunctionType computeFuncType =
      getParallelComputeFunctionType(op, rewriter);

  FunctionType type = computeFuncType.type;
  FuncOp func = FuncOp::create(op.getLoc(), "parallel_compute_fn", type);
  func.setPrivate();

  // Insert function into the module symbol table and assign it unique name.
  SymbolTable symbolTable(module);
  symbolTable.insert(func);
  rewriter.getListener()->notifyOperationInserted(func);

  // Create function entry block.
  Block *block = b.createBlock(&func.getBody(), func.begin(), type.getInputs());
  b.setInsertionPointToEnd(block);

  unsigned offset = 0; // argument offset for arguments decoding

  // Returns `numArguments` arguments starting from `offset` and updates offset
  // by moving forward to the next argument.
  auto getArguments = [&](unsigned numArguments) -> ArrayRef<Value> {
    auto args = block->getArguments();
    auto slice = args.drop_front(offset).take_front(numArguments);
    offset += numArguments;
    return {slice.begin(), slice.end()};
  };

  // Block iteration position defined by the block index and size.
  Value blockIndex = block->getArgument(offset++);
  Value blockSize = block->getArgument(offset++);

  // Constants used below.
  Value c0 = b.create<ConstantIndexOp>(0);
  Value c1 = b.create<ConstantIndexOp>(1);

  // Multi-dimensional parallel iteration space defined by the loop trip counts.
  ArrayRef<Value> tripCounts = getArguments(op.getNumLoops());

  // Compute a product of trip counts to get the size of the flattened
  // one-dimensional iteration space.
  Value tripCount = tripCounts[0];
  for (unsigned i = 1; i < tripCounts.size(); ++i)
    tripCount = b.create<MulIOp>(tripCount, tripCounts[i]);

  // Parallel operation lower bound and step.
  ArrayRef<Value> lowerBound = getArguments(op.getNumLoops());
  offset += op.getNumLoops(); // skip upper bound arguments
  ArrayRef<Value> step = getArguments(op.getNumLoops());

  // Remaining arguments are implicit captures of the parallel operation.
  ArrayRef<Value> captures = getArguments(block->getNumArguments() - offset);

  // Find one-dimensional iteration bounds: [blockFirstIndex, blockLastIndex]:
  //   blockFirstIndex = blockIndex * blockSize
  Value blockFirstIndex = b.create<MulIOp>(blockIndex, blockSize);

  // The last one-dimensional index in the block defined by the `blockIndex`:
  //   blockLastIndex = max(blockFirstIndex + blockSize, tripCount) - 1
  Value blockEnd0 = b.create<AddIOp>(blockFirstIndex, blockSize);
  Value blockEnd1 = b.create<CmpIOp>(CmpIPredicate::sge, blockEnd0, tripCount);
  Value blockEnd2 = b.create<SelectOp>(blockEnd1, tripCount, blockEnd0);
  Value blockLastIndex = b.create<SubIOp>(blockEnd2, c1);

  // Convert one-dimensional indices to multi-dimensional coordinates.
  auto blockFirstCoord = delinearize(b, blockFirstIndex, tripCounts);
  auto blockLastCoord = delinearize(b, blockLastIndex, tripCounts);

  // Compute loops upper bounds derived from the block last coordinates:
  //   blockEndCoord[i] = blockLastCoord[i] + 1
  //
  // Block first and last coordinates can be the same along the outer compute
  // dimension when inner compute dimension contains multiple blocks.
  SmallVector<Value> blockEndCoord(op.getNumLoops());
  for (size_t i = 0; i < blockLastCoord.size(); ++i)
    blockEndCoord[i] = b.create<AddIOp>(blockLastCoord[i], c1);

  // Construct a loop nest out of scf.for operations that will iterate over
  // all coordinates in [blockFirstCoord, blockLastCoord] range.
  using LoopBodyBuilder =
      std::function<void(OpBuilder &, Location, Value, ValueRange)>;
  using LoopNestBuilder = std::function<LoopBodyBuilder(size_t loopIdx)>;

  // Parallel region induction variables computed from the multi-dimensional
  // iteration coordinate using parallel operation bounds and step:
  //
  //   computeBlockInductionVars[loopIdx] =
  //       lowerBound[loopIdx] + blockCoord[loopIdx] * step[loopDdx]
  SmallVector<Value> computeBlockInductionVars(op.getNumLoops());

  // We need to know if we are in the first or last iteration of the
  // multi-dimensional loop for each loop in the nest, so we can decide what
  // loop bounds should we use for the nested loops: bounds defined by compute
  // block interval, or bounds defined by the parallel operation.
  //
  // Example: 2d parallel operation
  //                   i   j
  //   loop sizes:   [50, 50]
  //   first coord:  [25, 25]
  //   last coord:   [30, 30]
  //
  // If `i` is equal to 25 then iteration over `j` should start at 25, when `i`
  // is between 25 and 30 it should start at 0. The upper bound for `j` should
  // be 50, except when `i` is equal to 30, then it should also be 30.
  //
  // Value at ith position specifies if all loops in [0, i) range of the loop
  // nest are in the first/last iteration.
  SmallVector<Value> isBlockFirstCoord(op.getNumLoops());
  SmallVector<Value> isBlockLastCoord(op.getNumLoops());

  // Builds inner loop nest inside async.execute operation that does all the
  // work concurrently.
  LoopNestBuilder workLoopBuilder = [&](size_t loopIdx) -> LoopBodyBuilder {
    return [&, loopIdx](OpBuilder &nestedBuilder, Location loc, Value iv,
                        ValueRange args) {
      ImplicitLocOpBuilder nb(loc, nestedBuilder);

      // Compute induction variable for `loopIdx`.
      computeBlockInductionVars[loopIdx] = nb.create<AddIOp>(
          lowerBound[loopIdx], nb.create<MulIOp>(iv, step[loopIdx]));

      // Check if we are inside first or last iteration of the loop.
      isBlockFirstCoord[loopIdx] =
          nb.create<CmpIOp>(CmpIPredicate::eq, iv, blockFirstCoord[loopIdx]);
      isBlockLastCoord[loopIdx] =
          nb.create<CmpIOp>(CmpIPredicate::eq, iv, blockLastCoord[loopIdx]);

      // Check if the previous loop is in its first or last iteration.
      if (loopIdx > 0) {
        isBlockFirstCoord[loopIdx] = nb.create<AndOp>(
            isBlockFirstCoord[loopIdx], isBlockFirstCoord[loopIdx - 1]);
        isBlockLastCoord[loopIdx] = nb.create<AndOp>(
            isBlockLastCoord[loopIdx], isBlockLastCoord[loopIdx - 1]);
      }

      // Keep building loop nest.
      if (loopIdx < op.getNumLoops() - 1) {
        // Select nested loop lower/upper bounds depending on out position in
        // the multi-dimensional iteration space.
        auto lb = nb.create<SelectOp>(isBlockFirstCoord[loopIdx],
                                      blockFirstCoord[loopIdx + 1], c0);

        auto ub = nb.create<SelectOp>(isBlockLastCoord[loopIdx],
                                      blockEndCoord[loopIdx + 1],
                                      tripCounts[loopIdx + 1]);

        nb.create<scf::ForOp>(lb, ub, c1, ValueRange(),
                              workLoopBuilder(loopIdx + 1));
        nb.create<scf::YieldOp>(loc);
        return;
      }

      // Copy the body of the parallel op into the inner-most loop.
      BlockAndValueMapping mapping;
      mapping.map(op.getInductionVars(), computeBlockInductionVars);
      mapping.map(computeFuncType.captures, captures);

      for (auto &bodyOp : op.getLoopBody().getOps())
        nb.clone(bodyOp, mapping);
    };
  };

  b.create<scf::ForOp>(blockFirstCoord[0], blockEndCoord[0], c1, ValueRange(),
                       workLoopBuilder(0));
  b.create<ReturnOp>(ValueRange());

  return {func, std::move(computeFuncType.captures)};
}

// Creates recursive async dispatch function for the given parallel compute
// function. Dispatch function keeps splitting block range into halves until it
// reaches a single block, and then excecutes it inline.
//
// Function pseudocode (mix of C++ and MLIR):
//
//   func @async_dispatch(%block_start : index, %block_end : index, ...) {
//
//     // Keep splitting block range until we reached a range of size 1.
//     while (%block_end - %block_start > 1) {
//       %mid_index = block_start + (block_end - block_start) / 2;
//       async.execute { call @async_dispatch(%mid_index, %block_end); }
//       %block_end = %mid_index
//     }
//
//     // Call parallel compute function for a single block.
//     call @parallel_compute_fn(%block_start, %block_size, ...);
//   }
//
static FuncOp createAsyncDispatchFunction(ParallelComputeFunction &computeFunc,
                                          PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = computeFunc.func.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);

  ModuleOp module = computeFunc.func->getParentOfType<ModuleOp>();

  ArrayRef<Type> computeFuncInputTypes =
      computeFunc.func.type().cast<FunctionType>().getInputs();

  // Compared to the parallel compute function async dispatch function takes
  // additional !async.group argument. Also instead of a single `blockIndex` it
  // takes `blockStart` and `blockEnd` arguments to define the range of
  // dispatched blocks.
  SmallVector<Type> inputTypes;
  inputTypes.push_back(async::GroupType::get(rewriter.getContext()));
  inputTypes.push_back(rewriter.getIndexType()); // add blockStart argument
  inputTypes.append(computeFuncInputTypes.begin(), computeFuncInputTypes.end());

  FunctionType type = rewriter.getFunctionType(inputTypes, TypeRange());
  FuncOp func = FuncOp::create(loc, "async_dispatch_fn", type);
  func.setPrivate();

  // Insert function into the module symbol table and assign it unique name.
  SymbolTable symbolTable(module);
  symbolTable.insert(func);
  rewriter.getListener()->notifyOperationInserted(func);

  // Create function entry block.
  Block *block = b.createBlock(&func.getBody(), func.begin(), type.getInputs());
  b.setInsertionPointToEnd(block);

  Type indexTy = b.getIndexType();
  Value c1 = b.create<ConstantIndexOp>(1);
  Value c2 = b.create<ConstantIndexOp>(2);

  // Get the async group that will track async dispatch completion.
  Value group = block->getArgument(0);

  // Get the block iteration range: [blockStart, blockEnd)
  Value blockStart = block->getArgument(1);
  Value blockEnd = block->getArgument(2);

  // Create a work splitting while loop for the [blockStart, blockEnd) range.
  SmallVector<Type> types = {indexTy, indexTy};
  SmallVector<Value> operands = {blockStart, blockEnd};

  // Create a recursive dispatch loop.
  scf::WhileOp whileOp = b.create<scf::WhileOp>(types, operands);
  Block *before = b.createBlock(&whileOp.before(), {}, types);
  Block *after = b.createBlock(&whileOp.after(), {}, types);

  // Setup dispatch loop condition block: decide if we need to go into the
  // `after` block and launch one more async dispatch.
  {
    b.setInsertionPointToEnd(before);
    Value start = before->getArgument(0);
    Value end = before->getArgument(1);
    Value distance = b.create<SubIOp>(end, start);
    Value dispatch = b.create<CmpIOp>(CmpIPredicate::sgt, distance, c1);
    b.create<scf::ConditionOp>(dispatch, before->getArguments());
  }

  // Setup the async dispatch loop body: recursively call dispatch function
  // for the seconds half of the original range and go to the next iteration.
  {
    b.setInsertionPointToEnd(after);
    Value start = after->getArgument(0);
    Value end = after->getArgument(1);
    Value distance = b.create<SubIOp>(end, start);
    Value halfDistance = b.create<SignedDivIOp>(distance, c2);
    Value midIndex = b.create<AddIOp>(start, halfDistance);

    // Call parallel compute function inside the async.execute region.
    auto executeBodyBuilder = [&](OpBuilder &executeBuilder,
                                  Location executeLoc, ValueRange executeArgs) {
      // Update the original `blockStart` and `blockEnd` with new range.
      SmallVector<Value> operands{block->getArguments().begin(),
                                  block->getArguments().end()};
      operands[1] = midIndex;
      operands[2] = end;

      executeBuilder.create<CallOp>(executeLoc, func.sym_name(),
                                    func.getCallableResults(), operands);
      executeBuilder.create<async::YieldOp>(executeLoc, ValueRange());
    };

    // Create async.execute operation to dispatch half of the block range.
    auto execute = b.create<ExecuteOp>(TypeRange(), ValueRange(), ValueRange(),
                                       executeBodyBuilder);
    b.create<AddToGroupOp>(indexTy, execute.token(), group);
    b.create<scf::YieldOp>(ValueRange({start, midIndex}));
  }

  // After dispatching async operations to process the tail of the block range
  // call the parallel compute function for the first block of the range.
  b.setInsertionPointAfter(whileOp);

  // Drop async dispatch specific arguments: async group, block start and end.
  auto forwardedInputs = block->getArguments().drop_front(3);
  SmallVector<Value> computeFuncOperands = {blockStart};
  computeFuncOperands.append(forwardedInputs.begin(), forwardedInputs.end());

  b.create<CallOp>(computeFunc.func.sym_name(),
                   computeFunc.func.getCallableResults(), computeFuncOperands);
  b.create<ReturnOp>(ValueRange());

  return func;
}

// Launch async dispatch of the parallel compute function.
static void doAsyncDispatch(ImplicitLocOpBuilder &b, PatternRewriter &rewriter,
                            ParallelComputeFunction &parallelComputeFunction,
                            scf::ParallelOp op, Value blockSize,
                            Value blockCount,
                            const SmallVector<Value> &tripCounts) {
  MLIRContext *ctx = op->getContext();

  // Add one more level of indirection to dispatch parallel compute functions
  // using async operations and recursive work splitting.
  FuncOp asyncDispatchFunction =
      createAsyncDispatchFunction(parallelComputeFunction, rewriter);

  Value c0 = b.create<ConstantIndexOp>(0);
  Value c1 = b.create<ConstantIndexOp>(1);

  // Create an async.group to wait on all async tokens from the concurrent
  // execution of multiple parallel compute function. First block will be
  // executed synchronously in the caller thread.
  Value groupSize = b.create<SubIOp>(blockCount, c1);
  Value group = b.create<CreateGroupOp>(GroupType::get(ctx), groupSize);

  // Appends operands shared by async dispatch and parallel compute functions to
  // the given operands vector.
  auto appendBlockComputeOperands = [&](SmallVector<Value> &operands) {
    operands.append(tripCounts);
    operands.append(op.lowerBound().begin(), op.lowerBound().end());
    operands.append(op.upperBound().begin(), op.upperBound().end());
    operands.append(op.step().begin(), op.step().end());
    operands.append(parallelComputeFunction.captures);
  };

  // Check if the block size is one, in this case we can skip the async dispatch
  // completely. If this will be known statically, then canonicalization will
  // erase async group operations.
  Value isSingleBlock = b.create<CmpIOp>(CmpIPredicate::eq, blockCount, c1);

  auto syncDispatch = [&](OpBuilder &nestedBuilder, Location loc) {
    ImplicitLocOpBuilder nb(loc, nestedBuilder);

    // Call parallel compute function for the single block.
    SmallVector<Value> operands = {c0, blockSize};
    appendBlockComputeOperands(operands);

    nb.create<CallOp>(parallelComputeFunction.func.sym_name(),
                      parallelComputeFunction.func.getCallableResults(),
                      operands);
    nb.create<scf::YieldOp>();
  };

  auto asyncDispatch = [&](OpBuilder &nestedBuilder, Location loc) {
    ImplicitLocOpBuilder nb(loc, nestedBuilder);

    // Launch async dispatch function for [0, blockCount) range.
    SmallVector<Value> operands = {group, c0, blockCount, blockSize};
    appendBlockComputeOperands(operands);

    nb.create<CallOp>(asyncDispatchFunction.sym_name(),
                      asyncDispatchFunction.getCallableResults(), operands);
    nb.create<scf::YieldOp>();
  };

  // Dispatch either single block compute function, or launch async dispatch.
  b.create<scf::IfOp>(TypeRange(), isSingleBlock, syncDispatch, asyncDispatch);

  // Wait for the completion of all parallel compute operations.
  b.create<AwaitAllOp>(group);
}

// Dispatch parallel compute functions by submitting all async compute tasks
// from a simple for loop in the caller thread.
static void
doSequantialDispatch(ImplicitLocOpBuilder &b, PatternRewriter &rewriter,
                     ParallelComputeFunction &parallelComputeFunction,
                     scf::ParallelOp op, Value blockSize, Value blockCount,
                     const SmallVector<Value> &tripCounts) {
  MLIRContext *ctx = op->getContext();

  FuncOp compute = parallelComputeFunction.func;

  Value c0 = b.create<ConstantIndexOp>(0);
  Value c1 = b.create<ConstantIndexOp>(1);

  // Create an async.group to wait on all async tokens from the concurrent
  // execution of multiple parallel compute function. First block will be
  // executed synchronously in the caller thread.
  Value groupSize = b.create<SubIOp>(blockCount, c1);
  Value group = b.create<CreateGroupOp>(GroupType::get(ctx), groupSize);

  // Call parallel compute function for all blocks.
  using LoopBodyBuilder =
      std::function<void(OpBuilder &, Location, Value, ValueRange)>;

  // Returns parallel compute function operands to process the given block.
  auto computeFuncOperands = [&](Value blockIndex) -> SmallVector<Value> {
    SmallVector<Value> computeFuncOperands = {blockIndex, blockSize};
    computeFuncOperands.append(tripCounts);
    computeFuncOperands.append(op.lowerBound().begin(), op.lowerBound().end());
    computeFuncOperands.append(op.upperBound().begin(), op.upperBound().end());
    computeFuncOperands.append(op.step().begin(), op.step().end());
    computeFuncOperands.append(parallelComputeFunction.captures);
    return computeFuncOperands;
  };

  // Induction variable is the index of the block: [0, blockCount).
  LoopBodyBuilder loopBuilder = [&](OpBuilder &loopBuilder, Location loc,
                                    Value iv, ValueRange args) {
    ImplicitLocOpBuilder nb(loc, loopBuilder);

    // Call parallel compute function inside the async.execute region.
    auto executeBodyBuilder = [&](OpBuilder &executeBuilder,
                                  Location executeLoc, ValueRange executeArgs) {
      executeBuilder.create<CallOp>(executeLoc, compute.sym_name(),
                                    compute.getCallableResults(),
                                    computeFuncOperands(iv));
      executeBuilder.create<async::YieldOp>(executeLoc, ValueRange());
    };

    // Create async.execute operation to launch parallel computate function.
    auto execute = nb.create<ExecuteOp>(TypeRange(), ValueRange(), ValueRange(),
                                        executeBodyBuilder);
    nb.create<AddToGroupOp>(rewriter.getIndexType(), execute.token(), group);
    nb.create<scf::YieldOp>();
  };

  // Iterate over all compute blocks and launch parallel compute operations.
  b.create<scf::ForOp>(c1, blockCount, c1, ValueRange(), loopBuilder);

  // Call parallel compute function for the first block in the caller thread.
  b.create<CallOp>(compute.sym_name(), compute.getCallableResults(),
                   computeFuncOperands(c0));

  // Wait for the completion of all async compute operations.
  b.create<AwaitAllOp>(group);
}

LogicalResult
AsyncParallelForRewrite::matchAndRewrite(scf::ParallelOp op,
                                         PatternRewriter &rewriter) const {
  // We do not currently support rewrite for parallel op with reductions.
  if (op.getNumReductions() != 0)
    return failure();

  ImplicitLocOpBuilder b(op.getLoc(), rewriter);

  // Compute trip count for each loop induction variable:
  //   tripCount = ceil_div(upperBound - lowerBound, step);
  SmallVector<Value> tripCounts(op.getNumLoops());
  for (size_t i = 0; i < op.getNumLoops(); ++i) {
    auto lb = op.lowerBound()[i];
    auto ub = op.upperBound()[i];
    auto step = op.step()[i];
    auto range = b.create<SubIOp>(ub, lb);
    tripCounts[i] = b.create<SignedCeilDivIOp>(range, step);
  }

  // Compute a product of trip counts to get the 1-dimensional iteration space
  // for the scf.parallel operation.
  Value tripCount = tripCounts[0];
  for (size_t i = 1; i < tripCounts.size(); ++i)
    tripCount = b.create<MulIOp>(tripCount, tripCounts[i]);

  // With large number of threads the value of creating many compute blocks
  // is reduced because the problem typically becomes memory bound. For small
  // number of threads it helps with stragglers.
  float overshardingFactor = numWorkerThreads <= 4    ? 8.0
                             : numWorkerThreads <= 8  ? 4.0
                             : numWorkerThreads <= 16 ? 2.0
                             : numWorkerThreads <= 32 ? 1.0
                             : numWorkerThreads <= 64 ? 0.8
                                                      : 0.6;

  // Do not overload worker threads with too many compute blocks.
  Value maxComputeBlocks = b.create<ConstantIndexOp>(
      std::max(1, static_cast<int>(numWorkerThreads * overshardingFactor)));

  // Target block size from the pass parameters.
  Value targetComputeBlockSize = b.create<ConstantIndexOp>(targetBlockSize);

  // Compute parallel block size from the parallel problem size:
  //   blockSize = min(tripCount,
  //                   max(ceil_div(tripCount, maxComputeBlocks),
  //                       targetComputeBlockSize))
  Value bs0 = b.create<SignedCeilDivIOp>(tripCount, maxComputeBlocks);
  Value bs1 = b.create<CmpIOp>(CmpIPredicate::sge, bs0, targetComputeBlockSize);
  Value bs2 = b.create<SelectOp>(bs1, bs0, targetComputeBlockSize);
  Value bs3 = b.create<CmpIOp>(CmpIPredicate::sle, tripCount, bs2);
  Value blockSize0 = b.create<SelectOp>(bs3, tripCount, bs2);
  Value blockCount0 = b.create<SignedCeilDivIOp>(tripCount, blockSize0);

  // Compute balanced block size for the estimated block count.
  Value blockSize = b.create<SignedCeilDivIOp>(tripCount, blockCount0);
  Value blockCount = b.create<SignedCeilDivIOp>(tripCount, blockSize);

  // Create a parallel compute function that takes a block id and computes the
  // parallel operation body for a subset of iteration space.
  ParallelComputeFunction parallelComputeFunction =
      createParallelComputeFunction(op, rewriter);

  // Dispatch parallel compute function using async recursive work splitting, or
  // by submitting compute task sequentially from a caller thread.
  if (asyncDispatch) {
    doAsyncDispatch(b, rewriter, parallelComputeFunction, op, blockSize,
                    blockCount, tripCounts);
  } else {
    doSequantialDispatch(b, rewriter, parallelComputeFunction, op, blockSize,
                         blockCount, tripCounts);
  }

  // Parallel operation was replaced with a block iteration loop.
  rewriter.eraseOp(op);

  return success();
}

void AsyncParallelForPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<AsyncParallelForRewrite>(ctx, asyncDispatch, numWorkerThreads,
                                        targetBlockSize);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createAsyncParallelForPass() {
  return std::make_unique<AsyncParallelForPass>();
}

std::unique_ptr<Pass>
mlir::createAsyncParallelForPass(bool asyncDispatch, int32_t numWorkerThreads,
                                 int32_t targetBlockSize) {
  return std::make_unique<AsyncParallelForPass>(asyncDispatch, numWorkerThreads,
                                                targetBlockSize);
}
