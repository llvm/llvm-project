//===- SparseKokkosCodegen.cpp - Generates Kokkos code --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "LoopEmitter.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Constructs a new function for launching a Kokkos kernel.
/// The Kokkos lambda can directly capture the arguments to this function,
/// and nothing else. All memrefs are expected to be in device space
/// (DefaultExecutionSpace::memory_space), not HostSpace.
static func::FuncOp genKokkosFunc(OpBuilder &builder, ModuleOp module,
                                 SmallVectorImpl<Value> &args) {
  static unsigned kernelNumber = 0;
  SmallString<16> kernelName;
  ("__kokkos_sparse_kernel_" + Twine(kernelNumber++)).toStringRef(kernelName);
  // Then we insert a new kernel with given arguments into the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  SmallVector<Type> argsTp;
  for (unsigned i = 0, e = args.size(); i < e; i++)
    argsTp.push_back(args[i].getType());
  FunctionType type = FunctionType::get(builder.getContext(), argsTp, {});
  return builder.create<func::FuncOp>(module.getLoc(), kernelName, type);
}

static void genCallFunc(OpBuilder &builder, func::FuncOp func,
                              SmallVectorImpl<Value> &args) {
  Location loc = func->getLoc();
  return builder
      .create<func::CallOp>(loc, gpuFunc, gridSize, blckSize,
                                 /*dynSharedMemSz*/ none, args,
                                 builder.getType<gpu::AsyncTokenType>(), tokens);
}

/// Allocates memory on the device.
static gpu::AllocOp genAllocMemRef(OpBuilder &builder, Location loc, Value mem,
                                   Value token) {
  auto tp = cast<ShapedType>(mem.getType());
  auto elemTp = tp.getElementType();
  auto shape = tp.getShape();
  auto memTp = MemRefType::get(shape, elemTp);
  SmallVector<Value> dynamicSizes;
  for (unsigned r = 0, rank = tp.getRank(); r < rank; r++) {
    if (shape[r] == ShapedType::kDynamic) {
      Value dimOp = linalg::createOrFoldDimOp(builder, loc, mem, r);
      dynamicSizes.push_back(dimOp);
    }
  }
  return builder.create<gpu::AllocOp>(loc, TypeRange({memTp, token.getType()}),
                                      token, dynamicSizes, ValueRange());
}

/// Deallocates memory from the device.
static Value genDeallocMemRef(OpBuilder &builder, Location loc, Value mem,
                              Value token) {
  return builder.create<gpu::DeallocOp>(loc, token.getType(), token, mem)
      .getAsyncToken();
}

/// Copies memory between host and device (direction is implicit).
static Value genCopyMemRef(OpBuilder &builder, Location loc, Value dst,
                           Value src, Value token) {
  return builder.create<gpu::MemcpyOp>(loc, token.getType(), token, dst, src)
      .getAsyncToken();
}

/// Generates an alloc/copy pair.
static Value genAllocCopy(OpBuilder &builder, Location loc, Value b,
                          SmallVectorImpl<Value> &tokens) {
  Value firstToken = genFirstWait(builder, loc);
  auto alloc = genAllocMemRef(builder, loc, b, firstToken);
  Value devMem = alloc.getResult(0);
  Value depToken = alloc.getAsyncToken(); // copy-after-alloc
  tokens.push_back(genCopyMemRef(builder, loc, devMem, b, depToken));
  return devMem;
}

/// Generates a memref from tensor operation.
static Value genTensorToMemref(PatternRewriter &rewriter, Location loc,
                               Value tensor) {
  auto tensorType = llvm::cast<ShapedType>(tensor.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  return rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, tensor);
}

/// Prepares the outlined arguments, passing scalars and buffers in. Here we
/// assume that the first buffer is the one allocated for output. We create
/// a set of properly chained asynchronous allocation/copy pairs to increase
/// overlap before launching the kernel.
static void genParametersIn(OpBuilder &builder, Location loc,
                             SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args,
                             SmallVectorImpl<Value> &tokens) {
  // Scalars are passed by value.
  for (Value s : scalars)
    args.push_back(s);
  // Buffers are need to be made visible on device.
  for (Value b : buffers) {
    args.push_back(genAllocCopy(builder, loc, b, tokens));
  }
}

/// Finalizes the outlined arguments. The output buffer is copied depending
/// on the kernel token and then deallocated. All other buffers are simply
/// deallocated. Then we wait for all operations to complete.
static void genParametersOut(OpBuilder &builder, Location loc,
                             Value kernelToken, SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args,
                             SmallVectorImpl<Value> &tokens) {
  unsigned base = scalars.size();
  for (unsigned i = base, e = args.size(); i < e; i++) {
    Value firstToken;
    if (i == base) {
      firstToken =
          genCopyMemRef(builder, loc, buffers[0], args[i], kernelToken);
    } else {
      firstToken = genFirstWait(builder, loc);
    }
    tokens.push_back(genDeallocMemRef(builder, loc, args[i], firstToken));
  }
}

/// Constructs code for new GPU kernel.
static void genGPUCode(PatternRewriter &rewriter, gpu::GPUFuncOp gpuFunc,
                       scf::ParallelOp forallOp,
                       SmallVectorImpl<Value> &constants,
                       SmallVectorImpl<Value> &scalars,
                       SmallVectorImpl<Value> &buffers) {
  Location loc = gpuFunc->getLoc();
  Block &block = gpuFunc.getBody().front();
  rewriter.setInsertionPointToStart(&block);

  // Re-generate the constants, recapture all arguments.
  unsigned arg = 0;
  IRMapping irMap;
  for (Value c : constants)
    irMap.map(c, rewriter.clone(*c.getDefiningOp())->getResult(0));
  for (Value s : scalars)
    irMap.map(s, block.getArgument(arg++));
  for (Value b : buffers)
    irMap.map(b, block.getArgument(arg++));

  // Assume 1-dimensional grid/block configuration (only x dimension),
  // so that:
  //   row = blockIdx.x * blockDim.x + threadIdx.x
  //   inc = blockDim.x * gridDim.x
  Value bid = rewriter.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
  Value bsz = rewriter.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
  Value tid = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value gsz = rewriter.create<gpu::GridDimOp>(loc, gpu::Dimension::x);
  Value mul = rewriter.create<arith::MulIOp>(loc, bid, bsz);
  Value row = rewriter.create<arith::AddIOp>(loc, mul, tid);
  Value inc = rewriter.create<arith::MulIOp>(loc, bsz, gsz);

  // Construct the iteration over the computational space that
  // accounts for the fact that the total number of threads and
  // the amount of work to be done usually do not match precisely.
  //   for (r = row; r < N; r += inc) {
  //     <loop-body>
  //   }
  Value upper = irMap.lookup(forallOp.getUpperBound()[0]);
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, row, upper, inc);
  rewriter.cloneRegionBefore(forallOp.getLoopBody(), forOp.getLoopBody(),
                             forOp.getLoopBody().begin(), irMap);

  // Done.
  rewriter.setInsertionPointAfter(forOp);
  rewriter.create<gpu::ReturnOp>(gpuFunc->getLoc());
}

struct KokkosForallRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  KokkosForallRewriter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp forallOp,
                                PatternRewriter &rewriter) const override {
    // Check the forallOp for operations that can't be executed on device
    bool canBeOffloaded = true;
    forallOp->walk([&](func::CallOp op) {
        canBeOffloaded = false;
    });
    forallOp->walk([&](memref::AllocOp op) {
        canBeOffloaded = false;
    });
    forallOp->walk([&](memref::AllocaOp op) {
        canBeOffloaded = false;
    });
    if(!canBeOffloaded)
    {
      // Leave this scf.parallel unchanged;
      // it will execute on host (but can still be parallel).
      return failure();
    }
    // Collect every value that is computed outside the parallel loop.
    SetVector<Value> invariants; // stable iteration!
    forallOp->walk([&](Operation *op) {
      // Collect all values of admissible ops.
      for (OpOperand &o : op->getOpOperands()) {
        Value val = o.get();
        Block *block;
        if (auto arg = dyn_cast<BlockArgument>(val))
          block = arg.getOwner();
        else
          block = val.getDefiningOp()->getBlock();
        if (!isNestedIn(block, forallOp))
          invariants.insert(val);
      }
    });
    // Outline the outside values as proper parameters. Fail when sharing
    // value between host and device is not straightforward.
    SmallVector<Value> constants;
    SmallVector<Value> scalars;
    SmallVector<Value> buffers;
    for (Value val : invariants) {
      Type tp = val.getType();
      if (val.getDefiningOp<arith::ConstantOp>())
        constants.push_back(val);
      else if (isa<FloatType>(tp) || tp.isIntOrIndex())
        scalars.push_back(val);
      else if (isa<MemRefType>(tp))
        buffers.push_back(val);
      else
        return failure(); // don't know how to share
    }
    // Pass outlined non-constant values.
    Location loc = forallOp->getLoc();
    SmallVector<Value> args;
    genParametersIn(rewriter, loc, scalars, buffers, args);
    auto saveIp = rewriter.saveInsertionPoint();
    ModuleOp module = forallOp->getParentOfType<ModuleOp>();
    auto func = genKokkosFunc(rewriter, module, args);
    genKokkosCode(rewriter, func, forallOp, constants, scalars, buffers);

    rewriter.restoreInsertionPoint(saveIp);

    genBlockingWait(rewriter, loc);
    genCallFunc(rewriter, gpuFunc, args, numThreads);
    // Finalize the outlined arguments.
    genParametersOut(rewriter, loc, scalars, buffers, args);
    rewriter.eraseOp(forallOp);
    return success();
  }

private:
  // Helper method to see if block appears in given loop.
  static bool isNestedIn(Block *block, scf::ParallelOp forallOp) {
    for (Operation *o = block->getParentOp(); o; o = o->getParentOp()) {
      if (o == forallOp)
        return true;
    }
    return false;
  }
};

} // namespace

void mlir::populateSparseKokkosCodegenPatterns(RewritePatternSet &patterns) {
  patterns.add<KokkosForallRewriter>(patterns.getContext(), numThreads);
}

