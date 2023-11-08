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
  ("kokkos_sparse_kernel_" + Twine(kernelNumber++)).toStringRef(kernelName);
  // Then we insert a new kernel with given arguments into the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());
  SmallVector<Type> argsTp;
  for (unsigned i = 0, e = args.size(); i < e; i++)
    argsTp.push_back(args[i].getType());
  FunctionType type = FunctionType::get(builder.getContext(), argsTp, {});
  func::FuncOp func = builder.create<func::FuncOp>(module.getLoc(), kernelName, type);
  func.setPrivate();
  // Add a body to the function (not done by default)
  func.addEntryBlock();
  return func;
}

/// Allocates memory on the device.
static gpu::AllocOp genAllocMemRef(OpBuilder &builder, Location loc, Value mem) {
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
  // Create an async alloc with no dependencies
  return builder.create<gpu::AllocOp>(loc, memTp, builder.getType<gpu::AsyncTokenType>(), ValueRange(), dynamicSizes, ValueRange());
}

/// Deallocates memory from the device.
static void genDeallocMemRef(OpBuilder &builder, Location loc, Value mem) {
  builder.create<gpu::DeallocOp>(loc, TypeRange(), ValueRange(), mem);
}

/// Copies memory between host and device (direction is implicit).
static void genCopyMemRef(OpBuilder &builder, Location loc, Value dst, Value src) {
  builder.create<gpu::MemcpyOp>(loc, TypeRange(), ValueRange(), dst, src);
}

/// Generates an alloc/copy pair.
static Value genAllocCopy(OpBuilder &builder, Location loc, Value b) {
  auto alloc = genAllocMemRef(builder, loc, b);
  Value devMem = alloc.getResult(0);
  genCopyMemRef(builder, loc, devMem, b);
  return devMem;
}

/// Prepares the outlined arguments, passing scalars and buffers in. Here we
/// assume that the first buffer is the one allocated for output. We create
/// a set of properly chained asynchronous allocation/copy pairs to increase
/// overlap before launching the kernel.
static void genParametersIn(OpBuilder &builder, Location loc,
                             SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args) {
  // Scalars are passed by value.
  for (Value s : scalars)
    args.push_back(s);
  // Buffers are need to be made visible on device.
  for (Value b : buffers) {
    args.push_back(genAllocCopy(builder, loc, b));
  }
}

/// Finalizes the outlined arguments. The output buffer is copied depending
/// on the kernel token and then deallocated. All other buffers are simply
/// deallocated. Then we wait for all operations to complete.
static void genParametersOut(OpBuilder &builder, Location loc,
                             SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args) {
  unsigned base = scalars.size();
  // Go through the buffer (memref-typed) args.
  // The first one is the output buffer, and must be copied back to host.
  // The others are inputs and just need to be deallocated.
  genCopyMemRef(builder, loc, buffers[0], args[base]);
  // Sequence the output tensor dealloc after it's copied to host.
  for (unsigned i = base, e = args.size(); i < e; i++) {
    genDeallocMemRef(builder, loc, args[i]);
  }
}

static void genKokkosCode(PatternRewriter &rewriter, func::FuncOp func,
                       scf::ParallelOp forallOp,
                       SmallVectorImpl<Value> &constants,
                       SmallVectorImpl<Value> &scalars,
                       SmallVectorImpl<Value> &buffers) {
  Block &block = func.getBody().front();
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

  // Copy the outer scf.parallel to this function, but redefine
  // all constants, and replace scalars and buffers with corresponding arguments to func.
  rewriter.clone(*forallOp, irMap);
  
  //TODO: if forallOp has at least one reduction at the top level, func should return those result(s)
  rewriter.create<func::ReturnOp>(forallOp.getLoc());
}

struct KokkosForallRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  KokkosForallRewriter(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(scf::ParallelOp forallOp,
                                PatternRewriter &rewriter) const override {
    // Only run on outermost ParallelOps
    if(forallOp->getParentOfType<scf::ParallelOp>()) {
      return failure();
    }
    // Do not run on ParallelOps that are already in 
    auto enclosingFuncName = forallOp->getParentOfType<func::FuncOp>().getSymName();
    if(enclosingFuncName.starts_with("kokkos_sparse_kernel_"))
      return failure();
    // Check the forallOp for operations that can't be executed on device
    // NOTE: Kokkos emitter will have to check for this again, since we have
    // no way to mark an scf.parallel as executing in a particular place.
    bool canBeOffloaded = true;
    forallOp->walk([&](func::CallOp) {
        canBeOffloaded = false;
    });
    forallOp->walk([&](memref::AllocOp) {
        canBeOffloaded = false;
    });
    forallOp->walk([&](memref::AllocaOp) {
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
    // Move the rewriter back to the kernel launch site
    rewriter.restoreInsertionPoint(saveIp);
    // Call the Kokkos function
    rewriter.create<func::CallOp>(loc, func, args);
    // Finalize the outlined arguments.
    genParametersOut(rewriter, loc, scalars, buffers, args);
    // Lastly, add a Kokkos::fence() to make sure the kernel
    // and following deep copy have completed
    rewriter.create<emitc::CallOp>(loc, TypeRange(),
        "Kokkos::fence", ArrayAttr(), ArrayAttr(), ValueRange());
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
  patterns.add<KokkosForallRewriter>(patterns.getContext());
}

