//===- SparseGPUCodegen.cpp - Generates GPU code (using CUDA) -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a prototype GPU codegenerator for the sparse compiler.
// The objective is to eventually use the right combination of
// direct code generation and libary calls into vendor-specific
// highly optimized sparse libraries (e.g. cuSparse for CUDA).
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "LoopEmitter.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Marks the given top module as a GPU container module.
static void markAsGPUContainer(ModuleOp topModule) {
  topModule->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                     UnitAttr::get(topModule->getContext()));
}

/// Constructs a new GPU module (for GPU kernels) inside the given top module,
/// or returns an existing GPU module if one was built previously.
static gpu::GPUModuleOp genGPUModule(OpBuilder &builder, ModuleOp topModule) {
  for (auto op : topModule.getBodyRegion().getOps<gpu::GPUModuleOp>())
    return op; // existing
  markAsGPUContainer(topModule);
  builder.setInsertionPointToStart(&topModule.getBodyRegion().front());
  return builder.create<gpu::GPUModuleOp>(topModule->getLoc(),
                                          "sparse_kernels");
}

/// Constructs a new GPU kernel in the given GPU module.
static gpu::GPUFuncOp genGPUFunc(OpBuilder &builder, gpu::GPUModuleOp gpuModule,
                                 SmallVectorImpl<Value> &args) {
  // Get a unique kernel name. Not very creative,
  // but we simply try kernel0, kernel1, etc.
  unsigned kernelNumber = 0;
  SmallString<16> kernelName;
  do {
    kernelName.clear();
    ("kernel" + Twine(kernelNumber++)).toStringRef(kernelName);
  } while (gpuModule.lookupSymbol(kernelName));
  // Then we insert a new kernel with given arguments into the module.
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  SmallVector<Type> argsTp;
  for (unsigned i = 0, e = args.size(); i < e; i++)
    argsTp.push_back(args[i].getType());
  FunctionType type = FunctionType::get(gpuModule->getContext(), argsTp, {});
  auto gpuFunc =
      builder.create<gpu::GPUFuncOp>(gpuModule->getLoc(), kernelName, type);
  gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                   builder.getUnitAttr());
  return gpuFunc;
}

/// Constructs code to launch GPU kernel.
static void genLaunchGPUFunc(OpBuilder &builder, gpu::GPUFuncOp gpuFunc,
                             SmallVectorImpl<Value> &args,
                             unsigned numThreads) {
  Location loc = gpuFunc->getLoc();
  Value none = TypedValue<::mlir::IntegerType>{};
  Value one = constantIndex(builder, loc, 1);
  Value numT = constantIndex(builder, loc, numThreads);
  gpu::KernelDim3 gridSize = {one, one, one};
  gpu::KernelDim3 blckSize = {numT, one, one};
  builder.create<gpu::LaunchFuncOp>(loc, gpuFunc, gridSize, blckSize,
                                    /*dynSharedMemSz*/ none, args);
}

/// Maps the provided ranked host buffer into the device address space.
/// Writes from the host are guaranteed to be visible to device kernels
/// that are launched afterwards. Writes from the device are guaranteed
/// to be visible on the host after synchronizing with the device kernel
/// completion.
static Value genHostRegisterMemref(OpBuilder &builder, Location loc,
                                   Value mem) {
  MemRefType memTp = mem.getType().cast<MemRefType>();
  UnrankedMemRefType resTp =
      UnrankedMemRefType::get(memTp.getElementType(), /*memorySpace=*/0);
  Value cast = builder.create<memref::CastOp>(loc, resTp, mem);
  builder.create<gpu::HostRegisterOp>(loc, cast);
  return mem; // convenience pass-through
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

//===----------------------------------------------------------------------===//
// Rewriting rules.
//===----------------------------------------------------------------------===//

/// Proof-of-concept rewriter. This rule generates a CUDA implementation
/// for each outermost forall loop generated by the sparse compiler.
//
// TODO: right works with parallelization-strategy=dense-outer-loop
//       but give this its own flags in the future
//
struct ForallRewriter : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  ForallRewriter(MLIRContext *context, unsigned nT)
      : OpRewritePattern(context), numThreads(nT){};

  LogicalResult matchAndRewrite(scf::ParallelOp forallOp,
                                PatternRewriter &rewriter) const override {
    // Reject inadmissible loop form.
    // Essentially only accept a loop, generated by the sparse compiler,
    // of the form
    //   forall (i = 0; i < N; i++)
    // so that cyclic scheduling over the threads is easy.
    if (!forallOp->hasAttr(LoopEmitter::getLoopEmitterLoopAttrName()) ||
        forallOp.getNumReductions() != 0 || forallOp.getNumLoops() != 1 ||
        !matchPattern(forallOp.getLowerBound()[0], m_Zero()) ||
        !matchPattern(forallOp.getStep()[0], m_One()))
      return failure();
    // Collect every value that is computed outside the parallel loop.
    SetVector<Value> invariants; // stable iteration!
    forallOp->walk([&](Operation *op) {
      // Collect all values of admissible ops.
      for (OpOperand &o : op->getOpOperands()) {
        Value val = o.get();
        Block *block;
        if (auto arg = val.dyn_cast<BlockArgument>())
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
      else if (tp.isa<FloatType>() || tp.isIntOrIndex())
        scalars.push_back(val);
      else if (isa<MemRefType>(tp))
        buffers.push_back(val);
      else
        return failure(); // don't know how to share
    }
    // Prepare the outlined arguments, register buffers.
    Location loc = forallOp->getLoc();
    SmallVector<Value> args;
    for (Value s : scalars)
      args.push_back(s);
    for (Value b : buffers)
      args.push_back(genHostRegisterMemref(rewriter, loc, b));
    auto saveIp = rewriter.saveInsertionPoint();
    // Set up GPU module and construct GPU function.
    ModuleOp topModule = forallOp->getParentOfType<ModuleOp>();
    auto gpuModule = genGPUModule(rewriter, topModule);
    auto gpuFunc = genGPUFunc(rewriter, gpuModule, args);
    genGPUCode(rewriter, gpuFunc, forallOp, constants, scalars, buffers);
    // Generate code that launches the kernel.
    rewriter.restoreInsertionPoint(saveIp);
    genLaunchGPUFunc(rewriter, gpuFunc, args, numThreads);
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

  unsigned numThreads;
};

} // namespace

//===----------------------------------------------------------------------===//
// Public method for populating GPU rewriting rules.
//===----------------------------------------------------------------------===//

void mlir::populateSparseGPUCodegenPatterns(RewritePatternSet &patterns,
                                            unsigned numThreads) {
  patterns.add<ForallRewriter>(patterns.getContext(), numThreads);
}
