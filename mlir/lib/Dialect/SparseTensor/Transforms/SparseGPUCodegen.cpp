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
static Value genLaunchGPUFunc(OpBuilder &builder, gpu::GPUFuncOp gpuFunc,
                              SmallVectorImpl<Value> &args,
                              SmallVectorImpl<Value> &tokens,
                              unsigned numThreads) {
  Location loc = gpuFunc->getLoc();
  Value none = TypedValue<::mlir::IntegerType>{};
  Value one = constantIndex(builder, loc, 1);
  Value numT = constantIndex(builder, loc, numThreads);
  gpu::KernelDim3 gridSize = {one, one, one};
  gpu::KernelDim3 blckSize = {numT, one, one};
  return builder
      .create<gpu::LaunchFuncOp>(loc, gpuFunc, gridSize, blckSize,
                                 /*dynSharedMemSz*/ none, args,
                                 builder.getType<gpu::AsyncTokenType>(), tokens)
      .getAsyncToken();
}

/// Maps the provided ranked host buffer into the device address space.
/// Writes from the host are guaranteed to be visible to device kernels
/// that are launched afterwards. Writes from the device are guaranteed
/// to be visible on the host after synchronizing with the device kernel
/// completion. Needs to cast the buffer to a unranked buffer.
static Value genHostRegisterMemref(OpBuilder &builder, Location loc,
                                   Value mem) {
  MemRefType memTp = cast<MemRefType>(mem.getType());
  UnrankedMemRefType resTp =
      UnrankedMemRefType::get(memTp.getElementType(), /*memorySpace=*/0);
  Value cast = builder.create<memref::CastOp>(loc, resTp, mem);
  builder.create<gpu::HostRegisterOp>(loc, cast);
  return cast;
}

/// Unmaps the provided buffer, expecting the casted buffer.
static void genHostUnregisterMemref(OpBuilder &builder, Location loc,
                                    Value cast) {
  builder.create<gpu::HostUnregisterOp>(loc, cast);
}

/// Generates first wait in an asynchronous chain.
static Value genFirstWait(OpBuilder &builder, Location loc) {
  Type tokenType = builder.getType<gpu::AsyncTokenType>();
  return builder.create<gpu::WaitOp>(loc, tokenType, ValueRange())
      .getAsyncToken();
}

/// Generates last, blocking wait in an asynchronous chain.
static void genBlockingWait(OpBuilder &builder, Location loc,
                            ValueRange operands) {
  builder.create<gpu::WaitOp>(loc, Type(), operands);
}

/// Allocates memory on the device.
/// TODO: A `host_shared` attribute could be used to indicate that
///       the buffer is visible by both host and device, but lowering
///       that feature does not seem to be fully supported yet.
static gpu::AllocOp genAllocMemRef(OpBuilder &builder, Location loc, Value mem,
                                   Value token) {
  auto tp = cast<ShapedType>(mem.getType());
  auto elemTp = tp.getElementType();
  auto shape = tp.getShape();
  auto memTp = MemRefType::get(shape, elemTp);
  SmallVector<Value> dynamicSizes;
  for (unsigned r = 0, rank = tp.getRank(); r < rank; r++) {
    if (shape[r] == ShapedType::kDynamic) {
      Value dim = constantIndex(builder, loc, r);
      Value dimOp = builder.create<memref::DimOp>(loc, mem, dim);
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

/// Prepares the outlined arguments, passing scalars and buffers in. Here we
/// assume that the first buffer is the one allocated for output. We create
/// a set of properly chained asynchronous allocation/copy pairs to increase
/// overlap before launching the kernel.
/// TODO: the output assumption may be a bit too brittle
static Value genParametersIn(OpBuilder &builder, Location loc,
                             SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args,
                             SmallVectorImpl<Value> &tokens,
                             bool useHostRegistrationForOut) {
  Value out;
  // Scalars are passed by value.
  for (Value s : scalars)
    args.push_back(s);
  // Buffers are need to be made visible on device.
  for (Value b : buffers) {
    if (useHostRegistrationForOut) {
      out = genHostRegisterMemref(builder, loc, b);
      args.push_back(b);
      useHostRegistrationForOut = false;
      continue;
    }
    Value firstToken = genFirstWait(builder, loc);
    auto alloc = genAllocMemRef(builder, loc, b, firstToken);
    Value devMem = alloc.getResult(0);
    Value depToken = alloc.getAsyncToken(); // copy-after-alloc
    args.push_back(devMem);
    tokens.push_back(genCopyMemRef(builder, loc, devMem, b, depToken));
  }
  return out;
}

/// Finalizes the outlined arguments. The output buffer is copied depending
/// on the kernel token and then deallocated. All other buffers are simply
/// deallocated. Then we wait for all operations to complete.
static void genParametersOut(OpBuilder &builder, Location loc, Value out,
                             Value kernelToken, SmallVectorImpl<Value> &scalars,
                             SmallVectorImpl<Value> &buffers,
                             SmallVectorImpl<Value> &args,
                             SmallVectorImpl<Value> &tokens) {
  unsigned base = scalars.size();
  for (unsigned i = base, e = args.size(); i < e; i++) {
    Value firstToken;
    if (i == base) {
      // Assumed output parameter: unregister or copy-out.
      if (out) {
        genHostUnregisterMemref(builder, loc, out);
        out = Value();
        continue;
      }
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

//===----------------------------------------------------------------------===//
// Rewriting rules.
//===----------------------------------------------------------------------===//

/// Proof-of-concept rewriter. This rule generates a CUDA implementation
/// for each outermost forall loop generated by the sparse compiler.
/// TODO: right works with parallelization-strategy=dense-outer-loop
///       but give this its own flags in the future
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
    // TODO: Experiment with `useHostRegistrationForOut` to see if we want to
    //       keep the feature at all (either through a heuristic or compiler
    //       option for gpu codegen).
    Location loc = forallOp->getLoc();
    SmallVector<Value> args;
    SmallVector<Value> tokens;
    Value out = genParametersIn(rewriter, loc, scalars, buffers, args, tokens,
                                /*useHostRegistrationForOut=*/false);
    // Set up GPU module and construct GPU function.
    auto saveIp = rewriter.saveInsertionPoint();
    ModuleOp topModule = forallOp->getParentOfType<ModuleOp>();
    auto gpuModule = genGPUModule(rewriter, topModule);
    auto gpuFunc = genGPUFunc(rewriter, gpuModule, args);
    genGPUCode(rewriter, gpuFunc, forallOp, constants, scalars, buffers);
    // Generate code that launches the kernel asynchronously, blocking on all
    // opens tokens and yielding a new token for the output.
    // TODO: Passing in tokens to launch up does not seem to be properly lowered
    //       by cubin yet, hence the current blocking wait.
    rewriter.restoreInsertionPoint(saveIp);
    genBlockingWait(rewriter, loc, tokens);
    tokens.clear();
    Value kernelToken =
        genLaunchGPUFunc(rewriter, gpuFunc, args, tokens, numThreads);
    // Finalize the outlined arguments.
    genParametersOut(rewriter, loc, out, kernelToken, scalars, buffers, args,
                     tokens);
    genBlockingWait(rewriter, loc, tokens);
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
