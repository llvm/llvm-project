//===- ACCSpecializeForHost.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts OpenACC operations to host-compatible representations,
// enabling execution on the host rather than on accelerator devices.
//
// Overview:
// ---------
// The pass operates in two modes depending on the `enableHostFallback` option:
//
// 1. Default Mode (Orphan Operations Only):
//    Only converts "orphan" ACC operations that are not inside or attached to
//    compute regions. This is used for host routines (acc routine marked for
//    host) where structured/unstructured data constructs, compute constructs,
//    and their associated data operations should be preserved.
//
// 2. Host Fallback Mode (enableHostFallback=true):
//    Converts ALL ACC operations within the region to host equivalents. This
//    is used when the `if` clause evaluates to false at runtime and the
//    entire ACC region needs to fall back to host execution.
//
// Transformations (Orphan Mode):
// ------------------------------
// The following orphan operations are converted:
//
// 1. Atomic Ops (converted to load/store):
//    acc.atomic.update -> load + compute + store
//    acc.atomic.read -> load + store (copy)
//    acc.atomic.write -> store
//    acc.atomic.capture -> inline region contents
//
// 2. Loop Ops (converted to SCF):
//    acc.loop (structured) -> scf.for
//    acc.loop (unstructured) -> scf.execute_region
//
// 3. Orphan Data Entry Ops (replaced with var operand):
//    acc.cache, acc.private, acc.firstprivate, acc.reduction
//    (only if NOT connected to compute constructs or loop)
//
// Transformations (Host Fallback Mode):
// -------------------------------------
// In addition to orphan transformations, ALL of the following are converted:
//
// 1. Data Entry Ops (replaced with var operand):
//    acc.copyin, acc.create, acc.attach, acc.present, acc.deviceptr,
//    acc.get_deviceptr, acc.nocreate, acc.declare_device_resident,
//    acc.declare_link, acc.use_device, acc.update_device
//
// 2. Data Exit Ops (erased):
//    acc.copyout, acc.delete, acc.detach, acc.update_host
//
// 3. Structured Data/Compute Constructs (region inlined):
//    acc.data, acc.host_data, acc.kernel_environment, acc.declare,
//    acc.parallel, acc.serial, acc.kernels
//
// 4. Unstructured Data Ops (erased):
//    acc.enter_data, acc.exit_data, acc.update
//
// 5. Declare Ops (erased):
//    acc.declare_enter, acc.declare_exit
//
// 6. Runtime Ops (erased):
//    acc.init, acc.shutdown, acc.set, acc.wait, acc.terminator
//
// Requirements:
// -------------
// For atomic operation conversion, variables must implement the
// `acc::PointerLikeType` interface to enable generating load/store operations.
//
// The pass uses `OpenACCSupport::emitNYI()` to report unsupported cases.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsLoop.h"
#include "mlir/Dialect/OpenACC/Transforms/ACCSpecializePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCSPECIALIZEFORHOST
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-specialize-for-host"

using namespace mlir;
using namespace mlir::acc;

/// Check if an operation is inside an ACC compute construct.
static bool isInsideACCComputeConstruct(Operation *op) {
  while ((op = op->getParentOp()))
    if (isa<ACC_COMPUTE_CONSTRUCT_OPS>(op))
      return true;
  return false;
}

namespace {

// Lower orphan acc.atomic.update by: load from addr, clone region expr with
// the loaded value, then store the computed result back to addr.
// Only matches if NOT inside a compute region.
class ACCOrphanAtomicUpdateOpConversion
    : public OpRewritePattern<acc::AtomicUpdateOp> {
public:
  ACCOrphanAtomicUpdateOpConversion(MLIRContext *ctx, OpenACCSupport &support)
      : OpRewritePattern<acc::AtomicUpdateOp>(ctx), accSupport(support) {}

  LogicalResult matchAndRewrite(acc::AtomicUpdateOp atomicUpdateOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not inside an ACC compute construct
    if (isInsideACCComputeConstruct(atomicUpdateOp))
      return failure();

    Value x = atomicUpdateOp.getX();
    Type type = x.getType();
    auto ptrLikeType = dyn_cast<acc::PointerLikeType>(type);
    if (ptrLikeType) {
      auto xTyped = cast<TypedValue<acc::PointerLikeType>>(x);
      rewriter.setInsertionPointAfter(atomicUpdateOp);
      Value loadOp =
          ptrLikeType.genLoad(rewriter, atomicUpdateOp.getLoc(), xTyped, {});
      if (!loadOp) {
        accSupport.emitNYI(atomicUpdateOp.getLoc(),
                           "failed to generate load for atomic update");
        return failure();
      }
      IRMapping mapping;
      mapping.map(atomicUpdateOp.getRegion().front().getArgument(0), loadOp);
      Operation *expr = rewriter.clone(*atomicUpdateOp.getFirstOp(), mapping);
      if (!ptrLikeType.genStore(rewriter, atomicUpdateOp.getLoc(),
                                expr->getResult(0), xTyped)) {
        accSupport.emitNYI(atomicUpdateOp.getLoc(),
                           "failed to generate store for atomic update");
        return failure();
      }
      rewriter.eraseOp(atomicUpdateOp);
    } else {
      accSupport.emitNYI(atomicUpdateOp.getLoc(),
                         "unsupported type for atomic update");
      return failure();
    }
    return success();
  }

private:
  OpenACCSupport &accSupport;
};

// Lower orphan acc.atomic.read by: load from src, then store into dst.
// Only matches if NOT inside an ACC compute construct.
class ACCOrphanAtomicReadOpConversion
    : public OpRewritePattern<acc::AtomicReadOp> {
public:
  ACCOrphanAtomicReadOpConversion(MLIRContext *ctx, OpenACCSupport &support)
      : OpRewritePattern<acc::AtomicReadOp>(ctx), accSupport(support) {}

  LogicalResult matchAndRewrite(acc::AtomicReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not inside an ACC compute construct
    if (isInsideACCComputeConstruct(readOp))
      return failure();

    Value x = readOp.getX();
    Value v = readOp.getV();
    auto xPtrType = dyn_cast<acc::PointerLikeType>(x.getType());
    auto vPtrType = dyn_cast<acc::PointerLikeType>(v.getType());
    if (xPtrType && vPtrType) {
      auto xTyped = cast<TypedValue<acc::PointerLikeType>>(x);
      auto vTyped = cast<TypedValue<acc::PointerLikeType>>(v);
      rewriter.setInsertionPointAfter(readOp);

      // Use genCopy which does load + store
      if (!xPtrType.genCopy(rewriter, readOp.getLoc(), vTyped, xTyped, {})) {
        accSupport.emitNYI(readOp.getLoc(),
                           "failed to generate copy for atomic read");
        return failure();
      }
      rewriter.eraseOp(readOp);
    } else {
      accSupport.emitNYI(readOp.getLoc(), "unsupported type for atomic read");
      return failure();
    }
    return success();
  }

private:
  OpenACCSupport &accSupport;
};

// Lower orphan acc.atomic.write by: store value into addr.
// Only matches if NOT inside an ACC compute construct.
class ACCOrphanAtomicWriteOpConversion
    : public OpRewritePattern<acc::AtomicWriteOp> {
public:
  ACCOrphanAtomicWriteOpConversion(MLIRContext *ctx, OpenACCSupport &support)
      : OpRewritePattern<acc::AtomicWriteOp>(ctx), accSupport(support) {}

  LogicalResult matchAndRewrite(acc::AtomicWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not inside an ACC compute construct
    if (isInsideACCComputeConstruct(writeOp))
      return failure();

    Value x = writeOp.getX();
    Value expr = writeOp.getExpr();
    auto ptrLikeType = dyn_cast<acc::PointerLikeType>(x.getType());
    if (ptrLikeType) {
      auto xTyped = cast<TypedValue<acc::PointerLikeType>>(x);
      rewriter.setInsertionPointAfter(writeOp);
      if (!ptrLikeType.genStore(rewriter, writeOp.getLoc(), expr, xTyped)) {
        accSupport.emitNYI(writeOp.getLoc(),
                           "failed to generate store for atomic write");
        return failure();
      }
      rewriter.eraseOp(writeOp);
    } else {
      accSupport.emitNYI(writeOp.getLoc(), "unsupported type for atomic write");
      return failure();
    }
    return success();
  }

private:
  OpenACCSupport &accSupport;
};

// Lower orphan acc.atomic.capture by: unwrap the capture region and erase the
// wrapper; inner ops are lowered in-order (e.g., read+update becomes load/store
// to dst then load/compute/store to addr).
// Only matches if NOT inside an ACC compute construct.
class ACCOrphanAtomicCaptureOpConversion
    : public OpRewritePattern<acc::AtomicCaptureOp> {
  using OpRewritePattern<acc::AtomicCaptureOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::AtomicCaptureOp captureOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not inside an ACC compute construct
    if (isInsideACCComputeConstruct(captureOp))
      return failure();

    assert(captureOp.getRegion().hasOneBlock() && "expected one block");
    Block *block = &captureOp.getRegion().front();
    // Remove the terminator before inlining
    rewriter.eraseOp(block->getTerminator());
    rewriter.inlineBlockBefore(block, captureOp);
    rewriter.eraseOp(captureOp);
    return success();
  }
};

// Convert orphan acc.loop to scf.for or scf.execute_region.
// Only matches if NOT inside an ACC compute construct.
class ACCOrphanLoopOpConversion : public OpRewritePattern<acc::LoopOp> {
  using OpRewritePattern<acc::LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(acc::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not inside an ACC compute construct
    if (isInsideACCComputeConstruct(loopOp))
      return failure();

    if (loopOp.getUnstructured()) {
      auto executeRegion =
          acc::convertUnstructuredACCLoopToSCFExecuteRegion(loopOp, rewriter);
      if (!executeRegion)
        return failure();
      rewriter.replaceOp(loopOp, executeRegion);
    } else {
      auto forOp =
          acc::convertACCLoopToSCFFor(loopOp, /*enableCollapse=*/false);
      if (!forOp)
        return failure();
      rewriter.replaceOp(loopOp, forOp);
    }
    return success();
  }
};

/// Check if an operation is used by a compute construct or loop op
static bool isUsedByComputeOrLoop(Operation *op) {
  for (auto *user : op->getUsers())
    if (isa<acc::ParallelOp, acc::SerialOp, acc::KernelsOp, acc::LoopOp>(user))
      return true;
  return false;
}

/// Orphan data entry ops - only match if NOT connected to compute/loop and
/// NOT inside a compute region. Used for acc.cache, acc.private,
/// acc.firstprivate, acc.reduction.
template <typename OpTy>
class ACCOrphanDataEntryConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Only convert if this op is not used by a compute construct or loop,
    // and not inside an ACC compute construct.
    if (isUsedByComputeOrLoop(op) || isInsideACCComputeConstruct(op))
      return failure();

    if (op->use_empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, op.getVar());
    return success();
  }
};

class ACCSpecializeForHost
    : public acc::impl::ACCSpecializeForHostBase<ACCSpecializeForHost> {
public:
  using ACCSpecializeForHostBase<
      ACCSpecializeForHost>::ACCSpecializeForHostBase;

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Enter ACCSpecializeForHost()\n");

    func::FuncOp funcOp = getOperation();
    if (!acc::isSpecializedAccRoutine(funcOp)) {
      // Convert orphan operations to host, or all ACC operations if
      // host fallback patterns are enabled.
      auto *context = &getContext();
      RewritePatternSet patterns(context);
      OpenACCSupport &accSupport = getAnalysis<OpenACCSupport>();
      if (enableHostFallback)
        populateACCHostFallbackPatterns(patterns, accSupport);
      else
        populateACCOrphanToHostPatterns(patterns, accSupport);
      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config)))
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Exit ACCSpecializeForHost()\n");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population functions
//===----------------------------------------------------------------------===//

void mlir::acc::populateACCOrphanToHostPatterns(RewritePatternSet &patterns,
                                                OpenACCSupport &accSupport,
                                                bool enableLoopConversion) {
  MLIRContext *context = patterns.getContext();

  // For host routines (acc routine marked for host), we only convert orphan
  // operations that are not allowed outside compute regions. All patterns
  // here check that the operation is NOT inside a compute region before
  // converting:
  // - acc.atomic.* -> load/store operations
  // - acc.loop -> scf.for or scf.execute_region
  // - acc.cache -> replaced with var
  // - acc.private, acc.reduction, acc.firstprivate -> replaced with var
  //   (only if NOT connected to compute constructs or loop)
  //
  // We do NOT remove structured/unstructured data constructs, compute
  // constructs, or their associated data operations - those are valid
  // in host routines and will be processed by other passes.

  // Loop conversion (orphan only)
  if (enableLoopConversion)
    patterns.insert<ACCOrphanLoopOpConversion>(context);

  // Atomic operations - convert to non-atomic load/store (orphan only)
  patterns.insert<ACCOrphanAtomicUpdateOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicReadOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicWriteOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicCaptureOpConversion>(context);

  // Orphan data entry ops - only convert if NOT connected to compute/loop
  // and NOT inside a compute region
  patterns.insert<ACCOrphanDataEntryConversion<acc::CacheOp>,
                  ACCOrphanDataEntryConversion<acc::PrivateOp>,
                  ACCOrphanDataEntryConversion<acc::FirstprivateOp>,
                  ACCOrphanDataEntryConversion<acc::ReductionOp>>(context);
}

void mlir::acc::populateACCHostFallbackPatterns(RewritePatternSet &patterns,
                                                OpenACCSupport &accSupport,
                                                bool enableLoopConversion) {
  MLIRContext *context = patterns.getContext();

  // For host fallback path (when `if` clause evaluates to false), ALL ACC
  // operations within the region should be converted to host equivalents.
  // This includes structured/unstructured data, compute constructs, and
  // their associated data operations.

  // Loop conversion - OK to use the orphan loop conversion pattern here
  // because the parent compute constructs will also be converted.
  if (enableLoopConversion)
    patterns.insert<ACCOrphanLoopOpConversion>(context);

  // Atomic operations - convert to non-atomic load/store. OK to use the orphan
  // atomic conversion patterns here because the parent compute constructs will
  // also be converted.
  patterns.insert<ACCOrphanAtomicUpdateOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicReadOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicWriteOpConversion>(context, accSupport);
  patterns.insert<ACCOrphanAtomicCaptureOpConversion>(context);

  // acc.cache - convert ALL cache ops (including those inside compute regions)
  patterns.insert<ACCOpReplaceWithVarConversion<acc::CacheOp>>(context);

  // Privatization ops - convert ALL (including those attached to compute/loop)
  patterns.insert<ACCOpReplaceWithVarConversion<acc::PrivateOp>,
                  ACCOpReplaceWithVarConversion<acc::FirstprivateOp>,
                  ACCOpReplaceWithVarConversion<acc::ReductionOp>>(context);

  // Data entry ops - replaced with their var operand
  patterns.insert<ACCOpReplaceWithVarConversion<acc::CopyinOp>,
                  ACCOpReplaceWithVarConversion<acc::CreateOp>,
                  ACCOpReplaceWithVarConversion<acc::AttachOp>,
                  ACCOpReplaceWithVarConversion<acc::PresentOp>,
                  ACCOpReplaceWithVarConversion<acc::DevicePtrOp>,
                  ACCOpReplaceWithVarConversion<acc::GetDevicePtrOp>,
                  ACCOpReplaceWithVarConversion<acc::NoCreateOp>,
                  ACCOpReplaceWithVarConversion<acc::DeclareDeviceResidentOp>,
                  ACCOpReplaceWithVarConversion<acc::DeclareLinkOp>,
                  ACCOpReplaceWithVarConversion<acc::UseDeviceOp>,
                  ACCOpReplaceWithVarConversion<acc::UpdateDeviceOp>>(context);

  // Data exit ops - simply erased (no results)
  patterns.insert<ACCOpEraseConversion<acc::CopyoutOp>,
                  ACCOpEraseConversion<acc::DeleteOp>,
                  ACCOpEraseConversion<acc::DetachOp>,
                  ACCOpEraseConversion<acc::UpdateHostOp>>(context);

  // Structured data constructs - unwrap their regions
  patterns.insert<ACCRegionUnwrapConversion<acc::DataOp>,
                  ACCRegionUnwrapConversion<acc::HostDataOp>,
                  ACCRegionUnwrapConversion<acc::KernelEnvironmentOp>>(context);

  // Declare ops
  patterns.insert<ACCDeclareEnterOpConversion,
                  ACCRegionUnwrapConversion<acc::DeclareOp>>(context);

  // Unstructured data operations - erase them
  patterns.insert<ACCOpEraseConversion<acc::EnterDataOp>,
                  ACCOpEraseConversion<acc::ExitDataOp>,
                  ACCOpEraseConversion<acc::UpdateOp>>(context);

  // Runtime operations - erase them
  patterns.insert<
      ACCOpEraseConversion<acc::InitOp>, ACCOpEraseConversion<acc::ShutdownOp>,
      ACCOpEraseConversion<acc::SetOp>, ACCOpEraseConversion<acc::WaitOp>,
      ACCOpEraseConversion<acc::TerminatorOp>>(context);

  // Compute constructs - unwrap their regions
  patterns.insert<ACCRegionUnwrapConversion<acc::ParallelOp>,
                  ACCRegionUnwrapConversion<acc::SerialOp>,
                  ACCRegionUnwrapConversion<acc::KernelsOp>>(context);
}
