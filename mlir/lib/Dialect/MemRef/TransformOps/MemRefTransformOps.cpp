//===- MemRefTransformOps.cpp - Implementation of Memref transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "memref-transforms"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

std::unique_ptr<TypeConverter>
transform::MemrefToLLVMTypeConverterOp::getTypeConverter() {
  LowerToLLVMOptions options(getContext());
  options.allocLowering =
      (getUseAlignedAlloc() ? LowerToLLVMOptions::AllocLowering::AlignedAlloc
                            : LowerToLLVMOptions::AllocLowering::Malloc);
  options.useGenericFunctions = getUseGenericFunctions();
  options.useOpaquePointers = getUseOpaquePointers();

  if (getIndexBitwidth() != kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(getIndexBitwidth());

  // TODO: the following two options don't really make sense for
  // memref_to_llvm_type_converter specifically but we should have a single
  // to_llvm_type_converter.
  if (getDataLayout().has_value())
    options.dataLayout = llvm::DataLayout(getDataLayout().value());
  options.useBarePtrCallConv = getUseBarePtrCallConv();

  return std::make_unique<LLVMTypeConverter>(getContext(), options);
}

StringRef transform::MemrefToLLVMTypeConverterOp::getTypeConverterType() {
  return "LLVMTypeConverter";
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyExpandOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  memref::populateExpandOpsPatterns(patterns);
}

void transform::ApplyExpandStridedMetadataPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  memref::populateExpandStridedMetadataPatterns(patterns);
}

void transform::ApplyExtractAddressComputationsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  memref::populateExtractAddressComputationsPatterns(patterns);
}

void transform::ApplyFoldMemrefAliasOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  memref::populateFoldMemRefAliasOpPatterns(patterns);
}

void transform::ApplyResolveRankedShapedTypeResultDimsPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// MemRefMultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MemRefMultiBufferOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  SmallVector<Operation *> results;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    bool canApplyMultiBuffer = true;
    auto target = cast<memref::AllocOp>(op);
    LLVM_DEBUG(DBGS() << "Start multibuffer transform op: " << target << "\n";);
    // Skip allocations not used in a loop.
    for (Operation *user : target->getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;
      auto loop = user->getParentOfType<LoopLikeOpInterface>();
      if (!loop) {
        LLVM_DEBUG(DBGS() << "--allocation not used in a loop\n";
                   DBGS() << "----due to user: " << *user;);
        canApplyMultiBuffer = false;
        break;
      }
    }
    if (!canApplyMultiBuffer) {
      LLVM_DEBUG(DBGS() << "--cannot apply multibuffering -> Skip\n";);
      continue;
    }

    auto newBuffer =
        memref::multiBuffer(rewriter, target, getFactor(), getSkipAnalysis());

    if (failed(newBuffer)) {
      LLVM_DEBUG(DBGS() << "--op failed to multibuffer\n";);
      return emitSilenceableFailure(target->getLoc())
             << "op failed to multibuffer";
    }

    results.push_back(*newBuffer);
  }
  transformResults.set(cast<OpResult>(getResult()), results);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MemRefEraseDeadAllocAndStoresOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MemRefEraseDeadAllocAndStoresOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Apply store to load forwarding and dead store elimination.
  vector::transferOpflowOpt(rewriter, target);
  memref::eraseDeadAllocAndStores(rewriter, target);
  return DiagnosedSilenceableFailure::success();
}

void transform::MemRefEraseDeadAllocAndStoresOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}
void transform::MemRefEraseDeadAllocAndStoresOp::build(OpBuilder &builder,
                                                       OperationState &result,
                                                       Value target) {
  result.addOperands(target);
}

//===----------------------------------------------------------------------===//
// MemRefMakeLoopIndependentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MemRefMakeLoopIndependentOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Gather IVs.
  SmallVector<Value> ivs;
  Operation *nextOp = target;
  for (uint64_t i = 0, e = getNumLoops(); i < e; ++i) {
    nextOp = nextOp->getParentOfType<scf::ForOp>();
    if (!nextOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "could not find " << i
                                         << "-th enclosing loop";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    ivs.push_back(cast<scf::ForOp>(nextOp).getInductionVar());
  }

  // Rewrite IR.
  FailureOr<Value> replacement = failure();
  if (auto allocaOp = dyn_cast<memref::AllocaOp>(target)) {
    replacement = memref::replaceWithIndependentOp(rewriter, allocaOp, ivs);
  } else {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "unsupported target op";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  if (failed(replacement)) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError() << "could not make target op loop-independent";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  results.push_back(replacement->getDefiningOp());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MemRefTransformDialectExtension
    : public transform::TransformDialectExtension<
          MemRefTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<nvgpu::NVGPUDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"

void mlir::memref::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<MemRefTransformDialectExtension>();
}
