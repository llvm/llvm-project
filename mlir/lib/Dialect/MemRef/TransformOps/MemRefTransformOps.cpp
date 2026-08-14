//===- MemRefTransformOps.cpp - Implementation of Memref transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
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
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
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

namespace {
class AllocToAllocaPattern : public OpRewritePattern<memref::AllocOp> {
public:
  explicit AllocToAllocaPattern(Operation *analysisRoot, int64_t maxSize = 0)
      : OpRewritePattern<memref::AllocOp>(analysisRoot->getContext()),
        dataLayoutAnalysis(analysisRoot), maxSize(maxSize) {}

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    return success(memref::allocToAlloca(
        rewriter, op, [this](memref::AllocOp alloc, memref::DeallocOp dealloc) {
          MemRefType type = alloc.getMemref().getType();
          if (!type.hasStaticShape())
            return false;

          const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(alloc);
          int64_t elementSize = dataLayout.getTypeSize(type.getElementType());
          return maxSize == 0 || type.getNumElements() * elementSize < maxSize;
        }));
  }

private:
  DataLayoutAnalysis dataLayoutAnalysis;
  int64_t maxSize;
};
} // namespace

void transform::ApplyAllocToAllocaOp::populatePatterns(
    RewritePatternSet &patterns) {}

void transform::ApplyAllocToAllocaOp::populatePatternsWithState(
    RewritePatternSet &patterns, transform::TransformState &state) {
  patterns.insert<AllocToAllocaPattern>(
      state.getTopLevel(), static_cast<int64_t>(getSizeLimit().value_or(0)));
}

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
// Alloc and alloca to global utilities
//===----------------------------------------------------------------------===//

/// Checks whether an allocation operation can be converted to a
/// `memref.global`.
template <typename AllocLikeOp>
static DiagnosedSilenceableFailure
checkAllocToGlobalPreconditions(AllocLikeOp allocLikeOp) {
  MemRefType memrefType = allocLikeOp.getType();
  if (!memrefType.hasStaticShape()) {
    return emitSilenceableFailure(allocLikeOp)
           << "conversion to a global op requires statically shaped memrefs, "
              "but got "
           << memrefType;
  }

  if (!allocLikeOp.getSymbolOperands().empty()) {
    return emitSilenceableFailure(allocLikeOp)
           << "conversion to a global op does not support symbol operands, but "
              "got "
           << memrefType;
  }

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(memrefType.getStridesAndOffset(strides, offset))) {
    return emitSilenceableFailure(allocLikeOp)
           << "conversion to a global op requires strided layout, but got "
           << memrefType;
  }
  if (!ShapedType::isStatic(offset) || !ShapedType::isStaticShape(strides)) {
    return emitSilenceableFailure(allocLikeOp)
           << "conversion to a global op does not support dynamic offset or "
              "strides, but got "
           << memrefType;
  }

  return DiagnosedSilenceableFailure::success();
}

/// Converts an allocation operation (`memref.alloca` or `memref.alloc`) to a
/// `memref.global` operation in the nearest symbol table, and replaces the
/// allocation with a `memref.get_global` operation. Any `memref.dealloc`
/// operations referencing the allocation are erased.
template <typename AllocLikeOp>
static DiagnosedSilenceableFailure
allocLikeToGlobal(transform::TransformRewriter &rewriter,
                  AllocLikeOp allocLikeOp, StringRef globalName,
                  memref::GlobalOp &globalOp,
                  memref::GetGlobalOp &getGlobalOp) {
  if (DiagnosedSilenceableFailure failure =
          checkAllocToGlobalPreconditions(allocLikeOp);
      !failure.succeeded())
    return failure;

  MLIRContext *ctx = rewriter.getContext();
  Location loc = allocLikeOp->getLoc();

  // Find nearest symbol table.
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(allocLikeOp);
  assert(symbolTableOp && "expected payload to be in symbol table");
  SymbolTable symbolTable(symbolTableOp);

  // Insert a `memref.global` into the symbol table.
  Type resultType = allocLikeOp.getResult().getType();
  OpBuilder builder(rewriter.getContext());
  // TODO: Add a better builder for this.
  globalOp = memref::GlobalOp::create(
      builder, loc, StringAttr::get(ctx, globalName),
      StringAttr::get(ctx, "private"), TypeAttr::get(resultType), Attribute{},
      UnitAttr{}, allocLikeOp.getAlignmentAttr());
  symbolTable.insert(globalOp);

  // Remove any `memref.dealloc` operations referencing this allocation.
  // We assume that the allocation does not escape the current container
  // (e.g., via return or interprocedural function calls) and is not passed
  // through control-flow or alias operations (e.g., `scf.if`, `cf.cond_br`,
  // `select`, `memref.subview`), so any deallocation is a direct user of the
  // allocation. Indirect deallocations are not removed and must be handled
  // separately.
  for (Operation *user : llvm::make_early_inc_range(allocLikeOp->getUsers())) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
      rewriter.eraseOp(dealloc);
  }

  // Replace the allocation with a `memref.get_global` accessing the
  // global symbol inserted above.
  rewriter.setInsertionPoint(allocLikeOp);
  getGlobalOp = rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
      allocLikeOp, globalOp.getType(), globalOp.getName());

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// AllocaToGlobalOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MemRefAllocaToGlobalOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {
  auto allocaOps = state.getPayloadOps(getAlloca());

  SmallVector<memref::GlobalOp> globalOps;
  SmallVector<memref::GetGlobalOp> getGlobalOps;

  // Transform `memref.alloca`s.
  for (auto *op : allocaOps) {
    auto alloca = cast<memref::AllocaOp>(op);
    memref::GlobalOp globalOp;
    memref::GetGlobalOp getGlobalOp;
    DiagnosedSilenceableFailure diag =
        allocLikeToGlobal(rewriter, alloca, "alloca", globalOp, getGlobalOp);
    if (!diag.succeeded())
      return diag;

    globalOps.push_back(globalOp);
    getGlobalOps.push_back(getGlobalOp);
  }

  // Assemble results.
  results.set(cast<OpResult>(getGlobal()), globalOps);
  results.set(cast<OpResult>(getGetGlobal()), getGlobalOps);

  return DiagnosedSilenceableFailure::success();
}

void transform::MemRefAllocaToGlobalOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
  consumesHandle(getAllocaMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// AllocToGlobalOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MemRefAllocToGlobalOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  auto allocOps = state.getPayloadOps(getAlloc());

  SmallVector<memref::GlobalOp> globalOps;
  SmallVector<memref::GetGlobalOp> getGlobalOps;

  // Transform `memref.alloc`s.
  for (auto *op : allocOps) {
    auto alloc = cast<memref::AllocOp>(op);
    memref::GlobalOp globalOp;
    memref::GetGlobalOp getGlobalOp;
    DiagnosedSilenceableFailure diag =
        allocLikeToGlobal(rewriter, alloc, "alloc", globalOp, getGlobalOp);
    if (!diag.succeeded())
      return diag;

    globalOps.push_back(globalOp);
    getGlobalOps.push_back(getGlobalOp);
  }

  // Assemble results.
  results.set(cast<OpResult>(getGlobal()), globalOps);
  results.set(cast<OpResult>(getGetGlobal()), getGlobalOps);

  return DiagnosedSilenceableFailure::success();
}

void transform::MemRefAllocToGlobalOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
  consumesHandle(getAllocMutable(), effects);
  modifiesPayload(effects);
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
  transform::onlyReadsHandle(getTargetMutable(), effects);
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefTransformDialectExtension)

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
