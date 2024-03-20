//===- LegalizeVectorStorage.cpp - Ensures SVE loads/stores are legal -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::arm_sve {
#define GEN_PASS_DEF_LEGALIZEVECTORSTORAGE
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h.inc"
} // namespace mlir::arm_sve

using namespace mlir;
using namespace mlir::arm_sve;

// A tag to mark unrealized_conversions produced by this pass. This is used to
// detect IR this pass failed to completely legalize, and report an error.
// If everything was successfully legalized, no tagged ops will remain after
// this pass.
constexpr StringLiteral kSVELegalizerTag("__arm_sve_legalize_vector_storage__");

/// Definitions:
///
/// [1] svbool = vector<...x[16]xi1>, which maps to some multiple of full SVE
/// predicate registers. A full predicate is the smallest quantity that can be
/// loaded/stored.
///
/// [2] SVE mask = hardware-sized SVE predicate mask, i.e. its trailing
/// dimension matches the size of a legal SVE vector size (such as
/// vector<[4]xi1>), but is too small to be stored to memory (i.e smaller than
/// a svbool).

namespace {

/// Checks if a vector type is a SVE mask [2].
bool isSVEMaskType(VectorType type) {
  return type.getRank() > 0 && type.getElementType().isInteger(1) &&
         type.getScalableDims().back() && type.getShape().back() < 16 &&
         llvm::isPowerOf2_32(type.getShape().back()) &&
         !llvm::is_contained(type.getScalableDims().drop_back(), true);
}

VectorType widenScalableMaskTypeToSvbool(VectorType type) {
  assert(isSVEMaskType(type));
  return VectorType::Builder(type).setDim(type.getRank() - 1, 16);
}

/// A helper for cloning an op and replacing it will a new version, updated by a
/// callback.
template <typename TOp, typename TLegalizerCallback>
void replaceOpWithLegalizedOp(PatternRewriter &rewriter, TOp op,
                              TLegalizerCallback callback) {
  // Clone the previous op to preserve any properties/attributes.
  auto newOp = op.clone();
  rewriter.insert(newOp);
  rewriter.replaceOp(op, callback(newOp));
}

/// A helper for cloning an op and replacing it with a new version, updated by a
/// callback, and an unrealized conversion back to the type of the replaced op.
template <typename TOp, typename TLegalizerCallback>
void replaceOpWithUnrealizedConversion(PatternRewriter &rewriter, TOp op,
                                       TLegalizerCallback callback) {
  replaceOpWithLegalizedOp(rewriter, op, [&](TOp newOp) {
    // Mark our `unrealized_conversion_casts` with a pass label.
    return rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange{op.getResult().getType()},
        ValueRange{callback(newOp)},
        NamedAttribute(rewriter.getStringAttr(kSVELegalizerTag),
                       rewriter.getUnitAttr()));
  });
}

/// Extracts the widened SVE memref value (that's legal to store/load) from the
/// `unrealized_conversion_cast`s added by this pass.
static FailureOr<Value> getSVELegalizedMemref(Value illegalMemref) {
  Operation *definingOp = illegalMemref.getDefiningOp();
  if (!definingOp || !definingOp->hasAttr(kSVELegalizerTag))
    return failure();
  auto unrealizedConversion =
      llvm::cast<UnrealizedConversionCastOp>(definingOp);
  return unrealizedConversion.getOperand(0);
}

/// The default alignment of an alloca in LLVM may request overaligned sizes for
/// SVE types, which will fail during stack frame allocation. This rewrite
/// explicitly adds a reasonable alignment to allocas of scalable types.
struct RelaxScalableVectorAllocaAlignment
    : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const override {
    auto memrefElementType = allocaOp.getType().getElementType();
    auto vectorType = llvm::dyn_cast<VectorType>(memrefElementType);
    if (!vectorType || !vectorType.isScalable() || allocaOp.getAlignment())
      return failure();

    // Set alignment based on the defaults for SVE vectors and predicates.
    unsigned aligment = vectorType.getElementType().isInteger(1) ? 2 : 16;
    rewriter.modifyOpInPlace(allocaOp,
                             [&] { allocaOp.setAlignment(aligment); });

    return success();
  }
};

/// Replaces allocations of SVE predicates smaller than an svbool [1] (_illegal_
/// to load/store) with a wider allocation of svbool (_legal_ to load/store)
/// followed by a tagged unrealized conversion to the original type.
///
/// Example
/// ```
/// %alloca = memref.alloca() : memref<vector<[4]xi1>>
/// ```
/// is rewritten into:
/// ```
/// %widened = memref.alloca() {alignment = 1 : i64} : memref<vector<[16]xi1>>
/// %alloca = builtin.unrealized_conversion_cast %widened
///   : memref<vector<[16]xi1>> to memref<vector<[4]xi1>>
///     {__arm_sve_legalize_vector_storage__}
/// ```
template <typename AllocLikeOp>
struct LegalizeSVEMaskAllocation : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp allocLikeOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType =
        llvm::dyn_cast<VectorType>(allocLikeOp.getType().getElementType());

    if (!vectorType || !isSVEMaskType(vectorType))
      return failure();

    // Replace this alloc-like op of an SVE mask [2] with one of a (storable)
    // svbool mask [1]. A temporary unrealized_conversion_cast is added to the
    // old type to allow local rewrites.
    replaceOpWithUnrealizedConversion(
        rewriter, allocLikeOp, [&](AllocLikeOp newAllocLikeOp) {
          newAllocLikeOp.getResult().setType(
              llvm::cast<MemRefType>(newAllocLikeOp.getType().cloneWith(
                  {}, widenScalableMaskTypeToSvbool(vectorType))));
          return newAllocLikeOp;
        });

    return success();
  }
};

/// Replaces vector.type_casts of unrealized conversions to SVE predicate memref
/// types that are _illegal_ to load/store from (!= svbool [1]), with type casts
/// of memref types that are _legal_ to load/store, followed by unrealized
/// conversions.
///
/// Example:
/// ```
/// %alloca = builtin.unrealized_conversion_cast %widened
///   : memref<vector<[16]xi1>> to memref<vector<[8]xi1>>
///     {__arm_sve_legalize_vector_storage__}
/// %cast = vector.type_cast %alloca
///   : memref<vector<3x[8]xi1>> to memref<3xvector<[8]xi1>>
/// ```
/// is rewritten into:
/// ```
/// %widened_cast = vector.type_cast %widened
///   : memref<vector<3x[16]xi1>> to memref<3xvector<[16]xi1>>
/// %cast = builtin.unrealized_conversion_cast %widened_cast
///   : memref<3xvector<[16]xi1>> to memref<3xvector<[8]xi1>>
///     {__arm_sve_legalize_vector_storage__}
/// ```
struct LegalizeSVEMaskTypeCastConversion
    : public OpRewritePattern<vector::TypeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TypeCastOp typeCastOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = typeCastOp.getResultMemRefType();
    auto vectorType = llvm::dyn_cast<VectorType>(resultType.getElementType());

    if (!vectorType || !isSVEMaskType(vectorType))
      return failure();

    auto legalMemref = getSVELegalizedMemref(typeCastOp.getMemref());
    if (failed(legalMemref))
      return failure();

    // Replace this vector.type_cast with one of a (storable) svbool mask [1].
    replaceOpWithUnrealizedConversion(
        rewriter, typeCastOp, [&](vector::TypeCastOp newTypeCast) {
          newTypeCast.setOperand(*legalMemref);
          newTypeCast.getResult().setType(
              llvm::cast<MemRefType>(newTypeCast.getType().cloneWith(
                  {}, widenScalableMaskTypeToSvbool(vectorType))));
          return newTypeCast;
        });

    return success();
  }
};

/// Replaces stores to unrealized conversions to SVE predicate memref types that
/// are _illegal_ to load/store from (!= svbool [1]), with
/// `arm_sve.convert_to_svbool`s followed by (legal) wider stores.
///
/// Example:
/// ```
/// memref.store %mask, %alloca[] : memref<vector<[8]xi1>>
/// ```
/// is rewritten into:
/// ```
/// %svbool = arm_sve.convert_to_svbool %mask : vector<[8]xi1>
/// memref.store %svbool, %widened[] : memref<vector<[16]xi1>>
/// ```
struct LegalizeSVEMaskStoreConversion
    : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();

    Value valueToStore = storeOp.getValueToStore();
    auto vectorType = llvm::dyn_cast<VectorType>(valueToStore.getType());

    if (!vectorType || !isSVEMaskType(vectorType))
      return failure();

    auto legalMemref = getSVELegalizedMemref(storeOp.getMemref());
    if (failed(legalMemref))
      return failure();

    auto legalMaskType = widenScalableMaskTypeToSvbool(
        llvm::cast<VectorType>(valueToStore.getType()));
    auto convertToSvbool = rewriter.create<arm_sve::ConvertToSvboolOp>(
        loc, legalMaskType, valueToStore);
    // Replace this store with a conversion to a storable svbool mask [1],
    // followed by a wider store.
    replaceOpWithLegalizedOp(rewriter, storeOp,
                             [&](memref::StoreOp newStoreOp) {
                               newStoreOp.setOperand(0, convertToSvbool);
                               newStoreOp.setOperand(1, *legalMemref);
                               return newStoreOp;
                             });

    return success();
  }
};

/// Replaces loads from unrealized conversions to SVE predicate memref types
/// that are _illegal_ to load/store from (!= svbool [1]), types with (legal)
/// wider loads, followed by `arm_sve.convert_from_svbool`s.
///
/// Example:
/// ```
/// %reload = memref.load %alloca[] : memref<vector<[4]xi1>>
/// ```
/// is rewritten into:
/// ```
/// %svbool = memref.load %widened[] : memref<vector<[16]xi1>>
/// %reload = arm_sve.convert_from_svbool %reload : vector<[4]xi1>
/// ```
struct LegalizeSVEMaskLoadConversion : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    Value loadedMask = loadOp.getResult();
    auto vectorType = llvm::dyn_cast<VectorType>(loadedMask.getType());

    if (!vectorType || !isSVEMaskType(vectorType))
      return failure();

    auto legalMemref = getSVELegalizedMemref(loadOp.getMemref());
    if (failed(legalMemref))
      return failure();

    auto legalMaskType = widenScalableMaskTypeToSvbool(vectorType);
    // Replace this load with a legal load of an svbool type, followed by a
    // conversion back to the original type.
    replaceOpWithLegalizedOp(rewriter, loadOp, [&](memref::LoadOp newLoadOp) {
      newLoadOp.setMemRef(*legalMemref);
      newLoadOp.getResult().setType(legalMaskType);
      return rewriter.create<arm_sve::ConvertFromSvboolOp>(
          loc, loadedMask.getType(), newLoadOp);
    });

    return success();
  }
};

} // namespace

void mlir::arm_sve::populateLegalizeVectorStoragePatterns(
    RewritePatternSet &patterns) {
  patterns.add<RelaxScalableVectorAllocaAlignment,
               LegalizeSVEMaskAllocation<memref::AllocaOp>,
               LegalizeSVEMaskAllocation<memref::AllocOp>,
               LegalizeSVEMaskTypeCastConversion,
               LegalizeSVEMaskStoreConversion, LegalizeSVEMaskLoadConversion>(
      patterns.getContext());
}

namespace {
struct LegalizeVectorStorage
    : public arm_sve::impl::LegalizeVectorStorageBase<LegalizeVectorStorage> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLegalizeVectorStoragePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp unrealizedConversion) {
          return !unrealizedConversion->hasAttr(kSVELegalizerTag);
        });
    // This detects if we failed to completely legalize the IR.
    if (failed(applyPartialConversion(getOperation(), target, {})))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arm_sve::createLegalizeVectorStoragePass() {
  return std::make_unique<LegalizeVectorStorage>();
}
