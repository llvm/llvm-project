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

constexpr StringLiteral kPassLabel("__arm_sve_legalize_vector_storage__");

namespace {

/// A (legal) SVE predicate mask that has a logical size, i.e. the number of
/// bits match the number of lanes it masks (such as vector<[4]xi1>), but is too
/// small to be stored to memory.
bool isLogicalSVEPredicateType(VectorType type) {
  return type.getRank() > 0 && type.getElementType().isInteger(1) &&
         type.getScalableDims().back() && type.getShape().back() < 16 &&
         llvm::isPowerOf2_32(type.getShape().back()) &&
         !llvm::is_contained(type.getScalableDims().drop_back(), true);
}

VectorType widenScalableMaskTypeToSvbool(VectorType type) {
  assert(isLogicalSVEPredicateType(type));
  return VectorType::Builder(type).setDim(type.getRank() - 1, 16);
}

template <typename TOp, typename TLegalizerCallback>
void replaceOpWithLegalizedOp(PatternRewriter &rewriter, TOp op,
                              TLegalizerCallback callback) {
  // Clone the previous op to preserve any properties/attributes.
  auto newOp = op.clone();
  rewriter.insert(newOp);
  rewriter.replaceOp(op, callback(newOp));
}

template <typename TOp, typename TLegalizerCallback>
void replaceOpWithUnrealizedConversion(PatternRewriter &rewriter, TOp op,
                                       TLegalizerCallback callback) {
  replaceOpWithLegalizedOp(rewriter, op, [&](TOp newOp) {
    // Mark our `unrealized_conversion_casts` with a pass label.
    return rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), TypeRange{op.getResult().getType()},
        ValueRange{callback(newOp)},
        NamedAttribute(rewriter.getStringAttr(kPassLabel),
                       rewriter.getUnitAttr()));
  });
}

/// Extracts the legal memref value from the `unrealized_conversion_casts` added
/// by this pass.
static FailureOr<Value> getLegalMemRef(Value illegalMemref) {
  Operation *definingOp = illegalMemref.getDefiningOp();
  if (!definingOp || !definingOp->hasAttr(kPassLabel))
    return failure();
  auto unrealizedConversion =
      llvm::cast<UnrealizedConversionCastOp>(definingOp);
  return unrealizedConversion.getOperand(0);
}

/// The default alignment of an alloca may request overaligned sizes for SVE
/// types, which will fail during stack frame allocation. This rewrite
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
    allocaOp.setAlignment(aligment);

    return success();
  }
};

/// Replaces allocations of SVE predicates smaller than an svbool with a wider
/// allocation and a tagged unrealized conversion.
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
struct LegalizeAllocLikeOpConversion : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp allocLikeOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType =
        llvm::dyn_cast<VectorType>(allocLikeOp.getType().getElementType());

    if (!vectorType || !isLogicalSVEPredicateType(vectorType))
      return failure();

    // Replace this alloc-like op of an SVE mask with one of a (storable)
    // svbool_t mask. A temporary unrealized_conversion_cast is added to the old
    // type to allow local rewrites.
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

/// Replaces vector.type_casts of unrealized conversions to illegal memref types
/// with legal type casts, followed by unrealized conversions.
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
struct LegalizeVectorTypeCastConversion
    : public OpRewritePattern<vector::TypeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TypeCastOp typeCastOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = typeCastOp.getResultMemRefType();
    auto vectorType = llvm::dyn_cast<VectorType>(resultType.getElementType());

    if (!vectorType || !isLogicalSVEPredicateType(vectorType))
      return failure();

    auto legalMemref = getLegalMemRef(typeCastOp.getMemref());
    if (failed(legalMemref))
      return failure();

    // Replace this vector.type_cast with one of a (storable) svbool_t mask.
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

/// Replaces stores to unrealized conversions to illegal memref types with
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
struct LegalizeMemrefStoreConversion
    : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();

    Value valueToStore = storeOp.getValueToStore();
    auto vectorType = llvm::dyn_cast<VectorType>(valueToStore.getType());

    if (!vectorType || !isLogicalSVEPredicateType(vectorType))
      return failure();

    auto legalMemref = getLegalMemRef(storeOp.getMemref());
    if (failed(legalMemref))
      return failure();

    auto legalMaskType = widenScalableMaskTypeToSvbool(
        llvm::cast<VectorType>(valueToStore.getType()));
    auto convertToSvbool = rewriter.create<arm_sve::ConvertToSvboolOp>(
        loc, legalMaskType, valueToStore);
    // Replace this store with a conversion to a storable svbool_t mask,
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

/// Replaces loads from unrealized conversions to illegal memref types with
/// (legal) wider loads, followed by `arm_sve.convert_from_svbool`s.
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
struct LegalizeMemrefLoadConversion : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    Value loadedMask = loadOp.getResult();
    auto vectorType = llvm::dyn_cast<VectorType>(loadedMask.getType());

    if (!vectorType || !isLogicalSVEPredicateType(vectorType))
      return failure();

    auto legalMemref = getLegalMemRef(loadOp.getMemref());
    if (failed(legalMemref))
      return failure();

    auto legalMaskType = widenScalableMaskTypeToSvbool(vectorType);
    // Replace this load with a legal load of an svbool_t type, followed by a
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
               LegalizeAllocLikeOpConversion<memref::AllocaOp>,
               LegalizeAllocLikeOpConversion<memref::AllocOp>,
               LegalizeVectorTypeCastConversion, LegalizeMemrefStoreConversion,
               LegalizeMemrefLoadConversion>(patterns.getContext());
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
          return !unrealizedConversion->hasAttr(kPassLabel);
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
