#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/IR/Polygeist.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

using namespace mlir;
using namespace mlir::polygeist;

#define GET_OP_CLASSES
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.cpp.inc"

namespace {
/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
class Memref2Pointer2MemrefCast final
    : public OpRewritePattern<Pointer2MemrefOp> {
public:
  using OpRewritePattern<Pointer2MemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Pointer2MemrefOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Memref2PointerOp>();
    if (!src)
      return failure();
    auto smt = cast<MemRefType>(src.getSource().getType());
    auto omt = cast<MemRefType>(op.getType());
    if (smt.getShape().size() != omt.getShape().size())
      return failure();
    for (unsigned i = 1; i < smt.getShape().size(); i++) {
      if (smt.getShape()[i] != omt.getShape()[i])
        return failure();
    }
    if (smt.getElementType() != omt.getElementType())
      return failure();
    if (smt.getMemorySpace() != omt.getMemorySpace())
      return failure();

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(),
                                                src.getSource());
    return success();
  }
};

/// Simplify memref2pointer(pointer2memref(x)) to cast(x)
class Memref2PointerBitCast final : public OpRewritePattern<LLVM::BitcastOp> {
public:
  using OpRewritePattern<LLVM::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getOperand().getDefiningOp<Memref2PointerOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<Memref2PointerOp>(op, op.getType(),
                                                  src.getSource());
    return success();
  }
};

/// Simplify load(pointer2memref(gep(...(x)))) to load(x, idx)
template <typename T>
class LoadStorePointer2MemrefGEP final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  SmallVector<Value> newIndex(T op, Value finalIndex,
                              PatternRewriter &rewriter) const;

  void createNewOp(T op, Value baseMemref, SmallVector<Value> vals,
                   PatternRewriter &rewriter) const;

  Value getMemref(T op) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (op.getMemRefType().getRank() != 1)
      return failure();

    auto src = getMemref(op).template getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    Type elementType = op.getMemRefType().getElementType();
    unsigned elementSize = elementType.isIntOrFloat()
                               ? elementType.getIntOrFloatBitWidth() / 8
                               : 0;
    if (elementSize == 0)
      return failure();

    SmallVector<std::pair<LLVM::GEPOp, unsigned>> gepOps;
    Value ptr = src.getSource();

    while (auto gep = ptr.getDefiningOp<LLVM::GEPOp>()) {
      if (gep.getIndices().size() != 1)
        break;

      unsigned gepElemSize = 1;
      auto elemTy = gep.getElemType();
      if (elemTy.isIntOrFloat()) {
        gepElemSize = elemTy.getIntOrFloatBitWidth() / 8;
      } else if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(elemTy)) {
        auto baseTy = arrayTy.getElementType();
        if (baseTy.isIntOrFloat()) {
          gepElemSize =
              (baseTy.getIntOrFloatBitWidth() / 8) * arrayTy.getNumElements();
        } else {
          break;
        }
      } else {
        break;
      }

      gepOps.emplace_back(gep, gepElemSize);
      ptr = gep.getBase();
    }

    if (gepOps.empty())
      return failure();

    Location loc = op.getLoc();
    auto baseMemref = Pointer2MemrefOp::create(
        rewriter, loc, cast<MemRefType>(src.getType()), ptr);

    Value finalIndex = nullptr;
    for (auto [gep, gepElemSize] : llvm::reverse(gepOps)) {
      PointerUnion<IntegerAttr, Value> rawIdx = gep.getIndices()[0];
      Value idx = dyn_cast_if_present<Value>(rawIdx);
      if (!idx)
        idx = arith::ConstantIndexOp::create(
            rewriter, loc, cast<IntegerAttr>(rawIdx).getValue().getSExtValue());

      if (auto constIdx = idx.getDefiningOp<arith::ConstantIndexOp>()) {
        if ((constIdx.value() * gepElemSize) % elementSize != 0) {
          return failure();
        }
      }

      if (!idx.getType().isIndex()) {
        idx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                                  idx);
      }

      unsigned gcd = std::gcd(gepElemSize, elementSize);
      unsigned scaledGep = gepElemSize / gcd;
      unsigned scaledElement = elementSize / gcd;

      Value scaledIdx =
          (scaledGep != 1)
              ? arith::MulIOp::create(
                    rewriter, loc, idx,
                    arith::ConstantIndexOp::create(rewriter, loc, scaledGep))
              : idx;

      Value elemOffset =
          (scaledElement != 1)
              ? arith::DivSIOp::create(
                    rewriter, loc, scaledIdx,
                    arith::ConstantIndexOp::create(rewriter, loc, scaledElement))
              : scaledIdx;

      if (finalIndex)
        finalIndex =
            arith::AddIOp::create(rewriter, loc, finalIndex, elemOffset);
      else
        finalIndex = elemOffset;
    }

    createNewOp(op, baseMemref, newIndex(op, finalIndex, rewriter), rewriter);
    return success();
  }
};

template <>
Value LoadStorePointer2MemrefGEP<memref::LoadOp>::getMemref(
    memref::LoadOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<memref::StoreOp>::getMemref(
    memref::StoreOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::getMemref(
    affine::AffineLoadOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::getMemref(
    affine::AffineStoreOp op) const {
  return op.getMemref();
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<memref::LoadOp>::newIndex(
    memref::LoadOp op, Value finalIndex, PatternRewriter &rewriter) const {
  auto operands = llvm::to_vector(op.getIndices());
  operands[0] =
      arith::AddIOp::create(rewriter, op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::newIndex(
    affine::AffineLoadOp op, Value finalIndex,
    PatternRewriter &rewriter) const {
  auto apply = affine::AffineApplyOp::create(
      rewriter, op.getLoc(), op.getAffineMap(), op.getMapOperands());

  SmallVector<Value> operands;
  for (auto op : apply->getResults())
    operands.push_back(op);
  operands[0] =
      arith::AddIOp::create(rewriter, op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<memref::StoreOp>::newIndex(
    memref::StoreOp op, Value finalIndex, PatternRewriter &rewriter) const {
  auto operands = llvm::to_vector(op.getIndices());
  operands[0] =
      arith::AddIOp::create(rewriter, op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::newIndex(
    affine::AffineStoreOp op, Value finalIndex,
    PatternRewriter &rewriter) const {
  auto apply = affine::AffineApplyOp::create(
      rewriter, op.getLoc(), op.getAffineMap(), op.getMapOperands());

  SmallVector<Value> operands;
  for (auto op : apply->getResults())
    operands.push_back(op);
  operands[0] =
      arith::AddIOp::create(rewriter, op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
void LoadStorePointer2MemrefGEP<memref::LoadOp>::createNewOp(
    memref::LoadOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, baseMemref, idxs);
}

template <>
void LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::createNewOp(
    affine::AffineLoadOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, baseMemref, idxs);
}

template <>
void LoadStorePointer2MemrefGEP<memref::StoreOp>::createNewOp(
    memref::StoreOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), baseMemref,
                                               idxs);
}

template <>
void LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::createNewOp(
    affine::AffineStoreOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), baseMemref,
                                               idxs);
}

/// Simplify cast(pointer2memref(x)) to pointer2memref(x)
class Pointer2MemrefCast final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<Pointer2MemrefOp>(op, op.getType(),
                                                  src.getSource());
    return success();
  }
};

/// Simplify memref2pointer(pointer2memref(x)) to cast(x)
class Pointer2Memref2PointerCast final
    : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getType(),
                                                 src.getSource());
    return success();
  }
};

} // namespace

void Memref2PointerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<Memref2Pointer2MemrefCast, Memref2PointerBitCast>(context);
}

OpFoldResult Memref2PointerOp::fold(FoldAdaptor adaptor) {
  /// Simplify memref2pointer(cast(x)) to memref2pointer(x)
  if (auto mc = getSource().getDefiningOp<memref::CastOp>()) {
    getSourceMutable().assign(mc.getSource());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<Pointer2MemrefOp>()) {
    if (mc.getSource().getType() == getType()) {
      return mc.getSource();
    }
  }
  return nullptr;
}

void Pointer2MemrefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<Pointer2MemrefCast, Pointer2Memref2PointerCast,
                 LoadStorePointer2MemrefGEP<memref::LoadOp>,
                 LoadStorePointer2MemrefGEP<affine::AffineLoadOp>,
                 LoadStorePointer2MemrefGEP<memref::StoreOp>,
                 LoadStorePointer2MemrefGEP<affine::AffineStoreOp>>(context);
}

OpFoldResult Pointer2MemrefOp::fold(FoldAdaptor adaptor) {
  /// Simplify pointer2memref(cast(x)) to pointer2memref(x)
  if (auto mc = getSource().getDefiningOp<LLVM::BitcastOp>()) {
    getSourceMutable().assign(mc.getOperand());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<LLVM::AddrSpaceCastOp>()) {
    getSourceMutable().assign(mc.getOperand());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<LLVM::GEPOp>()) {
    for (auto idx : mc.getDynamicIndices()) {
      assert(idx);
      if (!matchPattern(idx, m_Zero()))
        return nullptr;
    }
    auto staticIndices = mc.getRawConstantIndices();
    for (auto pair : llvm::enumerate(staticIndices)) {
      if (pair.value() != LLVM::GEPOp::kDynamicIndex)
        if (pair.value() != 0)
          return nullptr;
    }

    getSourceMutable().assign(mc.getBase());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<Memref2PointerOp>()) {
    if (mc.getSource().getType() == getType()) {
      return mc.getSource();
    }
  }
  return nullptr;
}
