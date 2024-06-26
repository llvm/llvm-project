//===- EnableAMXTileBinding.cpp - Enable tile binding for Intel AMX -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass enables the tile register binding semantic for Intel® Advanced
// Matrix Extensions (Intel® AMX). Intuitively, this pass analyses the tile
// binding hints set by users, legalize the hints and automatically configures
// needed hardware context. The AMX tile register usage in lowered intrinsics
// would strictly respect the given hints, enforced in lowering pass
// `--convert-vector-to-llvm`.
//
// Note that if this pass is not invoked prior to `--convert-vector-to-llvm`,
// the AMX lowering would ignore the binding info and fallback to original
// scheme.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enable-amx-tile-binding"

namespace mlir {
namespace amx {

#define GEN_PASS_DEF_ENABLEAMXTILEBINDING
#include "mlir/Dialect/AMX/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// A class for analyzing (propagating) tile register binding for each tile
/// vector.
class TileBindingAnalysis {
private:
  bool isValidAnalysis;
  DenseMap<Value, int> bindings;

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBindingAnalysis)
  explicit TileBindingAnalysis(Operation *);
  bool isValid() const { return isValidAnalysis; }
  // void setValid(bool v) { isValidAnalysis = v; }
  int getBinding(Value val) const {
    auto iter = bindings.find(val);
    if (iter == bindings.end())
      return -1;
    return iter->second;
  }
  void setBinding(Value val, int index) { bindings[val] = index; }
};

static bool isTileOp(Operation *op) {
  return llvm::isa<TileZeroOp>(op) || llvm::isa<TileLoadOp>(op) ||
         llvm::isa<TileMulFOp>(op) || llvm::isa<TileMulIOp>(op) ||
         llvm::isa<TileStoreOp>(op);
}

template <typename Op>
static bool TileMulCheck(Operation *op) {
  auto tile_mul = dyn_cast_or_null<Op>(op);
  assert(tile_mul);

  auto lhsOp = tile_mul.getLhs().getDefiningOp();
  auto rhsOp = tile_mul.getRhs().getDefiningOp();
  auto accOp = tile_mul.getAcc().getDefiningOp();
  if (!isTileOp(lhsOp) || !isTileOp(rhsOp) || !isTileOp(accOp))
    return false;
  return true;
}

// Not allow mixed use of tile Ops and normal vector Ops, any mixing is
// considered unacceptable
static bool isAcceptableTileOp(Operation *op) {
  if (!isTileOp(op))
    return false;

  if (llvm::isa<TileMulFOp>(op)) {
    return TileMulCheck<TileMulFOp>(op);
  } else if (llvm::isa<TileMulIOp>(op)) {
    return TileMulCheck<TileMulIOp>(op);
  } else if (auto tileStore = dyn_cast_or_null<TileStoreOp>(op)) {
    auto valOp = tileStore.getVal().getDefiningOp();
    if (!isTileOp(valOp))
      return false;
  }
  return true;
}

template <typename Op>
static bool TileDstPropagate(TileBindingAnalysis *analysis, Operation *op) {
  auto tileDst = dyn_cast_or_null<Op>(op);
  assert(tileDst);
  std::optional<int8_t> tmmIndex = tileDst.getDstRegIndex();
  if (!tmmIndex) {
    return false;
  }
  analysis->setBinding(tileDst.getRes(), *tmmIndex);
  return true;
}

template <typename Op>
static bool TileMulPropagate(TileBindingAnalysis *analysis, Operation *op) {
  auto tileMul = dyn_cast_or_null<Op>(op);
  assert(tileMul);
  auto accVal = tileMul.getAcc();
  auto accIndex = analysis->getBinding(accVal);
  if (accIndex < 0)
    return false;

  analysis->setBinding(tileMul.getRes(), accIndex);
  return true;
}

TileBindingAnalysis::TileBindingAnalysis(Operation *root) {
  isValidAnalysis = false;
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(root);
  if (!func)
    return;

  isValidAnalysis = true;
  func->walk<WalkOrder::PreOrder>([this](Operation *op) {
    if (!isValidAnalysis)
      return;
    if (!isTileOp(op))
      return;
    if (!isAcceptableTileOp(op)) {
      isValidAnalysis = false;
      return;
    }

    if (llvm::isa<TileZeroOp>(op)) {
      if (!TileDstPropagate<TileZeroOp>(this, op)) {
        isValidAnalysis = false;
        return;
      }
    } else if (llvm::isa<TileLoadOp>(op)) {
      if (!TileDstPropagate<TileLoadOp>(this, op)) {
        isValidAnalysis = false;
        return;
      }
    } else if (llvm::isa<TileMulFOp>(op)) {
      if (!TileMulPropagate<TileMulFOp>(this, op)) {
        isValidAnalysis = false;
        return;
      }
    } else if (llvm::isa<TileMulIOp>(op)) {
      if (!TileMulPropagate<TileMulIOp>(this, op)) {
        isValidAnalysis = false;
        return;
      }
    }
  });
}

// A class for analyzing tile configuration domination (a.k.a. tile scope)
class TileScopeAnalysis {
private:
  typedef llvm::iterator_range<Block::iterator, Block::iterator> BlockSeg;
  typedef SmallVector<SmallVector<int, 2>, 8> Palette;
  struct TileScope {
    BlockSeg seg;
    Palette palette;
  };

  bool isValidAnalysis;
  // Storing Ops that would break tile context & scope (usually parallel Ops)
  DenseSet<Operation *> scopeBreaker;
  DenseMap<Operation *, BlockSeg> tileUsage;
  SmallVector<TileScope, 10> tileScopes;

  void addScopeBreaker(Operation *op) { scopeBreaker.insert(op); }
  bool isScopeBreaker(Operation *op) {
    return scopeBreaker.find(op) == scopeBreaker.end();
  }

  void setTileUsage(Operation *op, BlockSeg seg);
  BlockSeg getTileUsage();
  void doTileScope(Block &block);
  void doTileScope(BlockSeg seg);
  void doTileScope(Operation *op);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileScopeAnalysis)
  explicit TileScopeAnalysis(Operation *);
  bool isValid() const { return isValidAnalysis; }
};

static bool isTileOp(Operation *op) {
  return llvm::isa<TileZeroOp>(op) || llvm::isa<TileLoadOp>(op) ||
         llvm::isa<TileMulFOp>(op) || llvm::isa<TileMulIOp>(op) ||
         llvm::isa<TileStoreOp>(op);
}

TileScopeAnalysis::TileScopeAnalysis(Operation *root) {
  isValidAnalysis = false;
  func::FuncOp func = dyn_cast_or_null<func::FuncOp>(root);
  if (!func)
    return;

  isValidAnalysis = true;
  // 0. First walk to mark tile scope breakers
  func->walk<WalkOrder::PostOrder>([this](Operation *op) {
    if (!isScopeBreaker(op))
      return;

    if (llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op) ||
        llvm::isa<omp::ParallelOp>(op) || llvm::isa<omp::WsloopOp>(op)) {
      while (op != root) {
        addScopeBreaker(op);
        op = op->getParentOp();
      }
    }
  });

  // 1. Second walk to analyse usage scope for each tile Op
  func->walk<WalkOrder::PreOrder>([this](Operation *op) {
    if (!isValidAnalysis)
      return;
    if (!isTileOp(op))
      return;
    Operation *lastUser = nullptr;
    for (auto user : op->getUsers())
      lastUser = user;
    while (lastUser && op->getBlock() != lastUser->getBlock()) {
      lastUser = lastUser->getParentOp();
      if (!lastUser)
        isValidAnalysis = false;
    }
    setTileUsage(op, BlockSeg(Block::iterator(op), Block::iterator(lastUser)));
  });
  if (!isValidAnalysis)
    return;

  // 2. Tile scoping for each segmented region in a recursive manner
  doTileScope(func.getRegion(0).front());
}

void TileScopeAnalysis::doTileScope(Block &block) {
  doTileScope(BlockSeg(block.begin(), block.end()));
}

void TileScopeAnalysis::doTileScope(BlockSeg seg) {
  if (seg.empty())
    return;
  for (auto probe = seg.begin(); probe < seg.end(); probe++) {
  }
}

void TileScopeAnalysis::doTileScope(Operation *op) {}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class TileStoreBindingRewriter : public OpRewritePattern<TileStoreOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileStoreOp>::OpRewritePattern;

  TileStoreBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileStoreOp op,
                                PatternRewriter &rewriter) const final {
    auto val = op.getVal();
    auto srcIndex = analysis.getBinding(val);
    if (srcIndex < 0)
      return failure();
    auto existingAccIndex = op.getSrcRegIndex();
    if (existingAccIndex && *existingAccIndex != srcIndex)
      return failure();

    rewriter.replaceOpWithNewOp<TileStoreOp>(
        op, op.getBase(), op.getIndices(), val,
        rewriter.getI8IntegerAttr(srcIndex));
    return success();
  }
};

class TileMulFBindingRewriter : public OpRewritePattern<TileMulFOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileMulFOp>::OpRewritePattern;

  TileMulFBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileMulFOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsVal = op.getLhs();
    auto rhsVal = op.getRhs();
    auto accVal = op.getAcc();
    auto lhsIndex = analysis.getBinding(lhsVal);
    auto rhsIndex = analysis.getBinding(rhsVal);
    auto accIndex = analysis.getBinding(accVal);
    if (lhsIndex < 0 || rhsIndex < 0 || accIndex < 0)
      return failure();
    auto existingLhsIndex = op.getLhsRegIndex();
    auto existingRhsIndex = op.getRhsRegIndex();
    auto existingAccIndex = op.getAccRegIndex();
    if ((existingLhsIndex && *existingLhsIndex != lhsIndex) ||
        (existingRhsIndex && *existingRhsIndex != rhsIndex) ||
        (existingAccIndex && *existingAccIndex != accIndex))
      return failure();

    rewriter.replaceOpWithNewOp<TileMulFOp>(
        op, op.getRes().getType(), lhsVal, rhsVal, accVal,
        rewriter.getI8IntegerAttr(lhsIndex),
        rewriter.getI8IntegerAttr(rhsIndex),
        rewriter.getI8IntegerAttr(accIndex));
    return success();
  }
};

class TileMulIBindingRewriter : public OpRewritePattern<TileMulIOp> {
private:
  TileBindingAnalysis &analysis;

public:
  using OpRewritePattern<TileMulIOp>::OpRewritePattern;

  TileMulIBindingRewriter(MLIRContext *context, TileBindingAnalysis &ana)
      : OpRewritePattern(context), analysis{ana} {}

  LogicalResult matchAndRewrite(TileMulIOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsVal = op.getLhs();
    auto rhsVal = op.getRhs();
    auto accVal = op.getAcc();
    auto lhsIndex = analysis.getBinding(lhsVal);
    auto rhsIndex = analysis.getBinding(rhsVal);
    auto accIndex = analysis.getBinding(accVal);
    if (lhsIndex < 0 || rhsIndex < 0 || accIndex < 0)
      return failure();
    auto existingLhsIndex = op.getLhsRegIndex();
    auto existingRhsIndex = op.getRhsRegIndex();
    auto existingAccIndex = op.getAccRegIndex();
    if ((existingLhsIndex && *existingLhsIndex != lhsIndex) ||
        (existingRhsIndex && *existingRhsIndex != rhsIndex) ||
        (existingAccIndex && *existingAccIndex != accIndex))
      return failure();

    rewriter.replaceOpWithNewOp<TileMulIOp>(
        op, op.getRes().getType(), lhsVal, rhsVal, accVal, op.getIsZextLhs(),
        op.getIsZextRhs(), rewriter.getI8IntegerAttr(lhsIndex),
        rewriter.getI8IntegerAttr(rhsIndex),
        rewriter.getI8IntegerAttr(accIndex));
    return success();
  }
};

struct EnableAMXTileBindingPass
    : public impl::EnableAMXTileBindingBase<EnableAMXTileBindingPass> {
  void runOnOperation() override {
    // 0. Get AnalyseInfo for each concerned Value (Does not allow mixed used of
    // tmul & normal vector operations)
    TileBindingAnalysis &bindingAna = getAnalysis<TileBindingAnalysis>();
    if (!bindingAna.isValid())
      return;

    // 1. Set propagated binding info to AMX Ops
    RewritePatternSet patterns(&getContext());
    patterns.add<TileStoreBindingRewriter>(&getContext(), analysis);
    patterns.add<TileMulFBindingRewriter>(&getContext(), analysis);
    patterns.add<TileMulIBindingRewriter>(&getContext(), analysis);
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      return;

    // 2. Analyse tile scopes & expand them maximally
    TileScopeAnalysis &scopeAna = getAnalysis<TileScopeAnalysis>();
    if (!scopeAna.isValid())
      return;

    // 3. insert tile config/release according to tile scopes
  }
};

} // namespace amx
} // namespace mlir
