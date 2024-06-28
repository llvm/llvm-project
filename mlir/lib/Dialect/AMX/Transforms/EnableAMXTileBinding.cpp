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
  typedef llvm::iterator_range<Block::iterator> BlockSeg;
  // A list of 2-dim shapes representing tmm register shape, the length should
  // always be 8
  struct PaletteInfo {
    bool overflow;
    SmallVector<pair<int, int>, 8> palette;
    PaletteInfo() {
      palette.resize(8, {0, 0});
      clear();
    }
    void clear();
    bool isEmpty(int idx) {
      return palette[idx].first == 0 && palette[idx].second == 0;
    }
    void set(int idx, pair<int, int> shape) { palette[idx] = shape; }
    void merge(const PaletteInfo &rhs);
    bool isConflict(const PaletteInfo &rhs);
  };
  struct TileScope {
    // The BlockSeg here is inclusive (including end Op)
    BlockSeg seg;
    PaletteInfo pi;
    TileScope() { clear(); }
    void clear() { pi.clear(); }
  };

  bool isValidAnalysis;
  // Storing parallel Ops that would break tile context & scope
  DenseSet<Operation *> parallelOps;
  // Storing needed palette info for each concerned Op
  DenseMap<Operation *, PaletteInfo> neededPalette;
  // Storing the usage scope for each concerned tile Op
  // The BlockSeg here is inclusive (including end Op)
  DenseMap<Operation *, BlockSeg> tileUsage;
  // Storing final tile scope results for injecting tilecfg/tilerelease
  SmallVector<TileScope, 10> tileScopes;

  void addParallelOp(Operation *op) { parallelOps.insert(op); }
  bool isParallelOp(Operation *op) {
    return parallelOps.find(op) == parallelOps.end();
  }

  void setTileUsage(Operation *op, BlockSeg seg) { tileUsage[op] = seg; }

  PaletteInfo collectRegionPalette(Region &region);
  PaletteInfo collectPalette(Operation *op);
  // Below two functions are the leaf functinos of recursive collection, will
  // actually insert PaletteInfo into map storage
  PaletteInfo collectPaletteForScf(Operation *op);
  PaletteInfo collectPaletteForTile(Operation *op);
  std::optional<PaletteInfo> getPalette(Operation *op);
  std::optional<PaletteInfo> getPalette(BlockSeg seg);

  void doTileScope(Block &block);
  // The BlockSeg here is exclusive (excluding end Op)
  void doTileScope(BlockSeg seg);
  void doTileScope(Operation *op);

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileScopeAnalysis)
  explicit TileScopeAnalysis(Operation *);
  bool isValid() const { return isValidAnalysis; }
};

void TileScopeAnalysis::PaletteInfo::clear() {
  overflow = false;
  for (int idx = 0; idx < 8; idx++)
    palette[idx] = {0, 0};
}

void TileScopeAnalysis::PaletteInfo::merge(const PaletteInfo &rhs) {
  if (overflow || rhs.overflow) {
    overflow = true;
    return;
  }
  for (int idx = 0; idx < 8; idx++) {
    if (!isEmpty(idx) && !rhs.isEmpty(idx)) {
      if (palette[idx].first != rhs.palette[idx].first ||
          palette[idx].second != rhs.palette[idx].second) {
        overflow = true;
        break;
      }
    } else if (!rhs.isEmpty(idx)) {
      palette[idx] = rhs.palette[idx];
    }
  }
}

bool TileScopeAnalysis::PaletteInfo::isConflict(const PaletteInfo &rhs) {
  if (overflow || rhs.overflow) {
    return true;
  }
  for (int idx = 0; idx < 8; idx++) {
    if (!isEmpty(idx) && !rhs.isEmpty(idx)) {
      if (palette[idx].first != rhs.palette[idx].first ||
          palette[idx].second != rhs.palette[idx].second) {
        return true;
      }
    }
  }
}

static bool isConcernedScfOp(Operation *op) {
  return llvm::isa<scf::ExecuteRegionOp>(op) || llvm::isa<scf::ForOp>(op) ||
         llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::IfOp>(op) ||
         llvm::isa<scf::IndexSwitchOp>(op) || llvm::isa<scf::ParallelOp>(op) ||
         llvm::isa<scf::WhileOp>(op);
}

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
  // 0. First walk to mark parallel Ops
  func->walk<WalkOrder::PostOrder>([this](Operation *op) {
    if (!isParallelOp(op))
      return;

    if (llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op) ||
        llvm::isa<omp::ParallelOp>(op) || llvm::isa<omp::WsloopOp>(op)) {
      while (op != root) {
        addParallelOp(op);
        op = op->getParentOp();
      }
    }
  });

  // 1. Second walk to collect needed palette for each concerned Op
  collectNeededPalette(root);

  // 2. Third walk to analyse usage scope for each tile Op
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

  // 3. Tile scoping for each segmented region in a recursive manner
  doTileScope(func.getRegion(0).front());
}

PaletteInfo TileScopeAnalysis::collectRegionPalette(Block &block) {
  PaletteInfo pi;
  for (auto op : block.getOps())
    pi.merge(collectPalette(op));
  return pi;
}

PaletteInfo TileScopeAnalysis::collectPalette(Operation *op) {
  if (!isValidAnalysis)
    return PaletteInfo();
  if (auto func = dyn_cast_or_null<func::FuncOp>(root))
    // No need to store PaletteInfo for func
    return collectRegionPalette(func.getRegion(0).front());

  auto iter = neededPalette.find(op);
  if (iter != neededPalette.end())
    return iter->second;

  // For now, we only concern certain control flow Ops and tile Ops
  if (isConcernedScfOp(op))
    return collectPaletteForScf(op);
  if (isTileOp(op))
    return collectPaletteForTile(op);
  return PaletteInfo();
}

PaletteInfo TileScopeAnalysis::collectPaletteForScf(Operation *op) {
  if (!isConcerendScfOp(op))
    return PaletteInfo();

  PaletteInfo pi;
  if (llvm::isa<scf::ExecuteRegionOp>(op) || llvm::isa<scf::ForOp>(op) ||
      llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op)) {
    pi = collectNeededPalette(op->getRegion(0).front());
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto thenPalette = collectRegionPalette(ifOp.getThenRegion().front());
    auto elsePalette = collectRegionPalette(ifOp.getElseRegion().front());
    pi.merge(thenPalette);
    pi.merge(elsePalette);
  } else if (auto indexOp = dyn_cast<scf::IndexSwitchOp>(op)) {
    pi = collectRegionPalette(indexOp.getDefaultRegion().front());
    for (auto &caseRegion : indexOp.getCaseRegions()) {
      pi.merge(collectRegionPalette(caseRegion.front()));
    }
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    auto beforePalette = collectRegionPalette(whileOp.getRegion(0).front());
    auto afterPalette = collectRegionPalette(whileOp.getRegion(1).front());
    pi.merge(beforePalette);
    pi.merge(afterPalette);
  }
  neededPalette[op] = pi;
  return pi;
}

static inline pair<int, int> getPaletteShape(VectorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  auto elementType = type.getElementType();
  int typeSize;
  if (elementType.isInteger(8))
    typeSize = 1;
  else if (elementType.isBF16())
    typeSize = 2;
  else if (elementType.isInteger(32) || elementType.isF32())
    typeSize = 4;
  else
    assert(false && "Invalid type for palette");

  // Palette shape is { rows, colBytes }
  return {shape[0], shape[1] * typeSize};
}

PaletteInfo TileScopeAnalysis::collectPaletteForTile(Operation *op) {
  if (!isTileOp(op))
    return PaletteInfo();

#define PROCESS_UNARY_TILE_OP(op, method)                                      \
  auto index = op.method();                                                    \
  if (!index) {                                                                \
    isValidAnalysis = false;                                                   \
    return PaletteInfo();                                                      \
  }                                                                            \
  pi.set(*index, getPaletteShape(op.getVectorType()));

#define PROCESS_TRINARY_TILE_OP(op)                                            \
  auto lhsIndex = tileMulFOp.getLhsRegIndex();                                 \
  auto rhsIndex = tileMulFOp.getRhsRegIndex();                                 \
  auto accIndex = tileMulFOp.getAccRegIndex();                                 \
  if (!lhsIndex || !rhsIndex || !accIndex) {                                   \
    isValidAnalysis = false;                                                   \
    return PaletteInfo();                                                      \
  }                                                                            \
  pi.set(*lhsIndex, getPaletteShape(op.getLhsVectorType()));                   \
  pi.set(*rhsIndex, getPaletteShape(op.getRhsVectorType()));                   \
  pi.set(*accIndex, getPaletteShape(op.getAccVectorType()));

  PaletteInfo pi;
  if (auto tileLoadOp = dyn_cast<TileLoadOp>(op)) {
    PROCESS_UNARY_TILE_OP(tileLoadOp, getDstRegIndex);
  } else if (auto tileMulFOp = dyn_cast<TileMulFOp>(op)) {
    PROCESS_TRINARY_TILE_OP(tileMulFOp);
  } else if (auto tileMulIOp = dyn_cast<TileMulIOp>(op)) {
    PROCESS_TRINARY_TILE_OP(tileMulIOp);
  } else if (auto tileStoreOp = dyn_cast<TileStoreOp>(op)) {
    PROCESS_UNARY_TILE_OP(tileStoreOp, getSrcRegIndex);
  } else if (auto tileZeroOp = dyn_cast<TileZeroOp>(op)) {
    PROCESS_UNARY_TILE_OP(tileZeroOp, getDstRegIndex);
  }
  neededPalette[op] = pi;
  return pi;
}

std::optional<PaletteInfo> TileScopeAnalysis::getPalette(Operation *op) {
  auto iter = neededPalette.find(op);
  if (iter == neededPalette.end()) {
    return std::null_opt;
  }
  return iter->second;
}

std::optional<PaletteInfo> TileScopeAnalysis::getPalette(BlockSeg seg) {
  bool hasPaletteInfo = false;
  PaletteInfo pi;
  for (Operation &opIns : seg) {
    auto *op = &opIns;
    auto tmpPi = getPalette(&opIns);
    if (tmpPi) {
      hasPaletteInfo = true;
      pi.merge(*tmpPi);
    }
  }
  return hasPaletteInfo ? pi : std::null_opt;
}

void TileScopeAnalysis::doTileScope(Block &block) {
  doTileScope(BlockSeg(block.begin(), block.end()));
}

void TileScopeAnalysis::doTileScope(BlockSeg seg) {
  if (!isValidAnalysis)
    return;
  if (seg.empty())
    return;
  SmallVector<BlockSeg, 3> blockSegs;
  SmallVector<Operation *, 3> paraOps;
  auto currBegin = seg.begin();
  for (auto probe = seg.begin(); probe != seg.end(); probe++) {
    Operation *op = &(*probe);
    if (isParallelOp(op) {
      blockSegs.push_back(BlockSeg(currBegin, probe));
      paraOps.push_back(op);
      currBegin = probe;
      currBegin++;
    }
  }
  if (breakers.size()) {
    assert(blockSegs.size() == paraOps.size());
    for (int idx = 0; idx < paraOps.size(); idx++) {
      doTileScope(blockSegs[idx]);
      doTileScope(paraOps[idx]);
    }
    doTileScope(BlockSeg(currBegin, seg.end()));
    return;
  }

  // Do tile scope on parallel-free BlockSeg
  TileScope currScope;
  std::optional<Block::iterator> currSegStart;
  Block::iterator currIter = seg.begin();
  // Traverse BlockSeg and greedily do tile scoping without look ahead
  while (currIter != seg.end()) {
    Operation *currOp = &(*currIter);
    if (!currSegStart)
      currSegStart = currIter;

    Block::iterator nextIterIfMerge;
    std::optional<PaletteInfo> pi = std::null_opt;
    if (isConcernedScfOp(currOp)) {
      pi = getPalette(currOp);
      nextIterIfMerge = currIter;
      nextIterIfMerge++;
    } else if (isTileOp(currOp)) {
      auto iter = tileUsage.find(currOp);
      if (iter == tileUsage.end()) {
        isValidAnalysis = false;
        return;
      }
      pi = getPalette(iter->second);
      if (pi && pi->overflow) {
        // This means the binding info exceeds the hardware capability
        isValidAnalysis = false;
        return;
      }
      nextIterIfMerge = iter->second.end();
      nextIterIfMerge++;
    }
    if (!pi) {
      currIter++;
      continue;
    }

#define ADD_PREVIOUS_SCOPE()                                                   \
  auto prevIter = currIter;                                                    \
  prevIter--;                                                                  \
  currScope.seg = BlockSeg(*currSegStart, prevIter);                           \
  tileScopes.push_back(currScope);                                             \
  currScope.clear();                                                           \
  currSegStart = std::null_opt;

    if (pi->overflow) {
      // Only scf Ops could go through this possibility
      if (currSegStart && *currSegStart != currIter) {
        ADD_PREVIOUS_SCOPE()
      }
      doTileScope(currOp);
      currIter++;
    } else {
      if (currScope.pi.isConflict(*pi)) {
        ADD_PREVIOUS_SCOPE();
        currIter++;
      } else {
        currScope.pi.merge(*pi);
        currIter = nextIterIfMerge;
      }
    }
  }

  ADD_PREVIOUS_SCOPE();
}

void TileScopeAnalysis::doTileScope(Operation *op) {
  // TODO: do tile scope for scf here
}

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
