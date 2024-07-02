//===- AMXBindingAnalysis.cpp - Binding info analysis for Intel AMX -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementations of two analysis:
// 1. TileBindingAnalysis
// 2. TileScopeAnalysis
// For more details, please refer to header definition.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/Analysis/AMXBindingAnalysis.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "amx-binding-analysis"

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
// considered unacceptable.
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

// Currently we only operate on scf Ops.
static bool isConcernedControlFlowOp(Operation *op) {
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
  // 0. First walk to mark parallel Ops.
  func->walk<WalkOrder::PostOrder>([this](Operation *op) {
    if (!isParallelOp(op))
      return;

    if (llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op)) {
      while (op != root) {
        addParallelOp(op);
        op = op->getParentOp();
      }
    }
  });

  // 1. Second walk to collect needed palette for each concerned Op.
  collectNeededPalette(root);

  // 2. Third walk to analyse usage scope for each tile Op.
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

  // 3. Tile scoping for each segmented region in a recursive manner.
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
    // No need to store PaletteInfo for func.
    return collectRegionPalette(func.getRegion(0).front());

  auto iter = neededPalette.find(op);
  if (iter != neededPalette.end())
    return iter->second;

  // For now, we only concern certain control flow Ops and tile Ops.
  if (isConcernedControlFlowOp(op))
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

  // Palette shape is { rows, colBytes }.
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

  // Do tile scope on parallel-free BlockSeg.
  TileScope currScope;
  std::optional<Block::iterator> currSegStart;
  Block::iterator currIter = seg.begin();
  // Traverse BlockSeg and greedily do tile scoping without look ahead.
  while (currIter != seg.end()) {
    Operation *currOp = &(*currIter);
    if (!currSegStart)
      currSegStart = currIter;

    Block::iterator nextIterIfMerge;
    std::optional<PaletteInfo> pi = std::null_opt;
    if (isConcernedControlFlowOp(currOp)) {
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
        // This means the binding info in tile Ops exceeds the hardware
        // capability.
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

#define TRY_ADD_PREVIOUS_SCOPE()                                               \
  if (currSegStart && *currSegStart != currIter) {                             \
    auto prevIter = currIter;                                                  \
    prevIter--;                                                                \
    currScope.seg = BlockSeg(*currSegStart, prevIter);                         \
    tileScopes.push_back(currScope);                                           \
    currScope.clear();                                                         \
    currSegStart = std::null_opt;                                              \
  }

    if (pi->overflow) {
      // Only scf Ops could go through this possibility.
      TRY_ADD_PREVIOUS_SCOPE();
      doTileScope(currOp);
      currIter++;
    } else {
      if (currScope.pi.isConflict(*pi)) {
        TRY_ADD_PREVIOUS_SCOPE();
        currScope.pi = *pi;
        currSegStart = currIter;
        currIter++;
      } else {
        currScope.pi.merge(*pi);
        currIter = nextIterIfMerge;
      }
    }
  }

  TRY_ADD_PREVIOUS_SCOPE();
}

void TileScopeAnalysis::doTileScope(Operation *op) {
  // This func try to collect tile scopes for a single control flow Op
  // This func is not for tile Ops.
  if (isTileOp(op))
    return;
  // Ops that invoke this func are either parallelOps or scfOps with overflowed
  // paletteInfo, and neither of them can form a tile scope by itself, so we
  // omit checking self-formed tile scope in this func.
  if (llvm::isa<scf::ExecuteRegionOp>(op) || llvm::isa<scf::ForOp>(op) ||
      llvm::isa<scf::ForallOp>(op) || llvm::isa<scf::ParallelOp>(op)) {
    auto &block = op->getRegion(0).front();
    doTileScope(BlockSeg(block.begin(), block.end()));
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto &ifBlock = op->getThenRegion().front();
    auto &elseBlock = op->getElseRegion().front();
    doTileScope(BlockSeg(ifBlock.begin(), ifBlock.end()));
    doTileScope(BlockSeg(elseBlock.begin(), elseBlock.end()));
  } else if (auto indexOp = dyn_cast<scf::IndexSwitchOp>(op)) {
    auto &defaultBlock = indexOp.getDefaultRegion().front();
    doTileScope(BlockSeg(defaultBlock.begin(), defaultBlock.end()));
    for (auto &caseRegion : indexOp.getCaseRegions()) {
      auto &caseBlock = indexOp.getDefaultRegion().front();
      doTileScope(BlockSeg(caseBlock.begin(), caseBlock.end()));
    }
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    auto &beforeBlock = whileOp.getRegion(0).front();
    auto &afterBlock = whileOp.getRegion(1).front();
    doTileScope(BlockSeg(beforeBlock.begin(), beforeBlock.end()));
    doTileScope(BlockSeg(afterBlock.begin(), afterBlock.end()));
  }
}
