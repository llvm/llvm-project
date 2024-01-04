//===- TileAllocation.cpp - Allocate SME ZA tiles -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass allocates SME tiles at the 'func.func' op level for ArmSME
// operations. It does this using a 16-bit tile mask that has a bit for each
// 128-bit element tile (ZA0.Q-ZA15.Q), the smallest ZA tile granule.
//
// The 128-bit tiles overlap with other element tiles as follows (see section
// B2.3.2 of SME spec [1]):
//
//   Tile    Overlaps
//   ---------------------------------------------------------------------------
//   ZA0.B   ZA0.Q, ZA1.Q, ZA2.Q, ZA3.Q, ZA4.Q, ZA5.Q, ZA6.Q, ZA7.Q, ZA8.Q,
//           ZA9.Q, ZA10.Q, ZA11.Q, ZA12.Q, ZA13.Q, ZA14.Q, ZA15.Q
//   ZA0.H   ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
//   ZA1.H   ZA1.Q, ZA3.Q, ZA5.Q, ZA7.Q, ZA9.Q, ZA11.Q, ZA13.Q, ZA15.Q
//   ZA0.S   ZA0.Q, ZA4.Q, ZA8.Q, ZA12.Q
//   ZA1.S   ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
//   ZA2.S   ZA2.Q, ZA6.Q, ZA10.Q, ZA14.Q
//   ZA3.S   ZA3.Q, ZA7.Q, ZA11.Q, ZA15.Q
//   ZA0.D   ZA0.Q, ZA8.Q
//   ZA1.D   ZA1.Q, ZA9.Q
//   ZA2.D   ZA2.Q, ZA10.Q
//   ZA3.D   ZA3.Q, ZA11.Q
//   ZA4.D   ZA4.Q, ZA12.Q
//   ZA5.D   ZA5.Q, ZA13.Q
//   ZA6.D   ZA6.Q, ZA14.Q
//   ZA7.D   ZA7.Q, ZA15.Q
//
// The tiles in use are tracked via a function attribute 'arm_sme.tiles_in_use'
// that is initalized during the first tile allocation within a function and
// updated on each subsequent allocation.
//
// [1] https://developer.arm.com/documentation/ddi0616/aa
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "allocate-arm-sme-tiles"

namespace mlir {
namespace arm_sme {
#define GEN_PASS_DEF_TILEALLOCATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace arm_sme
} // namespace mlir

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

static constexpr char kTilesInUseAttr[] = "arm_sme.tiles_in_use";

enum class TileMask : unsigned {
  // clang-format off
  kZA0B  = 0xffff, // 1111 1111 1111 1111

  kZA0H  = 0xaaaa, // 1010 1010 1010 1010
  kZA1H  = 0x5555, // 0101 0101 0101 0101

  kZA0S  = 0x8888, // 1000 1000 1000 1000
  kZA1S  = 0x4444, // 0100 0100 0100 0100
  kZA2S  = 0x2222, // 0010 0010 0010 0010
  kZA3S  = 0x1111, // 0001 0001 0001 0001

  kZA0D  = 0x8080, // 1000 0000 1000 0000
  kZA1D  = 0x4040, // 0100 0000 0100 0000
  kZA2D  = 0x2020, // 0010 0000 0010 0000
  kZA3D  = 0x1010, // 0001 0000 0001 0000
  kZA4D  = 0x808,  // 0000 1000 0000 1000
  kZA5D  = 0x404,  // 0000 0100 0000 0100
  kZA6D  = 0x202,  // 0000 0010 0000 0010
  kZA7D  = 0x101,  // 0000 0001 0000 0001

  kZA0Q  = 0x8000, // 1000 0000 0000 0000
  kZA1Q  = 0x4000, // 0100 0000 0000 0000
  kZA2Q  = 0x2000, // 0010 0000 0000 0000
  kZA3Q  = 0x1000, // 0001 0000 0000 0000
  kZA4Q  = 0x800,  // 0000 1000 0000 0000
  kZA5Q  = 0x400,  // 0000 0100 0000 0000
  kZA6Q  = 0x200,  // 0000 0010 0000 0000
  kZA7Q  = 0x100,  // 0000 0001 0000 0000
  kZA8Q  = 0x80,   // 0000 0000 1000 0000
  kZA9Q  = 0x40,   // 0000 0000 0100 0000
  kZA10Q = 0x20,   // 0000 0000 0010 0000
  kZA11Q = 0x10,   // 0000 0000 0001 0000
  kZA12Q = 0x8,    // 0000 0000 0000 1000
  kZA13Q = 0x4,    // 0000 0000 0000 0100
  kZA14Q = 0x2,    // 0000 0000 0000 0010
  kZA15Q = 0x1,    // 0000 0000 0000 0001

  kNone = 0x0,     // 0000 0000 0000 0000
  // clang-format on

  LLVM_MARK_AS_BITMASK_ENUM(kZA0B)
};

/// Returns the set of masks relevant for the given type.
static ArrayRef<TileMask> getMasks(ArmSMETileType type) {
  static constexpr std::array ZA_B_MASKS = {TileMask::kZA0B};
  static constexpr std::array ZA_H_MASKS = {TileMask::kZA0H, TileMask::kZA1H};
  static constexpr std::array ZA_S_MASKS = {TileMask::kZA0S, TileMask::kZA1S,
                                            TileMask::kZA2S, TileMask::kZA3S};
  static constexpr std::array ZA_D_MASKS = {
      TileMask::kZA0D, TileMask::kZA1D, TileMask::kZA2D, TileMask::kZA3D,
      TileMask::kZA4D, TileMask::kZA5D, TileMask::kZA6D, TileMask::kZA7D};
  static constexpr std::array ZA_Q_MASKS = {
      TileMask::kZA0Q,  TileMask::kZA1Q,  TileMask::kZA2Q,  TileMask::kZA3Q,
      TileMask::kZA4Q,  TileMask::kZA5Q,  TileMask::kZA6Q,  TileMask::kZA7Q,
      TileMask::kZA8Q,  TileMask::kZA9Q,  TileMask::kZA10Q, TileMask::kZA11Q,
      TileMask::kZA12Q, TileMask::kZA13Q, TileMask::kZA14Q, TileMask::kZA15Q};
  switch (type) {
  case ArmSMETileType::ZAB:
    return ZA_B_MASKS;
  case ArmSMETileType::ZAH:
    return ZA_H_MASKS;
  case ArmSMETileType::ZAS:
    return ZA_S_MASKS;
  case ArmSMETileType::ZAD:
    return ZA_D_MASKS;
  case ArmSMETileType::ZAQ:
    return ZA_Q_MASKS;
  }
}

/// Allocates and returns a tile ID. Returns an error if there are no tiles
/// left.
static FailureOr<unsigned> allocateTileId(ArmSMETileType tileType,
                                          TileMask &tilesInUse) {
  auto masks = getMasks(tileType);
  for (auto [tileId, tileMask] : llvm::enumerate(masks)) {
    if ((tilesInUse & tileMask) == TileMask::kNone) {
      tilesInUse |= tileMask;
      return tileId;
    }
  }
  return failure();
}

/// Collects transitive uses of a root value through control flow. This can
/// handle basic SCF constructs, along with control flow (br and cond_br).
/// Simple loops work at the SCF level, while more complex control flow can be
/// dealt with after lowering to CF. This is used to implement basic tile
/// allocation.
static void findDependantOps(Value rootValue,
                             SetVector<Operation *> &dependantOps) {
  auto traverseCorrespondingValues = [&](auto inputValues, auto exitValues) {
    for (auto [idx, value] : llvm::enumerate(inputValues)) {
      if (value == rootValue)
        findDependantOps(exitValues[idx], dependantOps);
    }
  };
  for (Operation *user : rootValue.getUsers()) {
    if (dependantOps.contains(user))
      continue;
    dependantOps.insert(user);
    TypeSwitch<Operation *>(user)
        .Case<cf::BranchOp>([&](auto branchOp) {
          // (CF) Follow branch.
          traverseCorrespondingValues(branchOp.getDestOperands(),
                                      branchOp.getDest()->getArguments());
        })
        .Case<cf::CondBranchOp>([&](auto condBranchOp) {
          // (CF) Follow true branch.
          traverseCorrespondingValues(
              condBranchOp.getTrueOperands(),
              condBranchOp.getTrueDest()->getArguments());
          // (CF) Follow false branch.
          traverseCorrespondingValues(
              condBranchOp.getFalseOperands(),
              condBranchOp.getFalseDest()->getArguments());
        })
        .Case<LoopLikeOpInterface>([&](auto loopOp) {
          // (SCF) Follow iter_args of (basic) loops (e.g. for loops).
          traverseCorrespondingValues(loopOp.getInits(),
                                      loopOp.getRegionIterArgs());
        })
        .Case<scf::YieldOp>([&](auto yieldOp) {
          // (SCF) Follow yields of (basic) control flow (e.g. for loops).
          auto parent = user->getParentOp();
          traverseCorrespondingValues(user->getOperands(),
                                      parent->getResults());
        })
        .Default([&](auto) {
          // Otherwise, assume users of _any_ result are dependant.
          for (Value result : user->getResults())
            findDependantOps(result, dependantOps);
        });
  }
}

struct AssignTileIDsPattern
    : public OpInterfaceRewritePattern<ArmSMETileOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(ArmSMETileOpInterface tileOp,
                                PatternRewriter &rewriter) const override {
    if (tileOp.getTileId())
      return failure();

    std::optional<ArmSMETileType> tileType = tileOp.getAllocatedTileType();
    if (!tileType)
      return rewriter.notifyMatchFailure(tileOp, "op does not allocate a tile");

    auto func = tileOp->getParentOfType<FunctionOpInterface>();
    TileMask tilesInUse = TileMask::kNone;
    if (auto tilesInUseAttr = llvm::dyn_cast_or_null<IntegerAttr>(
            func->getDiscardableAttr(kTilesInUseAttr)))
      tilesInUse = static_cast<TileMask>(tilesInUseAttr.getInt());

    auto tileId = allocateTileId(*tileType, tilesInUse);
    if (failed(tileId))
      return tileOp.emitError("ran out of SME virtual tiles!");

    rewriter.updateRootInPlace(func, [&]() {
      func->setDiscardableAttr(
          kTilesInUseAttr, rewriter.getI32IntegerAttr((unsigned)tilesInUse));
    });

    // Find all the ops that (transitively) depend on this tile.
    SetVector<Operation *> dependantOps;
    findDependantOps(tileOp->getResult(0), dependantOps);

    // Set all operations dependent on `tileOp` to use the same tile ID.
    // This is a naive tile allocation scheme, but works for common cases. For
    // example, as this only allocates tile IDs to existing ops, it can't solve
    // cases like this (%tileA and %tileB come from different root operations):
    //
    // %tile = scf.if %some_cond -> vector<[4]x[4]xi32> {
    //   scf.yield %tileA {tile_id = 0} : vector<[4]x[4]xi32>
    // } else {
    //   scf.yield %tileB {tile_id = 1} : vector<[4]x[4]xi32>
    // }
    //
    // This case would require allocating a new tile for the result of the
    // scf.if, and moving the contents of %tileA or %tileB to result tile (based
    // on the %some_cond).
    auto tileIDAttr = rewriter.getI32IntegerAttr(*tileId);
    rewriter.updateRootInPlace(tileOp, [&]() { tileOp.setTileId(tileIDAttr); });
    for (auto *op : dependantOps) {
      if (auto tileOp = llvm::dyn_cast<ArmSMETileOpInterface>(op)) {
        auto currentTileId = tileOp.getTileId();
        if (currentTileId && unsigned(currentTileId.getInt()) != tileId)
          return tileOp.emitOpError(
              "already assigned different SME virtual tile!");
        rewriter.updateRootInPlace(tileOp,
                                   [&]() { tileOp.setTileId(tileIDAttr); });
      }
    }

    return success();
  }
};

struct TileAllocationPass
    : public arm_sme::impl::TileAllocationBase<TileAllocationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AssignTileIDsPattern>(patterns.getContext());
    GreedyRewriteConfig config;
    // Setting useTopDownTraversal ensures tiles are allocated in program
    // order.
    config.useTopDownTraversal = true;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createTileAllocationPass() {
  return std::make_unique<TileAllocationPass>();
}
