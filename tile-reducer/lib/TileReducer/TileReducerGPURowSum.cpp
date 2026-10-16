//===- TileReducerGPURowSum.cpp - Milestones 17-19 --------------*- C++ -*-===//
//
// Lower the canonical row-sum to GPU dialect:
//
//   logical 128x128 tile
//           |
//           | fused load / reduce  (no 128x128 temporary, no smem)
//           v
//   coalesced global loads
//           |
//           v
//   per-lane register accumulation (4 elems / lane)
//           |
//           v
//   gpu.subgroup_reduce
//
// Physical map (256 threads, 8 warps, warp size 32):
//   warp w owns rows w, w+8, ..., 120   (16 rows, sequential)
//   lane L owns columns L, L+32, L+64, L+96
//
// tr.program_id stays a logical program instance. It is not threadIdx.
// The host (later) launches one block per program id.
//
// Tails (Milestone 19): ceilDiv(K, tileCols); OOB columns contribute 0.
// The K-reduction is reassociated (lane tree, then warp tree) relative to
// a sequential left-fold. TileReducer treats row-sum as reassociative,
// matching the Linalg reduction generic. Masked zeros are add identities.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/GPUTargetInfo.h"
#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerPasses.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::tr;

namespace mlir::tr {
#define GEN_PASS_DEF_CONVERTTRROWSUMTOGPU
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

struct RowSumMatch {
  LoadOp load;
  ReduceSumOp reduce;
  StoreOp store;
  ProgramIdOp programId;
  TileType tileTy;
};

static std::optional<RowSumMatch> matchRowSum(func::FuncOp func) {
  RowSumMatch found;
  bool ok = false;
  func.walk([&](ReduceSumOp reduce) {
    if (ok)
      return;
    if (reduce.getAxis() != 1)
      return;
    auto tileTy = dyn_cast<TileType>(reduce.getInput().getType());
    if (!tileTy || tileTy.getRank() != 2)
      return;
    auto load = reduce.getInput().getDefiningOp<LoadOp>();
    if (!load)
      return;
    StoreOp store;
    ProgramIdOp programId;
    func.walk([&](StoreOp s) {
      if (!store)
        store = s;
    });
    func.walk([&](ProgramIdOp p) {
      if (!programId)
        programId = p;
    });
    if (!store || !programId)
      return;
    found = RowSumMatch{load, reduce, store, programId, tileTy};
    ok = true;
  });
  if (!ok)
    return std::nullopt;
  return found;
}

static Type bufferToMemRef(Type type) {
  if (auto buffer = dyn_cast<BufferType>(type))
    return MemRefType::get(buffer.getShape(), buffer.getElementType());
  return type;
}

static void convertBufferSignature(func::FuncOp func) {
  auto fty = func.getFunctionType();
  SmallVector<Type> ins, outs;
  for (Type t : fty.getInputs())
    ins.push_back(bufferToMemRef(t));
  for (Type t : fty.getResults())
    outs.push_back(bufferToMemRef(t));
  func.setType(FunctionType::get(func.getContext(), ins, outs));
  for (auto [arg, ty] : llvm::zip(func.getArguments(), ins))
    arg.setType(ty);
}

static Value scalarZero(OpBuilder &b, Location loc, Type elem) {
  if (auto ft = dyn_cast<FloatType>(elem))
    return arith::ConstantOp::create(b, loc, FloatAttr::get(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elem))
    return arith::ConstantOp::create(b, loc, IntegerAttr::get(it, 0));
  llvm_unreachable("row-sum element type must be float or integer");
}

static Value scalarAdd(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::AddFOp::create(b, loc, lhs, rhs);
  return arith::AddIOp::create(b, loc, lhs, rhs);
}

/// Emit the physical row-sum body at the current insertion point.
static void emitRowSumGPU(OpBuilder &b, Location loc, Value in, Value out,
                          const GPUTargetInfo &target, int64_t tileRows,
                          int64_t tileCols, Type elemTy) {
  const int warpSize = target.warpSize;
  const int warps = target.warpsPerBlock();
  const int rowsPerWarp = target.rowsPerWarp(static_cast<int>(tileRows));
  const int elemsPerLane = target.elementsPerLane(static_cast<int>(tileCols));

  // Logical program instance. Not threadIdx / lane / warp.
  Value pid = ProgramIdOp::create(b, loc, b.getIndexType(),
                                  b.getI64IntegerAttr(0));

  // Physical ids. thread_id is recorded so the 256-thread block is visible;
  // the map uses subgroup (warp) and lane.
  (void)gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value lane = gpu::LaneIdOp::create(b, loc, /*upper_bound=*/IntegerAttr());
  Value warp = gpu::SubgroupIdOp::create(b, loc, b.getIndexType(),
                                        /*upper_bound=*/IntegerAttr());

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value cWarp = arith::ConstantIndexOp::create(b, loc, warpSize);
  Value cWarps = arith::ConstantIndexOp::create(b, loc, warps);
  Value cRows = arith::ConstantIndexOp::create(b, loc, rowsPerWarp);
  Value cElems = arith::ConstantIndexOp::create(b, loc, elemsPerLane);
  Value cTileRows = arith::ConstantIndexOp::create(b, loc, tileRows);
  Value cTileCols = tileCols == tileRows
                        ? cTileRows
                        : arith::ConstantIndexOp::create(b, loc, tileCols);
  Value zero = scalarZero(b, loc, elemTy);

  Value m = memref::DimOp::create(b, loc, in, c0);
  Value k = memref::DimOp::create(b, loc, in, c1);
  // Milestone 19: source used truncating divui; the GPU kernel uses ceil.
  Value numTiles = arith::CeilDivUIOp::create(b, loc, k, cTileCols);
  Value rowBase = arith::MulIOp::create(b, loc, pid, cTileRows);

  scf::ForOp::create(
      b, loc, c0, cRows, c1, ValueRange{},
      [&](OpBuilder &rb, Location rloc, Value s, ValueRange) {
        Value sScaled = arith::MulIOp::create(rb, rloc, s, cWarps);
        Value localRow = arith::AddIOp::create(rb, rloc, warp, sScaled);
        Value globalRow = arith::AddIOp::create(rb, rloc, rowBase, localRow);
        Value rowOK = arith::CmpIOp::create(rb, rloc, arith::CmpIPredicate::ult,
                                            globalRow, m);
        scf::IfOp::create(rb, rloc, rowOK, [&](OpBuilder &ib, Location iloc) {
          auto kLoop = scf::ForOp::create(
              ib, iloc, c0, numTiles, c1, ValueRange{zero},
              [&](OpBuilder &kb, Location kloc, Value kt, ValueRange kargs) {
                Value acc = kargs[0];
                Value kBase = arith::MulIOp::create(kb, kloc, kt, cTileCols);
                auto jLoop = scf::ForOp::create(
                    kb, kloc, c0, cElems, c1, ValueRange{acc},
                    [&](OpBuilder &jb, Location jloc, Value j,
                        ValueRange jargs) {
                      Value laneAcc = jargs[0];
                      Value jOff = arith::MulIOp::create(jb, jloc, j, cWarp);
                      Value col0 =
                          arith::AddIOp::create(jb, jloc, kBase, lane);
                      Value col = arith::AddIOp::create(jb, jloc, col0, jOff);
                      Value colOK = arith::CmpIOp::create(
                          jb, jloc, arith::CmpIPredicate::ult, col, k);
                      auto masked = scf::IfOp::create(
                          jb, jloc, TypeRange{elemTy}, colOK,
                          /*withElseRegion=*/true);
                      {
                        OpBuilder::InsertionGuard guard(jb);
                        jb.setInsertionPointToStart(masked.thenBlock());
                        Value ld = memref::LoadOp::create(
                            jb, jloc, in, ValueRange{globalRow, col});
                        scf::YieldOp::create(jb, jloc, ld);
                        jb.setInsertionPointToStart(masked.elseBlock());
                        scf::YieldOp::create(jb, jloc, zero);
                      }
                      Value sum =
                          scalarAdd(jb, jloc, laneAcc, masked.getResult(0));
                      scf::YieldOp::create(jb, jloc, sum);
                    });
                scf::YieldOp::create(kb, kloc, jLoop.getResult(0));
              });
          Value reduced = gpu::SubgroupReduceOp::create(
              ib, iloc, kLoop.getResult(0), gpu::AllReduceOperation::ADD,
              /*uniform=*/true);
          Value isLane0 = arith::CmpIOp::create(
              ib, iloc, arith::CmpIPredicate::eq, lane, c0);
          scf::IfOp::create(ib, iloc, isLane0, [&](OpBuilder &sb, Location sloc) {
            memref::StoreOp::create(sb, sloc, reduced, out,
                                    ValueRange{globalRow});
            scf::YieldOp::create(sb, sloc);
          });
          scf::YieldOp::create(ib, iloc);
        });
        scf::YieldOp::create(rb, rloc);
      });
}

struct ConvertTRRowSumToGPU
    : impl::ConvertTRRowSumToGPUBase<ConvertTRRowSumToGPU> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto match = matchRowSum(func);
    if (!match)
      return;

    convertBufferSignature(func);

    Value in = match->load.getBuffer();
    Value out = match->store.getBuffer();
    if (isa<BufferType>(in.getType()) || isa<BufferType>(out.getType())) {
      // Signature conversion updated block args; reload from the match
      // ops after their operand types were rewritten with the args.
      in = match->load.getBuffer();
      out = match->store.getBuffer();
    }
    if (!isa<MemRefType>(in.getType()) || !isa<MemRefType>(out.getType())) {
      func.emitError("row-sum GPU lowering expects memref buffers");
      signalPassFailure();
      return;
    }

    GPUTargetInfo target = GPUTargetInfo::fromOp(func);
    int64_t tileRows = match->tileTy.getDimSize(0);
    int64_t tileCols = match->tileTy.getDimSize(1);
    Type elemTy = match->tileTy.getElementType();
    if (tileRows % target.warpsPerBlock() != 0 ||
        tileCols % target.warpSize != 0) {
      func.emitError("tile shape is not divisible by the baseline GPU map");
      signalPassFailure();
      return;
    }

    func->setAttr("gpu.known_block_size",
                  DenseI32ArrayAttr::get(func.getContext(),
                                         {target.threadsPerBlock(), 1, 1}));
    target.applyTo(func);

    IRRewriter rewriter(func.getContext());
    Block *block = &func.getBody().front();
    while (!block->empty())
      rewriter.eraseOp(&block->back());

    rewriter.setInsertionPointToEnd(block);
    emitRowSumGPU(rewriter, func.getLoc(), in, out, target, tileRows, tileCols,
                  elemTy);
    func::ReturnOp::create(rewriter, func.getLoc());
  }
};

} // namespace
} // namespace mlir::tr
