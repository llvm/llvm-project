//===- TileReducerGPUKernels.cpp - Milestones 20-22 -------------*- C++ -*-===//
//
// Outline reductions into gpu.module @tr_kernels and launch them by
// SymbolRefAttr. SymbolTable is used for:
//   - lookup of an existing @tr_kernels
//   - insertion of generated gpu.func symbols
//   - uniqueness when a kernel name is already taken
//   - nested references @tr_kernels::@kernel on gpu.launch_func
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::tr;

namespace mlir::tr {
#define GEN_PASS_DEF_EMITTRGPUKERNELS
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

constexpr StringRef kKernelModuleName = "tr_kernels";
constexpr StringRef kRowSumKernel = "row_sum_kernel";
constexpr StringRef kRowSumSplitK1 = "row_sum_splitk_stage1";
constexpr StringRef kRowSumSplitK2 = "row_sum_splitk_stage2";
constexpr StringRef kFullSumStage1 = "full_sum_stage1";
constexpr StringRef kFullSumStage2 = "full_sum_stage2";
constexpr StringRef kColumnSumKernel = "column_sum_kernel";

enum class KernelKind { None, Row, Full, Column };

struct ReductionMatch {
  KernelKind kind = KernelKind::None;
  LoadOp load;
  StoreOp store;
  ProgramIdOp programId;
  TileType tileTy;
};

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
  llvm_unreachable("expected integer or float element");
}

static Value scalarAdd(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::AddFOp::create(b, loc, lhs, rhs);
  return arith::AddIOp::create(b, loc, lhs, rhs);
}

static MemRefType workgroupMemRef(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  Type elem) {
  auto space = gpu::AddressSpaceAttr::get(
      ctx, gpu::GPUDialect::getWorkgroupAddressSpace());
  return MemRefType::get(shape, elem, MemRefLayoutAttrInterface{}, space);
}

/// Look up @tr_kernels or insert a new gpu.module. Marks the host module
/// as a gpu.container_module so launch_func can form nested SymbolRefAttrs.
static gpu::GPUModuleOp getOrCreateKernelModule(ModuleOp module) {
  SymbolTable hostSymbols(module);
  if (Operation *found = hostSymbols.lookup(kKernelModuleName)) {
    if (auto gpuMod = dyn_cast<gpu::GPUModuleOp>(found))
      return gpuMod;
  }
  OpBuilder b(module.getBody(), module.getBody()->begin());
  auto gpuMod =
      gpu::GPUModuleOp::create(b, module.getLoc(), kKernelModuleName);
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  UnitAttr::get(module.getContext()));
  return gpuMod;
}

/// Insert `kernel` into the gpu.module symbol table. If `baseName` is taken,
/// SymbolTable::insert uniquifies the symbol.
static StringRef insertUniqueKernel(SymbolTable &gpuSymbols,
                                    gpu::GPUFuncOp kernel) {
  gpuSymbols.insert(kernel);
  return kernel.getName();
}

static SymbolRefAttr nestedKernelRef(gpu::GPUModuleOp gpuMod,
                                     StringRef kernelName) {
  return SymbolRefAttr::get(
      gpuMod.getNameAttr(),
      {FlatSymbolRefAttr::get(gpuMod.getContext(), kernelName)});
}

static ReductionMatch matchReduction(func::FuncOp func) {
  ReductionMatch found;
  func.walk([&](ReduceSumOp reduce) {
    auto inTy = dyn_cast<TileType>(reduce.getInput().getType());
    auto outTy = dyn_cast<TileType>(reduce.getType());
    if (!inTy || !outTy)
      return;
    LoadOp load;
    if (auto l = reduce.getInput().getDefiningOp<LoadOp>())
      load = l;
    else if (auto inner = reduce.getInput().getDefiningOp<ReduceSumOp>())
      load = inner.getInput().getDefiningOp<LoadOp>();
    if (!load)
      return;
    auto tileTy = dyn_cast<TileType>(load.getType());
    if (!tileTy || tileTy.getRank() != 2)
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

    KernelKind kind = KernelKind::None;
    if (outTy.getRank() == 0)
      kind = KernelKind::Full;
    else if (reduce.getAxis() == 1 && inTy.getRank() == 2)
      kind = KernelKind::Row;
    else if (reduce.getAxis() == 0 && inTy.getRank() == 2)
      kind = KernelKind::Column;
    if (kind == KernelKind::None)
      return;
    // Prefer Full over a preceding row reduce in the same function.
    if (found.kind == KernelKind::Full && kind != KernelKind::Full)
      return;
    found = ReductionMatch{kind, load, store, programId, tileTy};
  });
  return found;
}

/// Emit the fused row-sum body. `pid` is the logical program / block_id x.
/// Optional `ktLb`/`ktUb` restrict the K-tile loop (large-K split).
/// Optional `kPart` stores a rank-2 partial `out[row, kPart]`.
static void emitRowSumBody(OpBuilder &b, Location loc, Value in, Value out,
                           Value pid, const GPUTargetInfo &target,
                           int64_t tileRows, int64_t tileCols, Type elemTy,
                           Value ktLb = Value(), Value ktUb = Value(),
                           Value kPart = Value()) {
  const int warpSize = target.warpSize;
  const int warps = target.warpsPerBlock();
  const int rowsPerWarp = target.rowsPerWarp(static_cast<int>(tileRows));
  const int elemsPerLane = target.elementsPerLane(static_cast<int>(tileCols));

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
  Value numTiles = arith::CeilDivUIOp::create(b, loc, k, cTileCols);
  if (!ktLb)
    ktLb = c0;
  if (!ktUb)
    ktUb = numTiles;
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
              ib, iloc, ktLb, ktUb, c1, ValueRange{zero},
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
                      scf::YieldOp::create(
                          jb, jloc,
                          scalarAdd(jb, jloc, laneAcc, masked.getResult(0)));
                    });
                scf::YieldOp::create(kb, kloc, jLoop.getResult(0));
              });
          Value reduced = gpu::SubgroupReduceOp::create(
              ib, iloc, kLoop.getResult(0), gpu::AllReduceOperation::ADD,
              /*uniform=*/true);
          Value isLane0 = arith::CmpIOp::create(
              ib, iloc, arith::CmpIPredicate::eq, lane, c0);
          scf::IfOp::create(ib, iloc, isLane0, [&](OpBuilder &sb, Location sloc) {
            if (kPart)
              memref::StoreOp::create(sb, sloc, reduced, out,
                                      ValueRange{globalRow, kPart});
            else
              memref::StoreOp::create(sb, sloc, reduced, out,
                                      ValueRange{globalRow});
            scf::YieldOp::create(sb, sloc);
          });
          scf::YieldOp::create(ib, iloc);
        });
        scf::YieldOp::create(rb, rloc);
      });
}

/// Thread-local sum -> warp reduce -> smem[warp] -> barrier -> warp-0 tree.
/// Every warp executes the second subgroup_reduce on the same 8 smem values
/// so the op stays uniform. Only tid==0's result is stored.
static Value emitBlockReduce(OpBuilder &b, Location loc, Value threadAcc,
                             Value smem, Value lane, Value warp, int warps,
                             Type elemTy) {
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value cWarps = arith::ConstantIndexOp::create(b, loc, warps);
  Value zero = scalarZero(b, loc, elemTy);
  Value warpSum = gpu::SubgroupReduceOp::create(
      b, loc, threadAcc, gpu::AllReduceOperation::ADD, /*uniform=*/true);
  Value isLane0 =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, c0);
  scf::IfOp::create(b, loc, isLane0, [&](OpBuilder &sb, Location sloc) {
    memref::StoreOp::create(sb, sloc, warpSum, smem, ValueRange{warp});
    scf::YieldOp::create(sb, sloc);
  });
  gpu::BarrierOp::create(b, loc);
  Value inRange =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, lane, cWarps);
  auto loaded = scf::IfOp::create(b, loc, TypeRange{elemTy}, inRange,
                                  /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loaded.thenBlock());
    Value v = memref::LoadOp::create(b, loc, smem, ValueRange{lane});
    scf::YieldOp::create(b, loc, v);
    b.setInsertionPointToStart(loaded.elseBlock());
    scf::YieldOp::create(b, loc, zero);
  }
  return gpu::SubgroupReduceOp::create(b, loc, loaded.getResult(0),
                                       gpu::AllReduceOperation::ADD,
                                       /*uniform=*/true);
}

static void emitFullSumStage1Body(OpBuilder &b, Location loc, Value in,
                                  Value partials, Value smem, Value bid,
                                  const GPUTargetInfo &target, int64_t tileRows,
                                  int64_t tileCols, Type elemTy) {
  const int warpSize = target.warpSize;
  const int warps = target.warpsPerBlock();
  const int rowsPerWarp = target.rowsPerWarp(static_cast<int>(tileRows));
  const int elemsPerLane = target.elementsPerLane(static_cast<int>(tileCols));

  Value tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value lane = gpu::LaneIdOp::create(b, loc, IntegerAttr());
  Value warp = gpu::SubgroupIdOp::create(b, loc, b.getIndexType(), IntegerAttr());

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
  Value numTiles = arith::CeilDivUIOp::create(b, loc, k, cTileCols);
  Value rowBase = arith::MulIOp::create(b, loc, bid, cTileRows);

  auto rowLoop = scf::ForOp::create(
      b, loc, c0, cRows, c1, ValueRange{zero},
      [&](OpBuilder &rb, Location rloc, Value s, ValueRange args) {
        Value acc = args[0];
        Value sScaled = arith::MulIOp::create(rb, rloc, s, cWarps);
        Value localRow = arith::AddIOp::create(rb, rloc, warp, sScaled);
        Value globalRow = arith::AddIOp::create(rb, rloc, rowBase, localRow);
        Value rowOK = arith::CmpIOp::create(rb, rloc, arith::CmpIPredicate::ult,
                                            globalRow, m);
        auto rowIf = scf::IfOp::create(rb, rloc, TypeRange{elemTy}, rowOK,
                                       /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard guard(rb);
          rb.setInsertionPointToStart(rowIf.thenBlock());
          auto kLoop = scf::ForOp::create(
              rb, rloc, c0, numTiles, c1, ValueRange{acc},
              [&](OpBuilder &kb, Location kloc, Value kt, ValueRange kargs) {
                Value kAcc = kargs[0];
                Value kBase = arith::MulIOp::create(kb, kloc, kt, cTileCols);
                auto jLoop = scf::ForOp::create(
                    kb, kloc, c0, cElems, c1, ValueRange{kAcc},
                    [&](OpBuilder &jb, Location jloc, Value j,
                        ValueRange jargs) {
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
                        OpBuilder::InsertionGuard g2(jb);
                        jb.setInsertionPointToStart(masked.thenBlock());
                        Value ld = memref::LoadOp::create(
                            jb, jloc, in, ValueRange{globalRow, col});
                        scf::YieldOp::create(jb, jloc, ld);
                        jb.setInsertionPointToStart(masked.elseBlock());
                        scf::YieldOp::create(jb, jloc, zero);
                      }
                      scf::YieldOp::create(
                          jb, jloc,
                          scalarAdd(jb, jloc, jargs[0], masked.getResult(0)));
                    });
                scf::YieldOp::create(kb, kloc, jLoop.getResult(0));
              });
          scf::YieldOp::create(rb, rloc, kLoop.getResult(0));
          rb.setInsertionPointToStart(rowIf.elseBlock());
          scf::YieldOp::create(rb, rloc, acc);
        }
        scf::YieldOp::create(rb, rloc, rowIf.getResult(0));
      });

  Value blockSum = emitBlockReduce(b, loc, rowLoop.getResult(0), smem, lane,
                                   warp, warps, elemTy);
  Value isTid0 =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tid, c0);
  scf::IfOp::create(b, loc, isTid0, [&](OpBuilder &sb, Location sloc) {
    memref::StoreOp::create(sb, sloc, blockSum, partials, ValueRange{bid});
    scf::YieldOp::create(sb, sloc);
  });
}

static void emitFullSumStage2Body(OpBuilder &b, Location loc, Value partials,
                                  Value out, Value smem,
                                  const GPUTargetInfo &target, Type elemTy) {
  const int warps = target.warpsPerBlock();
  Value tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value lane = gpu::LaneIdOp::create(b, loc, IntegerAttr());
  Value warp = gpu::SubgroupIdOp::create(b, loc, b.getIndexType(), IntegerAttr());
  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value c256 = arith::ConstantIndexOp::create(b, loc, target.threadsPerBlock());
  Value zero = scalarZero(b, loc, elemTy);
  Value nPart = memref::DimOp::create(b, loc, partials, c0);

  auto loop = scf::ForOp::create(
      b, loc, tid, nPart, c256, ValueRange{zero},
      [&](OpBuilder &lb, Location lloc, Value i, ValueRange args) {
        Value v = memref::LoadOp::create(lb, lloc, partials, ValueRange{i});
        scf::YieldOp::create(lb, lloc, scalarAdd(lb, lloc, args[0], v));
      });
  Value blockSum = emitBlockReduce(b, loc, loop.getResult(0), smem, lane, warp,
                                   warps, elemTy);
  Value isTid0 =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, tid, c0);
  scf::IfOp::create(b, loc, isTid0, [&](OpBuilder &sb, Location sloc) {
    memref::StoreOp::create(sb, sloc, blockSum, out, ValueRange{c0});
    scf::YieldOp::create(sb, sloc);
  });
}

/// Coalesced row-major loads into smem, barrier, then each of 128 threads
/// reduces one column. 128 is already a multiple of the warp size, so a
/// padded 128x132 layout is not justified for the baseline.
static void emitColumnSumBody(OpBuilder &b, Location loc, Value in, Value out,
                              Value smem, Value pid,
                              const GPUTargetInfo &target, int64_t tileRows,
                              int64_t tileCols, Type elemTy) {
  const int warpSize = target.warpSize;
  const int warps = target.warpsPerBlock();
  const int rowsPerWarp = target.rowsPerWarp(static_cast<int>(tileRows));
  const int elemsPerLane = target.elementsPerLane(static_cast<int>(tileCols));

  Value tid = gpu::ThreadIdOp::create(b, loc, gpu::Dimension::x);
  Value lane = gpu::LaneIdOp::create(b, loc, IntegerAttr());
  Value warp = gpu::SubgroupIdOp::create(b, loc, b.getIndexType(), IntegerAttr());

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
  Value numRowTiles = arith::CeilDivUIOp::create(b, loc, m, cTileRows);
  Value colBase = arith::MulIOp::create(b, loc, pid, cTileCols);

  Value ownsCol =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, tid, cTileCols);
  Value globalCol = arith::AddIOp::create(b, loc, colBase, tid);
  Value colOK =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, globalCol, k);

  auto mtLoop = scf::ForOp::create(
      b, loc, c0, numRowTiles, c1, ValueRange{zero},
      [&](OpBuilder &mb, Location mloc, Value mt, ValueRange margs) {
        Value colAcc = margs[0];
        Value rowBase = arith::MulIOp::create(mb, mloc, mt, cTileRows);
        scf::ForOp::create(
            mb, mloc, c0, cRows, c1, ValueRange{},
            [&](OpBuilder &rb, Location rloc, Value s, ValueRange) {
              Value sScaled = arith::MulIOp::create(rb, rloc, s, cWarps);
              Value localRow = arith::AddIOp::create(rb, rloc, warp, sScaled);
              Value globalRow =
                  arith::AddIOp::create(rb, rloc, rowBase, localRow);
              Value rowOK = arith::CmpIOp::create(
                  rb, rloc, arith::CmpIPredicate::ult, globalRow, m);
              scf::ForOp::create(
                  rb, rloc, c0, cElems, c1, ValueRange{},
                  [&](OpBuilder &jb, Location jloc, Value j, ValueRange) {
                    Value jOff = arith::MulIOp::create(jb, jloc, j, cWarp);
                    Value localCol =
                        arith::AddIOp::create(jb, jloc, lane, jOff);
                    Value gCol =
                        arith::AddIOp::create(jb, jloc, colBase, localCol);
                    Value inB = arith::AndIOp::create(
                        jb, jloc, rowOK,
                        arith::CmpIOp::create(jb, jloc,
                                              arith::CmpIPredicate::ult, gCol,
                                              k));
                    auto masked = scf::IfOp::create(
                        jb, jloc, TypeRange{elemTy}, inB,
                        /*withElseRegion=*/true);
                    {
                      OpBuilder::InsertionGuard guard(jb);
                      jb.setInsertionPointToStart(masked.thenBlock());
                      Value ld = memref::LoadOp::create(
                          jb, jloc, in, ValueRange{globalRow, gCol});
                      scf::YieldOp::create(jb, jloc, ld);
                      jb.setInsertionPointToStart(masked.elseBlock());
                      scf::YieldOp::create(jb, jloc, zero);
                    }
                    memref::StoreOp::create(jb, jloc, masked.getResult(0), smem,
                                            ValueRange{localRow, localCol});
                    scf::YieldOp::create(jb, jloc);
                  });
              scf::YieldOp::create(rb, rloc);
            });
        gpu::BarrierOp::create(mb, mloc);
        auto reduced = scf::IfOp::create(mb, mloc, TypeRange{elemTy}, ownsCol,
                                         /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard guard(mb);
          mb.setInsertionPointToStart(reduced.thenBlock());
          auto rLoop = scf::ForOp::create(
              mb, mloc, c0, cTileRows, c1, ValueRange{colAcc},
              [&](OpBuilder &rb, Location rloc, Value r, ValueRange rargs) {
                Value v =
                    memref::LoadOp::create(rb, rloc, smem, ValueRange{r, tid});
                scf::YieldOp::create(rb, rloc, scalarAdd(rb, rloc, rargs[0], v));
              });
          scf::YieldOp::create(mb, mloc, rLoop.getResult(0));
          mb.setInsertionPointToStart(reduced.elseBlock());
          scf::YieldOp::create(mb, mloc, colAcc);
        }
        gpu::BarrierOp::create(mb, mloc);
        scf::YieldOp::create(mb, mloc, reduced.getResult(0));
      });

  Value canStore = arith::AndIOp::create(b, loc, ownsCol, colOK);
  scf::IfOp::create(b, loc, canStore, [&](OpBuilder &sb, Location sloc) {
    memref::StoreOp::create(sb, sloc, mtLoop.getResult(0), out,
                            ValueRange{globalCol});
    scf::YieldOp::create(sb, sloc);
  });
}

static gpu::GPUFuncOp createKernel(OpBuilder &b, Location loc, StringRef name,
                                   TypeRange inputs, TypeRange workgroup,
                                   const GPUTargetInfo &target) {
  auto fnTy = FunctionType::get(b.getContext(), inputs, {});
  auto kernel = gpu::GPUFuncOp::create(b, loc, name, fnTy, workgroup);
  kernel.setKernel(true);
  kernel.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(
      b.getContext(), {target.threadsPerBlock(), 1, 1}));
  return kernel;
}

static void emitHostLaunch(OpBuilder &b, Location loc, gpu::GPUFuncOp kernel,
                           ValueRange args, Value gridX,
                           const GPUTargetInfo &target, Value gridY = Value()) {
  Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
  Value cBlock =
      arith::ConstantIndexOp::create(b, loc, target.threadsPerBlock());
  if (!gridY)
    gridY = c1;
  gpu::KernelDim3 grid{gridX, gridY, c1};
  gpu::KernelDim3 block{cBlock, c1, c1};
  gpu::LaunchFuncOp::create(b, loc, kernel, grid, block, /*dynSmem=*/Value(),
                            args);
}

static void clearBody(func::FuncOp func) {
  Block *block = &func.getBody().front();
  while (!block->empty())
    block->back().erase();
}

static LogicalResult emitRowSumKernel(func::FuncOp func, ReductionMatch match,
                                      gpu::GPUModuleOp gpuMod,
                                      SymbolTable &gpuSymbols,
                                      const GPUTargetInfo &target) {
  convertBufferSignature(func);
  Value in = match.load.getBuffer();
  Value out = match.store.getBuffer();
  if (!isa<MemRefType>(in.getType()) || !isa<MemRefType>(out.getType()))
    return func.emitError("row-sum kernel expects memref buffers");

  Location loc = func.getLoc();
  Type elemTy = match.tileTy.getElementType();
  int64_t tileRows = match.tileTy.getDimSize(0);
  int64_t tileCols = match.tileTy.getDimSize(1);

  OpBuilder kb(gpuMod.getBody(), gpuMod.getBody()->end());
  auto kernel = createKernel(kb, loc, kRowSumKernel,
                             TypeRange{in.getType(), out.getType()}, {},
                             target);
  insertUniqueKernel(gpuSymbols, kernel);
  // Re-lookup to exercise SymbolTable::lookup after insert.
  auto lookedUp = gpuSymbols.lookup<gpu::GPUFuncOp>(kernel.getName());
  if (!lookedUp)
    return func.emitError("symbol lookup failed for ") << kernel.getName();

  Block &kentry = kernel.getBody().front();
  OpBuilder body(&kentry, kentry.begin());
  Value pid = gpu::BlockIdOp::create(body, loc, gpu::Dimension::x);
  emitRowSumBody(body, loc, kentry.getArgument(0), kentry.getArgument(1), pid,
                 target, tileRows, tileCols, elemTy);
  gpu::ReturnOp::create(body, loc);

  clearBody(func);
  OpBuilder hb(&func.getBody().front(), func.getBody().front().end());
  Value c0 = arith::ConstantIndexOp::create(hb, loc, 0);
  Value cTile = arith::ConstantIndexOp::create(hb, loc, tileRows);
  Value m = memref::DimOp::create(hb, loc, in, c0);
  Value gridX = arith::CeilDivUIOp::create(hb, loc, m, cTile);
  emitHostLaunch(hb, loc, lookedUp, ValueRange{in, out}, gridX, target);
  func::ReturnOp::create(hb, loc);
  (void)nestedKernelRef(gpuMod, lookedUp.getName());
  return success();
}

/// One logical program (block_id x) is refined into many physical blocks
/// (block_id y) that each own a K-slice and write a partial. Stage 2 reduces
/// those partials. This is the M=1, K=1e8 case: one row is not one block.
static LogicalResult emitRowSumSplitK(func::FuncOp func, ReductionMatch match,
                                      gpu::GPUModuleOp gpuMod,
                                      SymbolTable &gpuSymbols,
                                      const GPUTargetInfo &target,
                                      int kSplits) {
  convertBufferSignature(func);
  Value in = match.load.getBuffer();
  Value out = match.store.getBuffer();
  if (!isa<MemRefType>(in.getType()) || !isa<MemRefType>(out.getType()))
    return func.emitError("row-sum split-K expects memref buffers");

  Location loc = func.getLoc();
  Type elemTy = match.tileTy.getElementType();
  int64_t tileRows = match.tileTy.getDimSize(0);
  int64_t tileCols = match.tileTy.getDimSize(1);
  auto partialsTy =
      MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elemTy);

  OpBuilder kb(gpuMod.getBody(), gpuMod.getBody()->end());
  auto stage1 = createKernel(kb, loc, kRowSumSplitK1,
                             TypeRange{in.getType(), partialsTy}, {}, target);
  insertUniqueKernel(gpuSymbols, stage1);
  auto stage2 = createKernel(kb, loc, kRowSumSplitK2,
                             TypeRange{partialsTy, out.getType()}, {}, target);
  insertUniqueKernel(gpuSymbols, stage2);
  if (!gpuSymbols.lookup<gpu::GPUFuncOp>(stage1.getName()) ||
      !gpuSymbols.lookup<gpu::GPUFuncOp>(stage2.getName()))
    return func.emitError("split-K symbol lookup failed");

  {
    Block &e = stage1.getBody().front();
    OpBuilder body(&e, e.begin());
    Value pid = gpu::BlockIdOp::create(body, loc, gpu::Dimension::x);
    Value kPart = gpu::BlockIdOp::create(body, loc, gpu::Dimension::y);
    Value nSplits = gpu::GridDimOp::create(body, loc, gpu::Dimension::y);
    Value c1 = arith::ConstantIndexOp::create(body, loc, 1);
    Value cTile = arith::ConstantIndexOp::create(body, loc, tileCols);
    Value k = memref::DimOp::create(body, loc, e.getArgument(0), c1);
    Value nTiles = arith::CeilDivUIOp::create(body, loc, k, cTile);
    Value chunk = arith::CeilDivUIOp::create(body, loc, nTiles, nSplits);
    Value ktLb = arith::MulIOp::create(body, loc, kPart, chunk);
    Value ktUb = arith::AddIOp::create(body, loc, ktLb, chunk);
    Value over =
        arith::CmpIOp::create(body, loc, arith::CmpIPredicate::ugt, ktUb, nTiles);
    auto clamped = scf::IfOp::create(body, loc, TypeRange{body.getIndexType()},
                                     over, /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(body);
      body.setInsertionPointToStart(clamped.thenBlock());
      scf::YieldOp::create(body, loc, nTiles);
      body.setInsertionPointToStart(clamped.elseBlock());
      scf::YieldOp::create(body, loc, ktUb);
    }
    emitRowSumBody(body, loc, e.getArgument(0), e.getArgument(1), pid, target,
                   tileRows, tileCols, elemTy, ktLb, clamped.getResult(0),
                   kPart);
    gpu::ReturnOp::create(body, loc);
  }
  {
    Block &e = stage2.getBody().front();
    OpBuilder body(&e, e.begin());
    Value pid = gpu::BlockIdOp::create(body, loc, gpu::Dimension::x);
    Value lane = gpu::LaneIdOp::create(body, loc, IntegerAttr());
    Value warp =
        gpu::SubgroupIdOp::create(body, loc, body.getIndexType(), IntegerAttr());
    Value c0 = arith::ConstantIndexOp::create(body, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(body, loc, 1);
    Value c8 = arith::ConstantIndexOp::create(body, loc, target.warpsPerBlock());
    Value c16 = arith::ConstantIndexOp::create(
        body, loc, target.rowsPerWarp(static_cast<int>(tileRows)));
    Value c128 = arith::ConstantIndexOp::create(body, loc, tileRows);
    Value zero = scalarZero(body, loc, elemTy);
    Value nSplits = memref::DimOp::create(body, loc, e.getArgument(0), c1);
    Value m = memref::DimOp::create(body, loc, e.getArgument(1), c0);
    Value rowBase = arith::MulIOp::create(body, loc, pid, c128);
    scf::ForOp::create(
        body, loc, c0, c16, c1, ValueRange{},
        [&](OpBuilder &rb, Location rloc, Value s, ValueRange) {
          Value local =
              arith::AddIOp::create(rb, rloc, warp,
                                    arith::MulIOp::create(rb, rloc, s, c8));
          Value row = arith::AddIOp::create(rb, rloc, rowBase, local);
          Value rowOK = arith::CmpIOp::create(rb, rloc, arith::CmpIPredicate::ult,
                                              row, m);
          scf::IfOp::create(rb, rloc, rowOK, [&](OpBuilder &ib, Location iloc) {
            auto red = scf::ForOp::create(
                ib, iloc, c0, nSplits, c1, ValueRange{zero},
                [&](OpBuilder &kb, Location kloc, Value p, ValueRange args) {
                  Value v = memref::LoadOp::create(
                      kb, kloc, e.getArgument(0), ValueRange{row, p});
                  scf::YieldOp::create(kb, kloc,
                                       scalarAdd(kb, kloc, args[0], v));
                });
            Value isLane0 = arith::CmpIOp::create(
                ib, iloc, arith::CmpIPredicate::eq, lane, c0);
            scf::IfOp::create(ib, iloc, isLane0, [&](OpBuilder &sb, Location sloc) {
              memref::StoreOp::create(sb, sloc, red.getResult(0),
                                      e.getArgument(1), ValueRange{row});
              scf::YieldOp::create(sb, sloc);
            });
            scf::YieldOp::create(ib, iloc);
          });
          scf::YieldOp::create(rb, rloc);
        });
    gpu::ReturnOp::create(body, loc);
  }

  clearBody(func);
  OpBuilder hb(&func.getBody().front(), func.getBody().front().end());
  Value c0 = arith::ConstantIndexOp::create(hb, loc, 0);
  Value cTile = arith::ConstantIndexOp::create(hb, loc, tileRows);
  Value cSplits = arith::ConstantIndexOp::create(hb, loc, kSplits);
  Value m = memref::DimOp::create(hb, loc, in, c0);
  Value nRowTiles = arith::CeilDivUIOp::create(hb, loc, m, cTile);
  Value partials =
      memref::AllocOp::create(hb, loc, partialsTy, ValueRange{m, cSplits});
  emitHostLaunch(hb, loc, stage1, ValueRange{in, partials}, nRowTiles, target,
                 cSplits);
  emitHostLaunch(hb, loc, stage2, ValueRange{partials, out}, nRowTiles, target);
  memref::DeallocOp::create(hb, loc, partials);
  func::ReturnOp::create(hb, loc);
  return success();
}

static LogicalResult emitFullSumKernels(func::FuncOp func, ReductionMatch match,
                                        gpu::GPUModuleOp gpuMod,
                                        SymbolTable &gpuSymbols,
                                        const GPUTargetInfo &target) {
  convertBufferSignature(func);
  Value in = match.load.getBuffer();
  Value out = match.store.getBuffer();
  if (!isa<MemRefType>(in.getType()) || !isa<MemRefType>(out.getType()))
    return func.emitError("full-sum kernel expects memref buffers");

  Location loc = func.getLoc();
  MLIRContext *ctx = func.getContext();
  Type elemTy = match.tileTy.getElementType();
  int64_t tileRows = match.tileTy.getDimSize(0);
  int64_t tileCols = match.tileTy.getDimSize(1);
  auto partialsTy = MemRefType::get({ShapedType::kDynamic}, elemTy);
  auto smemTy = workgroupMemRef(ctx, {target.warpsPerBlock()}, elemTy);

  OpBuilder kb(gpuMod.getBody(), gpuMod.getBody()->end());
  auto stage1 = createKernel(kb, loc, kFullSumStage1,
                             TypeRange{in.getType(), partialsTy},
                             TypeRange{smemTy}, target);
  insertUniqueKernel(gpuSymbols, stage1);
  auto stage2 = createKernel(kb, loc, kFullSumStage2,
                             TypeRange{partialsTy, out.getType()},
                             TypeRange{smemTy}, target);
  insertUniqueKernel(gpuSymbols, stage2);

  if (!gpuSymbols.lookup<gpu::GPUFuncOp>(stage1.getName()) ||
      !gpuSymbols.lookup<gpu::GPUFuncOp>(stage2.getName()))
    return func.emitError("full-sum symbol lookup failed");

  {
    Block &e = stage1.getBody().front();
    OpBuilder body(&e, e.begin());
    Value bid = gpu::BlockIdOp::create(body, loc, gpu::Dimension::x);
    emitFullSumStage1Body(body, loc, e.getArgument(0), e.getArgument(1),
                          e.getArgument(2), bid, target, tileRows, tileCols,
                          elemTy);
    gpu::ReturnOp::create(body, loc);
  }
  {
    Block &e = stage2.getBody().front();
    OpBuilder body(&e, e.begin());
    emitFullSumStage2Body(body, loc, e.getArgument(0), e.getArgument(1),
                          e.getArgument(2), target, elemTy);
    gpu::ReturnOp::create(body, loc);
  }

  clearBody(func);
  OpBuilder hb(&func.getBody().front(), func.getBody().front().end());
  Value c0 = arith::ConstantIndexOp::create(hb, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(hb, loc, 1);
  Value cTile = arith::ConstantIndexOp::create(hb, loc, tileRows);
  Value m = memref::DimOp::create(hb, loc, in, c0);
  Value nBlocks = arith::CeilDivUIOp::create(hb, loc, m, cTile);
  Value partials = memref::AllocOp::create(hb, loc, partialsTy, ValueRange{nBlocks});
  emitHostLaunch(hb, loc, stage1, ValueRange{in, partials}, nBlocks, target);
  emitHostLaunch(hb, loc, stage2, ValueRange{partials, out}, c1, target);
  memref::DeallocOp::create(hb, loc, partials);
  func::ReturnOp::create(hb, loc);
  return success();
}

static LogicalResult emitColumnSumKernel(func::FuncOp func, ReductionMatch match,
                                         gpu::GPUModuleOp gpuMod,
                                         SymbolTable &gpuSymbols,
                                         const GPUTargetInfo &target) {
  convertBufferSignature(func);
  Value in = match.load.getBuffer();
  Value out = match.store.getBuffer();
  if (!isa<MemRefType>(in.getType()) || !isa<MemRefType>(out.getType()))
    return func.emitError("column-sum kernel expects memref buffers");

  Location loc = func.getLoc();
  Type elemTy = match.tileTy.getElementType();
  int64_t tileRows = match.tileTy.getDimSize(0);
  int64_t tileCols = match.tileTy.getDimSize(1);
  auto smemTy =
      workgroupMemRef(func.getContext(), {tileRows, tileCols}, elemTy);

  OpBuilder kb(gpuMod.getBody(), gpuMod.getBody()->end());
  auto kernel = createKernel(kb, loc, kColumnSumKernel,
                             TypeRange{in.getType(), out.getType()},
                             TypeRange{smemTy}, target);
  insertUniqueKernel(gpuSymbols, kernel);
  auto lookedUp = gpuSymbols.lookup<gpu::GPUFuncOp>(kernel.getName());
  if (!lookedUp)
    return func.emitError("symbol lookup failed for ") << kernel.getName();

  Block &e = kernel.getBody().front();
  OpBuilder body(&e, e.begin());
  Value pid = gpu::BlockIdOp::create(body, loc, gpu::Dimension::x);
  emitColumnSumBody(body, loc, e.getArgument(0), e.getArgument(1),
                    e.getArgument(2), pid, target, tileRows, tileCols, elemTy);
  gpu::ReturnOp::create(body, loc);

  clearBody(func);
  OpBuilder hb(&func.getBody().front(), func.getBody().front().end());
  Value c1 = arith::ConstantIndexOp::create(hb, loc, 1);
  Value cTile = arith::ConstantIndexOp::create(hb, loc, tileCols);
  Value k = memref::DimOp::create(hb, loc, in, c1);
  Value gridX = arith::CeilDivUIOp::create(hb, loc, k, cTile);
  emitHostLaunch(hb, loc, lookedUp, ValueRange{in, out}, gridX, target);
  func::ReturnOp::create(hb, loc);
  return success();
}

struct EmitTRGPUKernels : impl::EmitTRGPUKernelsBase<EmitTRGPUKernels> {
  using impl::EmitTRGPUKernelsBase<EmitTRGPUKernels>::EmitTRGPUKernelsBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<std::pair<func::FuncOp, ReductionMatch>> work;
    module.walk([&](func::FuncOp func) {
      if (func->getParentOfType<gpu::GPUModuleOp>())
        return;
      ReductionMatch match = matchReduction(func);
      if (match.kind != KernelKind::None)
        work.emplace_back(func, match);
    });
    if (work.empty())
      return;

    gpu::GPUModuleOp gpuMod = getOrCreateKernelModule(module);
    SymbolTable gpuSymbols(gpuMod);
    GPUTargetInfo target = GPUTargetInfo::fromOp(module);

    int splits = kSplits;
    if (splits <= 1) {
      if (auto attr = module->getAttrOfType<IntegerAttr>("tr.tune.k_splits"))
        splits = static_cast<int>(attr.getInt());
    }

    for (auto [func, match] : work) {
      LogicalResult ok = success();
      switch (match.kind) {
      case KernelKind::Row:
        ok = splits > 1 ? emitRowSumSplitK(func, match, gpuMod, gpuSymbols,
                                           target, splits)
                        : emitRowSumKernel(func, match, gpuMod, gpuSymbols,
                                           target);
        break;
      case KernelKind::Full:
        ok = emitFullSumKernels(func, match, gpuMod, gpuSymbols, target);
        break;
      case KernelKind::Column:
        ok = emitColumnSumKernel(func, match, gpuMod, gpuSymbols, target);
        break;
      case KernelKind::None:
        break;
      }
      if (failed(ok)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
} // namespace mlir::tr
