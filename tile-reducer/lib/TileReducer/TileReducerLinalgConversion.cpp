//===- TileReducerLinalgConversion.cpp - Milestones 8-9 ---------*- C++ -*-===//
//
// Lower tile compute to Linalg over MemRefs. A loaded tile is a subview of
// the input buffer, not a 128x128 allocation.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerPasses.h"

#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tr {
#define GEN_PASS_DEF_CONVERTTRTOLINALG
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

static Type convertTRType(Type type) {
  if (auto tile = dyn_cast<TileType>(type))
    return MemRefType::get(tile.getShape(), tile.getElementType());
  if (auto buffer = dyn_cast<BufferType>(type))
    return MemRefType::get(buffer.getShape(), buffer.getElementType());
  return type;
}

static int64_t shapedRank(Type type) {
  if (auto tile = dyn_cast<TileType>(type))
    return tile.getRank();
  if (auto memref = dyn_cast<MemRefType>(type))
    return memref.getRank();
  if (auto buffer = dyn_cast<BufferType>(type))
    return buffer.getRank();
  return -1;
}

static MemRefType memrefForTileLike(Type type) {
  if (auto memref = dyn_cast<MemRefType>(type))
    return memref;
  if (auto tile = dyn_cast<TileType>(type))
    return MemRefType::get(tile.getShape(), tile.getElementType());
  llvm_unreachable("expected a tile or memref");
}

static Value createScalarAdd(OpBuilder &b, Location loc, Value lhs, Value rhs) {
  if (isa<FloatType>(lhs.getType()))
    return arith::AddFOp::create(b, loc, lhs, rhs);
  return arith::AddIOp::create(b, loc, lhs, rhs);
}

static Value materializeScalar(OpBuilder &b, Location loc, Attribute attr,
                               Type elem) {
  if (auto ft = dyn_cast<FloatType>(elem)) {
    APFloat value(ft.getFloatSemantics(), APInt::getZero(ft.getWidth()));
    if (auto fattr = dyn_cast<FloatAttr>(attr)) {
      value = fattr.getValue();
      bool losesInfo = false;
      value.convert(ft.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
    }
    return arith::ConstantOp::create(b, loc, FloatAttr::get(ft, value));
  }
  if (auto it = dyn_cast<IntegerType>(elem)) {
    APInt value(it.getWidth(), 0);
    if (auto iattr = dyn_cast<IntegerAttr>(attr))
      value = iattr.getValue().sextOrTrunc(it.getWidth());
    return arith::ConstantOp::create(b, loc, IntegerAttr::get(it, value));
  }
  return {};
}

static Value createZeroScalar(OpBuilder &b, Location loc, Type elem) {
  if (auto ft = dyn_cast<FloatType>(elem))
    return arith::ConstantOp::create(b, loc, b.getFloatAttr(ft, 0.0));
  if (auto it = dyn_cast<IntegerType>(elem))
    return arith::ConstantOp::create(b, loc, b.getIntegerAttr(it, 0));
  return {};
}

static Value createAlloca(OpBuilder &b, Location loc, MemRefType type) {
  return memref::AllocaOp::create(b, loc, type);
}

static void fillMemRef(OpBuilder &b, Location loc, Value dest, Value scalar) {
  linalg::FillOp::create(b, loc, ValueRange{scalar}, ValueRange{dest});
}

static void createReductionGeneric(OpBuilder &b, Location loc, Value input,
                                   Value dest, ArrayRef<int64_t> reducedDims) {
  auto inTy = cast<MemRefType>(input.getType());
  int64_t rank = inTy.getRank();
  MLIRContext *ctx = b.getContext();

  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  SmallVector<char> isReduced(rank, 0);
  for (int64_t dim : reducedDims) {
    iters[dim] = utils::IteratorType::reduction;
    isReduced[dim] = 1;
  }

  AffineMap inMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
  SmallVector<AffineExpr> outExprs;
  for (int64_t i = 0; i < rank; ++i)
    if (!isReduced[i])
      outExprs.push_back(getAffineDimExpr(i, ctx));
  AffineMap outMap = AffineMap::get(rank, /*symbolCount=*/0, outExprs, ctx);

  linalg::GenericOp::create(
      b, loc, ValueRange{input}, ValueRange{dest},
      ArrayRef<AffineMap>{inMap, outMap}, iters,
      [&](OpBuilder &rb, Location rloc, ValueRange args) {
        linalg::YieldOp::create(rb, rloc,
                                createScalarAdd(rb, rloc, args[0], args[1]));
      });
}

static void createAddGeneric(OpBuilder &b, Location loc, Value lhs, Value rhs,
                             Value dest) {
  auto ty = cast<MemRefType>(dest.getType());
  int64_t rank = ty.getRank();
  MLIRContext *ctx = b.getContext();
  AffineMap id = AffineMap::getMultiDimIdentityMap(rank, ctx);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  linalg::GenericOp::create(
      b, loc, ValueRange{lhs, rhs}, ValueRange{dest},
      ArrayRef<AffineMap>{id, id, id}, iters,
      [&](OpBuilder &rb, Location rloc, ValueRange args) {
        linalg::YieldOp::create(rb, rloc,
                                createScalarAdd(rb, rloc, args[0], args[1]));
      });
}

static FailureOr<Value> createTileSubview(OpBuilder &b, Location loc,
                                          Value buffer, ValueRange indices,
                                          ArrayRef<int64_t> tileShape) {
  auto memTy = dyn_cast<MemRefType>(buffer.getType());
  if (!memTy)
    return failure();
  if (static_cast<int64_t>(indices.size()) != memTy.getRank() ||
      indices.size() != tileShape.size())
    return failure();

  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (auto [idx, extent] : llvm::zip(indices, tileShape)) {
    Value offset = idx;
    if (extent != 1) {
      Value c = arith::ConstantIndexOp::create(b, loc, extent);
      offset = arith::MulIOp::create(b, loc, idx, c);
    }
    offsets.push_back(offset);
    sizes.push_back(b.getIndexAttr(extent));
    strides.push_back(b.getIndexAttr(1));
  }
  return memref::SubViewOp::create(b, loc, buffer, offsets, sizes, strides)
      .getResult();
}

static void convertSignature(func::FuncOp func) {
  auto oldTy = func.getFunctionType();
  SmallVector<Type> ins, outs;
  for (Type t : oldTy.getInputs())
    ins.push_back(convertTRType(t));
  for (Type t : oldTy.getResults())
    outs.push_back(convertTRType(t));
  if (ins == oldTy.getInputs() && outs == oldTy.getResults())
    return;
  func.setType(FunctionType::get(func.getContext(), ins, outs));
  for (auto [arg, ty] : llvm::zip(func.getArguments(), ins))
    if (arg.getType() != ty)
      arg.setType(ty);
}

static void convertForTypes(ForOp op) {
  for (BlockArgument arg : op.getRegionIterArgs()) {
    Type converted = convertTRType(arg.getType());
    if (converted != arg.getType())
      arg.setType(converted);
  }
  for (OpResult res : op->getResults()) {
    Type converted = convertTRType(res.getType());
    if (converted != res.getType())
      res.setType(converted);
  }
}

static bool isInnerOfFusedFullReduce(ReduceSumOp op) {
  if (!op->hasOneUse())
    return false;
  auto outer = dyn_cast<ReduceSumOp>(*op->getUsers().begin());
  if (!outer)
    return false;
  return shapedRank(op.getInput().getType()) == 2 &&
         shapedRank(op.getType()) == 1 && shapedRank(outer.getType()) == 0;
}

static LogicalResult lowerConstant(RewriterBase &rewriter, ConstantOp op) {
  auto tile = dyn_cast<TileType>(op.getType());
  if (!tile)
    return op.emitOpError("expected !tr.tile result");
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  auto memTy = MemRefType::get(tile.getShape(), tile.getElementType());
  Value dest = createAlloca(rewriter, loc, memTy);
  Value scalar =
      materializeScalar(rewriter, loc, op.getValue(), tile.getElementType());
  if (!scalar)
    return op.emitOpError("unsupported constant element type");
  fillMemRef(rewriter, loc, dest, scalar);
  rewriter.replaceOp(op, dest);
  return success();
}

static LogicalResult lowerLoad(RewriterBase &rewriter, LoadOp op) {
  auto tile = dyn_cast<TileType>(op.getType());
  if (!tile)
    return op.emitOpError("expected !tr.tile result");
  rewriter.setInsertionPoint(op);
  FailureOr<Value> view = createTileSubview(
      rewriter, op.getLoc(), op.getBuffer(), op.getIndices(), tile.getShape());
  if (failed(view))
    return op.emitOpError("load buffer must be a memref");
  rewriter.replaceOp(op, *view);
  return success();
}

static LogicalResult lowerStore(RewriterBase &rewriter, StoreOp op) {
  auto tile = dyn_cast<TileType>(op.getValue().getType());
  ArrayRef<int64_t> shape;
  if (tile) {
    shape = tile.getShape();
  } else if (auto memref = dyn_cast<MemRefType>(op.getValue().getType())) {
    shape = memref.getShape();
  } else {
    return op.emitOpError("store value must be a tile or memref");
  }
  rewriter.setInsertionPoint(op);
  FailureOr<Value> view = createTileSubview(
      rewriter, op.getLoc(), op.getBuffer(), op.getIndices(), shape);
  if (failed(view))
    return op.emitOpError("store buffer must be a memref");
  memref::CopyOp::create(rewriter, op.getLoc(), op.getValue(), *view);
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult lowerReduce(RewriterBase &rewriter, ReduceSumOp op) {
  if (isInnerOfFusedFullReduce(op))
    return success();

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  Value input = op.getInput();
  SmallVector<int64_t> reducedDims;
  ReduceSumOp innerToErase;
  if (auto inner = input.getDefiningOp<ReduceSumOp>()) {
    // Fuse 2D -> 1D -> 0D into one generic with both reduction iterators.
    input = inner.getInput();
    int64_t rank = shapedRank(input.getType());
    for (int64_t i = 0; i < rank; ++i)
      reducedDims.push_back(i);
    innerToErase = inner;
  } else {
    reducedDims.push_back(op.getAxis());
  }

  if (!isa<MemRefType>(input.getType()))
    return op.emitOpError("reduce input must already be a memref");

  MemRefType destTy = memrefForTileLike(op.getType());
  Value dest = createAlloca(rewriter, loc, destTy);
  Value zero = createZeroScalar(rewriter, loc, destTy.getElementType());
  if (!zero)
    return op.emitOpError("unsupported reduce element type");
  fillMemRef(rewriter, loc, dest, zero);
  createReductionGeneric(rewriter, loc, input, dest, reducedDims);
  rewriter.replaceOp(op, dest);
  if (innerToErase)
    rewriter.eraseOp(innerToErase);
  return success();
}

static LogicalResult lowerAdd(RewriterBase &rewriter, AddOp op) {
  if (!isa<MemRefType>(op.getLhs().getType()) ||
      !isa<MemRefType>(op.getRhs().getType()))
    return op.emitOpError("add operands must already be memrefs");
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  MemRefType destTy = memrefForTileLike(op.getType());
  Value dest = createAlloca(rewriter, loc, destTy);
  createAddGeneric(rewriter, loc, op.getLhs(), op.getRhs(), dest);
  rewriter.replaceOp(op, dest);
  return success();
}

static LogicalResult lowerDim(RewriterBase &rewriter, DimOp op) {
  if (!isa<MemRefType>(op.getBuffer().getType()))
    return op.emitOpError("dim buffer must already be a memref");
  rewriter.setInsertionPoint(op);
  Value idx =
      arith::ConstantIndexOp::create(rewriter, op.getLoc(), op.getAxis());
  rewriter.replaceOpWithNewOp<memref::DimOp>(op, op.getBuffer(), idx);
  return success();
}

struct ConvertTRToLinalg : impl::ConvertTRToLinalgBase<ConvertTRToLinalg> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());
    convertSignature(func);

    WalkResult walk = func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forOp = dyn_cast<ForOp>(op)) {
        convertForTypes(forOp);
        return WalkResult::advance();
      }
      LogicalResult status = success();
      if (auto dim = dyn_cast<DimOp>(op))
        status = lowerDim(rewriter, dim);
      else if (auto cst = dyn_cast<ConstantOp>(op))
        status = lowerConstant(rewriter, cst);
      else if (auto load = dyn_cast<LoadOp>(op))
        status = lowerLoad(rewriter, load);
      else if (auto red = dyn_cast<ReduceSumOp>(op))
        status = lowerReduce(rewriter, red);
      else if (auto add = dyn_cast<AddOp>(op))
        status = lowerAdd(rewriter, add);
      else if (auto store = dyn_cast<StoreOp>(op))
        status = lowerStore(rewriter, store);
      if (failed(status))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walk.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tr
