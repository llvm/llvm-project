//===- SparseTensorRewriting.cpp - Sparse tensor rewriting rules ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewriting rules that are specific to sparse tensors.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;
using namespace mlir::sparse_tensor;

//===---------------------------------------------------------------------===//
// Helper methods for the actual rewriting rules.
//===---------------------------------------------------------------------===//

// Helper method to match any typed zero.
static bool isZeroValue(Value val) {
  return matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat());
}

// Helper to detect a sparse tensor type operand.
static bool isSparseTensor(OpOperand *op) {
  if (auto enc = getSparseTensorEncoding(op->get().getType())) {
    if (llvm::is_contained(enc.getDimLevelType(), DimLevelType::Compressed))
      return true;
  }
  return false;
}

// Helper method to find zero/uninitialized allocation.
static bool isAlloc(OpOperand *op, bool isZero) {
  Value val = op->get();
  if (auto alloc = val.getDefiningOp<AllocTensorOp>()) {
    Value copy = alloc.getCopy();
    if (isZero)
      return copy && isZeroValue(copy);
    return !copy;
  }
  return false;
}

// Helper to detect sampling operation.
static bool isSampling(GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::MulFOp>(def) || isa<arith::MulIOp>(def)) {
      // Both scalar input arguments used exactly once.
      Value s1 = op.getBlock()->getArgument(0);
      Value s2 = op.getBlock()->getArgument(1);
      return (def->getOperand(0) == s1 && def->getOperand(1) == s2) ||
             (def->getOperand(1) == s1 && def->getOperand(0) == s2);
    }
  }
  return false;
}

// Helper to detect chain of multiplications that do not involve x.
static bool isMulChain(Value val, Value x) {
  if (auto arg = val.dyn_cast<BlockArgument>())
    return arg != x;
  if (auto *def = val.getDefiningOp()) {
    if (isa<arith::MulFOp>(def) || isa<arith::MulIOp>(def))
      return isMulChain(def->getOperand(0), x) &&
             isMulChain(def->getOperand(1), x);
  }
  return false;
}

// Helper to detect x = x + <multiplications>.
static bool isSumOfMul(GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::AddFOp>(def) || isa<arith::AddIOp>(def)) {
      Value x = op.getBlock()->getArguments().back();
      return (def->getOperand(0) == x && isMulChain(def->getOperand(1), x)) ||
             (def->getOperand(1) == x && isMulChain(def->getOperand(0), x));
    }
  }
  return false;
}

// Helper to detect direct yield of a zero value.
static bool isZeroYield(GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
  if (auto arg = yieldOp.getOperand(0).dyn_cast<BlockArgument>()) {
    if (arg.getOwner()->getParentOp() == op) {
      return isZeroValue(op->getOperand(arg.getArgNumber()));
    }
  }
  return isZeroValue(yieldOp.getOperand(0));
}

/// Populates given sizes array from type (for static sizes) and from
/// the tensor (for dynamic sizes).
static void sizesForTensor(OpBuilder &builder, SmallVector<Value, 4> &sizes,
                           Location loc, ShapedType stp, Value tensor) {
  for (const auto &d : enumerate(stp.getShape())) {
    Value dim;
    if (d.value() == ShapedType::kDynamicSize)
      dim = builder.create<tensor::DimOp>(loc, tensor, d.index());
    else
      dim = constantIndex(builder, loc, d.value());
    sizes.push_back(dim);
  }
}

// TODO: The dim level property of the COO type relies on input tensors, the
// shape relies on the output tensor
// Helpers to setup a COO type.
static RankedTensorType getUnorderedCOOFromType(RankedTensorType src) {
  auto *ctx = src.getContext();
  auto rank = src.getRank();
  SmallVector<DimLevelType, 4> dims;

  // An unordered and non-unique compressed dim at beginning.
  dims.push_back(DimLevelType::CompressedNuNo);

  if (rank > 1) {
    // TODO: it is actually ordered at the level for ordered input.
    // Followed by unordered non-unique n-2 singleton levels.
    std::fill_n(std::back_inserter(dims), rank - 2,
                DimLevelType::SingletonNuNo);
    // TODO: only if all the inputs (for concatentate) are unique at the last
    // level should the COO has a unique level at the end. Ends by a unordered
    // unique singleton level unless the tensor rank is 1.
    dims.push_back(DimLevelType::SingletonNo);
  }
  SparseTensorEncodingAttr encSrc = getSparseTensorEncoding(src);
  // TODO: Maybe pick the bitwidth based on input/output tensors (probably the
  // largest one among them) in the original operation instead of using the
  // default value.
  auto enc = SparseTensorEncodingAttr::get(
      ctx, dims, AffineMap::getMultiDimIdentityMap(rank, ctx), AffineMap(),
      encSrc.getPointerBitWidth(), encSrc.getIndexBitWidth());
  return RankedTensorType::get(src.getShape(), src.getElementType(), enc);
}

//===---------------------------------------------------------------------===//
// The actual sparse tensor rewriting rules.
//===---------------------------------------------------------------------===//

namespace {

/// Rewriting rule that converts direct yield of zero with initial allocation.
struct FoldInvariantYield : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasTensorSemantics() || op.getNumResults() != 1 ||
        !isAlloc(op.getDpsInitOperand(0), /*isZero=*/false) || !isZeroYield(op))
      return failure();
    auto outputType = op.getResult(0).getType().cast<RankedTensorType>();
    // Yielding zero on newly allocated (all-zero) sparse tensors can be
    // optimized out directly (regardless of dynamic or static size).
    if (getSparseTensorEncoding(outputType)) {
      rewriter.replaceOp(op, op.getDpsInitOperand(0)->get());
      return success();
    }
    // Incorporate zero value into allocation copy.
    if (!outputType.hasStaticShape())
      return failure();
    Value zero = constantZero(rewriter, op.getLoc(), op.getResult(0).getType());
    AllocTensorOp a =
        op.getDpsInitOperand(0)->get().getDefiningOp<AllocTensorOp>();
    rewriter.updateRootInPlace(a, [&]() { a.getCopyMutable().assign(zero); });
    rewriter.replaceOp(op, op.getDpsInitOperand(0)->get());
    return success();
  }
};

/// Rewriting rule that converts two kernels:
///
///      T(i,j) = SUM(k, A(i,j,k) * B(i,j,k) * ... )
///      X(i,j) = S(i,j) * T(i,j)
///
/// into a single kernel, using distributive law:
///
///      X(i,j) = SUM(k, S(i,j) * A(i,j,k) * B(i,j,k) * ... )
///
/// This kind of fusion (merging two ops into one but using arithmetic
/// equalities that may not hold for floating-point computations) would
/// be undesirable in the dense case, since we distribute the multiplication
/// into the reduction loop. However, for sparse sampling tensor S, such
/// a fusion may actually reduce the asymptotic complexity of the kernel,
/// since intermediate results may be nullified.
struct FuseSparseMultiplyOverAdd : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check consumer.
    if (!op.hasTensorSemantics() || op.getNumDpsInputs() != 2 ||
        op.getNumResults() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops() ||
        !op.getMatchingIndexingMap(op.getDpsInitOperand(0)).isIdentity() ||
        !op.getMatchingIndexingMap(op.getDpsInputOperand(0)).isIdentity() ||
        !op.getMatchingIndexingMap(op.getDpsInputOperand(1)).isIdentity())
      return failure();
    // Find consuming OP2(sparse, other) or OP2(other, sparse). The other
    // operand can be sparse or dense, since the point of this rewriting rule
    // is detecting a situation in which *more* sparsity is introduced into
    // a computation, be it already sparse or still dense.
    unsigned other = 0;
    if (isSparseTensor(op.getDpsInputOperand(0)))
      other = 1;
    else if (!isSparseTensor(op.getDpsInputOperand(1)))
      return failure();
    // Check producer.
    auto prod = dyn_cast_or_null<GenericOp>(
        op.getDpsInputOperand(other)->get().getDefiningOp());
    if (!prod || !prod.hasTensorSemantics() || prod.getNumResults() != 1 ||
        !prod.getResult(0).hasOneUse())
      return failure();
    // Sampling consumer and sum of multiplication chain producer.
    if (!isAlloc(op.getDpsInitOperand(0), /*isZero=*/false) ||
        !isAlloc(prod.getDpsInitOperand(0), /*isZero=*/true) ||
        !isSampling(op) || !isSumOfMul(prod))
      return failure();
    // Modify operand structure of producer and consumer.
    Location loc = prod.getLoc();
    SmallVector<Value> inputOps = prod.getInputs();
    SmallVector<Value> outputOps = op.getOutputs();
    SmallVector<AffineMap> fusedIndexMaps = prod.getIndexingMapsArray();
    inputOps.push_back(op.getDpsInputOperand(1 - other)->get());
    fusedIndexMaps.push_back(fusedIndexMaps.back()); // mimic other
    // Fuse producer and consumer into a new generic op.
    auto fusedOp = rewriter.create<GenericOp>(
        loc, op.getResult(0).getType(), inputOps, outputOps,
        rewriter.getAffineMapArrayAttr(fusedIndexMaps), prod.getIteratorTypes(),
        /*doc=*/nullptr, /*library_call=*/nullptr);
    Block &prodBlock = prod.getRegion().front();
    Block &consBlock = op.getRegion().front();
    BlockAndValueMapping mapper;
    Block *fusedBlock = new Block();
    fusedOp.getRegion().push_back(fusedBlock);
    unsigned num = prodBlock.getNumArguments();
    for (unsigned i = 0; i < num - 1; i++)
      addArg(mapper, fusedBlock, prodBlock.getArgument(i));
    addArg(mapper, fusedBlock, consBlock.getArgument(1 - other));
    addArg(mapper, fusedBlock, prodBlock.getArgument(num - 1));
    // Clone bodies of the producer and consumer in new evaluation order.
    auto *acc = prodBlock.getTerminator()->getOperand(0).getDefiningOp();
    auto *sampler = consBlock.getTerminator()->getOperand(0).getDefiningOp();
    rewriter.setInsertionPointToStart(fusedBlock);
    Value last;
    for (auto &op : prodBlock.without_terminator())
      if (&op != acc) {
        last = op.getResult(0);
        rewriter.clone(op, mapper);
      }
    mapper.map(consBlock.getArgument(other), fusedBlock->back().getResult(0));
    mapper.map(last, rewriter.clone(*sampler, mapper)->getResult(0));
    last = rewriter.clone(*acc, mapper)->getResult(0);
    rewriter.create<linalg::YieldOp>(loc, last);
    // Force initial value on merged allocation for dense outputs.
    if (!getSparseTensorEncoding(op.getResult(0).getType())) {
      Value init = prod.getDpsInitOperand(0)
                       ->get()
                       .getDefiningOp<AllocTensorOp>()
                       .getCopy();
      AllocTensorOp a =
          op.getDpsInitOperand(0)->get().getDefiningOp<AllocTensorOp>();
      rewriter.updateRootInPlace(a, [&]() { a.getCopyMutable().assign(init); });
    }
    // Replace consumer with fused operation. Old producer
    // and consumer ops will be removed by DCE.
    rewriter.replaceOp(op, fusedOp->getResults());
    return success();
  }

private:
  // Helper to add argument and record the mapping.
  static void addArg(BlockAndValueMapping &mapper, Block *b, BlockArgument a) {
    mapper.map(a, b->addArgument(a.getType(), a.getLoc()));
  }
};

/// Sparse rewriting rule for sparse-to-sparse reshape operator.
template <typename ReshapeOp>
struct Sparse2SparseReshapeRewriter : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value srcTensor = op.getSrc();
    auto srcTp = srcTensor.getType().template cast<RankedTensorType>();
    auto dstTp = op.getResult().getType().template cast<RankedTensorType>();
    SparseTensorEncodingAttr encSrc = getSparseTensorEncoding(srcTp);
    SparseTensorEncodingAttr encDst = getSparseTensorEncoding(dstTp);
    if (!encDst || !encSrc) {
      return failure();
    }

    // Generate code to represent the static dimension constants or compute
    // the dynamic dimension values.
    SmallVector<Value, 4> srcSizes;
    sizesForTensor(rewriter, srcSizes, loc, srcTp, srcTensor);
    SmallVector<Value, 4> dstSizes;
    SmallVector<Value, 4> dstDynSizes;
    if (dstTp.hasStaticShape()) {
      for (auto d : dstTp.getShape())
        dstSizes.push_back(constantIndex(rewriter, loc, d));
    } else {
      ArrayRef<int64_t> dstShape = dstTp.getShape();
      genReshapeDstShape(loc, rewriter, dstSizes, srcSizes, dstShape,
                         op.getReassociationIndices());
      for (auto &d : llvm::enumerate(dstShape)) {
        if (d.value() == ShapedType::kDynamicSize)
          dstDynSizes.push_back(dstSizes[d.index()]);
      }
    }

    // Implement the sparse2sparse reshape as follows:
    //   %tmp = bufferization.alloc_tensor : unordered COO
    //   foreach srcCoords %srcTensor
    //     insert translateIndicesArray(srcCoords), %tmp
    //   %t = sparse_tensor.cast %tmp
    RankedTensorType cooTp = getUnorderedCOOFromType(dstTp);
    auto cooBuffer =
        rewriter.create<AllocTensorOp>(loc, cooTp, dstDynSizes).getResult();
    rewriter.create<ForeachOp>(
        loc, srcTensor, [&](OpBuilder &builder, Location loc, ValueRange args) {
          SmallVector<Value, 4> srcIndices;
          SmallVector<Value, 4> dstIndices;
          for (int64_t i = 0, e = srcTp.getRank(); i < e; i++) {
            uint64_t dim = toStoredDim(encSrc, i);
            srcIndices.push_back(args[dim]);
          }
          translateIndicesArray(builder, loc, op.getReassociationIndices(),
                                srcIndices, srcSizes, dstSizes, dstIndices);
          builder.create<InsertOp>(loc, args.back(), cooBuffer, dstIndices);
          builder.create<sparse_tensor::YieldOp>(loc);
        });

    rewriter.replaceOpWithNewOp<ConvertOp>(op, dstTp, cooBuffer);
    return success();
  }
};

/// Sparse rewriting rule for sparse-to-dense and dense-to-sparse reshape
/// operator.
template <typename ReshapeOp>
struct ReshapeRewriter : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto encDst = getSparseTensorEncoding(op.getResult().getType());
    auto encSrc = getSparseTensorEncoding(op.getSrc().getType());
    // Since a pure dense expansion is very cheap (change of view), for
    // a sparse2dense or dense2sparse, we can simply unfuse a sparse
    // conversion from the reshape operation itself.
    // All other cases are handled elsewhere.
    if (encDst && encSrc) {
      return failure();
    }
    if (encSrc) {
      RankedTensorType rtp =
          op.getSrc().getType().template cast<RankedTensorType>();
      auto denseTp =
          RankedTensorType::get(rtp.getShape(), rtp.getElementType());
      auto convert = rewriter.create<ConvertOp>(loc, denseTp, op.getSrc());
      op->setOperand(0, convert);
      return success();
    }
    if (encDst) {
      RankedTensorType rtp =
          op.getResult().getType().template cast<RankedTensorType>();
      auto denseTp =
          RankedTensorType::get(rtp.getShape(), rtp.getElementType());
      auto reshape = rewriter.create<ReshapeOp>(loc, denseTp, op.getSrc(),
                                                op.getReassociation());
      Value convert = rewriter.create<ConvertOp>(loc, rtp, reshape);
      rewriter.replaceOp(op, convert);
      return success();
    }
    return failure();
  }
};

struct ConcatenateRewriter : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rtp = op.getType().cast<RankedTensorType>();
    // TODO: Build the output shape if needed.
    assert(rtp.hasStaticShape());
    auto rank = rtp.getRank();
    size_t conDim = op.getDimension().getZExtValue();
    // %t = concatenate %s1, %s2, %s3 {dim = 1}
    // ==>
    // %tmp = bufferization.alloc_tensor : unordered COO
    // foreach in %s1 : insert d0, d1, %tmp
    // foreach in %s2 : insert d0, d1 + size(s1), %tmp
    // foreach in %s3 : insert d0, d1 + size(s1) + size(s2), %tmp
    // %t = sparse_tensor.cast %tmp
    auto cooTp = getUnorderedCOOFromType(rtp);
    auto cooBuffer =
        rewriter.create<AllocTensorOp>(loc, cooTp, ValueRange()).getResult();

    Value offset = constantIndex(rewriter, loc, 0);
    for (Value input : op.getInputs()) {
      // Builds the indexing map.

      // Build a for op for each input tensor to append new values into the
      // output tensor.
      rewriter.create<ForeachOp>(
          loc, input, [&](OpBuilder &builder, Location loc, ValueRange args) {
            SmallVector<Value, 4> indices;
            for (int64_t i = 0; i < rank; i++) {
              uint64_t dim =
                  toStoredDim(getSparseTensorEncoding(input.getType()), i);
              Value idx = args[dim];
              if (i == static_cast<int64_t>(conDim))
                // transform coordinates on matching dim
                idx = builder.create<arith::AddIOp>(loc, idx, offset);
              indices.push_back(idx);
            }
            builder.create<InsertOp>(loc, args.back(), cooBuffer, indices);
            builder.create<sparse_tensor::YieldOp>(loc);
          });
      // Accumulates the offset. Note that only static-shaped inputs are allowed
      // by concatenate op verifier, which saves us from computing the offset
      // dynamically.
      auto d = input.getType().cast<RankedTensorType>().getShape()[conDim];
      assert(!ShapedType::isDynamic(d));
      offset = rewriter.create<arith::AddIOp>(loc, offset,
                                              constantIndex(rewriter, loc, d));
    }
    rewriter.replaceOpWithNewOp<ConvertOp>(op, rtp, cooBuffer);
    return success();
  }
};

/// Sparse rewriting rule for the foreach operator.
struct ForeachRewriter : public OpRewritePattern<ForeachOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ForeachOp op,
                                PatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value input = op.getTensor();
    auto rtp = input.getType().cast<RankedTensorType>();
    int64_t rank = rtp.getRank();
    auto enc = getSparseTensorEncoding(rtp);

    // 1. Generates loop for the sparse input.
    SparseTensorLoopEmitter loopEmitter(ValueRange{input});
    loopEmitter.initializeLoopEmit(rewriter, loc);
    for (int64_t i = 0; i < rank; i++) {
      // TODO: provide utility function for loop sequences that only contains
      // one for loop?
      loopEmitter.enterNewLoopSeq(rewriter, loc, 0, static_cast<size_t>(i));
      loopEmitter.enterLoopOverTensorAtDim(rewriter, loc, 0, i);
    }

    SmallVector<Value, 4> coords;
    coords.reserve(rank);
    loopEmitter.getCoordinateArray(coords);

    Value vals = loopEmitter.getValBuffer()[0];
    Value pidx = loopEmitter.getPidxs()[0].back();
    // Loads the value from sparse tensor using pointer index;
    // loads the value from dense tensor using coordinate array.
    Value val = enc ? rewriter.create<memref::LoadOp>(loc, vals, pidx)
                    : rewriter.create<memref::LoadOp>(loc, vals, coords);

    // 2. Inline the block in the foreach operator.
    Block::iterator inlinePos = rewriter.getInsertionPoint();
    Block *srcBlock = op.getBody();
    // Remove sparse_tensor.yield.
    rewriter.eraseOp(srcBlock->getTerminator());

    for (int64_t i = 0; i < rank; i++) {
      loopEmitter.exitCurrentLoop(rewriter, loc);
      loopEmitter.exitCurrentLoopSeq();
    }

    SmallVector<Value, 4> args;
    // Remap coordinates.
    for (int64_t i = 0; i < rank; i++) {
      Value actual = coords[toOrigDim(enc, i)];
      args.push_back(actual);
    }
    // Remap value.
    args.push_back(val);

    // Inline body.
    rewriter.mergeBlockBefore(srcBlock, &*inlinePos, args);
    // delete the foreach operator.
    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse rewriting rule for the new operator.
struct NewRewriter : public OpRewritePattern<NewOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NewOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto dstTp = op.getResult().getType().template cast<RankedTensorType>();
    SparseTensorEncodingAttr encDst = getSparseTensorEncoding(dstTp);
    if (!encDst) {
      return failure();
    }

    // Create a sparse tensor reader.
    Value fileName = op.getSource();
    Type opaqueTp = getOpaquePointerType(rewriter);
    Value reader = createFuncCall(rewriter, loc, "createSparseTensorReader",
                                  {opaqueTp}, {fileName}, EmitCInterface::Off)
                       .getResult(0);

    // Allocate a temporary buffer for storing dimension sizes and indices.
    Type indexTp = rewriter.getIndexType();
    uint64_t rank = dstTp.getRank();
    Value dimSizes = genAlloca(rewriter, loc, rank, indexTp);

    // If the result tensor has dynamic dimensions, get the dynamic sizes from
    // the sparse tensor reader.
    SmallVector<Value, 4> dynSizesArray;
    if (!dstTp.hasStaticShape()) {
      createFuncCall(rewriter, loc, "getSparseTensorReaderDimSizes", {},
                     {reader, dimSizes}, EmitCInterface::On)
          .getResult(0);
      ArrayRef<int64_t> dstShape = dstTp.getShape();
      for (auto &d : llvm::enumerate(dstShape)) {
        if (d.value() == ShapedType::kDynamicSize) {
          dynSizesArray.push_back(rewriter.create<memref::LoadOp>(
              loc, dimSizes, constantIndex(rewriter, loc, d.index())));
        }
      }
    }

    // Implement the NewOp as follows:
    //   %tmp = bufferization.alloc_tensor : an unordered COO with identity
    //                                       storage ordering
    //   for i = 0 to nnz
    //     get the next element from the input file
    //     insert the element to %tmp
    //   %t = sparse_tensor.ConvertOp %tmp
    RankedTensorType cooTp = getUnorderedCOOFromType(dstTp);
    auto cooBuffer =
        rewriter.create<AllocTensorOp>(loc, cooTp, dynSizesArray).getResult();

    Value c0 = constantIndex(rewriter, loc, 0);
    Value c1 = constantIndex(rewriter, loc, 1);
    Value nnz = createFuncCall(rewriter, loc, "getSparseTensorReaderNNZ",
                               {indexTp}, {reader}, EmitCInterface::Off)
                    .getResult(0);
    Type eltTp = dstTp.getElementType();
    Value value = genAllocaScalar(rewriter, loc, eltTp);
    scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, c0, nnz, c1);
    rewriter.setInsertionPointToStart(forOp.getBody());

    SmallString<18> getNextFuncName{"getSparseTensorReaderNext",
                                    primaryTypeFunctionSuffix(eltTp)};
    Value indices = dimSizes; // Reuse the indices memref to store indices.
    createFuncCall(rewriter, loc, getNextFuncName, {eltTp},
                   {reader, indices, value}, EmitCInterface::On)
        .getResult(0);
    SmallVector<Value, 4> indicesArray;
    for (uint64_t i = 0; i < rank; i++) {
      indicesArray.push_back(rewriter.create<memref::LoadOp>(
          loc, indices, constantIndex(rewriter, loc, i)));
    }
    Value v = rewriter.create<memref::LoadOp>(loc, value);
    rewriter.create<InsertOp>(loc, v, cooBuffer, indicesArray);
    rewriter.setInsertionPointAfter(forOp);

    // Release the sparse tensor reader.
    createFuncCall(rewriter, loc, "delSparseTensorReader", {}, {reader},
                   EmitCInterface::Off);

    Value newOp = rewriter.replaceOpWithNewOp<ConvertOp>(op, dstTp, cooBuffer);

    // Release the unordered COO tensor buffer.
    rewriter.setInsertionPointAfterValue(newOp);
    rewriter.create<DeallocTensorOp>(loc, cooBuffer);

    return success();
  }
};

struct OutRewriter : public OpRewritePattern<OutOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // Calculate NNZ.
    Value src = op.getTensor();
    Value nnz = rewriter.create<NumberOfEntriesOp>(loc, src);

    // Allocate a temporary buffer for storing dimension sizes and indices.
    auto srcTp = src.getType().template cast<RankedTensorType>();
    uint64_t rank = srcTp.getRank();
    Type indexTp = rewriter.getIndexType();
    Value dimSizes = genAlloca(rewriter, loc, rank, indexTp);

    // Generate code to calculate dimension size values and store the values to
    // the buffer.
    SmallVector<Value, 4> dims;
    sizesForTensor(rewriter, dims, loc, srcTp, src);
    for (uint64_t i = 0; i < rank; i++) {
      rewriter.create<memref::StoreOp>(loc, dims[i], dimSizes,
                                       constantIndex(rewriter, loc, i));
    }

    // Create a sparse tensor writer and output meta data.
    Type opaqueTp = getOpaquePointerType(rewriter);
    Value writer =
        createFuncCall(rewriter, loc, "createSparseTensorWriter", {opaqueTp},
                       {op.getDest()}, EmitCInterface::Off)
            .getResult(0);
    Value rankValue = constantIndex(rewriter, loc, rank);
    createFuncCall(rewriter, loc, "outSparseTensorWriterMetaData", {},
                   {writer, rankValue, nnz, dimSizes}, EmitCInterface::On);

    Value indices = dimSizes; // Reuse the dimSizes buffer for indices.
    Type eltTp = srcTp.getElementType();
    SmallString<18> outNextFuncName{"outSparseTensorWriterNext",
                                    primaryTypeFunctionSuffix(eltTp)};
    Value value = genAllocaScalar(rewriter, loc, eltTp);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // For each element in the source tensor, output the element.
    rewriter.create<ForeachOp>(
        loc, src, [&](OpBuilder &builder, Location loc, ValueRange args) {
          for (uint64_t i = 0; i < rank; i++) {
            rewriter.create<memref::StoreOp>(loc, args[i], indices,
                                             constantIndex(builder, loc, i));
          }
          rewriter.create<memref::StoreOp>(loc, args.back(), value);
          SmallVector<Value, 4> operands{writer, rankValue, indices, value};
          FlatSymbolRefAttr fn = getFunc(module, outNextFuncName, {}, operands,
                                         EmitCInterface::On);
          builder.create<func::CallOp>(loc, TypeRange(), fn, operands);
          builder.create<sparse_tensor::YieldOp>(loc);
        });

    // Release the writer.
    createFuncCall(rewriter, loc, "delSparseTensorWriter", {}, {writer},
                   EmitCInterface::Off);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===---------------------------------------------------------------------===//
// Methods that add patterns described in this file to a pattern list.
//===---------------------------------------------------------------------===//
void mlir::populateSparseTensorRewriting(RewritePatternSet &patterns,
                                         bool enableRT, bool enableForeach,
                                         bool /*enableConvert*/) {
  patterns.add<FoldInvariantYield, FuseSparseMultiplyOverAdd,
               ReshapeRewriter<tensor::ExpandShapeOp>,
               ReshapeRewriter<tensor::CollapseShapeOp>>(patterns.getContext());
  if (enableForeach)
    patterns.add<ForeachRewriter>(patterns.getContext());

  // TODO: If RT not enabled, rewrite concatenate ops, etc here.
  if (!enableRT)
    patterns.add<ConcatenateRewriter, NewRewriter, OutRewriter,
                 Sparse2SparseReshapeRewriter<tensor::ExpandShapeOp>,
                 Sparse2SparseReshapeRewriter<tensor::CollapseShapeOp>>(
        patterns.getContext());
}
