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
#include "LoopEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
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
  auto enc = getSparseTensorEncoding(op->get().getType());
  return enc &&
         llvm::is_contained(enc.getDimLevelType(), DimLevelType::Compressed);
}

// Helper method to find zero/uninitialized allocation.
static bool isAlloc(OpOperand *op, bool isZero) {
  Value val = op->get();
  // Check allocation, with zero alloc when required.
  if (auto alloc = val.getDefiningOp<AllocTensorOp>()) {
    Value copy = alloc.getCopy();
    if (isZero)
      return copy && isZeroValue(copy);
    return !copy;
  }
  // Last resort for zero alloc: the whole value is zero.
  return isZero && isZeroValue(val);
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
static void sizesForTensor(OpBuilder &builder, SmallVectorImpl<Value> &sizes,
                           Location loc, ShapedType stp, Value tensor) {
  for (const auto &d : enumerate(stp.getShape())) {
    Value dim;
    if (d.value() == ShapedType::kDynamic)
      dim = builder.create<tensor::DimOp>(loc, tensor, d.index());
    else
      dim = constantIndex(builder, loc, d.value());
    sizes.push_back(dim);
  }
}

// TODO: The dim level property of the COO type relies on input tensors, the
// shape relies on the output tensor
static RankedTensorType
getUnorderedCOOFromTypeWithOrdering(RankedTensorType src, AffineMap ordering) {
  return getCOOFromTypeWithOrdering(src, ordering, false);
}

static RankedTensorType getUnorderedCOOFromType(RankedTensorType src) {
  return getCOOFromType(src, false);
}

/// Collects the dynamic dimension sizes for `tp` with the assumption that
/// `sizes` are the dimension sizes for the type. Stores the dynamic dimension
/// sizes to dynSizes.
static void getDynamicSizes(RankedTensorType tp,
                            const SmallVectorImpl<Value> &sizes,
                            SmallVectorImpl<Value> &dynSizes) {
  for (const auto &d : enumerate(tp.getShape())) {
    if (d.value() == ShapedType::kDynamic)
      dynSizes.push_back(sizes[d.index()]);
  }
}

static LogicalResult genForeachOnSparseConstant(ForeachOp op,
                                                RewriterBase &rewriter,
                                                SparseElementsAttr attr) {
  auto loc = op.getLoc();
  SmallVector<Value> reduc = op.getInitArgs();

  // Foreach on constant.
  foreachInSparseConstant(
      loc, rewriter, attr, op.getOrder().value_or(AffineMap()),
      [&reduc, &rewriter, op](ArrayRef<Value> cvs, Value v) mutable {
        SmallVector<Value> args;
        args.append(cvs.begin(), cvs.end());
        args.push_back(v);
        args.append(reduc);
        // Clones the foreach op to get a copy of the loop body.
        auto cloned = cast<ForeachOp>(rewriter.clone(*op.getOperation()));
        assert(args.size() == cloned.getBody()->getNumArguments());
        Operation *yield = cloned.getBody()->getTerminator();
        rewriter.inlineBlockBefore(cloned.getBody(), op, args);
        // clean up
        rewriter.eraseOp(cloned);
        reduc = yield->getOperands();
        rewriter.eraseOp(yield);
      });

  rewriter.replaceOp(op, reduc);
  return success();
}

/// Populates the given sizes array for concatenation from types (for static
/// sizes) and from the source tensors (for dynamic sizes).
static void concatSizesFromInputs(OpBuilder &builder,
                                  SmallVectorImpl<Value> &sizes, Location loc,
                                  ShapedType dstTp, ValueRange srcs,
                                  unsigned dim) {
  auto dstShape = dstTp.getShape();
  sizesFromSrc(builder, sizes, loc, srcs[0]);

  // Sum up on the `dim` if the dimension is dynamic.
  if (dstShape[dim] != ShapedType::kDynamic) {
    // Faithfully take the static size.
    sizes[dim] = constantIndex(builder, loc, dstShape[dim]);
  } else {
    // Else, compute the shape dynamically.
    for (const auto &src : srcs.drop_front()) {
      Value srcSz = linalg::createOrFoldDimOp(builder, loc, src, dim);
      // Sum up all the sizes.
      sizes[dim] = builder.create<arith::AddIOp>(loc, sizes[dim], srcSz);
    }
  }
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
        !isAlloc(op.getDpsInitOperand(0), /*isZero=*/false) ||
        !isZeroYield(op) || !op.getDpsInitOperand(0)->get().hasOneUse())
      return failure();
    auto outputType = getRankedTensorType(op.getResult(0));
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
    IRMapping mapper;
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
  static void addArg(IRMapping &mapper, Block *b, BlockArgument a) {
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
    auto srcTp = getRankedTensorType(srcTensor);
    auto dstTp = getRankedTensorType(op.getResult());
    SparseTensorEncodingAttr encSrc = getSparseTensorEncoding(srcTp);
    SparseTensorEncodingAttr encDst = getSparseTensorEncoding(dstTp);
    if (!encDst || !encSrc) {
      return failure();
    }

    // Generate code to represent the static dimension constants or compute
    // the dynamic dimension values.
    SmallVector<Value> srcSizes;
    sizesForTensor(rewriter, srcSizes, loc, srcTp, srcTensor);
    SmallVector<Value> dstSizes;
    SmallVector<Value> dstDynSizes;
    if (dstTp.hasStaticShape()) {
      for (auto d : dstTp.getShape())
        dstSizes.push_back(constantIndex(rewriter, loc, d));
    } else {
      ArrayRef<int64_t> dstShape = dstTp.getShape();
      genReshapeDstShape(loc, rewriter, dstSizes, srcSizes, dstShape,
                         op.getReassociationIndices());
      for (auto [idx, shape] : llvm::enumerate(dstShape)) {
        if (shape == ShapedType::kDynamic)
          dstDynSizes.push_back(dstSizes[idx]);
      }
    }

    // Implement the sparse2sparse reshape as follows:
    //   %tmp = bufferization.alloc_tensor : unordered COO
    //   foreach srcCoords %srcTensor
    //     insert reshapeCvs(srcCoords), %tmp
    //   %t = sparse_tensor.cast %tmp
    Value nnz = rewriter.create<NumberOfEntriesOp>(loc, srcTensor);
    RankedTensorType cooTp = getUnorderedCOOFromType(dstTp);
    Value cooBuffer =
        rewriter
            .create<AllocTensorOp>(loc, cooTp, dstDynSizes, Value(),
                                   /*sizeHint=*/nnz, Attribute())
            .getResult();

    ForeachOp foreachOp = rewriter.create<ForeachOp>(
        loc, srcTensor, cooBuffer,
        [&](OpBuilder &builder, Location loc, ValueRange srcLcvs, Value v,
            ValueRange reduc) {
          const Dimension dimRank = srcTp.getRank();
          SmallVector<Value> srcDcvs;
          srcDcvs.reserve(dimRank);
          for (Dimension d = 0; d < dimRank; d++) {
            // FIXME: `toStoredDim` is deprecated
            Level lvl = toStoredDim(encSrc, d);
            srcDcvs.push_back(srcLcvs[lvl]);
          }
          SmallVector<Value> dstDcvs;
          reshapeCvs(builder, loc, op.getReassociationIndices(), srcSizes,
                     srcDcvs, dstSizes, dstDcvs);
          auto t = builder.create<InsertOp>(loc, v, reduc.front(), dstDcvs);
          builder.create<sparse_tensor::YieldOp>(loc, t);
        });
    auto t = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    auto converted = rewriter.create<ConvertOp>(loc, dstTp, t).getResult();
    rewriter.create<DeallocTensorOp>(loc, t);
    rewriter.replaceOp(op, converted);
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
      auto rtp = getRankedTensorType(op.getSrc());
      auto denseTp =
          RankedTensorType::get(rtp.getShape(), rtp.getElementType());
      auto convert = rewriter.create<ConvertOp>(loc, denseTp, op.getSrc());
      rewriter.updateRootInPlace(op, [&]() { op->setOperand(0, convert); });
      return success();
    }
    if (encDst) {
      auto rtp = getRankedTensorType(op.getResult());
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
    const Location loc = op.getLoc();
    const auto dstTp = getSparseTensorType(op);
    const Dimension dimRank = dstTp.getDimRank();
    const Dimension conDim = op.getDimension();
    SmallVector<Value> sizes;
    concatSizesFromInputs(rewriter, sizes, loc, dstTp, op.getInputs(), conDim);

    // %t = concatenate %s1, %s2, %s3 {dim = 1}
    // ==>
    // if (isSparseDst)
    //   if (allDense)
    //     %tmp = bufferization.alloc_tensor dstTp
    //   else
    //     %tmp = bufferization.alloc_tensor : unordered COO
    // else
    //   %tmp = memref.alloc : dense tensor
    // foreach in %s1 : insert d0, d1, %tmp
    // foreach in %s2 : insert d0, d1 + size(s1), %tmp
    // foreach in %s3 : insert d0, d1 + size(s1) + size(s2), %tmp
    // %t = convert_to_dest_tensor(%tmp)
    //
    // NOTE: this cannot be `const` because it will be changed when
    // `needTmpCOO`, but that's buried in the conditional below and
    // thus not easily extracted.
    auto encDst = dstTp.getEncoding();
    Value dst; // Destination tensor for inserting source tensor values.
    bool needTmpCOO = true;
    const bool allDense = dstTp.hasEncoding() && dstTp.isAllDense();
    Value annotatedDenseDst;
    if (dstTp.hasEncoding()) {
      bool allOrdered = false;
      // When concatenating on dimension 0, and all inputs are sorted and have
      // an identity dimOrdering, the concatenate will generate coords in
      // lexOrder thus no need for the tmp COO buffer.
      // TODO: When conDim != 0, as long as conDim is the first dimension
      // in all input/output buffers, and all input/output buffers have the same
      // dimOrdering, the tmp COO buffer is still unnecessary (e.g, concatenate
      // CSC matrices along column).
      if (!allDense && conDim == 0 && dstTp.isIdentity()) {
        for (auto i : op.getInputs()) {
          const auto stt = getSparseTensorType(i);
          allOrdered = stt.isAllOrdered() && stt.isIdentity();
          if (!allOrdered)
            break;
        }
      }

      needTmpCOO = !allDense && !allOrdered;
      const RankedTensorType tp = needTmpCOO ? getUnorderedCOOFromType(dstTp)
                                             : dstTp.getRankedTensorType();
      encDst = needTmpCOO ? getSparseTensorEncoding(tp) : encDst;
      SmallVector<Value> dynSizes;
      getDynamicSizes(dstTp, sizes, dynSizes);
      dst = rewriter.create<AllocTensorOp>(loc, tp, dynSizes).getResult();
      if (allDense) {
        // Create a view of the values buffer to match the unannotated dense
        // tensor.
        Value valuesBuffer = genToValues(rewriter, loc, dst);
        Value dimCoords =
            genAlloca(rewriter, loc, dimRank, rewriter.getIndexType(),
                      /*staticShape=*/true);
        annotatedDenseDst = dst;
        dst = reshapeValuesToLevels(rewriter, loc, encDst, sizes, valuesBuffer,
                                    dimCoords);
      }
    } else {
      // TODO: Dense buffers should be allocated/deallocated via the callback
      // in BufferizationOptions.
      dst = allocDenseTensor(rewriter, loc, dstTp, sizes);
    }

    Value offset = constantIndex(rewriter, loc, 0);
    SmallVector<Value> initArgs;
    if (encDst && !allDense)
      initArgs.push_back(dst);
    ForeachOp foreachOp;
    for (Value input : op.getInputs()) {
      // Build a for op for each input tensor to append new values into the
      // output tensor.
      foreachOp = rewriter.create<ForeachOp>(
          loc, input, initArgs,
          [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
              ValueRange reduc) {
            SmallVector<Value> dstLcvs(dstTp.getLvlRank());
            for (Dimension d = 0; d < dimRank; d++) {
              Value crd = dcvs[d];
              if (d == conDim)
                // Transform coordinates for the concatenating dim.
                crd = builder.create<arith::AddIOp>(loc, crd, offset);
              // FIXME: `toStoredDim` is deprecated
              dstLcvs[toStoredDim(encDst, d)] = crd;
            }
            if (encDst && !allDense) {
              Value cond = genIsNonzero(rewriter, loc, v);
              scf::IfOp ifOp = builder.create<scf::IfOp>(
                  loc, TypeRange(reduc.front().getType()), cond, /*else*/ true);
              builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
              Value t =
                  builder.create<InsertOp>(loc, v, reduc.front(), dstLcvs);
              rewriter.create<scf::YieldOp>(loc, t);
              rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
              rewriter.create<scf::YieldOp>(loc, reduc.front());
              rewriter.setInsertionPointAfter(ifOp);
              rewriter.create<sparse_tensor::YieldOp>(loc, ifOp.getResult(0));
            } else {
              builder.create<memref::StoreOp>(loc, v, dst, dstLcvs);
              builder.create<sparse_tensor::YieldOp>(loc);
            }
          });
      // Accumulates the offset. Note that only static-shaped inputs are allowed
      // by concatenate op verifier, which saves us from computing the offset
      // dynamically.
      const auto sh = getSparseTensorType(input).getStaticDimSize(conDim);
      assert(sh.has_value());
      offset = rewriter.create<arith::AddIOp>(
          loc, offset, constantIndex(rewriter, loc, *sh));
      if (encDst && !allDense) {
        dst = foreachOp.getResult(0);
        initArgs[0] = dst;
      }
    }

    // Temp variable to avoid needing to call `getRankedTensorType`
    // in the three use-sites below.
    const RankedTensorType dstRTT = dstTp;
    if (!encDst) {
      rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, dstRTT, dst);
    } else if (allDense) {
      rewriter.replaceOp(
          op, rewriter.create<ConvertOp>(loc, dstRTT, annotatedDenseDst)
                  .getResult());
    } else {
      dst = rewriter.create<LoadOp>(loc, dst, true);
      if (needTmpCOO) {
        Value tmpCoo = dst;
        dst = rewriter.create<ConvertOp>(loc, dstRTT, tmpCoo).getResult();
        rewriter.create<DeallocTensorOp>(loc, tmpCoo);
      }
      rewriter.replaceOp(op, dst);
    }
    return success();
  }
};

/// Sparse rewriting rule for the convert operator.
struct ConvertRewriter : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto encDst = getSparseTensorEncoding(op.getType());
    auto encSrc = getSparseTensorEncoding(op.getSource().getType());
    if (encDst && encSrc && !encSrc.isSlice() &&
        encSrc.withoutBitWidths() == encDst.withoutBitWidths()) {
      // Trivial tensor conversion and simple element type conversion is handled
      // in codegen.
      return failure();
    }
    // TODO: Add a cast before generating InsertOp.
    assert(op.getSource().getType().getElementType() ==
           op.getDest().getType().getElementType());
    if (encSrc && encDst)
      return sparse2SparseRewrite(op, rewriter);
    if (encSrc && !encDst)
      return sparse2DenseRewrite(op, rewriter);
    if (!encSrc && encDst)
      return dense2SparseRewrite(op, rewriter);

    // Dense-to-dense convert is a nop and handled by canonicalization.
    return failure();
  }

private:
  // Handles sparse constant to sparse tensor or dense tensor to sparse tensor
  // conversion as follows:
  //   t = new sparse COO tensor
  //   fill t using src
  //   dst = convert t
  //
  // To fill the COO tensor from a dense tensor:
  //   for i1 in dim1
  //    ..
  //     for ik in dimk
  //       val = a[i1,..,ik]
  //       if val != 0
  //         t->add(val, [i1,..,ik], [p1,..,pk])
  //
  // To fill the COO tensor from a sparse constant in COO format:
  //   for i in range(NNZ)
  //     val = values[i]
  //     [i1,..,ik] = coordinates[i]
  //     t->add(val, [i1,..,ik], [p1,..,pk])
  LogicalResult dense2SparseRewrite(ConvertOp op,
                                    PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Value src = op.getSource();
    const auto dstTp = getSparseTensorType(op);
    SmallVector<Value> sizes;
    sizesFromSrc(rewriter, sizes, loc, src);
    SmallVector<Value> dynSizes;
    getDynamicSizes(dstTp, sizes, dynSizes);

    bool fromSparseConst = false;
    if (auto constOp = op.getSource().getDefiningOp<arith::ConstantOp>()) {
      if (constOp.getValue().dyn_cast<SparseElementsAttr>()) {
        fromSparseConst = true;
      }
    }

    const auto encDst = dstTp.getEncoding();
    // We don't need a temporary COO tensor if the destination has an identity
    // ordering. Otherwise, we use the destination ordering for the temporary
    // COO tensor.
    // TODO: enhance foreachOp to take ordering to remove the need of a
    // temporary COO tensor here.
    const RankedTensorType bufferTp = dstTp.isIdentity() || fromSparseConst
                                          ? dstTp.getRankedTensorType()
                                          : getUnorderedCOOFromTypeWithOrdering(
                                                dstTp, dstTp.getDimToLvlMap());
    // Only imposes foreach order on dense constant (which will be statically
    // sorted by the sparse compiler), otherwise the rotated loop sequence
    // results to bad cache locality.
    AffineMapAttr foreachOrder = nullptr;
    if (encDst.getDimOrdering() && fromSparseConst)
      foreachOrder = AffineMapAttr::get(encDst.getDimOrdering());

    auto buffer =
        rewriter.create<AllocTensorOp>(loc, bufferTp, dynSizes).getResult();
    auto foreachOp = rewriter.create<ForeachOp>(
        loc, src, buffer, foreachOrder,
        [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
            ValueRange reduc) {
          Value input = reduc.front();
          const Dimension dimRank = dstTp.getDimRank();
          const Level lvlRank = dstTp.getLvlRank();
          SmallVector<Value> lcvs(lvlRank);
          for (Dimension d = 0; d < dimRank; d++)
            // FIXME: `toStoredDim` is deprecated
            lcvs[toStoredDim(encDst, d)] = dcvs[d];
          if (fromSparseConst) {
            input = builder.create<InsertOp>(loc, v, input, lcvs);
          } else {
            Value cond = genIsNonzero(builder, loc, v);
            auto ifOp = builder.create<scf::IfOp>(
                loc, TypeRange(input.getType()), cond, /*else*/ true);
            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            Value insert = builder.create<InsertOp>(loc, v, input, lcvs);
            builder.create<scf::YieldOp>(loc, insert);
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            builder.create<scf::YieldOp>(loc, input);
            builder.setInsertionPointAfter(ifOp);
            input = ifOp.getResult(0);
          }
          builder.create<sparse_tensor::YieldOp>(loc, input);
        });
    rewriter.setInsertionPointAfter(op);
    src = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    if (bufferTp != dstTp) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, dstTp.getRankedTensorType(),
                                             src);
      rewriter.create<DeallocTensorOp>(loc, src);
    } else {
      rewriter.replaceOp(op, src);
    }

    return success();
  }

  // Handles sparse tensor to dense tensor conversion as follows:
  //   dst = new dense tensor;
  //   foreach elemment in src
  //     dst[element.coords] = element.value
  LogicalResult sparse2DenseRewrite(ConvertOp op,
                                    PatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    RankedTensorType dstTp = getRankedTensorType(op);
    Value src = op.getSource();
    RankedTensorType srcTp = getRankedTensorType(src);

    SmallVector<Value> sizes;
    sizesForTensor(rewriter, sizes, loc, srcTp, src);

    Value dst = allocDenseTensor(rewriter, loc, dstTp, sizes);
    Block *insertionBlock = rewriter.getInsertionBlock();
    bool noEscape = bufferization::allocationDoesNotEscape(op->getOpResult(0));

    rewriter.create<ForeachOp>(loc, src, std::nullopt,
                               [&](OpBuilder &builder, Location loc,
                                   ValueRange args, Value v, ValueRange reduc) {
                                 builder.create<memref::StoreOp>(loc, v, dst,
                                                                 args);
                                 builder.create<sparse_tensor::YieldOp>(loc);
                               });

    rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, dstTp, dst);

    // Deallocate the buffer.
    if (noEscape) {
      rewriter.setInsertionPoint(insertionBlock->getTerminator());
      deallocDenseTensor(rewriter, loc, dst);
    }
    return success();
  }

  // Handles sparse tensor to sparse tensor conversion as follows:
  //   if src is not COO
  //       construct a COO to represent the src
  //   sort the src COO
  //   foreach elemment in the sorted src COO
  //     insert element to dst
  LogicalResult sparse2SparseRewrite(ConvertOp op,
                                     PatternRewriter &rewriter) const {
    const Location loc = op->getLoc();
    // These two variables cannot be `const` because they're conditionally
    // changed below.  Ideally we'd use `SparseTensorType` for `srcRTT`;
    // however that class's copy-ctor is implicitly deleted.
    Value src = op.getSource();
    auto srcRTT = getRankedTensorType(src);
    const auto dstTp = getSparseTensorType(op);
    const auto encDst = dstTp.getEncoding();
    const Level dstLvlRank = dstTp.getLvlRank();
    const Dimension dimRank = dstTp.getDimRank();
    // This assertion should be guaranteed by validity of the op,
    // but just for paranoia's sake.
    assert(static_cast<Dimension>(srcRTT.getRank()) == dimRank);

    SmallVector<Value> srcSizes;
    sizesForTensor(rewriter, srcSizes, loc, srcRTT, src);
    Value tmpCoo = Value();
    Value nnz = rewriter.create<NumberOfEntriesOp>(loc, src);
    // We need a tmp COO buffer if and only if
    // 1. the src tensor is not a COO and
    // 2. the src tensor is not ordered in the same way as the target
    // tensor (e.g., src tensor is not ordered or src tensor haves a different
    // dimOrdering).
    if (const SparseTensorType srcTp(srcRTT);
        !(srcTp.isAllOrdered() && srcTp.hasSameDimToLvlMap(dstTp))) {
      // Construct a COO tensor from the src tensor.
      // TODO: there may be cases for which more efficiently without
      // going through an intermediate COO, such as cases that only change
      // the overhead types.
      SmallVector<Value> dynSrcSizes;
      getDynamicSizes(srcRTT, srcSizes, dynSrcSizes);
      srcRTT =
          getUnorderedCOOFromTypeWithOrdering(srcRTT, dstTp.getDimToLvlMap());
      // Ensure that mutating `srcRTT` didn't invalidate `dimRank`.
      assert(static_cast<Dimension>(srcRTT.getRank()) == dimRank);
      tmpCoo = rewriter
                   .create<AllocTensorOp>(loc, srcRTT, dynSrcSizes, Value(),
                                          /*sizeHint=*/nnz, Attribute())
                   .getResult();
      auto foreachOp = rewriter.create<ForeachOp>(
          loc, src, tmpCoo,
          [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
              ValueRange reduc) {
            SmallVector<Value> dstLcvs(dstLvlRank);
            for (Dimension d = 0; d < dimRank; d++) {
              // FIXME: `toStoredDim` is deprecated
              Level l = toStoredDim(encDst, d);
              dstLcvs[l] = dcvs[d];
            }
            auto t = builder.create<InsertOp>(loc, v, reduc.front(), dstLcvs);
            builder.create<sparse_tensor::YieldOp>(loc, t);
          });
      src = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    }

    // Now that the conditional is done, we can use `SparseTensorType`.
    const SparseTensorType srcTp(srcRTT);

    // Only need to sort if the srcTp is not already sorted (we faithfully take
    // the guarantee from the sparse tensor encoding).
    if (!srcTp.isAllOrdered()) {
      // Retrieve the values-array.
      Value y = genToValues(rewriter, loc, src);
      const auto encSrc = srcTp.getEncoding();
      // Sort the COO tensor so that its elements are ordered via increasing
      // coordinates for the storage ordering of the dst tensor.  Use SortCoo
      // if the COO tensor has the same ordering as the dst tensor.
      if (dimRank > 1 && srcTp.hasSameDimToLvlMap(dstTp)) {
        MemRefType coordsTp =
            get1DMemRefType(encSrc.getCrdType(), /*withLayout=*/false);
        Value xs = rewriter.create<ToCoordinatesBufferOp>(loc, coordsTp, src);
        rewriter.create<SortCooOp>(
            loc, nnz, xs, ValueRange{y}, rewriter.getIndexAttr(dimRank),
            rewriter.getIndexAttr(0), SparseTensorSortKind::HybridQuickSort);
      } else {
        // Gather the coordinates-arrays in the dst tensor storage order.
        SmallVector<Value> xs(dstLvlRank);
        const Level srcLvlRank = srcTp.getLvlRank();
        for (Level srcLvl = 0; srcLvl < srcLvlRank; srcLvl++) {
          // FIXME: `toOrigDim` is deprecated
          Dimension dim = toOrigDim(encSrc, srcLvl);
          // FIXME: `toStoredDim` is deprecated
          Level dstLvl = toStoredDim(encDst, dim);
          xs[dstLvl] =
              genToCoordinates(rewriter, loc, src, srcLvl, /*cooStart=*/0);
        }
        rewriter.create<SortOp>(loc, nnz, xs, ValueRange{y},
                                SparseTensorSortKind::HybridQuickSort);
      }
    }

    // For each element in the COO tensor, insert the element to the dst tensor.
    SmallVector<Value> dynDstSizes;
    getDynamicSizes(dstTp, srcSizes, dynDstSizes);
    Value dst = rewriter
                    .create<AllocTensorOp>(loc, dstTp.getRankedTensorType(),
                                           dynDstSizes, Value(),
                                           /*sizeHint=*/nnz, Attribute())
                    .getResult();
    SmallVector<Value> dstLcvs(dstLvlRank);
    auto foreachOp = rewriter.create<ForeachOp>(
        loc, src, dst,
        [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
            ValueRange reduc) {
          for (Dimension d = 0; d < dimRank; d++) {
            // FIXME: `toStoredDim` is deprecated
            Level l = toStoredDim(encDst, d);
            dstLcvs[l] = dcvs[d];
          }
          auto t = builder.create<InsertOp>(loc, v, reduc.front(), dstLcvs);
          builder.create<sparse_tensor::YieldOp>(loc, t);
        });

    // Release the temporary COO if it is created. Note that tmpCoo is
    // invalidated due to foreach and updated to src.
    if (tmpCoo)
      rewriter.create<DeallocTensorOp>(loc, src);

    // Directly replace op with dst results in bufferization error message
    // "sparse tensor allocation should not escape function".
    // As such, we insert a trivial tensor convert which will be removed by
    // codegen.
    rewriter.setInsertionPointAfter(op);
    auto t = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    rewriter.replaceOpWithNewOp<ConvertOp>(op, dstTp.getRankedTensorType(), t);
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
    SmallVector<Value> reduc = op.getInitArgs();
    const auto stt = getSparseTensorType(input);
    const Dimension dimRank = stt.getDimRank();
    const Level lvlRank = stt.getLvlRank();

    // Special-case: for each over a sparse constant uses its own rewriting
    // rule.
    if (auto constOp = input.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = constOp.getValue().dyn_cast<SparseElementsAttr>()) {
        return genForeachOnSparseConstant(op, rewriter, attr);
      }
    }

    // Otherwise, use loop emitter to generate loops.
    const auto enc = stt.getEncoding();

    // 1. Generates loop for the sparse input.
    LoopEmitter loopEmitter(
        ValueRange{input},
        StringAttr::get(getContext(), ForeachOp::getOperationName()));
    loopEmitter.initializeLoopEmit(rewriter, loc);
    for (Level l = 0; l < lvlRank; l++) {
      // TODO: provide utility function for loop sequences that only contains
      // one for loop?
      // FIXME(wrengr): what is this "ld" supposed to be really?
      const Level ld = op.getOrder() ? op.getOrder()->getDimPosition(l) : l;
      const SmallVector<TensorId, 1> tids{0};
      loopEmitter.enterNewLoopSeq(rewriter, loc, tids, ld);
      // Note that reduc will be taken care of by loop emitter and get updated
      // in place.

      loopEmitter.enterLoopOverTensorAtLvl(rewriter, loc, tids, l, reduc);
    }

    SmallVector<Value> lcvs;
    lcvs.reserve(lvlRank);
    loopEmitter.getLoopIVs(lcvs);

    if (op.getOrder()) {
      // FIXME: There is some dim/lvl confusion here since `dimRank != lvlRank`
      SmallVector<Value> dcvs = lcvs; // keep a copy
      for (Dimension d = 0; d < dimRank; d++) {
        auto l = op.getOrder()->getDimPosition(d);
        lcvs[l] = dcvs[d];
      }
    }
    Value vals = loopEmitter.getValBuffer()[0];
    Value pos = loopEmitter.getPosits()[0].back();
    // Loads the value from sparse tensor using position-index;
    // loads the value from dense tensor using coords.
    Value val = enc ? rewriter.create<memref::LoadOp>(loc, vals, pos)
                    : rewriter.create<memref::LoadOp>(loc, vals, lcvs);

    // 2. Inline the block in the foreach operator.
    Block *srcBlock = op.getBody();

    // Remap coordinates.
    SmallVector<Value> args;
    for (Dimension d = 0; d < dimRank; d++) {
      // FIXME: `toStoredDim` is deprecated
      Value dimCrd = lcvs[toStoredDim(enc, d)];
      args.push_back(dimCrd);
    }
    // Remap value.
    args.push_back(val);
    // Remap reduction variables.
    args.append(reduc);

    // Remove sparse_tensor.yield.
    SmallVector<Value> reducValue = srcBlock->getTerminator()->getOperands();
    rewriter.eraseOp(srcBlock->getTerminator());

    // Inline body.
    if (!reducValue.empty()) {
      rewriter.mergeBlocks(srcBlock, rewriter.getBlock(), args);
    } else {
      // This is annoying, since scf.for inserts a implicit yield op when
      // there is no reduction variable upon creation, in this case we need to
      // merge the block *before* the yield op.
      rewriter.inlineBlockBefore(srcBlock, &*rewriter.getInsertionPoint(),
                                 args);
    }

    for (Dimension d = 0; d < dimRank; d++) {
      // Link the reduction chain. Note that loop emitter update the reducValue
      // in place.
      loopEmitter.exitCurrentLoop(rewriter, loc, reducValue);
      loopEmitter.exitCurrentLoopSeq();
    }

    // Replace the foreach operator with the value returned by the outtermost
    // for loop.
    rewriter.replaceOp(op, reducValue);
    return success();
  }
};

/// Sparse rewriting rule for the new operator.
struct NewRewriter : public OpRewritePattern<NewOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NewOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const auto dstTp = getSparseTensorType(op.getResult());
    const auto encDst = dstTp.getEncoding();
    if (!dstTp.hasEncoding() || getCOOStart(encDst) == 0)
      return failure();

    // Implement the NewOp as follows:
    //   %orderedCoo = sparse_tensor.new %filename
    //   %t = sparse_tensor.convert %orderedCoo
    RankedTensorType cooTp =
        getCOOFromTypeWithOrdering(dstTp, encDst.getDimOrdering(), true);
    Value cooTensor = rewriter.create<NewOp>(loc, cooTp, op.getSource());
    Value convert = rewriter.replaceOpWithNewOp<ConvertOp>(
        op, dstTp.getRankedTensorType(), cooTensor);

    // Release the ordered COO tensor.
    rewriter.setInsertionPointAfterValue(convert);
    rewriter.create<DeallocTensorOp>(loc, cooTensor);

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

    // Allocate a temporary buffer for storing dimension-sizes/coordinates.
    const auto srcTp = getSparseTensorType(src);
    const Dimension dimRank = srcTp.getDimRank();
    Type indexTp = rewriter.getIndexType();
    Value dimSizes = genAlloca(rewriter, loc, dimRank, indexTp);

    // Generate code to calculate dimension size values and store the values to
    // the buffer.
    SmallVector<Value> dims;
    sizesForTensor(rewriter, dims, loc, srcTp, src);
    for (Dimension d = 0; d < dimRank; d++) {
      rewriter.create<memref::StoreOp>(loc, dims[d], dimSizes,
                                       constantIndex(rewriter, loc, d));
    }

    // Create a sparse tensor writer and output meta data.
    Type opaqueTp = getOpaquePointerType(rewriter);
    Value writer =
        createFuncCall(rewriter, loc, "createSparseTensorWriter", {opaqueTp},
                       {op.getDest()}, EmitCInterface::Off)
            .getResult(0);
    Value rankValue = constantIndex(rewriter, loc, dimRank);
    createFuncCall(rewriter, loc, "outSparseTensorWriterMetaData", {},
                   {writer, rankValue, nnz, dimSizes}, EmitCInterface::On);

    Value dimCoords = dimSizes; // Reuse the dimSizes buffer for dimCoords.
    Type eltTp = srcTp.getElementType();
    SmallString<29> outNextFuncName{"outSparseTensorWriterNext",
                                    primaryTypeFunctionSuffix(eltTp)};
    Value value = genAllocaScalar(rewriter, loc, eltTp);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // For each element in the source tensor, output the element.
    rewriter.create<ForeachOp>(
        loc, src, std::nullopt,
        [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
            ValueRange reduc) {
          for (Dimension d = 0; d < dimRank; d++) {
            rewriter.create<memref::StoreOp>(loc, dcvs[d], dimCoords,
                                             constantIndex(builder, loc, d));
          }
          rewriter.create<memref::StoreOp>(loc, v, value);
          SmallVector<Value> operands{writer, rankValue, dimCoords, value};
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

void mlir::populatePreSparsificationRewriting(RewritePatternSet &patterns) {
  patterns.add<FoldInvariantYield, FuseSparseMultiplyOverAdd>(
      patterns.getContext());
}

void mlir::populatePostSparsificationRewriting(RewritePatternSet &patterns,
                                               bool enableRT,
                                               bool enableForeach,
                                               bool enableConvert) {
  patterns.add<ReshapeRewriter<tensor::ExpandShapeOp>,
               ReshapeRewriter<tensor::CollapseShapeOp>>(patterns.getContext());
  if (enableForeach)
    patterns.add<ForeachRewriter>(patterns.getContext());
  // TODO: If RT not enabled, rewrite concatenate ops, etc here.
  if (!enableRT) {
    patterns.add<ConcatenateRewriter, NewRewriter, OutRewriter,
                 Sparse2SparseReshapeRewriter<tensor::ExpandShapeOp>,
                 Sparse2SparseReshapeRewriter<tensor::CollapseShapeOp>>(
        patterns.getContext());
    if (enableConvert)
      patterns.add<ConvertRewriter>(patterns.getContext());
  }
}
