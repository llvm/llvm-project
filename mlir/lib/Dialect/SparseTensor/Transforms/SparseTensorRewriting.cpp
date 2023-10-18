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
static bool isSparseTensor(Value v) {
  auto enc = getSparseTensorEncoding(v.getType());
  return enc && !llvm::all_of(enc.getLvlTypes(), [](auto dlt) {
           return dlt == DimLevelType::Dense;
         });
}
static bool isSparseTensor(OpOperand *op) { return isSparseTensor(op->get()); }

// Helper method to find zero/uninitialized tensor materialization.
static bool isMaterializing(OpOperand *op, bool isZero) {
  Value val = op->get();
  // Check allocation, with zero alloc when required.
  if (auto alloc = val.getDefiningOp<AllocTensorOp>()) {
    Value copy = alloc.getCopy();
    if (isZero)
      return copy && isZeroValue(copy);
    return !copy;
  }
  // Check for empty tensor materialization.
  if (auto empty = val.getDefiningOp<tensor::EmptyOp>())
    return !isZero;
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
  if (auto arg = dyn_cast<BlockArgument>(val))
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
  if (auto arg = dyn_cast<BlockArgument>(yieldOp.getOperand(0))) {
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
static RankedTensorType getCOOType(const SparseTensorType &stt, bool ordered) {
  return getCOOFromTypeWithOrdering(stt, stt.getDimToLvl(), ordered);
}

static RankedTensorType getBufferType(const SparseTensorType &stt,
                                      bool needTmpCOO) {
  return needTmpCOO ? getCOOType(stt, /*ordered=*/false)
                    : stt.getRankedTensorType();
}

/// Collects the dynamic dimension sizes for `tp` with the assumption that
/// `sizes` are the dimension sizes for the type. Stores the dynamic dimension
/// sizes to dynSizes.
static void getDynamicSizes(RankedTensorType tp, ValueRange sizes,
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
      rewriter, loc, attr, op.getOrder().value_or(AffineMap()),
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
        !isMaterializing(op.getDpsInitOperand(0), /*isZero=*/false) ||
        !isZeroYield(op) || !op.getDpsInitOperand(0)->get().hasOneUse())
      return failure();
    auto outputType = getRankedTensorType(op.getResult(0));
    // Yielding zero on newly materialized sparse tensor can be
    // optimized directly (regardless of dynamic or static size).
    if (getSparseTensorEncoding(outputType)) {
      rewriter.replaceOp(op, op.getDpsInitOperand(0)->get());
      return success();
    }
    // Use static zero value directly instead of materialization.
    if (!outputType.hasStaticShape())
      return failure();
    Operation *def = op.getDpsInitOperand(0)->get().getDefiningOp();
    rewriter.replaceOp(op, constantZero(rewriter, op.getLoc(), outputType));
    rewriter.eraseOp(def);
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
    if (!isMaterializing(op.getDpsInitOperand(0), /*isZero=*/false) ||
        !isMaterializing(prod.getDpsInitOperand(0), /*isZero=*/true) ||
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
    // TODO: deal with non alloc tensor here one day
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

// Fuse a tensor cast into producing operation. Note that a tensor.cast
// should really not be used to convert between sparse encodings. Since
// the pattern currently appears as a result of some prior rewriting
// we make an attempt to repair very obvious cases.
// TODO: audit the pure tensor dialect rewriting rules
struct FuseTensorCast : public OpRewritePattern<tensor::CastOp> {
public:
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    Type srcType = op.getSource().getType();
    Type dstType = op.getDest().getType();
    // A nop cast simply folds away.
    if (srcType == dstType) {
      rewriter.replaceOp(op, op->getResults());
      return success();
    }
    // See if a sparsity changing cast can be fused into producer.
    if (tensor::isSameTypeWithoutEncoding(srcType, dstType)) {
      if (Operation *def = op.getSource().getDefiningOp()) {
        if (def->hasOneUse() && isa<tensor::ExtractSliceOp>(def)) {
          rewriter.updateRootInPlace(def, [&]() {
            def->getResult(0).setType(op->getResultTypes()[0]);
          });
          rewriter.replaceOp(op, def->getResult(0));
          return success();
        }
      }
    }
    // Repair tensor casts with at least one sparse operand into the
    // the properly supported sparse_tensor.convert.
    if (getSparseTensorEncoding(srcType) || getSparseTensorEncoding(dstType)) {
      rewriter.replaceOpWithNewOp<ConvertOp>(op, dstType, op.getSource());
      return success();
    }
    // Fail otherwise.
    return failure();
  }
};

/// Rewrites a sequence of operations for sparse tensor selections in to
/// semi-ring operations such that they can be compiled correctly by the sparse
/// compiler. E.g., transforming the following sequence
///
/// %sel = arith.select %cond, %sp1, %sp2
///
/// to
///
/// %sel = binary %sp1, %sp2:
///         both  (%l, %r) {yield select %cond, %l, %r}
///         left  (%l)     {yield select %cond, %l,  0}
///         right (%r)     {yield select %cond,  0, %r}
///
/// TODO: We require that the tensor used for extracting conditions to be dense
/// to sparsify the code. To support a sparse condition tensor, we need a
/// tri-nary operation.
struct GenSemiRingSelect : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Rejects non sparse kernels.
    if (!op.hasTensorSemantics() || !hasAnySparseOperand(op))
      return failure();

    Location loc = op.getLoc();
    SmallVector<std::pair<Operation *, sparse_tensor::BinaryOp>> semiRings;
    for (Operation &inst : *op.getBody()) {
      // Matches pattern.
      auto matched = isRewritablePattern(op, &inst);
      if (!matched.has_value())
        continue;

      rewriter.setInsertionPoint(&inst);
      auto [c, t, f] = matched.value();
      assert(t.getType() == f.getType());
      auto selTp = t.getType();
      auto c0 = constantZero(rewriter, loc, selTp);
      auto binOp = rewriter.create<sparse_tensor::BinaryOp>(loc, selTp, t, f);
      // Initializes all the blocks.
      rewriter.createBlock(&binOp.getOverlapRegion(), {}, {selTp, selTp},
                           {t.getLoc(), f.getLoc()});
      rewriter.createBlock(&binOp.getRightRegion(), {}, selTp, f.getLoc());
      rewriter.createBlock(&binOp.getLeftRegion(), {}, selTp, t.getLoc());

      for (auto *r : binOp.getRegions()) {
        Block *b = &r->front();
        rewriter.setInsertionPointToStart(b);

        IRMapping irMap;
        // Clones the cmp operations into the region to make the binary op
        // admissible.
        Value newC = c;
        if (auto *def = c.getDefiningOp())
          newC = rewriter.clone(*def, irMap)->getResult(0);

        irMap.map(c, newC);
        if (r == &binOp.getLeftRegion()) {
          irMap.map(t, b->getArgument(0));
          irMap.map(f, c0);
        } else if (r == &binOp.getRightRegion()) {
          irMap.map(t, c0);
          irMap.map(f, b->getArgument(0));
        } else {
          irMap.map(t, b->getArgument(0));
          irMap.map(f, b->getArgument(1));
        }
        auto y = rewriter.clone(inst, irMap)->getResult(0);
        rewriter.create<sparse_tensor::YieldOp>(loc, y);
      }

      // We successfully rewrited a operation. We can not do replacement here
      // becuase it invalidate the iterator for the current loop to traverse
      // the instructions.
      semiRings.emplace_back(&inst, binOp);
    }

    // Finalizes the replacement.
    for (auto [sel, semi] : semiRings)
      rewriter.replaceOp(sel, semi->getResults());

    return success(!semiRings.empty());
  }

private:
  static std::optional<std::tuple<Value, BlockArgument, BlockArgument>>
  isRewritablePattern(GenericOp op, Operation *v) {
    auto sel = dyn_cast<arith::SelectOp>(v);
    if (!sel)
      return std::nullopt;

    auto tVal = sel.getTrueValue().dyn_cast<BlockArgument>();
    auto fVal = sel.getFalseValue().dyn_cast<BlockArgument>();
    // TODO: For simplicity, we only handle cases where both true/false value
    // are directly loaded the input tensor. We can probably admit more cases
    // in theory.
    if (!tVal || !fVal)
      return std::nullopt;

    // Helper lambda to determine whether the value is loaded from a dense input
    // or is a loop invariant.
    auto isValFromDenseInputOrInvariant = [&op](Value v) -> bool {
      if (auto bArg = v.dyn_cast<BlockArgument>();
          bArg && !isSparseTensor(op.getDpsInputOperand(bArg.getArgNumber())))
        return true;
      // If the value is defined outside the loop, it is a loop invariant.
      return v.getDefiningOp() && v.getDefiningOp()->getBlock() != op.getBody();
    };

    // If the condition value is load directly from a dense tensor or
    // loop-invariants, we can sparsify the kernel.
    auto cond = sel.getCondition();
    if (isValFromDenseInputOrInvariant(cond))
      return std::make_tuple(cond, tVal, fVal);

    Value cmpL, cmpR;
    if (matchPattern(cond, m_Op<arith::CmpIOp>(matchers::m_Any(&cmpL),
                                               matchers::m_Any(&cmpR))) ||
        matchPattern(cond, m_Op<arith::CmpFOp>(matchers::m_Any(&cmpL),
                                               matchers::m_Any(&cmpR)))) {
      // TODO: we can do it recursively to check whether all the leaf values are
      // loaded from dense tensors or are loop invariants.
      if (isValFromDenseInputOrInvariant(cmpL) ||
          isValFromDenseInputOrInvariant(cmpR))
        return std::make_tuple(cond, tVal, fVal);
    }

    return std::nullopt;
  };
};

/// Rewrites a sparse reduction that would not sparsify directly since
/// doing so would only iterate over the stored elements, ignoring the
/// implicit zeros, into a semi-ring. Applies to all prod/and/min/max
/// (note that reductions like add/sub/or/xor can directly be sparsified
/// since the implicit zeros do not contribute to the final result).
/// Note that prod/and are still included since, even though they often
/// are nullified in sparse data, they may still occur for special
/// situations in which e.g. some rows in a sparse matrix are fully
/// dense. For min/max, including the implicit zeros is a much more
/// common situation.
///
/// TODO: this essentially "densifies" the operation; we want to implement
///       this much more efficiently by performing the reduction over the
///       stored values, and feed in the zero once if there were *any*
///       implicit zeros as well; but for now, at least we provide
///       the functionality
///
struct GenSemiRingReduction : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Reject non-reductions.
    if (!op.hasTensorSemantics() || op.getNumDpsInputs() != 1 ||
        op.getNumReductionLoops() == 0 || op.getNumResults() != 1)
      return failure();
    auto inp = op.getDpsInputOperand(0);
    auto init = op.getDpsInitOperand(0);
    if (!isSparseTensor(inp))
      return failure();
    // Look for direct x = x OP y for semi-ring ready reductions.
    auto red = cast<linalg::YieldOp>(op.getRegion().front().getTerminator())
                   .getOperand(0)
                   .getDefiningOp();
    if (!isa<arith::AndIOp, arith::MulIOp, arith::MulFOp, arith::MinimumFOp,
             arith::MinSIOp, arith::MinUIOp, arith::MaximumFOp, arith::MaxSIOp,
             arith::MaxUIOp>(red))
      return failure();
    Value s0 = op.getBlock()->getArgument(0);
    Value s1 = op.getBlock()->getArgument(1);
    if ((red->getOperand(0) != s0 || red->getOperand(1) != s1) &&
        (red->getOperand(0) != s1 || red->getOperand(1) != s0))
      return failure();
    // Identity.
    Location loc = op.getLoc();
    Value identity =
        rewriter.create<tensor::ExtractOp>(loc, init->get(), ValueRange());
    // Unary {
    //    present -> value
    //    absent  -> zero.
    // }
    Type rtp = s0.getType();
    rewriter.setInsertionPointToStart(&op.getRegion().front());
    auto semiring = rewriter.create<sparse_tensor::UnaryOp>(loc, rtp, s0);
    Block *present =
        rewriter.createBlock(&semiring.getPresentRegion(), {}, rtp, loc);
    rewriter.setInsertionPointToStart(&semiring.getPresentRegion().front());
    rewriter.create<sparse_tensor::YieldOp>(loc, present->getArgument(0));
    rewriter.createBlock(&semiring.getAbsentRegion(), {}, {}, {});
    rewriter.setInsertionPointToStart(&semiring.getAbsentRegion().front());
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(rtp));
    rewriter.create<sparse_tensor::YieldOp>(loc, zero);
    rewriter.setInsertionPointAfter(semiring);
    // CustomReduce {
    //    x = x REDUC y, identity
    // }
    auto custom = rewriter.create<sparse_tensor::ReduceOp>(
        loc, rtp, semiring.getResult(), s1, identity);
    Block *region =
        rewriter.createBlock(&custom.getRegion(), {}, {rtp, rtp}, {loc, loc});
    rewriter.setInsertionPointToStart(&custom.getRegion().front());
    IRMapping irMap;
    irMap.map(red->getOperand(0), region->getArgument(0));
    irMap.map(red->getOperand(1), region->getArgument(1));
    auto cloned = rewriter.clone(*red, irMap);
    rewriter.create<sparse_tensor::YieldOp>(loc, cloned->getResult(0));
    rewriter.setInsertionPointAfter(custom);
    rewriter.replaceOp(red, custom.getResult());
    return success();
  }
};

/// Sparse rewriting rule for sparse-to-sparse reshape operator.
struct TensorReshapeRewriter : public OpRewritePattern<tensor::ReshapeOp> {
public:
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value srcTensor = op.getSource();
    const auto srcTp = getSparseTensorType(srcTensor);
    const auto dstTp = getSparseTensorType(op.getResult());

    if (!srcTp.hasEncoding() || !dstTp.hasEncoding() ||
        !dstTp.hasStaticDimShape())
      return failure();

    SmallVector<Value> srcSizes;
    sizesForTensor(rewriter, srcSizes, loc, srcTp, srcTensor);
    SmallVector<Value> dstSizes;
    for (Dimension d : dstTp.getDimShape())
      dstSizes.push_back(constantIndex(rewriter, loc, d));

    Value nnz = rewriter.create<NumberOfEntriesOp>(loc, srcTensor);
    // Only need an unordered COO buffer if input and output are not sorted
    // in the same way.
    Type bufferTp = getBufferType(
        dstTp.withoutDimToLvl(),
        !srcTp.isAllOrdered() || !srcTp.isIdentity() || !dstTp.isIdentity());
    SmallVector<Value> dynSizes;
    Value buffer = rewriter
                       .create<AllocTensorOp>(loc, bufferTp, dynSizes, Value(),
                                              nnz, Attribute())
                       .getResult();

    // Convert src coordinates to dst coordinates by first collapsing it to 1D
    // and then expand it to the match the rank of the destination tensor.
    // Implemented as follows:
    //   foreach srcCoords %srcTensor
    //     collapsedCoords = reshapeCvs(srcCoords, [1, ..., srcRank])
    //     expandedCoords = reshapeCvs(collapsedCoords, [1, ..., dstRank])
    //     insert expandedCoords, %buffer
    //
    // followed by an optional
    //   %t = sparse_tensor.cast %tmp
    // depending on whether the input/output are sorted in the same way.
    const auto encSrc = srcTp.getEncoding();
    ForeachOp foreachOp = rewriter.create<ForeachOp>(
        loc, srcTensor, buffer,
        [&](OpBuilder &builder, Location loc, ValueRange srcLcvs, Value v,
            ValueRange reduc) {
          const Dimension srcRank = srcTp.getDimRank();
          SmallVector<Value> srcDcvs;
          srcDcvs.reserve(srcRank);
          for (Dimension d = 0; d < srcRank; d++) {
            // FIXME: `toStoredDim` is deprecated
            Level lvl = toStoredDim(encSrc, d);
            srcDcvs.push_back(srcLcvs[lvl]);
          }

          Value collapseSize = constantIndex(builder, loc, 1);
          for (Dimension d = 0; d < srcRank; d++)
            collapseSize =
                builder.create<arith::MulIOp>(loc, collapseSize, srcSizes[d]);
          SmallVector<Value, 1> collapsedSizes = {collapseSize};

          ReassociationIndices collapseIdx;
          for (Dimension i = 0; i < srcRank; i++)
            collapseIdx.push_back(i);
          SmallVector<ReassociationIndices, 1> collapseReass = {collapseIdx};
          SmallVector<Value, 1> collapsedDcvs;
          reshapeCvs(builder, loc, collapseReass, srcSizes, srcDcvs,
                     collapsedSizes, collapsedDcvs);

          ReassociationIndices expandIdx;
          for (Dimension i = 0; i < dstTp.getDimRank(); i++)
            expandIdx.push_back(i);
          SmallVector<ReassociationIndices, 1> expandReass = {expandIdx};
          SmallVector<Value> dstDcvs;
          reshapeCvs(builder, loc, expandReass, collapsedSizes, collapsedDcvs,
                     dstSizes, dstDcvs);

          auto t = builder.create<InsertOp>(loc, v, reduc.front(), dstDcvs);
          builder.create<sparse_tensor::YieldOp>(loc, t);
        });

    Value t = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    if (bufferTp != dstTp) {
      auto dstRTT = dstTp.getRankedTensorType();
      Value converted = rewriter.create<ConvertOp>(loc, dstRTT, t).getResult();
      rewriter.create<DeallocTensorOp>(loc, t);
      t = converted;
    }
    rewriter.replaceOp(op, t);
    return success();
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
    const auto srcTp = getSparseTensorType(srcTensor);
    const auto dstTp = getSparseTensorType(op.getResult());
    if (!srcTp.hasEncoding() || !dstTp.hasEncoding())
      return failure();

    // Generate code to represent the static dimension constants or compute
    // the dynamic dimension values.
    SmallVector<Value> srcSizes;
    sizesForTensor(rewriter, srcSizes, loc, srcTp, srcTensor);
    SmallVector<Value> dstSizes;
    SmallVector<Value> dstDynSizes;
    if (dstTp.hasStaticDimShape()) {
      for (Dimension d : dstTp.getDimShape())
        dstSizes.push_back(constantIndex(rewriter, loc, d));
    } else {
      ArrayRef<DynSize> dstShape = dstTp.getDimShape();
      genReshapeDstShape(rewriter, loc, dstSizes, srcSizes, dstShape,
                         op.getReassociationIndices());
      for (auto [idx, shape] : llvm::enumerate(dstShape)) {
        if (shape == ShapedType::kDynamic)
          dstDynSizes.push_back(dstSizes[idx]);
      }
    }
    Value nnz = rewriter.create<NumberOfEntriesOp>(loc, srcTensor);
    // Only need a unordered COO buffer if input and output are not sorted
    // in the same way.
    Type bufferTp = getBufferType(
        dstTp.withoutDimToLvl(),
        !srcTp.isAllOrdered() || !srcTp.isIdentity() || !dstTp.isIdentity());

    Value buffer =
        rewriter
            .create<AllocTensorOp>(loc, bufferTp, dstDynSizes, Value(),
                                   /*sizeHint=*/nnz, Attribute())
            .getResult();

    // Implement the sparse2sparse reshape as follows:
    //   foreach srcCoords %srcTensor
    //     insert reshapeCvs(srcCoords), %buffer
    //
    // followed by an optional
    //   %t = sparse_tensor.cast %tmp
    // depending on whether the input/output are sorted in the same way.
    const auto encSrc = srcTp.getEncoding();
    ForeachOp foreachOp = rewriter.create<ForeachOp>(
        loc, srcTensor, buffer,
        [&](OpBuilder &builder, Location loc, ValueRange srcLcvs, Value v,
            ValueRange reduc) {
          const Dimension dimRank = srcTp.getDimRank();
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

    Value t = rewriter.create<LoadOp>(loc, foreachOp.getResult(0), true);
    if (bufferTp != dstTp) {
      auto dstRTT = dstTp.getRankedTensorType();
      Value converted = rewriter.create<ConvertOp>(loc, dstRTT, t).getResult();
      rewriter.create<DeallocTensorOp>(loc, t);
      t = converted;
    }
    rewriter.replaceOp(op, t);
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

struct TensorLike {
  TensorLike(OpBuilder &builder, Location loc, RankedTensorType rtt,
             ValueRange sizes)
      : isSparse(rtt.getEncoding() != nullptr) {
    SmallVector<Value> dynSzs;
    getDynamicSizes(rtt, sizes, dynSzs);

    if (isSparse)
      val = builder.create<AllocTensorOp>(loc, rtt, dynSzs);
    else
      val = allocDenseTensor(builder, loc, rtt, sizes);
  };

  void insertOrStore(OpBuilder &builder, Location loc, Value v,
                     ValueRange crds) {
    if (isSparse)
      val = builder.create<InsertOp>(loc, v, val, crds);
    else
      builder.create<memref::StoreOp>(loc, v, val, crds);
  }

  Value getSSA() const {
    // We don't need to maintain the SSA chain for a memref value.
    return isSparse ? val : nullptr;
  }

  Value finalize(OpBuilder &builder, Location loc, RankedTensorType rtp) const {
    if (isSparse)
      return builder.create<LoadOp>(loc, val, true);
    return builder.create<bufferization::ToTensorOp>(loc, rtp, val);
  }

  void updateSSA(Value v) {
    // Dense memref is a non-SSA value.
    assert(isSparse);
    val = v;
  }

private:
  bool isSparse;
  Value val; // either a memref (for dense tensor) or a sparse tensor.
};

struct ConcatenateRewriter : public OpRewritePattern<ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.needsExtraSort())
      op.emitError("ConcatenateOp not staged");

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

    TensorLike dstBuf(rewriter, loc, dstTp.getRankedTensorType(), sizes);
    Value offset = constantIndex(rewriter, loc, 0);
    Value iterArg = dstBuf.getSSA();

    ForeachOp foreachOp;
    for (Value input : op.getInputs()) {
      // Builds a for op for each input tensor to append new values into the
      // output tensor.
      foreachOp = rewriter.create<ForeachOp>(
          loc, input, iterArg ? ValueRange{iterArg} : ValueRange{},
          [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
              ValueRange reduc) {
            SmallVector<Value> dstLcvs(dstTp.getLvlRank());
            for (Dimension d = 0; d < dimRank; d++) {
              Value crd = dcvs[d];
              // Transforms coordinates for the concatenating dim.
              if (d == conDim)
                crd = builder.create<arith::AddIOp>(loc, crd, offset);
              // FIXME: `toStoredDim` is deprecated
              dstLcvs[toStoredDim(dstTp.getEncoding(), d)] = crd;
            }

            if (!reduc.empty())
              dstBuf.updateSSA(reduc.front());

            if (!dstTp.isAllDense()) {
              Value cond = genIsNonzero(builder, loc, v);
              auto ifOp = builder.create<scf::IfOp>(loc, reduc.getTypes(), cond,
                                                    /*else*/ true);
              builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
              builder.create<scf::YieldOp>(loc, dstBuf.getSSA());

              builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
              dstBuf.insertOrStore(builder, loc, v, dstLcvs);
              builder.create<scf::YieldOp>(loc, dstBuf.getSSA());

              // Exits the ifOp, update the sparse tensor SSA value.
              builder.setInsertionPointAfter(ifOp);
              assert(!reduc.empty());
              dstBuf.updateSSA(ifOp.getResult(0));
            } else {
              dstBuf.insertOrStore(builder, loc, v, dstLcvs);
            }
            if (reduc.empty())
              builder.create<sparse_tensor::YieldOp>(loc);
            else
              builder.create<sparse_tensor::YieldOp>(loc, dstBuf.getSSA());
          });
      // Accumulates the offset. Note that only static-shaped inputs are allowed
      // by concatenate op verifier, which saves us from computing the offset
      // dynamically.
      const auto sh = getSparseTensorType(input).getStaticDimSize(conDim);
      assert(sh.has_value());
      offset = rewriter.create<arith::AddIOp>(
          loc, offset, constantIndex(rewriter, loc, *sh));

      if (!foreachOp.getResults().empty()) {
        iterArg = foreachOp.getResult(0);
        dstBuf.updateSSA(iterArg);
      }
    }

    if (!foreachOp.getResults().empty())
      dstBuf.updateSSA(iterArg);

    Value ret = dstBuf.finalize(rewriter, loc, dstTp.getRankedTensorType());
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct DirectConvertRewriter : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter &rewriter) const override {
    if (op.needsExtraSort())
      return op.emitError("ConvertOp not staged.");

    // TODO: Maybe we want a different operation for this too.
    auto encDst = getSparseTensorEncoding(op.getType());
    auto encSrc = getSparseTensorEncoding(op.getSource().getType());
    if (encDst && encSrc && !encSrc.isSlice() &&
        encSrc.withoutBitWidths() == encDst.withoutBitWidths()) {
      // Trivial tensor conversion and simple element type conversion is handled
      // in codegen.
      return failure();
    }

    Location loc = op.getLoc();
    Value src = op.getSource();

    SparseTensorType srcStt = getSparseTensorType(op.getSource());
    SparseTensorType dstStt = getSparseTensorType(op.getDest());

    bool fromSparseConst = false;
    if (auto constOp = op.getSource().getDefiningOp<arith::ConstantOp>())
      if (dyn_cast<SparseElementsAttr>(constOp.getValue()))
        fromSparseConst = true;

    const AffineMapAttr foreachOrder =
        (!dstStt.isIdentity() && fromSparseConst)
            ? AffineMapAttr::get(dstStt.getExpandedDimToLvl())
            : nullptr;

    bool skipZeroCheck = srcStt.hasEncoding() || fromSparseConst;

    SmallVector<Value> sizes;
    sizesFromSrc(rewriter, sizes, loc, src);
    ValueRange vs;
    TensorLike dstBuf(rewriter, loc, dstStt.getRankedTensorType(), sizes);

    Value iterArg = dstBuf.getSSA();
    auto foreachOp = rewriter.create<ForeachOp>(
        loc, src, iterArg ? ValueRange{iterArg} : ValueRange{}, foreachOrder,
        [&](OpBuilder &builder, Location loc, ValueRange dcvs, Value v,
            ValueRange reduc) {
          // Enters the loop, update the SSA value for insertion chain.
          if (!reduc.empty())
            dstBuf.updateSSA(reduc.front());

          const Dimension dimRank = dstStt.getDimRank();
          const Level lvlRank = dstStt.getLvlRank();
          SmallVector<Value> lcvs(lvlRank);
          for (Dimension d = 0; d < dimRank; d++) {
            // FIXME: `toStoredDim` is deprecated
            lcvs[toStoredDim(dstStt.getEncoding(), d)] = dcvs[d];
          }

          if (!skipZeroCheck) {
            assert(!reduc.empty());
            Value cond = genIsNonzero(builder, loc, v);
            auto ifOp = builder.create<scf::IfOp>(loc, reduc.getTypes(), cond,
                                                  /*else*/ true);
            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            builder.create<scf::YieldOp>(loc, dstBuf.getSSA());

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            dstBuf.insertOrStore(builder, loc, v, lcvs);
            builder.create<scf::YieldOp>(loc, dstBuf.getSSA());

            // Exits the ifOp, update the sparse tensor SSA value.
            builder.setInsertionPointAfter(ifOp);
            dstBuf.updateSSA(ifOp.getResult(0));
          } else {
            dstBuf.insertOrStore(builder, loc, v, lcvs);
          }
          if (reduc.empty())
            builder.create<sparse_tensor::YieldOp>(loc);
          else
            builder.create<sparse_tensor::YieldOp>(loc, dstBuf.getSSA());
        });

    rewriter.setInsertionPointAfter(foreachOp);

    // Exits the for loop, links the SSA chain.
    if (!foreachOp.getResults().empty())
      dstBuf.updateSSA(foreachOp.getResult(0));

    Value ret = dstBuf.finalize(rewriter, loc, dstStt.getRankedTensorType());
    rewriter.replaceOp(op, ret);
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
      if (auto attr = dyn_cast<SparseElementsAttr>(constOp.getValue())) {
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
      const SmallVector<TensorLevel, 1> tidLvls{
          loopEmitter.makeTensorLevel(0, l)};
      loopEmitter.enterNewLoopSeq(rewriter, loc, tidLvls);
      // Note that reduc will be taken care of by loop emitter and get updated
      // in place.
      loopEmitter.enterCoIterationOverTensorsAtLvls(rewriter, loc, tidLvls,
                                                    reduc);
    }

    SmallVector<Value> lcvs = loopEmitter.getLoopIVs();
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
      loopEmitter.exitCurrentLoopSeq(rewriter, loc);
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
    RankedTensorType cooTp = getCOOType(dstTp, /*ordered=*/true);
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
  patterns.add<FoldInvariantYield, FuseSparseMultiplyOverAdd, FuseTensorCast,
               GenSemiRingReduction, GenSemiRingSelect>(patterns.getContext());
}

void mlir::populatePostSparsificationRewriting(RewritePatternSet &patterns,
                                               bool enableRT,
                                               bool enableForeach,
                                               bool enableConvert) {
  patterns.add<ConcatenateRewriter, ReshapeRewriter<tensor::ExpandShapeOp>,
               ReshapeRewriter<tensor::CollapseShapeOp>,
               Sparse2SparseReshapeRewriter<tensor::ExpandShapeOp>,
               Sparse2SparseReshapeRewriter<tensor::CollapseShapeOp>,
               TensorReshapeRewriter>(patterns.getContext());
  if (enableForeach)
    patterns.add<ForeachRewriter>(patterns.getContext());
  if (enableConvert)
    patterns.add<DirectConvertRewriter>(patterns.getContext());
  if (!enableRT)
    patterns.add<NewRewriter, OutRewriter>(patterns.getContext());
}
