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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

// Helper to detect a sparse tensor type operand.
static bool isSparseTensor(OpOperand *op) {
  if (auto enc = getSparseTensorEncoding(op->get().getType())) {
    ArrayRef<SparseTensorEncodingAttr::DimLevelType> dimTypes =
        enc.getDimLevelType();
    for (auto dimType : dimTypes)
      if (dimType == SparseTensorEncodingAttr::DimLevelType::Compressed)
        return true; // at least one compressed
  }
  return false;
}

// Helper method to find zero or empty initialization.
static bool isEmptyInit(OpOperand *op) {
  Value val = op->get();
  return matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()) ||
         val.getDefiningOp<InitTensorOp>() ||
         val.getDefiningOp<AllocTensorOp>();
}

// Helper to detect sampling operation.
static bool isSampling(GenericOp op) {
  auto yieldOp = cast<linalg::YieldOp>(op.region().front().getTerminator());
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
  auto yieldOp = cast<linalg::YieldOp>(op.region().front().getTerminator());
  if (auto *def = yieldOp.getOperand(0).getDefiningOp()) {
    if (isa<arith::AddFOp>(def) || isa<arith::AddIOp>(def)) {
      Value x = op.getBlock()->getArguments().back();
      return (def->getOperand(0) == x && isMulChain(def->getOperand(1), x)) ||
             (def->getOperand(1) == x && isMulChain(def->getOperand(0), x));
    }
  }
  return false;
}

//===---------------------------------------------------------------------===//
// The actual sparse tensor rewriting rules.
//===---------------------------------------------------------------------===//

namespace {

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
    if (!op.hasTensorSemantics() || op.getNumInputs() != 2 ||
        op.getNumResults() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops() ||
        !op.getTiedIndexingMap(op.getOutputOperand(0)).isIdentity() ||
        !op.getTiedIndexingMap(op.getInputOperand(0)).isIdentity() ||
        !op.getTiedIndexingMap(op.getInputOperand(1)).isIdentity())
      return failure();
    // Find consuming OP2(sparse, other) or OP2(other, sparse). The other
    // operand can be sparse or dense, since the point of this rewriting rule
    // is detecting a situation in which *more* sparsity is introduced into
    // a computation, be it already sparse or still dense.
    unsigned other = 0;
    if (isSparseTensor(op.getInputOperand(0)))
      other = 1;
    else if (!isSparseTensor(op.getInputOperand(1)))
      return failure();
    // Check producer.
    auto prod = dyn_cast_or_null<GenericOp>(
        op.getInputOperand(other)->get().getDefiningOp());
    if (!prod || !prod.hasTensorSemantics() || prod.getNumResults() != 1 ||
        !prod.getResult(0).hasOneUse())
      return failure();
    // Sampling consumer and sum of multiplication chain producer.
    if (!isEmptyInit(op.getOutputOperand(0)) ||
        !isEmptyInit(prod.getOutputOperand(0)) || !isSampling(op) ||
        !isSumOfMul(prod))
      return failure();
    // Modify operand structure of producer and consumer.
    Location loc = prod.getLoc();
    SmallVector<Value> inputOps = prod.getInputOperands();
    SmallVector<Value> outputOps = op.getOutputOperands();
    SmallVector<AffineMap> fusedIndexMaps = prod.getIndexingMapsArray();
    inputOps.push_back(op.getInputOperand(1 - other)->get());
    fusedIndexMaps.push_back(fusedIndexMaps.back()); // mimic other
    // Fuse producer and consumer into a new generic op.
    auto fusedOp = rewriter.create<GenericOp>(
        loc, op.getResult(0).getType(), inputOps, outputOps,
        rewriter.getAffineMapArrayAttr(fusedIndexMaps), prod.iterator_types(),
        /*doc=*/nullptr, /*library_call=*/nullptr);
    Block &prodBlock = prod.region().front();
    Block &consBlock = op.region().front();
    BlockAndValueMapping mapper;
    Block *fusedBlock = new Block();
    fusedOp.region().push_back(fusedBlock);
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

/// Sparse rewriting rule for reshape operator.
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
    } else if (encSrc) {
      RankedTensorType rtp =
          op.getSrc().getType().template cast<RankedTensorType>();
      auto denseTp =
          RankedTensorType::get(rtp.getShape(), rtp.getElementType());
      auto convert = rewriter.create<ConvertOp>(loc, denseTp, op.getSrc());
      op->setOperand(0, convert);
      return success();
    } else if (encDst) {
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

} // namespace

//===---------------------------------------------------------------------===//
// Methods that add patterns described in this file to a pattern list.
//===---------------------------------------------------------------------===//

void mlir::populateSparseTensorRewriting(RewritePatternSet &patterns) {
  // TODO(springerm): enable FuseSparseMultiplyOverAdd
  patterns.add<ReshapeRewriter<tensor::ExpandShapeOp>,
               ReshapeRewriter<tensor::CollapseShapeOp>>(patterns.getContext());
}
