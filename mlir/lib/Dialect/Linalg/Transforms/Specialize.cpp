//===- Specialize.cpp - linalg generic ops to named ops  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a method to specialize generic operations to named
// operations. Conceptually it is the opposite of generalize.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGSPECIALIZEGENERICOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-specialization"

#define REPLACE_BINARY_OP(NEWOP, OPERANDS_SWAP)                                \
  (rewriter.replaceOpWithNewOp<NEWOP>(                                         \
      genericOp,                                                               \
      ValueRange{genericOp.getDpsInputs()[(OPERANDS_SWAP) ? 1 : 0],            \
                 genericOp.getDpsInputs()[(OPERANDS_SWAP) ? 0 : 1]},           \
      ValueRange{genericOp.getDpsInits()[0]}))

#define REPLACE_UNARY_OP(NEWOP)                                                \
  (rewriter.replaceOpWithNewOp<NEWOP>(genericOp,                               \
                                      ValueRange{genericOp.getDpsInputs()[0]}, \
                                      ValueRange{genericOp.getDpsInits()[0]}))

using namespace mlir;
using namespace mlir::linalg;

// Given a elementwise single binary linalg generic op, checks whether the
// binary op accesses operands as swapped. e.g.
// this differentiates between a linalg-generic body that contains:
//    ^bb0(%a: f32, %b: f32, %c : f32):
//         %0 = arith.subf %a, %b : f32
//         linalg.yield %0: f32
// against:
//    ^bb0(%a: f32, %b: f32, %c : f32):
//         %0 = arith.subf %b, %a : f32
//         linalg.yield %0: f32
// Former is linalg.sub(a,b), latter is linalg.sub(b,a).
static bool areBinOpsSwapped(GenericOp genericOp) {
  Block *body = genericOp.getBody();
  Operation *op = &body->front();
  bool swapped = false;
  if (op->getOpOperand(0).get() != body->getArgument(0)) {
    swapped = true;
    assert(op->getOpOperand(0).get() == body->getArgument(1) &&
           op->getOpOperand(1).get() == body->getArgument(0) &&
           "binary op uses just one block arg");
  }
  return swapped;
}

//===----------------------------------------------------------------------===//
// Specialize linalg generic to matmul variants.
//===----------------------------------------------------------------------===//
/// Identifies linalg.generic that is essentially named op of the form:
//    ` linalg.{batch_}?matmul{_transpose_a | _transpose_b}? `
//
// It is possible that a linalg.generic may be implementing one of matmul
// variants but not in a straight-forward way, or the linalg.generic's
// affine map per operand capture more semantics than is possible with
// named op (which has implicit map interpreted via name).
//
// But a named linalg matmul variant that was 'generalized' should be
// convertible back to named op here.
//
namespace {
enum class IndexMatchResult {
  Match = 0,  // identity map.
  Transposed, // transposed map.
  Mismatch    // none of the above.
};

// Looks at the affine map of an operand and works out if generic accesses
// the element as identity-map, transposed, or 'cant work out'.
// This check skips the `offset` batch indices and focuses on the matmul part.
static IndexMatchResult matchOperandMap(AffineMap m, unsigned offset,
                                        unsigned i, unsigned j) {
  auto expr_ei = dyn_cast<AffineDimExpr>(m.getResults()[offset]);
  auto expr_ej = dyn_cast<AffineDimExpr>(m.getResults()[offset + 1]);
  if (!expr_ei || !expr_ej)
    return IndexMatchResult::Mismatch;

  auto ei = expr_ei.getPosition();
  auto ej = expr_ej.getPosition();

  if (ei == i && ej == j)
    return IndexMatchResult::Match;

  if (ei == j && ej == i)
    return IndexMatchResult::Transposed;

  return IndexMatchResult::Mismatch;
}

//  All the variants `linalg.{batch_}?matmul{_transpose_a | _transpose_b}?`
//  have same number of input/output.
template <typename Variant>
static LinalgOp replaceWithMatmulVariant(RewriterBase &rewriter, GenericOp op) {
  LinalgOp namedOp = rewriter.replaceOpWithNewOp<Variant>(
      op, ValueRange{op.getDpsInputs()[0], op.getDpsInputs()[1]},
      ValueRange{op.getDpsInits()[0]});
  return namedOp;
}

// Converts linalg.generic to named linalg.*matmul* where possible.
static FailureOr<LinalgOp> specializeLinalgContractions(RewriterBase &rewriter,
                                                        GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1)
    return failure();

  // Linalg generic contraction can be across multiple axis but for matmul
  // variants it must be one.
  if (genericOp.getNumReductionLoops() != 1)
    return failure();

  // Must be projected permutations.
  auto mapRange = genericOp.getIndexingMapsArray();
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); }))
    return failure();

  //  matmul contractions are of the form:
  //  %0 = <elemwise>(permutation-of(cu(block-argument-0),
  //                                 cu(block-argument-1)))
  //  %1 = <reduce>(permutation-of(cu(%0), cu(block-argument-2)))
  //
  //  where <elemwise> and <reduce> are binary operations constituting a
  //  contraction (in the canonical case, <elemwise> is a multiplication and
  //  <reduce> is an addition). All operands of all operations may be supplied
  //  through a chain of side effect-free unary operations, such as casts,
  //  which is denoted as `cu` above.
  if (!mlir::linalg::detail::isContractionBody(
          *genericOp.getBlock(), [](Operation *first, Operation *second) {
            if ((isa<arith::MulFOp>(first) && isa<arith::AddFOp>(second)) ||
                (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second)) ||
                (isa<complex::MulOp>(first) && isa<complex::AddOp>(second)))
              return true;
            return false;
          }))
    return failure();

  // Finds 2 parallel (m and n) and 1 reduction (k) dimension candidates that
  // form a matmul subcomputation. These dimensions are such that:
  //   1. The m dimension is involved in an outer-product along LHS
  //      (i.e. it is a permutation on RES and LHS and does not appear in RHS).
  //   2. The n dimension is involved in an outer-product along RHS
  //      (i.e. it is a permutation on RES and RHS and does not appear in LHS).
  //   3. The k dimension appears as a permutation on LHS and RHS.
  //   4. m, n and k appear only once in any given indexing.
  //   5. Optional batch dimensions that appear in all operands are captured.
  auto res = inferContractionDims(genericOp);
  assert(succeeded(res) && "unexpected failure to infer contraction dims");
  auto dims = *res;

  // Other than `batch`, other dim sizes must be 1 for linalg.*_matmul_*.
  if (dims.m.size() != 1 || dims.n.size() != 1 || dims.k.size() != 1)
    return failure();

  // Check rank of operands
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (llvm::any_of(indexingMaps, [&dims](AffineMap m) {
        return m.getResults().size() !=
               dims.batch.size() + 2 /*two from {m,n,k}*/;
      }))
    return failure();

  auto batchSize = dims.batch.size();
  if (indexingMaps[0].getNumDims() != batchSize + 3) {
  }
  if (batchSize) {
    // Each operand in a linalg generic contraction  could express different
    // permutations for its batch dimension. But for named op it must be
    // identity since separate maps are not specified.
    if (llvm::any_of(indexingMaps, [batchSize](AffineMap m) {
          for (unsigned i = 0; i < batchSize; ++i) {
            auto expr = dyn_cast<AffineDimExpr>(m.getResults()[i]);
            if (!expr || expr.getPosition() != i)
              return true;
          }
          return false;
        }))
      return failure();
  }

  auto a = matchOperandMap(indexingMaps[0], batchSize, dims.m[0], dims.k[0]);
  auto b = matchOperandMap(indexingMaps[1], batchSize, dims.k[0], dims.n[0]);
  auto c = matchOperandMap(indexingMaps[2], batchSize, dims.m[0], dims.n[0]);

  if (llvm::any_of(ArrayRef<IndexMatchResult>{a, b, c}, [](IndexMatchResult r) {
        return r == IndexMatchResult::Mismatch;
      }))
    return failure();

  if (c != IndexMatchResult::Match ||
      (a == IndexMatchResult::Transposed && b == IndexMatchResult::Transposed))
    return failure();

  /// Codegen the different matmul variants.
  if (batchSize) {
    if (a == IndexMatchResult::Transposed)
      return replaceWithMatmulVariant<BatchMatmulTransposeAOp>(rewriter,
                                                               genericOp);
    if (b == IndexMatchResult::Transposed)
      return replaceWithMatmulVariant<BatchMatmulTransposeBOp>(rewriter,
                                                               genericOp);
    return replaceWithMatmulVariant<BatchMatmulOp>(rewriter, genericOp);
  }

  if (a == IndexMatchResult::Transposed)
    return replaceWithMatmulVariant<MatmulTransposeAOp>(rewriter, genericOp);
  if (b == IndexMatchResult::Transposed)
    return replaceWithMatmulVariant<MatmulTransposeBOp>(rewriter, genericOp);
  return replaceWithMatmulVariant<MatmulOp>(rewriter, genericOp);
}

} // namespace

//===----------------------------------------------------------------------===//
// Categorize linalg generic to named op where possible.
//===----------------------------------------------------------------------===//
FailureOr<LinalgOp> mlir::linalg::specializeGenericOp(RewriterBase &rewriter,
                                                      GenericOp genericOp) {
  if (isaCopyOpInterface(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<CopyOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }

  if (isaFillOpInterface(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<FillOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }

  if (isaElemwiseSingleUnaryOpInterface(genericOp)) {
    Operation *op = &genericOp.getBody()->front();
    if (isa<math::ExpOp>(op)) {
      LinalgOp namedOp = REPLACE_UNARY_OP(ExpOp);
      return namedOp;
    }
  }

  if (isaElemwiseSingleBinaryOpInterface(genericOp)) {
    bool swap = areBinOpsSwapped(genericOp);
    Operation *op = &genericOp.getBody()->front();
    if (isa<arith::AddFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(AddOp, swap);
      return namedOp;
    }
    if (isa<arith::SubFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(SubOp, swap);
      return namedOp;
    }
    if (isa<arith::MulFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(MulOp, swap);
      return namedOp;
    }
    if (isa<arith::DivFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(DivOp, swap);
      return namedOp;
    }
  }

  if (isaContractionOpInterface(genericOp)) {
    return specializeLinalgContractions(rewriter, genericOp);
  }
  return failure();
}

namespace {
struct LinalgSpecializeGenericOpsPass
    : public impl::LinalgSpecializeGenericOpsPassBase<
          LinalgSpecializeGenericOpsPass> {

  using impl::LinalgSpecializeGenericOpsPassBase<
      LinalgSpecializeGenericOpsPass>::LinalgSpecializeGenericOpsPassBase;
  void runOnOperation() override;
};
} // namespace

void LinalgSpecializeGenericOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgGenericOpsSpecializationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void mlir::linalg::populateLinalgGenericOpsSpecializationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgSpecializationPattern>(patterns.getContext());
}
