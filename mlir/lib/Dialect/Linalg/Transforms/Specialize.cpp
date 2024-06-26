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
// It is possible that a linalg.generic may be implementing a matmul but not
// in a straight-forward way e.g. below is matrix multiply over some slice
// ```
//  %0 = linalg.generic {
//          indexing_maps = [affine_map<(d0, d1, d2) -> (3, d1, d0)>,
//                           affine_map<(d0, d1, d2) -> (d0, 5, d2)>,
//                           affine_map<(d0, d1, d2) -> (d2, d1, 13)>],
//          iterator_types = ["parallel", "parallel", "parallel"]}
//          ins(%A, %B : tensor<20x20x20xf32>,  tensor<20x20x20xf32>)
//          outs(%C : tensor<20x20x20xf32>) {
//             ^bb0(%a: f32, %b: f32, %c : f32):
//                %mul = arith.mulf %a, %b : f32
//                %add = arith.addf %mul, %c : f32
//                linalg.yield %add : f32
//       } -> tensor<20x20x20xf32>
// ```
// It is not possible to represent above as named op.
// e.g. linalg.batch_matmul(%A, %B :  tensor<20x20x20xf32>, ...) is
// not  the same as linalg.generic above.
namespace {
enum class IndexMatchResult {
  Match = 0,  // identity map.
  Transposed, // transposed map.
  Mismatch    // none of the above.
};

// Matches position of indices appearing the affine map of operand
// with what is expected in non-transposed case. e.g.
//  consider the A matrix in `C[M,N] = A[M,K] * B[K,N]`. Below, we
//  check whether the index map of A is identity (match), transposed, or
//  something completely different (mis-match).
// The naming and explanation is in terms of A, but the function checks
// effectively maps for all A, B, C i.e. C<M,N>, A<M, K>, B<K,N>.
static IndexMatchResult matchOperandMap(AffineMap map, unsigned batchSize,
                                        unsigned expectedPosOfM,
                                        unsigned expectedPosOfK) {
  // Get the matrix multiply indices. They are past the batch indices.
  auto exprOfM = map.getResults()[batchSize];
  auto exprOfK = map.getResults()[batchSize + 1];

  // They should be pure dim ids.
  if (exprOfM.getKind() != AffineExprKind::DimId ||
      exprOfK.getKind() != AffineExprKind::DimId)
    return IndexMatchResult::Mismatch;

  auto posM = cast<AffineDimExpr>(exprOfM).getPosition();
  auto posK = cast<AffineDimExpr>(exprOfK).getPosition();

  if (expectedPosOfM == posM && expectedPosOfK == posK)
    return IndexMatchResult::Match;

  if (expectedPosOfM == posK && expectedPosOfK == posM)
    return IndexMatchResult::Transposed;

  return IndexMatchResult::Mismatch;
}

// Replaces genericOp with `NamedOpTy` op, supplied as a template arg.
//  All the variants expressed as pseudo regular expression:
//      `linalg.{batch_}?matmul{_transpose_a | _transpose_b}?`
//  have same number of ins/out, so its easy to stamp different versions.
template <typename NamedOpTy>
static LinalgOp replaceWithMatmulVariant(RewriterBase &rewriter, GenericOp op) {
  LinalgOp namedOp = rewriter.replaceOpWithNewOp<NamedOpTy>(
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

  if (!mlir::linalg::detail::isContractionBody(
          *genericOp.getBlock(), [](Operation *first, Operation *second) {
            if ((isa<arith::MulFOp>(first) && isa<arith::AddFOp>(second)) ||
                (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second)) ||
                (isa<complex::MulOp>(first) && isa<complex::AddOp>(second)))
              return true;
            return false;
          }))
    return failure();

  auto res = inferContractionDims(genericOp);
  assert(succeeded(res) && "unexpected failure to infer contraction dims");
  auto dims = *res;

  // Other than `batch`, other dim sizes must be 1 for linalg.*_matmul_*.
  // Note that linalg contraction can have more than one contraction dimension.
  if (dims.m.size() != 1 || dims.n.size() != 1 || dims.k.size() != 1)
    return failure();

  // Check rank of operands
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (llvm::any_of(indexingMaps, [&dims](AffineMap m) {
        return m.getResults().size() !=
               dims.batch.size() + 2 /* any two of {m,n,k} */;
      }))
    return failure();

  auto batchSize = dims.batch.size();
  if (indexingMaps[0].getNumDims() != batchSize + 3)
    return failure();

  if (batchSize) {
    // Each operand in a linalg generic contraction  could express different
    // permutations for its batch dimension. But for named op it must be
    // identity since separate maps are not specified.
    if (llvm::any_of(indexingMaps, [batchSize](AffineMap m) {
          for (unsigned i = 0; i < batchSize; ++i) {
            auto expr = m.getResults()[i];
            if (expr.getKind() != AffineExprKind::DimId ||
                cast<AffineDimExpr>(expr).getPosition() != i)
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

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void mlir::linalg::populateLinalgGenericOpsSpecializationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgSpecializationPattern>(patterns.getContext());
}
