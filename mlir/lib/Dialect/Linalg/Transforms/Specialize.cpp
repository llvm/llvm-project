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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

// Checks whether the input Affine `map` contains two consecutive dims that
// can be interpreted as accessing a 2D matrix. It is assumed that the row
// column dimension are adjacent axis (in this order) and start at
// `rowDimIdx` in the input map.
//
//  e.g. consider A matrix in `C[M,N] = A[M,K] * B[K,N]`. We will check
//  whether the map of A is identity (match), transposed, or something
//  completely different (mis-match). Similar for B and C.
static IndexMatchResult matchOperandMap(AffineMap map, unsigned rowDimIdx,
                                        unsigned expectedPosOfRowDim,
                                        unsigned expectedPosOfColDim) {
  // Get the matrix multiply indices. They are past the batch indices.
  auto exprOfRowDim = map.getResults()[rowDimIdx];
  auto exprOfColDim = map.getResults()[rowDimIdx + 1];

  // They should be pure dimension ids.
  if (exprOfRowDim.getKind() != AffineExprKind::DimId ||
      exprOfColDim.getKind() != AffineExprKind::DimId)
    return IndexMatchResult::Mismatch;

  auto posRowDim = cast<AffineDimExpr>(exprOfRowDim).getPosition();
  auto posColDim = cast<AffineDimExpr>(exprOfColDim).getPosition();

  if (expectedPosOfRowDim == posRowDim && expectedPosOfColDim == posColDim)
    return IndexMatchResult::Match;

  if (expectedPosOfRowDim == posColDim && expectedPosOfColDim == posRowDim)
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

  // Early exit if not projected permutations.
  auto mapRange = genericOp.getIndexingMapsArray();
  if (llvm::any_of(mapRange,
                   [](AffineMap m) { return !m.isProjectedPermutation(); }))
    return failure();

  // Linalg generic contraction can be across multiple axis e.g.
  // ```
  //      linalg.generic
  //           {indexing_maps = [affine_map<(m, n, k1, k2) -> (m, k1, k2)>,
  //                             affine_map<(m, n, k1, k2) -> (k2, k1, n)>,
  //                             affine_map<(m, n, k1, k2) -> (m, n)>],
  //           iterator_types = ["parallel", "parallel",
  //                             "reduction", "reduction"]}
  //           ins(%A, %B : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
  //           outs(%C : tensor<10x40xf32>) {
  //           ^bb0(%a: f32, %b: f32, %c: f32):
  //                 %1 = arith.mulf %a, %b : f32
  //                 %2 = arith.addf %c, %1 : f32
  //                 linalg.yield %2 : f32
  //      } -> tensor<10x40xf32>
  //  ```
  //  In above contraction, there are two reduction dimensions {k1, k2}
  //  and although a valid linalg contraction, it is not a named-op
  //  matrix multiply kind. Therefore, reject multi-dim reduction.
  auto res = inferContractionDims(genericOp);
  if (!succeeded(res))
    return failure();
  auto dims = *res;
  if (dims.m.size() != 1 || dims.n.size() != 1 || dims.k.size() != 1)
    return failure();

  if (!mlir::linalg::detail::isContractionBody(
          *genericOp.getBlock(), [](Operation *first, Operation *second) {
            return (isa<arith::MulFOp>(first) && isa<arith::AddFOp>(second)) ||
                   (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second)) ||
                   (isa<complex::MulOp>(first) && isa<complex::AddOp>(second));
          }))
    return failure();

  // Check rank of operands
  auto indexingMaps = genericOp.getIndexingMapsArray();
  if (llvm::any_of(indexingMaps, [&dims](AffineMap m) {
        return m.getResults().size() !=
               dims.batch.size() + 2 /* any two of {m,n,k} */;
      }))
    return failure();

  auto numOfBatchDims = dims.batch.size();
  if (indexingMaps[0].getNumDims() != numOfBatchDims + 3)
    return failure();

  if (numOfBatchDims) {
    // Each operand in a linalg generic contraction  could express different
    // permutations for its batch dimension. But for named op it must be
    // identity since separate maps are not specified.
    if (llvm::any_of(indexingMaps, [numOfBatchDims](AffineMap m) {
          for (unsigned i = 0; i < numOfBatchDims; ++i) {
            auto expr = m.getResults()[i];
            if (expr.getKind() != AffineExprKind::DimId ||
                cast<AffineDimExpr>(expr).getPosition() != i)
              return true;
          }
          return false;
        }))
      return failure();
  }

  auto a =
      matchOperandMap(indexingMaps[0], numOfBatchDims, dims.m[0], dims.k[0]);
  auto b =
      matchOperandMap(indexingMaps[1], numOfBatchDims, dims.k[0], dims.n[0]);
  auto c =
      matchOperandMap(indexingMaps[2], numOfBatchDims, dims.m[0], dims.n[0]);

  if (llvm::is_contained({a, b, c}, IndexMatchResult::Mismatch))
    return failure();

  if (c != IndexMatchResult::Match ||
      (a == IndexMatchResult::Transposed && b == IndexMatchResult::Transposed))
    return failure();

  /// Codegen the different matmul variants.
  if (numOfBatchDims) {
    return replaceWithMatmulVariant<BatchMatmulOp>(rewriter, genericOp);
  }
  return replaceWithMatmulVariant<MatmulOp>(rewriter, genericOp);
}

/// Utility to match block body for linalg.pool* ops.
template <typename... OpTypes>
static bool bodyMatcherForPoolOps(Value yieldVal, Block *body) {
  Operation *defOp = yieldVal.getDefiningOp();
  // if (!defOp) return false;
  if (!(isa_and_present<OpTypes>(defOp) || ...)) return false;

  BlockArgument lhsArg =  dyn_cast<BlockArgument>(defOp->getOperand(0));
  BlockArgument rhsArg =  dyn_cast<BlockArgument>(defOp->getOperand(1));
  if (!lhsArg || !rhsArg) return false;
  return true;
}

static bool bodyMatcherForMaxSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaximumFOp, arith::MaxSIOp>(yieldVal, body);
}

static bool bodyMatcherForMaxUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MaximumFOp, arith::MaxUIOp>(yieldVal, body);
}

static bool bodyMatcherForMinSignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinimumFOp, arith::MinSIOp>(yieldVal, body);
}

static bool bodyMatcherForMinUnsignedPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::MinimumFOp, arith::MinUIOp>(yieldVal, body);
}

static bool bodyMatcherForSumPoolOps(Value yieldVal, Block *body) {
  return bodyMatcherForPoolOps<arith::AddIOp, arith::AddFOp>(yieldVal, body);
}

static mlir::AffineExpr getAffineMapDim(ArrayAttr indexingMaps,
                                        uint32_t mapIndex, uint32_t dimIndex) {
  auto affineMap = cast<AffineMapAttr>(indexingMaps[mapIndex]).getValue();
  if (dimIndex < affineMap.getNumResults())
    return affineMap.getResult(dimIndex);
  return nullptr;
}

// Check if `expr` is either:
// - a dimension expr alone (implying *1), or
// - a multiplication of dimension expr by constant.
bool isDimTimesConstantOrDimOnly(AffineExpr expr, AffineExpr &dim, int64_t &constantValue) {
  if (auto dExpr = dyn_cast<AffineDimExpr>(expr)) {
    dim = dExpr;
    constantValue = 1;
    return true;
  }

  auto mulExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  AffineExpr lhs = mulExpr.getLHS();
  AffineExpr rhs = mulExpr.getRHS();

  if (auto dExpr = dyn_cast<AffineDimExpr>(lhs)) {
    if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
      dim = dExpr;
      constantValue = cst.getValue();
      return true;
    }
  }
  if (auto cst = dyn_cast<AffineConstantExpr>(lhs)) {
    if (auto dExpr = dyn_cast<AffineDimExpr>(rhs)) {
      dim = dExpr;
      constantValue = cst.getValue();
      return true;
    }
  }
  return false;
}

bool matchConvDimAddExprPattern(ArrayAttr indexingMaps, unsigned iDim, unsigned fDim, unsigned oDim) {
  unsigned iIndex = 0, fIndex = 1, oIndex = indexingMaps.size() - 1;
  AffineExpr inpExpr = getAffineMapDim(indexingMaps, iIndex, iDim);
  auto addExpr = dyn_cast<AffineBinaryOpExpr>(inpExpr);
  if (!addExpr || addExpr.getKind() != AffineExprKind::Add)
    return false;

  AffineExpr dim0, dim1;
  // TODO(Abhishek-Varma): Use this information in specialize.cpp.
  int64_t c0, c1;

  if (isDimTimesConstantOrDimOnly(addExpr.getLHS(), dim0, c0) &&
      isDimTimesConstantOrDimOnly(addExpr.getRHS(), dim1, c1)) {
    // Pattern matched with dims and constants extracted.
    AffineExpr fExpr = getAffineMapDim(indexingMaps, fIndex, fDim);
    AffineExpr oExpr = getAffineMapDim(indexingMaps, oIndex, oDim);
    return ((dim0 == fExpr && dim1 == oExpr) || (dim1 == fExpr && dim0 == oExpr));
  }
  return false;
}

bool matchConvDimExprPattern(ArrayAttr indexingMaps, unsigned aIndex, unsigned aDim, unsigned bIndex, unsigned bDim) {
  return getAffineMapDim(indexingMaps, aIndex, aDim) == getAffineMapDim(indexingMaps, bIndex, bDim);
}

static std::string inferBasedOnRank2ConvIteratorTypes(GenericOp genericOp) {
  if (isaConv1DOp(genericOp)) return "linalg.conv_1d";
  return "";
}

static std::string inferBasedOnRank4ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return "";
  // depthwise_conv_1d_ncw_cw
  // #map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1 + d3)>
  // #map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
  // #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
  if (isaDepthwiseConv1DNcwCwOp(genericOp))
    return "linalg.depthwise_conv_1d_ncw_cw";
  // depthwise_conv_1d_nwc_wc
  // #map = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d3, d2)>
  // #map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  // #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  if (isaDepthwiseConv1DNwcWcOp(genericOp))
    return "linalg.depthwise_conv_1d_nwc_wc";
  // conv_2d
  // #map = affine_map<(d0, d1, d2, d3) -> (d0 + d2, d1 + d3)>
  // #map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
  // #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
  if (isaConv2DOp(genericOp))
    return "linalg.conv_2d";
  
  unsigned iIndex = 0, fIndex = 1, oIndex = indexingMaps.size() - 1;
  Block *body = genericOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  // pooling_ncw_max
  // pooling_ncw_sum
  // #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + d3)>
  // #map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
  // #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 1, oIndex, 1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/0, /*oDim=*/2)) {
    if (bodyMatcherForMaxSignedPoolOps(yieldVal, body))
      return "linalg.pooling_ncw_max";
    if (bodyMatcherForSumPoolOps(yieldVal, body))
      return "linalg.pooling_ncw_sum";
  }
  // pooling_nwc_max
  // pooling_nwc_min
  // pooling_nwc_sum
  // #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1 + d3, d2)>
  // #map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
  // #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 2, oIndex, 2)) {
    if (bodyMatcherForMaxSignedPoolOps(yieldVal, body))
      return "linalg.pooling_nwc_max";
    if (bodyMatcherForMinSignedPoolOps(yieldVal, body))
      return "linalg.pooling_nwc_min";
    if (bodyMatcherForSumPoolOps(yieldVal, body))
      return "linalg.pooling_nwc_sum";
  }
  return "";
}

static std::string inferBasedOnRank5ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return "";
  // depthwise_conv_1d_nwc_wcm
  // #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d4, d2)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d2, d3)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
  if (isaDepthwiseConv1DNwcWcmOp(genericOp))
    return "linalg.depthwise_conv_1d_nwc_wcm";
  // conv_1d_nwc_wcf
  // #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
  if (isaConv1DNwcWcfOp(genericOp))
    return "linalg.conv_1d_nwc_wcf";
  // conv_1d_ncw_fcw
  // #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2 + d4)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
  if (isaConv1DNcwFcwOp(genericOp))
    return "linalg.conv_1d_ncw_fcw";
  return "";
}

static std::string inferBasedOnRank6ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() < 3) return "";
  unsigned iIndex = 0, fIndex = 1, oIndex = indexingMaps.size() - 1;
  // depthwise_conv_2d_nchw_chw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 + d4, d2 + d5)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1, d2)>
  if (isaDepthwiseConv2DNchwChwOp(genericOp))
    return "linalg.depthwise_conv_2d_nchw_chw";
  // depthwise_conv_2d_nhwc_hwc
  // #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d3)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
  if (isaDepthwiseConv2DNhwcHwcOp(genericOp))
    return "linalg.depthwise_conv_2d_nhwc_hwc";
  // conv_3d
  // #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 + d3, d1 + d4, d2 + d5)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
  if (matchConvDimAddExprPattern(indexingMaps, /*iDim=*/0, /*fDim=*/0, /*oDim=*/0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/1, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/2, /*oDim=*/2))
    return "linalg.conv_3d";

  Block *body = genericOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  // pooling_nchw_max
  // pooling_nchw_sum
  // #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d4, d3 + d5)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 1, oIndex, 1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/0, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/1, /*oDim=*/3)) {
    if (bodyMatcherForMaxSignedPoolOps(yieldVal, body))
      return "linalg.pooling_nchw_max";
    if (bodyMatcherForSumPoolOps(yieldVal, body))
      return "linalg.pooling_nchw_sum";
  }
  // pooling_nhwc_max
  // pooling_nhwc_min
  // pooling_nhwc_sum
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
  // #map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 3, oIndex, 3)) {
    if (bodyMatcherForMaxSignedPoolOps(yieldVal, body))
      return "linalg.pooling_nhwc_max";
    if (bodyMatcherForMinSignedPoolOps(yieldVal, body))
      return "linalg.pooling_nhwc_min";
    if (bodyMatcherForSumPoolOps(yieldVal, body))
      return "linalg.pooling_nhwc_sum";
  }
  // pooling_nhwc_max_unsigned
  // pooling_nhwc_min_unsigned
  // #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 3, oIndex, 3)) {
    if (bodyMatcherForMaxUnsignedPoolOps(yieldVal, body))
      return "linalg.pooling_nhwc_max_unsigned";
    if (bodyMatcherForMinUnsignedPoolOps(yieldVal, body))
      return "linalg.pooling_nhwc_min_unsigned";
  }
  return "";
}

static std::string inferBasedOnRank7ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() < 3) return "";
  // conv_2d_nhwc_fhwc
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
  if (isaConv2DNhwcFhwcOp(genericOp))
    return "linalg.conv_2d_nhwc_fhwc";
  // conv_2d_nhwc_hwcf
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
  if (isaConv2DNhwcHwcfOp(genericOp))
    return "linalg.conv_2d_nhwc_hwcf";
  // conv_2d_nchw_fchw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
  if (isaConv2DNchwFchwOp(genericOp))
    return "linalg.conv_2d_nchw_fchw";
  // conv_2d_nhwc_fhwc_q (same as conv_2d_nhwc_fhwc + check total 4 indexing maps)
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
  if (isaConv2DNhwcFhwcQOp(genericOp))
    return "linalg.conv_2d_nhwc_fhwc_q";
  // conv_2d_nchw_fchw_q (same as conv_2d_nchw_fchw + check total 4 indexing maps)
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
  if (isaConv2DNchwFchwQOp(genericOp))
    return "linalg.conv_2d_nchw_fchw_q";
  // depthwise_conv_2d_nhwc_hwcm
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
  if (isaDepthwiseConv2DNhwcHwcmOp(genericOp))
    return "linalg.depthwise_conv_2d_nhwc_hwcm";
  // depthwise_conv_2d_nhwc_hwcm_q
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
  if (isaDepthwiseConv2DNhwcHwcmQOp(genericOp))
    return "linalg.depthwise_conv_2d_nhwc_hwcm_q";
  return "";
}

static std::string inferBasedOnRank8ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() < 3) return "";
  unsigned iIndex = 0, fIndex = 1, oIndex = indexingMaps.size() - 1;
  // conv_2d_ngchw_fgchw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d1, d5, d6, d7)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  if (isaConv2DNgchwFgchwOp(genericOp))
    return "linalg.conv_2d_ngchw_fgchw";
  // conv_2d_ngchw_gfchw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  if (isaConv2DNgchwGfchwOp(genericOp))
    return "linalg.conv_2d_ngchw_gfchw";
  // conv_2d_ngchw_gfchw_q
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> ()>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  if (isaConv2DNgchwGfchwQOp(genericOp))
    return "linalg.conv_2d_ngchw_gfchw_q";
  // conv_2d_nhwgc_gfhwc
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  if (isaConv2DNhwgcGfhwcOp(genericOp))
    return "linalg.conv_2d_nhwgc_gfhwc";
  // depthwise_conv_3d_ncdhw_cdhw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d1 + d4, d2 + d5, d3 + d6)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d4, d5, d6)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d1, d2, d3)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 1, fIndex, 0) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 1, oIndex, 1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2, /*oDim=*/3) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/4, /*fDim=*/3, /*oDim=*/4))
    return "linalg.depthwise_conv_3d_ncdhw_cdhw";
  // depthwise_conv_3d_ndhwc_dhwc
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d4, d2 + d5, d3 + d6, d7)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d7)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2, /*oDim=*/3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, fIndex, 3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, oIndex, 4))
    return "linalg.depthwise_conv_3d_ndhwc_dhwc";

  Block *body = genericOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  // pooling_ndhwc_max
  // pooling_ndhwc_min
  // pooling_ndhwc_sum
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3 + d7, d4)>
  // #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>
  // #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2, /*oDim=*/3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, oIndex, 4)) {
    if (bodyMatcherForMaxSignedPoolOps(yieldVal, body))
      return "linalg.pooling_ndhwc_max";
    if (bodyMatcherForMinSignedPoolOps(yieldVal, body))
      return "linalg.pooling_ndhwc_min";
    if (bodyMatcherForSumPoolOps(yieldVal, body))
      return "linalg.pooling_ndhwc_sum";
  }
  return "";
}

static std::string inferBasedOnRank9ConvIteratorTypes(GenericOp genericOp) {
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() < 3) return "";
  unsigned iIndex = 0, fIndex = 1, oIndex = indexingMaps.size() - 1;
  // conv_3d_ncdhw_fcdhw
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d4 + d8)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 1, fIndex, 1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/2, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/3, /*oDim=*/3) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/4, /*fDim=*/4, /*oDim=*/4) &&
      matchConvDimExprPattern(indexingMaps, fIndex, 0, oIndex, 1))
    return "linalg.conv_3d_ncdhw_fcdhw";
  // conv_3d_ndhwc_dhwcf
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 + d5, d2 + d6, d3 + d7, d8)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5, d6, d7, d8, d4)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2, /*oDim=*/3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, fIndex, 3) &&
      matchConvDimExprPattern(indexingMaps, fIndex, 4, oIndex, 4))
    return "linalg.conv_3d_ndhwc_dhwcf";
  // depthwise_conv_3d_ndhwc_dhwcm
  // #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 + d5, d2 + d6, d3 + d7, d8)>
  // #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5, d6, d7, d8, d4)>
  // #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d8, d4)>
  if (matchConvDimExprPattern(indexingMaps, iIndex, 0, oIndex, 0) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/1, /*fDim=*/0, /*oDim=*/1) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/2, /*fDim=*/1, /*oDim=*/2) &&
      matchConvDimAddExprPattern(indexingMaps, /*iDim=*/3, /*fDim=*/2, /*oDim=*/3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, fIndex, 3) &&
      matchConvDimExprPattern(indexingMaps, iIndex, 4, oIndex, 4) &&
      matchConvDimExprPattern(indexingMaps, fIndex, 4, oIndex, 5))
    return "linalg.depthwise_conv_3d_ndhwc_dhwcm";
  return "";
}

static std::string inferConvolutionKind(GenericOp genericOp) {
  SmallVector<utils::IteratorType> iteratorTypes = genericOp.getIteratorTypesArray();
  unsigned totalIterators = iteratorTypes.size();
  switch(totalIterators) {
    case 2:
      return inferBasedOnRank2ConvIteratorTypes(genericOp);
    case 4:
      return inferBasedOnRank4ConvIteratorTypes(genericOp);
    case 5:
      return inferBasedOnRank5ConvIteratorTypes(genericOp);
    case 6:
      return inferBasedOnRank6ConvIteratorTypes(genericOp);
    case 7:
      return inferBasedOnRank7ConvIteratorTypes(genericOp);
    case 8:
      return inferBasedOnRank8ConvIteratorTypes(genericOp);
    case 9:
      return inferBasedOnRank9ConvIteratorTypes(genericOp);
  }
  return "";
}

// Converts linalg.generic to named linalg.*conv* where possible.
static FailureOr<LinalgOp> specializeLinalgConvolutions(RewriterBase &rewriter,
                                                        GenericOp genericOp) {
  std::string convKind = inferConvolutionKind(genericOp);
  if (convKind == "") return failure();
  SmallVector<Value> inputs = genericOp.getDpsInputs();
  ValueRange outputs = genericOp.getDpsInits();
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  SmallVector<Type> resultTypes = genericOp.hasPureTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};
  LinalgOp namedOp;
  if (convKind == "linalg.conv_1d") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv1DOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_1d_nwc_wcf") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv1DNwcWcfOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_1d_ncw_fcw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv1DNcwFcwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_1d_ncw_cw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv1DNcwCwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_1d_nwc_wc") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv1DNwcWcOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_1d_nwc_wcm") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv1DNwcWcmOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nhwc_fhwc") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcFhwcOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nhwc_hwcf") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcHwcfOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nchw_fchw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nhwc_fhwc_q") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNhwcFhwcQOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nchw_fchw_q") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNchwFchwQOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_ngchw_fgchw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNgchwFgchwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_ngchw_gfchw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNgchwGfchwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_ngchw_gfchw_q") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNgchwGfchwQOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_2d_nhwgc_gfhwc") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv2DNhwgcGfhwcOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_2d_nchw_chw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNchwChwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_2d_nhwc_hwc") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_2d_nhwc_hwcm") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcmOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_2d_nhwc_hwcm_q") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNhwcHwcmQOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_3d") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv3DOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_3d_ncdhw_fcdhw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv3DNcdhwFcdhwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.conv_3d_ndhwc_dhwcf") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::Conv3DNdhwcDhwcfOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_3d_ndhwc_dhwcm") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv3DNdhwcDhwcmOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_3d_ncdhw_cdhw") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv3DNcdhwCdhwOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.depthwise_conv_3d_ndhwc_dhwc") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::DepthwiseConv3DNdhwcDhwcOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nchw_max") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNchwMaxOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nchw_sum") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNchwSumOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nhwc_max") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nhwc_min") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMinOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nhwc_sum") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNhwcSumOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nhwc_max_unsigned") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMaxUnsignedOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nhwc_min_unsigned") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNhwcMinUnsignedOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_ncw_max") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNcwMaxOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_ncw_sum") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNcwSumOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nwc_max") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNwcMaxOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nwc_min") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNwcMinOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_nwc_sum") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNwcSumOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_ndhwc_max") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNdhwcMaxOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_ndhwc_min") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNdhwcMinOp>(genericOp, resultTypes, inputs, outputs);
  } else if (convKind == "linalg.pooling_ndhwc_sum") {
    namedOp = rewriter.replaceOpWithNewOp<linalg::PoolingNdhwcSumOp>(genericOp, resultTypes, inputs, outputs);
  }
  return namedOp;

  return failure();
}

} // namespace

//===----------------------------------------------------------------------===//
// Categorize linalg generic to named op where possible.
//===----------------------------------------------------------------------===//
FailureOr<LinalgOp> mlir::linalg::specializeGenericOp(RewriterBase &rewriter,
                                                      GenericOp genericOp) {
  // Copy
  if (isaCopyOpInterface(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<CopyOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }

  // Fill
  if (std::optional<Value> fillValue = isaFillOpInterface(genericOp)) {
    // Always use the detected fill value, regardless of pattern
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<FillOp>(
        genericOp, *fillValue, genericOp.getDpsInits()[0]);
    return namedOp;
  }

  // Broadcast
  std::optional<SmallVector<int64_t>> equivalentToBroadcast =
      isaBroadcastOpInterface(genericOp);
  if (equivalentToBroadcast) {
    auto dims = *equivalentToBroadcast;
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<BroadcastOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0],
        dims);
    return namedOp;
  }

  // Transpose
  std::optional<SmallVector<int64_t>> equivalentToTranspose =
      isaTransposeOpInterface(genericOp);
  if (equivalentToTranspose) {
    auto permutation = *equivalentToTranspose;
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<TransposeOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0],
        permutation);
    return namedOp;
  }

  // Elementwise Unary
  if (isaElemwiseSingleUnaryOpInterface(genericOp)) {
    Operation *op = &genericOp.getBody()->front();
    if (isa<math::ExpOp>(op)) {
      LinalgOp namedOp = REPLACE_UNARY_OP(ExpOp);
      return namedOp;
    }
  }

  // Elementwise Binary
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

  // Contraction - e.g. matmul
  if (isaContractionOpInterface(genericOp)) {
    return specializeLinalgContractions(rewriter, genericOp);
  }

  // Convolution - e.g. *conv*
  if (isaConvolutionOpInterface(genericOp)) {
    return specializeLinalgConvolutions(rewriter, genericOp);
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
  populateDecomposeProjectedPermutationPatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void mlir::linalg::populateLinalgGenericOpsSpecializationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgSpecializationPattern>(patterns.getContext());
}
