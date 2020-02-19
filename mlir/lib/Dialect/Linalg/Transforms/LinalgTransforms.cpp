//===- LinalgTransforms.cpp - Linalg transformations as patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for transforming Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/LinalgTransforms.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/VectorOps/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

#define DEBUG_TYPE "linalg-transforms"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;
using llvm::SetVector;

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral mlir::linalg::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

LogicalResult mlir::linalg::tileLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    StringRef linalgMarker, ArrayRef<unsigned> permutation) {
  assert(permutation.empty() || permutation.size() == sizes.size());
  auto tileRes = tileLinalgOperation(rewriter, op, sizes, permutation);
  if (!tileRes)
    return failure();
  tileRes->op.setAttr(LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getStringAttr(linalgMarker));
  return success();
}

LogicalResult mlir::linalg::tileAndFuseLinalgOpAndSetMarker(
    PatternRewriter &rewriter, Operation *op, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> operandIndicesToFuse, StringRef linalgMarker) {
  auto tileRes = tileLinalgOperation(rewriter, op, sizes);
  if (!tileRes)
    return failure();
  tileRes->op.setAttr(LinalgTransforms::kLinalgTransformMarker,
                      rewriter.getStringAttr(linalgMarker));
  Aliases aliases;
  auto G = LinalgDependenceGraph::buildDependenceGraph(
      aliases, op->getParentOfType<FuncOp>());
  SmallVector<Operation *, 4> originalProducers;
  for (auto operandIdx : operandIndicesToFuse) {
    auto fusionRes = fuseProducerOf(rewriter, tileRes->op, operandIdx, G);
    if (!fusionRes) {
      // Linalg fusion requires tiled loops to even determine whether it is
      // possible to fuse. As a consequence, the pattern may fail even though a
      // tiled version of op has already been introduced.
      // So we need to remove the tiled version ourselves in case of failure.
      // Another possibility is to ensure the constraints on the pattern
      // guarantee that fusion will occur and just assert here. As we develop
      // more complex patterns we can choose what is best.
      rewriter.eraseOp(tileRes->loops[0]);
      return failure();
    }
    fusionRes->fusedProducer.setAttr(LinalgTransforms::kLinalgTransformMarker,
                                     rewriter.getStringAttr(linalgMarker));
    originalProducers.push_back(fusionRes->originalProducer);
  }

  // The originalProducers can now be safely erased. This is similar to
  // SSA-value use-def but in the world of buffer + structured ops.
  for (auto *originalProducer : originalProducers)
    rewriter.eraseOp(originalProducer);
  return success();
}

bool mlir::linalg::detail::isProducedByOpOfTypeImpl(
    Operation *consumerOp, Value consumedView,
    function_ref<bool(Operation *)> isaOpType) {
  LinalgOp consumer = dyn_cast<LinalgOp>(consumerOp);
  assert(consumer.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  if (!consumer)
    return false;

  auto maybeConsumerIndex = consumer.getIndexOfInput(consumedView);
  if (!maybeConsumerIndex)
    return false;

  Aliases aliases;
  auto G = LinalgDependenceGraph::buildDependenceGraph(
      aliases, consumer.getParentOfType<FuncOp>());
  for (auto dependence : G.getDependencesInto(
           consumer, LinalgDependenceGraph::DependenceType::RAW)) {
    auto producer = cast<LinalgOp>(dependence.dependentOpView.op);
    if (!isProducerLastWriteOfView(G, consumer, consumedView, producer))
      continue;
    if (isaOpType(dependence.dependentOpView.op))
      return true;
  }
  return false;
}

//============================================================================//
// Precondition and transformation for vectorization of Linalg generic ops.
//============================================================================//
static bool hasMultiplyAddBody(linalg::GenericOp op) {
  auto &r = op.region();
  if (r.empty())
    return false;
  if (r.getBlocks().size() != 1)
    return false;
  auto &ops = r.front().getOperations();
  if (ops.size() != 3)
    return false;

  using mlir::matchers::m_Val;
  auto a = m_Val(r.front().getArgument(0));
  auto b = m_Val(r.front().getArgument(1));
  auto c = m_Val(r.front().getArgument(2));
  // TODO(ntv) Update this detection once we have  matcher support for
  // specifying that any permutation of operands matches.
  auto pattern1 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(a, b), c));
  auto pattern2 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(a, b)));
  auto pattern3 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(b, a), c));
  auto pattern4 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(b, a)));
  return pattern1.match(&ops.back()) || pattern2.match(&ops.back()) ||
         pattern3.match(&ops.back()) || pattern4.match(&ops.back());
}

// TODO(ntv) should be Tablegen'd from a single source that generates the op
// itself.
static bool isMatmul(linalg::GenericOp genericOp) {
  auto *ctx = genericOp.getContext();
  auto m = getAffineDimExpr(0, ctx);
  auto n = getAffineDimExpr(1, ctx);
  auto k = getAffineDimExpr(2, ctx);
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {k, n}));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}));
  auto maps = ArrayAttr::get({mapA, mapB, mapC}, ctx);
  return genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
         genericOp.indexing_maps() == maps && hasMultiplyAddBody(genericOp);
}

// TODO(ntv, ataei): This is in fact much more general than just vectorization
// for matmul and fill ops.
LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  for (Value operand : linalgOp.getInputsAndOutputBuffers())
    if (!operand.getType().cast<ShapedType>().hasStaticShape())
      return failure();
  for (Type outputTensorType : linalgOp.getOutputTensorTypes())
    if (!outputTensorType.cast<ShapedType>().hasStaticShape())
      return failure();
  if (isa<linalg::MatmulOp>(op) || isa<linalg::FillOp>(op))
    return success();

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || !isMatmul(genericOp))
    return failure();

  // TODO(ntv): non-identity layout.
  auto isStaticMemRefWithIdentityLayout = [](Value v) {
    auto m = v.getType().dyn_cast<MemRefType>();
    if (!m || !m.hasStaticShape() || !m.getAffineMaps().empty())
      return false;
    return true;
  };
  if (!llvm::all_of(genericOp.getInputsAndOutputBuffers(),
                    isStaticMemRefWithIdentityLayout))
    return failure();
  return success();
}

SmallVector<Value, 0> mlir::linalg::vectorizeLinalgOp(PatternRewriter &rewriter,
                                                      Operation *op) {
  using vector_contract = edsc::intrinsics::ValueBuilder<vector::ContractionOp>;
  using vector_broadcast = edsc::intrinsics::ValueBuilder<vector::BroadcastOp>;
  using vector_type_cast = edsc::intrinsics::ValueBuilder<vector::TypeCastOp>;

  assert(succeeded(vectorizeLinalgOpPrecondition(op)) &&
         "DRR failure case must be a precondition");
  auto linalgOp = cast<linalg::LinalgOp>(op);
  assert(linalgOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  edsc::ScopedContext scope(rewriter, op->getLoc());

  if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    // Vectorize fill as a vector.broadcast.
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                         "]: Rewrite linalg.fill as vector.broadcast: "
                      << *op << ":\n");
    auto dstMemrefVec = vector_type_cast(fillOp.getOutputBuffer(0));
    auto dstVec = std_load(dstMemrefVec);
    auto resVec = vector_broadcast(dstVec, fillOp.value());
    std_store(resVec, dstMemrefVec);
  } else {
    // Vectorize other ops as vector contraction (currently only matmul).
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                         "]: Rewrite linalg op as vector.contract: "
                      << *op << ":\n");
    auto vA = std_load(vector_type_cast(linalgOp.getInput(0)));
    auto vB = std_load(vector_type_cast(linalgOp.getInput(1)));
    auto vectorMemRefC = vector_type_cast(linalgOp.getOutputBuffer(0));
    auto vC = std_load(vectorMemRefC);
    auto vRes = vector_contract(vA, vB, vC, linalgOp.indexing_maps(),
                                linalgOp.iterator_types());
    std_store(vRes, vectorMemRefC);
  }
  return {};
}

//============================================================================//
// Precondition and transformation for permutation of Linalg generic ops.
//============================================================================//
LogicalResult mlir::linalg::permuteGenericLinalgOpPrecondition(
    Operation *op, ArrayRef<unsigned> permutation) {
  if (permutation.empty())
    return failure();
  // Transformation applies to generic ops only.
  if (!isa<GenericOp>(op) && !isa<IndexedGenericOp>(op))
    return failure();
  LinalgOp linOp = cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linOp.hasBufferSemantics())
    return failure();
  return success();
}

SmallVector<Value, 0>
mlir::linalg::permuteGenericLinalgOp(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<unsigned> permutation,
                                     StringRef linalgMarker) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: Permute dims for linalg op: " << *op
                    << ":\n");

  assert(succeeded(permuteGenericLinalgOpPrecondition(op, permutation)) &&
         "DRR failure case must be a precondition");

  auto linOp = cast<LinalgOp>(op);
  auto permutationMap = inversePermutation(
      AffineMap::getPermutationMap(permutation, rewriter.getContext()));
  SmallVector<AffineMap, 4> newIndexingMap;
  auto indexingMaps = linOp.indexing_maps().getValue();
  for (unsigned i = 0, e = linOp.getNumInputsAndOutputs(); i != e; ++i) {
    AffineMap m = indexingMaps[i].cast<AffineMapAttr>().getValue().compose(
        permutationMap);
    newIndexingMap.push_back(m);
  }
  auto itTypes = linOp.iterator_types().getValue();
  SmallVector<Attribute, 4> itTypesVector;
  for (unsigned i = 0, e = itTypes.size(); i != e; ++i)
    itTypesVector.push_back(itTypes[i]);
  applyPermutationToVector(itTypesVector, permutation);
  op->setAttr(getIndexingMapsAttrName(),
              rewriter.getAffineMapArrayAttr(newIndexingMap));
  op->setAttr(getIteratorTypesAttrName(), rewriter.getArrayAttr(itTypesVector));
  op->setAttr(LinalgTransforms::kLinalgTransformMarker,
              rewriter.getStringAttr(linalgMarker));
  linOp.clone(rewriter, linOp.getLoc(), op->getOperands());
  return {};
}

//============================================================================//
// Precondition and transformation for Linalg subview promotion.
//============================================================================//
LogicalResult mlir::linalg::promoteSubviewsLinalgOpPrecondition(Operation *op) {
  LinalgOp linOp = dyn_cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linOp || !linOp.hasBufferSemantics())
    return failure();
  if (llvm::none_of(linOp.getInputsAndOutputBuffers(), [](Value v) {
        return isa_and_nonnull<SubViewOp>(v.getDefiningOp());
      }))
    return failure();
  return success();
}

SmallVector<Value, 0>
mlir::linalg::promoteSubviewsLinalgOp(PatternRewriter &rewriter,
                                      Operation *op) {
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE "]: Promote subviews for linalg op: "
                    << *op << ":\n");

  assert(succeeded(promoteSubviewsLinalgOpPrecondition(op)) &&
         "DRR failure case must be a precondition");

  LinalgOp linOp = cast<LinalgOp>(op);
  assert(linOp.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  SetVector<Value> subViews;
  for (auto it : linOp.getInputsAndOutputBuffers())
    if (auto sv = dyn_cast_or_null<SubViewOp>(it.getDefiningOp()))
      subViews.insert(sv);
  if (!subViews.empty()) {
    promoteSubViewOperands(rewriter, linOp, subViews);
    return {};
  }
  llvm_unreachable("DRR failure case must be a precondition");
}
