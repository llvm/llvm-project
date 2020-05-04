//===- Vectorization.cpp - Implementation of linalg Vectorization ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Vectorization transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;

using llvm::dbgs;

#define DEBUG_TYPE "linalg-vectorization"

static bool hasMultiplyAddBody(linalg::GenericOp op) {
  auto &r = op.region();
  if (!llvm::hasSingleElement(r))
    return false;
  if (!llvm::hasNItems(r.front().begin(), r.front().end(), 3))
    return false;

  using mlir::matchers::m_Val;
  auto a = m_Val(r.front().getArgument(0));
  auto b = m_Val(r.front().getArgument(1));
  auto c = m_Val(r.front().getArgument(2));
  // TODO: Update this detection once we have  matcher support for specifying
  // that any permutation of operands matches.
  auto pattern1 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(a, b), c));
  auto pattern2 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(a, b)));
  auto pattern3 = m_Op<YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(b, a), c));
  auto pattern4 = m_Op<YieldOp>(m_Op<AddFOp>(c, m_Op<MulFOp>(b, a)));
  return pattern1.match(&r.front().back()) ||
         pattern2.match(&r.front().back()) ||
         pattern3.match(&r.front().back()) || pattern4.match(&r.front().back());
}

// TODO: Should be Tablegen'd from a single source that generates the op itself.
static bool isRowMajorMatmul(linalg::GenericOp genericOp) {
  return genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
         isRowMajorMatmul(genericOp.indexing_maps()) &&
         hasMultiplyAddBody(genericOp);
}

// TODO: This is in fact much more general than just vectorization for matmul
// and fill ops.
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
  if (!genericOp || !::isRowMajorMatmul(genericOp))
    return failure();

  // TODO(ntv): non-identity layout.
  auto isStaticMemRefWithIdentityLayout = [](Value v) {
    auto m = v.getType().dyn_cast<MemRefType>();
    if (!m || !m.hasStaticShape() || !m.getAffineMaps().empty())
      return false;
    return true;
  };
  return success(llvm::all_of(genericOp.getInputsAndOutputBuffers(),
                              isStaticMemRefWithIdentityLayout));
}

void mlir::linalg::vectorizeLinalgOp(OpBuilder &builder, Operation *op) {
  assert(succeeded(vectorizeLinalgOpPrecondition(op)));

  if (auto convOp = dyn_cast<linalg::ConvOp>(op)) {
    // TODO: add a level of indirection to linalg.generic.
    if (convOp.padding())
      llvm_unreachable("Unexpected conv with padding");
  }

  edsc::ScopedContext scope(builder, op->getLoc());
  if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    // Vectorize fill as a vector.broadcast.
    LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                         "]: Rewrite linalg.fill as vector.broadcast: "
                      << *op << ":\n");
    Value memref = vector_type_cast(fillOp.getOutputBuffer(0));
    Value dst = std_load(memref);
    Value res = vector_broadcast(dst.getType(), fillOp.value());
    std_store(res, memref);
    return;
  }

  // Vectorize other ops as vector contraction (currently only matmul).
  LLVM_DEBUG(dbgs() << "\n[" DEBUG_TYPE
                       "]: Rewrite linalg op as vector.contract: "
                    << *op << ":\n");
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Value a = std_load(vector_type_cast(linalgOp.getInput(0)));
  Value b = std_load(vector_type_cast(linalgOp.getInput(1)));
  Value memref = vector_type_cast(linalgOp.getOutputBuffer(0));
  Value c = std_load(memref);
  Value res = vector_contract(a, b, c, linalgOp.indexing_maps(),
                              linalgOp.iterator_types());
  std_store(res, memref);
}
