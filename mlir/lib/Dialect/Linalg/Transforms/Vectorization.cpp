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

static bool hasMultiplyAddBody(Region &r) {
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
  auto pattern5 = m_Op<YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(a, b), c));
  auto pattern6 = m_Op<YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(a, b)));
  auto pattern7 = m_Op<YieldOp>(m_Op<AddIOp>(m_Op<MulIOp>(b, a), c));
  auto pattern8 = m_Op<YieldOp>(m_Op<AddIOp>(c, m_Op<MulIOp>(b, a)));
  return pattern1.match(&r.front().back()) ||
         pattern2.match(&r.front().back()) ||
         pattern3.match(&r.front().back()) ||
         pattern4.match(&r.front().back()) ||
         pattern5.match(&r.front().back()) ||
         pattern6.match(&r.front().back()) ||
         pattern7.match(&r.front().back()) || pattern8.match(&r.front().back());
}

// TODO: Should be Tablegen'd from a single source that generates the op itself.
static LogicalResult isContraction(Operation *op) {
  // TODO: interface for named ops.
  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::MatvecOp,
          linalg::DotOp>(op))
    return success();

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return failure();

  auto mapRange =
      genericOp.indexing_maps().getAsRange<AffineMapAttr, AffineMap>();

  return success(
      genericOp.getNumInputs() == 2 && genericOp.getNumOutputs() == 1 &&
      llvm::all_of(mapRange,
                   [](AffineMap m) { return m.isProjectedPermutation(); }) &&
      hasMultiplyAddBody(genericOp.region()));
}

LogicalResult mlir::linalg::vectorizeLinalgOpPrecondition(Operation *op) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  // All types must be static shape to go to vector.
  for (Value operand : linalgOp.getInputsAndOutputBuffers())
    if (!operand.getType().cast<ShapedType>().hasStaticShape())
      return failure();
  for (Type outputTensorType : linalgOp.getOutputTensorTypes())
    if (!outputTensorType.cast<ShapedType>().hasStaticShape())
      return failure();

  if (isa<linalg::FillOp, linalg::CopyOp>(op))
    return success();

  return isContraction(op);
}

void mlir::linalg::vectorizeLinalgOp(OpBuilder &builder, Operation *op) {
  assert(succeeded(vectorizeLinalgOpPrecondition(op)));

  StringRef dbgPref = "\n[" DEBUG_TYPE "]: ";
  (void)dbgPref;
  edsc::ScopedContext scope(builder, op->getLoc());
  if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    // Vectorize fill as a vector.broadcast.
    LLVM_DEBUG(dbgs() << dbgPref
                      << "Rewrite linalg.fill as vector.broadcast: " << *op);
    Value memref = vector_type_cast(fillOp.getOutputBuffer(0));
    Value dst = std_load(memref);
    Value res = vector_broadcast(dst.getType(), fillOp.value());
    std_store(res, memref);
    return;
  }

  // In the case of 0-D memrefs, return null and special case to scalar load or
  // store later.
  auto extractVectorTypeFromScalarView = [](Value v) {
    MemRefType mt = v.getType().cast<MemRefType>();
    return mt.getShape().empty()
               ? VectorType()
               : VectorType::get(mt.getShape(), mt.getElementType());
  };

  if (auto copyOp = dyn_cast<linalg::CopyOp>(op)) {
    // Vectorize copy as a vector.transfer_read+vector.transfer_write.
    LLVM_DEBUG(dbgs() << dbgPref
                      << "Rewrite linalg.copy as vector.transfer_read + "
                         "vector.transfer_write: "
                      << *op);
    Value zero = std_constant_index(0);
    Value viewInput = copyOp.input();
    Value viewOutput = copyOp.output();
    Value vector;
    if (VectorType inputType = extractVectorTypeFromScalarView(viewInput)) {
      SmallVector<Value, 4> indicesInput(inputType.getRank(), zero);
      if (copyOp.inputPermutation())
        vector = vector_transfer_read(
            extractVectorTypeFromScalarView(viewInput), viewInput, indicesInput,
            copyOp.inputPermutation().getValue());
      else
        vector =
            vector_transfer_read(extractVectorTypeFromScalarView(viewInput),
                                 viewInput, indicesInput);
    } else {
      vector = std_load(viewInput).value;
    }
    if (VectorType outputType = extractVectorTypeFromScalarView(viewOutput)) {
      SmallVector<Value, 4> indicesOutput(outputType.getRank(), zero);
      if (copyOp.outputPermutation())
        vector_transfer_write(vector, viewOutput, indicesOutput,
                              copyOp.outputPermutation().getValue());
      else
        vector_transfer_write(vector, viewOutput, indicesOutput);
    } else {
      std_store(vector, viewOutput);
    }
    return;
  }

  assert(succeeded(isContraction(op)) && "Expected contraction");

  // Vectorize other ops as vector contraction.
  // TODO: interface.
  LLVM_DEBUG(dbgs() << dbgPref
                    << "Rewrite linalg op as vector.contract: " << *op);
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Value viewA = linalgOp.getInput(0);
  Value viewB = linalgOp.getInput(1);
  Value viewC = linalgOp.getOutputBuffer(0);
  VectorType vtA = extractVectorTypeFromScalarView(viewA);
  VectorType vtB = extractVectorTypeFromScalarView(viewB);
  VectorType vtC = extractVectorTypeFromScalarView(viewC);
  Value zero = std_constant_index(0);
  SmallVector<Value, 4> indicesA, indicesB, indicesC;
  if (vtA)
    indicesA = SmallVector<Value, 4>(vtA.getRank(), zero);
  if (vtB)
    indicesB = SmallVector<Value, 4>(vtB.getRank(), zero);
  if (vtC)
    indicesC = SmallVector<Value, 4>(vtC.getRank(), zero);
  Value a = vtA ? vector_transfer_read(vtA, viewA, indicesA).value
                : std_load(viewA, indicesA).value;
  Value b = vtB ? vector_transfer_read(vtB, viewB, indicesB).value
                : std_load(viewB, indicesB).value;
  Value c = vtC ? vector_transfer_read(vtC, viewC, indicesC).value
                : std_load(viewC, indicesC).value;
  Value res = vector_contract(a, b, c, linalgOp.indexing_maps(),
                              linalgOp.iterator_types());
  if (vtC)
    vector_transfer_write(res, viewC, indicesC);
  else
    std_store(res, viewC, indicesC);
}

/// Check whether there is any interleaved use of any `values` between `firstOp`
/// and `secondOp`. Conservatively return `true` if any op or value is in a
/// different block.
static bool mayExistInterleavedUses(Operation *firstOp, Operation *secondOp,
                                    ValueRange values) {
  StringRef dbgPref = "\n[" DEBUG_TYPE "]: ";
  (void)dbgPref;
  if (firstOp->getBlock() != secondOp->getBlock() ||
      !firstOp->isBeforeInBlock(secondOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << dbgPref << "interleavedUses precondition failed, firstOp: "
               << *firstOp << ", second op: " << *secondOp);
    return true;
  }
  for (auto v : values) {
    for (auto &u : v.getUses()) {
      Operation *owner = u.getOwner();
      if (owner == firstOp || owner == secondOp)
        continue;
      // TODO: this is too conservative, use dominance info in the future.
      if (owner->getBlock() == firstOp->getBlock() &&
          (owner->isBeforeInBlock(firstOp) || secondOp->isBeforeInBlock(owner)))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << dbgPref << " found interleaved op " << *owner
                 << ", firstOp: " << *firstOp << ", second op: " << *secondOp);
      return true;
    }
  }
  return false;
}

/// Return the unique subview use of `v` if it is indeed unique, null otherwise.
static SubViewOp getSubViewUseIfUnique(Value v) {
  SubViewOp subViewOp;
  for (auto &u : v.getUses()) {
    if (auto newSubViewOp = dyn_cast<SubViewOp>(u.getOwner())) {
      if (subViewOp)
        return SubViewOp();
      subViewOp = newSubViewOp;
    }
  }
  return subViewOp;
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTRForwardingPattern::matchAndRewrite(
    vector::TransferReadOp xferOp, PatternRewriter &rewriter) const {

  // Transfer into `view`.
  Value viewOrAlloc = xferOp.memref();
  if (!viewOrAlloc.getDefiningOp<ViewOp>() &&
      !viewOrAlloc.getDefiningOp<AllocOp>())
    return failure();

  StringRef dbgPref = "\n[" DEBUG_TYPE "]: VTRForwarding: ";
  (void)dbgPref;
  LLVM_DEBUG(llvm::dbgs() << dbgPref << viewOrAlloc);

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();
  LLVM_DEBUG(llvm::dbgs() << dbgPref << "with subView " << subView);

  // Find the copy into `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subView.getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getOutputBuffer(0) != subView)
        continue;
      LLVM_DEBUG(llvm::dbgs() << dbgPref << "copy candidate " << *newCopyOp);
      if (mayExistInterleavedUses(newCopyOp, xferOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << dbgPref << "with copy " << *copyOp);

  // Find the fill into `viewOrAlloc` without interleaved uses before the copy.
  FillOp maybeFillOp;
  for (auto &u : viewOrAlloc.getUses()) {
    if (auto newFillOp = dyn_cast<FillOp>(u.getOwner())) {
      if (newFillOp.getOutputBuffer(0) != viewOrAlloc)
        continue;
      LLVM_DEBUG(llvm::dbgs() << dbgPref << "fill candidate " << *newFillOp);
      if (mayExistInterleavedUses(newFillOp, copyOp, {viewOrAlloc, subView}))
        continue;
      maybeFillOp = newFillOp;
      break;
    }
  }
  // Ensure padding matches.
  if (maybeFillOp && xferOp.padding() != maybeFillOp.value())
    return failure();
  if (maybeFillOp)
    LLVM_DEBUG(llvm::dbgs() << dbgPref << "with maybeFillOp " << *maybeFillOp);

  // `in` is the subview that linalg.copy reads. Replace it.
  Value in = copyOp.getInput(0);

  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_read, the attribute must be reset
  // conservatively.
  Value res = rewriter.create<vector::TransferReadOp>(
      xferOp.getLoc(), xferOp.getVectorType(), in, xferOp.indices(),
      xferOp.permutation_map(), xferOp.padding(), ArrayAttr());

  if (maybeFillOp)
    rewriter.eraseOp(maybeFillOp);
  rewriter.eraseOp(copyOp);
  rewriter.replaceOp(xferOp, res);

  return success();
}

/// TODO: use interfaces, side-effects and aliasing analysis as appropriate,
/// when available.
LogicalResult LinalgCopyVTWForwardingPattern::matchAndRewrite(
    vector::TransferWriteOp xferOp, PatternRewriter &rewriter) const {
  // Transfer into `viewOrAlloc`.
  Value viewOrAlloc = xferOp.memref();
  if (!viewOrAlloc.getDefiningOp<ViewOp>() &&
      !viewOrAlloc.getDefiningOp<AllocOp>())
    return failure();

  // Ensure there is exactly one subview of `viewOrAlloc` defining `subView`.
  SubViewOp subViewOp = getSubViewUseIfUnique(viewOrAlloc);
  if (!subViewOp)
    return failure();
  Value subView = subViewOp.getResult();

  // Find the copy from `subView` without interleaved uses.
  CopyOp copyOp;
  for (auto &u : subViewOp.getResult().getUses()) {
    if (auto newCopyOp = dyn_cast<CopyOp>(u.getOwner())) {
      if (newCopyOp.getInput(0) != subView)
        continue;
      if (mayExistInterleavedUses(xferOp, newCopyOp, {viewOrAlloc, subView}))
        continue;
      copyOp = newCopyOp;
      break;
    }
  }
  if (!copyOp)
    return failure();

  // `out` is the subview copied into that we replace.
  Value out = copyOp.getOutputBuffer(0);

  // Forward vector.transfer into copy.
  // linalg.copy + linalg.fill can be used to create a padded local buffer.
  // The `masked` attribute is only valid on this padded buffer.
  // When forwarding to vector.transfer_write, the attribute must be reset
  // conservatively.
  rewriter.create<vector::TransferWriteOp>(
      xferOp.getLoc(), xferOp.vector(), out, xferOp.indices(),
      xferOp.permutation_map(), ArrayAttr());

  rewriter.eraseOp(copyOp);
  rewriter.eraseOp(xferOp);

  return success();
}
