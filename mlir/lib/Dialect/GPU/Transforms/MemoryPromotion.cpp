//===- MemoryPromotion.cpp - Utilities for moving data across GPU memories ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities that allow one to create IR moving the data
// across different levels of the GPU memory hierarchy.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/MemoryPromotion.h"

#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::gpu;

/// Emits the (imperfect) loop nest performing the copy between "from" and "to"
/// values using the bounds derived from the "from" value. Emits at least
/// GPUDialect::getNumWorkgroupDimensions() loops, completing the nest with
/// single-iteration loops. Maps the innermost loops to thread dimensions, in
/// reverse order to enable access coalescing in the innermost loop.
static void insertCopyLoops(ImplicitLocOpBuilder &b, Value from, Value to) {
  auto memRefType = from.getType().cast<MemRefType>();
  auto rank = memRefType.getRank();

  SmallVector<Value, 4> lbs, ubs, steps;
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  // Make sure we have enough loops to use all thread dimensions, these trivial
  // loops should be outermost and therefore inserted first.
  if (rank < GPUDialect::getNumWorkgroupDimensions()) {
    unsigned extraLoops = GPUDialect::getNumWorkgroupDimensions() - rank;
    lbs.resize(extraLoops, zero);
    ubs.resize(extraLoops, one);
    steps.resize(extraLoops, one);
  }

  // Add existing bounds.
  lbs.append(rank, zero);
  ubs.reserve(lbs.size());
  steps.reserve(lbs.size());
  for (auto idx = 0; idx < rank; ++idx) {
    ubs.push_back(b.createOrFold<memref::DimOp>(
        from, b.create<arith::ConstantIndexOp>(idx)));
    steps.push_back(one);
  }

  // Obtain thread identifiers and block sizes, necessary to map to them.
  auto indexType = b.getIndexType();
  SmallVector<Value, 3> threadIds, blockDims;
  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    threadIds.push_back(b.create<gpu::ThreadIdOp>(indexType, dim));
    blockDims.push_back(b.create<gpu::BlockDimOp>(indexType, dim));
  }

  // Produce the loop nest with copies.
  SmallVector<Value, 8> ivs(lbs.size());
  mlir::scf::buildLoopNest(
      b, b.getLoc(), lbs, ubs, steps,
      [&](OpBuilder &b, Location loc, ValueRange loopIvs) {
        ivs.assign(loopIvs.begin(), loopIvs.end());
        auto activeIvs = llvm::makeArrayRef(ivs).take_back(rank);
        Value loaded = b.create<memref::LoadOp>(loc, from, activeIvs);
        b.create<memref::StoreOp>(loc, loaded, to, activeIvs);
      });

  // Map the innermost loops to threads in reverse order.
  for (const auto &en :
       llvm::enumerate(llvm::reverse(llvm::makeArrayRef(ivs).take_back(
           GPUDialect::getNumWorkgroupDimensions())))) {
    Value v = en.value();
    auto loop = cast<scf::ForOp>(v.getParentRegion()->getParentOp());
    mapLoopToProcessorIds(loop, {threadIds[en.index()]},
                          {blockDims[en.index()]});
  }
}

/// Emits the loop nests performing the copy to the designated location in the
/// beginning of the region, and from the designated location immediately before
/// the terminator of the first block of the region. The region is expected to
/// have one block. This boils down to the following structure
///
///   ^bb(...):
///     <loop-bound-computation>
///     for %arg0 = ... to ... step ... {
///       ...
///         for %argN = <thread-id-x> to ... step <block-dim-x> {
///           %0 = load %from[%arg0, ..., %argN]
///           store %0, %to[%arg0, ..., %argN]
///         }
///       ...
///     }
///     gpu.barrier
///     <... original body ...>
///     gpu.barrier
///     for %arg0 = ... to ... step ... {
///       ...
///         for %argN = <thread-id-x> to ... step <block-dim-x> {
///           %1 = load %to[%arg0, ..., %argN]
///           store %1, %from[%arg0, ..., %argN]
///         }
///       ...
///     }
///
/// Inserts the barriers unconditionally since different threads may be copying
/// values and reading them. An analysis would be required to eliminate barriers
/// in case where value is only used by the thread that copies it. Both copies
/// are inserted unconditionally, an analysis would be required to only copy
/// live-in and live-out values when necessary. This copies the entire memref
/// pointed to by "from". In case a smaller block would be sufficient, the
/// caller can create a subview of the memref and promote it instead.
static void insertCopies(Region &region, Location loc, Value from, Value to) {
  auto fromType = from.getType().cast<MemRefType>();
  auto toType = to.getType().cast<MemRefType>();
  (void)fromType;
  (void)toType;
  assert(fromType.getShape() == toType.getShape());
  assert(fromType.getRank() != 0);
  assert(llvm::hasSingleElement(region) &&
         "unstructured control flow not supported");

  auto b = ImplicitLocOpBuilder::atBlockBegin(loc, &region.front());
  insertCopyLoops(b, from, to);
  b.create<gpu::BarrierOp>();

  b.setInsertionPoint(&region.front().back());
  b.create<gpu::BarrierOp>();
  insertCopyLoops(b, to, from);
}

/// Promotes a function argument to workgroup memory in the given function. The
/// copies will be inserted in the beginning and in the end of the function.
void mlir::promoteToWorkgroupMemory(GPUFuncOp op, unsigned arg) {
  Value value = op.getArgument(arg);
  auto type = value.getType().dyn_cast<MemRefType>();
  assert(type && type.hasStaticShape() && "can only promote memrefs");

  // Get the type of the buffer in the workgroup memory.
  int workgroupMemoryAddressSpace = gpu::GPUDialect::getWorkgroupAddressSpace();
  auto bufferType = MemRefType::get(type.getShape(), type.getElementType(), {},
                                    workgroupMemoryAddressSpace);
  Value attribution = op.addWorkgroupAttribution(bufferType, value.getLoc());

  // Replace the uses first since only the original uses are currently present.
  // Then insert the copies.
  value.replaceAllUsesWith(attribution);
  insertCopies(op.getBody(), op.getLoc(), value, attribution);
}
