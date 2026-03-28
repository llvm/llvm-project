//===- StaticMemoryPlannerAnalysis.cpp - Analysis for static memory -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Walk each block in a function and find memref.alloc / memref.dealloc pairs
// that can be reasoned about statically. For each eligible alloc compute a
// conservative alias-aware lifetime interval via BufferViewFlowAnalysis and
// report it as an op remark. Ineligible allocs get a skip reason instead.
//
// Meant to run after ownership-based-buffer-deallocation followed by
// bufferization-lower-deallocations, once all pairs are explicit in the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "static-memory-planner-analysis"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_STATICMEMORYPLANNERANALYSISPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Returns true if `op` lives inside a loop or conditional body. Stops
/// walking at function/isolated-region boundaries.
static bool isNestedInLoopOrConditional(Operation *op) {
  Operation *parent = op->getParentOp();
  while (parent) {
    if (isa<scf::ForOp, scf::ForallOp, scf::WhileOp, scf::ParallelOp,
            scf::IfOp>(parent))
      return true;
    // Do not cross function/isolated-region boundaries.
    if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
      break;
    parent = parent->getParentOp();
  }
  return false;
}

/// Returns the single user of `value` with a MemoryEffects::Free effect, or
/// nullptr if there are zero or more than one.
static Operation *findUniqueFreeSideEffectUser(Value value) {
  Operation *freeUser = nullptr;
  for (Operation *user : value.getUsers()) {
    auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(user);
    if (!memEffectOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    memEffectOp.getEffects(effects);
    for (const auto &effect : effects) {
      if (isa<MemoryEffects::Free>(effect.getEffect())) {
        if (freeUser)
          return nullptr; // more than one free user, not uniquely deallocated
        freeUser = user;
      }
    }
  }
  return freeUser;
}

/// Numbers each op in `block` by its position (0-based).
static DenseMap<Operation *, unsigned> buildOpIndexMap(Block *block) {
  DenseMap<Operation *, unsigned> indexMap;
  unsigned idx = 0;
  for (Operation &op : *block)
    indexMap[&op] = idx++;
  return indexMap;
}

/// Returns the size of `type` in bytes, or -1 for non int/float element types.
static int64_t getStaticSizeBytes(MemRefType type) {
  Type elemType = type.getElementType();
  if (!elemType.isIntOrFloat())
    return -1;
  int64_t numElems = type.getNumElements();
  unsigned elemBits = type.getElementTypeBitWidth();
  return (numElems * static_cast<int64_t>(elemBits) + 7) / 8;
}

//===----------------------------------------------------------------------===//
// StaticMemoryPlannerAnalysisPass
//===----------------------------------------------------------------------===//

struct StaticMemoryPlannerAnalysisPass
    : public bufferization::impl::StaticMemoryPlannerAnalysisPassBase<
          StaticMemoryPlannerAnalysisPass> {
public:
  StaticMemoryPlannerAnalysisPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    // Build alias analysis once; we call resolve() per alloc below.
    BufferViewFlowAnalysis aliasAnalysis(func);

    // Op index maps, built on demand per block.
    DenseMap<Block *, DenseMap<Operation *, unsigned>> blockIndexMaps;

    func.walk([&](memref::AllocOp allocOp) {
      auto memrefType = allocOp.getType();

      // Skip dynamic shapes; size is not known at compile time.
      if (!memrefType.hasStaticShape()) {
        ++numSkipDynamic;
        (void)allocOp.emitRemark("static-memory-planner: skip: dynamic shape");
        return;
      }

      // Skip allocs inside loops or conditionals.
      if (isNestedInLoopOrConditional(allocOp)) {
        ++numSkipNested;
        (void)allocOp.emitRemark(
            "static-memory-planner: skip: nested in loop or conditional");
        return;
      }

      // Need exactly one dealloc in the same block to form a pair.
      Value allocResult = allocOp.getResult();
      Operation *deallocOp = findUniqueFreeSideEffectUser(allocResult);
      if (!deallocOp || deallocOp->getBlock() != allocOp->getBlock()) {
        ++numSkipNoDealloc;
        (void)allocOp.emitRemark(
            "static-memory-planner: skip: no unique same-block dealloc");
        return;
      }

      Block *block = allocOp->getBlock();

      // Skip if any alias escapes or is used in a different block.
      const BufferViewFlowAnalysis::ValueSetT &aliases =
          aliasAnalysis.resolve(allocResult);
      bool escapes = false;
      for (Value alias : aliases) {
        for (Operation *user : alias.getUsers()) {
          if (user == deallocOp)
            continue;
          if (user->getBlock() != block || isa<func::ReturnOp>(user)) {
            escapes = true;
            break;
          }
        }
        if (escapes)
          break;
      }
      if (escapes) {
        ++numSkipEscaping;
        (void)allocOp.emitRemark(
            "static-memory-planner: skip: escaping or cross-block alias");
        return;
      }

      // Interval: [allocIdx, max(deallocIdx, last alias use in block)].
      auto &indexMap = blockIndexMaps.try_emplace(block).first->second;
      if (indexMap.empty())
        indexMap = buildOpIndexMap(block);

      unsigned allocIdx = indexMap.lookup(allocOp);
      unsigned deallocIdx = indexMap.lookup(deallocOp);
      unsigned endIdx = deallocIdx;

      for (Value alias : aliases) {
        for (Operation *user : alias.getUsers()) {
          if (user == deallocOp)
            continue;
          // Users in nested regions: lift to the ancestor in this block.
          if (Operation *ancestor = block->findAncestorOpInBlock(*user)) {
            auto it = indexMap.find(ancestor);
            if (it != indexMap.end() && it->second > endIdx)
              endIdx = it->second;
          }
        }
      }

      int64_t sizeBytes = getStaticSizeBytes(memrefType);
      std::optional<uint64_t> alignment = allocOp.getAlignment();

      ++numEligible;

      LDBG() << "eligible: " << allocOp << " size=" << sizeBytes
             << " interval=[" << allocIdx << "," << endIdx << "]";

      std::string msg = "static-memory-planner: eligible";
      if (sizeBytes >= 0)
        msg += ": size=" + std::to_string(sizeBytes) + " bytes";
      else
        msg += ": size=unknown";

      if (alignment)
        msg += ", align=" + std::to_string(*alignment);

      msg += ", interval=[" + std::to_string(allocIdx) + "," +
             std::to_string(endIdx) + "]";

      (void)allocOp.emitRemark(msg);
    });

    LDBG() << "[" << func.getName()
           << "] summary: eligible=" << (unsigned)numEligible
           << " skip-dynamic=" << (unsigned)numSkipDynamic
           << " skip-nested=" << (unsigned)numSkipNested
           << " skip-no-dealloc=" << (unsigned)numSkipNoDealloc
           << " skip-escaping=" << (unsigned)numSkipEscaping;
  }
};

} // end anonymous namespace
