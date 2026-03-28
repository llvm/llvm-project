//===- StaticMemoryPlannerAnalysis.cpp - Analysis for static memory -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an analysis-only pass that discovers same-block
// memref.alloc / memref.dealloc pairs eligible for static memory planning.
//
// For each eligible allocation the pass:
//   - Computes a conservative alias-aware lifetime interval using
//     BufferViewFlowAnalysis.
//   - Collects metadata (static size in bytes, alignment, memory space).
//   - Emits the results as op remarks on the alloc op.
//
// Ineligible allocations also receive a remark describing the skip reason.
//
// This pass is the first upstream step for the static memory planner project.
// It is intentionally analysis-only (no IR mutations) and covers the simplest
// structured case: same-block, static-shape, non-escaping allocations not
// nested inside loops or conditionals.
//
// Expected pipeline position: after ownership-based-buffer-deallocation and
// bufferization-lower-deallocations.
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

/// Returns true if `op` is nested inside any loop or conditional region,
/// i.e., any ancestor op (up to but not including the nearest
/// IsIsolatedFromAbove boundary) is a loop or conditional.
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

/// Returns the unique user of `value` that carries a MemoryEffects::Free
/// effect, or nullptr when there are zero or multiple such users.
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
          return nullptr; // Multiple free users — not uniquely deallocated.
        freeUser = user;
      }
    }
  }
  return freeUser;
}

/// Builds a block-local operation index map: op → position in block order.
static DenseMap<Operation *, unsigned> buildOpIndexMap(Block *block) {
  DenseMap<Operation *, unsigned> indexMap;
  unsigned idx = 0;
  for (Operation &op : *block)
    indexMap[&op] = idx++;
  return indexMap;
}

/// Returns the static allocation size in bytes for `type`, or -1 if it cannot
/// be determined (e.g., non-integer/float element types such as index).
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

    // Build alias/view-flow analysis once for the entire function.
    // BufferViewFlowAnalysis::resolve(v) gives the transitive closure of all
    // values derived from v (subviews, expands, casts, etc.).
    BufferViewFlowAnalysis aliasAnalysis(func);

    // Lazily-populated per-block operation index maps.
    // Keyed by Block*; maps each Op* to its zero-based position in the block.
    DenseMap<Block *, DenseMap<Operation *, unsigned>> blockIndexMaps;

    // Walk every memref.alloc in the function and classify it.
    func.walk([&](memref::AllocOp allocOp) {
      auto memrefType = allocOp.getType();

      //----------------------------------------------------------------
      // Eligibility check 1: static shape.
      //----------------------------------------------------------------
      if (!memrefType.hasStaticShape()) {
        ++numSkipDynamic;
        (void)allocOp.emitRemark("static-memory-planner: skip: dynamic shape");
        return;
      }

      //----------------------------------------------------------------
      // Eligibility check 2: not nested inside a loop or conditional.
      //----------------------------------------------------------------
      if (isNestedInLoopOrConditional(allocOp)) {
        ++numSkipNested;
        (void)allocOp.emitRemark(
            "static-memory-planner: skip: nested in loop or conditional");
        return;
      }

      //----------------------------------------------------------------
      // Eligibility check 3: unique same-block dealloc.
      //----------------------------------------------------------------
      Value allocResult = allocOp.getResult();
      Operation *deallocOp = findUniqueFreeSideEffectUser(allocResult);
      if (!deallocOp || deallocOp->getBlock() != allocOp->getBlock()) {
        ++numSkipNoDealloc;
        (void)allocOp.emitRemark(
            "static-memory-planner: skip: no unique same-block dealloc");
        return;
      }

      Block *block = allocOp->getBlock();

      //----------------------------------------------------------------
      // Eligibility check 4: no cross-block alias or escaping use.
      // Resolve the full alias set and verify every user is in the same
      // block (or is the dealloc itself). A use in func.return also
      // counts as escaping.
      //----------------------------------------------------------------
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

      //----------------------------------------------------------------
      // Compute alias-aware lifetime interval.
      // Start  = block-local index of the alloc op.
      // End    = max(dealloc index, last use index of any alias in block).
      //----------------------------------------------------------------
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
          // Lift the user to the ancestor op that lives directly in `block`
          // (handles users inside nested regions attached to block-level ops).
          if (Operation *ancestor = block->findAncestorOpInBlock(*user)) {
            auto it = indexMap.find(ancestor);
            if (it != indexMap.end() && it->second > endIdx)
              endIdx = it->second;
          }
        }
      }

      //----------------------------------------------------------------
      // Collect metadata.
      //----------------------------------------------------------------
      int64_t sizeBytes = getStaticSizeBytes(memrefType);
      std::optional<uint64_t> alignment = allocOp.getAlignment();

      //----------------------------------------------------------------
      // Emit eligibility remark and update statistics.
      //----------------------------------------------------------------
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
