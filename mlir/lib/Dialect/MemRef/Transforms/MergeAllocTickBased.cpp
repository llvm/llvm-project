//===- MergeAllocTickBased.cpp - Ticked based merge alloc implementation---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/MergeAllocTickBased.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/StaticMemoryPlanning.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace memref {

using namespace special_ticks;

/// Return `true` if the given MemRef type has a static identity layout (i.e.,
/// no layout).
static bool hasStaticIdentityLayout(MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity();
}

void Tick::access(int64_t tick) {
  if (tick == COMPLEX_ACCESS) {
    firstAccess = COMPLEX_ACCESS;
    lastAccess = COMPLEX_ACCESS;
  }
  if (firstAccess == COMPLEX_ACCESS) {
    return;
  }
  if (firstAccess == NO_ACCESS) {
    firstAccess = tick;
  } else {
    firstAccess = std::min(firstAccess, tick);
  }
  lastAccess = std::max(lastAccess, tick);
}

bool TickCollecter::needsResetTick(TickCollecterStates *s, Operation *scope,
                                   Operation *allocation) const {
  // if the allocation is not in the scope, conservatively set the ticks
  if (!scope->isProperAncestor(allocation)) {
    return true;
  }
  // if the allocation and its alias are used outside of the scope
  for (auto &&alias : s->aliasAnaly.resolve(allocation->getResult(0))) {
    for (auto &&userOp : alias.getUsers()) {
      if (!scope->isProperAncestor(userOp) && !isMemoryEffectFree(userOp)) {
        return true;
      }
    }
  }
  return false;
}

LogicalResult TickCollecter::onPopComplexScope(TickCollecterStates *s,
                                               int64_t endTick) const {
  const auto &scope = s->complexScopeStack.back();
  // if the complex scope is not recognized by us, and if it accesses memory,
  // raise an error
  if (!isa<RegionBranchOpInterface>(scope.scope) &&
      !isa<LoopLikeOpInterface>(scope.scope) && !scope.operations.empty()) {
    return scope.scope->emitOpError("expecting RegionBranchOpInterface or "
                                    "LoopLikeOpInterface for merge-alloc");
  }
  for (auto op : scope.operations) {
    if (needsResetTick(s, scope.scope, op)) {
      // let all referenced buffers have overlapped lifetime
      auto &tick = s->allocTicks[op];
      tick.access(scope.startTick);
      tick.access(endTick);
    }
  }
  return success();
}

LogicalResult TickCollecter::popScopeIfNecessary(TickCollecterStates *s,
                                                 Operation *op) const {
  // first check if we have walked outside of the previous ComplexScope
  while (!s->complexScopeStack.empty()) {
    auto &scope = s->complexScopeStack.back();
    if (!op || !scope.scope->isProperAncestor(op)) {
      if (failed(onPopComplexScope(s, s->curTick))) {
        return failure();
      }
      s->complexScopeStack.pop_back();
    } else {
      break;
    }
  }
  return success();
}

void TickCollecter::forwardTick(TickCollecterStates *s) const { s->curTick++; }

void TickCollecter::accessValue(TickCollecterStates *s, Value v,
                                bool complex) const {
  if (auto refv = dyn_cast<TypedValue<MemRefType>>(v)) {
    for (auto &&base : s->aliasAnaly.resolveReverse(refv)) {
      auto defop = base.getDefiningOp();
      if (isa_and_present<memref::AllocOp>(defop)) {
        s->allocTicks[defop].access(complex ? COMPLEX_ACCESS : s->curTick);
        if (!s->complexScopeStack.empty()) {
          s->complexScopeStack.back().operations.insert(defop);
        }
      }
    }
  }
}

void TickCollecter::onMemrefViews(TickCollecterStates *s,
                                  ViewLikeOpInterface op) const {
  auto viewSrc = op.getViewSource();
  // don't need to access the first operand, which is "source".
  // The "source" operand is not really read or written at this point
  for (auto val : op.getOperation()->getOperands()) {
    if (val != viewSrc)
      accessValue(s, val, false);
  }
}

void TickCollecter::onReturnOp(TickCollecterStates *s, Operation *op) const {
  bool isTopLevel = isa<func::FuncOp>(op->getParentOp());
  for (auto val : op->getOperands()) {
    accessValue(s, val, isTopLevel);
  }
}

void TickCollecter::onAllocOp(TickCollecterStates *s, Operation *op) const {
  s->allocTicks[op].allocTick = s->curTick;
}

void TickCollecter::onGeneralOp(TickCollecterStates *s, Operation *op) const {
  for (auto val : op->getOperands()) {
    accessValue(s, val, false);
  }
}

void TickCollecter::pushComplexScope(TickCollecterStates *s,
                                     Operation *op) const {
  s->complexScopeStack.emplace_back(op, s->curTick);
}

bool TickCollecter::isMergeableAlloc(TickCollecterStates *s, Operation *op,
                                     int64_t tick) const {
  if (tick == COMPLEX_ACCESS) {
    return false;
  }
  if (!hasStaticIdentityLayout(
          cast<MemRefType>(op->getResultTypes().front()))) {
    return false;
  }
  auto alignment = cast<memref::AllocOp>(op).getAlignment();
  if (!alignment) {
    return true; // ok if no alignment
  }
  return alignment > 0 && (s->opt.alignment % alignment.value() == 0);
}

// find the closest surrounding parent operation with AutomaticAllocationScope
// trait, and is not scf.for
Operation *TickCollecter::getAllocScope(TickCollecterStates *s,
                                        Operation *op) const {
  auto parent = op;
  for (;;) {
    parent = parent->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
    if (!parent) {
      return nullptr;
    }
    if (!isa<scf::ForOp>(parent)) {
      return parent;
    }
  }
}

FailureOr<size_t> TickCollecter::getAllocSize(TickCollecterStates *s,
                                              Operation *op) const {
  auto refType = cast<MemRefType>(op->getResultTypes().front());
  int64_t size = refType.getElementTypeBitWidth() / 8;
  // treat bool (i1) as 1 byte. It may not be true for all targets, but we at
  // least have a large enough size for i1
  size = (size != 0) ? size : 1;
  for (auto v : refType.getShape()) {
    size *= v;
  }
  if (size > 0) {
    return static_cast<size_t>(size);
  }
  return op->emitError("Expecting static shaped allocation");
}

FailureOr<MemoryTraceScopes>
TickCollecter::getTrace(TickCollecterStates *s) const {
  struct TraceWithTick {
    // just a tie-breaker when 2 tick are the same
    int64_t allocTick;
    int64_t tick;
    memoryplan::MemoryTrace trace;
    TraceWithTick(int64_t allocTick, int64_t tick, uintptr_t bufferId,
                  size_t size)
        : allocTick{allocTick}, tick{tick}, trace{bufferId, size} {}
  };
  llvm::DenseMap<Operation *, llvm::SmallVector<TraceWithTick, 8>> raw;
  for (auto &[op, tick] : s->allocTicks) {
    if (!isMergeableAlloc(s, op, tick.firstAccess)) {
      continue;
    }
    auto scope = getAllocScope(s, op);
    if (!scope) {
      return op->emitError(
          "This op should be surrounded by an AutomaticAllocationScope");
    }
    auto allocSize = getAllocSize(s, op);
    if (failed(allocSize)) {
      return failure();
    }
    // tick.firstAccess * 2 and tick.lastAccess * 2 + 1 to make sure "dealloc"
    // overlaps "alloc"
    raw[scope].emplace_back(tick.allocTick, tick.firstAccess * 2,
                            reinterpret_cast<uintptr_t>(op), *allocSize);
    raw[scope].emplace_back(tick.allocTick, tick.lastAccess * 2 + 1,
                            reinterpret_cast<uintptr_t>(op), 0);
  }
  MemoryTraceScopes ret;
  for (auto &[scope, trace] : raw) {
    std::stable_sort(trace.begin(), trace.end(),
                     [](const TraceWithTick &a, const TraceWithTick &b) {
                       if (a.tick == b.tick) {
                         return a.allocTick < b.allocTick;
                       }
                       return a.tick < b.tick;
                     });
    auto retTrace = std::make_unique<TickTraceResult>();
    retTrace->traces.reserve(trace.size());
    for (auto &tr : trace) {
      retTrace->traces.emplace_back(tr.trace);
    }
    ret.scopeToTraces[scope] = std::move(retTrace);
  }
  return ret;
}

FailureOr<MemoryTraceScopes>
TickCollecter::operator()(Operation *root,
                          const mlir::BufferViewFlowAnalysis &aliasAnaly,
                          const MergeAllocationOptions &option) const {
  TickCollecterStates s{aliasAnaly, option};
  TickCollecter collecter;
  LogicalResult result = success();
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (failed(collecter.popScopeIfNecessary(&s, op))) {
      result = failure();
    }
    collecter.forwardTick(&s);
    if (auto viewop = dyn_cast<ViewLikeOpInterface>(op)) {
      collecter.onMemrefViews(&s, viewop);
    } else if (op->hasTrait<OpTrait::ReturnLike>()) {
      collecter.onReturnOp(&s, op);
    } else if (isa<AllocOp>(op)) {
      collecter.onAllocOp(&s, op);
    } else if (!isMemoryEffectFree(op)) {
      // if the op has no memory effects, it don't contribute to liveness
      collecter.onGeneralOp(&s, op);
    }
    if (op->getNumRegions() > 0 && !isa<func::FuncOp>(op)) {
      // finally, if op is complex scope, push one ComplexScope
      collecter.pushComplexScope(&s, op);
    }
  });
  if (failed(result)) {
    return result;
  }
  if (failed(collecter.popScopeIfNecessary(&s, nullptr))) {
    return failure();
  }
  if (option.checkOnly) {
    for (auto &[alloc, tick] : s.allocTicks) {
      auto allocscope = getAllocScope(&s, alloc);
      alloc->setAttr(
          "__mergealloc_lifetime",
          DenseI64ArrayAttr::get(root->getContext(),
                                 {reinterpret_cast<int64_t>(allocscope),
                                  tick.firstAccess, tick.lastAccess}));
      allocscope->setAttr(
          "__mergealloc_scope",
          IntegerAttr::get(mlir::IntegerType::get(root->getContext(), 64),
                           reinterpret_cast<int64_t>(allocscope)));
    }
    return MemoryTraceScopes();
  }
  return collecter.getTrace(&s);
}

FailureOr<MemorySchedule> tickBasedPlanMemory(Operation *op,
                                              const LifetimeTrace &tr,
                                              const MergeAllocationOptions &o) {
  auto traceObj = dyn_cast<TickTraceResult>(&tr);
  if (!traceObj) {
    return failure();
  }
  auto &traces = traceObj->traces;
  if (traces.empty()) {
    return MemorySchedule{};
  }
  std::unordered_map<uintptr_t, std::size_t> outSchedule;
  std::unordered_map<uintptr_t, std::vector<uintptr_t>> dummy;
  auto total = memoryplan::scheduleMemoryAllocations(
      traces, o.alignment, !o.noLocalityFirst, memoryplan::InplaceInfoMap(),
      outSchedule, dummy);
  MemorySchedule ret;
  ret.totalSize = total;
  for (auto [k, offset] : outSchedule) {
    ret.allocToOffset[reinterpret_cast<Operation *>(k)] =
        static_cast<int64_t>(offset);
  }
  return std::move(ret);
}

Value MergeAllocDefaultMutator::buildAlloc(OpBuilder &builder, Operation *scope,
                                           int64_t size,
                                           int64_t alignmentInt) const {
  auto &block = scope->getRegion(0).getBlocks().front();
  builder.setInsertionPointToStart(&block);
  auto alignment = builder.getIntegerAttr(
      IntegerType::get(builder.getContext(), 64), alignmentInt);
  auto alloc = builder.create<memref::AllocOp>(
      scope->getLoc(), MemRefType::get({size}, builder.getI8Type()), alignment);
  return alloc;
}
Value MergeAllocDefaultMutator::buildView(OpBuilder &builder, Operation *scope,
                                          Operation *origAllocOp,
                                          Value mergedAlloc,
                                          int64_t byteOffset) const {
  builder.setInsertionPoint(origAllocOp);
  auto byteShift =
      builder.create<arith::ConstantIndexOp>(origAllocOp->getLoc(), byteOffset);
  return builder.create<memref::ViewOp>(origAllocOp->getLoc(),
                                        origAllocOp->getResultTypes().front(),
                                        mergedAlloc, byteShift, ValueRange{});
}

LogicalResult
MergeAllocDefaultMutator::operator()(Operation *op, Operation *scope,
                                     const MemorySchedule &schedule,
                                     const MergeAllocationOptions &o) const {
  if (schedule.allocToOffset.empty()) {
    return success();
  }
  OpBuilder builder{op->getContext()};
  auto alloc = buildAlloc(
      builder, scope, static_cast<int64_t>(schedule.totalSize), o.alignment);
  for (auto &[origBuf, offset] : schedule.allocToOffset) {
    origBuf->replaceAllUsesWith(
        buildView(builder, scope, origBuf, alloc, static_cast<int64_t>(offset))
            .getDefiningOp()
            ->getResults());
    origBuf->remove();
  }
  return success();
}

} // namespace memref
} // namespace mlir
