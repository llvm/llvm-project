//===- MergeAllocTickBased.cpp - Ticked based merge alloc implementation---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/MergeAlloc.h"
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

/// Return `true` if the given MemRef type has a static identity layout (i.e.,
/// no layout).
static bool hasStaticIdentityLayout(MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity();
}

namespace {
static constexpr int64_t NO_ACCESS = -1;
static constexpr int64_t COMPLEX_ACCESS = -2;
struct Tick {
  int64_t firstAccess = NO_ACCESS;
  int64_t lastAccess = NO_ACCESS;

  void access(int64_t tick) {
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
};

bool isMergeableAlloc(Operation *op, int64_t tick) {
  if (tick == COMPLEX_ACCESS) {
    return false;
  }
  if (!hasStaticIdentityLayout(
          cast<MemRefType>(op->getResultTypes().front()))) {
    return false;
  }
  // currently only support alignment: none, 1, 2, 4, 8, 16, 32, 64
  auto alignment = cast<memref::AllocOp>(op).getAlignment();
  if (!alignment) {
    return true; // ok if no alignment
  }
  return alignment > 0 && (64 % alignment.value() == 0);
}

// find the closest surrounding parent operation with AutomaticAllocationScope
// trait, and is not scf.for
Operation *getAllocScope(Operation *op) {
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

FailureOr<size_t> getAllocSize(Operation *op) {
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

// A complex scope object is addition info for a RegionBranchOpInterface or
// LoopLikeOpInterface. It contains the scope itself, and the referenced alloc
// ops inside this scope. We use this object to track which buffers this scope
// accesses. These buffers must have overlapped lifetime
struct ComplexScope {
  Operation *scope;
  int64_t startTick;
  llvm::SmallPtrSet<Operation *, 8> operations;
  ComplexScope(Operation *scope, int64_t startTick)
      : scope{scope}, startTick{startTick} {}
  // returns true of an allocation either is not defined in the scope, or the
  // allocation escapes from the scope
  bool needsResetTick(Operation *scope, Operation *allocation,
                      const mlir::BufferViewFlowAnalysis &aliasAnaly) const {
    // if the allocation is not in the scope, conservatively set the ticks
    if (!scope->isProperAncestor(allocation)) {
      return true;
    }
    // if the allocation and its alias are used outside of the scope
    for (auto &&alias : aliasAnaly.resolve(allocation->getResult(0))) {
      for (auto &&userOp : alias.getUsers()) {
        if (!scope->isProperAncestor(userOp) && !isMemoryEffectFree(userOp)) {
          return true;
        }
      }
    }
    return false;
  }

  // called when walk() runs outside of the scope
  LogicalResult onPop(int64_t endTick,
                      const mlir::BufferViewFlowAnalysis &aliasAnaly,
                      llvm::DenseMap<Operation *, Tick> &allocTicks) {
    // if the complex scope is not recognized by us, and if it accesses memory,
    // raise an error
    if (!isa<RegionBranchOpInterface>(scope) &&
        !isa<LoopLikeOpInterface>(scope) && !operations.empty()) {
      return scope->emitOpError("expecting RegionBranchOpInterface or "
                                "LoopLikeOpInterface for merge-alloc");
    }
    for (auto op : operations) {
      if (needsResetTick(scope, op, aliasAnaly)) {
        // let all referenced buffers have overlapped lifetime
        auto &tick = allocTicks[op];
        tick.access(startTick);
        tick.access(endTick);
      }
    }
    return success();
  }
};

struct TickTraceResult : public LifetimeTrace {
  memoryplan::Traces traces;
  TickTraceResult() : LifetimeTrace{TK_TICK} {}
  static bool classof(const LifetimeTrace *S) {
    return S->getKind() == TK_TICK;
  }
};

struct TickCollecter {
  const mlir::BufferViewFlowAnalysis &aliasAnaly;
  int64_t curTick = 0;
  llvm::DenseMap<Operation *, Tick> allocTicks;
  llvm::SmallVector<ComplexScope> complexScopeStack;
  TickCollecter(const mlir::BufferViewFlowAnalysis &aliasAnaly)
      : aliasAnaly{aliasAnaly} {}
  LogicalResult popScopeIfNecessary(Operation *op) {
    // first check if we have walked outside of the previous ComplexScope
    while (!complexScopeStack.empty()) {
      auto &scope = complexScopeStack.back();
      if (!op || !scope.scope->isProperAncestor(op)) {
        if (failed(scope.onPop(curTick, aliasAnaly, allocTicks))) {
          return failure();
        }
        complexScopeStack.pop_back();
      } else {
        break;
      }
    }
    return success();
  }

  void forwardTick() { curTick++; }

  void accessValue(Value v, bool complex) {
    if (auto refv = dyn_cast<TypedValue<MemRefType>>(v)) {
      for (auto &&base : aliasAnaly.resolveReverse(refv)) {
        auto defop = base.getDefiningOp();
        if (isa_and_present<memref::AllocOp>(defop)) {
          allocTicks[defop].access(complex ? COMPLEX_ACCESS : curTick);
          if (!complexScopeStack.empty()) {
            complexScopeStack.back().operations.insert(defop);
          }
        }
      }
    }
  }

  void onMemrefViews(ViewLikeOpInterface op) {
    auto viewSrc = op.getViewSource();
    // don't need to access the first operand, which is "source".
    // The "source" operand is not really read or written at this point
    for (auto val : op.getOperation()->getOperands()) {
      if (val != viewSrc)
        accessValue(val, false);
    }
  }

  void onReturnOp(Operation *op) {
    bool isTopLevel = isa<func::FuncOp>(op->getParentOp());
    for (auto val : op->getOperands()) {
      accessValue(val, isTopLevel);
    }
  }

  void onGeneralOp(Operation *op) {
    for (auto val : op->getOperands()) {
      accessValue(val, false);
    }
  }

  void pushComplexScope(Operation *op) {
    complexScopeStack.emplace_back(op, curTick);
  }

  FailureOr<MemoryTraceScopes> getTrace() {
    struct TraceWithTick {
      Operation *op;
      int64_t tick;
      memoryplan::MemoryTrace trace;
      TraceWithTick(int64_t tick, uintptr_t bufferId, size_t size)
          : tick{tick}, trace{bufferId, size} {}
    };
    llvm::DenseMap<Operation *, llvm::SmallVector<TraceWithTick, 8>> raw;
    for (auto &[op, tick] : allocTicks) {
      if (!isMergeableAlloc(op, tick.firstAccess)) {
        continue;
      }
      auto scope = getAllocScope(op);
      if (!scope) {
        return op->emitError(
            "This op should be surrounded by an AutomaticAllocationScope");
      }
      auto allocSize = getAllocSize(op);
      if (failed(allocSize)) {
        return failure();
      }
      // tick.firstAccess * 2 and tick.lastAccess * 2 + 1 to make sure "dealloc"
      // overlaps "alloc"
      raw[scope].emplace_back(tick.firstAccess * 2,
                              reinterpret_cast<uintptr_t>(op), *allocSize);
      raw[scope].emplace_back(tick.lastAccess * 2 + 1,
                              reinterpret_cast<uintptr_t>(op), 0);
    }
    MemoryTraceScopes ret;
    for (auto &[scope, trace] : raw) {
      std::stable_sort(trace.begin(), trace.end(),
                       [](const TraceWithTick &a, const TraceWithTick &b) {
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
};
} // namespace

FailureOr<MemoryTraceScopes>
tickBasedCollectMemoryTrace(Operation *root,
                            const mlir::BufferViewFlowAnalysis &aliasAnaly,
                            const MergeAllocationOptions &option) {
  TickCollecter collecter{aliasAnaly};
  LogicalResult result = success();
  root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (failed(collecter.popScopeIfNecessary(op))) {
      result = failure();
    }
    collecter.forwardTick();
    if (auto viewop = dyn_cast<ViewLikeOpInterface>(op)) {
      collecter.onMemrefViews(viewop);
    } else if (op->hasTrait<OpTrait::ReturnLike>()) {
      collecter.onReturnOp(op);
    } else if (!isMemoryEffectFree(op)) {
      // if the op has no memory effects, it don't contribute to liveness
      collecter.onGeneralOp(op);
    }
    if (op->getNumRegions() > 0 && !isa<func::FuncOp>(op)) {
      // finally, if op is complex scope, push one ComplexScope
      collecter.pushComplexScope(op);
    }
  });
  if (failed(result)) {
    return result;
  }
  if (failed(collecter.popScopeIfNecessary(nullptr))) {
    return failure();
  }
  if (option.checkOnly) {
    for (auto &[alloc, tick] : collecter.allocTicks) {
      auto allocscope = getAllocScope(alloc);
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
  return collecter.getTrace();
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
      traces, 64, !o.noLocalityFirst, memoryplan::InplaceInfoMap(),
      outSchedule, dummy);
  MemorySchedule ret;
  ret.totalSize = total;
  for (auto [k, offset] : outSchedule) {
    ret.allocToOffset[reinterpret_cast<Operation *>(k)] =
        static_cast<int64_t>(offset);
  }
  return std::move(ret);
}

LogicalResult tickBasedMutateAllocations(Operation *op, Operation *scope,
                                         const MemorySchedule &schedule,
                                         const MergeAllocationOptions &o) {
  if (schedule.allocToOffset.empty()) {
    return success();
  }
  auto &block = scope->getRegion(0).getBlocks().front();
  OpBuilder builder{&block.front()};
  auto alignment =
      builder.getIntegerAttr(IntegerType::get(op->getContext(), 64), 64);
  auto alloc = builder.create<memref::AllocOp>(
      scope->getLoc(),
      MemRefType::get({static_cast<int64_t>(schedule.totalSize)},
                      builder.getI8Type()),
      alignment);
  for (auto &[origBuf, offset] : schedule.allocToOffset) {
    builder.setInsertionPoint(origBuf);
    auto byteShift = builder.create<arith::ConstantIndexOp>(
        origBuf->getLoc(), static_cast<int64_t>(offset));
    auto view = builder.create<memref::ViewOp>(
        origBuf->getLoc(), origBuf->getResultTypes().front(), alloc, byteShift,
        ValueRange{});
    origBuf->replaceAllUsesWith(view->getResults());
    origBuf->remove();
  }
  return success();
}

} // namespace memref
} // namespace mlir
