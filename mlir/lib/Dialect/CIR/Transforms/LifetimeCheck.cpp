//===- Lifetimecheck.cpp - emit diagnostic checks for lifetime violations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CIR/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace cir;

namespace {
struct LifetimeCheckPass : public LifetimeCheckBase<LifetimeCheckPass> {
  LifetimeCheckPass() = default;

  // Prints the resultant operation statistics post iterating over the module.
  void runOnOperation() override;

  void checkOperation(Operation *op);
  void checkBlock(Block &block);
  void checkRegion(Region &region);

  void checkAlloca(AllocaOp op);
  void checkStore(StoreOp op);
  void checkLoad(LoadOp op);

  struct State {
    using DataTy = enum { Invalid, NullPtr, LocalValue };
    DataTy data = Invalid;
    State() = default;
    State(DataTy d) : data(d) {}
    State(mlir::Value v) : data(LocalValue), value(v) {}
    // FIXME: use int/ptr pair to save space
    std::optional<mlir::Value> value = std::nullopt;

    /// Provide less/equal than operator for sorting / set ops.
    bool operator<(const State &RHS) const {
      // FIXME: note that this makes the ordering non-deterministic, do
      // we really care?
      if (data == LocalValue && RHS.data == LocalValue)
        return value->getAsOpaquePointer() < RHS.value->getAsOpaquePointer();
      else
        return data < RHS.data;
    }
    bool operator==(const State &RHS) const {
      if (data == LocalValue && RHS.data == LocalValue)
        return *value == *RHS.value;
      else
        return data == RHS.data;
    }

    void dump();

    static State getInvalid() { return {}; }
    static State getNullPtr() { return {NullPtr}; }
    static State getLocalValue(mlir::Value v) { return {v}; }
  };

  using PSetType = llvm::SmallSet<State, 4>;

  // FIXME: this should be a ScopedHashTable for consistency.
  using PMapType = llvm::DenseMap<mlir::Value, PSetType>;

  PMapType pmap;
  SmallPtrSet<mlir::Value, 8> ptrs;

  // Represents the scope context for IR operations (cir.scope, cir.if,
  // then/else regions, etc). Tracks the declaration of variables in the current
  // local scope.
  struct LexicalScopeContext {
    unsigned Depth = 0;
    LexicalScopeContext() = default;
    ~LexicalScopeContext() = default;

    // Track all local values added in this scope
    llvm::SmallVector<mlir::Value, 4> localValues;

    void dumpLocalValues();
  };

  class LexicalScopeGuard {
    LifetimeCheckPass &Pass;
    LexicalScopeContext *OldVal = nullptr;

  public:
    LexicalScopeGuard(LifetimeCheckPass &p, LexicalScopeContext *L) : Pass(p) {
      if (Pass.currScope) {
        OldVal = Pass.currScope;
        L->Depth++;
      }
      Pass.currScope = L;
    }

    LexicalScopeGuard(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(const LexicalScopeGuard &) = delete;
    LexicalScopeGuard &operator=(LexicalScopeGuard &&other) = delete;

    void cleanup();
    void restore() { Pass.currScope = OldVal; }
    ~LexicalScopeGuard() {
      cleanup();
      restore();
    }
  };

  LexicalScopeContext *currScope = nullptr;
  void dumpPset(PSetType &pset);
  void dumpPmap();
};
} // namespace

static StringRef getVarNameFromValue(mlir::Value v) {
  if (auto allocaOp = dyn_cast<::mlir::cir::AllocaOp>(v.getDefiningOp()))
    return allocaOp.getName();
  assert(0 && "how did it get here?");
  return "";
}

void LifetimeCheckPass::LexicalScopeGuard::cleanup() {
  auto *localScope = Pass.currScope;
  auto &pmap = Pass.pmap;
  // If we are cleaning up at the function level, nothing
  // to do here cause we are past all possible deference points
  if (localScope->Depth == 0)
    return;

  // 2.3 - KILL(x) means to replace all occurrences of x and x' and x'' (etc.)
  // in the pmap with invalid. For example, if pmap is {(p1,{a}), (p2,{a'})},
  // KILL(a') would invalidate only p2, and KILL(a) would invalidate both p1 and
  // p2.
  for (auto value : localScope->localValues) {
    for (auto &mapEntry : pmap) {

      // We are deleting this entry anyways, nothing to do here.
      if (value == mapEntry.first)
        continue;

      // If the local value is part of this pset, it means
      // we need to invalidate it, otherwise keep searching.
      auto &pset = mapEntry.second;
      State valState = State::getLocalValue(value);
      if (!pset.contains(valState))
        continue;

      // Erase the reference and mark this invalid.
      // FIXME: add a way to just mutate the state.
      // FIXME: right now we are piling up invalids, if it's already
      // invalid we don't need to add again? only if tracking the path.
      pset.erase(valState);
      pset.insert(State::getInvalid());
    }
    // Delete the local value from pmap, since its gone now.
    pmap.erase(value);
  }
}

void LifetimeCheckPass::checkBlock(Block &block) {
  // Block main role is to hold a list of Operations: let's recurse.
  for (Operation &op : block.getOperations())
    checkOperation(&op);
}

void LifetimeCheckPass::checkRegion(Region &region) {
  // FIXME: if else-then blocks have their own scope too.
  for (Block &block : region.getBlocks())
    checkBlock(block);
}

void LifetimeCheckPass::checkAlloca(AllocaOp allocaOp) {
  auto addr = allocaOp.getAddr();
  assert(!pmap.count(addr) && "only one alloca for any given address");

  pmap[addr] = {};
  if (!allocaOp.isPointerType()) {
    // 2.4.2 - When a local Value x is declared, add (x, {x}) to pmap.
    pmap[addr].insert(State::getLocalValue(addr));
    currScope->localValues.push_back(addr);
    return;
  }

  // 2.4.2 - When a non-parameter non-member Pointer p is declared, add
  // (p, {invalid}) to pmap.
  ptrs.insert(addr);
  pmap[addr].insert(State::getInvalid());

  // If other styles of initialization gets added, required to add support
  // here.
  assert(allocaOp.getInitAttr().getValue() == mlir::cir::InitStyle::cinit &&
         "other init styles tbd");
}

void LifetimeCheckPass::checkStore(StoreOp storeOp) {
  auto addr = storeOp.getAddr();

  // We only care about stores that change local pointers, local values
  // are not interesting here (just yet).
  if (!ptrs.count(addr))
    return;

  auto data = storeOp.getValue();
  // 2.4.2 - If the declaration includes an initialization, the
  // initialization is treated as a separate operation
  if (auto cstOp = dyn_cast<::mlir::cir::ConstantOp>(data.getDefiningOp())) {
    assert(cstOp.isNullPtr() && "not implemented");
    // 2.4.2 - If the initialization is default initialization or zero
    // initialization, set pset(p) = {null}; for example:
    //
    //  int* p; => pset(p) == {invalid}
    //  int* p{}; or string_view p; => pset(p) == {null}.
    //  int *p = nullptr; => pset(p) == {nullptr} => pset(p) == {null}
    pmap[addr] = {};
    pmap[addr].insert(State::getNullPtr());
    return;
  }

  if (auto allocaOp = dyn_cast<::mlir::cir::AllocaOp>(data.getDefiningOp())) {
    // p = &x;
    pmap[addr] = {};
    pmap[addr].insert(State::getLocalValue(data));
    return;
  }

  storeOp.dump();
  // FIXME: asserts here should become remarks for non-implemented parts.
  assert(0 && "not implemented");
}

void LifetimeCheckPass::checkLoad(LoadOp loadOp) {
  auto addr = loadOp.getAddr();
  // Only interested in checking deference on top of pointer types.
  if (!pmap.count(addr) || !ptrs.count(addr))
    return;

  if (!loadOp.getIsDeref())
    return;

  // 2.4.2 - On every dereference of a Pointer p, enforce that p is not
  // invalid.
  if (!pmap[addr].count(State::getInvalid())) {
    // FIXME: perhaps add a remark that we got a valid dereference
    return;
  }

  // Looks like we found a invalid path leading to this deference point,
  // diagnose it.
  //
  // Note that usually the use of the invalid address happens at the
  // load or store using the result of this loadOp.
  emitWarning(loadOp.getLoc())
      << "use of invalid pointer '" << getVarNameFromValue(addr) << "'";
}

void LifetimeCheckPass::checkOperation(Operation *op) {
  if (isa<::mlir::ModuleOp>(op)) {
    for (Region &region : op->getRegions())
      checkRegion(region);
    return;
  }

  bool isLexicalScopeOp =
      isa<::mlir::FuncOp>(op) || isa<::mlir::cir::ScopeOp>(op);
  if (isLexicalScopeOp) {
    // Add a new scope. Note that as part of the scope cleanup process
    // we apply section 2.3 KILL(x) functionality, turning relevant
    // references invalid.
    LexicalScopeContext lexScope{};
    LexicalScopeGuard scopeGuard{*this, &lexScope};
    for (Region &region : op->getRegions())
      checkRegion(region);
    return;
  }

  if (auto allocaOp = dyn_cast<::mlir::cir::AllocaOp>(op))
    return checkAlloca(allocaOp);
  if (auto storeOp = dyn_cast<::mlir::cir::StoreOp>(op))
    return checkStore(storeOp);
  if (auto loadOp = dyn_cast<::mlir::cir::LoadOp>(op))
    return checkLoad(loadOp);
}

void LifetimeCheckPass::runOnOperation() {
  Operation *op = getOperation();
  checkOperation(op);
}

std::unique_ptr<Pass> mlir::createLifetimeCheckPass() {
  return std::make_unique<LifetimeCheckPass>();
}

//===----------------------------------------------------------------------===//
// Dump helpers
//===----------------------------------------------------------------------===//

void LifetimeCheckPass::LexicalScopeContext::dumpLocalValues() {
  llvm::errs() << "Local values: { ";
  for (auto value : localValues) {
    llvm::errs() << getVarNameFromValue(value);
    llvm::errs() << ", ";
  }
  llvm::errs() << "}\n";
}

void LifetimeCheckPass::State::dump() {
  switch (data) {
  case Invalid:
    llvm::errs() << "invalid";
    break;
  case NullPtr:
    llvm::errs() << "nullptr";
    break;
  case LocalValue:
    llvm::errs() << getVarNameFromValue(*value);
    break;
  }
}

void LifetimeCheckPass::dumpPset(PSetType &pset) {
  llvm::errs() << "{ ";
  for (auto s : pset) {
    s.dump();
    llvm::errs() << ", ";
  }
  llvm::errs() << "}";
}

void LifetimeCheckPass::dumpPmap() {
  llvm::errs() << "pmap {\n";
  int entry = 0;
  for (auto &mapEntry : pmap) {
    llvm::errs() << "  " << entry << ": " << getVarNameFromValue(mapEntry.first)
                 << "  "
                 << "=> ";
    dumpPset(mapEntry.second);
    llvm::errs() << "\n";
    entry++;
  }
  llvm::errs() << "}\n";
}
