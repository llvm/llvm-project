//===- DependenceAnalysis.cpp - Dependence analysis on SSA views ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements view-based alias and dependence analyses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-dependence-analysis"

using namespace mlir;
using namespace mlir::linalg;

using llvm::dbgs;

Value Aliases::find(Value v) {
  if (v.isa<BlockArgument>())
    return v;

  auto it = aliases.find(v);
  if (it != aliases.end()) {
    assert(it->getSecond().getType().isa<BaseMemRefType>() &&
           "Memref expected");
    return it->getSecond();
  }

  while (true) {
    if (v.isa<BlockArgument>())
      return v;

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return v;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(defOp)) {
      // Collect all memory effects on `v`.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffectsOnValue(v, effects);

      // If we have the 'Allocate' memory effect on `v`, then `v` should be the
      // original buffer.
      if (llvm::any_of(
              effects, [](const MemoryEffects::EffectInstance &instance) {
                return isa<MemoryEffects::Allocate>(instance.getEffect());
              }))
        return v;
    }

    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(defOp)) {
      auto it =
          aliases.insert(std::make_pair(v, find(viewLikeOp.getViewSource())));
      return it.first->second;
    }

    llvm::errs() << "View alias analysis reduces to: " << v << "\n";
    llvm_unreachable("unsupported view alias case");
  }
}

StringRef LinalgDependenceGraph::getDependenceTypeStr(DependenceType depType) {
  switch (depType) {
  case LinalgDependenceGraph::DependenceType::RAW:
    return "RAW";
  case LinalgDependenceGraph::DependenceType::RAR:
    return "RAR";
  case LinalgDependenceGraph::DependenceType::WAR:
    return "WAR";
  case LinalgDependenceGraph::DependenceType::WAW:
    return "WAW";
  default:
    break;
  }
  llvm_unreachable("Unexpected DependenceType");
}

LinalgDependenceGraph
LinalgDependenceGraph::buildDependenceGraph(Aliases &aliases, FuncOp f) {
  SmallVector<Operation *, 8> linalgOps;
  f.walk([&](LinalgOp op) { linalgOps.push_back(op); });
  return LinalgDependenceGraph(aliases, linalgOps);
}

LinalgDependenceGraph::LinalgDependenceGraph(Aliases &aliases,
                                             ArrayRef<Operation *> ops)
    : aliases(aliases), linalgOps(ops.begin(), ops.end()) {
  for (auto en : llvm::enumerate(linalgOps)) {
    assert(isa<LinalgOp>(en.value()) && "Expected value for LinalgOp");
    linalgOpPositions.insert(std::make_pair(en.value(), en.index()));
  }
  for (unsigned i = 0, e = ops.size(); i < e; ++i) {
    for (unsigned j = i + 1; j < e; ++j) {
      addDependencesBetween(cast<LinalgOp>(ops[i]), cast<LinalgOp>(ops[j]));
    }
  }
}

void LinalgDependenceGraph::addDependenceElem(DependenceType dt,
                                              LinalgOpView indexingOpView,
                                              LinalgOpView dependentOpView) {
  LLVM_DEBUG(dbgs() << "\nAdd dep type " << getDependenceTypeStr(dt) << ":\t"
                    << *indexingOpView.op << " -> " << *dependentOpView.op);
  dependencesFromGraphs[dt][indexingOpView.op].push_back(
      LinalgDependenceGraphElem{dependentOpView, indexingOpView.view});
  dependencesIntoGraphs[dt][dependentOpView.op].push_back(
      LinalgDependenceGraphElem{indexingOpView, dependentOpView.view});
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    LinalgOp src, LinalgDependenceGraph::DependenceType dt) const {
  return getDependencesFrom(src.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesFrom(
    Operation *src, LinalgDependenceGraph::DependenceType dt) const {
  auto iter = dependencesFromGraphs[dt].find(src);
  if (iter == dependencesFromGraphs[dt].end())
    return llvm::make_range(nullptr, nullptr);
  return llvm::make_range(iter->second.begin(), iter->second.end());
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    LinalgOp dst, LinalgDependenceGraph::DependenceType dt) const {
  return getDependencesInto(dst.getOperation(), dt);
}

LinalgDependenceGraph::dependence_range
LinalgDependenceGraph::getDependencesInto(
    Operation *dst, LinalgDependenceGraph::DependenceType dt) const {
  auto iter = dependencesIntoGraphs[dt].find(dst);
  if (iter == dependencesIntoGraphs[dt].end())
    return llvm::make_range(nullptr, nullptr);
  return llvm::make_range(iter->second.begin(), iter->second.end());
}

void LinalgDependenceGraph::addDependencesBetween(LinalgOp src, LinalgOp dst) {
  assert(src.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  assert(dst.hasBufferSemantics() &&
         "expected linalg op with buffer semantics");
  for (auto srcView : src.getOutputBuffers()) { // W
    // RAW graph
    for (auto dstView : dst.getInputs()) {   // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAW
        addDependenceElem(DependenceType::RAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAW graph
    for (auto dstView : dst.getOutputBuffers()) { // W
      if (aliases.alias(srcView, dstView)) {      // if alias, fill WAW
        addDependenceElem(DependenceType::WAW,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
  for (auto srcView : src.getInputs()) { // R
    // RAR graph
    for (auto dstView : dst.getInputs()) {   // R
      if (aliases.alias(srcView, dstView)) { // if alias, fill RAR
        addDependenceElem(DependenceType::RAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
    // WAR graph
    for (auto dstView : dst.getOutputBuffers()) { // W
      if (aliases.alias(srcView, dstView)) {      // if alias, fill WAR
        addDependenceElem(DependenceType::WAR,
                          LinalgOpView{src.getOperation(), srcView},
                          LinalgOpView{dst.getOperation(), dstView});
      }
    }
  }
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findCoveringDependences(LinalgOp srcLinalgOp,
                                               LinalgOp dstLinalgOp) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, nullptr,
      {DependenceType::WAW, DependenceType::WAR, DependenceType::RAW});
}

SmallVector<Operation *, 8> LinalgDependenceGraph::findCoveringWrites(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::WAW, DependenceType::WAR});
}

SmallVector<Operation *, 8> LinalgDependenceGraph::findCoveringReads(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view) const {
  return findOperationsWithCoveringDependences(
      srcLinalgOp, dstLinalgOp, view,
      {DependenceType::RAR, DependenceType::RAW});
}

SmallVector<Operation *, 8>
LinalgDependenceGraph::findOperationsWithCoveringDependences(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp, Value view,
    ArrayRef<DependenceType> types) const {
  auto *src = srcLinalgOp.getOperation();
  auto *dst = dstLinalgOp.getOperation();
  auto srcPos = linalgOpPositions.lookup(src);
  auto dstPos = linalgOpPositions.lookup(dst);
  assert(srcPos < dstPos && "expected dst after src in IR traversal order");

  SmallVector<Operation *, 8> res;
  // Consider an intermediate interleaved `interim` op, look for any dependence
  // to an aliasing view on a src -> op -> dst path.
  // TODO: we are not considering paths yet, just interleaved positions.
  for (auto dt : types) {
    for (auto dependence : getDependencesFrom(src, dt)) {
      auto interimPos = linalgOpPositions.lookup(dependence.dependentOpView.op);
      // Skip if not interleaved.
      if (interimPos >= dstPos || interimPos <= srcPos)
        continue;
      if (view && !aliases.alias(view, dependence.indexingView))
        continue;
      auto *op = dependence.dependentOpView.op;
      LLVM_DEBUG(dbgs() << "\n***Found covering dependence of type "
                        << getDependenceTypeStr(dt) << ": " << *src << " -> "
                        << *op << " on " << dependence.indexingView);
      res.push_back(op);
    }
  }
  return res;
}

bool LinalgDependenceGraph::hasDependenceFrom(
    LinalgOp srcLinalgOp, LinalgOp dstLinalgOp,
    ArrayRef<LinalgDependenceGraph::DependenceType> depTypes) const {
  for (auto dep : depTypes) {
    for (auto dependence : getDependencesInto(dstLinalgOp, dep)) {
      if (dependence.dependentOpView.op == srcLinalgOp)
        return true;
    }
  }
  return false;
}

bool LinalgDependenceGraph::hasDependentOperations(
    LinalgOp linalgOp,
    ArrayRef<LinalgDependenceGraph::DependenceType> depTypes) const {
  for (auto dep : depTypes) {
    if (!getDependencesFrom(linalgOp, dep).empty() ||
        !getDependencesInto(linalgOp, dep).empty())
      return true;
  }
  return false;
}
