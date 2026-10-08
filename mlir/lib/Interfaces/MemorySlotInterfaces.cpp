//===-- MemorySlotInterfaces.cpp - MemorySlot interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/MemorySlotInterfaces.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Interfaces/MemorySlotOpInterfaces.cpp.inc"
#include "mlir/Interfaces/MemorySlotTypeInterfaces.cpp.inc"

using namespace mlir;

/// Returns the slot describing `aliasPtr`: `rootSlot` if it is the root,
/// the entry in `aliasMap` if it's a known alias, or `nullopt` otherwise.
static std::optional<MemorySlot>
getParentSlot(Value aliasPtr, const MemorySlot &rootSlot,
              const PromotableAliasMap &aliasMap) {
  if (aliasPtr == rootSlot.ptr)
    return rootSlot;
  auto it = aliasMap.find(aliasPtr);
  if (it == aliasMap.end())
    return std::nullopt;
  return it->second.slot;
}

void mlir::populatePromotableAliasMap(PromotableAliaserInterface aliaser,
                                      const MemorySlot &rootSlot,
                                      PromotableAliasMap &aliasMap) {
  for (OpOperand &operand : aliaser->getOpOperands()) {
    std::optional<MemorySlot> parentSlot =
        getParentSlot(operand.get(), rootSlot, aliasMap);
    if (!parentSlot)
      continue;
    SmallVector<MemorySlot, 2> newSlots;
    aliaser.getPromotableSlotAliases(operand, *parentSlot, newSlots);
    for (const MemorySlot &alias : newSlots)
      aliasMap.try_emplace(alias.ptr, PromotableSlotAliasInfo{alias, &operand});
  }
}

std::optional<MemorySlot>
mlir::getOpAliasSlot(Operation *op, const MemorySlot &rootSlot,
                     const PromotableAliasMap &aliasMap) {
  for (Value operand : op->getOperands())
    if (std::optional<MemorySlot> slot =
            getParentSlot(operand, rootSlot, aliasMap))
      return slot;
  return std::nullopt;
}

bool mlir::referencesAtMostOneAliasOfSlot(Operation *op,
                                          const MemorySlot &rootSlot,
                                          const PromotableAliasMap &aliasMap) {
  Value uniqueAliasPtr;
  for (Value operand : op->getOperands()) {
    std::optional<MemorySlot> slot = getParentSlot(operand, rootSlot, aliasMap);
    if (!slot)
      continue;
    if (uniqueAliasPtr && uniqueAliasPtr != slot->ptr)
      return false;
    uniqueAliasPtr = slot->ptr;
  }
  return true;
}

namespace {
/// A step in an alias chain, from leaf to root. `parentSlot` is one step
/// closer to the root; `aliasSlot` is the slot exposed at this step.
struct ChainStep {
  PromotableAliaserInterface aliaser;
  OpOperand *aliasedSlotPointerOperand;
  MemorySlot parentSlot;
  MemorySlot aliasSlot;
};
} // namespace

/// Walks from `aliasSlot` back to `rootSlot` via `aliasMap`. Returns the
/// leaf-to-root chain, or `nullopt` if `aliasSlot` is not a known alias.
static std::optional<SmallVector<ChainStep>>
buildAliasChain(const MemorySlot &aliasSlot, const MemorySlot &rootSlot,
                const PromotableAliasMap &aliasMap) {
  SmallVector<ChainStep> chain;
  Value current = aliasSlot.ptr;
  while (current != rootSlot.ptr) {
    auto it = aliasMap.find(current);
    if (it == aliasMap.end())
      return std::nullopt;
    OpOperand *operand = it->second.aliasedSlotPointerOperand;
    auto aliaser = cast<PromotableAliaserInterface>(operand->getOwner());
    std::optional<MemorySlot> parent =
        getParentSlot(operand->get(), rootSlot, aliasMap);
    if (!parent)
      return std::nullopt;
    chain.push_back(ChainStep{aliaser, operand, *parent, it->second.slot});
    current = operand->get();
  }
  return chain;
}

Value mlir::convertSlotValueToAliasValue(Value slotValue,
                                         const MemorySlot &aliasSlot,
                                         const MemorySlot &rootSlot,
                                         const PromotableAliasMap &aliasMap,
                                         OpBuilder &builder) {
  std::optional<SmallVector<ChainStep>> chain =
      buildAliasChain(aliasSlot, rootSlot, aliasMap);
  if (!chain)
    return {};
  Value current = slotValue;
  // Root-to-leaf walk: reverse the leaf-first chain.
  for (ChainStep &step : llvm::reverse(*chain)) {
    current = step.aliaser.projectSlotValueToAliasValue(
        *step.aliasedSlotPointerOperand, step.parentSlot, step.aliasSlot,
        current, builder);
    if (!current)
      return {};
  }
  return current;
}

Value mlir::convertAliasValueToSlotValue(Value aliasValue,
                                         const MemorySlot &aliasSlot,
                                         Value rootReachingDef,
                                         const MemorySlot &rootSlot,
                                         const PromotableAliasMap &aliasMap,
                                         OpBuilder &builder) {
  std::optional<SmallVector<ChainStep>> chainOpt =
      buildAliasChain(aliasSlot, rootSlot, aliasMap);
  if (!chainOpt)
    return {};
  SmallVector<ChainStep> &chain = *chainOpt;

  // Project `rootReachingDef` down to each step's parent level so the
  // per-step projector can use it (needed for partial sub-aliases; full
  // aliases ignore it). The chain is leaf-first, so `chain.back()` is the
  // root-most step (parent = rootSlot) and `chain.front()` is the leaf.
  SmallVector<Value> perStepReachingDef(chain.size());
  Value current = rootReachingDef;
  for (int i = static_cast<int>(chain.size()) - 1; i >= 0; --i) {
    perStepReachingDef[i] = current;
    current = chain[i].aliaser.projectSlotValueToAliasValue(
        *chain[i].aliasedSlotPointerOperand, chain[i].parentSlot,
        chain[i].aliasSlot, current, builder);
    if (!current)
      return {};
  }

  // Walk leaf-to-root, combining `aliasValue` with the projected reaching
  // definition at each step.
  current = aliasValue;
  for (size_t i = 0; i < chain.size(); ++i) {
    current = chain[i].aliaser.projectAliasValueToSlotValue(
        *chain[i].aliasedSlotPointerOperand, chain[i].parentSlot,
        chain[i].aliasSlot, current, perStepReachingDef[i], builder);
    if (!current)
      return {};
  }
  return current;
}
