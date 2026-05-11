//===-- MemorySlotInterfaces.cpp - MemorySlot interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/MemorySlotInterfaces.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Interfaces/MemorySlotOpInterfaces.cpp.inc"
#include "mlir/Interfaces/MemorySlotTypeInterfaces.cpp.inc"

using namespace mlir;

namespace {
/// One step in a view chain, leaf-first. `inputElemType` is the elemType
/// of the slot one step closer to root; `outputElemType` is the elemType
/// this step exposes.
struct ViewStep {
  PromotableOpInterface view;
  Type inputElemType;
  Type outputElemType;
};
} // namespace

/// Walks back from `value` to `rootSlot.ptr` along
/// `getPromotableSlotView` chains. On success, populates `chainOut` with
/// the view ops leaf-to-root and writes the type at which `value` aliases
/// the underlying slot to `*outViewElemType`.
static bool walkPromotableSlotViewChain(Value value, const MemorySlot &rootSlot,
                                        SmallVectorImpl<ViewStep> &chainOut,
                                        Type *outViewElemType) {
  if (value == rootSlot.ptr) {
    if (outViewElemType)
      *outViewElemType = rootSlot.elemType;
    return true;
  }

  Value current = value;
  Type aliasElemType{};
  llvm::SmallPtrSet<Value, 4> seen;
  while (current != rootSlot.ptr) {
    if (!seen.insert(current).second)
      return false;
    auto promotable =
        dyn_cast_or_null<PromotableOpInterface>(current.getDefiningOp());
    if (!promotable)
      return false;
    std::optional<PromotableSlotView> info = promotable.getPromotableSlotView();
    if (!info || info->view.ptr != current)
      return false;
    if (!aliasElemType)
      aliasElemType = info->view.elemType;
    chainOut.push_back(ViewStep{promotable, /*inputElemType=*/Type{},
                                /*outputElemType=*/info->view.elemType});
    current = info->slotPointerOperand;
  }

  // Fill in each step's `inputElemType` from the previous step's output
  // (or `rootSlot.elemType` for the root-most step).
  Type prevOutput = rootSlot.elemType;
  for (ViewStep &step : llvm::reverse(chainOut)) {
    step.inputElemType = prevOutput;
    prevOutput = step.outputElemType;
  }

  if (outViewElemType)
    *outViewElemType = aliasElemType ? aliasElemType : rootSlot.elemType;
  return true;
}

bool mlir::isPromotableSlotView(Value value, const MemorySlot &rootSlot,
                                Type *outViewElemType) {
  SmallVector<ViewStep> chain;
  return walkPromotableSlotViewChain(value, rootSlot, chain, outViewElemType);
}

std::optional<MemorySlot> mlir::getOpViewSlot(Operation *op,
                                              const MemorySlot &rootSlot) {
  for (Value operand : op->getOperands()) {
    Type viewElemType;
    if (isPromotableSlotView(operand, rootSlot, &viewElemType))
      return MemorySlot{operand, viewElemType};
  }
  return std::nullopt;
}

Value mlir::convertSlotValueToViewValue(Value slotValue, Value viewPtr,
                                        const MemorySlot &rootSlot,
                                        OpBuilder &builder) {
  SmallVector<ViewStep> chain;
  if (!walkPromotableSlotViewChain(viewPtr, rootSlot, chain, /*out=*/nullptr))
    return {};
  Value current = slotValue;
  // Root-to-leaf walk: reverse the leaf-first chain.
  for (ViewStep &step : llvm::reverse(chain)) {
    current = step.view.convertSlotValue(current, step.outputElemType, builder);
    if (!current)
      return {};
  }
  return current;
}

Value mlir::convertViewValueToSlotValue(Value viewValue, Value viewPtr,
                                        const MemorySlot &rootSlot,
                                        OpBuilder &builder) {
  SmallVector<ViewStep> chain;
  if (!walkPromotableSlotViewChain(viewPtr, rootSlot, chain, /*out=*/nullptr))
    return {};
  Value current = viewValue;
  for (ViewStep &step : chain) {
    current = step.view.convertSlotValue(current, step.inputElemType, builder);
    if (!current)
      return {};
  }
  return current;
}
