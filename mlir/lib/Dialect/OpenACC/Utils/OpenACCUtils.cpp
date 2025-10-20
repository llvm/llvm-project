//===- OpenACCUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtils.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/TypeSwitch.h"

mlir::Operation *mlir::acc::getEnclosingComputeOp(mlir::Region &region) {
  mlir::Operation *parentOp = region.getParentOp();
  while (parentOp) {
    if (mlir::isa<ACC_COMPUTE_CONSTRUCT_OPS>(parentOp))
      return parentOp;
    parentOp = parentOp->getParentOp();
  }
  return nullptr;
}

template <typename OpTy>
static bool isOnlyUsedByOpClauses(mlir::Value val, mlir::Region &region) {
  auto checkIfUsedOnlyByOpInside = [&](mlir::Operation *user) {
    // For any users which are not in the current acc region, we can ignore.
    // Return true so that it can be used in a `all_of` check.
    if (!region.isAncestor(user->getParentRegion()))
      return true;
    return mlir::isa<OpTy>(user);
  };

  return llvm::all_of(val.getUsers(), checkIfUsedOnlyByOpInside);
}

bool mlir::acc::isOnlyUsedByPrivateClauses(mlir::Value val,
                                           mlir::Region &region) {
  return isOnlyUsedByOpClauses<mlir::acc::PrivateOp>(val, region);
}

bool mlir::acc::isOnlyUsedByReductionClauses(mlir::Value val,
                                             mlir::Region &region) {
  return isOnlyUsedByOpClauses<mlir::acc::ReductionOp>(val, region);
}

std::optional<mlir::acc::ClauseDefaultValue>
mlir::acc::getDefaultAttr(Operation *op) {
  std::optional<mlir::acc::ClauseDefaultValue> defaultAttr;
  Operation *currOp = op;

  // Iterate outwards until a default clause is found (since OpenACC
  // specification notes that a visible default clause is the nearest default
  // clause appearing on the compute construct or a lexically containing data
  // construct.
  while (!defaultAttr.has_value() && currOp) {
    defaultAttr =
        llvm::TypeSwitch<mlir::Operation *,
                         std::optional<mlir::acc::ClauseDefaultValue>>(currOp)
            .Case<ACC_COMPUTE_CONSTRUCT_OPS, mlir::acc::DataOp>(
                [&](auto op) { return op.getDefaultAttr(); })
            .Default([&](Operation *) { return std::nullopt; });
    currOp = currOp->getParentOp();
  }

  return defaultAttr;
}

mlir::acc::VariableTypeCategory mlir::acc::getTypeCategory(mlir::Value var) {
  mlir::acc::VariableTypeCategory typeCategory =
      mlir::acc::VariableTypeCategory::uncategorized;
  if (auto mappableTy = dyn_cast<mlir::acc::MappableType>(var.getType()))
    typeCategory = mappableTy.getTypeCategory(var);
  else if (auto pointerLikeTy =
               dyn_cast<mlir::acc::PointerLikeType>(var.getType()))
    typeCategory = pointerLikeTy.getPointeeTypeCategory(
        cast<TypedValue<mlir::acc::PointerLikeType>>(var),
        pointerLikeTy.getElementType());
  return typeCategory;
}
