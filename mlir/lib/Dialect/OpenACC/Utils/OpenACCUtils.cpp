//===- OpenACCUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtils.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"

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

std::string mlir::acc::getVariableName(mlir::Value v) {
  Value current = v;

  // Walk through view operations until a name is found or can't go further
  while (Operation *definingOp = current.getDefiningOp()) {
    // Check for `acc.var_name` attribute
    if (auto varNameAttr =
            definingOp->getAttrOfType<VarNameAttr>(getVarNameAttrName()))
      return varNameAttr.getName().str();

    // If it is a data entry operation, get name via getVarName
    if (isa<ACC_DATA_ENTRY_OPS>(definingOp))
      if (auto name = acc::getVarName(definingOp))
        return name->str();

    // If it's a view operation, continue to the source
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(definingOp)) {
      current = viewOp.getViewSource();
      continue;
    }

    break;
  }

  return "";
}

std::string mlir::acc::getRecipeName(mlir::acc::RecipeKind kind,
                                     mlir::Type type) {
  assert(kind == mlir::acc::RecipeKind::private_recipe ||
         kind == mlir::acc::RecipeKind::firstprivate_recipe ||
         kind == mlir::acc::RecipeKind::reduction_recipe);
  if (!llvm::isa<mlir::acc::PointerLikeType, mlir::acc::MappableType>(type))
    return "";

  std::string recipeName;
  llvm::raw_string_ostream ss(recipeName);
  ss << (kind == mlir::acc::RecipeKind::private_recipe ? "privatization_"
         : kind == mlir::acc::RecipeKind::firstprivate_recipe
             ? "firstprivatization_"
             : "reduction_");

  // Print the type using its dialect-defined textual format.
  type.print(ss);
  ss.flush();

  // Replace invalid characters (anything that's not a letter, number, or
  // period) since this needs to be a valid MLIR identifier.
  for (char &c : recipeName) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '_') {
      if (c == '?')
        c = 'U';
      else if (c == '*')
        c = 'Z';
      else if (c == '(' || c == ')' || c == '[' || c == ']' || c == '{' ||
               c == '}' || c == '<' || c == '>')
        c = '_';
      else
        c = 'X';
    }
  }

  return recipeName;
}

mlir::Value mlir::acc::getBaseEntity(mlir::Value val) {
  if (auto partialEntityAccessOp =
          dyn_cast<PartialEntityAccessOpInterface>(val.getDefiningOp())) {
    if (!partialEntityAccessOp.isCompleteView())
      return partialEntityAccessOp.getBaseEntity();
  }

  return val;
}

bool mlir::acc::isValidSymbolUse(mlir::Operation *user,
                                 mlir::SymbolRefAttr symbol,
                                 mlir::Operation **definingOpPtr) {
  mlir::Operation *definingOp =
      mlir::SymbolTable::lookupNearestSymbolFrom(user, symbol);

  // If there are no defining ops, we have no way to ensure validity because
  // we cannot check for any attributes.
  if (!definingOp)
    return false;

  if (definingOpPtr)
    *definingOpPtr = definingOp;

  // Check if the defining op is a recipe (private, reduction, firstprivate).
  // Recipes are valid as they get materialized before being offloaded to
  // device. They are only instructions for how to materialize.
  if (mlir::isa<mlir::acc::PrivateRecipeOp, mlir::acc::ReductionRecipeOp,
                mlir::acc::FirstprivateRecipeOp>(definingOp))
    return true;

  // Check if the defining op is a function
  if (auto func =
          mlir::dyn_cast_if_present<mlir::FunctionOpInterface>(definingOp)) {
    // If this symbol is actually an acc routine - then it is expected for it
    // to be offloaded - therefore it is valid.
    if (func->hasAttr(mlir::acc::getRoutineInfoAttrName()))
      return true;

    // If this symbol is a call to an LLVM intrinsic, then it is likely valid.
    // Check the following:
    // 1. The function is private
    // 2. The function has no body
    // 3. Name starts with "llvm."
    // 4. The function's name is a valid LLVM intrinsic name
    if (func.getVisibility() == mlir::SymbolTable::Visibility::Private &&
        func.getFunctionBody().empty() && func.getName().starts_with("llvm.") &&
        llvm::Intrinsic::lookupIntrinsicID(func.getName()) !=
            llvm::Intrinsic::not_intrinsic)
      return true;
  }

  // A declare attribute is needed for symbol references.
  bool hasDeclare = definingOp->hasAttr(mlir::acc::getDeclareAttrName());
  return hasDeclare;
}
