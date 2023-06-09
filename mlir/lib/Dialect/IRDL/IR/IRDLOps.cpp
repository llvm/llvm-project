//===- IRDLOps.cpp - IRDL dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"

using namespace mlir;
using namespace mlir::irdl;

std::unique_ptr<Constraint> IsOp::getVerifier(
    ArrayRef<Value> valueToConstr,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> const &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> const
        &attrs) {
  return std::make_unique<IsConstraint>(getExpectedAttr());
}

std::unique_ptr<Constraint> ParametricOp::getVerifier(
    ArrayRef<Value> valueToConstr,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> const &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> const
        &attrs) {
  SmallVector<unsigned> constraints;
  for (Value arg : getArgs()) {
    for (auto [i, value] : enumerate(valueToConstr)) {
      if (value == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  // Symbol reference case for the base
  SymbolRefAttr symRef = getBaseType();
  Operation *defOp =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), symRef);
  if (!defOp) {
    emitError() << symRef << " does not refer to any existing symbol";
    return nullptr;
  }

  if (auto typeOp = dyn_cast<TypeOp>(defOp))
    return std::make_unique<DynParametricTypeConstraint>(types.at(typeOp).get(),
                                                         constraints);

  if (auto attrOp = dyn_cast<AttributeOp>(defOp))
    return std::make_unique<DynParametricAttrConstraint>(attrs.at(attrOp).get(),
                                                         constraints);

  llvm_unreachable("verifier should ensure that the referenced operation is "
                   "either a type or an attribute definition");
}

std::unique_ptr<Constraint> AnyOfOp::getVerifier(
    ArrayRef<Value> valueToConstr,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> const &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> const
        &attrs) {
  SmallVector<unsigned> constraints;
  for (Value arg : getArgs()) {
    for (auto [i, value] : enumerate(valueToConstr)) {
      if (value == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  return std::make_unique<AnyOfConstraint>(constraints);
}

std::unique_ptr<Constraint> AllOfOp::getVerifier(
    ArrayRef<Value> valueToConstr,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> const &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> const
        &attrs) {
  SmallVector<unsigned> constraints;
  for (Value arg : getArgs()) {
    for (auto [i, value] : enumerate(valueToConstr)) {
      if (value == arg) {
        constraints.push_back(i);
        break;
      }
    }
  }

  return std::make_unique<AllOfConstraint>(constraints);
}

std::unique_ptr<Constraint> AnyOp::getVerifier(
    ArrayRef<Value> valueToConstr,
    DenseMap<TypeOp, std::unique_ptr<DynamicTypeDefinition>> const &types,
    DenseMap<AttributeOp, std::unique_ptr<DynamicAttrDefinition>> const
        &attrs) {
  return std::make_unique<AnyAttributeConstraint>();
}
