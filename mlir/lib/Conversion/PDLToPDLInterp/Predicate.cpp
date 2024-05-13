//===- Predicate.cpp - Pattern predicates ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Predicate.h"

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

//===----------------------------------------------------------------------===//
// Positions
//===----------------------------------------------------------------------===//

Position::~Position() = default;

/// Returns the depth of the first ancestor operation position.
unsigned Position::getOperationDepth() const {
  if (const auto *operationPos = dyn_cast<OperationPosition>(this))
    return operationPos->getDepth();
  return parent ? parent->getOperationDepth() : 0;
}

//===----------------------------------------------------------------------===//
// AttributePosition

AttributePosition::AttributePosition(const KeyTy &key) : Base(key) {
  parent = key.first;
}

//===----------------------------------------------------------------------===//
// OperandPosition

OperandPosition::OperandPosition(const KeyTy &key) : Base(key) {
  parent = key.first;
}

//===----------------------------------------------------------------------===//
// OperandGroupPosition

OperandGroupPosition::OperandGroupPosition(const KeyTy &key) : Base(key) {
  parent = std::get<0>(key);
}

//===----------------------------------------------------------------------===//
// OperationPosition

bool OperationPosition::isOperandDefiningOp() const {
  return isa_and_nonnull<OperandPosition, OperandGroupPosition>(parent);
}
