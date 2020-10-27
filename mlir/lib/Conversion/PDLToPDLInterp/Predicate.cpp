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

Position::~Position() {}

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
// OperationPosition

OperationPosition *OperationPosition::get(StorageUniquer &uniquer,
                                          ArrayRef<unsigned> index) {
  assert(!index.empty() && "expected at least two indices");

  // Set the parent position if this isn't the root.
  Position *parent = nullptr;
  if (index.size() > 1) {
    auto *node = OperationPosition::get(uniquer, index.drop_back());
    parent = OperandPosition::get(uniquer, std::make_pair(node, index.back()));
  }
  return uniquer.get<OperationPosition>(
      [parent](OperationPosition *node) { node->parent = parent; }, index);
}
