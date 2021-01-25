//===- FunctionSupport.cpp - Utility types for function-like ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionSupport.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"

using namespace mlir;

/// Helper to call a callback once on each index in the range
/// [0, `totalIndices`), *except* for the indices given in `indices`.
/// `indices` is allowed to have duplicates and can be in any order.
inline void iterateIndicesExcept(unsigned totalIndices,
                                 ArrayRef<unsigned> indices,
                                 function_ref<void(unsigned)> callback) {
  llvm::BitVector skipIndices(totalIndices);
  for (unsigned i : indices)
    skipIndices.set(i);

  for (unsigned i = 0; i < totalIndices; ++i)
    if (!skipIndices.test(i))
      callback(i);
}

//===----------------------------------------------------------------------===//
// Function Arguments and Results.
//===----------------------------------------------------------------------===//

void mlir::impl::eraseFunctionArguments(Operation *op,
                                        ArrayRef<unsigned> argIndices,
                                        unsigned originalNumArgs,
                                        Type newType) {
  // There are 3 things that need to be updated:
  // - Function type.
  // - Arg attrs.
  // - Block arguments of entry block.
  Block &entry = op->getRegion(0).front();
  SmallString<8> nameBuf;

  // Collect arg attrs to set.
  SmallVector<DictionaryAttr, 4> newArgAttrs;
  iterateIndicesExcept(originalNumArgs, argIndices, [&](unsigned i) {
    newArgAttrs.emplace_back(getArgAttrDict(op, i));
  });

  // Remove any arg attrs that are no longer needed.
  for (unsigned i = newArgAttrs.size(), e = originalNumArgs; i < e; ++i)
    op->removeAttr(getArgAttrName(i, nameBuf));

  // Set the function type.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));

  // Set the new arg attrs, or remove them if empty.
  for (unsigned i = 0, e = newArgAttrs.size(); i != e; ++i) {
    auto nameAttr = getArgAttrName(i, nameBuf);
    if (newArgAttrs[i] && !newArgAttrs[i].empty())
      op->setAttr(nameAttr, newArgAttrs[i]);
    else
      op->removeAttr(nameAttr);
  }

  // Update the entry block's arguments.
  entry.eraseArguments(argIndices);
}

void mlir::impl::eraseFunctionResults(Operation *op,
                                      ArrayRef<unsigned> resultIndices,
                                      unsigned originalNumResults,
                                      Type newType) {
  // There are 2 things that need to be updated:
  // - Function type.
  // - Result attrs.
  SmallString<8> nameBuf;

  // Collect result attrs to set.
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  iterateIndicesExcept(originalNumResults, resultIndices, [&](unsigned i) {
    newResultAttrs.emplace_back(getResultAttrDict(op, i));
  });

  // Remove any result attrs that are no longer needed.
  for (unsigned i = newResultAttrs.size(), e = originalNumResults; i < e; ++i)
    op->removeAttr(getResultAttrName(i, nameBuf));

  // Set the function type.
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));

  // Set the new result attrs, or remove them if empty.
  for (unsigned i = 0, e = newResultAttrs.size(); i != e; ++i) {
    auto nameAttr = getResultAttrName(i, nameBuf);
    if (newResultAttrs[i] && !newResultAttrs[i].empty())
      op->setAttr(nameAttr, newResultAttrs[i]);
    else
      op->removeAttr(nameAttr);
  }
}

//===----------------------------------------------------------------------===//
// Function type signature.
//===----------------------------------------------------------------------===//

FunctionType mlir::impl::getFunctionType(Operation *op) {
  assert(op->hasTrait<OpTrait::FunctionLike>());
  return op->getAttrOfType<TypeAttr>(mlir::impl::getTypeAttrName())
      .getValue()
      .cast<FunctionType>();
}

void mlir::impl::setFunctionType(Operation *op, FunctionType newType) {
  assert(op->hasTrait<OpTrait::FunctionLike>());
  SmallVector<char, 16> nameBuf;
  FunctionType oldType = getFunctionType(op);

  for (int i = newType.getNumInputs(), e = oldType.getNumInputs(); i < e; i++)
    op->removeAttr(getArgAttrName(i, nameBuf));
  for (int i = newType.getNumResults(), e = oldType.getNumResults(); i < e; i++)
    op->removeAttr(getResultAttrName(i, nameBuf));
  op->setAttr(getTypeAttrName(), TypeAttr::get(newType));
}

//===----------------------------------------------------------------------===//
// Function body.
//===----------------------------------------------------------------------===//

Region &mlir::impl::getFunctionBody(Operation *op) {
  assert(op->hasTrait<OpTrait::FunctionLike>());
  return op->getRegion(0);
}
