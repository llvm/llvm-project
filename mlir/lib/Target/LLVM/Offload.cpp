//===- Offload.cpp - LLVM Target Offload ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines LLVM target offload utility classes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/Offload.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace mlir::LLVM;

std::string OffloadHandler::getBeginSymbol(StringRef suffix) {
  return ("__begin_offload_" + suffix).str();
}

std::string OffloadHandler::getEndSymbol(StringRef suffix) {
  return ("__end_offload_" + suffix).str();
}

namespace {
/// Returns the type of the entry array.
llvm::ArrayType *getEntryArrayType(llvm::Module &module, size_t numElems) {
  return llvm::ArrayType::get(llvm::offloading::getEntryTy(module), numElems);
}

/// Creates the initializer of the entry array.
llvm::Constant *getEntryArrayBegin(llvm::Module &module,
                                   ArrayRef<llvm::Constant *> entries) {
  // If there are no entries return a constant zero initializer.
  llvm::ArrayType *arrayTy = getEntryArrayType(module, entries.size());
  return entries.empty() ? llvm::ConstantAggregateZero::get(arrayTy)
                         : llvm::ConstantArray::get(arrayTy, entries);
}

/// Computes the end position of the entry array.
llvm::Constant *getEntryArrayEnd(llvm::Module &module,
                                 llvm::GlobalVariable *begin, size_t numElems) {
  llvm::Type *intTy = module.getDataLayout().getIntPtrType(module.getContext());
  return llvm::ConstantExpr::getGetElementPtr(
      llvm::offloading::getEntryTy(module), begin,
      ArrayRef<llvm::Constant *>({llvm::ConstantInt::get(intTy, numElems)}),
      true);
}
} // namespace

OffloadHandler::OffloadEntryArray
OffloadHandler::getEntryArray(StringRef suffix) {
  llvm::GlobalVariable *beginGV =
      module.getGlobalVariable(getBeginSymbol(suffix), true);
  llvm::GlobalVariable *endGV =
      module.getGlobalVariable(getEndSymbol(suffix), true);
  return {beginGV, endGV};
}

OffloadHandler::OffloadEntryArray
OffloadHandler::emitEmptyEntryArray(StringRef suffix) {
  llvm::ArrayType *arrayTy = getEntryArrayType(module, 0);
  auto *beginGV = new llvm::GlobalVariable(
      module, arrayTy, /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      getEntryArrayBegin(module, {}), getBeginSymbol(suffix));
  auto *endGV = new llvm::GlobalVariable(
      module, llvm::PointerType::get(module.getContext(), 0),
      /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      getEntryArrayEnd(module, beginGV, 0), getEndSymbol(suffix));
  return {beginGV, endGV};
}

LogicalResult OffloadHandler::insertOffloadEntry(StringRef suffix,
                                                 llvm::Constant *entry) {
  // Get the begin and end symbols to the entry array.
  std::string beginSymId = getBeginSymbol(suffix);
  llvm::GlobalVariable *beginGV = module.getGlobalVariable(beginSymId, true);
  llvm::GlobalVariable *endGV =
      module.getGlobalVariable(getEndSymbol(suffix), true);
  // Fail if the symbols are missing.
  if (!beginGV || !endGV)
    return failure();
  // Create the entry initializer.
  assert(beginGV->getInitializer() && "entry array initializer is missing.");
  // Add existing entries into the new entry array.
  SmallVector<llvm::Constant *> entries;
  if (auto beginInit = dyn_cast_or_null<llvm::ConstantAggregate>(
          beginGV->getInitializer())) {
    for (unsigned i = 0; i < beginInit->getNumOperands(); ++i)
      entries.push_back(beginInit->getOperand(i));
  }
  // Add the new entry.
  entries.push_back(entry);
  // Create a global holding the new updated set of entries.
  auto *arrayTy = llvm::ArrayType::get(llvm::offloading::getEntryTy(module),
                                       entries.size());
  auto *entryArr = new llvm::GlobalVariable(
      module, arrayTy, /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      getEntryArrayBegin(module, entries), beginSymId, endGV);
  // Replace the old entry array variable withe new one.
  beginGV->replaceAllUsesWith(entryArr);
  beginGV->eraseFromParent();
  entryArr->setName(beginSymId);
  // Update the end symbol.
  endGV->setInitializer(getEntryArrayEnd(module, entryArr, entries.size()));
  return success();
}
