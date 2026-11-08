//===-- MLIRMutator.cpp - Mutation engine for fuzzing MLIR ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/FuzzMutate/MLIRMutator.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/FuzzMutate/Random.h"

using namespace mlir;

void MLIRMutator::mutateModule(ModuleOp module, int Seed, size_t MaxSize) {
  RandomMLIRBuilder IB(Seed);
  auto RS = llvm::makeSampler<MLIRMutationStrategy *>(IB.getRand());
  for (const auto &Strategy : Strategies)
    RS.sample(Strategy.get(),
              Strategy->getWeight(0, MaxSize, RS.totalWeight()));
  if (RS.totalWeight() == 0)
    return;
  MLIRMutationStrategy *Strategy = RS.getSelection();
  Strategy->mutate(module, IB);
}

uint64_t TopLevelOpDuplicationStrategy::getWeight(size_t CurrentSize,
                                                  size_t MaxSize,
                                                  uint64_t CurrentWeight) {
  return 1;
}

void TopLevelOpDuplicationStrategy::mutate(ModuleOp module,
                                           RandomMLIRBuilder &IB) {
  llvm::SmallVector<Operation *, 8> ops;
  for (Operation &op : module.getBody()->getOperations())
    ops.push_back(&op);
  if (ops.empty())
    return;
  Operation *chosen = ops[std::uniform_int_distribution<size_t>(
      0, ops.size() - 1)(IB.getRand())];
  OpBuilder b(module->getContext());
  b.setInsertionPointToStart(module.getBody());
  Operation *clone = b.clone(*chosen);
  if (isa<SymbolOpInterface>(clone)) {
    StringAttr oldName = SymbolTable::getSymbolName(clone);
    if (oldName) {
      SymbolTable moduleTable(module);
      StringAttr newName;
      for (unsigned i = 0;; ++i) {
        std::string candidate = oldName.str() + "_" + std::to_string(i);
        if (!moduleTable.lookup(candidate)) {
          newName = b.getStringAttr(candidate);
          break;
        }
      }
      SymbolTable::setSymbolName(clone, newName);
      if (failed(SymbolTable::replaceAllSymbolUses(oldName, newName, clone)))
        clone->erase();
    }
  }
}

uint64_t PoisonAppendStrategy::getWeight(size_t CurrentSize, size_t MaxSize,
                                         uint64_t CurrentWeight) {
  return 1;
}

void PoisonAppendStrategy::mutate(ModuleOp module, RandomMLIRBuilder &IB) {
  OpBuilder b(module->getContext());
  b.setInsertionPointToEnd(module.getBody());
  b.getContext()->getOrLoadDialect<ub::UBDialect>();
  ub::PoisonOp::create(b, module->getLoc(), b.getI32Type());
}

OwningOpRef<ModuleOp> mlir::parseAndVerify(llvm::StringRef text,
                                           MLIRContext &context) {
  return parseSourceString<ModuleOp>(
      text, ParserConfig(&context, /*verifyAfterParse=*/true));
}
