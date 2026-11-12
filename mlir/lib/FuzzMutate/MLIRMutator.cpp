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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/FuzzMutate/Random.h"
#include <algorithm>

using namespace mlir;

RandomMLIRBuilder::RandomMLIRBuilder(int Seed, ModuleOp root)
    : Rand(Seed), root(root) {
  root->walk([&](Operation *op) {
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
        op->hasTrait<OpTrait::IsTerminator>())
      return;
    genes.push_back(op);
    geneBuckets[getSignature(op)].push_back(op);
  });
}

RandomMLIRBuilder::AlleleMenu
RandomMLIRBuilder::collectAlleles(Operation *at) {
  AlleleMenu menu;
  Operation *cur = at;
  bool first = true;
  while (cur) {
    Block *blk = cur->getBlock();
    if (!blk)
      break;
    for (BlockArgument arg : blk->getArguments())
      menu[arg.getType()].push_back(arg);
    auto end = first ? cur->getIterator() : std::next(cur->getIterator());
    for (auto it = blk->begin(); it != end; ++it)
      for (Value r : it->getResults())
        menu[r.getType()].push_back(r);
    first = false;
    cur = blk->getParentOp();
  }
  return menu;
}

Type RandomMLIRBuilder::getSignature(Operation *op) {
  return FunctionType::get(op->getContext(), op->getOperandTypes(),
                           op->getResultTypes());
}

llvm::ArrayRef<Operation *> RandomMLIRBuilder::bases() {
  return genes;
}

Operation *RandomMLIRBuilder::findSameGene(Operation *exemplar) {
  auto it = geneBuckets.find(getSignature(exemplar));
  if (it == geneBuckets.end())
    return nullptr;
  const auto &ops = it->second;
  unsigned total = ops.size() - 1;
  if (!total)
    return nullptr;
  unsigned pick = std::uniform_int_distribution<unsigned>(0, total - 1)(Rand);
  unsigned i = 0;
  for (Operation *op : ops) {
    if (op == exemplar)
      continue;
    if (i++ == pick)
      return op;
  }
  return nullptr;
}

Operation *RandomMLIRBuilder::findDiffGene(Operation *exemplar) {
  Type exemplarSig = getSignature(exemplar);
  llvm::SmallVector<const llvm::SmallVector<Operation *> *, 8> classes;
  for (auto &bucket : geneBuckets)
    if (bucket.getFirst() != exemplarSig)
      classes.push_back(&bucket.getSecond());
  if (classes.empty())
    return nullptr;
  const auto &ops =
      *classes[std::uniform_int_distribution<size_t>(0, classes.size() - 1)(
          Rand)];
  return ops[std::uniform_int_distribution<size_t>(0, ops.size() - 1)(Rand)];
}

Value RandomMLIRBuilder::pickAllele(const AlleleMenu &alleles, Type ty) {
  auto it = alleles.find(ty);
  if (it == alleles.end())
    return Value();
  const auto &vals = it->second;
  return vals[std::uniform_int_distribution<size_t>(0, vals.size() - 1)(Rand)];
}

void MLIRMutator::mutateModule(ModuleOp module, int Seed, size_t MaxSize) {
  RandomMLIRBuilder IB(Seed, module);
  auto RS = llvm::makeSampler<MLIRGeneMutationStrategy *>(IB.getRand());
  for (const auto &Strategy : Strategies)
    RS.sample(Strategy.get(),
              Strategy->getWeight(0, MaxSize, RS.totalWeight()));
  if (RS.totalWeight() == 0)
    return;
  MLIRGeneMutationStrategy *Strategy = RS.getSelection();
  Strategy->mutate(module, IB);
}

uint64_t NonsenseMutationStrategy::getWeight(size_t CurrentSize, size_t MaxSize,
                                         uint64_t CurrentWeight) {
  return 1;
}

void NonsenseMutationStrategy::mutate(ModuleOp module, RandomMLIRBuilder &IB) {
  llvm::SmallVector<Operation *, 8> bases;
  module.walk([&](Operation *op) {
    if (op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>())
      bases.push_back(op);
  });
  if (bases.empty())
    return;
  std::shuffle(bases.begin(), bases.end(), IB.getRand());
  size_t count =
      std::uniform_int_distribution<size_t>(1, bases.size())(IB.getRand());
  OpBuilder opBuilder(module->getContext());
  opBuilder.getContext()->getOrLoadDialect<ub::UBDialect>();
  for (size_t i = 0; i < count; ++i) {
    Operation *op = bases[i];
    opBuilder.setInsertionPoint(op);
    llvm::SmallVector<Value> replacements;
    for (Type ty : op->getResultTypes())
      replacements.push_back(ub::PoisonOp::create(opBuilder, op->getLoc(), ty));
    for (auto [j, res] : llvm::enumerate(op->getResults()))
      res.replaceAllUsesWith(replacements[j]);
    op->erase();
  }
}

uint64_t SynonymousMutationStrategy::getWeight(size_t CurrentSize,
                                               size_t MaxSize,
                                               uint64_t CurrentWeight) {
  return 1;
}

void SynonymousMutationStrategy::mutate(ModuleOp module,
                                        RandomMLIRBuilder &IB) {
  llvm::SmallVector<Operation *, 8> bases(IB.bases().begin(),
                                          IB.bases().end());
  if (bases.empty())
    return;
  OpBuilder opBuilder(module->getContext());
  // Shuffle the bases and stop after the first point mutation,
  // simulating a point mutation at a random position.
  std::shuffle(bases.begin(), bases.end(), IB.getRand());
  for (Operation *exemplar : bases) {
    Operation *donor = IB.findSameGene(exemplar);
    if (!donor)
      continue;
    opBuilder.setInsertionPoint(exemplar);
    Operation *clone = Operation::create(
        exemplar->getLoc(), donor->getName(), donor->getResultTypes(),
        exemplar->getOperands(), donor->getAttrDictionary(),
        donor->getPropertiesStorage(), /*successors=*/{}, /*numRegions=*/0);
    opBuilder.insert(clone);
    for (auto [i, res] : llvm::enumerate(exemplar->getResults()))
      res.replaceAllUsesWith(clone->getResult(i));
    exemplar->erase();
    return;
  }
}

uint64_t MissenseMutationStrategy::getWeight(size_t CurrentSize,
                                             size_t MaxSize,
                                             uint64_t CurrentWeight) {
  return 1;
}

void MissenseMutationStrategy::mutate(ModuleOp module,
                                      RandomMLIRBuilder &IB) {
  llvm::SmallVector<Operation *, 8> bases(IB.bases().begin(),
                                          IB.bases().end());
  if (bases.empty())
    return;
  OpBuilder opBuilder(module->getContext());
  opBuilder.getContext()->getOrLoadDialect<ub::UBDialect>();
  // Shuffle the bases and stop after the first point mutation,
  // simulating a point mutation at a random position.
  std::shuffle(bases.begin(), bases.end(), IB.getRand());
  for (Operation *exemplar : bases) {
    Operation *donor = IB.findDiffGene(exemplar);
    if (!donor)
      continue;
    opBuilder.setInsertionPoint(exemplar);
    // Donor inputs first: input poisons must precede the replacement, or
    // they fail to dominate it.
    RandomMLIRBuilder::AlleleMenu inputMenu = IB.collectAlleles(exemplar);
    llvm::SmallVector<Value> donorInputs;
    for (Type ty : donor->getOperandTypes()) {
      Value allele = IB.pickAllele(inputMenu, ty);
      if (!allele)
        allele = ub::PoisonOp::create(opBuilder, exemplar->getLoc(), ty);
      donorInputs.push_back(allele);
    }
    Operation *clone = Operation::create(
        exemplar->getLoc(), donor->getName(), donor->getResultTypes(),
        donorInputs, donor->getAttrDictionary(), donor->getPropertiesStorage(),
        /*successors=*/{}, /*numRegions=*/0);
    opBuilder.insert(clone);
    // Old outputs: consumers rewire from the menu (now includes the
    // clone's results); poison on miss.
    RandomMLIRBuilder::AlleleMenu outputMenu = IB.collectAlleles(exemplar);
    for (Value res : exemplar->getResults()) {
      Value allele = IB.pickAllele(outputMenu, res.getType());
      if (!allele)
        allele = ub::PoisonOp::create(opBuilder, exemplar->getLoc(), res.getType());
      res.replaceAllUsesWith(allele);
    }
    exemplar->erase();
    return;
  }
}

OwningOpRef<ModuleOp> mlir::parseAndVerify(llvm::StringRef text,
                                           MLIRContext &context) {
  return parseSourceString<ModuleOp>(
      text, ParserConfig(&context, /*verifyAfterParse=*/true));
}
