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
#include "mlir/IR/Dominance.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/FuzzMutate/Random.h"
#include <algorithm>

using namespace mlir;

RandomMLIRBuilder::RandomMLIRBuilder(int Seed, ModuleOp root)
    : Rand(Seed), root(root) {
  // A base is the smallest op a point mutation can act on: no regions, no
  // successors, not a terminator. Collected and bucketed by the
  // operand+result type table for donor lookup.
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
  DominanceInfo dom;
  auto collect = [&](Value v) {
    if (dom.properlyDominates(v, at))
      menu[v.getType()].push_back(v);
  };
  root->walk([&](Operation *op) {
    for (Value res : op->getResults())
      collect(res);
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        for (BlockArgument arg : block.getArguments())
          collect(arg);
  });
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
  auto donorSampler = llvm::makeSampler(
      Rand, llvm::make_filter_range(ops, [exemplar](Operation *op) {
        return op != exemplar;
      }));
  if (donorSampler.isEmpty())
    return nullptr;
  return donorSampler.getSelection();
}

Operation *RandomMLIRBuilder::findDiffGene(Operation *exemplar) {
  // Pick a signature class at random, then a random op within it — uniform
  // random no matter how ops are spread across signatures. The != compares
  // pointers safely: bucket keys and the query signature come from the same
  // type uniquing, so one type table is literally one instance.
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
  llvm::SmallVector<Operation *, 8> bases(IB.bases().begin(), IB.bases().end());
  if (bases.empty())
    return;
  // Unlike synonymous/missense (one op per call), nonsense shuffles the
  // bases and poisons the first k, with k random (at least 1).
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
    op->replaceAllUsesWith(replacements);
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
    exemplar->replaceAllUsesWith(clone);
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
  std::shuffle(bases.begin(), bases.end(), IB.getRand());
  for (Operation *exemplar : bases) {
    Operation *donor = IB.findDiffGene(exemplar);
    if (!donor)
      continue;
    opBuilder.setInsertionPoint(exemplar);
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
    RandomMLIRBuilder::AlleleMenu outputMenu = IB.collectAlleles(exemplar);
    llvm::SmallVector<Value> oldOutputs;
    for (Type ty : exemplar->getResultTypes()) {
      Value allele = IB.pickAllele(outputMenu, ty);
      if (!allele)
        allele = ub::PoisonOp::create(opBuilder, exemplar->getLoc(), ty);
      oldOutputs.push_back(allele);
    }
    exemplar->replaceAllUsesWith(oldOutputs);
    exemplar->erase();
    return;
  }
}

OwningOpRef<ModuleOp> mlir::parseAndVerify(llvm::StringRef text,
                                           MLIRContext &context) {
  return parseSourceString<ModuleOp>(
      text, ParserConfig(&context, /*verifyAfterParse=*/true));
}
