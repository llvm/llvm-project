//===-- MLIRMutator.h - Mutation engine for fuzzing MLIR ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides the MLIRMutator class, which drives mutations on MLIR modules
// based on a configurable set of strategies. Some common strategies are
// also included here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_FUZZMUTATE_MLIRMUTATOR_H
#define MLIR_FUZZMUTATE_MLIRMUTATOR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace mlir {
class MLIRContext;

class RandomMLIRBuilder {
public:
  RandomMLIRBuilder(int Seed, ModuleOp root);
  std::mt19937 &getRand() { return Rand; }

  using AlleleMenu = llvm::DenseMap<Type, llvm::SmallVector<Value>>;
  AlleleMenu collectAlleles(Operation *at);

  Operation *findSameGene(Operation *exemplar);

  Operation *findDiffGene(Operation *exemplar);

  llvm::ArrayRef<Operation *> bases();

  Value pickAllele(const AlleleMenu &alleles, Type ty);

private:
  Type getSignature(Operation *op);

  std::mt19937 Rand;
  ModuleOp root;
  llvm::SmallVector<Operation *> genes;
  llvm::DenseMap<Type, llvm::SmallVector<Operation *>> geneBuckets;
};

class MLIRGeneMutationStrategy {
public:
  virtual ~MLIRGeneMutationStrategy() = default;

  virtual uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                             uint64_t CurrentWeight) = 0;

  virtual void mutate(ModuleOp module, RandomMLIRBuilder &IB) = 0;
};

class MLIRMutator {
public:
  MLIRMutator(
      std::vector<std::unique_ptr<MLIRGeneMutationStrategy>> &&Strategies)
      : Strategies(std::move(Strategies)) {}

  void mutateModule(ModuleOp module, int Seed, size_t MaxSize);

private:
  std::vector<std::unique_ptr<MLIRGeneMutationStrategy>> Strategies;
};

class NonsenseMutationStrategy : public MLIRGeneMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;
  void mutate(ModuleOp module, RandomMLIRBuilder &IB) override;
};

class SynonymousMutationStrategy : public MLIRGeneMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;
  void mutate(ModuleOp module, RandomMLIRBuilder &IB) override;
};

class MissenseMutationStrategy : public MLIRGeneMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;
  void mutate(ModuleOp module, RandomMLIRBuilder &IB) override;
};

class MLIRStructureMutationStrategy {};

OwningOpRef<ModuleOp> parseAndVerify(llvm::StringRef text,
                                     MLIRContext &context);
} // namespace mlir

#endif
