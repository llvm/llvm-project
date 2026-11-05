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
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace mlir {
class MLIRContext;

/// Random source for mutation strategies. Starts minimal (a seeded RNG)
/// and grows with the strategies that need more from it.
class RandomMLIRBuilder {
public:
  RandomMLIRBuilder(int Seed) : Rand(Seed) {}
  std::mt19937 &getRand() { return Rand; }

private:
  std::mt19937 Rand;
};

/// Base class for describing how to mutate a module.
class MLIRMutationStrategy {
public:
  virtual ~MLIRMutationStrategy() = default;

  /// Provide a weight to bias towards choosing this strategy for a
  /// mutation.
  ///
  /// The value of the weight is arbitrary, but a good default is "the
  /// number of distinct ways in which this strategy can mutate a unit".
  virtual uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                             uint64_t CurrentWeight) = 0;

  /// Mutate the top-level structure of the module in place.
  virtual void mutate(ModuleOp module, RandomMLIRBuilder &IB) = 0;
};

/// Entry point for configuring and running MLIR mutations.
class MLIRMutator {
public:
  MLIRMutator(std::vector<std::unique_ptr<MLIRMutationStrategy>> &&Strategies)
      : Strategies(std::move(Strategies)) {}

  /// Mutate given module. No change will be made if no strategy is
  /// selected.
  ///
  /// \param module  module to mutate
  /// \param Seed    seed for random mutation
  /// \param MaxSize max size of the mutated module's textual form
  void mutateModule(ModuleOp module, int Seed, size_t MaxSize);

private:
  std::vector<std::unique_ptr<MLIRMutationStrategy>> Strategies;
};

/// Strategy that duplicates a top-level op of the module — an
/// always-valid structural edit.
class TopLevelOpDuplicationStrategy : public MLIRMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;
  void mutate(ModuleOp module, RandomMLIRBuilder &IB) override;
};

/// Strategy that appends a standalone ub.poison op — corpus novelty,
/// always-valid IR.
class PoisonAppendStrategy : public MLIRMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;
  void mutate(ModuleOp module, RandomMLIRBuilder &IB) override;
};

/// Try to parse the input as a module and verify it.
/// \return the module, or null on failure.
OwningOpRef<ModuleOp> parseAndVerify(llvm::StringRef text,
                                     MLIRContext &context);
} // namespace mlir

#endif
