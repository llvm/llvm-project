//===-- IRMutator.h - Mutation engine for fuzzing IR ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides the IRMutator class, which drives mutations on IR based on a
// configurable set of strategies. Some common strategies are also included
// here.
//
// Fuzzer-friendly (de)serialization functions are also provided, as these
// are usually needed when mutating IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZMUTATE_IRMUTATOR_H
#define LLVM_FUZZMUTATE_IRMUTATOR_H

#include "llvm/ADT/Optional.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
class BasicBlock;
class Function;
class Instruction;
class Module;

struct RandomIRBuilder;

/// Base class for describing how to mutate a module. mutation functions for
/// each IR unit forward to the contained unit.
class IRMutationStrategy {
public:
  virtual ~IRMutationStrategy() = default;

  /// Provide a weight to bias towards choosing this strategy for a mutation.
  ///
  /// The value of the weight is arbitrary, but a good default is "the number of
  /// distinct ways in which this strategy can mutate a unit". This can also be
  /// used to prefer strategies that shrink the overall size of the result when
  /// we start getting close to \c MaxSize.
  virtual uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                             uint64_t CurrentWeight) = 0;

  /// @{
  /// Mutators for each IR unit. By default these forward to a contained
  /// instance of the next smaller unit.
  virtual void mutate(Module &M, RandomIRBuilder &IB);
  virtual void mutate(Function &F, RandomIRBuilder &IB);
  virtual void mutate(BasicBlock &BB, RandomIRBuilder &IB);
  virtual void mutate(Instruction &I, RandomIRBuilder &IB) {
    llvm_unreachable("Strategy does not implement any mutators");
  }
  /// @}
};

using TypeGetter = std::function<Type *(LLVMContext &)>;

/// Entry point for configuring and running IR mutations.
class IRMutator {
  std::vector<TypeGetter> AllowedTypes;
  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;

public:
  IRMutator(std::vector<TypeGetter> &&AllowedTypes,
            std::vector<std::unique_ptr<IRMutationStrategy>> &&Strategies)
      : AllowedTypes(std::move(AllowedTypes)),
        Strategies(std::move(Strategies)) {}

  void mutateModule(Module &M, int Seed, size_t CurSize, size_t MaxSize);
};

/// Strategy that injects operations into the function.
class InjectorIRStrategy : public IRMutationStrategy {
  std::vector<fuzzerop::OpDescriptor> Operations;

  Optional<fuzzerop::OpDescriptor> chooseOperation(Value *Src,
                                                   RandomIRBuilder &IB);

public:
  InjectorIRStrategy(std::vector<fuzzerop::OpDescriptor> &&Operations)
      : Operations(std::move(Operations)) {}
  static std::vector<fuzzerop::OpDescriptor> getDefaultOps();

  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override {
    return Operations.size();
  }

  using IRMutationStrategy::mutate;
  void mutate(Function &F, RandomIRBuilder &IB) override;
  void mutate(BasicBlock &BB, RandomIRBuilder &IB) override;
};

/// Strategy that deletes instructions when the Module is too large.
class InstDeleterIRStrategy : public IRMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override;

  using IRMutationStrategy::mutate;
  void mutate(Function &F, RandomIRBuilder &IB) override;
  void mutate(Instruction &Inst, RandomIRBuilder &IB) override;
};

/// Strategy that modifies instruction attributes and operands.
class InstModificationIRStrategy : public IRMutationStrategy {
public:
  uint64_t getWeight(size_t CurrentSize, size_t MaxSize,
                     uint64_t CurrentWeight) override {
    return 4;
  }

  using IRMutationStrategy::mutate;
  void mutate(Instruction &Inst, RandomIRBuilder &IB) override;
};

/// Fuzzer friendly interface for the llvm bitcode parser.
///
/// \param Data Bitcode we are going to parse
/// \param Size Size of the 'Data' in bytes
/// \return New module or nullptr in case of error
std::unique_ptr<Module> parseModule(const uint8_t *Data, size_t Size,
                                    LLVMContext &Context);

/// Fuzzer friendly interface for the llvm bitcode printer.
///
/// \param M Module to print
/// \param Dest Location to store serialized module
/// \param MaxSize Size of the destination buffer
/// \return Number of bytes that were written. When module size exceeds MaxSize
///         returns 0 and leaves Dest unchanged.
size_t writeModule(const Module &M, uint8_t *Dest, size_t MaxSize);

/// Try to parse module and verify it. May output verification errors to the
/// errs().
/// \return New module or nullptr in case of error.
std::unique_ptr<Module> parseAndVerify(const uint8_t *Data, size_t Size,
                                       LLVMContext &Context);

} // namespace llvm

#endif // LLVM_FUZZMUTATE_IRMUTATOR_H
