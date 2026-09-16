//===- LoadStoreVec.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that vectorizes short store-load chains.
// Unlike generic bundle vectorization, this pass can vectorize instructions
// of different types.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

#include <optional>

namespace llvm {

class DataLayout;

namespace sandboxir {

class Context;
class Function;
class Value;
class Instruction;
class Type;

class LLVM_ABI LoadStoreVec final : public RegionPass {
  const DataLayout *DL = nullptr;
  /// The region saved by saveIR(), used by acceptOrRevert().
  Region *Rgn = nullptr;
  /// The region's cost at the time of saveIR().
  InstructionCost CostBefore = 0;
  const Analyses *A = nullptr;
  Context *Ctx = nullptr;
  std::unique_ptr<Scheduler> Sched;
  VecUtils::DeadInstructionMorgue DeadInstrMorgue;

  /// Saves the IR along with the current cost of \p Rgn, so that
  /// acceptOrRevert() can tell whether vectorizing was profitable.
  void saveIR(Region &Rgn);

  /// Accepts the transaction saved by saveIR() if vectorizing was profitable,
  /// reverts it otherwise. \returns true if the transaction was accepted.
  bool acceptOrRevert();

  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(BndlRef<Instruction *> Bndl);

  /// Builds a single vector load out of \p Loads. \returns the new load,
  /// or nullptr if \p Loads are not a vectorizable.
  LoadInst *createVectorLoad(BndlRef<Instruction *> Loads);

  /// Builds a ConstantVector from per-lane constant store operands in \p
  /// Constants. \returns the packed ConstantVector.
  Value *createConstantVector(BndlRef<Value *> Constants);

  /// Vectorizes \p Stores and their operands if constants or consecutive
  /// loads. \returns true on success.
  bool vectorizeStores(BndlRef<Instruction *> Stores, Region &Rgn);

  /// Vectorizes \p Loads into a single load. \return the packed load.
  LoadInst *vectorizeLoads(BndlRef<Instruction *> Loads, Region &Rgn);

public:
  LoadStoreVec(StringRef AuxArg) : RegionPass("load-store-vec") {
    assert(AuxArg.empty() && "This pass ignores aux arg!");
  }
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
