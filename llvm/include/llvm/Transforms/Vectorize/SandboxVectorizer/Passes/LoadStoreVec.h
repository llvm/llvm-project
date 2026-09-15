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
  Region *SavedRgn = nullptr;
  /// The region's cost at the time of saveIR().
  InstructionCost CostBefore = 0;
  const Analyses *A = nullptr;
  Context *Ctx = nullptr;
  std::optional<Scheduler> Sched;
  VecUtils::DeadInstructionMorgue DeadInstrMorgue;

  /// Initializes \c A, \c Ctx, and \c Sched for \p F using \p AnalysesRef.
  void initialize(Function &F, const Analyses &AnalysesRef);

  /// Saves the IR along with the current cost of \p Rgn, so that
  /// acceptOrRevert() can tell whether vectorizing was profitable.
  void saveIR(Region &Rgn);

  /// Accepts the transaction saved by saveIR() if vectorizing was profitable,
  /// reverts it otherwise. \returns true if the transaction was accepted.
  bool acceptOrRevert();

  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(ArrayRef<Instruction *> Bndl);

  /// Builds a single vector load out of the load operands in \p Loads.
  /// \returns the new load, or nullptr if \p Loads are not a vectorizable
  /// load chain.
  LoadInst *createVectorLoad(ArrayRef<Instruction *> Loads);

  /// Builds a ConstantVector from per-lane constant store operands in \p
  /// Constants. Aggregates and sequential constants contribute their elements
  /// in order; aggregate-zero and splat vector constants are expanded to one
  /// element per lane. \returns the packed ConstantVector.
  Value *createConstantVector(ArrayRef<Value *> Constants);

  /// Tries to vectorize the store bundle \p Bndl into a single vector store.
  /// Load value-operands are vectorized via vectorizeLoads(); constant
  /// operands are packed into a ConstantVector. \returns whether it succeeded.
  bool vectorizeStores(ArrayRef<Instruction *> Bndl, Region &Rgn);

  /// Tries to vectorize the load chain \p Loads into a single vector load.
  /// If \p ManageTransaction is true, also unpacks for remaining uses, erases
  /// dead original loads, and save/accept-or-reverts. If false, only emits
  /// the vector load; the caller owns the IR transaction.
  /// \returns the new vector load, or nullptr on failure.
  LoadInst *vectorizeLoads(ArrayRef<Instruction *> Loads, Region &Rgn,
                           bool ManageTransaction = true);

public:
  LoadStoreVec(StringRef AuxArg) : RegionPass("load-store-vec") {
    assert(AuxArg.empty() && "This pass ignores aux arg!");
  }
  bool runOnRegion(Region &Rgn, const Analyses &AnalysesRef) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
