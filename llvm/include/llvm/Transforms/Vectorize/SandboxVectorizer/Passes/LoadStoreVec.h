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

namespace llvm {

class DataLayout;

namespace sandboxir {

class Value;
class Instruction;
class Scheduler;
class Type;

class LLVM_ABI LoadStoreVec final : public RegionPass {
  const DataLayout *DL = nullptr;
  /// The region saved by saveIR(), used by acceptOrRevert().
  Region *SavedRgn = nullptr;
  /// The region's cost at the time of saveIR().
  InstructionCost CostBefore = 0;

  /// Saves the IR along with the current cost of \p Rgn, so that
  /// acceptOrRevert() can tell whether vectorizing was profitable.
  void saveIR(Region &Rgn);

  /// Accepts the transaction saved by saveIR() if vectorizing was profitable,
  /// reverts it otherwise. \Returns true if the transaction was accepted.
  bool acceptOrRevert();

  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(ArrayRef<Instruction *> Bndl,
                                     Scheduler &Sched);

  void tryEraseDeadInstrs(ArrayRef<Instruction *> Stores,
                          ArrayRef<Value *> Operands);

  /// Tries to vectorize the load/store/constant ops chain \p Bndl
  /// into a single vector store. \Returns whether it succeeded.
  bool vectorizeStores(ArrayRef<Instruction *> Bndl, Region &Rgn,
                       Scheduler &Sched, const Analyses &A);

public:
  LoadStoreVec(StringRef AuxArg) : RegionPass("load-store-vec") {
    assert(AuxArg.empty() && "This pass ignores aux arg!");
  }
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
