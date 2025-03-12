//===- StructInitVec.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that vectorizes struct initializations.
// Generic bottom-up vectorization cannot handle these because the
// initialization instructions can be of different types.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Pass.h"

namespace llvm {

class DataLayout;

namespace sandboxir {

class Value;
class Instruction;
class Scheduler;
class Type;

class StructInitVec final : public RegionPass {
  const DataLayout *DL = nullptr;
  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(ArrayRef<Instruction *> Bndl,
                                     Scheduler &Sched);

  void tryEraseDeadInstrs(ArrayRef<Instruction *> Stores,
                          ArrayRef<Instruction *> Loads);

public:
  StructInitVec() : RegionPass("struct-init-vec") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
