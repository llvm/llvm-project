//===- BottomUpVec.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A Bottom-Up Vectorizer pass.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

class BottomUpVec final : public RegionPass {
  bool Change = false;
  std::unique_ptr<LegalityAnalysis> Legality;
  /// The original instructions that are potentially dead after vectorization.
  DenseSet<Instruction *> DeadInstrCandidates;
  /// Maps scalars to vectors.
  std::unique_ptr<InstrMaps> IMaps;

  /// Creates and returns a vector instruction that replaces the instructions in
  /// \p Bndl. \p Operands are the already vectorized operands.
  Value *createVectorInstr(ArrayRef<Value *> Bndl, ArrayRef<Value *> Operands);
  /// Erases all dead instructions from the dead instruction candidates
  /// collected during vectorization.
  void tryEraseDeadInstrs();
  /// Creates a shuffle instruction that shuffles \p VecOp according to \p Mask.
  /// \p UserBB is the block of the user bundle.
  Value *createShuffle(Value *VecOp, const ShuffleMask &Mask,
                       BasicBlock *UserBB);
  /// Packs all elements of \p ToPack into a vector and returns that vector. \p
  /// UserBB is the block of the user bundle.
  Value *createPack(ArrayRef<Value *> ToPack, BasicBlock *UserBB);
  /// After we create vectors for groups of instructions, the original
  /// instructions are potentially dead and may need to be removed. This
  /// function helps collect these instructions (along with the pointer operands
  /// for loads/stores) so that they can be cleaned up later.
  void collectPotentiallyDeadInstrs(ArrayRef<Value *> Bndl);
  /// Recursively try to vectorize \p Bndl and its operands.
  Value *vectorizeRec(ArrayRef<Value *> Bndl, ArrayRef<Value *> UserBndl,
                      unsigned Depth);
  /// Entry point for vectorization starting from \p Seeds.
  bool tryVectorize(ArrayRef<Value *> Seeds);

public:
  BottomUpVec() : RegionPass("bottom-up-vec") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
