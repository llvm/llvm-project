//===- Legality.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/Debug.h"

namespace llvm::sandboxir {

#define DEBUG_TYPE "SBVec:Legality"

#ifndef NDEBUG
void LegalityResult::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

std::optional<ResultReason>
LegalityAnalysis::notVectorizableBasedOnOpcodesAndTypes(
    ArrayRef<Value *> Bndl) {
  // TODO: Unimplemented.
  return std::nullopt;
}

static void dumpBndl(ArrayRef<Value *> Bndl) {
  for (auto *V : Bndl)
    dbgs() << *V << "\n";
}

const LegalityResult &LegalityAnalysis::canVectorize(ArrayRef<Value *> Bndl) {
  // If Bndl contains values other than instructions, we need to Pack.
  if (any_of(Bndl, [](auto *V) { return !isa<Instruction>(V); })) {
    LLVM_DEBUG(dbgs() << "Not vectorizing: Not Instructions:\n";
               dumpBndl(Bndl););
    return createLegalityResult<Pack>(ResultReason::NotInstructions);
  }

  if (auto ReasonOpt = notVectorizableBasedOnOpcodesAndTypes(Bndl))
    return createLegalityResult<Pack>(*ReasonOpt);

  // TODO: Check for existing vectors containing values in Bndl.

  // TODO: Check with scheduler.

  return createLegalityResult<Widen>();
}
} // namespace llvm::sandboxir
