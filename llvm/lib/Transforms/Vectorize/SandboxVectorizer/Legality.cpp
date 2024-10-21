//===- Legality.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/SandboxIR/Value.h"
#include "llvm/Support/Debug.h"

namespace llvm::sandboxir {

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

LegalityResult &LegalityAnalysis::canVectorize(ArrayRef<Value *> Bndl) {
  if (auto ReasonOpt = notVectorizableBasedOnOpcodesAndTypes(Bndl))
    return createLegalityResult<Pack>(*ReasonOpt);

  // TODO: Check for existing vectors containing values in Bndl.

  // TODO: Check with scheduler.

  return createLegalityResult<Widen>();
}
} // namespace llvm::sandboxir
