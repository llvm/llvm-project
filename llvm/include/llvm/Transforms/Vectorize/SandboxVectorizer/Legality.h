//===- Legality.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legality checks for the Sandbox Vectorizer.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H

#include "llvm/SandboxIR/SandboxIR.h"

namespace llvm::sandboxir {

class LegalityAnalysis;

enum class LegalityResultID {
  Widen, ///> Vectorize by combining scalars to a vector.
};

/// The legality outcome is represented by a class rather than an enum class
/// because in some cases the legality checks are expensive and look for a
/// particular instruction that can be passed along to the vectorizer to avoid
/// repeating the same expensive computation.
class LegalityResult {
protected:
  LegalityResultID ID;
  /// Only Legality can create LegalityResults.
  LegalityResult(LegalityResultID ID) : ID(ID) {}
  friend class LegalityAnalysis;

public:
  LegalityResultID getSubclassID() const { return ID; }
};

class Widen final : public LegalityResult {
  friend class LegalityAnalysis;
  Widen() : LegalityResult(LegalityResultID::Widen) {}

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::Widen;
  }
};

/// Performs the legality analysis and returns a LegalityResult object.
class LegalityAnalysis {
public:
  LegalityAnalysis() = default;
  LegalityResult canVectorize(ArrayRef<Value *> Bndl) {
    // TODO: For now everything is legal.
    return Widen();
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H
