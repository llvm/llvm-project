//===- SLPReductionUtils.h - SLP reduction match helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal header used by SLPVectorizer.cpp. It declares free reduction
// pattern-match helpers that do not depend on BoUpSLP or any other SLP-private
// type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H

namespace llvm {
class Instruction;
class PHINode;
} // namespace llvm

namespace llvm::slpvectorizer {

/// \returns the first operand of \p I that does not match \p Phi. If
/// the operand is not an instruction, returns nullptr.
Instruction *getNonPhiOperand(Instruction *I, PHINode *Phi);

/// \returns true if \p I is a candidate instruction for reduction
/// vectorization.
bool isReductionCandidate(Instruction *I);

} // namespace llvm::slpvectorizer

#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_SLPVECTORIZER_SLPREDUCTIONUTILS_H
