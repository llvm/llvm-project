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
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

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
  auto *I0 = cast<Instruction>(Bndl[0]);
  auto Opcode = I0->getOpcode();
  // If they have different opcodes, then we cannot form a vector (for now).
  if (any_of(drop_begin(Bndl), [Opcode](Value *V) {
        return cast<Instruction>(V)->getOpcode() != Opcode;
      }))
    return ResultReason::DiffOpcodes;

  // If not the same scalar type, Pack. This will accept scalars and vectors as
  // long as the element type is the same.
  Type *ElmTy0 = VecUtils::getElementType(Utils::getExpectedType(I0));
  if (any_of(drop_begin(Bndl), [ElmTy0](Value *V) {
        return VecUtils::getElementType(Utils::getExpectedType(V)) != ElmTy0;
      }))
    return ResultReason::DiffTypes;

  // TODO: Missing checks

  return std::nullopt;
}

#ifndef NDEBUG
static void dumpBndl(ArrayRef<Value *> Bndl) {
  for (auto *V : Bndl)
    dbgs() << *V << "\n";
}
#endif // NDEBUG

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
