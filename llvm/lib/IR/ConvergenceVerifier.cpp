//===- ConvergenceVerifier.cpp - Verify convergence control -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NOTE: Including the following header causes a premature instantiation of the
// template, and the compiler complains about explicit specialization after
// instantiation. So don't include it.
//
// #include "llvm/IR/ConvergenceVerifier.h"
//===----------------------------------------------------------------------===//

#include "llvm/ADT/GenericConvergenceVerifierImpl.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/SSAContext.h"

using namespace llvm;

template <>
const Instruction *
GenericConvergenceVerifier<SSAContext>::findAndCheckConvergenceTokenUsed(
    const Instruction &I) {
  auto *CB = dyn_cast<CallBase>(&I);
  if (!CB)
    return nullptr;

  unsigned Count =
      CB->countOperandBundlesOfType(LLVMContext::OB_convergencectrl);
  CheckOrNull(Count <= 1,
              "The 'convergencetrl' bundle can occur at most once on a call",
              {Context.print(CB)});
  if (!Count)
    return nullptr;

  auto Bundle = CB->getOperandBundle(LLVMContext::OB_convergencectrl);
  CheckOrNull(Bundle->Inputs.size() == 1 &&
                  Bundle->Inputs[0]->getType()->isTokenTy(),
              "The 'convergencectrl' bundle requires exactly one token use.",
              {Context.print(CB)});
  auto *Token = Bundle->Inputs[0].get();
  auto *Def = dyn_cast<Instruction>(Token);

  CheckOrNull(
      Def && isConvergenceControlIntrinsic(SSAContext::getIntrinsicID(*Def)),
      "Convergence control tokens can only be produced by calls to the "
      "convergence control intrinsics.",
      {Context.print(Token), Context.print(&I)});

  if (Def)
    Tokens[&I] = Def;

  return Def;
}

template <>
bool GenericConvergenceVerifier<SSAContext>::isConvergent(
    const InstructionT &I) const {
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    return CB->isConvergent();
  }
  return false;
}

template <>
bool GenericConvergenceVerifier<SSAContext>::isControlledConvergent(
    const InstructionT &I) {
  // First find a token and place it in the map.
  if (findAndCheckConvergenceTokenUsed(I))
    return true;

  // The entry and anchor intrinsics do not use a token, so we do a broad check
  // here. The loop intrinsic will be checked separately for a missing token.
  if (isConvergenceControlIntrinsic(SSAContext::getIntrinsicID(I)))
    return true;

  return false;
}

template class llvm::GenericConvergenceVerifier<SSAContext>;
