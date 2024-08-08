//=== llvm/Analysis/CSADescriptors.cpp - CSA Descriptors -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file "describes" conditional scalar assignments (CSA).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CSADescriptors.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "csa-descriptors"

CSADescriptor CSADescriptor::isCSAPhi(PHINode *Phi, Loop *TheLoop) {
  // Return CSADescriptor that describes a CSA that matches one of these
  // patterns:
  //   phi loop_inv, (select cmp, value, phi)
  //   phi loop_inv, (select cmp, phi, value)
  //   phi (select cmp, value, phi), loop_inv
  //   phi (select cmp, phi, value), loop_inv
  // If the CSA does not match any of these paterns, return a CSADescriptor
  // that describes an InvalidCSA.

  // Must be a scalar
  Type *Type = Phi->getType();
  if (!Type->isIntegerTy() && !Type->isFloatingPointTy() &&
      !Type->isPointerTy())
    return CSADescriptor();

  // Match phi loop_inv, (select cmp, value, phi)
  //    or phi loop_inv, (select cmp, phi, value)
  //    or phi (select cmp, value, phi), loop_inv
  //    or phi (select cmp, phi, value), loop_inv
  if (Phi->getNumIncomingValues() != 2)
    return CSADescriptor();
  auto SelectInstIt = find_if(Phi->incoming_values(), [&Phi](Use &U) {
    return match(U.get(), m_Select(m_Value(), m_Specific(Phi), m_Value())) ||
           match(U.get(), m_Select(m_Value(), m_Value(), m_Specific(Phi)));
  });
  if (SelectInstIt == Phi->incoming_values().end())
    return CSADescriptor();
  auto LoopInvIt = find_if(Phi->incoming_values(), [&](Use &U) {
    return U.get() != *SelectInstIt && TheLoop->isLoopInvariant(U.get());
  });
  if (LoopInvIt == Phi->incoming_values().end())
    return CSADescriptor();

  // Phi or Sel must be used only outside the loop,
  // excluding if Phi use Sel or Sel use Phi
  auto IsOnlyUsedOutsideLoop = [=](Value *V, Value *Ignore) {
    return all_of(V->users(), [Ignore, TheLoop](User *U) {
      if (U == Ignore)
        return true;
      if (auto *I = dyn_cast<Instruction>(U))
        return !TheLoop->contains(I);
      return true;
    });
  };
  auto *Sel = cast<SelectInst>(SelectInstIt->get());
  auto *LoopInv = LoopInvIt->get();
  if (!IsOnlyUsedOutsideLoop(Phi, Sel) || !IsOnlyUsedOutsideLoop(Sel, Phi))
    return CSADescriptor();

  return CSADescriptor(Phi, Sel, LoopInv);
}
