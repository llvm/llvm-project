//===-- AArch64SMEAttributes.cpp - Helper for interpreting SME attributes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64SMEAttributes.h"
#include "llvm/IR/InstrTypes.h"
#include <cassert>

using namespace llvm;

void SMEAttrs::set(unsigned M, bool Enable) {
  if (Enable)
    Bitmask |= M;
  else
    Bitmask &= ~M;

  assert(!(hasStreamingInterface() && hasStreamingCompatibleInterface()) &&
         "SM_Enabled and SM_Compatible are mutually exclusive");
  assert(!(hasNewZAInterface() && hasSharedZAInterface()) &&
         "ZA_New and ZA_Shared are mutually exclusive");
  assert(!(hasNewZAInterface() && preservesZA()) &&
         "ZA_New and ZA_Preserved are mutually exclusive");
}

SMEAttrs::SMEAttrs(const CallBase &CB) {
  *this = SMEAttrs(CB.getAttributes());
  if (auto *F = CB.getCalledFunction())
    set(SMEAttrs(*F).Bitmask);
}

SMEAttrs::SMEAttrs(const AttributeList &Attrs) {
  Bitmask = 0;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_enabled"))
    Bitmask |= SM_Enabled;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_compatible"))
    Bitmask |= SM_Compatible;
  if (Attrs.hasFnAttr("aarch64_pstate_sm_body"))
    Bitmask |= SM_Body;
  if (Attrs.hasFnAttr("aarch64_pstate_za_shared"))
    Bitmask |= ZA_Shared;
  if (Attrs.hasFnAttr("aarch64_pstate_za_new"))
    Bitmask |= ZA_New;
  if (Attrs.hasFnAttr("aarch64_pstate_za_preserved"))
    Bitmask |= ZA_Preserved;
}

std::optional<bool>
SMEAttrs::requiresSMChange(const SMEAttrs &Callee,
                           bool BodyOverridesInterface) const {
  // If the transition is not through a call (e.g. when considering inlining)
  // and Callee has a streaming body, then we can ignore the interface of
  // Callee.
  if (BodyOverridesInterface && Callee.hasStreamingBody()) {
    return hasStreamingInterfaceOrBody() ? std::nullopt
                                         : std::optional<bool>(true);
  }

  if (Callee.hasStreamingCompatibleInterface())
    return std::nullopt;

  // Both non-streaming
  if (hasNonStreamingInterfaceAndBody() && Callee.hasNonStreamingInterface())
    return std::nullopt;

  // Both streaming
  if (hasStreamingInterfaceOrBody() && Callee.hasStreamingInterface())
    return std::nullopt;

  return Callee.hasStreamingInterface();
}
