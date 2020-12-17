//===- PseudoProbe.cpp - Pseudo Probe Helpers -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the helpers to manipulate pseudo probe IR intrinsic
// calls.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PseudoProbe.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"

using namespace llvm;

namespace llvm {

Optional<PseudoProbe> extractProbeFromDiscriminator(const Instruction &Inst) {
  assert(isa<CallBase>(&Inst) && !isa<IntrinsicInst>(&Inst) &&
         "Only call instructions should have pseudo probe encodes as their "
         "Dwarf discriminators");
  if (const DebugLoc &DLoc = Inst.getDebugLoc()) {
    const DILocation *DIL = DLoc;
    auto Discriminator = DIL->getDiscriminator();
    if (DILocation::isPseudoProbeDiscriminator(Discriminator)) {
      PseudoProbe Probe;
      Probe.Id =
          PseudoProbeDwarfDiscriminator::extractProbeIndex(Discriminator);
      Probe.Type =
          PseudoProbeDwarfDiscriminator::extractProbeType(Discriminator);
      Probe.Attr =
          PseudoProbeDwarfDiscriminator::extractProbeAttributes(Discriminator);
      return Probe;
    }
  }
  return None;
}

Optional<PseudoProbe> extractProbe(const Instruction &Inst) {
  if (const auto *II = dyn_cast<PseudoProbeInst>(&Inst)) {
    PseudoProbe Probe;
    Probe.Id = II->getIndex()->getZExtValue();
    Probe.Type = (uint32_t)PseudoProbeType::Block;
    Probe.Attr = II->getAttributes()->getZExtValue();
    return Probe;
  }

  if (isa<CallBase>(&Inst) && !isa<IntrinsicInst>(&Inst))
    return extractProbeFromDiscriminator(Inst);

  return None;
}
} // namespace llvm
