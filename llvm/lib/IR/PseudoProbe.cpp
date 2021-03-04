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
#include <unordered_set>

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
      Probe.Factor =
          PseudoProbeDwarfDiscriminator::extractProbeFactor(Discriminator) /
          (float)PseudoProbeDwarfDiscriminator::FullDistributionFactor;
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
    Probe.Factor = II->getFactor()->getZExtValue() /
                   (float)PseudoProbeFullDistributionFactor;
    return Probe;
  }

  if (isa<CallBase>(&Inst) && !isa<IntrinsicInst>(&Inst))
    return extractProbeFromDiscriminator(Inst);

  return None;
}

void setProbeDistributionFactor(Instruction &Inst, float Factor) {
  assert(Factor >= 0 && Factor <= 1 &&
         "Distribution factor must be in [0, 1.0]");
  if (auto *II = dyn_cast<PseudoProbeInst>(&Inst)) {
    IRBuilder<> Builder(&Inst);
    uint64_t IntFactor = PseudoProbeFullDistributionFactor;
    if (Factor < 1)
      IntFactor *= Factor;
    auto OrigFactor = II->getFactor()->getZExtValue();
    if (IntFactor != OrigFactor)
      II->replaceUsesOfWith(II->getFactor(), Builder.getInt64(IntFactor));
  } else if (isa<CallBase>(&Inst) && !isa<IntrinsicInst>(&Inst)) {
    if (const DebugLoc &DLoc = Inst.getDebugLoc()) {
      const DILocation *DIL = DLoc;
      auto Discriminator = DIL->getDiscriminator();
      if (DILocation::isPseudoProbeDiscriminator(Discriminator)) {
        auto Index =
            PseudoProbeDwarfDiscriminator::extractProbeIndex(Discriminator);
        auto Type =
            PseudoProbeDwarfDiscriminator::extractProbeType(Discriminator);
        auto Attr = PseudoProbeDwarfDiscriminator::extractProbeAttributes(
            Discriminator);
        // Round small factors to 0 to avoid over-counting.
        uint32_t IntFactor =
            PseudoProbeDwarfDiscriminator::FullDistributionFactor;
        if (Factor < 1)
          IntFactor *= Factor;
        uint32_t V = PseudoProbeDwarfDiscriminator::packProbeData(
            Index, Type, Attr, IntFactor);
        DIL = DIL->cloneWithDiscriminator(V);
        Inst.setDebugLoc(DIL);
      }
    }
  }
}

void addPseudoProbeAttribute(PseudoProbeInst &Inst,
                             PseudoProbeAttributes Attr) {
  IRBuilder<> Builder(&Inst);
  uint32_t OldAttr = Inst.getAttributes()->getZExtValue();
  uint32_t NewAttr = OldAttr | (uint32_t)Attr;
  if (OldAttr != NewAttr)
    Inst.replaceUsesOfWith(Inst.getAttributes(), Builder.getInt32(NewAttr));
}

/// A block emptied (i.e., with all instructions moved out of it) won't be
/// sampled at run time. In such cases, AutoFDO will be informed of zero samples
/// collected for the block. This is not accurate and could lead to misleading
/// weights assigned for the block. A way to mitigate that is to treat such
/// block as having unknown counts in the AutoFDO profile loader and allow the
/// counts inference tool a chance to calculate a relatively reasonable weight
/// for it. This can be done by moving all pseudo probes in the emptied block
/// i.e, /c From, to before /c To and tag them dangling. Note that this is
/// not needed for dead blocks which really have a zero weight. It's per
/// transforms to decide whether to call this function or not.
bool moveAndDanglePseudoProbes(BasicBlock *From, Instruction *To) {
  SmallVector<PseudoProbeInst *, 4> ToBeMoved;
  for (auto &I : *From) {
    if (auto *II = dyn_cast<PseudoProbeInst>(&I)) {
      addPseudoProbeAttribute(*II, PseudoProbeAttributes::Dangling);
      ToBeMoved.push_back(II);
    }
  }

  for (auto *I : ToBeMoved)
    I->moveBefore(To);

  return !ToBeMoved.empty();
}

/// Same dangling probes in one blocks are redundant since they all have the
/// same semantic that is to rely on the counts inference too to get reasonable
/// count for the same original block. Therefore, there's no need to keep
/// multiple copies of them.
bool removeRedundantPseudoProbes(BasicBlock *Block) {

  auto Hash = [](const PseudoProbeInst *I) {
    return std::hash<uint64_t>()(I->getFuncGuid()->getZExtValue()) ^
           std::hash<uint64_t>()(I->getIndex()->getZExtValue());
  };

  auto IsEqual = [](const PseudoProbeInst *Left, const PseudoProbeInst *Right) {
    return Left->getFuncGuid() == Right->getFuncGuid() &&
           Left->getIndex() == Right->getIndex() &&
           Left->getAttributes() == Right->getAttributes() &&
           Left->getDebugLoc() == Right->getDebugLoc();
  };

  SmallVector<PseudoProbeInst *, 4> ToBeRemoved;
  std::unordered_set<PseudoProbeInst *, decltype(Hash), decltype(IsEqual)>
      DanglingProbes(0, Hash, IsEqual);

  for (auto &I : *Block) {
    if (auto *II = dyn_cast<PseudoProbeInst>(&I)) {
      if (II->getAttributes()->getZExtValue() &
          (uint32_t)PseudoProbeAttributes::Dangling)
        if (!DanglingProbes.insert(II).second)
          ToBeRemoved.push_back(II);
    }
  }

  for (auto *I : ToBeRemoved)
    I->eraseFromParent();
  return !ToBeRemoved.empty();
}
} // namespace llvm
