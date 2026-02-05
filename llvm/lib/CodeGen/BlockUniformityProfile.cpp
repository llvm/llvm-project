//===- BlockUniformityProfile.cpp - Block uniformity from PGO -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BlockUniformityProfile.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;

static std::optional<bool> getIRBlockUniformity(const BasicBlock &BB) {
  const Instruction *TI = BB.getTerminator();
  if (!TI)
    return std::nullopt;

  MDNode *MD = TI->getMetadata(LLVMContext::MD_block_uniformity_profile);
  if (!MD)
    return std::nullopt;

  // Metadata format: !{i1 IsUniform} - structural validity assumed (verifier).
  // Returns true if uniform (not divergent).
  return mdconst::extract<ConstantInt>(MD->getOperand(0))->isOne();
}

void BlockUniformityProfile::compute(const MachineFunction &MF) {
  HasProfile = false;
  NumBlockIDs = MF.getNumBlockIDs();
  DivergentBlocks.clear();
  DivergentBlocks.resize(NumBlockIDs);

  // First determine whether any uniformity profile exists for this function.
  for (const MachineBasicBlock &MBB : MF) {
    const BasicBlock *BB = MBB.getBasicBlock();
    if (!BB)
      continue;
    if (getIRBlockUniformity(*BB).has_value()) {
      HasProfile = true;
      break;
    }
  }

  if (!HasProfile)
    return;

  // Conservative behavior: if profile exists for the function but we
  // cannot classify a particular (Machine)basic block, treat it as divergent.
  for (const MachineBasicBlock &MBB : MF) {
    const unsigned Num = MBB.getNumber();
    bool IsDivergent = true;
    if (const BasicBlock *BB = MBB.getBasicBlock()) {
      if (auto U = getIRBlockUniformity(*BB))
        IsDivergent = !*U; // Metadata stores IsUniform, we want IsDivergent
    }
    if (Num < DivergentBlocks.size() && IsDivergent)
      DivergentBlocks.set(Num);
  }
}

void BlockUniformityProfile::print(raw_ostream &OS,
                                   const MachineFunction &MF) const {
  OS << "BlockUniformityProfile for function: ";
  MF.getFunction().printAsOperand(OS, /*PrintType=*/false);
  OS << '\n';
  OS << "HasProfile: " << (HasProfile ? "true" : "false") << '\n';
  if (!HasProfile)
    return;

  for (const MachineBasicBlock &MBB : MF) {
    const BasicBlock *BB = MBB.getBasicBlock();
    if (!BB)
      continue;
    OS << "  " << printMBBReference(MBB);
    if (BB->hasName())
      OS << " (%" << BB->getName() << ")";
    if (auto U = getIRBlockUniformity(*BB)) {
      OS << ": " << (*U ? "uniform" : "divergent") << '\n';
      continue;
    }
    OS << ": no PGO annotation (treated divergent for spill placement)\n";
  }
}

bool BlockUniformityProfile::isDivergent(const MachineBasicBlock &MBB) const {
  if (!HasProfile)
    return false;
  assert(MBB.getParent()->getNumBlockIDs() == NumBlockIDs &&
         "MachineFunction was modified without invalidating "
         "BlockUniformityProfile");
  const unsigned Num = MBB.getNumber();
  assert(Num < DivergentBlocks.size() && "Block number out of range");
  return DivergentBlocks.test(Num);
}

AnalysisKey BlockUniformityProfileProxy::Key;

BlockUniformityProfileProxy::Result
BlockUniformityProfileProxy::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &) {
  BlockUniformityProfile Profile;
  Profile.compute(MF);
  return Profile;
}

PreservedAnalyses
BlockUniformityProfilePrinterPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  auto &Profile = MFAM.getResult<BlockUniformityProfileProxy>(MF);
  Profile.print(OS, MF);
  return PreservedAnalyses::all();
}
