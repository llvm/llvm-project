//===- BlockUniformityProfile.h - Block uniformity from PGO -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide per-(Machine)basic-block uniformity information from PGO profiles.
//
// The source of truth is IR metadata attached during PGO use:
//   - Metadata name: "block-uniformity-profile"
//   - Payload: i1 (true = uniform, false = divergent)
//
// This is intentionally target-agnostic: any backend that produces
// uniformity bits in the profile can attach the same metadata and reuse this
// proxy in codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H
#define LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MachineBasicBlock;
class MachineFunction;
class raw_ostream;

class BlockUniformityProfile {
public:
  LLVM_ABI void compute(const MachineFunction &MF);

  bool hasProfile() const { return HasProfile; }

  // Returns true if the block is considered divergent. If profile exists for
  // the function but a block has no explicit annotation, it is treated as
  // divergent (conservative).
  LLVM_ABI bool isDivergent(const MachineBasicBlock &MBB) const;

  LLVM_ABI void print(raw_ostream &OS, const MachineFunction &MF) const;

private:
  bool HasProfile = false;
  unsigned NumBlockIDs = 0;
  BitVector DivergentBlocks;
};

class BlockUniformityProfileProxy
    : public AnalysisInfoMixin<BlockUniformityProfileProxy> {
  friend AnalysisInfoMixin<BlockUniformityProfileProxy>;
  static AnalysisKey Key;

public:
  using Result = BlockUniformityProfile;
  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

class BlockUniformityProfilePrinterPass
    : public PassInfoMixin<BlockUniformityProfilePrinterPass> {
  raw_ostream &OS;

public:
  explicit BlockUniformityProfilePrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_BLOCKUNIFORMITYPROFILE_H
