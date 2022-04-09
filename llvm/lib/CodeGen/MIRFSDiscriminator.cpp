//===-------- MIRFSDiscriminator.cpp: Flow Sensitive Discriminator --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of a machine pass that adds the flow
// sensitive discriminator to the instruction debug information.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRFSDiscriminator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SampleProfileLoaderBaseUtil.h"

using namespace llvm;
using namespace sampleprof;
using namespace sampleprofutil;

#define DEBUG_TYPE "mirfs-discriminators"

char MIRAddFSDiscriminators::ID = 0;

INITIALIZE_PASS(MIRAddFSDiscriminators, DEBUG_TYPE,
                "Add MIR Flow Sensitive Discriminators",
                /* cfg = */ false, /* is_analysis = */ false)

char &llvm::MIRAddFSDiscriminatorsID = MIRAddFSDiscriminators::ID;

FunctionPass *llvm::createMIRAddFSDiscriminatorsPass(FSDiscriminatorPass P) {
  return new MIRAddFSDiscriminators(P);
}

// Compute a hash value using debug line number, and the line numbers from the
// inline stack.
static uint64_t getCallStackHash(const MachineBasicBlock &BB,
                                 const MachineInstr &MI,
                                 const DILocation *DIL) {
  auto updateHash = [](const StringRef &Str) -> uint64_t {
    if (Str.empty())
      return 0;
    return MD5Hash(Str);
  };
  uint64_t Ret = updateHash(std::to_string(DIL->getLine()));
  Ret ^= updateHash(BB.getName());
  Ret ^= updateHash(DIL->getScope()->getSubprogram()->getLinkageName());
  for (DIL = DIL->getInlinedAt(); DIL; DIL = DIL->getInlinedAt()) {
    Ret ^= updateHash(std::to_string(DIL->getLine()));
    Ret ^= updateHash(DIL->getScope()->getSubprogram()->getLinkageName());
  }
  return Ret;
}

// Traverse the CFG and assign FD discriminators. If two instructions
// have the same lineno and discriminator, but residing in different BBs,
// the latter instruction will get a new discriminator value. The new
// discriminator keeps the existing discriminator value but sets new bits
// b/w LowBit and HighBit.
bool MIRAddFSDiscriminators::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableFSDiscriminator)
    return false;
  if (!MF.getFunction().isDebugInfoForProfiling())
    return false;

  bool Changed = false;
  using LocationDiscriminator = std::tuple<StringRef, unsigned, unsigned>;
  using BBSet = DenseSet<const MachineBasicBlock *>;
  using LocationDiscriminatorBBMap = DenseMap<LocationDiscriminator, BBSet>;
  using LocationDiscriminatorCurrPassMap =
      DenseMap<LocationDiscriminator, unsigned>;

  LocationDiscriminatorBBMap LDBM;
  LocationDiscriminatorCurrPassMap LDCM;

  // Mask of discriminators before this pass.
  unsigned BitMaskBefore = getN1Bits(LowBit);
  // Mask of discriminators including this pass.
  unsigned BitMaskNow = getN1Bits(HighBit);
  // Mask of discriminators for bits specific to this pass.
  unsigned BitMaskThisPass = BitMaskNow ^ BitMaskBefore;
  unsigned NumNewD = 0;

  LLVM_DEBUG(dbgs() << "MIRAddFSDiscriminators working on Func: "
                    << MF.getFunction().getName() << "\n");
  for (MachineBasicBlock &BB : MF) {
    for (MachineInstr &I : BB) {
      const DILocation *DIL = I.getDebugLoc().get();
      if (!DIL)
        continue;
      unsigned LineNo = DIL->getLine();
      if (LineNo == 0)
        continue;
      unsigned Discriminator = DIL->getDiscriminator();
      LocationDiscriminator LD{DIL->getFilename(), LineNo, Discriminator};
      auto &BBMap = LDBM[LD];
      auto R = BBMap.insert(&BB);
      if (BBMap.size() == 1)
        continue;

      unsigned DiscriminatorCurrPass;
      DiscriminatorCurrPass = R.second ? ++LDCM[LD] : LDCM[LD];
      DiscriminatorCurrPass = DiscriminatorCurrPass << LowBit;
      DiscriminatorCurrPass += getCallStackHash(BB, I, DIL);
      DiscriminatorCurrPass &= BitMaskThisPass;
      unsigned NewD = Discriminator | DiscriminatorCurrPass;
      const auto *const NewDIL = DIL->cloneWithDiscriminator(NewD);
      if (!NewDIL) {
        LLVM_DEBUG(dbgs() << "Could not encode discriminator: "
                          << DIL->getFilename() << ":" << DIL->getLine() << ":"
                          << DIL->getColumn() << ":" << Discriminator << " "
                          << I << "\n");
        continue;
      }

      I.setDebugLoc(NewDIL);
      NumNewD++;
      LLVM_DEBUG(dbgs() << DIL->getFilename() << ":" << DIL->getLine() << ":"
                        << DIL->getColumn() << ": add FS discriminator, from "
                        << Discriminator << " -> " << NewD << "\n");
      Changed = true;
    }
  }

  if (Changed) {
    createFSDiscriminatorVariable(MF.getFunction().getParent());
    LLVM_DEBUG(dbgs() << "Num of FS Discriminators: " << NumNewD << "\n");
    (void) NumNewD;
  }

  return Changed;
}
