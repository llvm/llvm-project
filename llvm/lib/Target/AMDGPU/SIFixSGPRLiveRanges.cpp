//===-- SIFixSGPRLiveRanges.cpp - Fix SGPR live ranges ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file SALU instructions ignore the execution mask, so we need to modify the
/// live ranges of the registers they define in some cases.
///
/// The main case we need to handle is when a def is used in one side of a
/// branch and not another.  For example:
///
/// %def
/// IF
///   ...
///   ...
/// ELSE
///   %use
///   ...
/// ENDIF
///
/// Here we need the register allocator to avoid assigning any of the defs
/// inside of the IF to the same register as %def.  In traditional live
/// interval analysis %def is not live inside the IF branch, however, since
/// SALU instructions inside of IF will be executed even if the branch is not
/// taken, there is the chance that one of the instructions will overwrite the
/// value of %def, so the use in ELSE will see the wrong value.
///
/// The strategy we use for solving this is to add an extra use after the ENDIF:
///
/// %def
/// IF
///   ...
///   ...
/// ELSE
///   %use
///   ...
/// ENDIF
/// %use
///
/// Adding this use will make the def live throughout the IF branch, which is
/// what we want.

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sgpr-live-ranges"

namespace {

class SIFixSGPRLiveRanges : public MachineFunctionPass {
public:
  static char ID;

public:
  SIFixSGPRLiveRanges() : MachineFunctionPass(ID) {
    initializeSIFixSGPRLiveRangesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fix SGPR live ranges";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveVariables>();
    AU.addPreserved<LiveVariables>();

    AU.addRequired<MachinePostDominatorTree>();
    AU.addPreserved<MachinePostDominatorTree>();
    AU.setPreservesCFG();

    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIFixSGPRLiveRanges, DEBUG_TYPE,
                      "SI Fix SGPR Live Ranges", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariables)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(SIFixSGPRLiveRanges, DEBUG_TYPE,
                    "SI Fix SGPR Live Ranges", false, false)

char SIFixSGPRLiveRanges::ID = 0;

char &llvm::SIFixSGPRLiveRangesID = SIFixSGPRLiveRanges::ID;

FunctionPass *llvm::createSIFixSGPRLiveRangesPass() {
  return new SIFixSGPRLiveRanges();
}

static bool hasOnlyScalarBr(const MachineBasicBlock *MBB,
                            const SIInstrInfo *TII) {
  for (MachineBasicBlock::const_iterator I = MBB->getFirstTerminator(),
                                         E = MBB->end(); I != E; ++I) {
    if (!TII->isSOPP(*I))
      return false;
  }
  return true;
}

bool SIFixSGPRLiveRanges::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII =
      static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const SIRegisterInfo *TRI = static_cast<const SIRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  bool MadeChange = false;

  MachinePostDominatorTree *PDT = &getAnalysis<MachinePostDominatorTree>();
  SmallVector<unsigned, 16> SGPRLiveRanges;

  LiveVariables *LV = &getAnalysis<LiveVariables>();
  MachineBasicBlock *Entry = &MF.front();

  // Use a depth first order so that in SSA, we encounter all defs before
  // uses. Once the defs of the block have been found, attempt to insert
  // SGPR_USE instructions in successor blocks if required.
  for (MachineBasicBlock *MBB : depth_first(Entry)) {
    for (const MachineInstr &MI : *MBB) {
      for (const MachineOperand &MO : MI.defs()) {
        // We should never see a live out def of a physical register, so we also
        // do not need to worry about implicit_defs().
        unsigned Def = MO.getReg();
        if (TargetRegisterInfo::isVirtualRegister(Def)) {
          if (TRI->isSGPRClass(MRI.getRegClass(Def))) {
            // Only consider defs that are live outs. We don't care about def /
            // use within the same block.

            // LiveVariables does not consider registers that are only used in a
            // phi in a sucessor block as live out, unlike LiveIntervals.
            //
            // This is OK because SIFixSGPRCopies replaced any SGPR phis with
            // VGPRs.
            if (LV->isLiveOut(Def, *MBB))
              SGPRLiveRanges.push_back(Def);
          }
        }
      }
    }

    if (MBB->succ_size() < 2 || hasOnlyScalarBr(MBB, TII))
      continue;

    // We have structured control flow, so the number of successors should be
    // two.
    assert(MBB->succ_size() == 2);
    MachineBasicBlock *SuccA = *MBB->succ_begin();
    MachineBasicBlock *SuccB = *(++MBB->succ_begin());
    MachineBasicBlock *NCD = PDT->findNearestCommonDominator(SuccA, SuccB);

    if (!NCD)
      continue;

    MachineBasicBlock::iterator NCDTerm = NCD->getFirstTerminator();

    if (NCDTerm != NCD->end() && NCDTerm->getOpcode() == AMDGPU::SI_ELSE) {
      assert(NCD->succ_size() == 2);
      // We want to make sure we insert the Use after the ENDIF, not after
      // the ELSE.
      NCD = PDT->findNearestCommonDominator(*NCD->succ_begin(),
                                            *(++NCD->succ_begin()));
    }

    for (unsigned Reg : SGPRLiveRanges) {
      // FIXME: We could be smarter here. If the register is Live-In to one
      // block, but the other doesn't have any SGPR defs, then there won't be a
      // conflict. Also, if the branch condition is uniform then there will be
      // no conflict.
      bool LiveInToA = LV->isLiveIn(Reg, *SuccA);
      bool LiveInToB = LV->isLiveIn(Reg, *SuccB);

      if (!LiveInToA && !LiveInToB) {
        DEBUG(dbgs() << PrintReg(Reg, TRI, 0)
              << " is live into neither successor\n");
        continue;
      }

      if (LiveInToA && LiveInToB) {
        DEBUG(dbgs() << PrintReg(Reg, TRI, 0)
              << " is live into both successors\n");
        continue;
      }

      // This interval is live in to one successor, but not the other, so
      // we need to update its range so it is live in to both.
      DEBUG(dbgs() << "Possible SGPR conflict detected for "
            << PrintReg(Reg, TRI, 0)
            << " BB#" << SuccA->getNumber()
            << ", BB#" << SuccB->getNumber()
            << " with NCD = BB#" << NCD->getNumber() << '\n');

      assert(TargetRegisterInfo::isVirtualRegister(Reg) &&
             "Not expecting to extend live range of physreg");

      // FIXME: Need to figure out how to update LiveRange here so this pass
      // will be able to preserve LiveInterval analysis.
      MachineInstr *NCDSGPRUse =
        BuildMI(*NCD, NCD->getFirstNonPHI(), DebugLoc(),
                TII->get(AMDGPU::SGPR_USE))
        .addReg(Reg, RegState::Implicit);

      MadeChange = true;
      LV->HandleVirtRegUse(Reg, NCD, NCDSGPRUse);

      DEBUG(NCDSGPRUse->dump());
    }
  }

  return MadeChange;
}
