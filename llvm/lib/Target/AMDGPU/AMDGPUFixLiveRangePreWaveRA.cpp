//===- AMDGPUFixLiveRangePreWaveRA.cpp - Fix Phy-VGPR live-ranges ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass assumes that we have done register-allocation for per-thread
/// values. It extends the live-ranges of those physical VGPRs in order to
/// create the correct interference with those WWM/WQM values during the last
/// register-allocation pass for those WWM/WQM values.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-live-range-pre-wave-ra"

namespace {

class AMDGPUFixLiveRangePreWaveRA : public MachineFunctionPass {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  MachinePostDominatorTree *PDT;

  DenseMap<MachineBasicBlock *, SmallPtrSet<MachineBasicBlock *, 2>> CtrlDeps;

  void buildControlDependences(MachineFunction &MF);
  bool influences(MachineBasicBlock *CtrlMBB, MachineBasicBlock *DepMBB);

public:
  static char ID;

  AMDGPUFixLiveRangePreWaveRA() : MachineFunctionPass(ID) {
    initializeAMDGPUFixLiveRangePreWaveRAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachinePostDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(AMDGPUFixLiveRangePreWaveRA, DEBUG_TYPE,
                      "SI Fix Live Range before Wave-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(AMDGPUFixLiveRangePreWaveRA, DEBUG_TYPE,
                    "SI Fix Live Range before Wave-RA", false, false)

char AMDGPUFixLiveRangePreWaveRA::ID = 0;

char &llvm::AMDGPUFixLiveRangePreWaveRAID = AMDGPUFixLiveRangePreWaveRA::ID;

FunctionPass *llvm::createAMDGPUFixLiveRangePreWaveRAPass() {
  return new AMDGPUFixLiveRangePreWaveRA();
}

static bool MBBHasWWM(const MachineBasicBlock &MBB) {
  for (const MachineInstr &MI : MBB) {
    if (MI.getOpcode() == AMDGPU::V_SET_INACTIVE_B32 ||
        MI.getOpcode() == AMDGPU::V_SET_INACTIVE_B64 ||
        MI.getOpcode() == AMDGPU::SI_SPILL_S32_TO_VGPR ||
        MI.getOpcode() == AMDGPU::ENTER_STRICT_WWM ||
        MI.getOpcode() == AMDGPU::ENTER_STRICT_WQM ||
        MI.getOpcode() == AMDGPU::ENTER_PSEUDO_WM) {
      return true;
    }
  }
  return false;
}

void AMDGPUFixLiveRangePreWaveRA::buildControlDependences(MachineFunction &MF) {
  for (auto *MBB : nodes(&MF)) {
    // skip
    if (MBB->getSingleSuccessor())
      continue;

    // For each successor of MBB
    for (auto *SuccMBB : MBB->successors()) {
      auto *PostDomMBB = PDT->findNearestCommonDominator(MBB, SuccMBB);
      if (PostDomMBB == MBB) {
        if (auto *ParentNode = PDT->getNode(MBB)->getIDom())
          PostDomMBB = ParentNode->getBlock();
      }
      // walk PDT from SuccMBB to PostDomMBB
      // add MBB as the control-parent of the blocks along the path (except
      // PostDomBB)
      for (auto *Node = PDT->getNode(SuccMBB);
           Node && Node->getBlock() != PostDomMBB; Node = Node->getIDom()) {
        auto *PathMBB = Node->getBlock();
        CtrlDeps[PathMBB].insert(MBB);
      }
    }
  }
}

bool AMDGPUFixLiveRangePreWaveRA::influences(MachineBasicBlock *CtrlMBB,
                                             MachineBasicBlock *DepMBB) {
  if (CtrlDeps.find(DepMBB) == CtrlDeps.end())
    return false;

  SmallVector<MachineBasicBlock *, 8> WL;
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  for (auto *ParMBB : CtrlDeps[DepMBB]) {
    WL.push_back(ParMBB);
  }

  while (!WL.empty()) {
    auto *MBB = WL.back();
    WL.pop_back();
    Visited.insert(MBB);
    if (MBB == CtrlMBB)
      return true;
    if (CtrlDeps.find(MBB) != CtrlDeps.end()) {
      for (auto *ParMBB : CtrlDeps[DepMBB]) {
        if (!Visited.count(ParMBB))
          WL.push_back(ParMBB);
      }
    }
  }

  return false;
}

bool AMDGPUFixLiveRangePreWaveRA::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "AMDGPUFixLiveRangePreWaveRA: function " << MF.getName()
                    << "\n");

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  PDT = &getAnalysis<MachinePostDominatorTree>();

  buildControlDependences(MF);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  bool Changed = false;
  for (unsigned i = 0, e = TRI->getNumRegUnits(); i != e; ++i) {
    for (MCRegUnitRootIterator Root(i, TRI); Root.isValid(); ++Root) {
      auto RC = TRI->getPhysRegBaseClass(*Root);
      if (!RC || TRI->isSGPRClass(RC)) {
        // dbgs() << printReg(*Root, TRI) << "\n";
        // skip reg-class that is not relevant
        continue;
      }
      for (MCPhysReg Reg : TRI->superregs_inclusive(*Root)) {
        // if a reg is either not-seen or reserved, not a concern for RA
        if (MRI->reg_empty(Reg) || MRI->isReserved(Reg))
          continue;

        // iterate through the CFG, processing every divergent branch
        for (MachineBasicBlock *MBB : RPOT) {
          MachineBasicBlock *TrueMBB = nullptr;
          MachineBasicBlock *FalseMBB = nullptr;
          SmallVector<MachineOperand, 1> Cond;
          TII->analyzeBranch(*MBB, TrueMBB, FalseMBB, Cond);

          if (!Cond.size())
            break;

          auto CondOpnd = Cond.back();
          if (!FalseMBB)
            FalseMBB = MBB->getNextNode();

          // check if this is a divergent branch
          // is this the right way?
          if (CondOpnd.getReg() != AMDGPU::VCC &&
              CondOpnd.getReg() != AMDGPU::VCC_LO &&
              CondOpnd.getReg() != AMDGPU::VCC_HI &&
              CondOpnd.getReg() != AMDGPU::EXEC &&
              CondOpnd.getReg() != AMDGPU::EXEC_LO &&
              CondOpnd.getReg() != AMDGPU::EXEC_HI)
            continue;

          auto *IPD = PDT->getNode(MBB)->getIDom()->getBlock();
          // is register live at the join-point
          if (!IPD->isLiveIn(Reg))
            continue;

          auto CBR = CondOpnd.getParent();
          // add implicit use if a def is inside the influence region
          bool UseAdded = false;
          for (MachineOperand &MO : MRI->def_operands(Reg)) {
            MachineInstr &MI = *MO.getParent();
            auto DefMBB = MI.getParent();
            if (influences(MBB, DefMBB)) {
              // MI add implicit use for Reg;
              bool UseExists = false;
              for (auto Opnd : MI.all_uses()) {
                if (Opnd.isReg() && Opnd.getReg() == Reg) {
                  UseExists = true;
                  break;
                }
              }
              if (!UseExists) {
                MI.addOperand(MF, MachineOperand::CreateReg(Reg, false, true));
                UseAdded = true;
                Changed = true;
              }
            }
          }
          // add implicit def to branch in order to cap the liveness
          if (UseAdded && !FalseMBB->isLiveIn(Reg) && !TrueMBB->isLiveIn(Reg)) {
            bool DefExists = false;
            for (auto Opnd : CBR->all_defs()) {
              if (Opnd.isReg() && Opnd.getReg() == Reg) {
                DefExists = true;
                break;
              }
            }
            if (!DefExists) {
              CBR->addOperand(MF, MachineOperand::CreateReg(Reg, true, true));
              // should we try to merge implicit-def to make MIR concise?
            }
          }
        } // end the block-loop
      } // end the reg-loop
    } // end the root-loop
  } // end of the unit-loop

  if (Changed) {
    // recompute liveness
    std::vector<MachineBasicBlock *> PostOrder;
    for (auto MBB : reverse(RPOT)) {
      PostOrder.push_back(MBB);
    }
    fullyRecomputeLiveIns(PostOrder);
    for (auto *MBB : RPOT) {
      recomputeLivenessFlags(*MBB);
    }
  }
  CtrlDeps.clear();
  return Changed;
}
