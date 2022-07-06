//===- GCNCreateVOPD.cpp - Create VOPD Instructions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Combine VALU pairs into VOPD instructions
/// Only works on wave32
/// Has register requirements, we reject creating VOPD if the requirements are
/// not met.
/// shouldCombineVOPD mutator in postRA machine scheduler puts candidate
/// instructions for VOPD back-to-back
///
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "GCNVOPDUtils.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <utility>

#define DEBUG_TYPE "gcn-create-vopd"
STATISTIC(NumVOPDCreated, "Number of VOPD Insts Created.");

using namespace llvm;

namespace {

class GCNCreateVOPD : public MachineFunctionPass {
private:
public:
  static char ID;
  const GCNSubtarget *ST = nullptr;

  GCNCreateVOPD() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "GCN Create VOPD Instructions";
  }

  bool doReplace(const SIInstrInfo *SII,
                 std::pair<MachineInstr *, MachineInstr *> &Pair) {
    auto *FirstMI = Pair.first;
    auto *SecondMI = Pair.second;
    unsigned Opc1 = FirstMI->getOpcode();
    unsigned Opc2 = SecondMI->getOpcode();
    int NewOpcode = AMDGPU::getVOPDFull(AMDGPU::getVOPDOpcode(Opc1),
                                        AMDGPU::getVOPDOpcode(Opc2));
    assert(NewOpcode != -1 &&
           "Should have previously determined this as a possible VOPD\n");

    auto VOPDInst = BuildMI(*FirstMI->getParent(), FirstMI,
                            FirstMI->getDebugLoc(), SII->get(NewOpcode))
                        .setMIFlags(FirstMI->getFlags() | SecondMI->getFlags());
    VOPDInst.add(FirstMI->getOperand(0))
        .add(SecondMI->getOperand(0))
        .add(FirstMI->getOperand(1));

    switch (Opc1) {
    case AMDGPU::V_MOV_B32_e32:
      break;
    case AMDGPU::V_FMAMK_F32:
    case AMDGPU::V_FMAAK_F32:
      VOPDInst.add(FirstMI->getOperand(2));
      VOPDInst.add(FirstMI->getOperand(3));
      break;
    default:
      VOPDInst.add(FirstMI->getOperand(2));
      break;
    }

    VOPDInst.add(SecondMI->getOperand(1));

    switch (Opc2) {
    case AMDGPU::V_MOV_B32_e32:
      break;
    case AMDGPU::V_FMAMK_F32:
    case AMDGPU::V_FMAAK_F32:
      VOPDInst.add(SecondMI->getOperand(2));
      VOPDInst.add(SecondMI->getOperand(3));
      break;
    default:
      VOPDInst.add(SecondMI->getOperand(2));
      break;
    }

    VOPDInst.copyImplicitOps(*FirstMI);
    VOPDInst.copyImplicitOps(*SecondMI);

    LLVM_DEBUG(dbgs() << "VOPD Fused: " << *VOPDInst << " from\tX: "
                      << *Pair.first << "\tY: " << *Pair.second << "\n");
    FirstMI->eraseFromParent();
    SecondMI->eraseFromParent();
    ++NumVOPDCreated;
    return true;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    ST = &MF.getSubtarget<GCNSubtarget>();
    if (!AMDGPU::hasVOPD(*ST) || !ST->isWave32())
      return false;
    LLVM_DEBUG(dbgs() << "CreateVOPD Pass:\n");

    const SIInstrInfo *SII = ST->getInstrInfo();
    bool Changed = false;

    SmallVector<std::pair<MachineInstr *, MachineInstr *>> ReplaceCandidates;

    for (auto &MBB : MF) {
      auto MII = MBB.begin(), E = MBB.end();
      while (MII != E) {
        auto *FirstMI = &*MII;
        MII = next_nodbg(MII, MBB.end());
        if (MII == MBB.end())
          break;
        if (FirstMI->isDebugInstr())
          continue;
        auto *SecondMI = &*MII;
        unsigned Opc = FirstMI->getOpcode();
        unsigned Opc2 = SecondMI->getOpcode();
        llvm::AMDGPU::CanBeVOPD FirstCanBeVOPD = AMDGPU::getCanBeVOPD(Opc);
        llvm::AMDGPU::CanBeVOPD SecondCanBeVOPD = AMDGPU::getCanBeVOPD(Opc2);
        std::pair<MachineInstr *, MachineInstr *> Pair;

        if (FirstCanBeVOPD.X && SecondCanBeVOPD.Y)
          Pair = {FirstMI, SecondMI};
        else if (FirstCanBeVOPD.Y && SecondCanBeVOPD.X)
          Pair = {SecondMI, FirstMI};
        else
          continue;
        // checkVOPDRegConstraints cares about program order, but doReplace
        // cares about X-Y order in the constituted VOPD
        if (llvm::checkVOPDRegConstraints(*SII, *FirstMI, *SecondMI)) {
          ReplaceCandidates.push_back(Pair);
          ++MII;
        }
      }
    }
    for (auto &Pair : ReplaceCandidates) {
      Changed |= doReplace(SII, Pair);
    }

    return Changed;
  }
};

} // namespace

char GCNCreateVOPD::ID = 0;

char &llvm::GCNCreateVOPDID = GCNCreateVOPD::ID;

INITIALIZE_PASS(GCNCreateVOPD, DEBUG_TYPE, "GCN Create VOPD Instructions",
                false, false)
