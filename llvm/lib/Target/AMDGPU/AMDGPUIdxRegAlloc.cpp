//===----------- AMDGPUIdxRegAlloc.cpp - Allocate gpr-idx registers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Assign the physical idx-reg for v_load_idx and v_store_idx by
/// inserting set_gpr_idx_u32 instruction and replace the indexing operand
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-idx-reg-alloc"

namespace {

class AMDGPUIdxRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUIdxRegAlloc() : MachineFunctionPass(ID) {
    initializeAMDGPUIdxRegAllocPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Set GPR Indices"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool processMBB(MachineBasicBlock &MBB);

  const MachineRegisterInfo *MRI;
  const SIInstrInfo *TII;
};

} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUIdxRegAlloc, DEBUG_TYPE, "AMDGPU Set GPR Indices", false,
                false)

char AMDGPUIdxRegAlloc::ID = 0;

char &llvm::AMDGPUIdxRegAllocID = AMDGPUIdxRegAlloc::ID;

FunctionPass *llvm::createAMDGPUIdxRegAllocPass() {
  return new AMDGPUIdxRegAlloc();
}

constexpr int NumIDXReg = 4;

bool AMDGPUIdxRegAlloc::processMBB(MachineBasicBlock &MBB) {
  // GFX13-TODO:
  // Idx0 right now is always reserved. In non-wavegroup mode,
  // Idx0 simply holds constant-zero, perhaps we should
  // think about flexible usage of idx0 to get better result?
  struct IdxInfo {
    MachineInstr *SetIdxMI; // current setter
    Register IdxSrc;        // its virtual source
    int UseCnt;             // potential uses of this setter
    int UseTS;              // time-stamp for the most recent use
  } IdxInfo[NumIDXReg];
  // initialize the tracking struct
  for (int i = 0; i < NumIDXReg; ++i) {
    IdxInfo[i].SetIdxMI = nullptr;
    IdxInfo[i].IdxSrc = 0;
    IdxInfo[i].UseCnt = 0;
    IdxInfo[i].UseTS = 0;
  }
  auto createSetGPRIdx = [&](MachineBasicBlock &MBB, Register IdxSrc, int Idx) {
    auto DefMI = MRI->getVRegDef(IdxSrc);
    auto InsertPt = MBB.getFirstNonPHI();
    if (DefMI->getParent() == &MBB && !DefMI->isPHI()) {
      InsertPt = ++(DefMI->getIterator());
    }
    Register RegList[] = {AMDGPU::IDX0, AMDGPU::IDX1, AMDGPU::IDX2,
                          AMDGPU::IDX3};
    auto Setter = BuildMI(MBB, InsertPt, DefMI->getDebugLoc(),
                          TII->get(AMDGPU::S_SET_GPR_IDX_U32), RegList[Idx])
                      .addReg(IdxSrc);
    // count the number of local idx-uses
    int Cnt = 0;
    for (MachineInstr &Use : MRI->use_instructions(IdxSrc)) {
      if (Use.getParent() == &MBB &&
          (Use.getOpcode() == AMDGPU::V_LOAD_IDX ||
           Use.getOpcode() == AMDGPU::V_STORE_IDX) &&
          TII->getNamedOperand(Use, AMDGPU::OpName::idx)->getReg() == IdxSrc) {
        Cnt++;
      }
    }
    return std::pair(Setter, Cnt);
  };
  // iterate the MBB bottom-up from the exit to the entry
  int TimeStamp = 0;
  bool Changed = false;
  for (MachineBasicBlock::reverse_iterator I = MBB.rbegin(), E = MBB.rend();
       I != E; ++I) {
    MachineInstr &MI = (*I);
    if (MI.getOpcode() == AMDGPU::V_LOAD_IDX ||
        MI.getOpcode() == AMDGPU::V_STORE_IDX) {
      auto IdxOpnd = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
      auto SRegIdx = IdxOpnd->getReg();
      Changed = true;
      // first, try to find an existing setter
      bool Found = false;
      for (int i = 1; !Found && i < NumIDXReg; ++i) {
        if (IdxInfo[i].IdxSrc == SRegIdx) {
          // find an active setter
          assert(IdxInfo[i].SetIdxMI);
          // simply redirect the indexing operand to the setter
          IdxOpnd->setReg(IdxInfo[i].SetIdxMI->getOperand(0).getReg());
          IdxInfo[i].UseCnt--;
          IdxInfo[i].UseTS = TimeStamp;
          Found = true;
        }
      }
      if (Found)
        continue;
      // second, try to find a free idx-reg
      int FreeIdx = 0;
      for (int i = 1; !FreeIdx && i < NumIDXReg; ++i) {
        if (IdxInfo[i].SetIdxMI == nullptr) {
          FreeIdx = i;
        }
      }
      if (!FreeIdx) {
        // pick one of the existing setters:
        // first, try to pick one without any remaining use;
        // Second, try to pick one with the minimum TimeStamp;
        // move that setter below this use, then create a new setter for
        // this use
        FreeIdx = 1;
        int MinTS = IdxInfo[FreeIdx].UseTS;
        for (int i = 1; i < NumIDXReg; ++i) {
          if (IdxInfo[i].UseCnt <= 0) {
            FreeIdx = i;
            break;
          } else if (IdxInfo[i].UseTS < MinTS) {
            FreeIdx = i;
            MinTS = IdxInfo[i].UseTS;
          }
        }
        IdxInfo[FreeIdx].SetIdxMI->removeFromParent();
        MBB.insertAfter(MI.getIterator(), IdxInfo[FreeIdx].SetIdxMI);
      }
      MachineInstr *Setter;
      int Cnt;
      std::tie(Setter, Cnt) = createSetGPRIdx(MBB, SRegIdx, FreeIdx);
      IdxInfo[FreeIdx].SetIdxMI = Setter;
      IdxInfo[FreeIdx].IdxSrc = SRegIdx;
      IdxInfo[FreeIdx].UseCnt = Cnt - 1;
      IdxInfo[FreeIdx].UseTS = TimeStamp;
      IdxOpnd->setReg(IdxInfo[FreeIdx].SetIdxMI->getOperand(0).getReg());
    } else if (MI.getOpcode() == AMDGPU::S_SET_GPR_IDX_U32) {
      // clears the entry when we meet the setter
      for (int i = 0; i < NumIDXReg; ++i) {
        if (IdxInfo[i].SetIdxMI == &MI) {
          IdxInfo[i].SetIdxMI = nullptr;
          IdxInfo[i].IdxSrc = 0;
          IdxInfo[i].UseTS = TimeStamp;
          break;
        }
      }
    }
    TimeStamp++;
  }
  return Changed;
}

bool AMDGPUIdxRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);

  return Changed;
}
