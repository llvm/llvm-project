#if LLPC_BUILD_NPI
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
#include "SIMachineFunctionInfo.h"
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
  void createIdx0SaveDef(MachineFunction &MF);

  MachineRegisterInfo *MRI;
  const SIInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  SIMachineFunctionInfo *MFI;
  SmallVector<MachineInstr *, 4> InstrsToErase;
  Register Idx0Restore;
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
  // Idx0 is used for dynamic indexing and for
  // accessing wave-private space.
  struct IdxInfo {
    MachineInstr *SetIdxMI; // current setter
    Register IdxSrc;        // its virtual source
    int UseCnt;             // potential uses of this setter
    int UseTS;              // time-stamp for the most recent use
  } IdxInfo[NumIDXReg];
  BitVector ActiveBundleIdxUses(NumIDXReg, false);
  // initialize the tracking struct
  for (int i = 0; i < NumIDXReg; ++i) {
    IdxInfo[i].SetIdxMI = nullptr;
    IdxInfo[i].IdxSrc = 0;
    IdxInfo[i].UseCnt = 0;
    IdxInfo[i].UseTS = 0;
  }
  // iterate the MBB bottom-up from the exit to the entry
  int TimeStamp = 0;
  bool Changed = false;
  for (MachineBasicBlock::reverse_instr_iterator I = MBB.instr_rbegin(),
                                                 E = MBB.instr_rend();
       I != E; ++I) {
    MachineInstr &MI = (*I);
    if (MI.getOpcode() == TargetOpcode::BUNDLE) {
      ActiveBundleIdxUses.reset();
      continue;
    } else if (MI.getOpcode() == AMDGPU::V_LOAD_IDX ||
               MI.getOpcode() == AMDGPU::V_STORE_IDX) {
      auto IdxOpnd = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
      auto SRegIdx = IdxOpnd->getReg();
      auto DefMI = MRI->getVRegDef(SRegIdx);
      Changed = true;
      // first, try to find an existing setter
      bool Found = false;
      for (int i = 0; !Found && i < NumIDXReg; ++i) {
        if (IdxInfo[i].IdxSrc == SRegIdx) {
          // find an active setter
          assert(IdxInfo[i].SetIdxMI);
          // simply redirect the indexing operand to the setter
          MachineOperand &NewOp = IdxInfo[i].SetIdxMI->getOperand(0);
          if (MI.isBundled()) {
            updateReplacedRegInBundle(MI, NewOp, *IdxOpnd, TRI);
            ActiveBundleIdxUses.set(i);
          }
          IdxOpnd->setReg(NewOp.getReg());
          // The def-instr can become dead due to const-folding
          if (MRI->use_nodbg_empty(SRegIdx)) {
            InstrsToErase.push_back(DefMI);
          }
          IdxInfo[i].UseCnt--;
          IdxInfo[i].UseTS = TimeStamp;
          Found = true;
        }
      }
      if (Found)
        continue;
      // second, try to find a free non bundle idx-reg
      int FreeIdx = 0;
      for (int i = 1; !FreeIdx && i < NumIDXReg; ++i) {
        if (IdxInfo[i].SetIdxMI == nullptr) {
          FreeIdx = i;
        }
      }
      bool CanUseIdx0 =
          MI.isBundled() && ActiveBundleIdxUses.count() == (NumIDXReg - 1);
      if (!FreeIdx && !CanUseIdx0) {
        // pick one of the existing setters:
        // first, try to pick one without any remaining use;
        // Second, try to pick one with the minimum TimeStamp;
        // move that setter below this use, then create a new setter for
        // this use
        FreeIdx = 1;
        int MinTS = IdxInfo[FreeIdx].UseTS;
        for (int i = 1; i < NumIDXReg; ++i) {
          // UseCnt is not accurate in a Bundle, but UseTS is still useful
          if (IdxInfo[i].UseCnt <= 0 && !MI.isBundled()) {
            FreeIdx = i;
            break;
          } else if (IdxInfo[i].UseTS < MinTS) {
            FreeIdx = i;
            MinTS = IdxInfo[i].UseTS;
          }
        }
        IdxInfo[FreeIdx].SetIdxMI->removeFromParent();
        MBB.insertAfterBundle(MachineBasicBlock::instr_iterator(&MI),
                              IdxInfo[FreeIdx].SetIdxMI);
      }
      // Create idx[i] = s_set_gpr_idx_u32 SRegIdx (or its imm-src)
      auto InsertPt = MBB.getFirstNonPHI();
      if (FreeIdx == 0) {
        // idx0 setter has to be inserted right before the bundle
        InsertPt = getBundleStart(MI.getIterator());
      } else if (DefMI->getParent() == &MBB && !DefMI->isPHI()) {
        InsertPt = ++(DefMI->getIterator());
      }
      Register RegList[] = {AMDGPU::IDX0, AMDGPU::IDX1, AMDGPU::IDX2,
                            AMDGPU::IDX3};
      assert(FreeIdx >= 0 && FreeIdx < NumIDXReg && "Check bounds of FreeIdx");
      auto Setter =
          BuildMI(MBB, InsertPt, DefMI->getDebugLoc(),
                  TII->get(AMDGPU::S_SET_GPR_IDX_U32), RegList[FreeIdx]);
      if (DefMI->getOpcode() == AMDGPU::S_MOV_B32 &&
          DefMI->getOperand(1).isImm())
        // Fold the constant source from the def to the setter
        Setter.addImm(DefMI->getOperand(1).getImm());
      else
        Setter.addReg(SRegIdx);

      IdxInfo[FreeIdx].SetIdxMI = Setter;
      IdxInfo[FreeIdx].IdxSrc = SRegIdx;
      IdxInfo[FreeIdx].UseTS = TimeStamp;
      MachineOperand &NewOp = IdxInfo[FreeIdx].SetIdxMI->getOperand(0);
      if (MI.isBundled()) {
        ActiveBundleIdxUses.set(FreeIdx);
        updateReplacedRegInBundle(MI, NewOp, *IdxOpnd, TRI);
        if (FreeIdx == 0) {
          MFI->setNeedIdx0Restore(true);
          // insert restore
          auto RestoreInsertPt = getBundleEnd(MI.getIterator());
          BuildMI(MBB, RestoreInsertPt, DefMI->getDebugLoc(),
                  TII->get(AMDGPU::S_SET_GPR_IDX_U32), RegList[FreeIdx])
              .addReg(Idx0Restore);
        }
      }

      IdxOpnd->setReg(NewOp.getReg());
      // The def-instr can become dead due to const-folding
      if (MRI->use_nodbg_empty(SRegIdx)) {
        InstrsToErase.push_back(DefMI);
      }
      // Count the remaining local-uses of the virtual-sreg-idx
      int Cnt = 0;
      for (MachineInstr &Use : MRI->use_instructions(SRegIdx)) {
        if (Use.getParent() == &MBB &&
            (Use.getOpcode() == AMDGPU::V_LOAD_IDX ||
             Use.getOpcode() == AMDGPU::V_STORE_IDX) &&
            TII->getNamedOperand(Use, AMDGPU::OpName::idx)->getReg() ==
                SRegIdx) {
          Cnt++;
        }
      }
      IdxInfo[FreeIdx].UseCnt = Cnt;
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

void AMDGPUIdxRegAlloc::createIdx0SaveDef(MachineFunction &MF) {
  // Create the idx0 save here instead of frame lowering so regalloc is aware
  Register VReg =
      MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0_XEXECRegClass);
  MachineBasicBlock &EntryMBB = MF.front();
  auto InsertPt = EntryMBB.begin();
  BuildMI(EntryMBB, InsertPt, DebugLoc(), TII->get(AMDGPU::S_SET_GPR_IDX_U32),
          AMDGPU::IDX0)
      .addImm(0);
  BuildMI(EntryMBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), VReg)
      .addReg(AMDGPU::IDX0);
  MRI->replaceRegWith(Idx0Restore, VReg);
}

bool AMDGPUIdxRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  if (!ST.hasVGPRIndexingRegisters())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = &TII->getRegisterInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();

  bool Changed = false;
  InstrsToErase.clear();
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);

  if (MFI->getNeedIdx0Restore()) {
    createIdx0SaveDef(MF);
  }

  for (auto I : InstrsToErase)
    I->eraseFromParent();

  return Changed;
}
#endif /* LLPC_BUILD_NPI */
