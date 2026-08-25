//===- AMDGPUPrivateObjectVGPRs.cpp - Private object VGPRs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Mark the physical VGPRs an object in the VGPR ("as memory") address space
/// was allocated to as used, by giving the VGPR_LIFETIME_{START,END} pseudos
/// implicit operands and adding the registers to block live-ins, so register
/// allocation cannot hand them to anything else while the object is live.
///
/// AMDGPUPromoteAlloca records the allocation on the alloca as
/// !amdgpu.allocated.vgprs; it reaches this pass through the markers' MMO.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineInstrs.h"
#include "AMDGPUMemoryUtils.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-private-object-vgprs"

using ObjectRegs = SmallVector<MCPhysReg, 50>;

namespace {

class AMDGPUPrivateObjectVGPRsImpl {
public:
  AMDGPUPrivateObjectVGPRsImpl(const MachineFunction &MF, LiveIntervals *LIS)
      : TII(MF.getSubtarget<GCNSubtarget>().getInstrInfo()), LIS(LIS) {}

  bool run(MachineFunction &MF);

private:
  struct AllocaBBInfo {
    const AllocaInst *Alloca = nullptr;
    bool LiveIn = false;
    bool Starts = false;
    bool Ends = false;
  };

  ObjectRegs computeObjectRegs(const AllocaInst &Alloca) const;

  const SIInstrInfo *TII;
  LiveIntervals *LIS;

  DenseMap<const AllocaInst *, std::pair<ObjectRegs, MachineMemOperand *>>
      AllocaObjectRegs;
};

} // end anonymous namespace

// The registers an object occupies: one per dword, starting at the register its
// byte address in the address space names.
ObjectRegs AMDGPUPrivateObjectVGPRsImpl::computeObjectRegs(
    const AllocaInst &Alloca) const {
  const auto &MD = AMDGPU::AllocatedVGPRsMetadata::get(Alloca);
  unsigned Offset = MD.getAddress();
  assert(Offset % 4 == 0 && "Object is not dword aligned");

  // An object that does not fill its last register still owns the whole of it.
  unsigned BaseRegIdx = Offset / 4;
  unsigned NumRegs = divideCeil(MD.getSize(), 4);

  ObjectRegs Regs;
  Regs.reserve(NumRegs);
  for (unsigned I : seq(NumRegs))
    Regs.push_back(AMDGPU::VGPR_32RegClass.getRegister(BaseRegIdx + I));

  return Regs;
}

bool AMDGPUPrivateObjectVGPRsImpl::run(MachineFunction &MF) {
  // Reverse post-order, for the live-out/live-in propagation below.
  DenseMap<MachineBasicBlock *, unsigned> BlockToIndex;
  SmallVector<MachineBasicBlock *> IndexToBlock;
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF.begin());
  for (MachineBasicBlock *MBB : RPOT) {
    BlockToIndex[MBB] = IndexToBlock.size();
    IndexToBlock.push_back(MBB);
  }

  // Fixed-point iteration for block live-ins. The first pass also scans
  // instructions, augmenting the markers and recording per-block state.
  SmallVector<SmallVector<AllocaBBInfo>> BBInfos(IndexToBlock.size());
  SmallBitVector Worklist(IndexToBlock.size());
  bool Changed = false;

  for (bool Dirty = true, FirstPass = true; Dirty; FirstPass = false) {
    Dirty = false;

    for (auto [MBBI, MBB] : enumerate(IndexToBlock)) {
      auto &BBI = BBInfos[MBBI];

      if (FirstPass) {
        for (MachineInstr &MI : *MBB) {
          if (MI.getOpcode() != AMDGPU::VGPR_LIFETIME_START &&
              MI.getOpcode() != AMDGPU::VGPR_LIFETIME_END)
            continue;

          bool IsStart = MI.getOpcode() == AMDGPU::VGPR_LIFETIME_START;
          MachineMemOperand *MMO = *MI.memoperands_begin();
          const auto *Alloca = cast<AllocaInst>(MMO->getValue());

          auto ObjRegsIt = AllocaObjectRegs.find(Alloca);
          if (ObjRegsIt == AllocaObjectRegs.end())
            ObjRegsIt =
                AllocaObjectRegs
                    .try_emplace(Alloca, computeObjectRegs(*Alloca), MMO)
                    .first;

          for (MCPhysReg Reg : ObjRegsIt->second.first)
            MI.addOperand(MachineOperand::CreateReg(
                Reg, /*isDef=*/IsStart, /*isImp=*/true, /*isKill=*/!IsStart));

          auto It = find_if(BBI, [&](const AllocaBBInfo &Info) {
            return Info.Alloca == Alloca;
          });
          if (It == BBI.end()) {
            BBI.push_back({Alloca, false, false, false});
            It = std::prev(BBI.end());
          }
          It->Starts = IsStart;
          It->Ends = !IsStart;

          Changed = true;
        }
      } else {
        if (!Worklist[MBBI])
          continue;
        Worklist[MBBI] = false;
      }

      for (const auto &ABBI : BBI) {
        if (!((ABBI.LiveIn && !ABBI.Ends) || ABBI.Starts))
          continue;

        const ObjectRegs &Regs = AllocaObjectRegs.at(ABBI.Alloca).first;
        for (MachineBasicBlock *Succ : MBB->successors()) {
          unsigned SuccI = BlockToIndex.at(Succ);
          auto &SuccBBI = BBInfos[SuccI];
          auto It = find_if(SuccBBI, [&](const AllocaBBInfo &Info) {
            return Info.Alloca == ABBI.Alloca;
          });

          bool Update = false;
          if (It == SuccBBI.end()) {
            SuccBBI.push_back({ABBI.Alloca, true, false, false});
            It = std::prev(SuccBBI.end());
            Update = true;
          } else if (!It->LiveIn) {
            It->LiveIn = true;
            Update = true;
          }

          if (!Update)
            continue;

          // A block whose own marker decides the matter does not depend on
          // this live-in, so it needs no queueing.
          //
          // Later passes visit queued blocks alone, with none of the reverse
          // post-order sweep that revisits a successor for free, so every
          // changed successor must be queued - not just back edges. A start
          // inside a loop reaches the header only on the second pass, which
          // is exactly where that matters.
          if (!It->Starts && !It->Ends && (SuccI < MBBI || !FirstPass)) {
            Worklist[SuccI] = true;
            Dirty = true;
          }

          for (MCPhysReg Reg : Regs)
            Succ->addLiveIn(Reg);
        }
      }
    }
  }

  // A lifetime.start without a matching end is legal in the incoming IR, so
  // close the object off in every block that ends the function.
  for (auto [BBIdx, MBB] : enumerate(IndexToBlock)) {
    if (!MBB->succ_empty())
      continue;

    for (const auto &ABBI : BBInfos[BBIdx]) {
      if (ABBI.Ends || (!ABBI.LiveIn && !ABBI.Starts))
        continue;

      // A COPY to a conflicting physical VGPR may precede the return, so place
      // the end as late as possible: walk back over instructions that cannot
      // observe the object, stopping at anything the object must stay
      // reserved across.
      MachineBasicBlock::iterator IP = MBB->getFirstTerminator();
      while (IP != MBB->begin()) {
        --IP;
        if (IP->mayStore() || IP->mayLoad() || IP->isCall() ||
            IP->hasUnmodeledSideEffects()) {
          ++IP;
          break;
        }
      }

      const auto &[ObjRegs, MMO] = AllocaObjectRegs.at(ABBI.Alloca);
      MachineInstr *MI =
          BuildMI(*MBB, IP, {}, TII->get(AMDGPU::VGPR_LIFETIME_END))
              .addMemOperand(MMO);
      for (MCPhysReg Reg : ObjRegs)
        MI->addOperand(MachineOperand::CreateReg(
            Reg, /*isDef=*/false, /*isImp=*/true, /*isKill=*/true));

      Changed = true;
    }
  }

  // An object lives in caller-saved registers, so being live across a call is
  // diagnosed rather than left to read back whatever the callee left behind.
  // Inline asm is diagnosed too, but only when it clobbers registers the
  // object occupies: it names registers directly, so the liveness above
  // cannot keep it away from them. One diagnostic per object.
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  SmallPtrSet<const AllocaInst *, 4> Diagnosed;
  for (auto [BBIdx, MBB] : enumerate(IndexToBlock)) {
    for (const auto &ABBI : BBInfos[BBIdx]) {
      if (Diagnosed.contains(ABBI.Alloca))
        continue;

      bool Live = ABBI.LiveIn;
      for (const MachineInstr &MI : *MBB) {
        if (const auto *Marker = dyn_cast<AMDGPUMI::VGPRLifetimeInst>(&MI)) {
          if (&Marker->getObject() == ABBI.Alloca)
            Live = Marker->isStart();
          continue;
        }
        if (!Live)
          continue;

        std::string What;
        if (MI.isCall()) {
          What = "is live across a call";
        } else if (MI.isInlineAsm()) {
          const ObjectRegs &Regs = AllocaObjectRegs.at(ABBI.Alloca).first;
          if (any_of(Regs, [&](MCPhysReg Reg) {
                return MI.modifiesRegister(Reg, &TRI);
              })) {
            // Name the registers, so the conflict is actionable: either the
            // asm gives them up or the object is placed elsewhere.
            const auto &MD = AMDGPU::AllocatedVGPRsMetadata::get(*ABBI.Alloca);
            unsigned Begin = MD.getAddress() / 4;
            unsigned End = (MD.getAddress() + MD.getSize() - 1) / 4;
            What = ("at v[" + Twine(Begin) + ":" + Twine(End) +
                    "] is clobbered by inline asm")
                       .str();
          }
        }
        if (What.empty())
          continue;

        const Function &F = MF.getFunction();
        F.getContext().diagnose(DiagnosticInfoUnsupported(
            F,
            Twine("object in the VGPR 'as memory' address space (13) ") + What,
            MI.getDebugLoc()));
        Diagnosed.insert(ABBI.Alloca);
        break;
      }
    }
  }

  // Remove live ranges from LiveIntervals. They will be recalculated lazily.
  if (LIS) {
    for (const auto &[Alloca, RegsAndMMO] : AllocaObjectRegs) {
      for (MCPhysReg Reg : RegsAndMMO.first)
        LIS->removeAllRegUnitsForPhysReg(Reg);
    }
  }

  return Changed;
}

namespace {

class AMDGPUPrivateObjectVGPRsLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrivateObjectVGPRsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    LiveIntervals *LIS = nullptr;
    if (auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>())
      LIS = &LISWrapper->getLIS();
    return AMDGPUPrivateObjectVGPRsImpl(MF, LIS).run(MF);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // LiveVariables only tracks virtual registers and we only touch physical
    // registers.
    AU.addPreserved<LiveVariablesWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "AMDGPU Def/use private object VGPRs";
  }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPUPrivateObjectVGPRsPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  auto *LIS = MFAM.getCachedResult<LiveIntervalsAnalysis>(MF);
  if (!AMDGPUPrivateObjectVGPRsImpl(MF, LIS).run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveVariablesAnalysis>();
  PA.preserve<SlotIndexesAnalysis>();
  PA.preserve<LiveIntervalsAnalysis>();
  return PA;
}

char AMDGPUPrivateObjectVGPRsLegacy::ID = 0;

char &llvm::AMDGPUPrivateObjectVGPRsID = AMDGPUPrivateObjectVGPRsLegacy::ID;

INITIALIZE_PASS(AMDGPUPrivateObjectVGPRsLegacy, DEBUG_TYPE,
                "AMDGPU Def/use private object VGPRs", false, false)
