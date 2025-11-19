//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

static cl::opt<bool>
    EnableAntiHintsForMFMARegs("amdgpu-anti-hints-for-mfma", cl::Hidden,
                               cl::desc("Enable Anti-Hints for "
                                        "MFMA in GCNPreRAOptimizations stage."),
                               cl::init(true));

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  bool processReg(Register Reg);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {
    initializeGCNPreRAOptimizationsLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

FunctionPass *llvm::createGCNPreRAOptimizationsLegacyPass() {
  return new GCNPreRAOptimizationsLegacy();
}

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  // Add RA anti-hints to reduce MFMA hazard NOPs
  if (EnableAntiHintsForMFMARegs && ST.hasMAIInsts()) {
    // Max lookback window for RAW or WAW hazard
    constexpr unsigned MaxLookbackWindow = 19;
    for (const MachineBasicBlock &MBB : MF) {
      SmallVector<SmallVector<Register, 4>, 16> RecentMFMAs;
      for (const MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;

        // Handle MFMA instructions
        if (SIInstrInfo::isMFMA(MI)) {
          SmallVector<Register, 4> MFMARegisters;
          // Helper to get named operand
          auto collectNamedOperand = [&](AMDGPU::OpName OpName,
                                         const char *OpNameStr) {
            const MachineOperand *MO = TII->getNamedOperand(MI, OpName);
            if (!MO) {
              LLVM_DEBUG(dbgs() << "    Named operand " << OpNameStr
                                << " not found\n");
              return;
            }
            if (MO->isReg() && MO->getReg().isVirtual()) {
              Register Reg = MO->getReg();
              const TargetRegisterClass *RC = MRI->getRegClass(Reg);
              // Only consider VGPRs
              if (TRI->hasVGPRs(RC))
                MFMARegisters.push_back(Reg);
              LLVM_DEBUG(dbgs() << "    Collected " << OpNameStr << " : "
                                << printReg(Reg, TRI) << "\n");
            }
          };

          // Collect destination and source C registers
          collectNamedOperand(AMDGPU::OpName::vdst, "vdst"); // Destination
          collectNamedOperand(AMDGPU::OpName::src2,
                              "src2"); // Matrix C (accumulator)
          if (!MFMARegisters.empty()) {
            RecentMFMAs.emplace_back(std::move(MFMARegisters));
            // Maintain window
            if (RecentMFMAs.size() > MaxLookbackWindow)
              RecentMFMAs.erase(RecentMFMAs.begin());
          }
          continue;
        }
        bool ShouldCheckReuse = MI.mayLoad() || MI.mayStore() || MI.isCopy() ||
                                SIInstrInfo::isVALU(MI);
        // Skip non-relevant instructions, or skip until at least one MFMA is
        // encountered
        if (!ShouldCheckReuse || RecentMFMAs.empty())
          continue;

        // Process operands that might reuse MFMA registers
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;

          const Register CandidateReg = MO.getReg();
          const TargetRegisterClass *CandidateRC =
              MRI->getRegClass(CandidateReg);

          // Only process VGPR registers
          if (!TRI->isVGPRClass(CandidateRC))
            continue;
          for (auto It = RecentMFMAs.rbegin(); It != RecentMFMAs.rend(); ++It) {
            const SmallVector<Register, 4> &MFMARegs = *It;
            for (Register MFMAReg : MFMARegs) {
              // Check if MFMA register is dead at current instruction
              const LiveInterval &MFMAInterval = LIS->getInterval(MFMAReg);
              const SlotIndex CurrentSlot =
                  LIS->getInstructionIndex(MI).getRegSlot();
              if (!MFMAInterval.liveAt(CurrentSlot)) {
                // Add bi-directional anti-hints
                MRI->addRegAllocationAntiHints(CandidateReg, MFMAReg);
                MRI->addRegAllocationAntiHints(MFMAReg, CandidateReg);
              }
            }
          }
        }
      }
    }
  }

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->MC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  if (!ST.useRealTrue16Insts())
    return Changed;

  // Add RA hints to improve True16 COPY elimination.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::COPY)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      Register Src = MI.getOperand(1).getReg();
      if (Dst.isVirtual() &&
          MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          Src.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
      if (Src.isVirtual() &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
          Dst.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Dst) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
      if (!Dst.isVirtual() || !Src.isVirtual())
        continue;
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
        MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
      }
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
    }
  }

  return Changed;
}
