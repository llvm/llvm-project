//===- SIPreAllocateWWMRegs.cpp - WWM Register Pre-allocation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Pass to pre-allocated WWM registers
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "si-pre-allocate-wwm-regs"

static cl::opt<bool>
    EnablePreallocateSGPRSpillVGPRs("amdgpu-prealloc-sgpr-spill-vgprs",
                                    cl::init(false), cl::Hidden);

namespace {

class SIPreAllocateWWMRegs : public MachineFunctionPass {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  LiveRegMatrix *Matrix;
  VirtRegMap *VRM;
  MachineLoopInfo *MLI;
  RegisterClassInfo RegClassInfo;

  std::vector<Register> RegsToRewrite;
  SmallSet<SlotIndex, 4> CallIndexes;
  SmallSetVector<MCRegister, 16> PhysUsed;
  DenseMap<MCRegister, SmallVector<Register>> Assignments;

#ifndef NDEBUG
  void printWWMInfo(const MachineInstr &MI);
#endif

public:
  static char ID;

  SIPreAllocateWWMRegs() : MachineFunctionPass(ID) {
    initializeSIPreAllocateWWMRegsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMap>();
    AU.addRequired<LiveRegMatrix>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  bool processDef(MachineOperand &MO, bool CanReallocate);
  void rewriteRegs(MachineFunction &MF);
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(SIPreAllocateWWMRegs, DEBUG_TYPE,
                "SI Pre-allocate WWM Registers", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SIPreAllocateWWMRegs, DEBUG_TYPE,
                "SI Pre-allocate WWM Registers", false, false)

char SIPreAllocateWWMRegs::ID = 0;

char &llvm::SIPreAllocateWWMRegsID = SIPreAllocateWWMRegs::ID;

FunctionPass *llvm::createSIPreAllocateWWMRegsPass() {
  return new SIPreAllocateWWMRegs();
}

bool SIPreAllocateWWMRegs::processDef(MachineOperand &MO, bool CanReallocate) {
  Register Reg = MO.getReg();
  if (Reg.isPhysical())
    return false;

  if (!TRI->isVGPR(*MRI, Reg))
    return false;

  if (VRM->hasPhys(Reg))
    return false;

  LiveInterval &LI = LIS->getInterval(Reg);

  for (MCRegister PhysReg : RegClassInfo.getOrder(MRI->getRegClass(Reg))) {
    if (!MRI->isPhysRegUsed(PhysReg, /*SkipRegMaskTest=*/true) &&
        Matrix->checkInterference(LI, PhysReg) == LiveRegMatrix::IK_Free) {
      Matrix->assign(LI, PhysReg);
      assert(PhysReg != 0);
      RegsToRewrite.push_back(Reg);
      if (CanReallocate) {
        Assignments[PhysReg].push_back(Reg);
        PhysUsed.insert(PhysReg);
      }
      return true;
    }
  }

  llvm_unreachable("physreg not found for WWM expression");
}

void SIPreAllocateWWMRegs::rewriteRegs(MachineFunction &MF) {
  // Get first point of divergence, i.e. entry block end,
  // for use as live range extension point.
  MachineBasicBlock *EntryMBB = &MF.front();
  auto EntryInsertPoint = EntryMBB->getFirstTerminator();
  if (EntryInsertPoint != EntryMBB->instr_begin())
    EntryInsertPoint--;

  // For each used PhysReg, see if we can use a single virtual reg instead.
  DenseMap<Register, Register> Reassign;
  SmallVector<Register> NewRegisters;
  for (MCRegister PhysReg : PhysUsed) {
    bool CanMerge = true;

    LLVM_DEBUG(dbgs() << "Try to change " << printReg(PhysReg, TRI)
                      << " to virtual\n");

    // Test if all reallocable intervals for this physical register are
    // suitable for combining.
    SmallVector<MachineInstr *> Defs;
    SlotIndex FirstDef, LastUse;
    for (Register Reg : Assignments[PhysReg]) {
      LiveInterval &LI = LIS->getInterval(Reg);

      // Must have no subranges
      CanMerge = !LI.hasSubRanges();
      if (!CanMerge)
        break;

      // Out of an abundance of caution check that there are no PHI values,
      // and all values beyond the initial definition are tied operands.
      if (!LI.containsOneValue()) {
        for (unsigned Idx = 0; CanMerge && Idx < LI.getNumValNums(); ++Idx) {
          auto *VN = LI.getValNumInfo(Idx);
          MachineInstr *DefMI = LIS->getInstructionFromIndex(VN->def);
          CanMerge = !VN->isPHIDef() && DefMI;
          if (!CanMerge || Idx == 0)
            continue;
          MachineOperand &DefOp = DefMI->getOperand(0);
          CanMerge = DefOp.isReg() && DefOp.getReg() == Reg && DefOp.isTied() &&
                     DefMI->isRegTiedToUseOperand(0);
        }
        if (!CanMerge)
          break;
      }

      // Must be contained in a single basic block
      SlotIndex DefIdx = LI.beginIndex();
      MachineInstr *DefMI = LIS->getInstructionFromIndex(DefIdx);
      MachineBasicBlock *MBB = DefMI->getParent();
      CanMerge = LI.isLocal(LIS->getMBBStartIdx(MBB), LIS->getMBBEndIdx(MBB));
      if (!CanMerge)
        break;

      // Def must not be in a loop
      CanMerge = !MLI->getLoopFor(MBB);
      if (!CanMerge)
        break;

      // Update extents of whole range
      SlotIndex UseIdx = LI.endIndex();
      if (Defs.empty()) {
        FirstDef = DefIdx;
        LastUse = UseIdx;
      } else {
        if (DefIdx < FirstDef)
          FirstDef = DefIdx;
        if (UseIdx > LastUse)
          LastUse = UseIdx;
      }

      // Cache defs for later
      Defs.push_back(DefMI);
    }
    if (!CanMerge) {
      LLVM_DEBUG(dbgs() << "  intervals too complex\n");
      continue;
    }

    // Test that the combined live range would cross no calls.
    for (SlotIndex Idx : CallIndexes) {
      if (Idx > FirstDef && Idx < LastUse) {
        CanMerge = false;
        break;
      }
    }
    if (!CanMerge) {
      LLVM_DEBUG(dbgs() << "  crosses function call\n");
      continue;
    }

    // Replace physical register with a new virtual register.
    // Live range reaches from uniform control flow to last use.
    Register NewReg =
        MRI->createVirtualRegister(MRI->getRegClass(Assignments[PhysReg][0]));
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    MFI->setFlag(NewReg, AMDGPU::VirtRegFlag::WWM_REG);
    NewRegisters.push_back(NewReg);

    LLVM_DEBUG(dbgs() << "New virtual " << printReg(NewReg, TRI) << "\n");

    for (Register Reg : Assignments[PhysReg]) {
      LiveInterval &LI = LIS->getInterval(Reg);
      Matrix->unassign(LI);
      LLVM_DEBUG(dbgs() << "Map " << printReg(Reg, TRI) << " to "
                        << printReg(NewReg, TRI) << "\n");
      Reassign[Reg] = NewReg;
    }

    SlotIndex InsPtIdx = LIS->getInstructionIndex(*EntryInsertPoint);
    if (InsPtIdx < FirstDef) {
      // Implicitly define register at insertion point.
      auto MI = BuildMI(*EntryMBB, EntryInsertPoint, DebugLoc(),
                        TII->get(AMDGPU::IMPLICIT_DEF), NewReg);
      LLVM_DEBUG(dbgs() << "Add implicit def " << MI);
      FirstDef = LIS->InsertMachineInstrInMaps(*MI);
    }
    for (MachineInstr *DefMI : Defs) {
      SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
      if (SlotIndex::isSameInstr(DefIdx, FirstDef))
        continue;
      // Mark implicit previous value as used by next def to extend range.
      DefMI->addOperand(MF,
                        MachineOperand::CreateReg(NewReg, false, true, true));
    }
  }

  PhysUsed.clear();
  Assignments.clear();

  // Expand VRM if we added registers
  if (!NewRegisters.empty())
    VRM->grow();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;

        const Register VirtReg = MO.getReg();
        if (VirtReg.isPhysical())
          continue;

        if (Reassign.contains(VirtReg)) {
          MO.setReg(Reassign[VirtReg]);
          continue;
        }

        if (!VRM->hasPhys(VirtReg))
          continue;
        Register PhysReg = VRM->getPhys(VirtReg);
        const unsigned SubReg = MO.getSubReg();
        if (SubReg != 0) {
          PhysReg = TRI->getSubReg(PhysReg, SubReg);
          MO.setSubReg(0);
        }

        MO.setReg(PhysReg);
        MO.setIsRenamable(false);
      }
    }
  }

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  for (unsigned Reg : RegsToRewrite) {
    LIS->removeInterval(Reg);

    const Register PhysReg = VRM->getPhys(Reg);
    if (PhysReg)
      MFI->reserveWWMRegister(PhysReg);
  }

  RegsToRewrite.clear();

  for (Register Reg : NewRegisters) {
    // Compute new interval and mark to prevent splits and spills.
    LiveInterval &LI = LIS->createAndComputeVirtRegInterval(Reg);
    LI.markNotSpillable();
  }

  // Update the set of reserved registers to include WWM ones.
  MRI->freezeReservedRegs();
}

#ifndef NDEBUG
LLVM_DUMP_METHOD void
SIPreAllocateWWMRegs::printWWMInfo(const MachineInstr &MI) {

  unsigned Opc = MI.getOpcode();

  if (Opc == AMDGPU::ENTER_STRICT_WWM || Opc == AMDGPU::ENTER_STRICT_WQM) {
    dbgs() << "Entering ";
  } else {
    assert(Opc == AMDGPU::EXIT_STRICT_WWM || Opc == AMDGPU::EXIT_STRICT_WQM);
    dbgs() << "Exiting ";
  }

  if (Opc == AMDGPU::ENTER_STRICT_WWM || Opc == AMDGPU::EXIT_STRICT_WWM) {
    dbgs() << "Strict WWM ";
  } else {
    assert(Opc == AMDGPU::ENTER_STRICT_WQM || Opc == AMDGPU::EXIT_STRICT_WQM);
    dbgs() << "Strict WQM ";
  }

  dbgs() << "region: " << MI;
}

#endif

bool SIPreAllocateWWMRegs::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "SIPreAllocateWWMRegs: function " << MF.getName() << "\n");

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();

  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Matrix = &getAnalysis<LiveRegMatrix>();
  VRM = &getAnalysis<VirtRegMap>();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  RegClassInfo.runOnMachineFunction(MF);

  bool PreallocateSGPRSpillVGPRs =
      EnablePreallocateSGPRSpillVGPRs ||
      MF.getFunction().hasFnAttribute("amdgpu-prealloc-sgpr-spill-vgprs");
  bool AllowRealloc =
      MF.getFunction().getCallingConv() == CallingConv::AMDGPU_PS;

  bool RegsAssigned = false;

  // We use a reverse post-order traversal of the control-flow graph to
  // guarantee that we visit definitions in dominance order. Since WWM
  // expressions are guaranteed to never involve phi nodes, and we can only
  // escape WWM through the special WWM instruction, this means that this is a
  // perfect elimination order, so we can never do any better.
  ReversePostOrderTraversal<MachineFunction*> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    bool InWWM = false;
    bool InWQM = false;
    for (MachineInstr &MI : *MBB) {
      if (AllowRealloc && MI.isCall())
        CallIndexes.insert(LIS->getInstructionIndex(MI));

      if (MI.getOpcode() == AMDGPU::V_SET_INACTIVE_B32) {
        RegsAssigned |= processDef(MI.getOperand(0), false);
        // Prevent unexpected reload of operands in WWM region
        auto markUnspillable = [&](MachineOperand &Op) {
          if (Op.isReg() && Op.getReg().isVirtual())
            LIS->getInterval(Op.getReg()).markNotSpillable();
        };
        markUnspillable(MI.getOperand(2));
        markUnspillable(MI.getOperand(4));
      }

      if (MI.getOpcode() == AMDGPU::SI_SPILL_S32_TO_VGPR) {
        if (!PreallocateSGPRSpillVGPRs)
          continue;
        RegsAssigned |= processDef(MI.getOperand(0), false);
      }

      if (MI.getOpcode() == AMDGPU::ENTER_STRICT_WWM ||
          MI.getOpcode() == AMDGPU::ENTER_STRICT_WQM) {
        LLVM_DEBUG(printWWMInfo(MI));
        InWWM = MI.getOpcode() == AMDGPU::ENTER_STRICT_WWM;
        InWQM = MI.getOpcode() == AMDGPU::ENTER_STRICT_WQM;
        continue;
      }

      if (MI.getOpcode() == AMDGPU::EXIT_STRICT_WWM ||
          MI.getOpcode() == AMDGPU::EXIT_STRICT_WQM) {
        LLVM_DEBUG(printWWMInfo(MI));
        InWWM = false;
        InWQM = false;
      }

      if (!InWWM && !InWQM)
        continue;

      LLVM_DEBUG(dbgs() << "Processing " << MI);

      for (MachineOperand &DefOpnd : MI.defs()) {
        RegsAssigned |= processDef(DefOpnd, AllowRealloc && InWQM);
      }
    }
  }

  if (!RegsAssigned)
    return false;

  rewriteRegs(MF);
  return true;
}
