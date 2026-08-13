//===- GCNBreakLoadClusterDeps.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Post-RA pass that breaks false (WAR/WAW) anti-dependencies on the address
/// computation feeding clusterable memory loads, so that the downstream post-RA
/// load-clustering scheduler can issue the loads as a burst and expose more
/// memory-level parallelism.
///
/// Register allocation may pack the per-lane index/address computation of
/// several adjacent loads into a small set of registers (e.g. funneling every
/// extracted index through a single scratch VGPR, or reusing one address
/// register pair across two loads). Those reuses are anti-dependencies: they
/// serialize the address chains and pin the loads apart even though the loads
/// are semantically independent. The post-RA MachineScheduler's load-cluster
/// mutation cannot rename registers, so it cannot undo them.
///
/// This pass renames the reused registers on those address chains to free
/// registers scavenged from the function, bounded by the VGPR budget of the
/// current occupancy so it never spends registers that would drop the number of
/// concurrent waves. It performs no rescheduling itself: once the false
/// dependencies are gone the existing load-cluster scheduler does the reorder.
//===----------------------------------------------------------------------===//

#include "GCNBreakLoadClusterDeps.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#include <algorithm>
#include <bitset>
#include <unordered_set>
#include <utility>
#include <tuple>
#include <vector>

using std::bitset;
using std::pair;
using std::reverse;
using std::tie;
using std::unordered_set;
using std::vector;

using namespace llvm;

#define DEBUG_TYPE "amdgpu-break-load-cluster-deps"

namespace {

/// Target-independent-of-pass-manager implementation.
class GCNBreakLoadClusterDepsImpl {
  const GCNSubtarget *ST = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  unsigned occupancy_budget;

  bitset<AMDGPU::NUM_TARGET_REGS> getVGPR32Lanes(Register Reg) const;
  pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
  get_uses_and_defs_for(MachineInstr &MI) const;
  Register rename_register(Register from_reg, Register to_reg, Register rename_reg);

public:
  bool run(MachineFunction &MF);
  bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
};

/// Legacy-PM wrapper.
class GCNBreakLoadClusterDepsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNBreakLoadClusterDepsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Break Load Cluster Dependencies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

  // Append the 32-bit VGPR lanes of physical VGPR `Reg` (any width) to `Lanes`.
bitset<AMDGPU::NUM_TARGET_REGS> GCNBreakLoadClusterDepsImpl::getVGPR32Lanes(Register Reg) const {
  bitset<AMDGPU::NUM_TARGET_REGS> to_return;
  assert(Reg.isPhysical());
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  unsigned NumLanes = TRI->getRegSizeInBits(*RC).getFixedValue() / 32;
  if (NumLanes <= 1) { // already a VGPR_32
    to_return[Reg] = true;
    return to_return;
  }
  for (unsigned C = 0; C < NumLanes; ++C)
    to_return[TRI->getSubReg(Reg, TRI->getSubRegFromChannel(C))] = true;
  return to_return;
}

pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
GCNBreakLoadClusterDepsImpl::get_uses_and_defs_for(MachineInstr &MI) const {
  pair<bitset<AMDGPU::NUM_TARGET_REGS>, bitset<AMDGPU::NUM_TARGET_REGS>>
      to_return;
  for (unsigned i = 0; i < MI.getNumExplicitOperands(); i++)
    if (MI.getOperand(i).isReg())
      (*(MI.getOperand(i).isDef()
             ? &to_return.first
         : &to_return.second)) |= getVGPR32Lanes(MI.getOperand(i).getReg());
  return to_return;
}

Register GCNBreakLoadClusterDepsImpl::rename_register(Register from_reg, Register to_reg, Register rename_reg) {
  if (rename_reg == from_reg)
    return to_reg;
  if (unsigned Idx =
      TRI->getSubRegIndex(from_reg.asMCReg(), rename_reg.asMCReg()))
    return TRI->getSubReg(to_reg.asMCReg(), Idx);
  return rename_reg;
}

bool GCNBreakLoadClusterDepsImpl::runOnMachineBasicBlock(
    MachineBasicBlock &MBB) {
  bool to_return = false;

  // Find clusterable loads whose address operands share a register with an
  // earlier load's address, or whose address def chains funnel through a
  // common scratch register (WAR/WAW anti-dependencies).
  vector<MachineInstr *> all_vector_loads;
  for (MachineInstr &MI : MBB)
    if (MI.mayLoad() && MI.getOperand(0).isReg() && TRI->isVGPR(*MRI,MI.getOperand(0).getReg()))
      all_vector_loads.push_back(&MI);

  reverse(all_vector_loads.begin(), all_vector_loads.end()); // efficiency
  bitset<AMDGPU::NUM_TARGET_REGS> used_load_source_physregs, used_load_dest_physregs;
  while (!all_vector_loads.empty()) {
    MachineInstr& vec_load_ins = *all_vector_loads.back();
    bitset<AMDGPU::NUM_TARGET_REGS> ins_defs, ins_uses;
    tie(ins_defs, ins_uses) = get_uses_and_defs_for(vec_load_ins);
    
    // This means the load is not independent from previous loads
    if ((ins_defs & used_load_dest_physregs).any()) {
      used_load_source_physregs.reset();
      used_load_dest_physregs.reset();
      all_vector_loads.pop_back();
      continue;
    }

    // Check if we have something to rename
    bitset<AMDGPU::NUM_TARGET_REGS> war_conflicts =
        ins_uses & used_load_source_physregs;
    while (war_conflicts.any()) {
      Register old_reg = war_conflicts._Find_first();
      bitset<AMDGPU::NUM_TARGET_REGS> old_reg_war_conflicts;
      for (const MachineOperand &operand : vec_load_ins.operands())
        if (operand.isReg() && operand.isUse() &&
            TRI->regsOverlap(old_reg, operand.getReg())) {
          old_reg_war_conflicts |= getVGPR32Lanes(operand.getReg());
          if (TRI->getPhysRegBaseClass(operand.getReg())->getSizeInBits() >
              TRI->getPhysRegBaseClass(old_reg)->getSizeInBits())
            old_reg = operand.getReg();
        }

      bitset<AMDGPU::NUM_TARGET_REGS> rit_reg_war_conflicts;
      for (MachineBasicBlock::reverse_iterator RIt =
               ++vec_load_ins.getReverseIterator();
           RIt != MBB.rend(); ++RIt) {
        if (RIt->modifiesRegister(old_reg, TRI))
          rit_reg_war_conflicts |= get_uses_and_defs_for(*RIt).first;

        if((old_reg_war_conflicts & ~rit_reg_war_conflicts).any())
          continue;
      
        // First, make sure we don't modify EXEC before redefining register.
        bool exec_modified = false, redefined = false;
        for (MachineBasicBlock::iterator It = std::next(RIt->getIterator());
             It != MBB.end(); ++It)
          if (It->definesRegister(AMDGPU::EXEC, TRI)) {
            exec_modified = true;
            break;
          } else if (It->modifiesRegister(old_reg, TRI)) {
            redefined = true;
          }
        
        //Can't do anything if EXEC modified
        if (exec_modified)
          break;
        
        LiveRegUnits LRU(*TRI);
        LRU.addLiveOuts(MBB);
        
        // If we're live out of the block and the conflicing reg hasn't been
        // redefined, we can't do this with a block-local analysis.
        if (!redefined && !LRU.available(old_reg))
          break;

        // Find the instruction which kills the def in RIt
        bitset<AMDGPU::NUM_TARGET_REGS> killed_subregs;
        MachineBasicBlock::iterator KillerIns = vec_load_ins;
        for (MachineBasicBlock::iterator CandidateKiller = std::next(KillerIns);
             CandidateKiller != MBB.end(); ++CandidateKiller) {
          if (CandidateKiller->modifiesRegister(old_reg, TRI)) {
            killed_subregs |= get_uses_and_defs_for(*CandidateKiller).first;
            if((rit_reg_war_conflicts & ~killed_subregs).none())
              break;
          }
          if (CandidateKiller->readsRegister(old_reg, TRI))
            KillerIns = CandidateKiller;
        }
        
        // See what's free
        for (MachineBasicBlock::reverse_iterator LiveRIt = MBB.rbegin(); &*LiveRIt != &*KillerIns; ++LiveRIt)
          LRU.stepBackward(*LiveRIt);
          for (MachineBasicBlock::reverse_iterator AccumIt =
                   KillerIns->getReverseIterator();
               AccumIt != MBB.rend(); ++AccumIt)
            LRU.accumulate(*AccumIt);

          // Iterate over registers in physical register class
          const TargetRegisterClass &DefinedRegClass =
              *TRI->getPhysRegBaseClass(old_reg);
          unsigned i;
          for (i = 0;
               i < DefinedRegClass.getRegisters().size() &&
               i * DefinedRegClass.getSizeInBits() / 32 < occupancy_budget;
               i++)
            if (LRU.available(DefinedRegClass.getRegisters()[i]))
              break;

          // Actually rename the register
          if (i < DefinedRegClass.getRegisters().size() &&
              i * DefinedRegClass.getSizeInBits() / 32 < occupancy_budget) {
            for (unsigned op = 0; op < RIt->getNumExplicitOperands(); op++)
              if (RIt->getOperand(op).isReg() && RIt->getOperand(op).isDef() &&
                  TRI->regsOverlap(RIt->getOperand(op).getReg(),old_reg))
                RIt->getOperand(op).setReg(rename_register(old_reg,DefinedRegClass.getRegisters()[i],RIt->getOperand(op).getReg()));
            for (MachineBasicBlock::iterator RenameIt =
                     std::next(RIt->getIterator());
                 RenameIt != KillerIns; ++RenameIt)
              for (unsigned op = 0; op < RenameIt->getNumExplicitOperands(); op++)
                if (RenameIt->getOperand(op).isReg() &&
                    TRI->regsOverlap(RenameIt->getOperand(op).getReg(),old_reg))
                  RenameIt->getOperand(op).setReg(rename_register(old_reg,DefinedRegClass.getRegisters()[i],RenameIt->getOperand(op).getReg()));
            for (unsigned op = 0; op < KillerIns->getNumExplicitOperands(); op++)
              if (KillerIns->getOperand(op).isReg() &&
                  KillerIns->getOperand(op).isUse() &&
                  TRI->regsOverlap(KillerIns->getOperand(op).getReg(),old_reg))
                KillerIns->getOperand(op).setReg(rename_register(old_reg,DefinedRegClass.getRegisters()[i],KillerIns->getOperand(op).getReg()));
          }

          // If we renamed, we're done.  If we didn't rename, we can't get
          // around this conflict, so we're also done.
          break;
        }

      tie(ins_defs, ins_uses) = get_uses_and_defs_for(vec_load_ins);
      war_conflicts &= ~getVGPR32Lanes(old_reg);
    }

    // Coda
    used_load_dest_physregs |= ins_defs;
    used_load_source_physregs |= ins_uses;
    all_vector_loads.pop_back();
  }

  // Scavenge free VGPRs and rename along each address def chain so the chains
  // become register-disjoint, staying within the budget.  The downstream
  // post-RA load-cluster scheduler then reorders the now independent loads into
  // a burst.
  
  return to_return;
}

bool GCNBreakLoadClusterDepsImpl::run(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TRI = ST->getRegisterInfo();
  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  unsigned DynamicBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->isDynamicVGPREnabled()
          ? MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize()
          : false;
  occupancy_budget = ST->getMaxNumVGPRs(
      ST->getOccupancyWithNumVGPRs(
          TRI->getNumUsedPhysRegs(*MRI, AMDGPU::VGPR_32RegClass),
          DynamicBlockSize),
      DynamicBlockSize);

  bool to_return = false;
  for (MachineBasicBlock &MBB : MF)
    to_return |= runOnMachineBasicBlock(MBB);
  
  return to_return;
}

bool GCNBreakLoadClusterDepsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return GCNBreakLoadClusterDepsImpl().run(MF);
}

PreservedAnalyses
GCNBreakLoadClusterDepsPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!GCNBreakLoadClusterDepsImpl().run(MF))
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char GCNBreakLoadClusterDepsLegacy::ID = 0;

char &llvm::GCNBreakLoadClusterDepsID = GCNBreakLoadClusterDepsLegacy::ID;

INITIALIZE_PASS(GCNBreakLoadClusterDepsLegacy, DEBUG_TYPE,
                "AMDGPU Break Load Cluster Dependencies", false, false)
