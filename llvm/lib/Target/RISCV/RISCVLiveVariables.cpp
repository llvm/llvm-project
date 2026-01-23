//===- RISCVLiveVariables.cpp - Live Variable Analysis for RISC-V --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a live variable analysis pass for the RISC-V backend.
// The pass computes liveness information for virtual and physical registers
// in RISC-V machine functions, optimized for RV64 (64-bit RISC-V architecture).
//
// The analysis performs a backward dataflow analysis to compute
// liveness information. Also updates the kill flags on register operands.
// There is also a verification step to ensure consistency with MBB live-ins.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "riscv-live-variables"
#define RISCV_LIVE_VARIABLES_NAME "RISC-V Live Variable Analysis"

STATISTIC(NumLiveRegsAtEntry, "Number of registers live at function entry");
STATISTIC(NumLiveRegsTotal, "Total number of live registers across all blocks");

static cl::opt<bool> UpdateKills("riscv-liveness-update-kills",
                                 cl::desc("Update kill flags"), cl::init(true),
                                 cl::Hidden);

static cl::opt<bool> UpdateLiveIns("riscv-liveness-update-mbb-liveins",
                                   cl::desc("Update MBB live-in sets"),
                                   cl::init(true), cl::Hidden);

static cl::opt<unsigned> MaxVRegs("riscv-liveness-max-vregs",
                                  cl::desc("Maximum VRegs to track"),
                                  cl::init(1024), cl::Hidden);

static cl::opt<bool> VerifyLiveness("riscv-liveness-verify",
                                    cl::desc("Verify liveness information"),
                                    cl::init(false), cl::Hidden);

namespace {

/// LivenessInfo - Stores liveness information for a basic block
/// TODO: Optimize storage using BitVectors for large register sets.
struct LivenessInfo {
  /// Registers that are live into this block
  /// LiveIn[B] = Use[B] U (LiveOut[B] - Def[B])
  std::set<Register> LiveIn;

  /// Registers that are live out of this block.
  /// LiveOut[B] = U LiveIns[∀ Succ(B)].
  std::set<Register> LiveOut;

  /// Registers that are defined in this block
  std::set<Register> Gen;

  /// Registers that are used in this block before being defined (if at all).
  std::set<Register> Use;
};

class RISCVLiveVariables : public MachineFunctionPass {
public:
  static char ID;

  RISCVLiveVariables(RISCVTargetMachine &TM, bool PreRegAlloc)
      : MachineFunctionPass(ID), TM(TM), PreRegAlloc(PreRegAlloc) {
    initializeRISCVLiveVariablesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_LIVE_VARIABLES_NAME; }

  /// Returns the set of registers live at the entry of the given basic block
  const std::set<Register> &getLiveInSet(const MachineBasicBlock *MBB) const {
    auto It = BlockLiveness.find(MBB);
    assert(It != BlockLiveness.end() && "Block not analyzed");
    return It->second.LiveIn;
  }

  /// Returns the set of registers live at the exit of the given basic block
  const std::set<Register> &getLiveOutSet(const MachineBasicBlock *MBB) const {
    auto It = BlockLiveness.find(MBB);
    assert(It != BlockLiveness.end() && "Block not analyzed");
    return It->second.LiveOut;
  }

  /// Check if a register is live at a specific instruction
  bool isLiveAt(Register Reg, const MachineInstr &MI) const;

  /// Print liveness information for debugging
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  /// Verify the computed liveness information against MBB live-ins.
  /// TODO: Extend verification to live-outs and instruction-level liveness.
  void verifyLiveness(MachineFunction &MF) const;

  /// Mark operands that kill a register
  /// TODO: Add and remove kill flags as necessary.
  bool markKills(MachineFunction &MF);

private:
  /// Compute local liveness information (Use and Def sets) for each block
  void computeLocalLiveness(MachineFunction &MF);

  /// Compute global liveness information (LiveIn and LiveOut sets)
  void computeGlobalLiveness(MachineFunction &MF);

  /// Update MBB live-in sets based on computed liveness information
  bool updateMBBLiveIns(MachineFunction &MF);

  /// Process a single instruction to extract def/use information
  void processInstruction(const MachineInstr &MI, LivenessInfo &Info,
                          const TargetRegisterInfo *TRI);

  /// Check if a register is allocatable (relevant for liveness tracking)
  bool isTrackableRegister(Register Reg, const TargetRegisterInfo *TRI,
                           const MachineRegisterInfo *MRI) const;

  RISCVTargetMachine &TM;
  bool PreRegAlloc;
  unsigned RegCounter = 0;

  // PreRA can have large number of registers and basic block
  // level liveness may be expensive without a bitvector representation.
  std::unordered_map<unsigned, unsigned> TrackedRegisters;

  /// Liveness information for each basic block
  DenseMap<const MachineBasicBlock *, LivenessInfo> BlockLiveness;

  /// Cached pointer to MachineRegisterInfo
  const MachineRegisterInfo *MRI;

  /// Cached pointer to TargetRegisterInfo
  const TargetRegisterInfo *TRI;
};

} // end anonymous namespace

char RISCVLiveVariables::ID = 0;

INITIALIZE_PASS(RISCVLiveVariables, DEBUG_TYPE, RISCV_LIVE_VARIABLES_NAME,
                false, true)

FunctionPass *llvm::createRISCVLiveVariablesPass(RISCVTargetMachine &TM,
                                                 bool PreRegAlloc) {
  return new RISCVLiveVariables(TM, PreRegAlloc);
}

bool RISCVLiveVariables::isTrackableRegister(
    Register Reg, const TargetRegisterInfo *TRI,
    const MachineRegisterInfo *MRI) const {
  // Track all virtual registers but only allocatable physical registers.
  // 1. General purpose registers (X0-X31)
  // 2. Floating point registers (F0-F31)
  // 3. Vector registers if present

  if (Reg.isVirtual())
    return true;

  if (Reg.isPhysical())
    return TRI->isInAllocatableClass(Reg);

  return false;
}

void RISCVLiveVariables::processInstruction(const MachineInstr &MI,
                                            LivenessInfo &Info,
                                            const TargetRegisterInfo *TRI) {
  std::vector<Register> GenVec;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;

    Register Reg = MO.getReg();

    TrackedRegisters.insert(std::pair(Reg, RegCounter++));

    // Skip non-trackable registers
    if (!isTrackableRegister(Reg, TRI, MRI))
      continue;

    if (MO.isUse()) {
      // Only add to Use set if not already defined in this block.
      if (Info.Gen.find(Reg) == Info.Gen.end()) {
        Info.Use.insert(Reg);

        // Also handle sub-registers for physical registers
        if (Reg.isPhysical()) {
          for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/false);
               SubRegs.isValid(); ++SubRegs) {
            if (Info.Gen.find(*SubRegs) == Info.Gen.end()) {
              Info.Use.insert(*SubRegs);
            }
          }
        }
      }
    }

    // Handle implicit operands (like condition codes, stack pointer updates)
    if (MO.isImplicit() && MO.isUse() && Reg.isPhysical()) {
      if (Info.Gen.find(Reg) == Info.Gen.end()) {
        Info.Use.insert(Reg);
      }
    }

    if (MO.isDef()) // Collect defs for later processing.
      GenVec.push_back(Reg);
  }

  for (auto Reg : GenVec) {
    Info.Gen.insert(Reg);
    if (Reg.isPhysical()) {
      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/false);
           SubRegs.isValid(); ++SubRegs) {
        Info.Gen.insert(*SubRegs);
      }
    }
  }

  // Handle RegMasks (from calls) - they kill all non-preserved registers
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask()) {
      const uint32_t *RegMask = MO.getRegMask();

      // Iterate through all physical registers
      for (unsigned PhysReg = 1; PhysReg < TRI->getNumRegs(); ++PhysReg) {
        // If the register is not preserved by this mask, it's clobbered
        if (!MachineOperand::clobbersPhysReg(RegMask, PhysReg))
          continue;

        // Mark as defined (clobbered)
        if (isTrackableRegister(Register(PhysReg), TRI, MRI)) {
          Info.Gen.insert(Register(PhysReg));
        }
      }
    }
  }
}

void RISCVLiveVariables::computeLocalLiveness(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Computing local liveness for " << MF.getName() << "\n");

  // Process each basic block
  for (MachineBasicBlock &MBB : MF) {
    LivenessInfo &Info = BlockLiveness[&MBB];
    Info.Gen.clear();
    Info.Use.clear();

    // Process instructions in forward order to build Use and Def sets
    for (const MachineInstr &MI : MBB) {
      // Skip debug instructions and meta instructions
      if (MI.isDebugInstr() || MI.isMetaInstruction())
        continue;

      processInstruction(MI, Info, TRI);
    }

    LLVM_DEBUG({
      dbgs() << "Block " << MBB.getName() << ":\n";
      dbgs() << "  Use: ";
      for (Register Reg : Info.Use)
        dbgs() << printReg(Reg, TRI) << " ";
      dbgs() << "\n  Def: ";
      for (Register Reg : Info.Gen)
        dbgs() << printReg(Reg, TRI) << " ";
      dbgs() << "\n";
    });
  }
}

void RISCVLiveVariables::computeGlobalLiveness(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Computing global liveness (fixed-point iteration)\n");

  bool Changed = true;
  [[maybe_unused]] unsigned Iterations = 0;

  // Iterate until we reach a fixed point
  // Live-out[B] = Union of Live-in[S] for all successors S of B
  // Live-in[B] = Use[B] Union (Live-out[B] - Def[B])
  while (Changed) {
    Changed = false;
    ++Iterations;

    for (MachineBasicBlock *MBB : post_order(&MF)) {
      LivenessInfo &Info = BlockLiveness[MBB];
      std::set<Register> OldLiveIn = Info.LiveIn;
      std::set<Register> NewLiveOut;

      // Compute Live-out: Union of Live-in of all successors
      for (MachineBasicBlock *Succ : MBB->successors()) {
        LivenessInfo &SuccInfo = BlockLiveness[Succ];
        NewLiveOut.insert(SuccInfo.LiveIn.begin(), SuccInfo.LiveIn.end());
      }

      Info.LiveOut = NewLiveOut;

      // Compute Live-in: Use Union (Live-out - Def)
      std::set<Register> NewLiveIn = Info.Use;

      for (Register Reg : Info.LiveOut) {
        if (Info.Gen.find(Reg) == Info.Gen.end()) {
          NewLiveIn.insert(Reg);
        }
      }

      Info.LiveIn = NewLiveIn;

      // Check if anything changed
      if (Info.LiveIn != OldLiveIn) {
        Changed = true;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "Global liveness converged in " << Iterations
                    << " iterations\n");

  // Update statistics
  for (auto &Entry : BlockLiveness) {
    NumLiveRegsTotal += Entry.second.LiveIn.size();
  }

  // Count live registers at function entry
  if (!MF.empty()) {
    const MachineBasicBlock &EntryBB = MF.front();
    NumLiveRegsAtEntry += BlockLiveness[&EntryBB].LiveIn.size();
  }
}

bool RISCVLiveVariables::updateMBBLiveIns(MachineFunction &MF) {
  // Update each MBB's live-in set based on computed liveness
  // Only update physical register live-ins, as MBB live-in sets
  // track physical registers entering a block
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    auto It = BlockLiveness.find(&MBB);
    if (It == BlockLiveness.end())
      continue;

    const LivenessInfo &Info = It->second;
    Changed = true;

    // Clear existing live-ins
    MBB.clearLiveIns();

    // Add computed live-in physical registers to the MBB
    // Skip sub-registers - only add top-level registers
    // Also skip reserved registers (stack pointer, zero register, etc.)
    for (Register Reg : Info.LiveIn) {
      if (Reg.isPhysical()) {
        MCRegister MCReg = Reg.asMCReg();

        // Skip reserved registers - they're implicitly always live
        if (MRI->isReserved(Reg))
          continue;

        // Only add if this is not a sub-register of another register
        // We want top-level registers only (e.g., $x10, not $x10_w)
        bool IsSubReg = false;
        for (MCSuperRegIterator SR(MCReg, TRI, /*IncludeSelf=*/false);
             SR.isValid(); ++SR) {
          if (Info.LiveIn.count(Register(*SR))) {
            IsSubReg = true;
            break;
          }
        }

        if (!IsSubReg) {
          MBB.addLiveIn(MCReg);
          LLVM_DEBUG(dbgs() << "  Adding live-in " << printReg(Reg, TRI)
                            << " to block " << MBB.getName() << "\n");
        }
      }
    }

    // Sort and unique the live-ins for efficient lookup
    MBB.sortUniqueLiveIns();
  }
  return Changed;
}

bool RISCVLiveVariables::isLiveAt(Register Reg, const MachineInstr &MI) const {
  const MachineBasicBlock *MBB = MI.getParent();
  auto It = BlockLiveness.find(MBB);

  if (It == BlockLiveness.end())
    return false;

  const LivenessInfo &Info = It->second;

  // A register is live at an instruction if:
  // 1. It's in the live-in set of the block, OR
  // 2. It's defined before this instruction and used after (not yet killed)

  // Check if it's live-in to the block
  if (Info.LiveIn.count(Reg))
    return true;

  // For a more precise answer, we'd need to track instruction-level liveness
  // For now, conservatively return true if it's in live-out and not killed yet
  if (Info.LiveOut.count(Reg))
    return true;

  return false;
}

void RISCVLiveVariables::verifyLiveness(MachineFunction &MF) const {
  for (auto &BB : MF) {
    auto BBLiveness = BlockLiveness.find(&BB);
    assert(BBLiveness != BlockLiveness.end() && "Missing Liveness");
    auto &ComputedLivein = BBLiveness->second.LiveIn;
    for (auto &LI : BB.getLiveIns()) {
      if (!ComputedLivein.count(LI.PhysReg)) {
        LLVM_DEBUG(dbgs() << "Warning: Live-in register "
                          << printReg(LI.PhysReg, TRI)
                          << " missing from computed live-in set of block "
                          << BB.getName() << "\n");
        llvm_unreachable("Computed live-in set is inconsistent with MBB.");
      }
    }
  }
}

bool RISCVLiveVariables::markKills(MachineFunction &MF) {
  bool Changed = false;
  auto KillSetSize = PreRegAlloc ? RegCounter : TRI->getNumRegs();
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    // Set all the registers that are not live-out of the block.
    // Since the global liveness is available (even though a bit conservative),
    // this initialization is safe.
    llvm::BitVector KillSet(KillSetSize, true);
    LivenessInfo &Info = BlockLiveness[MBB];

    for (Register Reg : Info.LiveOut) {
      auto RegIdx = PreRegAlloc ? TrackedRegisters[Reg] : Reg.asMCReg().id();
      KillSet.reset(RegIdx);
    }

    for (MachineInstr &MI : reverse(*MBB)) {
      for (MachineOperand &MO : MI.all_defs()) {
        Register Reg = MO.getReg();
        // Does not track physical registers pre-regalloc.
        if ((PreRegAlloc && Reg.isPhysical()) ||
            !isTrackableRegister(Reg, TRI, MRI))
          continue;

        assert(TrackedRegisters.find(Reg) != TrackedRegisters.end() &&
               "Register not tracked");
        auto RegIdx = PreRegAlloc ? TrackedRegisters[Reg] : Reg.asMCReg().id();

        KillSet.set(RegIdx);

        // Also handle sub-registers for physical registers
        if (!PreRegAlloc && Reg.isPhysical()) {
          for (MCRegAliasIterator RA(Reg, TRI, true); RA.isValid(); ++RA)
            KillSet.set(*RA);
        }
      }

      for (MachineOperand &MO : MI.all_uses()) {
        Register Reg = MO.getReg();
        // Does not track physical registers pre-regalloc.
        if ((PreRegAlloc && Reg.isPhysical()) ||
            !isTrackableRegister(Reg, TRI, MRI))
          continue;

        assert(TrackedRegisters.find(Reg) != TrackedRegisters.end() &&
               "Register not tracked");
        auto RegIdx = PreRegAlloc ? TrackedRegisters[Reg] : Reg.asMCReg().id();

        if (KillSet[RegIdx]) {
          if (!MO.isKill() && !MI.isPHI()) {
            MO.setIsKill(true);
            Changed = true;
          }
          LLVM_DEBUG(dbgs() << "Marking kill of " << printReg(Reg, TRI)
                            << " at instruction: " << MI);
          KillSet.reset(RegIdx);
          if (Reg.isPhysical()) {
            for (MCRegAliasIterator RA(Reg, TRI, true); RA.isValid(); ++RA)
              KillSet.reset(*RA);
          }
        }
      }
    }
  }
  return Changed;
}

bool RISCVLiveVariables::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || MF.empty())
    return false;

  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();

  // Verify we're targeting RV64
  if (!TM.getTargetTriple().isRISCV64()) {
    LLVM_DEBUG(dbgs() << "Warning: RISCVLiveVariables only intended for RV64, "
                      << "but running on RV32\n");
    return false;
  }

  MRI = &MF.getRegInfo();
  TRI = Subtarget.getRegisterInfo();

  LLVM_DEBUG(dbgs() << "***** RISC-V Live Variable Analysis *****\n");
  LLVM_DEBUG(dbgs() << "Function: " << MF.getName() << "\n");

  // Clear any previous analysis
  BlockLiveness.clear();

  // Step 1: Compute local liveness (Use and Def sets)
  computeLocalLiveness(MF);

  // Step 2: Compute global liveness (LiveIn and LiveOut sets)
  computeGlobalLiveness(MF);

  bool Changed = false;
  // Step 3: Update live-in sets of MBBs based on computed liveness
  if (UpdateLiveIns)
    Changed = updateMBBLiveIns(MF);

  // Step 4: Mark kill flags on operands
  if (UpdateKills && MaxVRegs >= RegCounter)
    Changed |= markKills(MF);

  LLVM_DEBUG({
    dbgs() << "\n***** Final Liveness Information *****\n";
    print(dbgs());
  });

  if (VerifyLiveness)
    verifyLiveness(MF);

  return Changed;
}

void RISCVLiveVariables::print(raw_ostream &OS, const Module *M) const {
  OS << "RISC-V Live Variable Analysis Results:\n";
  OS << "======================================\n\n";

  for (const auto &Entry : BlockLiveness) {
    const MachineBasicBlock *MBB = Entry.first;
    const LivenessInfo &Info = Entry.second;

    OS << "Block: " << MBB->getName() << " (Number: " << MBB->getNumber()
       << ")\n";

    OS << "  Live-In:  { ";
    for (Register Reg : Info.LiveIn) {
      OS << printReg(Reg, TRI) << " ";
    }
    OS << "}\n";

    OS << "  Live-Out: { ";
    for (Register Reg : Info.LiveOut) {
      OS << printReg(Reg, TRI) << " ";
    }
    OS << "}\n";

    OS << "  Use:      { ";
    for (Register Reg : Info.Use) {
      OS << printReg(Reg, TRI) << " ";
    }
    OS << "}\n";

    OS << "  Def:      { ";
    for (Register Reg : Info.Gen) {
      OS << printReg(Reg, TRI) << " ";
    }
    OS << "}\n\n";
  }
}
