//===- AArch64CollectCPSpillInfo.cpp - Collect spilled const pool info ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces reloads of spilled constant pool values with direct loads
// from the constant pool. When all reloads from a spill slot are replaced, the
// spill instruction itself is also eliminated.
//
// Additionally, this pass handles spilled DUP instructions that broadcast a
// constant from a GPR to a vector register. These are rematerialized as
// MOV+DUP sequences instead of reloading from the stack.
//
// This pass runs after register allocation but before pseudo expansion.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64RegisterInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-cp-spill-info"

STATISTIC(NumSpillsTracked, "Number of constant pool spills tracked");
STATISTIC(NumSpillsEliminated, "Number of spill stores eliminated");
STATISTIC(NumReloadsReplaced, "Number of reloads replaced with CP loads");
STATISTIC(NumOrigLoadsEliminated, "Number of original CP loads eliminated");
STATISTIC(NumDUPSpillsTracked, "Number of DUP constant spills tracked");
STATISTIC(NumDUPReloadsReplaced, "Number of DUP reloads replaced");

namespace {

/// Information about a spilled constant pool value.
struct CPSpillInfo {
  unsigned CPIndex;        ///< Constant pool index.
  MachineInstr *ADRPMI;    ///< The original ADRP instruction.
  MachineInstr *LoadMI;    ///< The original LDR instruction.
  SmallVector<MachineInstr *, 4> SpillMIs; ///< The spill (store) instructions.
  unsigned ReloadCount;    ///< Number of reload instructions found.
  unsigned ReplacedCount;  ///< Number of reloads successfully replaced.

  CPSpillInfo(unsigned CPI, MachineInstr *ADRP, MachineInstr *Load,
              MachineInstr *Spill)
      : CPIndex(CPI), ADRPMI(ADRP), LoadMI(Load), ReloadCount(0), ReplacedCount(0) {
    SpillMIs.push_back(Spill);
  }
};

/// Information about a spilled DUP of constant value.
struct DUPSpillInfo {
  uint64_t ImmVal;          ///< The immediate value being DUP'd.
  unsigned DUPOpcode;       ///< The DUP opcode (e.g., DUPv4i32gpr).
  bool Is64Bit;             ///< True if the source GPR is 64-bit.
  SmallVector<MachineInstr *, 4> SpillMIs; ///< The spill (store) instructions.
  SmallVector<MachineInstr *, 4> DefMIs; ///< The MOV/MOVK instructions.
  MachineInstr *DUPMI;      ///< The DUP instruction.
  unsigned ReloadCount;     ///< Number of reload instructions found.
  unsigned ReplacedCount;   ///< Number of reloads successfully replaced.

  DUPSpillInfo(uint64_t Imm, unsigned Opc, bool Is64, MachineInstr *Spill,
               SmallVector<MachineInstr *, 4> Defs, MachineInstr *DUP)
      : ImmVal(Imm), DUPOpcode(Opc), Is64Bit(Is64),
        DefMIs(std::move(Defs)), DUPMI(DUP), ReloadCount(0), ReplacedCount(0) {
    SpillMIs.push_back(Spill);
  }
};

class AArch64CollectCPSpillInfo : public MachineFunctionPass {
public:
  static char ID;

  AArch64CollectCPSpillInfo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Collect Constant Pool Spill Info";
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

private:
  const AArch64InstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineFrameInfo *MFI = nullptr;

  /// Maps frame index to spill information.
  DenseMap<int, CPSpillInfo> FrameIndexToSpillInfo;

  /// Maps frame index to DUP spill information.
  DenseMap<int, DUPSpillInfo> FrameIndexToDUPSpillInfo;

  /// Collect information about which stack slots hold spilled CP values.
  void collectSpillInfo(MachineFunction &MF);

  /// Count reloads from each spilled frame index.
  void countReloads(MachineFunction &MF);

  /// Try to replace reloads from stack with loads from constant pool.
  bool optimizeReloads(MachineFunction &MF);

  /// Eliminate spill stores where all reloads were replaced.
  bool eliminateDeadSpills(MachineFunction &MF);

  /// Check if there's a free GPR for ADRP at the given instruction.
  Register findFreeGPRForADRP(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              LivePhysRegs &LiveRegs);

  /// Get the LDR opcode for loading from constant pool based on the reload.
  unsigned getCPLoadOpcode(unsigned ReloadOpc) const;

  /// Check if an instruction is a DUP from GPR to vector.
  bool isDUPFromGPR(unsigned Opc, bool &Is64Bit) const;

  /// Get the reload size in bytes for a given reload opcode.
  unsigned getReloadSize(unsigned ReloadOpc) const;

  /// Check if an instruction's def register is only used by the given user.
  bool hasOnlyOneUse(MachineInstr *MI, MachineInstr *User) const;

  /// Count all uses of a frame index in the function.
  unsigned countFrameIndexUses(MachineFunction &MF, int FrameIndex) const;
};

} // end anonymous namespace

char AArch64CollectCPSpillInfo::ID = 0;

INITIALIZE_PASS(AArch64CollectCPSpillInfo, "aarch64-cp-spill-info",
                "AArch64 Collect Constant Pool Spill Info", false, false)

void AArch64CollectCPSpillInfo::collectSpillInfo(MachineFunction &MF) {
  // Track register -> (CP index, ADRP MI, LDR MI) mappings.
  // When we see ADRP + LDR pattern loading from CP, track it.
  // When we see a store to stack slot, record the mapping.
  struct RegCPInfo {
    unsigned CPIndex;
    MachineInstr *ADRPMI;
    MachineInstr *LoadMI;
  };
  DenseMap<Register, RegCPInfo> RegToCPInfo;

  // Track GPR -> (immediate value, defining instructions) for DUP tracking.
  struct RegImmInfo {
    uint64_t ImmVal;
    SmallVector<MachineInstr *, 4> DefMIs;
  };
  DenseMap<Register, RegImmInfo> RegToImmInfo;

  // Track vector register -> DUP info for spill tracking.
  struct VecDUPInfo {
    uint64_t ImmVal;
    unsigned DUPOpcode;
    bool Is64Bit;
    SmallVector<MachineInstr *, 4> DefMIs;
    MachineInstr *DUPMI;
  };
  DenseMap<Register, VecDUPInfo> VecToDUPInfo;

  for (MachineBasicBlock &MBB : MF) {
    // Clear register tracking at block boundaries for safety.
    // A more sophisticated analysis could track across blocks.
    RegToCPInfo.clear();
    RegToImmInfo.clear();
    VecToDUPInfo.clear();

    for (MachineInstr &MI : MBB) {
      // Look for ADRP followed by LDR from constant pool.
      // Pattern: ADRP Xd, cpentry@PAGE
      //          LDR  Qd, [Xd, cpentry@PAGEOFF]
      //          STR  Qd, [SP, #imm]  ; spill

      // Track ADRP instructions that reference constant pool.
      if (MI.getOpcode() == AArch64::ADRP) {
        if (MI.getNumOperands() >= 2 && MI.getOperand(0).isReg() &&
            MI.getOperand(1).isCPI()) {
          Register DstReg = MI.getOperand(0).getReg();
          unsigned CPIndex = MI.getOperand(1).getIndex();
          RegToCPInfo[DstReg] = {CPIndex, &MI, nullptr};
          LLVM_DEBUG(dbgs() << "  Found ADRP to CP" << CPIndex << " in "
                            << printReg(DstReg, TRI) << "\n");
        }
        continue;
      }

      // Track MOVi32imm/MOVi64imm pseudo instructions that load immediates.
      if (MI.getOpcode() == AArch64::MOVi32imm ||
          MI.getOpcode() == AArch64::MOVi64imm) {
        if (MI.getNumOperands() >= 2 && MI.getOperand(0).isReg() &&
            MI.getOperand(1).isImm()) {
          Register DstReg = MI.getOperand(0).getReg();
          uint64_t Val = MI.getOperand(1).getImm();
          RegToImmInfo[DstReg] = {Val, {&MI}};
          LLVM_DEBUG(dbgs() << "  Found MOVimm: " << printReg(DstReg, TRI)
                            << " = 0x" << Twine::utohexstr(Val) << "\n");
        }
        continue;
      }

      // Track MOVZWi/MOVZXi instructions that start an immediate sequence.
      if (MI.getOpcode() == AArch64::MOVZWi ||
          MI.getOpcode() == AArch64::MOVZXi) {
        if (MI.getNumOperands() >= 3 && MI.getOperand(0).isReg() &&
            MI.getOperand(1).isImm() && MI.getOperand(2).isImm()) {
          Register DstReg = MI.getOperand(0).getReg();
          uint64_t Imm = MI.getOperand(1).getImm();
          uint64_t Shift = MI.getOperand(2).getImm();
          uint64_t Val = Imm << Shift;
          RegToImmInfo[DstReg] = {Val, {&MI}};
          LLVM_DEBUG(dbgs() << "  Found MOVZ: " << printReg(DstReg, TRI)
                            << " = 0x" << Twine::utohexstr(Val) << "\n");
        }
        continue;
      }

      // Track MOVKWi/MOVKXi instructions that add to an immediate sequence.
      if (MI.getOpcode() == AArch64::MOVKWi ||
          MI.getOpcode() == AArch64::MOVKXi) {
        if (MI.getNumOperands() >= 4 && MI.getOperand(0).isReg() &&
            MI.getOperand(1).isReg() && MI.getOperand(2).isImm() &&
            MI.getOperand(3).isImm()) {
          Register DstReg = MI.getOperand(0).getReg();
          Register SrcReg = MI.getOperand(1).getReg();
          // MOVK is a tied def/use, so DstReg == SrcReg.
          auto It = RegToImmInfo.find(SrcReg);
          if (It != RegToImmInfo.end()) {
            uint64_t Imm = MI.getOperand(2).getImm();
            uint64_t Shift = MI.getOperand(3).getImm();
            uint64_t Mask = 0xFFFFULL << Shift;
            uint64_t Val = (It->second.ImmVal & ~Mask) | (Imm << Shift);
            It->second.ImmVal = Val;
            It->second.DefMIs.push_back(&MI);
            LLVM_DEBUG(dbgs() << "  Found MOVK: " << printReg(DstReg, TRI)
                              << " updated to 0x" << Twine::utohexstr(Val)
                              << "\n");
          }
        }
        continue;
      }

      // Track DUP instructions that broadcast from GPR to vector.
      unsigned Opc = MI.getOpcode();
      bool Is64Bit;
      if (isDUPFromGPR(Opc, Is64Bit)) {
        if (MI.getNumOperands() >= 2 && MI.getOperand(0).isReg() &&
            MI.getOperand(1).isReg()) {
          Register VecReg = MI.getOperand(0).getReg();
          Register GPRReg = MI.getOperand(1).getReg();
          auto It = RegToImmInfo.find(GPRReg);
          if (It != RegToImmInfo.end()) {
            VecToDUPInfo[VecReg] = {It->second.ImmVal, Opc, Is64Bit,
                                    It->second.DefMIs, &MI};
            LLVM_DEBUG(dbgs() << "  Found DUP: " << printReg(VecReg, TRI)
                              << " = DUP(0x"
                              << Twine::utohexstr(It->second.ImmVal) << ")\n");
          }
        }
        continue;
      }

      // Track LDR instructions that load from constant pool via ADRP result.
      // These are LDRQui, LDRDui, LDRSui, LDRXui, LDRWui with CP operand.
      bool IsLDR = (Opc == AArch64::LDRQui || Opc == AArch64::LDRDui ||
                    Opc == AArch64::LDRSui || Opc == AArch64::LDRXui ||
                    Opc == AArch64::LDRWui);

      if (IsLDR && MI.getNumOperands() >= 3 && MI.getOperand(0).isReg() &&
          MI.getOperand(1).isReg() && MI.getOperand(2).isCPI()) {
        Register DstReg = MI.getOperand(0).getReg();
        Register BaseReg = MI.getOperand(1).getReg();
        unsigned CPIndex = MI.getOperand(2).getIndex();

        // Get the ADRP instruction from the base register if available.
        MachineInstr *ADRPMI = nullptr;
        auto BaseIt = RegToCPInfo.find(BaseReg);
        if (BaseIt != RegToCPInfo.end() && BaseIt->second.CPIndex == CPIndex) {
          ADRPMI = BaseIt->second.ADRPMI;
        }

        RegToCPInfo[DstReg] = {CPIndex, ADRPMI, &MI};
        // Clear the base register as it's been used.
        RegToCPInfo.erase(BaseReg);
        LLVM_DEBUG(dbgs() << "  Found LDR from CP" << CPIndex << " to "
                          << printReg(DstReg, TRI) << "\n");
        continue;
      }

      // Look for stores to stack slots (spills).
      int FrameIndex;
      if (Register SrcReg = TII->isStoreToStackSlot(MI, FrameIndex)) {
        // Only process actual register spills, not other stack stores
        // (e.g., argument passing, local variables, temporaries)
        if (!MFI->isSpillSlotObjectIndex(FrameIndex)) {
          LLVM_DEBUG(dbgs() << "  Skipping non-spill stack store to FI"
                            << FrameIndex << "\n");
          continue;
        }

        // Check for constant pool spill.
        auto CPIt = RegToCPInfo.find(SrcReg);
        if (CPIt != RegToCPInfo.end()) {
          RegCPInfo &Info = CPIt->second;
          auto SpillIt = FrameIndexToSpillInfo.find(FrameIndex);
          if (SpillIt != FrameIndexToSpillInfo.end()) {
            // Frame index already exists, add this spill to the list
            SpillIt->second.SpillMIs.push_back(&MI);
          } else {
            // New frame index, create new entry
            FrameIndexToSpillInfo.try_emplace(FrameIndex, Info.CPIndex,
                                              Info.ADRPMI, Info.LoadMI, &MI);
          }
          ++NumSpillsTracked;
          LLVM_DEBUG(dbgs() << "  Tracking CP spill: FI" << FrameIndex
                            << " -> CP" << Info.CPIndex << "\n");
          continue;
        }

        // Check for DUP of constant spill.
        auto DUPIt = VecToDUPInfo.find(SrcReg);
        if (DUPIt != VecToDUPInfo.end()) {
          VecDUPInfo &Info = DUPIt->second;
          auto SpillIt = FrameIndexToDUPSpillInfo.find(FrameIndex);
          if (SpillIt != FrameIndexToDUPSpillInfo.end()) {
            // Frame index already exists, add this spill to the list
            SpillIt->second.SpillMIs.push_back(&MI);
          } else {
            // New frame index, create new entry
            FrameIndexToDUPSpillInfo.try_emplace(
                FrameIndex, Info.ImmVal, Info.DUPOpcode, Info.Is64Bit, &MI,
                Info.DefMIs, Info.DUPMI);
          }
          ++NumDUPSpillsTracked;
          LLVM_DEBUG(dbgs() << "  Tracking DUP spill: FI" << FrameIndex
                            << " -> DUP(0x" << Twine::utohexstr(Info.ImmVal)
                            << ")\n");
          continue;
        }
        continue;
      }

      // Invalidate any registers that are defined by other instructions.
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
          Register Reg = MO.getReg();
          RegToCPInfo.erase(Reg);
          RegToImmInfo.erase(Reg);
          VecToDUPInfo.erase(Reg);
          // Also invalidate any aliases.
          for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
            RegToCPInfo.erase(*AI);
            RegToImmInfo.erase(*AI);
            VecToDUPInfo.erase(*AI);
          }
        }
      }
    }
  }
}

void AArch64CollectCPSpillInfo::countReloads(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      int FrameIndex;
      if (TII->isLoadFromStackSlot(MI, FrameIndex)) {
        auto CPIt = FrameIndexToSpillInfo.find(FrameIndex);
        if (CPIt != FrameIndexToSpillInfo.end()) {
          CPIt->second.ReloadCount++;
        }
        auto DUPIt = FrameIndexToDUPSpillInfo.find(FrameIndex);
        if (DUPIt != FrameIndexToDUPSpillInfo.end()) {
          DUPIt->second.ReloadCount++;
        }
      }
    }
  }
}

Register AArch64CollectCPSpillInfo::findFreeGPRForADRP(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    LivePhysRegs &LiveRegs) {
  // Compute live registers at this point.
  LiveRegs.clear();
  LiveRegs.addLiveOuts(MBB);

  // Walk backwards from end to current instruction.
  for (auto I = MBB.rbegin(), E = MBB.rend(); I != E; ++I) {
    if (&*I == &*MI)
      break;
    LiveRegs.stepBackward(*I);
  }

  // Look for a free GPR that we can use for ADRP.
  // Prefer callee-saved registers that are already saved.
  // Avoid X16, X17 (used by linker), X18 (platform reserved), X29, X30, SP.
  // Check reserved registers to avoid using argument/return registers.
  static const MCPhysReg GPRs[] = {
      AArch64::X0,  AArch64::X1,  AArch64::X2,  AArch64::X3,
      AArch64::X4,  AArch64::X5,  AArch64::X6,  AArch64::X7,
      AArch64::X8,  AArch64::X9,  AArch64::X10, AArch64::X11,
      AArch64::X12, AArch64::X13, AArch64::X14, AArch64::X15,
      AArch64::X19, AArch64::X20, AArch64::X21, AArch64::X22,
      AArch64::X23, AArch64::X24, AArch64::X25, AArch64::X26,
      AArch64::X27, AArch64::X28,
  };

  for (MCPhysReg Reg : GPRs) {
    // Skip reserved registers to avoid using argument/return registers
    const AArch64RegisterInfo *ARI = static_cast<const AArch64RegisterInfo *>(TRI);
    if (ARI->isReservedReg(*MBB.getParent(), Reg))
      continue;

    // Skip argument/return registers (X0-X7) to avoid interfering with calling convention
    if (Reg >= AArch64::X0 && Reg <= AArch64::X7)
      continue;

    if (LiveRegs.available(*MRI, Reg)) {
      LLVM_DEBUG(dbgs() << "  Found free GPR: " << printReg(Reg, TRI) << "\n");
      return Reg;
    }
  }

  return Register();
}

unsigned AArch64CollectCPSpillInfo::getCPLoadOpcode(unsigned ReloadOpc) const {
  // Map reload opcodes to appropriate LDR opcodes for constant pool access.
  switch (ReloadOpc) {
  case AArch64::LDRQui:
    return AArch64::LDRQui;
  case AArch64::LDRDui:
    return AArch64::LDRDui;
  case AArch64::LDRSui:
    return AArch64::LDRSui;
  case AArch64::LDRXui:
    return AArch64::LDRXui;
  case AArch64::LDRWui:
    return AArch64::LDRWui;
  default:
    return 0;
  }
}

bool AArch64CollectCPSpillInfo::isDUPFromGPR(unsigned Opc, bool &Is64Bit) const {
  switch (Opc) {
  case AArch64::DUPv2i64gpr:
    Is64Bit = true;
    return true;
  case AArch64::DUPv8i8gpr:
  case AArch64::DUPv16i8gpr:
  case AArch64::DUPv4i16gpr:
  case AArch64::DUPv8i16gpr:
  case AArch64::DUPv2i32gpr:
  case AArch64::DUPv4i32gpr:
    Is64Bit = false;
    return true;
  default:
    return false;
  }
}

/// Get the element size in bits for a DUP opcode.
static unsigned getDUPElementSize(unsigned DUPOpc) {
  switch (DUPOpc) {
  case AArch64::DUPv8i8gpr:
  case AArch64::DUPv16i8gpr:
    return 8;
  case AArch64::DUPv4i16gpr:
  case AArch64::DUPv8i16gpr:
    return 16;
  case AArch64::DUPv2i32gpr:
  case AArch64::DUPv4i32gpr:
    return 32;
  case AArch64::DUPv2i64gpr:
    return 64;
  default:
    return 0;
  }
}

/// Get the DUP opcode for a given element size and target vector size.
/// Returns 0 if no matching opcode exists.
static unsigned getDUPOpcodeForSize(unsigned ElemSize, unsigned VecSize,
                                    bool Is64BitGPR) {
  if (VecSize == 8) { // 64-bit vector (D register)
    switch (ElemSize) {
    case 8:  return AArch64::DUPv8i8gpr;
    case 16: return AArch64::DUPv4i16gpr;
    case 32: return AArch64::DUPv2i32gpr;
    default: return 0;
    }
  } else if (VecSize == 16) { // 128-bit vector (Q register)
    switch (ElemSize) {
    case 8:  return AArch64::DUPv16i8gpr;
    case 16: return AArch64::DUPv8i16gpr;
    case 32: return AArch64::DUPv4i32gpr;
    case 64: return Is64BitGPR ? AArch64::DUPv2i64gpr : 0;
    default: return 0;
    }
  }
  return 0;
}

unsigned AArch64CollectCPSpillInfo::getReloadSize(unsigned ReloadOpc) const {
  switch (ReloadOpc) {
  case AArch64::LDRQui:
    return 16;
  case AArch64::LDRDui:
  case AArch64::LDRXui:
    return 8;
  case AArch64::LDRSui:
  case AArch64::LDRWui:
    return 4;
  default:
    return 0;
  }
}

bool AArch64CollectCPSpillInfo::hasOnlyOneUse(MachineInstr *MI,
                                              MachineInstr *User) const {
  if (!MI || MI->getNumOperands() == 0 || !MI->getOperand(0).isReg())
    return false;

  Register DefReg = MI->getOperand(0).getReg();
  if (!DefReg.isPhysical())
    return false;

  // Count uses of DefReg (and aliases) in the function.
  // This is a simplified check - we just verify the only use is the User.
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MBB->getParent();

  unsigned UseCount = 0;
  for (MachineBasicBlock &BB : *MF) {
    for (MachineInstr &UseMI : BB) {
      if (&UseMI == MI)
        continue; // Skip the def itself

      for (const MachineOperand &MO : UseMI.operands()) {
        if (MO.isReg() && MO.isUse() && MO.getReg().isPhysical()) {
          if (TRI->regsOverlap(MO.getReg(), DefReg)) {
            UseCount++;
            if (UseCount > 1 || &UseMI != User)
              return false;
          }
        }
      }
    }
  }

  return UseCount == 1;
}

unsigned AArch64CollectCPSpillInfo::countFrameIndexUses(MachineFunction &MF,
                                                        int FrameIndex) const {
  unsigned Count = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isFI() && MO.getIndex() == FrameIndex) {
          Count++;
          break; // Count each instruction only once
        }
      }
    }
  }
  return Count;
}

bool AArch64CollectCPSpillInfo::optimizeReloads(MachineFunction &MF) {
  bool Changed = false;
  LivePhysRegs LiveRegs(*TRI);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      int FrameIndex;
      Register DstReg = TII->isLoadFromStackSlot(MI, FrameIndex);
      if (!DstReg)
        continue;

      // Check if this stack slot holds a spilled CP value.
      auto CPIt = FrameIndexToSpillInfo.find(FrameIndex);
      if (CPIt != FrameIndexToSpillInfo.end()) {
        CPSpillInfo &Info = CPIt->second;

        // Get the appropriate load opcode.
        unsigned LoadOpc = getCPLoadOpcode(MI.getOpcode());
        if (!LoadOpc)
          continue;

        // Check if reload size is compatible with original CP load size
        // to prevent alignment issues (e.g., 128-bit reload from 8-byte CP entry)
        unsigned ReloadSize = getReloadSize(MI.getOpcode());
        unsigned OrigSize = Info.LoadMI ? getReloadSize(Info.LoadMI->getOpcode()) : 0;
        if (ReloadSize > OrigSize && OrigSize > 0) {
          LLVM_DEBUG(dbgs() << "  Skipping incompatible reload: reload size ("
                            << ReloadSize << " bytes) > original CP load size ("
                            << OrigSize << " bytes)\n");
          continue;
        }

        // Check if we have a free GPR for ADRP.
        Register ADRPReg = findFreeGPRForADRP(MBB, MI.getIterator(), LiveRegs);
        if (!ADRPReg)
          continue;

        LLVM_DEBUG(dbgs() << "Replacing reload from FI" << FrameIndex
                          << " with CP" << Info.CPIndex << " load\n");
        LLVM_DEBUG(dbgs() << "  Original reload: " << MI << "\n");
        LLVM_DEBUG(dbgs() << "  Will create: ADRP " << printReg(ADRPReg, TRI)
                          << ", CP" << Info.CPIndex << "@PAGE; LDR "
                          << printReg(DstReg, TRI) << ", [" << printReg(ADRPReg, TRI)
                          << ", CP" << Info.CPIndex << "@PAGEOFF]\n");

        // Build ADRP + LDR sequence.
        MachineBasicBlock::iterator InsertPt = MI.getIterator();
        DebugLoc DL = MI.getDebugLoc();

        // ADRP Xd, cpentry@PAGE
        BuildMI(MBB, InsertPt, DL, TII->get(AArch64::ADRP), ADRPReg)
            .addConstantPoolIndex(Info.CPIndex, 0, AArch64II::MO_PAGE);

        // LDR Qd, [Xd, cpentry@PAGEOFF]
        BuildMI(MBB, InsertPt, DL, TII->get(LoadOpc), DstReg)
            .addReg(ADRPReg, RegState::Kill)
            .addConstantPoolIndex(Info.CPIndex, 0,
                                  AArch64II::MO_PAGEOFF | AArch64II::MO_NC);

        // Remove the original reload.
        MI.eraseFromParent();
        ++NumReloadsReplaced;
        Info.ReplacedCount++;
        Changed = true;
        continue;
      }

      // Check if this stack slot holds a spilled DUP value.
      auto DUPIt = FrameIndexToDUPSpillInfo.find(FrameIndex);
      if (DUPIt != FrameIndexToDUPSpillInfo.end()) {
        DUPSpillInfo &Info = DUPIt->second;

        // Get the reload size and determine the correct DUP opcode.
        // The reload may be a different size than the original DUP destination
        // (e.g., original DUP produces D register but spill/reload uses Q).
        unsigned ReloadSize = getReloadSize(MI.getOpcode());
        unsigned ElemSize = getDUPElementSize(Info.DUPOpcode);
        if (ElemSize == 0)
          continue;

        // Check register class compatibility: DUP produces FPR, so reload must be to FPR
        bool isDRegReload = AArch64::FPR64RegClass.contains(DstReg);
        bool isQRegReload = AArch64::FPR128RegClass.contains(DstReg);

        if (!isDRegReload && !isQRegReload) {
          LLVM_DEBUG(dbgs() << "  Skipping: DstReg is not FPR (is GPR), incompatible with DUP output\n");
          continue;
        }

        // Get the correct DUP opcode for this reload size.
        unsigned DUPOpc = getDUPOpcodeForSize(ElemSize, ReloadSize, Info.Is64Bit);
        if (DUPOpc == 0)
          continue;

        // Check if we have a free GPR for MOV.
        Register GPRReg = findFreeGPRForADRP(MBB, MI.getIterator(), LiveRegs);
        if (!GPRReg)
          continue;

        // Note: Keep GPRReg as X register since MOVi32imm/MOVi64imm pseudos
        // expect X registers even for 32-bit immediates. The DUP instruction
        // will get the correct register size based on the DUP opcode.

        // For 32-bit operations, verify we can get the W register for DUP
        // before creating any instructions.
        Register DUPSrcReg = GPRReg;
        if (!Info.Is64Bit) {
          DUPSrcReg = TRI->getSubReg(GPRReg, AArch64::sub_32);
          if (!DUPSrcReg) {
            // This should not happen given our register selection
            LLVM_DEBUG(dbgs() << "  Failed to get W register for DUP\n");
            continue;
          }
        }

        LLVM_DEBUG(dbgs() << "Replacing reload from FI" << FrameIndex
                          << " with MOV+DUP(0x" << Twine::utohexstr(Info.ImmVal)
                          << ")\n");

        MachineBasicBlock::iterator InsertPt = MI.getIterator();
        DebugLoc DL = MI.getDebugLoc();

        // Emit MOVi32imm or MOVi64imm pseudo to materialize the constant.
        // For 32-bit MOVs, mask the immediate to 32 bits to avoid issues with
        // sign-extended values.
        unsigned MOVOpc =
            Info.Is64Bit ? AArch64::MOVi64imm : AArch64::MOVi32imm;
        uint64_t ImmVal = Info.Is64Bit ? Info.ImmVal : (Info.ImmVal & 0xFFFFFFFFULL);
        BuildMI(MBB, InsertPt, DL, TII->get(MOVOpc), GPRReg)
            .addImm(ImmVal);

        // Emit DUP to broadcast the GPR to vector.
        BuildMI(MBB, InsertPt, DL, TII->get(DUPOpc), DstReg)
            .addReg(DUPSrcReg, RegState::Kill);

        // Remove the original reload.
        MI.eraseFromParent();
        ++NumDUPReloadsReplaced;
        Info.ReplacedCount++;
        Changed = true;
      }
    }
  }

  return Changed;
}

bool AArch64CollectCPSpillInfo::eliminateDeadSpills(MachineFunction &MF) {
  bool Changed = false;

  for (auto &Entry : FrameIndexToSpillInfo) {
    CPSpillInfo &Info = Entry.second;
    int FrameIndex = Entry.first;

    // If all reloads were replaced, check if we can eliminate the spill.
    if (Info.ReloadCount > 0 && Info.ReplacedCount == Info.ReloadCount) {
      // Verify the spill instructions are still in basic blocks (not already
      // deleted by a previous iteration).
      SmallVector<MachineInstr *, 4> ValidSpills;
      for (MachineInstr *SpillMI : Info.SpillMIs) {
        if (SpillMI && SpillMI->getParent()) {
          ValidSpills.push_back(SpillMI);
        }
      }

      if (ValidSpills.empty())
        continue;

      // After replacing all reloads, the only remaining use of the frame index
      // should be the spills themselves. If there are other uses (e.g., partial
      // accesses for fcopysign), we cannot delete the spills.
      unsigned FIUses = countFrameIndexUses(MF, FrameIndex);
      if (FIUses != ValidSpills.size()) {
        LLVM_DEBUG(dbgs() << "Cannot eliminate spills for FI" << FrameIndex
                          << " - still has " << FIUses << " uses (expected "
                          << ValidSpills.size() << " spills)\n");
        continue;
      }

      LLVM_DEBUG(dbgs() << "Eliminating " << ValidSpills.size()
                        << " dead spills for FI" << FrameIndex
                        << " (replaced " << Info.ReplacedCount << "/"
                        << Info.ReloadCount << " reloads)\n");

      // Erase all spill stores.
      for (MachineInstr *SpillMI : ValidSpills) {
        SpillMI->eraseFromParent();
        ++NumSpillsEliminated;
      }
      Changed = true;

      // Mark the stack slot as dead.
      MFI->RemoveStackObject(FrameIndex);

      // Do not erase the original LDR from constant pool to avoid invalidating
      // the constant pool entry that our replacement loads reference.
      // The original LDR will be cleaned up by later passes if it becomes dead.
      if (Info.LoadMI && Info.LoadMI->getParent()) {
        LLVM_DEBUG(dbgs() << "  Keeping original LDR to maintain CP entry\n");
      }

      // Do not erase the original ADRP to avoid any potential issues
      // with constant pool entry references. Dead code elimination
      // passes will clean these up if they become unreferenced.
      if (Info.ADRPMI && Info.ADRPMI->getParent()) {
        LLVM_DEBUG(dbgs() << "  Keeping original ADRP to maintain CP references\n");
      }
    }
  }

  for (auto &Entry : FrameIndexToDUPSpillInfo) {
    DUPSpillInfo &Info = Entry.second;
    int FrameIndex = Entry.first;

    // If all reloads were replaced, check if we can eliminate the spill.
    if (Info.ReloadCount > 0 && Info.ReplacedCount == Info.ReloadCount) {
      // Verify the spill instructions are still in basic blocks (not already
      // deleted by a previous iteration).
      SmallVector<MachineInstr *, 4> ValidSpills;
      for (MachineInstr *SpillMI : Info.SpillMIs) {
        if (SpillMI && SpillMI->getParent()) {
          ValidSpills.push_back(SpillMI);
        }
      }

      if (ValidSpills.empty())
        continue;

      // After replacing all reloads, the only remaining use of the frame index
      // should be the spills themselves.
      unsigned FIUses = countFrameIndexUses(MF, FrameIndex);
      if (FIUses != ValidSpills.size()) {
        LLVM_DEBUG(dbgs() << "Cannot eliminate DUP spill for FI" << FrameIndex
                          << " - still has " << FIUses << " uses (expected "
                          << ValidSpills.size() << " spills)\n");
        continue;
      }

      LLVM_DEBUG(dbgs() << "Eliminating " << ValidSpills.size()
                        << " dead DUP spills for FI" << FrameIndex
                        << " (replaced " << Info.ReplacedCount << "/"
                        << Info.ReloadCount << " reloads)\n");

      // Check if the DUP's only use was the spills. If so, we can delete
      // the DUP and its defining MOV instructions as well.
      // Also verify the DUP instruction is still in a basic block (not already
      // deleted by a previous iteration).
      bool CanDeleteDUP = Info.DUPMI && Info.DUPMI->getParent();
      if (CanDeleteDUP) {
        // Check if all uses of the DUP are only the spill instructions
        for (MachineInstr *SpillMI : ValidSpills) {
          if (!hasOnlyOneUse(Info.DUPMI, SpillMI)) {
            CanDeleteDUP = false;
            break;
          }
        }
        // For multiple spills, we need a more sophisticated check
        // For now, conservatively only delete if there's exactly one spill
        if (ValidSpills.size() > 1) {
          CanDeleteDUP = false;
        }
      }

      // Erase all spill stores.
      for (MachineInstr *SpillMI : ValidSpills) {
        SpillMI->eraseFromParent();
        ++NumSpillsEliminated;
      }
      Changed = true;

      // Mark the stack slot as dead.
      MFI->RemoveStackObject(FrameIndex);

      // Delete the DUP and MOV instructions if they have no other uses.
      if (CanDeleteDUP) {
        // First check if the MOV's only use was the DUP.
        bool CanDeleteMOVs = true;
        for (MachineInstr *DefMI : Info.DefMIs) {
          if (DefMI && DefMI->getParent() &&
              !hasOnlyOneUse(DefMI, Info.DUPMI)) {
            CanDeleteMOVs = false;
            break;
          }
        }

        LLVM_DEBUG(dbgs() << "  Eliminating original DUP\n");
        Info.DUPMI->eraseFromParent();
        ++NumOrigLoadsEliminated;

        if (CanDeleteMOVs) {
          for (MachineInstr *DefMI : Info.DefMIs) {
            // Check DefMI is still in a basic block (not already deleted).
            if (DefMI && DefMI->getParent()) {
              LLVM_DEBUG(dbgs() << "  Eliminating original MOV/MOVK\n");
              DefMI->eraseFromParent();
              ++NumOrigLoadsEliminated;
            }
          }
        }
      }
    }
  }

  return Changed;
}

bool AArch64CollectCPSpillInfo::runOnMachineFunction(MachineFunction &MF) {
  const AArch64Subtarget &ST = MF.getSubtarget<AArch64Subtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();

  FrameIndexToSpillInfo.clear();
  FrameIndexToDUPSpillInfo.clear();

  LLVM_DEBUG(dbgs() << "*** AArch64 Collect CP Spill Info: " << MF.getName()
                    << " ***\n");

  // First pass: collect information about CP and DUP spills.
  collectSpillInfo(MF);

  if (FrameIndexToSpillInfo.empty() && FrameIndexToDUPSpillInfo.empty()) {
    LLVM_DEBUG(dbgs() << "  No spills found.\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Found " << FrameIndexToSpillInfo.size()
                    << " CP spill(s) and " << FrameIndexToDUPSpillInfo.size()
                    << " DUP spill(s).\n");

  // Count reloads for each spilled frame index.
  countReloads(MF);

  // Second pass: try to optimize reloads.
  bool Changed = optimizeReloads(MF);

  // Third pass: eliminate dead spills where all reloads were replaced.
  Changed |= eliminateDeadSpills(MF);

  return Changed;
}

FunctionPass *llvm::createAArch64CollectCPSpillInfoPass() {
  return new AArch64CollectCPSpillInfo();
}
