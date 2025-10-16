//===- MachineLaneSSAUpdater.h - SSA repair for Machine IR (lane-aware) -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// === MachineLaneSSAUpdater Design Notes ===
//

#ifndef LLVM_CODEGEN_MACHINELANESSAUPDATER_H
#define LLVM_CODEGEN_MACHINELANESSAUPDATER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"        // SmallVector
#include "llvm/CodeGen/LiveInterval.h"    // LiveRange
#include "llvm/CodeGen/Register.h"       // Register
#include "llvm/CodeGen/SlotIndexes.h"    // SlotIndex
#include "llvm/CodeGen/TargetRegisterInfo.h" // For inline function
#include "llvm/MC/LaneBitmask.h"        // LaneBitmask

namespace llvm {

// Forward declarations to avoid heavy includes in the header.
class MachineFunction;
class MachineBasicBlock;
class MachineInstr;
class LiveIntervals;
class LiveRange;
class MachineDominatorTree;
class MachinePostDominatorTree; // optional if you choose to use it

//===----------------------------------------------------------------------===//
// MachineLaneSSAUpdater: universal SSA repair for Machine IR (lane-aware)
//
// Primary Use Case: repairSSAForNewDef()
//   - Caller creates a new instruction that defines an existing vreg (violating SSA)
//   - This function creates a new vreg (or uses a caller-provided one), 
//     replaces the operand, and repairs SSA
//   - Example: Insert "OrigVReg = ADD ..." and call repairSSAForNewDef()
//   - Works for full register and subregister definitions
//   - Handles all scenarios including spill/reload
//
// Advanced Usage: Caller-provided NewVReg
//   - By default, repairSSAForNewDef() creates a new virtual register automatically
//   - For special cases (e.g., subregister reloads where the spiller already 
//     created a register of a specific class), caller can provide NewVReg
//   - This gives full control over register class selection when needed
//===----------------------------------------------------------------------===//
class MachineLaneSSAUpdater {
public:
  MachineLaneSSAUpdater(MachineFunction &MF,
                        LiveIntervals &LIS,
                        MachineDominatorTree &MDT,
                        const TargetRegisterInfo &TRI)
      : MF(MF), LIS(LIS), MDT(MDT), TRI(TRI) {}

  // Repair SSA for a new definition that violates SSA form
  // 
  // Parameters:
  //   NewDefMI: Instruction with a def operand that currently defines OrigVReg (violating SSA)
  //   OrigVReg: The virtual register being redefined
  //   NewVReg:  (Optional) Pre-allocated virtual register to use instead of auto-creating one
  //
  // This function will:
  //   1. Find the def operand in NewDefMI that defines OrigVReg
  //   2. Derive the lane mask from the operand's subreg index (if any)
  //   3. Use NewVReg if provided, or create a new virtual register with appropriate class
  //   4. Replace the operand in NewDefMI to define the new vreg
  //   5. Perform SSA repair (insert PHIs, rewrite uses)
  //
  // When to provide NewVReg:
  //   - Leave it empty (default) for most cases - automatic class selection works well
  //   - Provide it when you need precise control over register class selection
  //   - Common use case: subregister spill/reload where target-specific constraints apply
  //   - Example: Reloading a 96-bit subregister requires vreg_96 class (not vreg_128)
  //
  // Returns: The SSA-repaired virtual register (either NewVReg or auto-created)
  Register repairSSAForNewDef(MachineInstr &NewDefMI, Register OrigVReg,
                             Register NewVReg = Register());

private:
  // Common SSA repair logic
  void performSSARepair(Register NewVReg, Register OrigVReg, 
                        LaneBitmask DefMask, MachineBasicBlock *DefBB);

  // Optional knobs (fluent style); no-ops until implemented in .cpp.
  MachineLaneSSAUpdater &setUndefEdgePolicy(bool MaterializeImplicitDef) {
    UndefEdgeAsImplicitDef = MaterializeImplicitDef; return *this; }
  MachineLaneSSAUpdater &setVerifyOnExit(bool Enable) {
    VerifyOnExit = Enable; return *this; }

  // --- Internal helpers ---

  // Index MI in SlotIndexes / LIS maps immediately after insertion.
  // Returns the SlotIndex assigned to the instruction.
  SlotIndex indexNewInstr(MachineInstr &MI);

  // Extend the main live range and the specific subranges at MI's index
  // for the lanes actually used/defined.
  void extendPreciselyAt(const Register VReg,
                         const SmallVector<LaneBitmask, 4> &LaneMasks,
                         const MachineInstr &AtMI);

  // Compute pruned IDF for a set of definition blocks (usually {block(NewDef)}),
  // intersected with blocks where OrigVReg lanes specified by DefMask are live-in.
  void computePrunedIDF(Register OrigVReg,
                        LaneBitmask DefMask,
                        ArrayRef<MachineBasicBlock *> NewDefBlocks,
                        SmallVectorImpl<MachineBasicBlock *> &OutIDFBlocks);

  // Insert lane-aware Machine PHIs with iterative worklist processing.
  // Seeds with InitialVReg definition, computes IDF, places PHIs, repeats until convergence.
  // Returns all PHI result registers created during the iteration.
  SmallVector<Register> insertLaneAwarePHI(Register InitialVReg,
                                            Register OrigVReg,
                                            LaneBitmask DefMask,
                                            MachineBasicBlock *InitialDefBB);

  // Helper: Create PHI in a specific block with per-edge lane analysis
  Register createPHIInBlock(MachineBasicBlock &JoinMBB,
                           Register OrigVReg,
                           Register NewVReg,
                           LaneBitmask DefMask);

  // Rewrite dominated uses of OrigVReg to NewSSA according to the
  // exact/subset/super policy; create REG_SEQUENCE only when needed.
  void rewriteDominatedUses(Register OrigVReg,
                            Register NewSSA,
                            LaneBitmask MaskToRewrite);

  // Internal helper methods for use rewriting
  VNInfo *incomingOnEdge(LiveInterval &LI, MachineInstr *Phi, MachineOperand &PhiOp);
  bool defReachesUse(MachineInstr *DefMI, MachineInstr *UseMI, MachineOperand &UseOp);
  LaneBitmask operandLaneMask(const MachineOperand &MO);
  Register buildRSForSuperUse(MachineInstr *UseMI, MachineOperand &MO,
                             Register OldVR, Register NewVR, LaneBitmask MaskToRewrite,
                             LiveInterval &LI, const TargetRegisterClass *OpRC,
                             SlotIndex &OutIdx, SmallVectorImpl<LaneBitmask> &LanesToExtend);
  void extendAt(LiveInterval &LI, SlotIndex Idx, ArrayRef<LaneBitmask> Lanes);
  void updateDeadFlags(Register Reg);

  // --- Data members ---
  MachineFunction &MF;
  LiveIntervals &LIS;
  MachineDominatorTree &MDT;
  const TargetRegisterInfo &TRI;

  bool UndefEdgeAsImplicitDef = true; // policy hook
  bool VerifyOnExit = true;           // run MF.verify()/LI.verify() at end
};

/// Get the subregister index that corresponds to the given lane mask.
/// \param Mask The lane mask to convert to a subregister index
/// \param TRI The target register info (provides target-specific subregister mapping)
/// \return The subregister index, or 0 if no single subregister matches
inline unsigned getSubRegIndexForLaneMask(LaneBitmask Mask, const TargetRegisterInfo *TRI) {
  if (Mask.none())
    return 0; // No subregister
  
  // Iterate through all subregister indices to find a match
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    if (SubMask == Mask) {
      return SubIdx;
    }
  }
  
  // No exact match found - this might be a composite mask requiring REG_SEQUENCE
  return 0;
}

// DenseMapInfo specialization for LaneBitmask
template<>
struct DenseMapInfo<LaneBitmask> {
  static inline LaneBitmask getEmptyKey() {
    // Use a specific bit pattern for empty key
    return LaneBitmask(~0U - 1);
  }
  
  static inline LaneBitmask getTombstoneKey() {
    // Use a different bit pattern for tombstone  
    return LaneBitmask(~0U);
  }
  
  static unsigned getHashValue(const LaneBitmask &Val) {
    return (unsigned)Val.getAsInteger();
  }
  
  static bool isEqual(const LaneBitmask &LHS, const LaneBitmask &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINELANESSAUPDATER_H