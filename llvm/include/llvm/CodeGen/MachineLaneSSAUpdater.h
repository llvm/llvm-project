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

#include "llvm/MC/LaneBitmask.h"        // LaneBitmask
#include "llvm/ADT/SmallVector.h"        // SmallVector
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Register.h"       // Register
#include "llvm/CodeGen/SlotIndexes.h"    // SlotIndex
#include "llvm/CodeGen/LiveInterval.h"    // LiveRange
#include "llvm/CodeGen/TargetRegisterInfo.h" // For inline function

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
// Use Case: repairSSAForNewDef()
//   - Caller creates a new instruction that defines an existing vreg (violating SSA)
//   - This function creates a new vreg, replaces the operand, and repairs SSA
//   - Example: Insert "OrigVReg = ADD ..." and call repairSSAForNewDef()
//   - This works for all scenarios including spill/reload
//===----------------------------------------------------------------------===//
class MachineLaneSSAUpdater {
public:
  MachineLaneSSAUpdater(MachineFunction &MF,
                        LiveIntervals &LIS,
                        MachineDominatorTree &MDT,
                        const TargetRegisterInfo &TRI)
      : MF(MF), LIS(LIS), MDT(MDT), TRI(TRI) {}

  // Use Case 1 (Common): Repair SSA for a new definition
  // 
  // NewDefMI: Instruction with a def operand that currently defines OrigVReg (violating SSA)
  // OrigVReg: The virtual register being redefined
  //
  // This function will:
  //   1. Find the def operand in NewDefMI that defines OrigVReg
  //   2. Derive the lane mask from the operand's subreg index (if any)
  //   3. Create a new virtual register with appropriate register class
  //   4. Replace the operand in NewDefMI to define the new vreg
  //   5. Perform SSA repair (insert PHIs, rewrite uses)
  //
  // Returns: The newly created virtual register
  Register repairSSAForNewDef(MachineInstr &NewDefMI, Register OrigVReg);

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