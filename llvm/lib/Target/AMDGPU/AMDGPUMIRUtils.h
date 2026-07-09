//===------- AMDGPUMIRUtils.h - Helpers for MIR passes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions shared by AMDGPU MIR passes, in particular the hot
/// block rematerialization pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/LaneBitmask.h"

namespace llvm {

class LiveIntervals;
class MachineInstr;
class MachineRegisterInfo;
class MachineDominatorTree;
class MachinePostDominatorTree;
class MachineLoopInfo;
class SIRegisterInfo;
class SIInstrInfo;
class TargetRegisterInfo;
class Register;

// VCC always reserves a 2-SGPR pair; add this to measured SGPR pressure.
constexpr unsigned RegForVCC = 2;

using LiveSet = DenseMap<unsigned, LaneBitmask>;

// Advances a reverse iterator past trailing debug instructions. Returns false
// if the block contains only debug instructions.
bool getNonDebugMBBEnd(MachineBasicBlock::reverse_iterator &BBEnd,
                       MachineBasicBlock &MBB);

// Returns the number of lanes that map to one 32-bit register for the given
// register file.
unsigned getNumLanesIn32BitReg(bool IsVGPR);

// Returns the effective live size of Reg in 32-bit dword units, honoring the
// partially-live lane mask. Lane masks are converted to dwords via the
// register class's lanes-per-dword ratio (not a raw lane count).
unsigned getRegSize(unsigned Reg, LaneBitmask &Mask,
                    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI);

// Sums VGPR and SGPR pressure (in dwords) over a live set.
void collectLiveSetPressure(const LiveSet &LiveSet,
                            const MachineRegisterInfo &MRI,
                            const SIRegisterInfo *SIRI, unsigned &VPressure,
                            unsigned &SPressure);

// Returns true if all defs of Reg are disjoint sub-register writes that
// together cover the full register exactly once (e.g. sub0/sub1 for a 64-bit
// pointer, or sub0..sub3 for a 128-bit descriptor). Replaces the PR's
// two-subreg-only isSub0Sub1SingleDef.
bool isConsistentMultiDef(Register Reg, const MachineRegisterInfo &MRI,
                          const TargetRegisterInfo *TRI);

// Walks the immediate dominator chain from BB until reaching a block outside
// all loops. Returns nullptr if none exists in the chain.
MachineBasicBlock *findNonLoopDominator(MachineBasicBlock *BB,
                                        MachineDominatorTree *DT,
                                        MachineLoopInfo *LI);

// Nearest common dominator of a set of blocks, with the EXEC-split fixup
// (see the .cpp) applied to the result.
MachineBasicBlock *
nearestCommonDominator(MachineDominatorTree *DT,
                       SmallPtrSetImpl<MachineBasicBlock *> &Blocks);

// True if a def can legally be relocated from FromBB to ToBB on CFG grounds
// alone (dominance / post-dominance / shared outermost loop). Memory ordering
// is enforced separately by the caller.
bool reachBlock(MachineBasicBlock *FromBB, MachineDominatorTree *DT,
                MachinePostDominatorTree *PDT, MachineLoopInfo *LI,
                MachineBasicBlock *ToBB);

// True if SCC is live at MI according to LiveIntervals; conservatively true
// when LIS is null.
bool isSccLiveAt(const MachineInstr &MI, LiveIntervals *LIS);

// Finds a safe location to insert an instruction that defines SCC, starting at
// BeforeInst and scanning backwards for a point where SCC is dead. If none is
// found, a COPY-based save/restore of SCC is synthesized around BeforeInst and
// the SCC live range is split accordingly; BeforeInst is then returned as a
// now-safe insertion point.
MachineBasicBlock::iterator findOrCreateInsertionPointForSccDef(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator BeforeInst,
    const TargetRegisterInfo *TRI, const SIInstrInfo *TII,
    MachineRegisterInfo *MRI, LiveIntervals *LIS);

void dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H
