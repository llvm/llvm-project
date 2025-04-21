//===------- AMDGPUMIRUtils.h - Helpers for MIR passes --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for MIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMIRUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class LiveInterval;
class SlotIndexes;
class MachineRegisterInfo;
class SIRegisterInfo;
class SIInstrInfo;
class MachineDominatorTree;
class MachinePostDominatorTree;

constexpr unsigned RegForVCC = 2;

bool getNonDebugMBBEnd(llvm::MachineBasicBlock::reverse_iterator &BBEnd,
                       llvm::MachineBasicBlock &MBB);

// Check if LI live cross basic blocks, save all touched basic block if is
// local.
bool isLocalLiveInterval(
    const llvm::LiveInterval &LI, llvm::SlotIndexes *Indexes,
    llvm::SmallDenseSet<llvm::MachineBasicBlock *, 2> &TouchedMBBSet);
bool isLocalLiveInterval(const llvm::LiveInterval &LI,
                         llvm::SlotIndexes *Indexes);

bool isSub0Sub1SingleDef(unsigned Reg, const llvm::MachineRegisterInfo &MRI);

using LiveSet = llvm::DenseMap<unsigned, llvm::LaneBitmask>;
void dumpLiveSet(const LiveSet &LiveSet, const llvm::SIRegisterInfo *SIRI);

bool isSccLiveAt(llvm::MachineBasicBlock *MBB,
                 llvm::MachineBasicBlock::iterator MI);

// An enum used to pass additional constraints to
// `FindOrCreateInsertionPointForSccDef()`. This will further
// constrain the location where the scc def can be inserted.
enum SccDefInsertPointConstraintFlags {
  None = 0,        // No additional constraints.
  NoExecWrite = 1, // Should be no modification of exec between BeforeInst and
                   // insert point.
};

// Look for a safe place to insert an instruction that defines scc.
//
//
// This function is useful for when we need to insert a new
// instruction that defines scc in a block and we need to find
// a location that will not smash the existing value.
//
// Starting at `BeforeInst` it will look backwards to try to find
// a place in the block where scc is dead so we can insert our new
// def there. If no location can be found it will save and restore
// scc around BeforeInst. This way BeforeInst can safely be used
// as the new insert location.
//
llvm::MachineBasicBlock::iterator findOrCreateInsertionPointForSccDef(
    llvm::MachineBasicBlock *MBB, llvm::MachineBasicBlock::iterator BeforeInst,
    const llvm::TargetRegisterInfo *TRI, const llvm::SIInstrInfo *TII,
    llvm::MachineRegisterInfo *MRI,
    SccDefInsertPointConstraintFlags Constraints =
        SccDefInsertPointConstraintFlags::None);

// For inst like S_BUFFER_LOAD_DWORDX16, change to S_BUFFER_LOAD_DWORDX4 if only
// used 4 lanes.
bool removeUnusedLanes(llvm::MachineInstr &MI, llvm::MachineRegisterInfo &MRI,
                       const llvm::SIRegisterInfo *TRI,
                       const llvm::SIInstrInfo *TII,
                       llvm::SlotIndexes *SlotIndexes);

unsigned getRegSize(unsigned Reg, llvm::LaneBitmask &Mask,
                    const llvm::MachineRegisterInfo &MRI,
                    const llvm::SIRegisterInfo *SIRI);
void collectLiveSetPressure(const LiveSet &LiveSet,
                            const llvm::MachineRegisterInfo &MRI,
                            const llvm::SIRegisterInfo *SIRI,
                            unsigned &VPressure, unsigned &SPressure);

bool reach_block(llvm::MachineBasicBlock *FromBB,
                 llvm::MachineDominatorTree *DT,
                 llvm::MachinePostDominatorTree *PDT, llvm::MachineLoopInfo *LI,
                 llvm::MachineBasicBlock *ToBB);
} // namespace llvm

#endif
