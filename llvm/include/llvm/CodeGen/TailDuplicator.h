//===-- llvm/CodeGen/TailDuplicator.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TailDuplicator class. Used by the
// TailDuplication pass, and MachineBlockPlacement.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TAILDUPLICATOR_H
#define LLVM_CODEGEN_TAILDUPLICATOR_H

#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

namespace llvm {

/// Utility class to perform tail duplication.
class TailDuplicator {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineBranchProbabilityInfo *MBPI;
  const MachineModuleInfo *MMI;
  MachineRegisterInfo *MRI;
  MachineFunction *MF;
  bool PreRegAlloc;
  unsigned TailDupSize;

  // A list of virtual registers for which to update SSA form.
  SmallVector<unsigned, 16> SSAUpdateVRs;

  // For each virtual register in SSAUpdateVals keep a list of source virtual
  // registers.
  typedef std::vector<std::pair<MachineBasicBlock *, unsigned>> AvailableValsTy;

  DenseMap<unsigned, AvailableValsTy> SSAUpdateVals;

public:
  /// Prepare to run on a specific machine function.
  /// @param TailDupSize - Maxmimum size of blocks to tail-duplicate.
  void initMF(MachineFunction &MF,
              const MachineBranchProbabilityInfo *MBPI,
              unsigned TailDupSize = 0);
  bool tailDuplicateBlocks();
  static bool isSimpleBB(MachineBasicBlock *TailBB);
  bool shouldTailDuplicate(bool IsSimple, MachineBasicBlock &TailBB);
  /// Returns true if TailBB can successfully be duplicated into PredBB
  bool canTailDuplicate(MachineBasicBlock *TailBB, MachineBasicBlock *PredBB);
  bool tailDuplicateAndUpdate(bool IsSimple, MachineBasicBlock *MBB);

private:
  typedef TargetInstrInfo::RegSubRegPair RegSubRegPair;

  void addSSAUpdateEntry(unsigned OrigReg, unsigned NewReg,
                         MachineBasicBlock *BB);
  void processPHI(MachineInstr *MI, MachineBasicBlock *TailBB,
                  MachineBasicBlock *PredBB,
                  DenseMap<unsigned, RegSubRegPair> &LocalVRMap,
                  SmallVectorImpl<std::pair<unsigned, RegSubRegPair>> &Copies,
                  const DenseSet<unsigned> &UsedByPhi, bool Remove);
  void duplicateInstruction(MachineInstr *MI, MachineBasicBlock *TailBB,
                            MachineBasicBlock *PredBB,
                            DenseMap<unsigned, RegSubRegPair> &LocalVRMap,
                            const DenseSet<unsigned> &UsedByPhi);
  void updateSuccessorsPHIs(MachineBasicBlock *FromBB, bool isDead,
                            SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                            SmallSetVector<MachineBasicBlock *, 8> &Succs);
  bool canCompletelyDuplicateBB(MachineBasicBlock &BB);
  bool duplicateSimpleBB(MachineBasicBlock *TailBB,
                         SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                         const DenseSet<unsigned> &RegsUsedByPhi,
                         SmallVectorImpl<MachineInstr *> &Copies);
  bool tailDuplicate(bool IsSimple, MachineBasicBlock *TailBB,
                     SmallVectorImpl<MachineBasicBlock *> &TDBBs,
                     SmallVectorImpl<MachineInstr *> &Copies);
  void appendCopies(MachineBasicBlock *MBB,
                 SmallVectorImpl<std::pair<unsigned,RegSubRegPair>> &CopyInfos,
                 SmallVectorImpl<MachineInstr *> &Copies);

  void removeDeadBlock(MachineBasicBlock *MBB);
};

} // End llvm namespace

#endif
