//=- MachineLoopUtils.h - Helper functions for manipulating loops -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MACHINELOOPUTILS_H
#define LLVM_LIB_CODEGEN_MACHINELOOPUTILS_H

namespace llvm {
class MachineLoop;
class MachineBasicBlock;
class MachineRegisterInfo;
class TargetInstrInfo;

enum LoopPeelDirection {
  LPD_Front, ///< Peel the first iteration of the loop.
  LPD_Back   ///< Peel the last iteration of the loop.
};

/// Peels a single block loop. Loop must have two successors, one of which
/// must be itself. Similarly it must have two predecessors, one of which must
/// be itself.
///
/// The loop block is copied and inserted into the CFG such that two copies of
/// the loop follow on from each other. The copy is inserted either before or
/// after the loop based on Direction.
///
/// Phis are updated and an unconditional branch inserted at the end of the
/// clone so as to execute a single iteration.
///
/// The trip count of Loop is not updated.
MachineBasicBlock *PeelSingleBlockLoop(LoopPeelDirection Direction,
                                       MachineBasicBlock *Loop,
                                       MachineRegisterInfo &MRI,
                                       const TargetInstrInfo *TII);

/// Return true if PhysReg is live outside the loop, i.e. determine if it
/// is live in the loop exit blocks, and false otherwise.
bool isRegLiveInExitBlocks(MachineLoop *Loop, int PhysReg);

} // namespace llvm

#endif // LLVM_LIB_CODEGEN_MACHINELOOPUTILS_H
