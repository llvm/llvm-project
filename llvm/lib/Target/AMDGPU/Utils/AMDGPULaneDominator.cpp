//===-- AMDGPULaneDominator.cpp - Determine Lane Dominators ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MBB A lane-dominates MBB B if
// 1. A dominates B in the usual sense, i.e. every path from the entry to B
//    goes through A, and
// 2. whenever B executes, every active lane during that execution of B was
//    also active during the most recent execution of A.
//
// The simplest example where A dominates B but does not lane-dominate it is
// where A is a loop:
//
//     |
//     +--+
//     A  |
//     +--+
//     |
//     B
//
// Unfortunately, the second condition is not fully captured by the control
// flow graph when it is unstructured (as may happen when branch conditions are
// uniform).
//
// The following replacement of the second condition is a conservative
// approximation. It is an equivalent condition when the CFG is fully
// structured:
//
// 2'. every cycle in the CFG that contains A also contains B.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULaneDominator.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

namespace AMDGPU {

// Given machine basic blocks A and B where A dominates B, check whether
// A lane-dominates B.
//
// The check is conservative, i.e. there can be false-negatives.
bool laneDominates(MachineBasicBlock *A, MachineBasicBlock *B) {
  // Check whether A is reachable from itself without going through B.
  DenseSet<MachineBasicBlock *> Reachable;
  SmallVector<MachineBasicBlock *, 8> Stack;

  Stack.push_back(A);
  do {
    MachineBasicBlock *MBB = Stack.back();
    Stack.pop_back();

    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (Succ == A)
        return false;
      if (Succ != B && Reachable.insert(Succ).second)
        Stack.push_back(Succ);
    }
  } while (!Stack.empty());

  return true;
}

} // namespace AMDGPU

} // namespace llvm
