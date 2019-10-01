
//===------------ MIRVRegNamerUtils.h - MIR VReg Renaming Utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of these utilities is to abstract out parts of the MIRCanon pass
// that are responsible for renaming virtual registers with the purpose of
// sharing code with a MIRVRegNamer pass that could be the analog of the
// opt -instnamer pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRVREGNAMERUTILS_H
#define LLVM_LIB_CODEGEN_MIRVREGNAMERUTILS_H

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/raw_ostream.h"

#include <queue>

namespace llvm {

/// NamedVRegCursor - The cursor is an object that keeps track of what the next
/// vreg name should be. It does book keeping to determine when to skip the
/// index value and by how much, or if the next vreg name should be an increment
/// from the previous.
class NamedVRegCursor {
  MachineRegisterInfo &MRI;

  /// virtualVRegNumber - Book keeping of the last vreg position.
  unsigned virtualVRegNumber;

  /// SkipGapSize - Used to calculate a modulo amount to skip by after every
  /// sequence of instructions starting from a given side-effecting
  /// MachineInstruction for a given MachineBasicBlock. The general idea is that
  /// for a given program compiled with two different opt pipelines, there
  /// shouldn't be greater than SkipGapSize difference in how many vregs are in
  /// play between the two and for every def-use graph of vregs we rename we
  /// will round up to the next SkipGapSize'th number so that we have a high
  /// change of landing on the same name for two given matching side-effects
  /// for the two compilation outcomes.
  const unsigned SkipGapSize;

  /// RenamedInOtherBB - VRegs that we already renamed: ie breadcrumbs.
  std::vector<Register> RenamedInOtherBB;

public:
  NamedVRegCursor() = delete;
  /// 1000 for the SkipGapSize was a good heuristic at the time of the writing
  /// of the MIRCanonicalizerPass. Adjust as needed.
  NamedVRegCursor(MachineRegisterInfo &MRI, unsigned SkipGapSize = 1000)
      : MRI(MRI), virtualVRegNumber(0), SkipGapSize(SkipGapSize) {}

  /// SkipGapSize - Skips modulo a gap value of indices. Indices are used to
  /// produce the next vreg name.
  void skipVRegs();

  unsigned getVirtualVReg() const { return virtualVRegNumber; }

  /// incrementVirtualVReg - This increments an index value that us used to
  /// create a new vreg name. This is not a Register.
  unsigned incrementVirtualVReg(unsigned incr = 1) {
    virtualVRegNumber += incr;
    return virtualVRegNumber;
  }

  /// createVirtualRegister - Given an existing vreg, create a named vreg to
  /// take its place.
  unsigned createVirtualRegister(unsigned VReg);

  /// renameVRegs - For a given MachineBasicBlock, scan for side-effecting
  /// instructions, walk the def-use from each side-effecting root (in sorted
  /// root order) and rename the encountered vregs in the def-use graph in a
  /// canonical ordering. This method maintains book keeping for which vregs
  /// were already renamed in RenamedInOtherBB.
  // @return changed
  bool renameVRegs(MachineBasicBlock *MBB);
};

} // namespace llvm

#endif
