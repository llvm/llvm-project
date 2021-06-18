//===- MBFIWrapper.cpp - MachineBlockFrequencyInfo wrapper ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class keeps track of branch frequencies of newly created blocks and
// tail-merged blocks. Used by the TailDuplication and MachineBlockPlacement.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MBFIWrapper.h"

using namespace llvm;

BlockFrequency MBFIWrapper::getBlockFreq(const MachineBasicBlock *MBB) const {
  auto I = MergedBBFreq.find(MBB);

  if (I != MergedBBFreq.end())
    return I->second;

  return MBFI.getBlockFreq(MBB);
}

void MBFIWrapper::setBlockFreq(const MachineBasicBlock *MBB,
                               BlockFrequency F) {
  MergedBBFreq[MBB] = F;
}

Optional<uint64_t>
MBFIWrapper::getBlockProfileCount(const MachineBasicBlock *MBB) const {
  auto I = MergedBBFreq.find(MBB);

  // Modified block frequency also impacts profile count. So we should compute
  // profile count from new block frequency if it has been changed.
  if (I != MergedBBFreq.end())
    return MBFI.getProfileCountFromFreq(I->second.getFrequency());

  return MBFI.getBlockProfileCount(MBB);
}

raw_ostream & MBFIWrapper::printBlockFreq(raw_ostream &OS,
                                          const MachineBasicBlock *MBB) const {
  return MBFI.printBlockFreq(OS, getBlockFreq(MBB));
}

raw_ostream & MBFIWrapper::printBlockFreq(raw_ostream &OS,
                                          const BlockFrequency Freq) const {
  return MBFI.printBlockFreq(OS, Freq);
}

void MBFIWrapper::view(const Twine &Name, bool isSimple) {
  MBFI.view(Name, isSimple);
}

uint64_t MBFIWrapper::getEntryFreq() const {
  return MBFI.getEntryFreq();
}
