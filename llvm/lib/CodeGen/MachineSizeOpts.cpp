//===- MachineSizeOpts.cpp - code size optimization related code ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some shared machine IR code size optimization related
// code.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"

using namespace llvm;

extern cl::opt<bool> EnablePGSO;
extern cl::opt<bool> PGSOLargeWorkingSetSizeOnly;
extern cl::opt<bool> ForcePGSO;
extern cl::opt<int> PgsoCutoffInstrProf;
extern cl::opt<int> PgsoCutoffSampleProf;

namespace {
namespace machine_size_opts_detail {

/// Like ProfileSummaryInfo::isColdBlock but for MachineBasicBlock.
bool isColdBlock(const MachineBasicBlock *MBB,
                 ProfileSummaryInfo *PSI,
                 const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getBlockProfileCount(MBB);
  return Count && PSI->isColdCount(*Count);
}

bool isColdBlock(BlockFrequency BlockFreq,
                 ProfileSummaryInfo *PSI,
                 const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getProfileCountFromFreq(BlockFreq.getFrequency());
  return Count && PSI->isColdCount(*Count);
}

/// Like ProfileSummaryInfo::isHotBlockNthPercentile but for MachineBasicBlock.
static bool isHotBlockNthPercentile(int PercentileCutoff,
                                    const MachineBasicBlock *MBB,
                                    ProfileSummaryInfo *PSI,
                                    const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getBlockProfileCount(MBB);
  return Count && PSI->isHotCountNthPercentile(PercentileCutoff, *Count);
}

static bool isHotBlockNthPercentile(int PercentileCutoff,
                                    BlockFrequency BlockFreq,
                                    ProfileSummaryInfo *PSI,
                                    const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getProfileCountFromFreq(BlockFreq.getFrequency());
  return Count && PSI->isHotCountNthPercentile(PercentileCutoff, *Count);
}

static bool isColdBlockNthPercentile(int PercentileCutoff,
                                     const MachineBasicBlock *MBB,
                                     ProfileSummaryInfo *PSI,
                                     const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getBlockProfileCount(MBB);
  return Count && PSI->isColdCountNthPercentile(PercentileCutoff, *Count);
}

static bool isColdBlockNthPercentile(int PercentileCutoff,
                                     BlockFrequency BlockFreq,
                                     ProfileSummaryInfo *PSI,
                                     const MachineBlockFrequencyInfo *MBFI) {
  auto Count = MBFI->getProfileCountFromFreq(BlockFreq.getFrequency());
  return Count && PSI->isColdCountNthPercentile(PercentileCutoff, *Count);
}

/// Like ProfileSummaryInfo::isFunctionColdInCallGraph but for
/// MachineFunction.
bool isFunctionColdInCallGraph(
    const MachineFunction *MF,
    ProfileSummaryInfo *PSI,
    const MachineBlockFrequencyInfo &MBFI) {
  if (auto FunctionCount = MF->getFunction().getEntryCount())
    if (!PSI->isColdCount(FunctionCount->getCount()))
      return false;
  for (const auto &MBB : *MF)
    if (!isColdBlock(&MBB, PSI, &MBFI))
      return false;
  return true;
}

/// Like ProfileSummaryInfo::isFunctionHotInCallGraphNthPercentile but for
/// MachineFunction.
bool isFunctionHotInCallGraphNthPercentile(
    int PercentileCutoff,
    const MachineFunction *MF,
    ProfileSummaryInfo *PSI,
    const MachineBlockFrequencyInfo &MBFI) {
  if (auto FunctionCount = MF->getFunction().getEntryCount())
    if (PSI->isHotCountNthPercentile(PercentileCutoff,
                                     FunctionCount->getCount()))
      return true;
  for (const auto &MBB : *MF)
    if (isHotBlockNthPercentile(PercentileCutoff, &MBB, PSI, &MBFI))
      return true;
  return false;
}

bool isFunctionColdInCallGraphNthPercentile(
    int PercentileCutoff, const MachineFunction *MF, ProfileSummaryInfo *PSI,
    const MachineBlockFrequencyInfo &MBFI) {
  if (auto FunctionCount = MF->getFunction().getEntryCount())
    if (!PSI->isColdCountNthPercentile(PercentileCutoff,
                                       FunctionCount->getCount()))
      return false;
  for (const auto &MBB : *MF)
    if (!isColdBlockNthPercentile(PercentileCutoff, &MBB, PSI, &MBFI))
      return false;
  return true;
}
} // namespace machine_size_opts_detail

struct MachineBasicBlockBFIAdapter {
  static bool isFunctionColdInCallGraph(const MachineFunction *MF,
                                        ProfileSummaryInfo *PSI,
                                        const MachineBlockFrequencyInfo &MBFI) {
    return machine_size_opts_detail::isFunctionColdInCallGraph(MF, PSI, MBFI);
  }
  static bool isFunctionHotInCallGraphNthPercentile(
      int CutOff,
      const MachineFunction *MF,
      ProfileSummaryInfo *PSI,
      const MachineBlockFrequencyInfo &MBFI) {
    return machine_size_opts_detail::isFunctionHotInCallGraphNthPercentile(
        CutOff, MF, PSI, MBFI);
  }
  static bool isFunctionColdInCallGraphNthPercentile(
      int CutOff, const MachineFunction *MF, ProfileSummaryInfo *PSI,
      const MachineBlockFrequencyInfo &MBFI) {
    return machine_size_opts_detail::isFunctionColdInCallGraphNthPercentile(
        CutOff, MF, PSI, MBFI);
  }
  static bool isColdBlock(const MachineBasicBlock *MBB,
                          ProfileSummaryInfo *PSI,
                          const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isColdBlock(MBB, PSI, MBFI);
  }
  static bool isColdBlock(BlockFrequency BlockFreq,
                          ProfileSummaryInfo *PSI,
                          const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isColdBlock(BlockFreq, PSI, MBFI);
  }
  static bool isHotBlockNthPercentile(int CutOff,
                                      const MachineBasicBlock *MBB,
                                      ProfileSummaryInfo *PSI,
                                      const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isHotBlockNthPercentile(
        CutOff, MBB, PSI, MBFI);
  }
  static bool isHotBlockNthPercentile(int CutOff,
                                      BlockFrequency BlockFreq,
                                      ProfileSummaryInfo *PSI,
                                      const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isHotBlockNthPercentile(
        CutOff, BlockFreq, PSI, MBFI);
  }
  static bool isColdBlockNthPercentile(int CutOff, const MachineBasicBlock *MBB,
                                       ProfileSummaryInfo *PSI,
                                       const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isColdBlockNthPercentile(CutOff, MBB, PSI,
                                                              MBFI);
  }
  static bool isColdBlockNthPercentile(int CutOff, BlockFrequency BlockFreq,
                                       ProfileSummaryInfo *PSI,
                                       const MachineBlockFrequencyInfo *MBFI) {
    return machine_size_opts_detail::isColdBlockNthPercentile(CutOff, BlockFreq,
                                                              PSI, MBFI);
  }
};
} // end anonymous namespace

bool llvm::shouldOptimizeForSize(const MachineFunction *MF,
                                 ProfileSummaryInfo *PSI,
                                 const MachineBlockFrequencyInfo *MBFI,
                                 PGSOQueryType QueryType) {
  return shouldFuncOptimizeForSizeImpl<MachineBasicBlockBFIAdapter>(
      MF, PSI, MBFI, QueryType);
}

bool llvm::shouldOptimizeForSize(const MachineBasicBlock *MBB,
                                 ProfileSummaryInfo *PSI,
                                 const MachineBlockFrequencyInfo *MBFI,
                                 PGSOQueryType QueryType) {
  assert(MBB);
  return shouldOptimizeForSizeImpl<MachineBasicBlockBFIAdapter>(
      MBB, PSI, MBFI, QueryType);
}

bool llvm::shouldOptimizeForSize(const MachineBasicBlock *MBB,
                                 ProfileSummaryInfo *PSI,
                                 MBFIWrapper *MBFIW,
                                 PGSOQueryType QueryType) {
  assert(MBB);
  if (!PSI || !MBFIW)
    return false;
  BlockFrequency BlockFreq = MBFIW->getBlockFreq(MBB);
  return shouldOptimizeForSizeImpl<MachineBasicBlockBFIAdapter>(
      BlockFreq, PSI, &MBFIW->getMBFI(), QueryType);
}
