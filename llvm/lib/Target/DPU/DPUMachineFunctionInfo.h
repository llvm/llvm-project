//===- DPUMachineFuctionInfo.h - DPU machine func info -------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares DPU-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_DPU_DPUMACHINEFUNCTIONINFO_H

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include <set>

#define UNDEF_FI INT_MAX

namespace llvm {

// DPUMachineFunctionInfo - This class is derived from MachineFunction and
// contains private DPU target-specific information for each MachineFunction.
class DPUMachineFunctionInfo : public MachineFunctionInfo {
  virtual void anchor();

  MachineFrameInfo &MFI;
  std::set<int> frameIndexOffsetSet;
  int FI;

public:
  explicit DPUMachineFunctionInfo(MachineFunction &MF)
      : MFI(MF.getFrameInfo()), frameIndexOffsetSet(), FI(UNDEF_FI) {}

  int getOffsetFromFrameIndex(int FrameIndex);
  int getTemporaryFrameIndex();
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DPUMACHINEFUNCTIONINFO_H
