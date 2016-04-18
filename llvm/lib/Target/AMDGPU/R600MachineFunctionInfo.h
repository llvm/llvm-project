//===-- R600MachineFunctionInfo.h - R600 Machine Function Info ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_R600MACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_AMDGPU_R600MACHINEFUNCTIONINFO_H

#include "AMDGPUMachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include <vector>

namespace llvm {

class R600MachineFunctionInfo final : public AMDGPUMachineFunction {
  void anchor() override;
public:
  R600MachineFunctionInfo(const MachineFunction &MF);
  SmallVector<unsigned, 4> LiveOuts;
  std::vector<unsigned> IndirectRegs;
  unsigned StackSize;
};

} // End llvm namespace

#endif
