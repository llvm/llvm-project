//===-- PISAMachineFunctionInfo.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_PISA_PISAMACHINEFUNCTIONINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {
class PISASubtarget;

class PISAMachineFunctionInfo : public MachineFunctionInfo {
  using ArgInfo = std::pair<unsigned, bool>;
  DenseMap<unsigned, ArgInfo> ArgInfos;

public:
  PISAMachineFunctionInfo(const Function &F, const PISASubtarget *STI);
  ~PISAMachineFunctionInfo() override;

  ArgInfo getArgInfo(unsigned Slot) const {
    auto I = ArgInfos.find(Slot);
    if (I == ArgInfos.end())
      return {0, false};
    return I->second;
  }
  void setArgInfo(unsigned Slot, unsigned ArgSize, bool IsByRef) {
    ArgInfos[Slot] = {ArgSize, IsByRef};
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAMACHINEFUNCTIONINFO_H
