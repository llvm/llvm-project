//===------------------- AMDGPUCustomBehaviour.h ----------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the AMDGPUCustomBehaviour class which inherits from
/// CustomBehaviour.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_LIB_AMDGPU_AMDGPUCUSTOMBEHAVIOUR_H
#define LLVM_TOOLS_LLVM_MCA_LIB_AMDGPU_AMDGPUCUSTOMBEHAVIOUR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Support/TargetParser.h"

namespace llvm {
namespace mca {

class AMDGPUInstrPostProcess : public InstrPostProcess {
public:
  AMDGPUInstrPostProcess(const MCSubtargetInfo &STI, const MCInstrInfo &MCII)
      : InstrPostProcess(STI, MCII) {}

  ~AMDGPUInstrPostProcess() {}

  void postProcessInstruction(std::unique_ptr<Instruction> &Inst,
                              const MCInst &MCI) override {}
};

class AMDGPUCustomBehaviour : public CustomBehaviour {
public:
  AMDGPUCustomBehaviour(const MCSubtargetInfo &STI, const SourceMgr &SrcMgr,
                        const MCInstrInfo &MCII);

  ~AMDGPUCustomBehaviour() {}

  /// This method is used to determine if an instruction
  /// should be allowed to be dispatched. The return value is
  /// how many cycles until the instruction can be dispatched.
  /// This method is called after MCA has already checked for
  /// register and hardware dependencies so this method should only
  /// implement custom behaviour and dependencies that are not picked up
  /// by MCA naturally.
  unsigned checkCustomHazard(ArrayRef<InstRef> IssuedInst,
                             const InstRef &IR) override;
};

} // namespace mca
} // namespace llvm

#endif /* LLVM_TOOLS_LLVM_MCA_LIB_AMDGPU_AMDGPUCUSTOMBEHAVIOUR_H */
