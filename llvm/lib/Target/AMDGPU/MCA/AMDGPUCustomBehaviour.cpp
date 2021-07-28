//===------------------ AMDGPUCustomBehaviour.cpp ---------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the AMDGPUCustomBehaviour class.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUCustomBehaviour.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace mca {

AMDGPUCustomBehaviour::AMDGPUCustomBehaviour(const MCSubtargetInfo &STI,
                                             const mca::SourceMgr &SrcMgr,
                                             const MCInstrInfo &MCII)
    : CustomBehaviour(STI, SrcMgr, MCII) {}

unsigned
AMDGPUCustomBehaviour::checkCustomHazard(ArrayRef<mca::InstRef> IssuedInst,
                                         const mca::InstRef &IR) {
  return 0;
}

} // namespace mca
} // namespace llvm

using namespace llvm;
using namespace mca;

static CustomBehaviour *
createAMDGPUCustomBehaviour(const MCSubtargetInfo &STI,
                            const mca::SourceMgr &SrcMgr,
                            const MCInstrInfo &MCII) {
  return new AMDGPUCustomBehaviour(STI, SrcMgr, MCII);
}

static InstrPostProcess *
createAMDGPUInstrPostProcess(const MCSubtargetInfo &STI,
                             const MCInstrInfo &MCII) {
  return new AMDGPUInstrPostProcess(STI, MCII);
}

/// Extern function to initialize the targets for the AMDGPU backend

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAMDGPUTargetMCA() {
  TargetRegistry::RegisterCustomBehaviour(getTheAMDGPUTarget(),
                                          createAMDGPUCustomBehaviour);
  TargetRegistry::RegisterInstrPostProcess(getTheAMDGPUTarget(),
                                           createAMDGPUInstrPostProcess);

  TargetRegistry::RegisterCustomBehaviour(getTheGCNTarget(),
                                          createAMDGPUCustomBehaviour);
  TargetRegistry::RegisterInstrPostProcess(getTheGCNTarget(),
                                           createAMDGPUInstrPostProcess);
}
