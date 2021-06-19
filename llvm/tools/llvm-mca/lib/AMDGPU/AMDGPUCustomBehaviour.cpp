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
#include "llvm/Support/WithColor.h"

namespace llvm {
namespace mca {

AMDGPUCustomBehaviour::AMDGPUCustomBehaviour(const MCSubtargetInfo &STI,
                                             const SourceMgr &SrcMgr,
                                             const MCInstrInfo &MCII)
    : CustomBehaviour(STI, SrcMgr, MCII) {}

unsigned AMDGPUCustomBehaviour::checkCustomHazard(ArrayRef<InstRef> IssuedInst,
                                                  const InstRef &IR) {
  return 0;
}

} // namespace mca
} // namespace llvm
