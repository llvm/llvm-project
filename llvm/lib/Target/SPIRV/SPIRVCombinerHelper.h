//===-- SPIRVCombinerHelper.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This contains common combine transformations that may be used in a combine
/// pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H

#include "SPIRVSubtarget.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"

namespace llvm {
class SPIRVCombinerHelper : public CombinerHelper {
protected:
  const SPIRVSubtarget &STI;

public:
  using CombinerHelper::CombinerHelper;
  SPIRVCombinerHelper(GISelChangeObserver &Observer, MachineIRBuilder &B,
                      bool IsPreLegalize, GISelValueTracking *VT,
                      MachineDominatorTree *MDT, const LegalizerInfo *LI,
                      const SPIRVSubtarget &STI);

  bool matchLengthToDistance(MachineInstr &MI) const;
  void applySPIRVDistance(MachineInstr &MI) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVCOMBINERHELPER_H
