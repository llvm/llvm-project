//===-- PISALegalizerInfo.h --- PISA Legalization Rules -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISALEGALIZERINFO_H
#define LLVM_LIB_TARGET_PISA_PISALEGALIZERINFO_H

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

namespace llvm {

class LLVMContext;
class PISASubtarget;

// This class provides the information for legalizing PISA instructions.
class PISALegalizerInfo : public LegalizerInfo {

public:
  bool legalizeIntrinsic(LegalizerHelper &Helper,
                         MachineInstr &MI) const override;
  bool legalizeCustom(LegalizerHelper &Helper, MachineInstr &MI,
                      LostDebugLocObserver &LocObserver) const override;
  PISALegalizerInfo(const PISASubtarget &ST);
};
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_PISALEGALIZERINFO_H
