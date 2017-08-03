//===-- ARMBaseInfo.h - Top level definitions for ARM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the ARM target useful for the compiler back-end and the MC libraries.
// As such, it deliberately does not include references to LLVM core
// code gen types, passes, etc..
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_UTILS_ARMBASEINFO_H
#define LLVM_LIB_TARGET_ARM_UTILS_ARMBASEINFO_H

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/MC/SubtargetFeature.h"
#include "MCTargetDesc/ARMMCTargetDesc.h"

namespace llvm {

// System Registers
namespace ARMSysReg {
  struct MClassSysReg {
    const char *Name;
    uint16_t M1Encoding12;
    uint16_t M2M3Encoding8;
    uint16_t Encoding;
    FeatureBitset FeaturesRequired;

    // return true if FeaturesRequired are all present in ActiveFeatures
    bool hasRequiredFeatures(FeatureBitset ActiveFeatures) const {
      return (FeaturesRequired & ActiveFeatures) == FeaturesRequired;
    }

    // returns true if TestFeatures are all present in FeaturesRequired
    bool isInRequiredFeatures(FeatureBitset TestFeatures) const {
      return (FeaturesRequired & TestFeatures) == TestFeatures;
    }
  };

  #define GET_MCLASSSYSREG_DECL
  #include "ARMGenSystemRegister.inc"

  // lookup system register using 12-bit SYSm value.
  // Note: the search is uniqued using M1 mask
  const MClassSysReg *lookupMClassSysRegBy12bitSYSmValue(unsigned SYSm);

  // returns APSR with _<bits> qualifier.
  // Note: ARMv7-M deprecates using MSR APSR without a _<bits> qualifier
  const MClassSysReg *lookupMClassSysRegAPSRNonDeprecated(unsigned SYSm);

  // lookup system registers using 8-bit SYSm value
  const MClassSysReg *lookupMClassSysRegBy8bitSYSmValue(unsigned SYSm);

} // end namespace ARMSysReg

// Banked Registers
namespace ARMBankedReg {
  struct BankedReg {
    const char *Name;
    uint16_t Encoding;
  };
  #define GET_BANKEDREG_DECL
  #include "ARMGenSystemRegister.inc"
} // end namespace ARMBankedReg

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_UTILS_ARMBASEINFO_H
