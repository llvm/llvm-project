//===-- PISAMCAsmInfo.h - PISA asm properties -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCASMINFO_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {

class Triple;

class PISAMCAsmInfo : public MCAsmInfo {
public:
  explicit PISAMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
  bool shouldOmitSectionDirective(StringRef SectionName) const override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAMCASMINFO_H
