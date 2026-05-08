//===- MCAsmInfoCOFF.h - COFF asm properties --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMINFOCOFF_H
#define LLVM_MC_MCASMINFOCOFF_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {

class MCAsmInfoCOFF : public MCAsmInfo {
  virtual void anchor();
  void printSwitchToSection(const MCSection &, uint32_t, const Triple &,
                            raw_ostream &) const final;
  bool useCodeAlign(const MCSection &Sec) const final;

protected:
  explicit MCAsmInfoCOFF(const MCTargetOptions &Options);
};

class MCAsmInfoMicrosoft : public MCAsmInfoCOFF {
  void anchor() override;

protected:
  explicit MCAsmInfoMicrosoft(const MCTargetOptions &Options);
};

class MCAsmInfoGNUCOFF : public MCAsmInfoCOFF {
  void anchor() override;

protected:
  explicit MCAsmInfoGNUCOFF(const MCTargetOptions &Options);
};

} // end namespace llvm

#endif // LLVM_MC_MCASMINFOCOFF_H
