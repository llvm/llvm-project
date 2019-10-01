//===- MCAsmInfoDarwin.h - Darwin asm properties ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMINFODARWIN_H
#define LLVM_MC_MCASMINFODARWIN_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {

class MCAsmInfoDarwin : public MCAsmInfo {
public:
  explicit MCAsmInfoDarwin();

  bool isSectionAtomizableBySymbols(const MCSection &Section) const override;
};

} // end namespace llvm

#endif // LLVM_MC_MCASMINFODARWIN_H
