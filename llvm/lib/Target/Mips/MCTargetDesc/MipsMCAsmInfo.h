//===-- MipsMCAsmInfo.h - Mips Asm Info ------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCASMINFO_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCASMINFO_H

#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

class MipsELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit MipsELFMCAsmInfo(const Triple &TheTriple,
                            const MCTargetOptions &Options);
};

class MipsCOFFMCAsmInfo : public MCAsmInfoGNUCOFF {
  void anchor() override;

public:
  explicit MipsCOFFMCAsmInfo();
};

} // namespace llvm

#endif
