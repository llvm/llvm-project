//===-- XtensaMCAsmInfo.h - Xtensa Asm Info --------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the XtensaMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSATARGETASMINFO_H
#define LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSATARGETASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {
class Triple;

class XtensaMCAsmInfo : public MCAsmInfoELF {
public:
  explicit XtensaMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
};

namespace Xtensa {
enum Specifier { S_None, S_TPOFF };
} // namespace Xtensa

} // namespace llvm

#endif // LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSATARGETASMINFO_H
