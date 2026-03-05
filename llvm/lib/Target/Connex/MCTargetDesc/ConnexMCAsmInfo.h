//===-- ConnexMCAsmInfo.h - Connex asm properties -------------*- C++ -*--====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ConnexMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_MCTARGETDESC_CONNEXMCASMINFO_H
#define LLVM_LIB_TARGET_CONNEX_MCTARGETDESC_CONNEXMCASMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Target;
class Triple;

class ConnexMCAsmInfo : public MCAsmInfo {
public:
  explicit ConnexMCAsmInfo(const Triple &TT, const MCTargetOptions &Options) {
    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";

    // Inspired from llvm.org/docs/doxygen/html/NVPTXMCAsmInfo_8cpp_source.html
    // Avoiding to add APP and NO_APP delimiters before ASM Inline Expressions
    CommentString = "//";
    InlineAsmStart = "";
    InlineAsmEnd = "";

    UsesELFSectionDirectiveForBSS = true;
    HasSingleParameterDotFile = false;
    HasDotTypeDotSizeDirective = false;

    SupportsDebugInformation = true;
  }
};
} // End namespace llvm

#endif
