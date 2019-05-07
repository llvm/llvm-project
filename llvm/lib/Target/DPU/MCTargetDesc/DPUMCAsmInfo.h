//===-- DPUMCAsmInfo.h - DPU asm properties -------------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the DPUMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Target;

class DPUMCAsmInfo : public MCAsmInfo {
public:
  explicit DPUMCAsmInfo(const Triple &TT) {
    WeakRefDirective = ".weak";
    PrivateGlobalPrefix = ".L";
    PrivateLabelPrefix = ".L";

    UseIntegratedAssembler = true;

    UsesELFSectionDirectiveForBSS = true;
    CommentString = "//";

    InlineAsmStart = " inline asm";
    InlineAsmEnd = " inline asm";

    SupportsDebugInformation = true;
  }
};
} // namespace llvm
