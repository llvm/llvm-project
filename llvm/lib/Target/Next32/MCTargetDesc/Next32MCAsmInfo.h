//===-- Next32MCAsmInfo.h - Next32 asm properties ------------------------====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Next32MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_MCTARGETDESC_Next32MCASMINFO_H
#define LLVM_LIB_TARGET_Next32_MCTARGETDESC_Next32MCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class Target;

class Next32MCAsmInfo : public MCAsmInfo {
public:
  explicit Next32MCAsmInfo(const Triple &TT, const MCTargetOptions &Options) {
    PrivateGlobalPrefix = ".L";
    SupportsDebugInformation = true;
    CodePointerSize = 8;
  }
};
} // namespace llvm

#endif
