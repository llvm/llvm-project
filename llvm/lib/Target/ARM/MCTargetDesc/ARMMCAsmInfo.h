//===-- ARMMCAsmInfo.h - ARM asm properties --------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARMTARGETASMINFO_H
#define LLVM_ARMTARGETASMINFO_H

#include "llvm/MC/MCAsmInfoDarwin.h"

namespace llvm {

  class ARMMCAsmInfoDarwin : public MCAsmInfoDarwin {
    virtual void anchor();
  public:
    explicit ARMMCAsmInfoDarwin();
  };

  class ARMELFMCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit ARMELFMCAsmInfo();
  };

} // namespace llvm

#endif
