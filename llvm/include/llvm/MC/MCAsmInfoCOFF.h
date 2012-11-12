//===-- MCAsmInfoCOFF.h - COFF asm properties -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COFF_TARGET_ASM_INFO_H
#define LLVM_COFF_TARGET_ASM_INFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class MCAsmInfoCOFF : public MCAsmInfo {
    virtual void anchor();
  protected:
    explicit MCAsmInfoCOFF();
  };

  class MCAsmInfoMicrosoft : public MCAsmInfoCOFF {
    virtual void anchor();
  protected:
    explicit MCAsmInfoMicrosoft();
  };

  class MCAsmInfoGNUCOFF : public MCAsmInfoCOFF {
    virtual void anchor();
  protected:
    explicit MCAsmInfoGNUCOFF();
  };
}


#endif // LLVM_COFF_TARGET_ASM_INFO_H
