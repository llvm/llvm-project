//===-- MBlazeMCAsmInfo.h - MBlaze asm properties --------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MBlazeMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZETARGETASMINFO_H
#define MBLAZETARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;

  class MBlazeMCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit MBlazeMCAsmInfo();
  };

} // namespace llvm

#endif
