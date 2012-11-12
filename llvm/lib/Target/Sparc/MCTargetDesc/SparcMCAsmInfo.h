//===-- SparcMCAsmInfo.h - Sparc asm properties ----------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SparcMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCTARGETASMINFO_H
#define SPARCTARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class StringRef;
  class Target;

  class SparcELFMCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit SparcELFMCAsmInfo(const Target &T, StringRef TT);
  };

} // namespace llvm

#endif
