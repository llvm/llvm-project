//===-- XCoreMCAsmInfo.h - XCore asm properties ----------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the XCoreMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETASMINFO_H
#define XCORETARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class StringRef;
  class Target;

  class XCoreMCAsmInfo : public MCAsmInfo {
    virtual void anchor();
  public:
    explicit XCoreMCAsmInfo(const Target &T, StringRef TT);
  };

} // namespace llvm

#endif
