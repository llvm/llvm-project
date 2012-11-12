//===-- NVPTXMCAsmInfo.h - NVPTX asm properties ----------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVPTXMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_MCASM_INFO_H
#define NVPTX_MCASM_INFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Target;
class StringRef;

class NVPTXMCAsmInfo : public MCAsmInfo {
  virtual void anchor();
public:
  explicit NVPTXMCAsmInfo(const Target &T, const StringRef &TT);
};
} // namespace llvm

#endif // NVPTX_MCASM_INFO_H
