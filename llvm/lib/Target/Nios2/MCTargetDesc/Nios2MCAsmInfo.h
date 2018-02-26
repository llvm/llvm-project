//===-- Nios2MCAsmInfo.h - Nios2 Asm Info ----------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Nios2MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCASMINFO_H
#define LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

class Nios2MCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit Nios2MCAsmInfo(const Triple &TheTriple);
};

} // namespace llvm

#endif
