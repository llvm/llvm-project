//=====-- IA64TargetAsmInfo.h - IA64 asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the IA64TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef IA64TARGETASMINFO_H
#define IA64TARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class TargetMachine;

  struct IA64TargetAsmInfo : public ELFTargetAsmInfo {
    explicit IA64TargetAsmInfo(const TargetMachine &TM);
  };


} // namespace llvm

#endif
