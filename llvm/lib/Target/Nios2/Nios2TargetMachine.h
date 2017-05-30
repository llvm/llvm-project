//===-- Nios2TargetMachine.h - Define TargetMachine for Nios2 ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Nios2 specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2TARGETMACHINE_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2TARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {
class Nios2TargetMachine : public LLVMTargetMachine {
public:
  Nios2TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, CodeModel::Model CM,
                     CodeGenOpt::Level OL);
  ~Nios2TargetMachine() override;
};
} // namespace llvm

#endif
