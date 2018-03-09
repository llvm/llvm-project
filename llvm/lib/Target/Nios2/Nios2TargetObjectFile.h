//===-- llvm/Target/Nios2TargetObjectFile.h - Nios2 Object Info -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2TARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2TARGETOBJECTFILE_H

#include "Nios2TargetMachine.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class Nios2TargetObjectFile : public TargetLoweringObjectFileELF {
  const Nios2TargetMachine *TM;

public:
  Nios2TargetObjectFile() : TargetLoweringObjectFileELF() {}

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};
} // end namespace llvm

#endif
