//===-- Next32TargetObjectFile.h - Next32 Object Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32TARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {
class MCContext;
class TargetMachine;

class Next32ELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  Next32ELFTargetObjectFile() : TargetLoweringObjectFileELF() {}

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;
};

} // end namespace llvm

#endif
