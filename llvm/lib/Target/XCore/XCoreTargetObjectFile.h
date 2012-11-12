//===-- XCoreTargetObjectFile.h - XCore Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_XCORE_TARGETOBJECTFILE_H
#define LLVM_TARGET_XCORE_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

  class XCoreTargetObjectFile : public TargetLoweringObjectFileELF {
  public:
    void Initialize(MCContext &Ctx, const TargetMachine &TM);

    // TODO: Classify globals as xcore wishes.
  };
} // end namespace llvm

#endif
