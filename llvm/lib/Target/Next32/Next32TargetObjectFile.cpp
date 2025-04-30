//===------- Next32TargetObjectFile.cpp - Next32 Object Info Impl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32TargetObjectFile.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

void Next32ELFTargetObjectFile::Initialize(MCContext &Ctx,
                                           const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}
