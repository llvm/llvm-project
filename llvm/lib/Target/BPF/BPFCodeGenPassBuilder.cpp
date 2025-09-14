//===-- BPFCodeGenPassBuilder.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements BPFCodeGenPassBuilder class.
//
//===----------------------------------------------------------------------===//

#include "BPFCodeGenPassBuilder.h"
#include "BPFTargetMachine.h"

using namespace llvm;

BPFCodeGenPassBuilder::BPFCodeGenPassBuilder(BPFTargetMachine &TM,
                                             const CGPassBuilderOption &Opts,
                                             PassInstrumentationCallbacks *PIC)
    : CodeGenPassBuilder(TM, Opts, PIC) {}

void BPFCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void BPFCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}
