//===-- AVRCodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains AVR CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "AVRTargetMachine.h"

#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {

class AVRCodeGenPassBuilder : public CodeGenPassBuilder<AVRCodeGenPassBuilder> {
public:
  explicit AVRCodeGenPassBuilder(LLVMTargetMachine &TM,
                                 CGPassBuilderOption Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
};

void AVRCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void AVRCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error AVRCodeGenPassBuilder::addInstSelector(AddMachinePass &) const {
  // TODO: Add instruction selector.
  return Error::success();
}

} // namespace

Error AVRTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, MachineFunctionPassManager &MFPM,
    MachineFunctionAnalysisManager &, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    CGPassBuilderOption Opt, PassInstrumentationCallbacks *PIC) {
  auto CGPB = AVRCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MFPM, Out, DwoOut, FileType);
}
