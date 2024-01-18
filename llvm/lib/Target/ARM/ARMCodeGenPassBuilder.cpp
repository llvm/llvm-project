//===-- ARMCodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains ARM CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "ARMTargetMachine.h"

#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {

class ARMCodeGenPassBuilder : public CodeGenPassBuilder<ARMCodeGenPassBuilder> {
public:
  explicit ARMCodeGenPassBuilder(LLVMTargetMachine &TM,
                                 CGPassBuilderOption Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
};

void ARMCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void ARMCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error ARMCodeGenPassBuilder::addInstSelector(AddMachinePass &) const {
  // TODO: Add instruction selector.
  return Error::success();
}

} // namespace

Error ARMBaseTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, MachineFunctionPassManager &MFPM,
    MachineFunctionAnalysisManager &, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    CGPassBuilderOption Opt, PassInstrumentationCallbacks *PIC) {
  auto CGPB = ARMCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MFPM, Out, DwoOut, FileType);
}
