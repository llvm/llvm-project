//===-- MipsCodeGenPassBuilder.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains Mips CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "MipsTargetMachine.h"

#include "llvm/CodeGen/CodeGenPassBuilder.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace {

class MipsCodeGenPassBuilder
    : public CodeGenPassBuilder<MipsCodeGenPassBuilder> {
public:
  explicit MipsCodeGenPassBuilder(LLVMTargetMachine &TM,
                                  CGPassBuilderOption Opts,
                                  PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
};

void MipsCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void MipsCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                           CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error MipsCodeGenPassBuilder::addInstSelector(AddMachinePass &) const {
  // TODO: Add instruction selector.
  return Error::success();
}

} // namespace

Error MipsTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, MachineFunctionPassManager &MFPM,
    MachineFunctionAnalysisManager &, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    CGPassBuilderOption Opt, PassInstrumentationCallbacks *PIC) {
  auto CGPB = MipsCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MFPM, Out, DwoOut, FileType);
}
