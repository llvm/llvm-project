//===-- X86CodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains X86 CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86ISelDAGToDAG.h"
#include "X86TargetMachine.h"

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class X86CodeGenPassBuilder
    : public CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine> {
public:
  explicit X86CodeGenPassBuilder(X86TargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}
  void addPreISel(ModulePassManagerWrapper &MPM,
                  FunctionPassManagerWrapper &FPM) const;
  void addAsmPrinter(ModulePassManagerWrapper &MPM, CreateMCStreamer) const;
  Error addInstSelector(ModulePassManagerWrapper &MPM,
                        MachineFunctionPassManagerWrapper &FPm) const;
};

void X86CodeGenPassBuilder::addPreISel(ModulePassManagerWrapper &MPM,
                                       FunctionPassManagerWrapper &FPM) const {
  // TODO: Add passes pre instruction selection.
}

void X86CodeGenPassBuilder::addAsmPrinter(ModulePassManagerWrapper &MPM,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error X86CodeGenPassBuilder::addInstSelector(
    ModulePassManagerWrapper &MPM,
    MachineFunctionPassManagerWrapper &MFPM) const {
  // TODO: Add instruction selector related passes.
  addMachineFunctionPass(X86ISelDAGToDAGPass(TM), MFPM);
  return Error::success();
}

} // namespace

void X86TargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "X86PassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error X86TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, const CGPassBuilderOption &Opt,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = X86CodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}
