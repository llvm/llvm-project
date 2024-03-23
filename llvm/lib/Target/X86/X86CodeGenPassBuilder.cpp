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

#include "X86ISelDAGToDAG.h"
#include "X86TargetMachine.h"

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class X86CodeGenPassBuilder : public CodeGenPassBuilder<X86CodeGenPassBuilder> {
public:
  explicit X86CodeGenPassBuilder(LLVMTargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
};

void X86CodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void X86CodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error X86CodeGenPassBuilder::addInstSelector(AddMachinePass &addPass) const {
  // TODO: Add instruction selector.
  addPass(X86ISelDAGToDAGPass(static_cast<X86TargetMachine &>(TM)));
  return Error::success();
}

} // namespace

void X86TargetMachine::registerPassBuilderCallbacks(
    PassBuilder &PB, bool PopulateClassToPassNames) {
  if (PopulateClassToPassNames) {
    auto *PIC = PB.getPassInstrumentationCallbacks();
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS)                               \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#include "X86PassRegistry.def"
  }

  PB.registerPipelineParsingCallback(
      [this](StringRef Name, MachineFunctionPassManager &MFPM,
             ArrayRef<PassBuilder::PipelineElement>) {
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS)                               \
  if (Name == NAME) {                                                          \
    MFPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#include "X86PassRegistry.def"
        return false;
      });
}

Error X86TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, const CGPassBuilderOption &Opt,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = X86CodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}
