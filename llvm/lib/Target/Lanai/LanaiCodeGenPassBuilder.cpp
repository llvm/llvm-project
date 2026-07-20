//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "LanaiAsmPrinter.h"
#include "LanaiTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"

using namespace llvm;

namespace {

class LanaiCodeGenPassBuilder
    : public CodeGenPassBuilder<LanaiCodeGenPassBuilder, LanaiTargetMachine> {
  using Base = CodeGenPassBuilder<LanaiCodeGenPassBuilder, LanaiTargetMachine>;

public:
  explicit LanaiCodeGenPassBuilder(LanaiTargetMachine &TM,
                                   const CGPassBuilderOption &Opts,
                                   PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(CodeGenModulePassManager &CGMPM) const;
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreSched2(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterBegin(CodeGenModulePassManager &CGMPM) const;
  void addAsmPrinter(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterEnd(CodeGenModulePassManager &CGMPM) const;
};

void LanaiCodeGenPassBuilder::addIRPasses(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addPass(AtomicExpandPass(TM));
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  Base::addIRPasses(CGMPM);
}

Error LanaiCodeGenPassBuilder::addInstSelector(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(LanaiISelDAGToDAGPass(TM));
  return Error::success();
}

void LanaiCodeGenPassBuilder::addPreSched2(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(LanaiMemAluCombinerPass());
}

void LanaiCodeGenPassBuilder::addPreEmitPass(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(LanaiDelaySlotFillerPass());
}

void LanaiCodeGenPassBuilder::addAsmPrinterBegin(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(LanaiAsmPrinterBeginPass());
}

void LanaiCodeGenPassBuilder::addAsmPrinter(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(LanaiAsmPrinterPass());
}

void LanaiCodeGenPassBuilder::addAsmPrinterEnd(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(LanaiAsmPrinterEndPass());
}

} // namespace

void LanaiTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "LanaiPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
  // TODO(boomanaiden154): Move this into the base CodeGenPassBuilder once all
  // targets that currently implement it have a ported asm-printer pass.
  if (PIC) {
    PIC->addClassToPassName(LanaiAsmPrinterBeginPass::name(),
                            "lanai-asm-printer-begin");
    PIC->addClassToPassName(LanaiAsmPrinterPass::name(), "lanai-asmprinter");
    PIC->addClassToPassName(LanaiAsmPrinterEndPass::name(),
                            "lanai-asm-printer-end");
  }
}

Error LanaiTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = LanaiCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
