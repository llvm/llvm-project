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

  void addIRPasses(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  void addPreSched2(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMW) const;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) const;
  void addAsmPrinter(PassManagerWrapper &PMW) const;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) const;
};

void LanaiCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  addFunctionPass(AtomicExpandPass(TM), PMW);

  Base::addIRPasses(PMW);
}

Error LanaiCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(LanaiISelDAGToDAGPass(TM), PMW);
  return Error::success();
}

void LanaiCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(LanaiMemAluCombinerPass(), PMW);
}

void LanaiCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(LanaiDelaySlotFillerPass(), PMW);
}

void LanaiCodeGenPassBuilder::addAsmPrinterBegin(
    PassManagerWrapper &PMW) const {
  addModulePass(LanaiAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void LanaiCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(LanaiAsmPrinterPass(), PMW);
}

void LanaiCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) const {
  addModulePass(LanaiAsmPrinterEndPass(), PMW, /*Force=*/true);
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
