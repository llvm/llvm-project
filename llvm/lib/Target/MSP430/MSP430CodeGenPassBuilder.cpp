//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
#include "MSP430AsmPrinter.h"
#include "MSP430TargetMachine.h"

#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"

using namespace llvm;

namespace {

class MSP430CodeGenPassBuilder
    : public CodeGenPassBuilder<MSP430CodeGenPassBuilder, MSP430TargetMachine> {
  using Base =
      CodeGenPassBuilder<MSP430CodeGenPassBuilder, MSP430TargetMachine>;

public:
  explicit MSP430CodeGenPassBuilder(MSP430TargetMachine &TM,
                                    const CGPassBuilderOption &Opts,
                                    PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(CodeGenModulePassManager &CGMPM) const;
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterBegin(CodeGenModulePassManager &CGMPM) const;
  void addAsmPrinter(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterEnd(CodeGenModulePassManager &CGMPM) const;
};

void MSP430CodeGenPassBuilder::addIRPasses(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addPass(AtomicExpandPass(TM));
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  Base::addIRPasses(CGMPM);
}

Error MSP430CodeGenPassBuilder::addInstSelector(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(MSP430ISelDAGToDAGPass(TM, getOptLevel()));
  return Error::success();
}

void MSP430CodeGenPassBuilder::addPreEmitPass(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(MSP430BranchSelectPass());
}

void MSP430CodeGenPassBuilder::addAsmPrinterBegin(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(MSP430AsmPrinterBeginPass());
}

void MSP430CodeGenPassBuilder::addAsmPrinter(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(MSP430AsmPrinterPass());
}

void MSP430CodeGenPassBuilder::addAsmPrinterEnd(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(MSP430AsmPrinterEndPass());
}

} // namespace

void MSP430TargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "MSP430PassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error MSP430TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = MSP430CodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
