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

class MSP430CodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  MSP430TargetMachine &getTM() const {
    return static_cast<MSP430TargetMachine &>(TM);
  }

public:
  explicit MSP430CodeGenPassBuilder(MSP430TargetMachine &TM,
                                    const CGPassBuilderOption &Opts,
                                    PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

void MSP430CodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  addFunctionPass(AtomicExpandPass(TM), PMW);

  Base::addIRPasses(PMW);
}

Error MSP430CodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(MSP430ISelDAGToDAGPass(getTM(), getOptLevel()), PMW);
  return Error::success();
}

void MSP430CodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  addMachineFunctionPass(MSP430BranchSelectPass(), PMW);
}

void MSP430CodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  addModulePass(MSP430AsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void MSP430CodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  addMachineFunctionPass(MSP430AsmPrinterPass(), PMW);
}

void MSP430CodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  addModulePass(MSP430AsmPrinterEndPass(), PMW);
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
