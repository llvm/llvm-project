//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AVR.h"
#include "AVRTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BranchRelaxation.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"
using namespace llvm;

namespace {

class AVRCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  AVRTargetMachine &getTM() const {
    return static_cast<AVRTargetMachine &>(TM);
  }

public:
  explicit AVRCodeGenPassBuilder(AVRTargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addPreSched2(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

void AVRCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  // Expand instructions like
  //   %result = shl i32 %n, %amount
  // to a loop so that library calls are avoided.
  addFunctionPass(AVRShiftExpandPass(), PMW);
  addFunctionPass(AtomicExpandPass(getTM()), PMW);

  Base::addIRPasses(PMW);
}

Error AVRCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  // Install an instruction selector.
  addMachineFunctionPass(AVRISelDAGToDAGPass(getTM(), getOptLevel()), PMW);
  // Create the frame analyzer pass used by the PEI pass.
  // TODO(boomanaiden154): Add AVRFrameAnalyzer
  return Error::success();
}

void AVRCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add AVRExpandPseudos
}

void AVRCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  // Must run branch selection immediately preceding the asm printer.
  addMachineFunctionPass(BranchRelaxationPass(), PMW);
}

void AVRCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add AVRAsmPrinterBegin
}

void AVRCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add AVRAsmPrinter
}

void AVRCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add AVRAsmPrinterEnd
}

} // namespace

void AVRTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "AVRPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error AVRTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = AVRCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
