//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sparc.h"
#include "SparcTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BranchRelaxation.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"

using namespace llvm;

namespace {

class SparcCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  SparcTargetMachine &getTM() const {
    return static_cast<SparcTargetMachine &>(TM);
  }

public:
  explicit SparcCodeGenPassBuilder(SparcTargetMachine &TM,
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

void SparcCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  addFunctionPass(AtomicExpandPass(getTM()), PMW);

  Base::addIRPasses(PMW);
}

Error SparcCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(SparcDAGToDAGISelPass(getTM(), getOptLevel()), PMW);
  return Error::success();
}

void SparcCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  if (getTM().getCLOpts().enable_branch_relax)
    addMachineFunctionPass(BranchRelaxationPass(), PMW);

  // TODO(boomanaiden154): Add SparcDelaySlotFillerPass
  // TODO(boomanaiden154): Add SparcInsertNOPLoadPass
  // TODO(boomanaiden154): Add SparcDetectRoundChangePass
  // TODO(boomanaiden154): Add SparcFixAllFDIVSQRTPass
  // TODO(boomanaiden154): Add SparcErrataWorkaroundPass
}

void SparcCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SparcAsmPrinterBegin
}

void SparcCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SparcAsmPrinter
}

void SparcCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SparcAsmPrinterEnd
}

} // namespace

void SparcTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "SparcPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error SparcTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = SparcCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
