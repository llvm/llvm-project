//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VE.h"
#include "VETargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class VECodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  VETargetMachine &getTM() const { return static_cast<VETargetMachine &>(TM); }

public:
  explicit VECodeGenPassBuilder(VETargetMachine &TM,
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

void VECodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  addFunctionPass(AtomicExpandPass(getTM()), PMW);

  Base::addIRPasses(PMW);
}

Error VECodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(VEDAGToDAGISelPass(getTM()), PMW);
  return Error::success();
}

void VECodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  // LVLGen should be called after scheduling and register allocation
  // TODO(boomanaiden154): Add VELVLGenPass
}

void VECodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add VEAsmPrinterBeginPass
}

void VECodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add VEAsmPrinterPass
}

void VECodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add VEAsmPrinterEndPass
}

} // namespace

void VETargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "VEPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error VETargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = VECodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
