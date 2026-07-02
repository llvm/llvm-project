//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
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

  void addIRPasses(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMW) const;
};

void MSP430CodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  addFunctionPass(AtomicExpandPass(TM), PMW);

  Base::addIRPasses(PMW);
}

Error MSP430CodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(MSP430ISelDAGToDAGPass(TM, getOptLevel()), PMW);
  return Error::success();
}

void MSP430CodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  // TODO(boomanaiden154): Add MSP430BranchSelectionPass when it has been
  // ported.
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
