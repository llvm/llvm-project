//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  // TODO(boomanaiden154): Add LanaiMemAluCombiner when it has been ported.
}

void LanaiCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  // TODO(boomanaiden154): Add LanaiDelaySlotFiller when it has been ported.
}

} // namespace

void LanaiTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "LanaiPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error LanaiTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = LanaiCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
