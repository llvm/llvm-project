//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFSubtarget.h"
#include "BPFTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"

using namespace llvm;

extern cl::opt<bool> DisableMIPeephole;

namespace {

class BPFCodeGenPassBuilder
    : public CodeGenPassBuilder<BPFCodeGenPassBuilder, BPFTargetMachine> {
  using Base = CodeGenPassBuilder<BPFCodeGenPassBuilder, BPFTargetMachine>;

public:
  explicit BPFCodeGenPassBuilder(BPFTargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMW) const;
};

void BPFCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  addFunctionPass(AtomicExpandPass(TM), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(BPFCheckAndAdjustIRPass(), PMW);

  Base::addIRPasses(PMW);
}

Error BPFCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(BPFISelDAGToDAGPass(TM), PMW);
  return Error::success();
}

void BPFCodeGenPassBuilder::addMachineSSAOptimization(
    PassManagerWrapper &PMW) const {
  // TODO(boomanaiden154): Add BPFMISimplifyPatchablePass when it has been
  // ported.

  Base::addMachineSSAOptimization(PMW);

  const BPFSubtarget *Subtarget = TM.getSubtargetImpl();
  if (!DisableMIPeephole) {
    if (Subtarget->getHasAlu32()) {
      // TODO(boomanaiden154): Add BPFMIPeepholePass when it has been ported.
    }
  }
}

void BPFCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  // TODO(boomanaiden154): Add BPFMIPreEmitCheckingPass when it has been ported.
  if (!DisableMIPeephole) {
    // TODO(boomanaiden154): Add BPFMIPreEmitPeepholePass when it has been
    // ported.
  }
}

} // namespace

static Expected<bool> parseBPFPreserveStaticOffsetOptions(StringRef Params) {
  return PassBuilder::parseSinglePassOption(Params, "allow-partial",
                                            "BPFPreserveStaticOffsetPass");
}

void BPFTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "BPFPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"

  PB.registerPipelineStartEPCallback(
      [=](ModulePassManager &MPM, OptimizationLevel) {
        FunctionPassManager FPM;
        FPM.addPass(BPFPreserveStaticOffsetPass(true));
        FPM.addPass(BPFAbstractMemberAccessPass(this));
        FPM.addPass(BPFPreserveDITypePass());
        FPM.addPass(BPFIRPeepholePass());
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
  PB.registerPeepholeEPCallback([=](FunctionPassManager &FPM,
                                    OptimizationLevel Level) {
    FPM.addPass(SimplifyCFGPass(SimplifyCFGOptions().hoistCommonInsts(true)));
    FPM.addPass(BPFASpaceCastSimplifyPass());
  });
  PB.registerScalarOptimizerLateEPCallback(
      [=](FunctionPassManager &FPM, OptimizationLevel Level) {
        // Run this after loop unrolling but before
        // SimplifyCFGPass(... .sinkCommonInsts(true))
        FPM.addPass(BPFPreserveStaticOffsetPass(false));
      });
  PB.registerPipelineEarlySimplificationEPCallback(
      [=](ModulePassManager &MPM, OptimizationLevel, ThinOrFullLTOPhase) {
        MPM.addPass(BPFAdjustOptPass());
      });
}

Error BPFTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = BPFCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
