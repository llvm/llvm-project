//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFAsmPrinter.h"
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

  void addIRPasses(CodeGenModulePassManager &CGMPM) const;
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void
  addMachineSSAOptimization(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterBegin(CodeGenModulePassManager &CGMPM) const;
  void addAsmPrinter(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterEnd(CodeGenModulePassManager &CGMPM) const;
};

void BPFCodeGenPassBuilder::addIRPasses(CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addPass(AtomicExpandPass(TM));
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  CGMPM.addPass(BPFCheckAndAdjustIRPass());

  Base::addIRPasses(CGMPM);
}

Error BPFCodeGenPassBuilder::addInstSelector(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(BPFISelDAGToDAGPass(TM));
  return Error::success();
}

void BPFCodeGenPassBuilder::addMachineSSAOptimization(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(BPFMISimplifyPatchablePass());

  Base::addMachineSSAOptimization(CGMFPM);

  const BPFSubtarget *Subtarget = TM.getSubtargetImpl();
  if (!DisableMIPeephole) {
    if (Subtarget->getHasAlu32())
      CGMFPM.addPass(BPFMIPeepholePass());
  }
}

void BPFCodeGenPassBuilder::addPreEmitPass(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(BPFMIPreEmitCheckingPass());
  if (!DisableMIPeephole) {
    CGMFPM.addPass(BPFMIExpandStackArgPseudosPass());
    CGMFPM.addPass(BPFMIPreEmitPeepholePass());
  }
}

void BPFCodeGenPassBuilder::addAsmPrinterBegin(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(BPFAsmPrinterBeginPass());
}

void BPFCodeGenPassBuilder::addAsmPrinter(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(BPFAsmPrinterPass());
}

void BPFCodeGenPassBuilder::addAsmPrinterEnd(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(BPFAsmPrinterEndPass());
}

} // namespace

static Expected<bool> parseBPFPreserveStaticOffsetOptions(StringRef Params) {
  return PassBuilder::parseSinglePassOption(Params, "allow-partial",
                                            "BPFPreserveStaticOffsetPass");
}

void BPFTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "BPFPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
  // TODO(boomanaiden154): Move this into the base CodeGenPassBuilder once all
  // targets that currently implement it have a ported asm-printer pass.
  if (PIC) {
    PIC->addClassToPassName(BPFAsmPrinterBeginPass::name(),
                            "bpf-asm-printer-begin");
    PIC->addClassToPassName(BPFAsmPrinterPass::name(), "bpf-asmprinter");
    PIC->addClassToPassName(BPFAsmPrinterEndPass::name(),
                            "bpf-asm-printer-end");
  }

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
