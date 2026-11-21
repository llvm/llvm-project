//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoongArch.h"
#include "LoongArchTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BranchRelaxation.h"
#include "llvm/CodeGen/MachineCombiner.h"
#include "llvm/CodeGen/TypePromotion.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"

using namespace llvm;

extern cl::opt<bool> EnableLoopDataPrefetch;
extern cl::opt<bool> EnableMachineCombiner;
extern cl::opt<bool> EnableMergeBaseOffset;
extern cl::opt<bool> EnableLoongArchDeadRegisterElimination;

namespace {

class LoongArchCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  LoongArchTargetMachine &getTM() const {
    return static_cast<LoongArchTargetMachine &>(TM);
  }

public:
  explicit LoongArchCodeGenPassBuilder(LoongArchTargetMachine &TM,
                                       const CGPassBuilderOption &Opts,
                                       PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

protected:
  void addIRPasses(PassManagerWrapper &PMW) override;
  void addCodeGenPrepare(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) override;
  void addILPOpts(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  Error addRegAssignAndRewriteFast(PassManagerWrapper &PMW) override;
  Expected<bool>
  addRegAssignAndRewriteOptimized(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addPreEmitPass2(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

} // namespace

void LoongArchCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  // Run LoopDataPrefetch
  //
  // Run this before LSR to remove the multiplies involved in computing the
  // pointer values N iterations ahead.
  if (getOptLevel() != CodeGenOptLevel::None && EnableLoopDataPrefetch)
    addFunctionPass(LoopDataPrefetchPass(), PMW);
  addFunctionPass(AtomicExpandPass(getTM()), PMW);

  Base::addIRPasses(PMW);
}

void LoongArchCodeGenPassBuilder::addCodeGenPrepare(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(TypePromotionPass(getTM()), PMW);

  Base::addCodeGenPrepare(PMW);
}

Error LoongArchCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(LoongArchISelDAGToDAGPass(getTM(), getOptLevel()),
                         PMW);

  return Error::success();
}

void LoongArchCodeGenPassBuilder::addMachineSSAOptimization(
    PassManagerWrapper &PMW) {
  Base::addMachineSSAOptimization(PMW);

  if (getTM().getTargetTriple().isLoongArch64()) {
    // TODO(boomanaiden154): Add LoongArchOptWInstrsPass
  }
}

void LoongArchCodeGenPassBuilder::addILPOpts(PassManagerWrapper &PMW) {
  if (EnableMachineCombiner)
    addMachineFunctionPass(MachineCombinerPass(), PMW);
}

void LoongArchCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add LoongArchPreRAExpandPseudoPass
  if (getOptLevel() != CodeGenOptLevel::None && EnableMergeBaseOffset) {
    // TODO(boomanaiden154): Add LoongArchMergeBaseOffsetOptPass
  }
}

Error LoongArchCodeGenPassBuilder::addRegAssignAndRewriteFast(
    PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None &&
      EnableLoongArchDeadRegisterElimination) {
    // TODO(boomanaiden154): Add LoongArchDeadRegisterDefinitionsPass
  }

  return Base::addRegAssignAndRewriteFast(PMW);
}

Expected<bool> LoongArchCodeGenPassBuilder::addRegAssignAndRewriteOptimized(
    PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None &&
      EnableLoongArchDeadRegisterElimination) {
    // TODO(boomanaiden154): Add LoongArchDeadRegisterDefinitionsPass
  }

  return Base::addRegAssignAndRewriteOptimized(PMW);
}

void LoongArchCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  addMachineFunctionPass(BranchRelaxationPass(), PMW);
}

void LoongArchCodeGenPassBuilder::addPreEmitPass2(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add LoongArchExpandPseudosPass
  // Schedule the expansion of AtomicPseudos at the last possible moment,
  // avoiding the possibility for other passes to break the requirements for
  // forward progress in the LL/SC block.
  // TODO(boomanaiden154): Add LoongArchExpandAtomicPseudoPass
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add LoongArchMemoryBarrierOptPass
  }
}

void LoongArchCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add LoongArchAsmPrinterBeginPass
}

void LoongArchCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add LoongArchAsmPrinterPass
}

void LoongArchCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add LoongArchAsmPrinterEndPass
}

void LoongArchTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "LoongArchPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error LoongArchTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = LoongArchCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
