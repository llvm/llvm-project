//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class MipsCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  MipsTargetMachine &getTM() const {
    return static_cast<MipsTargetMachine &>(TM);
  }

public:
  explicit MipsCodeGenPassBuilder(MipsTargetMachine &TM,
                                  const CGPassBuilderOption &Opts,
                                  PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

} // namespace

void MipsCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  Base::addIRPasses(PMW);
  addFunctionPass(AtomicExpandPass(getTM()), PMW);
  flushFPMsToMPM(PMW);
  if (getTM().getCLOpts().os16) {
    addModulePass(MipsOs16Pass(), PMW);
  }
  addModulePass(Mips16HardFloatPass(getTM()), PMW);
}

Error MipsCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(Mips16ISelDAGPass(getTM(), getOptLevel()), PMW);
  addMachineFunctionPass(MipsSEISelDAGPass(getTM(), getOptLevel()), PMW);
  return Error::success();
}

void MipsCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(MipsOptimizePICCallPass(), PMW);
}

void MipsCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  // Expand pseudo instructions that are sensitive to register allocation.
  addMachineFunctionPass(MipsExpandPseudoPass(), PMW);

  // The microMIPS size reduction pass performs instruction reselection for
  // instructions which can be remapped to a 16 bit instruction.
  addMachineFunctionPass(MipsMicroSizeReducePass(), PMW);

  // This pass inserts a nop instruction between two back-to-back multiplication
  // instructions when the "mfix4300" flag is passed.
  if (getTM().getCLOpts().mfix4300)
    addMachineFunctionPass(MipsMulMulBugFixPass(), PMW);

  // The delay slot filler pass can potientially create forbidden slot hazards
  // for MIPSR6 and therefore it should go before MipsBranchExpansion pass.
  addMachineFunctionPass(MipsDelaySlotFillerPass(), PMW);

  // This pass expands branches and takes care about the forbidden slot hazards.
  // Expanding branches may potentially create forbidden slot hazards for
  // MIPSR6, and fixing such hazard may potentially break a branch by extending
  // its offset out of range. That's why this pass combine these two tasks, and
  // runs them alternately until one of them finishes without any changes. Only
  // then we can be sure that all branches are expanded properly and no hazards
  // exists.
  // Any new pass should go before this pass.
  addMachineFunctionPass(MipsBranchExpansionPass(), PMW);

  // TODO(boomanaiden154): Add MipsConstantIslandPass
}

void MipsCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add MipsAsmPrinterBeginPass
}

void MipsCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add MipsAsmPrinterPass
}

void MipsCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add MipsAsmPrinterEndPass
}

void MipsTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "MipsPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error MipsTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = MipsCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
