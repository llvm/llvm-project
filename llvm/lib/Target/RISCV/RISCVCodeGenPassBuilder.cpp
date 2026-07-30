//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the RISC-V CodeGen pipeline builder.
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVAsmPrinter.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BranchRelaxation.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/KCFI.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineLICM.h"
#include "llvm/CodeGen/TypePromotion.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Vectorize/LoopIdiomVectorize.h"

using namespace llvm;

namespace {

class RISCVCodeGenPassBuilder
    : public CodeGenPassBuilder<RISCVCodeGenPassBuilder, RISCVTargetMachine> {
  using Base = CodeGenPassBuilder<RISCVCodeGenPassBuilder, RISCVTargetMachine>;

public:
  explicit RISCVCodeGenPassBuilder(RISCVTargetMachine &TM,
                                   const CGPassBuilderOption &Opts,
                                   PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {
    // TODO: See the FIXME on RISCVPassConfig::setEnableSinkAndFold in the
    // legacy pass manager pipeline. There is currently no way to plumb
    // -riscv-enable-sink-fold through to CGPassBuilderOption::EnableSinkAndFold
    // from a target-local CodeGenPassBuilder, so this NewPM pipeline always
    // uses the base class default (disabled).
  }

  void addIRPasses(PassManagerWrapper &PMW) const;
  void addCodeGenPrepare(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) const;
  void addPreRegAlloc(PassManagerWrapper &PMW) const;
  void addPostRegAlloc(PassManagerWrapper &PMW) const;
  void addPreSched2(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMW) const;
  void addPreEmitPass2(PassManagerWrapper &PMW) const;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) const;
  void addAsmPrinter(PassManagerWrapper &PMW) const;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) const;
};

void RISCVCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  addFunctionPass(AtomicExpandPass(TM), PMW);
  // TODO: RISCVZacasABIFixPass

  if (getOptLevel() != CodeGenOptLevel::None) {
    addFunctionPass(LoopDataPrefetchPass(), PMW);

    // TODO: RISCVGatherScatterLoweringPass
    addFunctionPass(InterleavedAccessPass(TM), PMW);
    addFunctionPass(RISCVCodeGenPreparePass(&TM), PMW);
  }

  Base::addIRPasses(PMW);

  // TODO: SelectOptimizePass is already added by the base class when
  // !Opt.DisableSelectOptimize. The legacy pipeline additionally gates this
  // on -riscv-select-opt and only runs it at -O3; that extra gating is not
  // yet replicated here.
}

void RISCVCodeGenPassBuilder::addCodeGenPrepare(PassManagerWrapper &PMW) const {
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(TypePromotionPass(TM), PMW);
  Base::addCodeGenPrepare(PMW);
}

Error RISCVCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(RISCVISelDAGToDAGPass(TM, getOptLevel()), PMW);
  return Error::success();
}

void RISCVCodeGenPassBuilder::addMachineSSAOptimization(
    PassManagerWrapper &PMW) const {
  // It's beneficial to reduce the VL to enable more
  // Machine SSA optimizations.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // RISCVVLOptimizer can make loop invariant instructions like vmv.v.i
    // loop variant by propagating a VL defined inside the loop. Run LICM and
    // hoist them early. Don't do this at -O0 to avoid the compile-time
    // overhead. Not reducing the VL of loop invariant pseudos results in more
    // vsetvli toggles, and still requires the MachineLoopInfo analysis to be
    // run.
    addMachineFunctionPass(EarlyMachineLICMPass(), PMW);
    // TODO: RISCVVLOptimizerPass
  }

  // TODO: RISCVVectorPeepholePass
  // TODO: RISCVFoldMemOffsetPass

  Base::addMachineSSAOptimization(PMW);

  if (TM.getTargetTriple().isRISCV64()) {
    // TODO: RISCVOptWInstrsPass
  }
}

void RISCVCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) const {
  // TODO: RISCVPreRAExpandPseudoPass
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVMergeBaseOffsetOptPass
    // TODO: RISCVPreAllocZilsdOptPass
  }

  // TODO: RISCVInsertReadWriteCSRPass
  // TODO: RISCVInsertWriteVXRMPass
  // TODO: RISCVLandingPadSetupPass

  // TODO: MachinePipelinerPass (no new pass manager port exists yet)

  // TODO: RISCVVMV0EliminationPass
}

void RISCVCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) const {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVRedundantCopyEliminationPass
  }
}

void RISCVCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) const {
  // TODO: RISCVPostRAExpandPseudoPass

  addMachineFunctionPass(MachineKCFIPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVLoadStoreOptPass
  }
}

void RISCVCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) const {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (getOptLevel() >= CodeGenOptLevel::Default) {
    addMachineFunctionPass(MachineCopyPropagationPass(true), PMW);
    // TODO: RISCVLateBranchOptPass
  }
  // The IndirectBranchTrackingPass inserts lpad and could have changed the
  // basic block alignment. It must be done before Branch Relaxation to
  // prevent the adjusted offset exceeding the branch range.
  // TODO: RISCVIndirectBranchTrackingPass
  addMachineFunctionPass(BranchRelaxationPass(), PMW);
  // TODO: RISCVMakeCompressibleOptPass
}

void RISCVCodeGenPassBuilder::addPreEmitPass2(PassManagerWrapper &PMW) const {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVMoveMergePass
    // TODO: RISCVPushPopOptimizationPass
  }
  // TODO: RISCVExpandPseudoPass

  // Add QC Relaxation Markers as late as possible, and only for RV32
  if (getOptLevel() != CodeGenOptLevel::None &&
      TM.getTargetTriple().isRISCV32()) {
    // TODO: RISCVQCRelaxMarkingPass
  }

  // TODO: RISCVExpandAtomicPseudoPass

  // KCFI indirect call checks are lowered to a bundle.
  addMachineFunctionPass(
      UnpackMachineBundlesPass([&](const MachineFunction &MF) {
        return MF.getFunction().getParent()->getModuleFlag("kcfi");
      }),
      PMW);

  // TODO: CFIInstrInserterPass
}

void RISCVCodeGenPassBuilder::addAsmPrinterBegin(
    PassManagerWrapper &PMW) const {
  addModulePass(RISCVAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void RISCVCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) const {
  addMachineFunctionPass(RISCVAsmPrinterPass(), PMW);
}

void RISCVCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) const {
  addModulePass(RISCVAsmPrinterEndPass(), PMW, /*Force=*/true);
}

} // namespace

void RISCVTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "RISCVPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"

  PB.registerLateLoopOptimizationsEPCallback([=](LoopPassManager &LPM,
                                                 OptimizationLevel Level) {
    if (Level != OptimizationLevel::O0)
      LPM.addPass(LoopIdiomVectorizePass(LoopIdiomVectorizeStyle::Predicated));
  });

  if (PIC) {
    PIC->addClassToPassName(RISCVAsmPrinterBeginPass::name(),
                            "riscv-asm-printer-begin");
    PIC->addClassToPassName(RISCVAsmPrinterPass::name(), "riscv-asm-printer");
    PIC->addClassToPassName(RISCVAsmPrinterEndPass::name(),
                            "riscv-asm-printer-end");
  }
}

Error RISCVTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = RISCVCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
