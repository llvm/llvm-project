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
#include "llvm/CodeGen/CFIInstrInserter.h"
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

class RISCVCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  RISCVTargetMachine &getTM() const {
    return static_cast<RISCVTargetMachine &>(TM);
  }

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

  void addIRPasses(PassManagerWrapper &PMW) override;
  void addCodeGenPrepare(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  void addPostRegAlloc(PassManagerWrapper &PMW) override;
  void addPreSched2(PassManagerWrapper &PMW) override;
  void addPreEmitPass(PassManagerWrapper &PMW) override;
  void addPreEmitPass2(PassManagerWrapper &PMW) override;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;
};

void RISCVCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  addFunctionPass(AtomicExpandPass(TM), PMW);
  addFunctionPass(RISCVZacasABIFixPass(&getTM()), PMW);

  if (getOptLevel() != CodeGenOptLevel::None) {
    addFunctionPass(LoopDataPrefetchPass(), PMW);

    addFunctionPass(RISCVGatherScatterLoweringPass(&getTM()), PMW);
    addFunctionPass(InterleavedAccessPass(TM), PMW);
    addFunctionPass(RISCVCodeGenPreparePass(&getTM()), PMW);
  }

  Base::addIRPasses(PMW);

  // TODO: SelectOptimizePass is already added by the base class when
  // !Opt.DisableSelectOptimize. The legacy pipeline additionally gates this
  // on -riscv-select-opt and only runs it at -O3; that extra gating is not
  // yet replicated here.
}

void RISCVCodeGenPassBuilder::addCodeGenPrepare(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(TypePromotionPass(TM), PMW);
  Base::addCodeGenPrepare(PMW);
}

Error RISCVCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVISelDAGToDAGPass(getTM(), getOptLevel()), PMW);
  return Error::success();
}

void RISCVCodeGenPassBuilder::addMachineSSAOptimization(
    PassManagerWrapper &PMW) {
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
    addMachineFunctionPass(RISCVVLOptimizerPass(), PMW);
  }

  addMachineFunctionPass(RISCVVectorPeepholePass(), PMW);
  addMachineFunctionPass(RISCVFoldMemOffsetPass(), PMW);

  Base::addMachineSSAOptimization(PMW);

  if (TM.getTargetTriple().isRISCV64())
    addMachineFunctionPass(RISCVOptWInstrsPass(), PMW);
}

void RISCVCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVPreRAExpandPseudoPass(), PMW);
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

void RISCVCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVRedundantCopyEliminationPass
  }
}

void RISCVCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVPostRAExpandPseudoPass(), PMW);

  addMachineFunctionPass(MachineKCFIPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO: RISCVLoadStoreOptPass
  }
}

void RISCVCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
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

void RISCVCodeGenPassBuilder::addPreEmitPass2(PassManagerWrapper &PMW) {
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

  // RISCVTargetMachine's constructor sets Options.EnableCFIFixup to the
  // inverse of -riscv-enable-cfi-instr-inserter (a flag private to
  // RISCVTargetMachine.cpp), so checking it here is equivalent to checking
  // that flag directly -- the two passes solve overlapping problems and
  // this target picks exactly one.
  if (!TM.Options.EnableCFIFixup)
    addMachineFunctionPass(CFIInstrInserterPass(), PMW);
}

void RISCVCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  addModulePass(RISCVAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void RISCVCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVAsmPrinterPass(), PMW);
}

void RISCVCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
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
}

Error RISCVTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = RISCVCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
