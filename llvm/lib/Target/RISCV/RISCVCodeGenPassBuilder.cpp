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
#include "llvm/CodeGen/GlobalMerge.h"
#include "llvm/CodeGen/InitUndef.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/KCFI.h"
#include "llvm/CodeGen/MachineCombiner.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineLICM.h"
#include "llvm/CodeGen/RegAllocFast.h"
#include "llvm/CodeGen/RegAllocGreedyPass.h"
#include "llvm/CodeGen/SelectOptimize.h"
#include "llvm/CodeGen/TypePromotion.h"
#include "llvm/CodeGen/VirtRegMap.h"
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
  void addPreISel(PassManagerWrapper &PMW) override;
  void addCodeGenPrepare(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addMachineSSAOptimization(PassManagerWrapper &PMW) override;
  void addILPOpts(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  Error addFastRegAlloc(PassManagerWrapper &PMW) override;
  Error addRegAssignAndRewriteFast(PassManagerWrapper &PMW) override;
  Expected<bool>
  addRegAssignAndRewriteOptimized(PassManagerWrapper &PMW) override;
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
    if (getTM().getCLOpts().enable_loop_data_prefetch)
      addFunctionPass(LoopDataPrefetchPass(), PMW);

    addFunctionPass(RISCVGatherScatterLoweringPass(&getTM()), PMW);
    addFunctionPass(InterleavedAccessPass(TM), PMW);
    addFunctionPass(RISCVCodeGenPreparePass(&getTM()), PMW);
  }

  Base::addIRPasses(PMW);

  if (getOptLevel() == CodeGenOptLevel::Aggressive &&
      getTM().getCLOpts().select_opt)
    addFunctionPass(SelectOptimizePass(getTM()), PMW);
}

void RISCVCodeGenPassBuilder::addPreISel(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None) {
    flushFPMsToMPM(PMW);
    addModulePass(RISCVPromoteConstantPass(&getTM()), PMW);
  }

  if (valueOr(getTM().getCLOpts().enable_global_merge,
              getOptLevel() != CodeGenOptLevel::None)) {
    // FIXME: Like AArch64, we disable extern global merging by default due to
    // concerns it might regress some workloads. Unlike AArch64, we don't
    // currently support enabling the pass in an "OnlyOptimizeForSize" mode.
    // Investigating and addressing both items are TODO.
    flushFPMsToMPM(PMW);
    GlobalMergeOptions GMOptions;
    GMOptions.MaxOffset = 2047;
    GMOptions.MergeExternal = true;
    GMOptions.SizeOnly = false;
    addModulePass(GlobalMergePass(&getTM(), GMOptions), PMW);
  }
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

void RISCVCodeGenPassBuilder::addILPOpts(PassManagerWrapper &PMW) {
  if (getTM().getCLOpts().enable_machine_combiner)
    addMachineFunctionPass(MachineCombinerPass(), PMW);
}

void RISCVCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVExpandPseudoPreRAPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineFunctionPass(RISCVMergeBaseOffsetPass(), PMW);
    // Add Zilsd pre-allocation load/store optimization
    addMachineFunctionPass(RISCVPreAllocZilsdOptPass(), PMW);
  }

  addMachineFunctionPass(RISCVInsertReadWriteCSRPass(), PMW);
  addMachineFunctionPass(RISCVInsertWriteVXRMPass(), PMW);
  addMachineFunctionPass(RISCVLandingPadSetupPass(), PMW);
  addMachineFunctionPass(RISCVVMV0EliminationPass(), PMW);
}

Error RISCVCodeGenPassBuilder::addFastRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(InitUndefPass(), PMW);
  return Base::addFastRegAlloc(PMW);
}

static bool onlyAllocateRVVReg(const TargetRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI,
                               const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return RISCVRegisterInfo::isRVVRegClass(RC);
}

Error RISCVCodeGenPassBuilder::addRegAssignAndRewriteFast(
    PassManagerWrapper &PMW) {
  addMachineFunctionPass(
      RegAllocFastPass({onlyAllocateRVVReg, "rvv", /*ClearVRegs=*/false}), PMW);
  addMachineFunctionPass(RISCVInsertVSETVLIPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None &&
      getTM().getCLOpts().enable_dead_defs)
    addMachineFunctionPass(RISCVDeadRegisterDefinitionsPass(), PMW);
  return Base::addRegAssignAndRewriteFast(PMW);
}

Expected<bool> RISCVCodeGenPassBuilder::addRegAssignAndRewriteOptimized(
    PassManagerWrapper &PMW) {
  addMachineFunctionPass(RAGreedyPass({onlyAllocateRVVReg, "rvv"}), PMW);
  addMachineFunctionPass(VirtRegRewriterPass(false), PMW);
  addMachineFunctionPass(RISCVInsertVSETVLIPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None &&
      getTM().getCLOpts().enable_dead_defs)
    addMachineFunctionPass(RISCVDeadRegisterDefinitionsPass(), PMW);
  return Base::addRegAssignAndRewriteOptimized(PMW);
}

void RISCVCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None &&
      getTM().getCLOpts().enable_copyelim)
    addMachineFunctionPass(RISCVRedundantCopyEliminationPass(), PMW);
}

void RISCVCodeGenPassBuilder::addPreSched2(PassManagerWrapper &PMW) {
  addMachineFunctionPass(RISCVExpandPseudoPostRAPass(), PMW);

  addMachineFunctionPass(MachineKCFIPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineFunctionPass(RISCVLoadStoreOptPass(), PMW);
}

void RISCVCodeGenPassBuilder::addPreEmitPass(PassManagerWrapper &PMW) {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (getOptLevel() >= CodeGenOptLevel::Default &&
      getTM().getCLOpts().enable_copy_propagation)
    addMachineFunctionPass(MachineCopyPropagationPass(true), PMW);
  if (getOptLevel() >= CodeGenOptLevel::Default)
    addMachineFunctionPass(RISCVLateBranchOptPass(), PMW);
  // The IndirectBranchTrackingPass inserts lpad and could have changed the
  // basic block alignment. It must be done before Branch Relaxation to
  // prevent the adjusted offset exceeding the branch range.
  addMachineFunctionPass(RISCVIndirectBranchTrackingPass(), PMW);
  addMachineFunctionPass(BranchRelaxationPass(), PMW);
  addMachineFunctionPass(RISCVMakeCompressiblePass(), PMW);
}

void RISCVCodeGenPassBuilder::addPreEmitPass2(PassManagerWrapper &PMW) {
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineFunctionPass(RISCVMoveMergePass(), PMW);
    // Schedule PushPop Optimization before expansion of Pseudo instruction,
    // ensuring return instruction is detected correctly.
    addMachineFunctionPass(RISCVPushPopOptPass(), PMW);
  }
  addMachineFunctionPass(RISCVExpandPseudoPreEmitPass(), PMW);

  // Add QC Relaxation Markers as late as possible, and only for RV32
  if (getOptLevel() != CodeGenOptLevel::None &&
      TM.getTargetTriple().isRISCV32()) {
    addMachineFunctionPass(RISCVQCRelaxMarkingPass(), PMW);
  }

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addMachineFunctionPass(RISCVExpandPseudoAtomicsPass(), PMW);

  // KCFI indirect call checks are lowered to a bundle.
  addMachineFunctionPass(
      UnpackMachineBundlesPass([&](const MachineFunction &MF) {
        return MF.getFunction().getParent()->getModuleFlag("kcfi");
      }),
      PMW);

  // RISCVTargetMachine's constructor sets Options.enable_cfi_instr_inserter to
  // the inverse of -riscv-enable-cfi-instr-inserter (a flag private to
  // RISCVTargetMachine.cpp), so checking it here is equivalent to checking
  // that flag directly -- the two passes solve overlapping problems and
  // this target picks exactly one.
  if (getTM().getCLOpts().enable_cfi_instr_inserter)
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
