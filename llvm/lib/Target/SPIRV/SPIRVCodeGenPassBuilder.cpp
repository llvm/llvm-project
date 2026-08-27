//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVConvergenceRegionAnalysis.h"
#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BranchFoldingPass.h"
#include "llvm/CodeGen/FuncletLayout.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/LiveDebugValuesPass.h"
#include "llvm/CodeGen/MachineBlockPlacement.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineLateInstrsCleanup.h"
#include "llvm/CodeGen/PatchableFunction.h"
#include "llvm/CodeGen/PostRAMachineSink.h"
#include "llvm/CodeGen/PostRASchedulerList.h"
#include "llvm/CodeGen/RemoveLoadsIntoFakeUses.h"
#include "llvm/CodeGen/ShrinkWrap.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/StripConvergenceIntrinsics.h"

using namespace llvm;

namespace {

class SPIRVCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  SPIRVTargetMachine &getTM() const {
    return static_cast<SPIRVTargetMachine &>(TM);
  }

public:
  explicit SPIRVCodeGenPassBuilder(SPIRVTargetMachine &TM,
                                   const CGPassBuilderOption &Opts,
                                   PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {
    // Disable passes that break from assuming no virtual registers exist.
    disablePass<MachineCopyPropagationPass, PostRAMachineSinkingPass,
                PostRASchedulerPass, FuncletLayoutPass, StackMapLivenessPass,
                PatchableFunctionPass, ShrinkWrapPass, LiveDebugValuesPass,
                MachineLateInstrsCleanupPass, RemoveLoadsIntoFakeUsesPass,
                BranchFolderPass, MachineBlockPlacementPass>();
  }

  void addIRPasses(PassManagerWrapper &PMW) override;
  void addISelPrepare(PassManagerWrapper &PMW) override;
  Error addIRTranslator(PassManagerWrapper &PMW) override;
  void addPreLegalizeMachineIR(PassManagerWrapper &PMW) override;
  Error addLegalizeMachineIR(PassManagerWrapper &PMW) override;
  Error addRegBankSelect(PassManagerWrapper &PMW) override;
  Error addGlobalInstructionSelect(PassManagerWrapper &PMW) override;

  Error addFastRegAlloc(PassManagerWrapper &PMW) override;

  Error addOptimizedRegAlloc(PassManagerWrapper &PMW) override;
};

void SPIRVCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  addFunctionPass(AtomicExpandPass(TM), PMW);

  Base::addIRPasses(PMW);

  flushFPMsToMPM(PMW);

  if (getTM().getSubtargetImpl()->isShader()) {
    if (getOptLevel() != CodeGenOptLevel::None) {
      addModulePass(SPIRVFinalizeShaderLinkagePass(getTM()), PMW);
    }
  } else {
    // Variadic function calls aren't supported in shader code.
    // This needs to come before SPIRVPrepareFunctions because this
    // may introduce intrinsic calls.
    addModulePass(ExpandVariadicsPass(ExpandVariadicsMode::Lowering), PMW);
  }

  addFunctionPass(SPIRVRegularizerPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(SPIRVCtorDtorLoweringPass(), PMW);
  addModulePass(SPIRVPrepareFunctionsPass(getTM()), PMW);
  addModulePass(SPIRVPrepareGlobalsPass(), PMW);
}

void SPIRVCodeGenPassBuilder::addISelPrepare(PassManagerWrapper &PMW) {
  SPIRVTargetMachine &TM = getTM();
  if (getTM().getSubtargetImpl()->isShader()) {
    // Vulkan does not allow address space casts. This pass is run to remove
    // address space casts that can be removed.
    // If an address space cast is not removed while targeting Vulkan, lowering
    // will fail during MIR lowering.
    addFunctionPass(InferAddressSpacesPass(), PMW);

    // 1.  Simplify loop for subsequent transformations. After this steps, loops
    // have the following properties:
    //  - loops have a single entry edge (pre-header to loop header).
    //  - all loop exits are dominated by the loop pre-header.
    //  - loops have a single back-edge.
    addFunctionPass(LoopSimplifyPass(), PMW);

    // 2. Removes registers whose lifetime spans across basic blocks. Also
    // removes phi nodes. This will greatly simplify the next steps.
    addFunctionPass(RegToMemPass(), PMW);

    // 3. Merge the convergence region exit nodes into one. After this step,
    // regions are single-entry, single-exit. This will help determine the
    // correct merge block.
    addFunctionPass(SPIRVMergeRegionExitTargetsPass(), PMW);

    // 4. Structurize.
    addFunctionPass(SPIRVStructurizerPass(), PMW);

    // 5. Reduce the amount of variables required by pushing some operations
    // back to virtual registers.
    addFunctionPass(PromotePass(), PMW);
  } else {
    // Canonicalize loops so they have a single latch and preheader.
    // This enables OpLoopMerge emission for non-shader targets.
    addFunctionPass(LoopSimplifyPass(), PMW);
  }
  addFunctionPass(StripConvergenceIntrinsicsPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(SPIRVLegalizeImplicitBindingPass(), PMW);
  addModulePass(SPIRVLegalizeZeroSizeArraysPass(getTM()), PMW);
  addModulePass(SPIRVCBufferAccessPass(), PMW);
  addModulePass(SPIRVPushConstantAccessPass(getTM()), PMW);
  addModulePass(SPIRVEmitIntrinsicsPass(getTM()), PMW);
  if (TM.getSubtargetImpl()->isLogicalSPIRV())
    addFunctionPass(SPIRVLegalizePointerCastPass(getTM()), PMW);
  Base::addISelPrepare(PMW);
}

Error SPIRVCodeGenPassBuilder::addIRTranslator(PassManagerWrapper &PMW) {
  addMachineFunctionPass(IRTranslatorPass(getOptLevel()), PMW);
  return Error::success();
}

void SPIRVCodeGenPassBuilder::addPreLegalizeMachineIR(PassManagerWrapper &PMW) {
  // TODO(boomanaiden154): Add SPIRVPreLegalizerCombiner when it has been
  // ported.
  // TODO(boomanaiden154): Add SPIRVPreLegalizerPass when it has been ported.
}

Error SPIRVCodeGenPassBuilder::addLegalizeMachineIR(PassManagerWrapper &PMW) {
  addMachineFunctionPass(LegalizerPass(), PMW);
  // TODO(boomanaiden154): Add SPIRVPostLegalizerPass when it has been ported.
  return Error::success();
}

Error SPIRVCodeGenPassBuilder::addRegBankSelect(PassManagerWrapper &PMW) {
  // We do not add RegBankSelectPass as we only ever need virtual registers.
  return Error::success();
}

Error SPIRVCodeGenPassBuilder::addGlobalInstructionSelect(
    PassManagerWrapper &PMW) {
  addMachineFunctionPass(InstructionSelectPass(getOptLevel()), PMW);
  return Error::success();
}

// We do nothing in register allocation as we keep virtual registers.
Error SPIRVCodeGenPassBuilder::addFastRegAlloc(PassManagerWrapper &PMW) {
  return Error::success();
}

Error SPIRVCodeGenPassBuilder::addOptimizedRegAlloc(PassManagerWrapper &PMW) {
  return Error::success();
}

} // namespace

void SPIRVTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "SPIRVPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error SPIRVTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = SPIRVCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}