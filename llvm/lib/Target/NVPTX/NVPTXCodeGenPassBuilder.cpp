//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the NVPTX CodeGen pipeline builder. It mirrors
/// NVPTXPassConfig in NVPTXTargetMachine.cpp; the two must be kept in sync
/// until the legacy pass manager path is removed.
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVPTXAliasAnalysis.h"
#include "NVPTXAsmPrinter.h"
#include "NVPTXSubtarget.h"
#include "NVPTXTargetMachine.h"
#include "llvm/Analysis/KernelInfo.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/DeadMachineInstructionElim.h"
#include "llvm/CodeGen/FuncletLayout.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineLateInstrsCleanup.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/PEI.h"
#include "llvm/CodeGen/PHIElimination.h"
#include "llvm/CodeGen/PatchableFunction.h"
#include "llvm/CodeGen/PostRAMachineSink.h"
#include "llvm/CodeGen/PostRASchedulerList.h"
#include "llvm/CodeGen/ProcessImplicitDefs.h"
#include "llvm/CodeGen/RegisterCoalescerPass.h"
#include "llvm/CodeGen/RemoveLoadsIntoFakeUses.h"
#include "llvm/CodeGen/ShrinkWrap.h"
#include "llvm/CodeGen/StackColoring.h"
#include "llvm/CodeGen/StackSlotColoring.h"
#include "llvm/CodeGen/TailDuplication.h"
#include "llvm/CodeGen/TwoAddressInstructionPass.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;

extern cl::opt<bool> DisableLoadStoreVectorizer;
extern cl::opt<bool> DisableNVPTXIRPeephole;

// byval arguments in NVPTX are special. We're only allowed to read from them
// using a special instruction, and if we ever need to write to them or take an
// address, we must make a local copy and use it, instead.
//
// The problem is that local copies are very expensive, and we create them very
// late in the compilation pipeline, so LLVM does not have much of a chance to
// eliminate them, if they turn out to be unnecessary.
//
// One way around that is to create such copies early on, and let them percolate
// through the optimizations. The copying itself will never trigger creation of
// another copy later on, as the reads are allowed. If LLVM can eliminate it,
// it's a win. It the full optimization pipeline can't remove the copy, that's
// as good as it gets in terms of the effort we could've done, and it's
// certainly a much better effort than what we do now.
//
// This early injection of the copies has potential to create undesireable
// side-effects, so it's disabled by default, for now, until it sees more
// testing.
static cl::opt<bool> EarlyByValArgsCopy(
    "nvptx-early-byval-copy",
    cl::desc("Create a copy of byval function arguments early."),
    cl::init(false), cl::Hidden);

namespace {

class NVPTXCodeGenPassBuilder : public CodeGenPassBuilder {
  using Base = CodeGenPassBuilder;

  NVPTXTargetMachine &getTM() const {
    return static_cast<NVPTXTargetMachine &>(TM);
  }

public:
  explicit NVPTXCodeGenPassBuilder(NVPTXTargetMachine &TM,
                                   const CGPassBuilderOption &Opts,
                                   PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {
    // The following passes are known to not play well with virtual regs
    // hanging around after register allocation (which in our case, is *all*
    // registers). We explicitly disable them here. We do, however, need some
    // functionality of the PrologEpilogCodeInserter pass, so we emulate that
    // behavior in the NVPTXPrologEpilog pass (see NVPTXPrologEpilogPass.cpp).
    disablePass<PrologEpilogInserterPass, MachineLateInstrsCleanupPass,
                MachineCopyPropagationPass, TailDuplicatePass,
                StackMapLivenessPass, PostRAMachineSinkingPass,
                PostRASchedulerPass, FuncletLayoutPass, PatchableFunctionPass,
                ShrinkWrapPass, RemoveLoadsIntoFakeUsesPass>();
  }

  void addIRPasses(PassManagerWrapper &PMW) override;
  Error addInstSelector(PassManagerWrapper &PMW) override;
  void addPreRegAlloc(PassManagerWrapper &PMW) override;
  void addPostRegAlloc(PassManagerWrapper &PMW) override;

  // NVPTX has no register allocation; virtual registers are emitted directly.
  void addTargetRegisterAllocator(PassManagerWrapper &PMW, bool) override {}
  Error addFastRegAlloc(PassManagerWrapper &PMW) override;
  Error addOptimizedRegAlloc(PassManagerWrapper &PMW) override;

  void addAsmPrinterBegin(PassManagerWrapper &PMW) override;
  void addAsmPrinter(PassManagerWrapper &PMW) override;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) override;

private:
  // If the opt level is aggressive, add GVN; otherwise, add EarlyCSE.
  void addEarlyCSEOrGVNPass(PassManagerWrapper &PMW);

  // Add passes that propagate special memory spaces.
  void addAddressSpaceInferencePasses(PassManagerWrapper &PMW);

  // Add passes that perform straight-line scalar optimizations.
  void addStraightLineScalarOptimizationPasses(PassManagerWrapper &PMW);
};

void NVPTXCodeGenPassBuilder::addEarlyCSEOrGVNPass(PassManagerWrapper &PMW) {
  if (getOptLevel() == CodeGenOptLevel::Aggressive)
    // Disable scalar PRE due to Register Pressure increase
    addFunctionPass(GVNPass(GVNOptions().setScalarPRE(false)), PMW);
  else
    addFunctionPass(EarlyCSEPass(), PMW);
}

void NVPTXCodeGenPassBuilder::addAddressSpaceInferencePasses(
    PassManagerWrapper &PMW) {
  // NVPTXLowerArgs emits alloca for byval parameters which can often
  // be eliminated by SROA.
  addFunctionPass(SROAPass(SROAOptions(SROAOptions::PreserveCFG,
                                       /*AggregateToVector=*/true)),
                  PMW);
  addFunctionPass(NVPTXLowerAllocaPass(), PMW);
  // TODO: Consider running InferAddressSpaces during opt, earlier in the
  // compilation flow.
  addFunctionPass(InferAddressSpacesPass(), PMW);
  addFunctionPass(NVPTXAtomicLowerPass(), PMW);
}

void NVPTXCodeGenPassBuilder::addStraightLineScalarOptimizationPasses(
    PassManagerWrapper &PMW) {
  addFunctionPass(SeparateConstOffsetFromGEPPass(), PMW);
  addFunctionPass(SpeculativeExecutionPass(), PMW);
  // ReassociateGEPs exposes more opportunites for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addFunctionPass(StraightLineStrengthReducePass(), PMW);
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN
  // or EarlyCSE can reuse. GVN generates significantly better code than
  // EarlyCSE for some of our benchmarks.
  addEarlyCSEOrGVNPass(PMW);
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addFunctionPass(NaryReassociatePass(), PMW);
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addFunctionPass(EarlyCSEPass(), PMW);
}

void NVPTXCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) {
  const NVPTXSubtarget &ST = *getTM().getSubtargetImpl();

  // NVVMReflectPass is added in the pipeline-start extension point, so
  // hopefully running it here does nothing. But since we need it for
  // correctness when lowering to NVPTX, run it here too, in case whoever built
  // our pass pipeline didn't add it.
  flushFPMsToMPM(PMW);
  addModulePass(NVVMReflectPass(ST.getSmVersion()), PMW);

  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(NVPTXImageOptimizerPass(), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(NVPTXAssignValidGlobalNamesPass(), PMW);
  addModulePass(GenericToNVVMPass(), PMW);

  // Lower variadic calls before address space inference.
  addModulePass(ExpandVariadicsPass(ExpandVariadicsMode::Lowering), PMW);

  // NVPTXLowerArgs is required for correctness and should be run right
  // before the address space inference passes.
  if (getTM().getDrvInterface() == NVPTX::CUDA) {
    addFunctionPass(NVPTXMarkKernelPtrsGlobalPass(), PMW);
    flushFPMsToMPM(PMW);
  }
  addModulePass(NVPTXPromoteParamAlignPass(), PMW);
  addModulePass(NVPTXLowerArgsPass(TM), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    addAddressSpaceInferencePasses(PMW);
    addStraightLineScalarOptimizationPasses(PMW);
  } else {
    // Required for correct stack lowering
    addFunctionPass(NVPTXLowerAllocaPass(), PMW);
  }

  addFunctionPass(AtomicExpandPass(TM), PMW);
  flushFPMsToMPM(PMW);
  addModulePass(NVPTXCtorDtorLoweringPass(), PMW);

  // === LSR and other generic IR passes ===
  Base::addIRPasses(PMW);
  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (getOptLevel() != CodeGenOptLevel::None) {
    addEarlyCSEOrGVNPass(PMW);
    if (!DisableLoadStoreVectorizer)
      addFunctionPass(LoadStoreVectorizerPass(), PMW);
    addFunctionPass(SROAPass(SROAOptions(SROAOptions::PreserveCFG,
                                         /*AggregateToVector=*/true)),
                    PMW);
    addFunctionPass(NVPTXTagInvariantLoadsPass(), PMW);
    if (!DisableNVPTXIRPeephole)
      addFunctionPass(NVPTXIRPeepholePass(), PMW);
  }

  if (ST.hasPTXASUnreachableBug()) {
    // Run LowerUnreachable to WAR a ptxas bug. See the commit description of
    // 1ee4d880e8760256c606fe55b7af85a4f70d006d for more details.
    addFunctionPass(NVPTXLowerUnreachablePass(TM.Options.TrapUnreachable,
                                              TM.Options.NoTrapAfterNoreturn),
                    PMW);
  }
}

Error NVPTXCodeGenPassBuilder::addInstSelector(PassManagerWrapper &PMW) {
  addFunctionPass(NVPTXLowerAggrCopiesPass(), PMW);
  addFunctionPass(NVPTXAllocaHoistingPass(), PMW);
  addMachineFunctionPass(NVPTXISelDAGToDAGPass(getTM(), getOptLevel()), PMW);
  addMachineFunctionPass(NVPTXReplaceImageHandlesPass(), PMW);
  return Error::success();
}

void NVPTXCodeGenPassBuilder::addPreRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(NVPTXForwardParamsPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineFunctionPass(NVPTXAddressFolderPass(), PMW);
  // Remove Proxy Register pseudo instructions used to keep `callseq_end` alive.
  addMachineFunctionPass(NVPTXProxyRegErasurePass(), PMW);
}

void NVPTXCodeGenPassBuilder::addPostRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(NVPTXPrologEpilogPass(), PMW);
  if (getOptLevel() != CodeGenOptLevel::None) {
    // NVPTXPrologEpilogPass calculates frame object offset and replaces frame
    // index with VRFrame register. NVPTXPeephole needs to be run after that
    // and will replace VRFrame with VRFrameLocal when possible.
    addMachineFunctionPass(NVPTXPeepholePass(), PMW);
  }
}

Error NVPTXCodeGenPassBuilder::addFastRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(PHIEliminationPass(), PMW);
  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  return Error::success();
}

Error NVPTXCodeGenPassBuilder::addOptimizedRegAlloc(PassManagerWrapper &PMW) {
  addMachineFunctionPass(ProcessImplicitDefsPass(), PMW);
  // LiveVariables requires pure SSA form and no unreachable blocks; the legacy
  // pass manager pulls UnreachableMachineBlockElim in as an implicit
  // dependency, so add it explicitly here.
  addMachineFunctionPass(UnreachableMachineBlockElimPass(), PMW);
  addMachineFunctionPass(
      RequireAnalysisPass<LiveVariablesAnalysis, MachineFunction>(), PMW);
  addMachineFunctionPass(
      RequireAnalysisPass<MachineLoopAnalysis, MachineFunction>(), PMW);
  addMachineFunctionPass(PHIEliminationPass(), PMW);

  addMachineFunctionPass(TwoAddressInstructionPass(), PMW);
  addMachineFunctionPass(RegisterCoalescerPass(), PMW);

  // PreRA instruction scheduling.
  addMachineFunctionPass(MachineSchedulerPass(&TM), PMW);

  addMachineFunctionPass(StackSlotColoringPass(), PMW);

  // FIXME: Needs physical registers
  // addMachineFunctionPass(MachineLICMPass(), PMW);

  return Error::success();
}

void NVPTXCodeGenPassBuilder::addAsmPrinterBegin(PassManagerWrapper &PMW) {
  addModulePass(NVPTXAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void NVPTXCodeGenPassBuilder::addAsmPrinter(PassManagerWrapper &PMW) {
  addMachineFunctionPass(NVPTXAsmPrinterPass(), PMW);
}

void NVPTXCodeGenPassBuilder::addAsmPrinterEnd(PassManagerWrapper &PMW) {
  addModulePass(NVPTXAsmPrinterEndPass(), PMW);
}

} // namespace

void NVPTXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "NVPTXPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"

  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        // We do not want to fold out calls to nvvm.reflect early if the user
        // has not provided a target architecture just yet.
        if (Subtarget.hasTargetName())
          PM.addPass(NVVMReflectPass(Subtarget.getSmVersion()));

        FunctionPassManager FPM;
        // Note: NVVMIntrRangePass was causing numerical discrepancies at one
        // point, if issues crop up, consider disabling.
        FPM.addPass(NVVMIntrRangePass());
        if (EarlyByValArgsCopy)
          FPM.addPass(NVPTXCopyByValArgsPass());
        PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });

  if (!NoKernelInfoEndLTO) {
    PB.registerFullLinkTimeOptimizationLastEPCallback(
        [this](ModulePassManager &PM, OptimizationLevel Level) {
          FunctionPassManager FPM;
          FPM.addPass(KernelInfoPrinter(this));
          PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
        });
  }
}

Error NVPTXTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = NVPTXCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
