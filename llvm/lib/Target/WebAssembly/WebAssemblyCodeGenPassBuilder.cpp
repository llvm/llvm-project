//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyAsmPrinter.h"
#include "WebAssemblyExceptionInfo.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/MachineBlockPlacement.h"
#include "llvm/CodeGen/MachineCopyPropagation.h"
#include "llvm/CodeGen/MachineLateInstrsCleanup.h"
#include "llvm/CodeGen/PatchableFunction.h"
#include "llvm/CodeGen/PostRAMachineSink.h"
#include "llvm/CodeGen/PostRASchedulerList.h"
#include "llvm/CodeGen/RegisterCoalescerPass.h"
#include "llvm/CodeGen/RemoveLoadsIntoFakeUses.h"
#include "llvm/CodeGen/ShrinkWrap.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/Utils/LowerGlobalDtors.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"

using namespace llvm;

namespace WebAssembly {
extern cl::opt<bool> WasmDisableExplicitLocals;
extern cl::opt<bool> WasmEnableEH;
extern cl::opt<bool> WasmEnableEmEH;
extern cl::opt<bool> WasmEnableEmSjLj;
extern cl::opt<bool> WasmEnableSjLj;
} // namespace WebAssembly

using llvm::WebAssembly::WasmDisableExplicitLocals;
using llvm::WebAssembly::WasmEnableEH;
using llvm::WebAssembly::WasmEnableEmEH;
using llvm::WebAssembly::WasmEnableEmSjLj;
using llvm::WebAssembly::WasmEnableSjLj;

namespace {

class WebAssemblyCodeGenPassBuilder
    : public CodeGenPassBuilder<WebAssemblyCodeGenPassBuilder,
                                WebAssemblyTargetMachine> {
  using Base = CodeGenPassBuilder<WebAssemblyCodeGenPassBuilder,
                                  WebAssemblyTargetMachine>;

public:
  explicit WebAssemblyCodeGenPassBuilder(WebAssemblyTargetMachine &TM,
                                         const CGPassBuilderOption &Opts,
                                         PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {
    disablePass<MachineLateInstrsCleanupPass, MachineCopyPropagationPass,
                PostRAMachineSinkingPass, PostRASchedulerPass,
                FuncletLayoutPass, StackMapLivenessPass, PatchableFunctionPass,
                ShrinkWrapPass, RemoveLoadsIntoFakeUsesPass,
                MachineBlockPlacementPass>();

    // Currently RegisterCoalesce degrades wasm debug info quality by a
    // significant margin. As a quick fix, disable this for -O1, which is often
    // used for debugging large applications. Disabling this increases code size
    // of Emscripten core benchmarks by ~5%, which is acceptable for -O1, which
    // is usually not used for production builds.
    // TODO Investigate why RegisterCoalesce degrades debug info quality and fix
    // it properly
    if (getOptLevel() == CodeGenOptLevel::Less)
      disablePass<RegisterCoalescerPass>();
  }

  void addIRPasses(PassManagerWrapper &PMW) const;
  void addISelPrepare(PassManagerWrapper &PMW) const;
  Error addInstSelector(PassManagerWrapper &PMW) const;
  Error addRegAssignAndRewriteFast(PassManagerWrapper &PMW) const;
  Expected<bool> addRegAssignAndRewriteOptimized(PassManagerWrapper &PMW) const;
  void addPreEmitPass(PassManagerWrapper &PMW) const;
  void addAsmPrinterBegin(PassManagerWrapper &PMW) const;
  void addAsmPrinter(PassManagerWrapper &PMW) const;
  void addAsmPrinterEnd(PassManagerWrapper &PMW) const;
};

void WebAssemblyCodeGenPassBuilder::addIRPasses(PassManagerWrapper &PMW) const {
  // Add signatures to prototype-less function declarations
  flushFPMsToMPM(PMW);
  addModulePass(WebAssemblyAddMissingPrototypesPass(), PMW);

  // Lower .llvm.global_dtors into .llvm.global_ctors with __cxa_atexit calls.
  addModulePass(LowerGlobalDtorsPass(), PMW);

  // Fix function bitcasts, as WebAssembly requires caller and callee signatures
  // to match.
  addModulePass(WebAssemblyFixFunctionBitcastsPass(), PMW);

  // Optimize "returned" function attributes.
  if (getOptLevel() != CodeGenOptLevel::None)
    addFunctionPass(WebAssemblyOptimizeReturnedPass(), PMW);

  // If exception handling is not enabled and setjmp/longjmp handling is
  // enabled, we lower invokes into calls and delete unreachable landingpad
  // blocks. Lowering invokes when there is no EH support is done in
  // TargetPassConfig::addPassesToHandleExceptions, but that runs after these IR
  // passes and Emscripten SjLj handling expects all invokes to be lowered
  // before.
  if (!WasmEnableEmEH && !WasmEnableEH) {
    addFunctionPass(LowerInvokePass(), PMW);
    // The lower invoke pass may create unreachable code. Remove it in order not
    // to process dead blocks in setjmp/longjmp handling.
    addFunctionPass(UnreachableBlockElimPass(), PMW);
  }

  // Handle exceptions and setjmp/longjmp if enabled. Unlike Wasm EH preparation
  // done in WasmEHPrepare pass, Wasm SjLj preparation shares libraries and
  // transformation algorithms with Emscripten SjLj, so we run
  // LowerEmscriptenEHSjLj pass also when Wasm SjLj is enabled.
  if (WasmEnableEmEH || WasmEnableEmSjLj || WasmEnableSjLj) {
    flushFPMsToMPM(PMW);
    addModulePass(WebAssemblyLowerEmscriptenEHSjLjPass(), PMW);
  }

  // Expand indirectbr instructions to switches.
  addFunctionPass(IndirectBrExpandPass(TM), PMW);

  // Try to expand `vecreduce_{and, or}` into `{any, all}_true`.
  addFunctionPass(WebAssemblyReduceToAnyAllTruePass(TM), PMW);

  Base::addIRPasses(PMW);
}

void WebAssemblyCodeGenPassBuilder::addISelPrepare(
    PassManagerWrapper &PMW) const {
  // We need to move reference type allocas to WASM_ADDRESS_SPACE_VAR so that
  // loads and stores are promoted to local.gets/local.sets.
  addFunctionPass(WebAssemblyRefTypeMem2LocalPass(), PMW);
  // Lower atomics and TLS if necessary
  flushFPMsToMPM(PMW);
  addModulePass(WebAssemblyCoalesceFeaturesAndStripAtomicsPass(TM), PMW);

  // This is a no-op if atomics are not used in the module
  addFunctionPass(AtomicExpandPass(TM), PMW);

  Base::addISelPrepare(PMW);
}

Error WebAssemblyCodeGenPassBuilder::addInstSelector(
    PassManagerWrapper &PMW) const {
  addMachineFunctionPass(WebAssemblyISelDAGToDAGPass(TM, getOptLevel()), PMW);

  // Run the argument-move pass immediately after the ScheduleDAG scheduler
  // so that we can fix up the ARGUMENT instructions before anything else
  // sees them in the wrong place.
  addMachineFunctionPass(WebAssemblyArgumentMovePass(), PMW);

  // Set the p2align operands. This information is present during ISel, however
  // it's inconvenient to collect. Collect it now, and update the immediate
  // operands.
  addMachineFunctionPass(WebAssemblySetP2AlignOperandsPass(), PMW);

  // Eliminate range checks and add default targets to br_table instructions.
  addMachineFunctionPass(WebAssemblyFixBrTableDefaultsPass(), PMW);

  // unreachable is terminator, non-terminator instruction after it is not
  // allowed.
  addMachineFunctionPass(WebAssemblyCleanCodeAfterTrapPass(), PMW);

  return Error::success();
}

Error WebAssemblyCodeGenPassBuilder::addRegAssignAndRewriteFast(
    PassManagerWrapper &PMW) const {
  return Error::success();
}

Expected<bool> WebAssemblyCodeGenPassBuilder::addRegAssignAndRewriteOptimized(
    PassManagerWrapper &PMW) const {
  return false;
}

void WebAssemblyCodeGenPassBuilder::addPreEmitPass(
    PassManagerWrapper &PMW) const {
  Base::addPreEmitPass(PMW);

  // Nullify DBG_VALUE_LISTs that we cannot handle.
  addMachineFunctionPass(WebAssemblyNullifyDebugValueListsPass(), PMW);

  // Remove any unreachable blocks that may be left floating around.
  // Rare, but possible. Needed for WebAssemblyFixIrreducibleControlFlow.
  addMachineFunctionPass(UnreachableMachineBlockElimPass(), PMW);

  // Eliminate multiple-entry loops.
  addMachineFunctionPass(WebAssemblyFixIrreducibleControlFlowPass(), PMW);

  // Do various transformations for exception handling.
  // Every CFG-changing optimizations should come before this.
  if (TM.Options.ExceptionModel == ExceptionHandling::Wasm)
    addMachineFunctionPass(WebAssemblyLateEHPreparePass(), PMW);

  // Now that we have a prologue and epilogue and all frame indices are
  // rewritten, eliminate SP and FP. This allows them to be stackified,
  // colored, and numbered with the rest of the registers.
  addMachineFunctionPass(WebAssemblyReplacePhysRegsPass(), PMW);

  // Preparations and optimizations related to register stackification.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // Depend on LiveIntervals and perform some optimizations on it.
    addMachineFunctionPass(WebAssemblyOptimizeLiveIntervalsPass(), PMW);

    // Prepare memory intrinsic calls for register stackifying.
    addMachineFunctionPass(WebAssemblyMemIntrinsicResultsPass(), PMW);
  }

  // Mark registers as representing wasm's value stack. This is a key
  // code-compression technique in WebAssembly. We run this pass (and
  // MemIntrinsicResults above) very late, so that it sees as much code as
  // possible, including code emitted by PEI and expanded by late tail
  // duplication.
  addMachineFunctionPass(WebAssemblyRegStackifyPass(getOptLevel()), PMW);

  if (getOptLevel() != CodeGenOptLevel::None) {
    // Run the register coloring pass to reduce the total number of registers.
    // This runs after stackification so that it doesn't consider registers
    // that become stackified.
    addMachineFunctionPass(WebAssemblyRegColoringPass(), PMW);
  }

  // Sort the blocks of the CFG into topological order, a prerequisite for
  // BLOCK and LOOP markers.
  addMachineFunctionPass(WebAssemblyCFGSortPass(), PMW);

  // Insert BLOCK and LOOP markers.
  addMachineFunctionPass(WebAssemblyCFGStackifyPass(), PMW);

  // Insert explicit local.get and local.set operators.
  if (!WasmDisableExplicitLocals)
    addMachineFunctionPass(WebAssemblyExplicitLocalsPass(), PMW);

  // Lower br_unless into br_if.
  addMachineFunctionPass(WebAssemblyLowerBrUnlessPass(), PMW);

  // Perform the very last peephole optimizations on the code.
  if (getOptLevel() != CodeGenOptLevel::None)
    addMachineFunctionPass(WebAssemblyPeepholePass(), PMW);

  // Create a mapping from LLVM CodeGen virtual registers to wasm registers.
  addMachineFunctionPass(WebAssemblyRegNumberingPass(), PMW);

  // Fix debug_values whose defs have been stackified.
  if (!WasmDisableExplicitLocals)
    addMachineFunctionPass(WebAssemblyDebugFixupPass(), PMW);

  // Collect information to prepare for MC lowering / asm printing.
  flushFPMsToMPM(PMW);
  addModulePass(WebAssemblyMCLowerPrePass(), PMW);
}

void WebAssemblyCodeGenPassBuilder::addAsmPrinterBegin(
    PassManagerWrapper &PMW) const {
  addModulePass(WebAssemblyAsmPrinterBeginPass(), PMW, /*Force=*/true);
}

void WebAssemblyCodeGenPassBuilder::addAsmPrinter(
    PassManagerWrapper &PMW) const {
  addMachineFunctionPass(WebAssemblyAsmPrinterPass(), PMW);
}

void WebAssemblyCodeGenPassBuilder::addAsmPrinterEnd(
    PassManagerWrapper &PMW) const {
  addModulePass(WebAssemblyAsmPrinterEndPass(), PMW);
}

} // namespace

void WebAssemblyTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "WebAssemblyPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
  // TODO(boomanaiden154): Move this into the base CodeGenPassBuilder once all
  // targets that currently implement it have a ported asm-printer pass.
  if (PIC) {
    PIC->addClassToPassName(WebAssemblyAsmPrinterBeginPass::name(),
                            "wasm-asm-printer-begin");
    PIC->addClassToPassName(WebAssemblyAsmPrinterPass::name(),
                            "wasm-asm-printer");
    PIC->addClassToPassName(WebAssemblyAsmPrinterEndPass::name(),
                            "wasm-asm-printer-end");
  }
}

Error WebAssemblyTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = WebAssemblyCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
