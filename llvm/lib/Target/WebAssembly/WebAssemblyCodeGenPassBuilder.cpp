//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
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

  void addIRPasses(CodeGenModulePassManager &CGMPM) const;
  void addISelPrepare(CodeGenModulePassManager &CGMPM) const;
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const;
};

void WebAssemblyCodeGenPassBuilder::addIRPasses(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  // Add signatures to prototype-less function declarations
  CGMPM.addPass(WebAssemblyAddMissingPrototypesPass());

  // Lower .llvm.global_dtors into .llvm.global_ctors with __cxa_atexit calls.
  CGMPM.addPass(LowerGlobalDtorsPass());

  // Fix function bitcasts, as WebAssembly requires caller and callee signatures
  // to match.
  CGMPM.addPass(WebAssemblyFixFunctionBitcastsPass());

  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  // Optimize "returned" function attributes.
  if (getOptLevel() != CodeGenOptLevel::None)
    CGFPM.addPass(WebAssemblyOptimizeReturnedPass());

  // If exception handling is not enabled and setjmp/longjmp handling is
  // enabled, we lower invokes into calls and delete unreachable landingpad
  // blocks. Lowering invokes when there is no EH support is done in
  // TargetPassConfig::addPassesToHandleExceptions, but that runs after these IR
  // passes and Emscripten SjLj handling expects all invokes to be lowered
  // before.
  if (!WasmEnableEmEH && !WasmEnableEH) {
    CGFPM.addPass(LowerInvokePass());
    // The lower invoke pass may create unreachable code. Remove it in order not
    // to process dead blocks in setjmp/longjmp handling.
    CGFPM.addPass(UnreachableBlockElimPass());
  }

  // Handle exceptions and setjmp/longjmp if enabled. Unlike Wasm EH preparation
  // done in WasmEHPrepare pass, Wasm SjLj preparation shares libraries and
  // transformation algorithms with Emscripten SjLj, so we run
  // LowerEmscriptenEHSjLj pass also when Wasm SjLj is enabled.
  if (WasmEnableEmEH || WasmEnableEmSjLj || WasmEnableSjLj) {
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
    CGMPM.addPass(WebAssemblyLowerEmscriptenEHSjLjPass());
  }

  // Expand indirectbr instructions to switches.
  CGFPM.addPass(IndirectBrExpandPass(TM));

  // Try to expand `vecreduce_{and, or}` into `{any, all}_true`.
  CGFPM.addPass(WebAssemblyReduceToAnyAllTruePass(TM));

  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  Base::addIRPasses(CGMPM);
}

void WebAssemblyCodeGenPassBuilder::addISelPrepare(
    CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  // We need to move reference type allocas to WASM_ADDRESS_SPACE_VAR so that
  // loads and stores are promoted to local.gets/local.sets.
  CGFPM.addPass(WebAssemblyRefTypeMem2LocalPass());
  // Lower atomics and TLS if necessary
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  CGMPM.addPass(WebAssemblyCoalesceFeaturesAndStripAtomicsPass(TM));

  // This is a no-op if atomics are not used in the module
  CGFPM.addPass(AtomicExpandPass(TM));

  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  Base::addISelPrepare(CGMPM);
}

Error WebAssemblyCodeGenPassBuilder::addInstSelector(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(WebAssemblyISelDAGToDAGPass(TM, getOptLevel()));

  // Run the argument-move pass immediately after the ScheduleDAG scheduler
  // so that we can fix up the ARGUMENT instructions before anything else
  // sees them in the wrong place.
  CGMFPM.addPass(WebAssemblyArgumentMovePass());

  // Set the p2align operands. This information is present during ISel, however
  // it's inconvenient to collect. Collect it now, and update the immediate
  // operands.
  CGMFPM.addPass(WebAssemblySetP2AlignOperandsPass());

  // Eliminate range checks and add default targets to br_table instructions.
  CGMFPM.addPass(WebAssemblyFixBrTableDefaultsPass());

  // unreachable is terminator, non-terminator instruction after it is not
  // allowed.
  CGMFPM.addPass(WebAssemblyCleanCodeAfterTrapPass());

  return Error::success();
}

void WebAssemblyCodeGenPassBuilder::addPreEmitPass(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  Base::addPreEmitPass(CGMFPM);

  // Nullify DBG_VALUE_LISTs that we cannot handle.
  CGMFPM.addPass(WebAssemblyNullifyDebugValueListsPass());

  // Remove any unreachable blocks that may be left floating around.
  // Rare, but possible. Needed for WebAssemblyFixIrreducibleControlFlow.
  CGMFPM.addPass(UnreachableMachineBlockElimPass());

  // Eliminate multiple-entry loops.
  CGMFPM.addPass(WebAssemblyFixIrreducibleControlFlowPass());

  // Do various transformations for exception handling.
  // Every CFG-changing optimizations should come before this.
  if (TM.Options.ExceptionModel == ExceptionHandling::Wasm)
    CGMFPM.addPass(WebAssemblyLateEHPreparePass());

  // Now that we have a prologue and epilogue and all frame indices are
  // rewritten, eliminate SP and FP. This allows them to be stackified,
  // colored, and numbered with the rest of the registers.
  CGMFPM.addPass(WebAssemblyReplacePhysRegsPass());

  // Preparations and optimizations related to register stackification.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // Depend on LiveIntervals and perform some optimizations on it.
    // TODO(boomanaiden154): WebAssemblyOptimizeLiveIntervals

    // Prepare memory intrinsic calls for register stackifying.
    // TODO(boomanaiden154): WebAssemblyMemIntrinsicResults
  }

  // Mark registers as representing wasm's value stack. This is a key
  // code-compression technique in WebAssembly. We run this pass (and
  // MemIntrinsicResults above) very late, so that it sees as much code as
  // possible, including code emitted by PEI and expanded by late tail
  // duplication.
  // TODO(boomanaiden154): WebAssemblyRegStackify

  if (getOptLevel() != CodeGenOptLevel::None) {
    // Run the register coloring pass to reduce the total number of registers.
    // This runs after stackification so that it doesn't consider registers
    // that become stackified.
    // TODO(boomanaiden154): WebAssemblyRegColoring
  }

  // Sort the blocks of the CFG into topological order, a prerequisite for
  // BLOCK and LOOP markers.
  // TODO(boomanaiden154): WebAssemblyCFGSort

  // Insert BLOCK and LOOP markers.
  // TODO(boomanaiden154): WebAssemblyCFGStackify

  // Insert explicit local.get and local.set operators.
  if (!WasmDisableExplicitLocals) {
    // TODO(boomanaiden154): WebAssemblyExplicitLocals
  }

  // Lower br_unless into br_if.
  // TODO(boomanaiden154): WebAssemblyLowerBrUnless

  // Perform the very last peephole optimizations on the code.
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): WebAssemblyPeephole
  }

  // Create a mapping from LLVM CodeGen virtual registers to wasm registers.
  // TODO(boomanaiden154): WebAssemblyRegNumbering

  // Fix debug_values whose defs have been stackified.
  if (!WasmDisableExplicitLocals) {
    // TODO(boomanaiden154): WebAssemblyDebugFixup
  }

  // Collect information to prepare for MC lowering / asm printing.
  // TODO(boomanaiden154): WebAssemblyMCLowerPrePass
}

} // namespace

void WebAssemblyTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "WebAssemblyPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

Error WebAssemblyTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = WebAssemblyCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
