//===-- X86CodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains X86 CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86AsmPrinter.h"
#include "X86TargetMachine.h"

#include "llvm/CodeGen/AtomicExpand.h"
#include "llvm/CodeGen/BreakFalseDeps.h"
#include "llvm/CodeGen/EarlyIfConversion.h"
#include "llvm/CodeGen/IndirectBrExpand.h"
#include "llvm/CodeGen/InterleavedAccess.h"
#include "llvm/CodeGen/JMCInstrumenter.h"
#include "llvm/CodeGen/KCFI.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Transforms/CFGuard.h"

using namespace llvm;

extern cl::opt<bool> X86EnableMachineCombinerPass;

namespace {

class X86CodeGenPassBuilder
    : public CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine> {
  using Base = CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine>;

public:
  explicit X86CodeGenPassBuilder(X86TargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder(TM, Opts, PIC) {}

  void addIRPasses(CodeGenModulePassManager &CGMPM) const;
  CodeGenModulePassManager addPreISel() const;
  Error addInstSelector(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreLegalizeMachineIR(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addILPOpts(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreRegBankSelect(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void
  addMachineSSAOptimization(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const;
  // TODO(boomanaiden154): We need to add addPostFastRegAllocRewrite here once
  // it is available to support AMX.
  void addPostRegAlloc(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreSched2(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addPreEmitPass2(CodeGenMachineFunctionPassManager &CGMFPM) const;
  // TODO(boomanaiden154): We need to add addRegAssignAndRewriteOptimized here
  // once it is available to support AMX.
  void addAsmPrinterBegin(CodeGenModulePassManager &CGMPM) const;
  void addAsmPrinter(CodeGenMachineFunctionPassManager &CGMFPM) const;
  void addAsmPrinterEnd(CodeGenModulePassManager &CGMPM) const;
};

void X86CodeGenPassBuilder::addIRPasses(CodeGenModulePassManager &CGMPM) const {
  CodeGenFunctionPassManager CGFPM;
  CGFPM.addPass(AtomicExpandPass(TM));

  // We add both pass anyway and when these two passes run, one will be a
  // no-op based on the optimization level/attributes.
  CGFPM.addPass(X86LowerAMXIntrinsicsPass(&TM));
  CGFPM.addPass(X86LowerAMXTypePass(&TM));
  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));

  Base::addIRPasses(CGMPM);

  if (getOptLevel() != CodeGenOptLevel::None) {
    CGFPM.addPass(InterleavedAccessPass(TM));
    CGFPM.addPass(X86PartialReductionPass(&TM));
  }

  // Add passes that handle indirect branch removal and insertion of a retpoline
  // thunk. These will be a no-op unless a function subtarget has the retpoline
  // feature enabled.
  CGFPM.addPass(IndirectBrExpandPass(TM));

  // Add Control Flow Guard checks.
  const Triple &TT = TM.getTargetTriple();
  if (TT.isOSWindows())
    CGFPM.addPass(CFGuardPass());

  CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
  if (TM.Options.JMCInstrument) {
    CGMPM.addCodeGenFunctionPassManager(std::move(CGFPM));
    CGMPM.addPass(JMCInstrumenterPass());
  }
}

CodeGenModulePassManager X86CodeGenPassBuilder::addPreISel() const {
  CodeGenModulePassManager CGMPM;
  // Only add this pass for 32-bit x86 Windows.
  const Triple &TT = TM.getTargetTriple();
  if (TT.isOSWindows() && TT.isX86_32())
    CGMPM.addPass(X86WinEHStatePass());
  return CGMPM;
}

Error X86CodeGenPassBuilder::addInstSelector(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86ISelDAGToDAGPass(TM));

  // For ELF, cleanup any local-dynamic TLS accesses
  if (TM.getTargetTriple().isOSBinFormatELF() &&
      getOptLevel() != CodeGenOptLevel::None) {
    CGMFPM.addPass(X86CleanupLocalDynamicTLSPass());
  }

  CGMFPM.addPass(X86GlobalBaseRegPass());
  CGMFPM.addPass(X86ArgumentStackSlotPass());
  return Error::success();
}

void X86CodeGenPassBuilder::addPreLegalizeMachineIR(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86PreLegalizerCombinerPass());
}

void X86CodeGenPassBuilder::addILPOpts(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(EarlyIfConverterPass());
  if (X86EnableMachineCombinerPass) {
    // TODO(boomanaiden154): Add the MachineCombinerPass here once it has been
    // ported to the new pass manager.
  }
  CGMFPM.addPass(X86CmovConversionPass());
}

void X86CodeGenPassBuilder::addPreRegBankSelect(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86PostLegalizerCombinerPass());
}

void X86CodeGenPassBuilder::addMachineSSAOptimization(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86DomainReassignmentPass());
  Base::addMachineSSAOptimization(CGMFPM);
}

void X86CodeGenPassBuilder::addPreRegAlloc(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  if (getOptLevel() != CodeGenOptLevel::None) {
    CGMFPM.addPass(LiveRangeShrinkPass());
    CGMFPM.addPass(X86FixupSetCCPass());
    CGMFPM.addPass(X86CallFrameOptimizationPass());
    CGMFPM.addPass(X86AvoidStoreForwardingBlocksPass());
  }

  CGMFPM.addPass(X86SuppressAPXForRelocationPass());
  CGMFPM.addPass(X86SpeculativeLoadHardeningPass());
  CGMFPM.addPass(X86FlagsCopyLoweringPass());
  CGMFPM.addPass(X86DynAllocaExpanderPass());

  if (getOptLevel() != CodeGenOptLevel::None)
    CGMFPM.addPass(X86PreTileConfigPass());
  else
    CGMFPM.addPass(X86FastPreTileConfigPass());
}

void X86CodeGenPassBuilder::addPostRegAlloc(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86LowerTileCopyPass());
  CGMFPM.addPass(X86FPStackifierPass());
  // When -O0 is enabled, the Load Value Injection Hardening pass will fall back
  // to using the Speculative Execution Side Effect Suppression pass for
  // mitigation. This is to prevent slow downs due to
  // analyses needed by the LVIHardening pass when compiling at -O0.
  if (getOptLevel() != CodeGenOptLevel::None) {
    CGMFPM.addPass(X86LoadValueInjectionLoadHardeningPass());
  }
}

void X86CodeGenPassBuilder::addPreSched2(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86ExpandPseudoPass());
  CGMFPM.addPass(MachineKCFIPass());
}

void X86CodeGenPassBuilder::addPreEmitPass(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  if (getOptLevel() != CodeGenOptLevel::None) {
    // TODO(boomanaiden154): Add X86ExecutionDomainFixPass here once it has
    // been ported.
    CGMFPM.addPass(BreakFalseDepsPass());
  }

  CGMFPM.addPass(X86IndirectBranchTrackingPass());
  CGMFPM.addPass(X86InsertVZeroUpperPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    CGMFPM.addPass(X86FixupBWInstsPass());
    // TODO(boomanaiden154): Add X86PadShortFunctionsPass here once it has been
    // ported.
    CGMFPM.addPass(X86FixupLEAsPass());
    CGMFPM.addPass(X86FixupInstTuningPass());
    CGMFPM.addPass(X86FixupVectorConstantsPass());
  }
  CGMFPM.addPass(X86CompressEVEXPass());
  CGMFPM.addPass(X86InsertX87WaitPass());
}

void X86CodeGenPassBuilder::addPreEmitPass2(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  const Triple &TT = TM.getTargetTriple();
  const MCAsmInfo &MAI = TM.getMCAsmInfo();

  // The X86 Speculative Execution Pass must run after all control
  // flow graph modifying passes. As a result it was listed to run right before
  // the X86 Retpoline Thunks pass. The reason it must run after control flow
  // graph modifications is that the model of LFENCE in LLVM has to be updated
  // (FIXME: https://bugs.llvm.org/show_bug.cgi?id=45167). Currently the
  // placement of this pass was hand checked to ensure that the subsequent
  // passes don't move the code around the LFENCEs in a way that will hurt the
  // correctness of this pass. This placement has been shown to work based on
  // hand inspection of the codegen output.
  CGMFPM.addPass(X86SpeculativeExecutionSideEffectSuppressionPass());
  // TODO(boomanaiden154): Add X86IndirectThunksPass here
  // once it has been ported.
  CGMFPM.addPass(X86ReturnThunksPass());

  // Insert extra int3 instructions after trailing call instructions to avoid
  // issues in the unwinder.
  if (TT.isOSWindows() && TT.isX86_64())
    CGMFPM.addPass(X86AvoidTrailingCallPass());

  // Verify basic block incoming and outgoing cfa offset and register values and
  // correct CFA calculation rule where needed by inserting appropriate CFI
  // instructions.
  if (!TT.isOSDarwin() &&
      (!TT.isOSWindows() ||
       MAI.getExceptionHandlingType() == ExceptionHandling::DwarfCFI)) {
    // TODO(boomanaiden154): Add CFInstrInserterPass here when it has been
    // ported.
  }

  if (TT.isOSWindows()) {
    // Identify valid longjmp targets for Windows Control Flow Guard.
    // TODO(boomanaiden154): Add CFGuardLongjmpPass here when it has been
    // ported.
    // Identify valid eh continuation targets for Windows EHCont Guard.
    // TODO(boomanaiden154): Add EHContGuardTargetsPass when it has been
    // ported.
  }

  CGMFPM.addPass(X86LoadValueInjectionRetHardeningPass());

  // Insert pseudo probe annotation for callsite profiling
  // TODO(boomanaiden154): Add PseudoProberInserterPass here once it has been
  // ported.

  // KCFI indirect call checks are lowered to a bundle, and on Darwin platforms,
  // also CALL_RVMARKER.
  // TODO(boomanaiden154): Add UnpackMachineBundlesPass here once it has been
  // ported.

  // Analyzes and emits pseudos to support Win x64 Unwind V2. This pass must run
  // after all real instructions have been added to the epilog.
  if (TT.isOSWindows() && TT.isX86_64()) {
    CGMFPM.addPass(X86WinEHUnwindV2Pass());
  }
}

void X86CodeGenPassBuilder::addAsmPrinterBegin(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(X86AsmPrinterBeginPass());
}

void X86CodeGenPassBuilder::addAsmPrinter(
    CodeGenMachineFunctionPassManager &CGMFPM) const {
  CGMFPM.addPass(X86AsmPrinterPass());
}

void X86CodeGenPassBuilder::addAsmPrinterEnd(
    CodeGenModulePassManager &CGMPM) const {
  CGMPM.addPass(X86AsmPrinterEndPass());
}

} // namespace

void X86TargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "X86PassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
  // TODO(boomanaiden154): Move this into the base CodeGenPassBuilder once all
  // targets that currently implement it have a ported asm-printer pass.
  if (PIC) {
    PIC->addClassToPassName(X86AsmPrinterBeginPass::name(),
                            "x86-asm-printer-begin");
    PIC->addClassToPassName(X86AsmPrinterPass::name(), "x86-asm-printer");
    PIC->addClassToPassName(X86AsmPrinterEndPass::name(),
                            "x86-asm-printer-end");
  }
}

Error X86TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, ModuleAnalysisManager &MAM, raw_pwrite_stream &Out,
    raw_pwrite_stream *DwoOut, CodeGenFileType FileType,
    const CGPassBuilderOption &Opt, MCContext &Ctx,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = X86CodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, MAM, Out, DwoOut, FileType, Ctx);
}
