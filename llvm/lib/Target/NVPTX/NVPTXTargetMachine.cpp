//===-- NVPTXTargetMachine.cpp - Define TargetMachine for NVPTX -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetMachine.h"
#include "NVPTX.h"
#include "NVPTXAliasAnalysis.h"
#include "NVPTXAllocaHoisting.h"
#include "NVPTXAtomicLower.h"
#include "NVPTXCtorDtorLowering.h"
#include "NVPTXLowerAggrCopies.h"
#include "NVPTXMachineFunctionInfo.h"
#include "NVPTXTargetObjectFile.h"
#include "NVPTXTargetTransformInfo.h"
#include "TargetInfo/NVPTXTargetInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include <cassert>
#include <optional>
#include <string>

using namespace llvm;

// LSV is still relatively new; this switch lets us turn it off in case we
// encounter (or suspect) a bug.
static cl::opt<bool>
    DisableLoadStoreVectorizer("disable-nvptx-load-store-vectorizer",
                               cl::desc("Disable load/store vectorizer"),
                               cl::init(false), cl::Hidden);

// TODO: Remove this flag when we are confident with no regressions.
static cl::opt<bool> DisableRequireStructuredCFG(
    "disable-nvptx-require-structured-cfg",
    cl::desc("Transitional flag to turn off NVPTX's requirement on preserving "
             "structured CFG. The requirement should be disabled only when "
             "unexpected regressions happen."),
    cl::init(false), cl::Hidden);

static cl::opt<bool> UseShortPointersOpt(
    "nvptx-short-ptr",
    cl::desc(
        "Use 32-bit pointers for accessing const/local/shared address spaces."),
    cl::init(false), cl::Hidden);

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

namespace llvm {

void initializeGenericToNVVMLegacyPassPass(PassRegistry &);
void initializeNVPTXAllocaHoistingPass(PassRegistry &);
void initializeNVPTXAssignValidGlobalNamesPass(PassRegistry &);
void initializeNVPTXAtomicLowerPass(PassRegistry &);
void initializeNVPTXCtorDtorLoweringLegacyPass(PassRegistry &);
void initializeNVPTXLowerAggrCopiesPass(PassRegistry &);
void initializeNVPTXLowerAllocaPass(PassRegistry &);
void initializeNVPTXLowerUnreachablePass(PassRegistry &);
void initializeNVPTXCtorDtorLoweringLegacyPass(PassRegistry &);
void initializeNVPTXLowerArgsPass(PassRegistry &);
void initializeNVPTXProxyRegErasurePass(PassRegistry &);
void initializeNVVMIntrRangePass(PassRegistry &);
void initializeNVVMReflectPass(PassRegistry &);
void initializeNVPTXAAWrapperPassPass(PassRegistry &);
void initializeNVPTXExternalAAWrapperPass(PassRegistry &);

} // end namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNVPTXTarget() {
  // Register the target.
  RegisterTargetMachine<NVPTXTargetMachine32> X(getTheNVPTXTarget32());
  RegisterTargetMachine<NVPTXTargetMachine64> Y(getTheNVPTXTarget64());

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  // FIXME: This pass is really intended to be invoked during IR optimization,
  // but it's very NVPTX-specific.
  initializeNVVMReflectPass(PR);
  initializeNVVMIntrRangePass(PR);
  initializeGenericToNVVMLegacyPassPass(PR);
  initializeNVPTXAllocaHoistingPass(PR);
  initializeNVPTXAssignValidGlobalNamesPass(PR);
  initializeNVPTXAtomicLowerPass(PR);
  initializeNVPTXLowerArgsPass(PR);
  initializeNVPTXLowerAllocaPass(PR);
  initializeNVPTXLowerUnreachablePass(PR);
  initializeNVPTXCtorDtorLoweringLegacyPass(PR);
  initializeNVPTXLowerAggrCopiesPass(PR);
  initializeNVPTXProxyRegErasurePass(PR);
  initializeNVPTXDAGToDAGISelLegacyPass(PR);
  initializeNVPTXAAWrapperPassPass(PR);
  initializeNVPTXExternalAAWrapperPass(PR);
}

static std::string computeDataLayout(bool is64Bit, bool UseShortPointers) {
  std::string Ret = "e";

  if (!is64Bit)
    Ret += "-p:32:32";
  else if (UseShortPointers)
    Ret += "-p3:32:32-p4:32:32-p5:32:32";

  Ret += "-i64:64-i128:128-v16:16-v32:32-n16:32:64";

  return Ret;
}

NVPTXTargetMachine::NVPTXTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool is64bit)
    // The pic relocation model is used regardless of what the client has
    // specified, as it is the only relocation model currently supported.
    : CodeGenTargetMachineImpl(T,
                               computeDataLayout(is64bit, UseShortPointersOpt),
                               TT, CPU, FS, Options, Reloc::PIC_,
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      is64bit(is64bit), TLOF(std::make_unique<NVPTXTargetObjectFile>()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this),
      StrPool(StrAlloc) {
  if (TT.getOS() == Triple::NVCL)
    drvInterface = NVPTX::NVCL;
  else
    drvInterface = NVPTX::CUDA;
  if (!DisableRequireStructuredCFG)
    setRequiresStructuredCFG(true);
  initAsmInfo();
}

NVPTXTargetMachine::~NVPTXTargetMachine() = default;

void NVPTXTargetMachine32::anchor() {}

NVPTXTargetMachine32::NVPTXTargetMachine32(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void NVPTXTargetMachine64::anchor() {}

NVPTXTargetMachine64::NVPTXTargetMachine64(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           std::optional<Reloc::Model> RM,
                                           std::optional<CodeModel::Model> CM,
                                           CodeGenOptLevel OL, bool JIT)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace {

class NVPTXPassConfig : public TargetPassConfig {
public:
  NVPTXPassConfig(NVPTXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  NVPTXTargetMachine &getNVPTXTargetMachine() const {
    return getTM<NVPTXTargetMachine>();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addMachineSSAOptimization() override;

  FunctionPass *createTargetRegisterAllocator(bool) override;
  void addFastRegAlloc() override;
  void addOptimizedRegAlloc() override;

  bool addRegAssignAndRewriteFast() override {
    llvm_unreachable("should not be used");
  }

  bool addRegAssignAndRewriteOptimized() override {
    llvm_unreachable("should not be used");
  }

private:
  // If the opt level is aggressive, add GVN; otherwise, add EarlyCSE. This
  // function is only called in opt mode.
  void addEarlyCSEOrGVNPass();

  // Add passes that propagate special memory spaces.
  void addAddressSpaceInferencePasses();

  // Add passes that perform straight-line scalar optimizations.
  void addStraightLineScalarOptimizationPasses();
};

} // end anonymous namespace

TargetPassConfig *NVPTXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new NVPTXPassConfig(*this, PM);
}

MachineFunctionInfo *NVPTXTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return NVPTXMachineFunctionInfo::create<NVPTXMachineFunctionInfo>(Allocator,
                                                                    F, STI);
}

void NVPTXTargetMachine::registerDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<NVPTXAA>();
}

void NVPTXTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "NVPTXPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"

  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        FunctionPassManager FPM;
        // We do not want to fold out calls to nvvm.reflect early if the user
        // has not provided a target architecture just yet.
        if (Subtarget.hasTargetName())
          FPM.addPass(NVVMReflectPass(Subtarget.getSmVersion()));
        // Note: NVVMIntrRangePass was causing numerical discrepancies at one
        // point, if issues crop up, consider disabling.
        FPM.addPass(NVVMIntrRangePass());
        if (EarlyByValArgsCopy)
          FPM.addPass(NVPTXCopyByValArgsPass());
        PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
}

TargetTransformInfo
NVPTXTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(NVPTXTTIImpl(this, F));
}

std::pair<const Value *, unsigned>
NVPTXTargetMachine::getPredicatedAddrSpace(const Value *V) const {
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_isspacep_const:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_CONST);
    case Intrinsic::nvvm_isspacep_global:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_GLOBAL);
    case Intrinsic::nvvm_isspacep_local:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_LOCAL);
    case Intrinsic::nvvm_isspacep_shared:
    case Intrinsic::nvvm_isspacep_shared_cluster:
      return std::make_pair(II->getArgOperand(0), llvm::ADDRESS_SPACE_SHARED);
    default:
      break;
    }
  }
  return std::make_pair(nullptr, -1);
}

void NVPTXPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOptLevel::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void NVPTXPassConfig::addAddressSpaceInferencePasses() {
  // NVPTXLowerArgs emits alloca for byval parameters which can often
  // be eliminated by SROA.
  addPass(createSROAPass());
  addPass(createNVPTXLowerAllocaPass());
  // TODO: Consider running InferAddressSpaces during opt, earlier in the
  // compilation flow.
  addPass(createInferAddressSpacesPass());
  addPass(createNVPTXAtomicLowerPass());
}

void NVPTXPassConfig::addStraightLineScalarOptimizationPasses() {
  addPass(createSeparateConstOffsetFromGEPPass());
  addPass(createSpeculativeExecutionPass());
  // ReassociateGEPs exposes more opportunites for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse. GVN generates significantly better code than EarlyCSE
  // for some of our benchmarks.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void NVPTXPassConfig::addIRPasses() {
  // The following passes are known to not play well with virtual regs hanging
  // around after register allocation (which in our case, is *all* registers).
  // We explicitly disable them here.  We do, however, need some functionality
  // of the PrologEpilogCodeInserter pass, so we emulate that behavior in the
  // NVPTXPrologEpilog pass (see NVPTXPrologEpilogPass.cpp).
  disablePass(&PrologEpilogCodeInserterID);
  disablePass(&MachineLateInstrsCleanupID);
  disablePass(&MachineCopyPropagationID);
  disablePass(&TailDuplicateLegacyID);
  disablePass(&StackMapLivenessID);
  disablePass(&PostRAMachineSinkingID);
  disablePass(&PostRASchedulerID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);
  disablePass(&ShrinkWrapID);
  disablePass(&RemoveLoadsIntoFakeUsesID);

  addPass(createNVPTXAAWrapperPass());
  addPass(createExternalAAWrapperPass([](Pass &P, Function &, AAResults &AAR) {
    if (auto *WrapperPass = P.getAnalysisIfAvailable<NVPTXAAWrapperPass>())
      AAR.addAAResult(WrapperPass->getResult());
  }));

  // NVVMReflectPass is added in addEarlyAsPossiblePasses, so hopefully running
  // it here does nothing.  But since we need it for correctness when lowering
  // to NVPTX, run it here too, in case whoever built our pass pipeline didn't
  // call addEarlyAsPossiblePasses.
  const NVPTXSubtarget &ST = *getTM<NVPTXTargetMachine>().getSubtargetImpl();
  addPass(createNVVMReflectPass(ST.getSmVersion()));

  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createNVPTXImageOptimizerPass());
  addPass(createNVPTXAssignValidGlobalNamesPass());
  addPass(createGenericToNVVMLegacyPass());

  // NVPTXLowerArgs is required for correctness and should be run right
  // before the address space inference passes.
  addPass(createNVPTXLowerArgsPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addAddressSpaceInferencePasses();
    addStraightLineScalarOptimizationPasses();
  }

  addPass(createAtomicExpandLegacyPass());
  addPass(createExpandVariadicsPass(ExpandVariadicsMode::Lowering));
  addPass(createNVPTXCtorDtorLoweringLegacyPass());

  // === LSR and other generic IR passes ===
  TargetPassConfig::addIRPasses();
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
    addEarlyCSEOrGVNPass();
    if (!DisableLoadStoreVectorizer)
      addPass(createLoadStoreVectorizerPass());
    addPass(createSROAPass());
  }

  if (ST.hasPTXASUnreachableBug()) {
    // Run LowerUnreachable to WAR a ptxas bug. See the commit description of
    // 1ee4d880e8760256c606fe55b7af85a4f70d006d for more details.
    const auto &Options = getNVPTXTargetMachine().Options;
    addPass(createNVPTXLowerUnreachablePass(Options.TrapUnreachable,
                                            Options.NoTrapAfterNoreturn));
  }
}

bool NVPTXPassConfig::addInstSelector() {
  addPass(createLowerAggrCopies());
  addPass(createAllocaHoisting());
  addPass(createNVPTXISelDag(getNVPTXTargetMachine(), getOptLevel()));
  addPass(createNVPTXReplaceImageHandlesPass());

  return false;
}

void NVPTXPassConfig::addPreRegAlloc() {
  // Remove Proxy Register pseudo instructions used to keep `callseq_end` alive.
  addPass(createNVPTXProxyRegErasurePass());
}

void NVPTXPassConfig::addPostRegAlloc() {
  addPass(createNVPTXPrologEpilogPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    // NVPTXPrologEpilogPass calculates frame object offset and replace frame
    // index with VRFrame register. NVPTXPeephole need to be run after that and
    // will replace VRFrame with VRFrameLocal when possible.
    addPass(createNVPTXPeephole());
  }
}

FunctionPass *NVPTXPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
}

void NVPTXPassConfig::addFastRegAlloc() {
  addPass(&PHIEliminationID);
  addPass(&TwoAddressInstructionPassID);
}

void NVPTXPassConfig::addOptimizedRegAlloc() {
  addPass(&ProcessImplicitDefsID);
  addPass(&LiveVariablesID);
  addPass(&MachineLoopInfoID);
  addPass(&PHIEliminationID);

  addPass(&TwoAddressInstructionPassID);
  addPass(&RegisterCoalescerID);

  // PreRA instruction scheduling.
  if (addPass(&MachineSchedulerID))
    printAndVerify("After Machine Scheduling");

  addPass(&StackSlotColoringID);

  // FIXME: Needs physical registers
  // addPass(&MachineLICMID);

  printAndVerify("After StackSlotColoring");
}

void NVPTXPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  if (addPass(&EarlyTailDuplicateLegacyID))
    printAndVerify("After Pre-RegAlloc TailDuplicate");

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(&OptimizePHIsLegacyID);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(&StackColoringLegacyID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(&LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(&DeadMachineInstructionElimID);
  printAndVerify("After codegen DCE pass");

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  if (addILPOpts())
    printAndVerify("After ILP optimizations");

  addPass(&EarlyMachineLICMID);
  addPass(&MachineCSELegacyID);

  addPass(&MachineSinkingID);
  printAndVerify("After Machine LICM, CSE and Sinking passes");

  addPass(&PeepholeOptimizerLegacyID);
  printAndVerify("After codegen peephole optimization pass");
}
